"""
Z-Score Data Manager - Depth / Trades / Price Window

CORRECTED: Matches official CoinSwitch WebSocket formats
UPDATED: Robust Volatility (ATR/Realized) calculation to prevent MISSING data
+ Vol-Regime Detection for dynamic parameter tuning
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Deque  # Add Tuple if missing
from collections import deque
from datetime import datetime
import threading
import numpy as np
import pandas as pd  # used for EMA, ATR, and feature calculations

import torch
import torch.nn as nn
import torch.optim as optim

from futures_api import FuturesAPI
from futures_websocket import FuturesWebSocket
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)

import random
random.seed(42)


class TrendLSTM(nn.Module):
    """
    Simple LSTM classifier for 3-state trend:
    0 = UPTREND, 1 = DOWNTREND, 2 = RANGEBOUND
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out


class ZScoreDataManager:
    """Data manager for Z-Score Imbalance Iceberg Hunter."""

    # Price history must be long enough to cover:
    # - HTF_EMA_SPAN + HTF_LOOKBACK_BARS worth of HTF_TREND_INTERVAL minutes
    # - ATR_WINDOW_MINUTES (with some buffer)
    _MAX_PRICE_HISTORY_SEC = max(
        900,
        getattr(config, "HTF_TREND_INTERVAL", 5)
        * (
            getattr(config, "HTF_EMA_SPAN", 80)
            + getattr(config, "HTF_LOOKBACK_BARS", 86)
            + 1
        )
        * 60,
        getattr(config, "ATR_WINDOW_MINUTES", 10) * 60 * 2,
    )
    _MAX_TRADES_HISTORY_SEC = 900

    def __init__(self) -> None:
        logger.info("=" * 80)
        logger.info("Z-SCORE DATA MANAGER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Symbol       : {config.SYMBOL}")
        logger.info(f"Exchange     : {config.EXCHANGE}")

        # WebSocket connection (actual connect happens in start())
        self.ws: Optional[FuturesWebSocket] = None

        # REST API client
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )

        # Data storage
        self._orderbook_bids: List[Tuple[float, float]] = []
        self._orderbook_asks: List[Tuple[float, float]] = []
        self._trades: Deque[Dict] = deque(maxlen=1000)
        self._price_window: Deque[Tuple[int, float]] = deque(
            maxlen=int(self._MAX_PRICE_HISTORY_SEC * 2)
        )
        self._candles_1m: Deque[Dict] = deque(maxlen=100)
        self._candles_5m: Deque[Dict] = deque(maxlen=100)

        # Additional series used elsewhere
        self._htf_5m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._bos_15m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._ltf_1m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._recent_trades: Deque[Dict] = deque(maxlen=2000)

        # Last price cache
        self._last_price: float = 0.0

        # Locks
        import threading
        self._orderbook_lock = threading.Lock()
        self._trades_lock = threading.Lock()
        self._price_lock = threading.Lock()
        self._candles_lock = threading.Lock()

        # Stats
        self.stats = {
            "orderbook_updates": 0,
            "trade_updates": 0,
            "candle_updates": 0,
            "prices_recorded": 0,
            "trades_received": 0,
            "candles_received": 0,
            "last_update": None,
        }

        # ═══════════════════════════════════════════════════════════════
        # ADDED: REST API rate limiting (FIX: Reduces warmup API calls)
        # ═══════════════════════════════════════════════════════════════
        self._rest_api_last_call = 0.0
        self._rest_api_min_interval = 2.0  # Min 2 seconds between REST calls
        logger.info("✓ REST API rate limiting enabled (2s minimum interval)")

        
        # LSTM / model placeholders
        self._lstm_model: Optional[nn.Module] = None
        self._lstm_seq_len = 10
        self._lstm_input_dim = 5

        self._htf_lstm: Optional[TrendLSTM] = None
        self._htf_lstm_trained: bool = False
        self._htf_norm_mean: float = 0.0
        self._htf_norm_std: float = 1.0
        self._htf_pending_trend: Optional[str] = None
        self._htf_confirm_count: int = 0
        self._last_htf_trend: Optional[str] = None

        self._ltf_lstm: Optional[TrendLSTM] = None
        self._ltf_lstm_trained: bool = False
        self._ltf_norm_mean: float = 0.0
        self._ltf_norm_std: float = 1.0
        self._ltf_pending_trend: Optional[str] = None
        self._ltf_confirm_count: int = 0
        self._last_ltf_trend: Optional[str] = None

        # Initialize Aether Oracle
        from aether_oracle import AetherOracle
        self._oracle = AetherOracle()

        # Streaming flag
        self.is_streaming: bool = False

        logger.info(f"Streams      : ORDERBOOK, TRADES, CANDLESTICK")
        logger.info("=" * 80)

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def _wait_for_rest_api_limit(self) -> None:
        """
        Enforce minimum interval between REST API calls.
        Prevents burst requests during warmup that trigger 429 errors.
        """
        now = time.time()
        elapsed = now - self._rest_api_last_call

        if elapsed < self._rest_api_min_interval:
            wait_time = self._rest_api_min_interval - elapsed
            logger.debug(f"⏸️  REST API rate limit: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)

        self._rest_api_last_call = time.time()


    def start(self) -> bool:
        """Connect WebSocket and subscribe to streams, then warm up from REST klines."""
        try:
            self.ws = FuturesWebSocket()
            logger.info("Connecting to CoinSwitch Futures WebSocket...")
            if not self.ws.connect():
                logger.error("WebSocket connection failed")
                return False

            # ORDERBOOK: just "BTCUSDT" (no interval)
            logger.info(f"Subscribing ORDERBOOK: {config.SYMBOL}")
            self.ws.subscribe_orderbook(config.SYMBOL, callback=self._on_orderbook)

            # TRADES: "BTCUSDT"
            logger.info(f"Subscribing TRADES: {config.SYMBOL}")
            self.ws.subscribe_trades(config.SYMBOL, callback=self._on_trade)

            # 1‑minute CANDLESTICKS (for EMA/ATR + LTF LSTM)
            logger.info(
                f"Subscribing CANDLESTICKS 1m: {config.SYMBOL}_{config.CANDLE_INTERVAL}"
            )
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=config.CANDLE_INTERVAL,  # typically 1
                callback=self._on_candlestick_1m,
            )

            # 5‑minute CANDLESTICKS (HTF LSTM)
            logger.info(f"Subscribing CANDLESTICKS 5m: {config.SYMBOL}_5")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=5,
                callback=self._on_candlestick_5m,
            )

            # 15‑minute CANDLESTICKS (BOS / structure)
            logger.info(f"Subscribing CANDLESTICKS 15m: {config.SYMBOL}_15")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=15,
                callback=self._on_candlestick_15m,
            )

            self.is_streaming = True
            logger.info("Z-Score data streams started")

            # REST warmup for 1m prices (EMA, ATR, LTF)
            self._warmup_from_klines_1m()

            # REST warmup for 5m HTF candles
            self._warmup_htf_klines_5m()

            # REST warmup for 15m BOS candles
            self._warmup_bos_klines_15m()

            return True

        except Exception as e:
            logger.error(f"Error starting ZScoreDataManager: {e}", exc_info=True)
            return False

    def stop(self) -> None:
        """Stop WebSocket."""
        try:
            if self.ws and self.is_streaming:
                self.ws.disconnect()
        finally:
            self.is_streaming = False
            logger.info("ZScoreDataManager stopped")

    # ======================================================================
    # Warmup - 1m, 5m, 15m
    # ======================================================================

    def _warmup_from_klines_1m(self) -> None:
        """Warmup 1m price window + LTF candles from REST klines."""
        try:
            window_min = getattr(config, "ATR_WINDOW_MINUTES", 10)
            ltf_lookback = getattr(config, "LTF_LOOKBACK_BARS", 60)
            limit = max(window_min, ltf_lookback) + 10
            logger.info(f"Warming up 1m klines (limit={limit})...")

            self._wait_for_rest_api_limit()  # ← ADDED: Rate limit enforcement
            resp = self._fetch_rest_klines(limit=limit, interval=1)

            if not resp or "data" not in resp:
                logger.warning("No 1m klines returned from REST API")
                return

            klines = resp["data"].get("klines", [])
            if not klines:
                logger.warning("Empty 1m klines array")
                return

            for k in klines:
                try:
                    ts_ms = int(k.get("openTime", 0))
                    close = float(k.get("close", 0.0))
                    if ts_ms > 0 and close > 0:
                        self._append_price(ts_ms, close)
                        self._append_ltf_1m_close(ts_ms, close)
                except Exception as e_inner:
                    logger.debug(f"Skipping malformed 1m kline: {e_inner}")
                    continue

            logger.info(
                f"1m warmup complete: {len(self._price_window)} ticks, "
                f"{len(self._ltf_1m_closes)} LTF candles"
            )

        except Exception as e:
            logger.error(f"Error in _warmup_from_klines_1m: {e}", exc_info=True)

    def _fetch_rest_klines(self, limit: int, interval: int = 1):
        """
        Try multiple method names on the FuturesAPI and common signatures.
        Returns the response dict or None if unavailable.
        """
        candidates = (
            getattr(self.api, "get_klines", None),
            getattr(self.api, "get_candles", None),
            getattr(self.api, "fetch_klines", None),
            getattr(self.api, "klines", None),
        )

        for fn in candidates:
            if callable(fn):
                try:
                    # Try named kwargs first
                    resp = fn(symbol=config.SYMBOL, interval=interval, limit=limit, exchange=config.EXCHANGE)
                    return resp
                except TypeError:
                    try:
                        # Try positional fallback
                        resp = fn(config.SYMBOL, interval, limit)
                        return resp
                    except Exception:
                        continue
                except Exception:
                    continue
        logger.warning("No REST klines method available on FuturesAPI; skipping REST warmup.")
        return None

    def _warmup_htf_klines_5m(self) -> None:
        """Warmup HTF 5m candles from REST."""
        try:
            htf_interval = getattr(config, "HTF_TREND_INTERVAL", 5)
            htf_span = getattr(config, "HTF_EMA_SPAN", 80)
            htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
            limit = htf_span + htf_lookback + 10
            logger.info(f"Warming up {htf_interval}m HTF klines (limit={limit})...")

            self._wait_for_rest_api_limit()  # ← ADDED: Rate limit enforcement
            resp = self._fetch_rest_klines(limit=limit, interval=htf_interval)

            if not resp or "data" not in resp:
                logger.warning("No 5m HTF klines returned from REST API")
                return

            klines = resp["data"].get("klines", [])
            if not klines:
                logger.warning("Empty 5m HTF klines array")
                return

            for k in klines:
                try:
                    ts_ms = int(k.get("openTime", 0))
                    close = float(k.get("close", 0.0))
                    if ts_ms > 0 and close > 0:
                        self._append_htf_5m_close(ts_ms, close)
                except Exception as e_inner:
                    logger.debug(f"Skipping malformed 5m kline: {e_inner}")
                    continue

            logger.info(f"HTF 5m warmup complete: {len(self._htf_5m_closes)} candles")

        except Exception as e:
            logger.error(f"Error in _warmup_htf_klines_5m: {e}", exc_info=True)

    def _warmup_bos_klines_15m(self) -> None:
        """Warmup BOS 15m candles from REST."""
        try:
            bos_interval = 15
            htf_span = getattr(config, "HTF_EMA_SPAN", 80)
            htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
            limit = htf_span + htf_lookback + 10
            logger.info(f"Warming up {bos_interval}m BOS klines (limit={limit})...")

            self._wait_for_rest_api_limit()  # ← ADDED: Rate limit enforcement
            resp = self._fetch_rest_klines(limit=limit, interval=bos_interval)


            if not resp or "data" not in resp:
                logger.warning("No 15m BOS klines returned from REST API")
                return

            klines = resp["data"].get("klines", [])
            if not klines:
                logger.warning("Empty 15m BOS klines array")
                return

            for k in klines:
                try:
                    ts_ms = int(k.get("openTime", 0))
                    close = float(k.get("close", 0.0))
                    if ts_ms > 0 and close > 0:
                        self._append_bos_15m_close(ts_ms, close)
                except Exception as e_inner:
                    logger.debug(f"Skipping malformed 15m kline: {e_inner}")
                    continue

            logger.info(f"BOS 15m warmup complete: {len(self._bos_15m_closes)} candles")

        except Exception as e:
            logger.error(f"Error in _warmup_bos_klines_15m: {e}", exc_info=True)

    # ======================================================================
    # WebSocket callbacks
    # ======================================================================

    def _on_orderbook(self, data: Dict) -> None:
        """Handle orderbook depth snapshot/update."""
        try:
            if not isinstance(data, dict):
                logger.error(f"Orderbook callback non-dict: {type(data)}")
                return

            bids_raw = data.get("bids") or data.get("b") or []
            asks_raw = data.get("asks") or data.get("a") or []

            if not bids_raw or not asks_raw:
                return

            def _parse_level(lvl) -> Optional[Tuple[float, float]]:
                try:
                    if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        return (float(lvl[0]), float(lvl[1]))
                    elif isinstance(lvl, dict):
                        p = float(lvl.get("price") or lvl.get("p") or 0)
                        q = float(lvl.get("quantity") or lvl.get("q") or 0)
                        if p > 0 and q > 0:
                            return (p, q)
                except Exception:
                    pass
                return None

            bids = [_parse_level(b) for b in bids_raw]
            bids = [x for x in bids if x is not None]

            asks = [_parse_level(a) for a in asks_raw]
            asks = [x for x in asks if x is not None]

            if not bids or not asks:
                return

            self._bids = sorted(bids, key=lambda x: x[0], reverse=True)
            self._asks = sorted(asks, key=lambda x: x[0])

            self.stats["orderbook_updates"] += 1
            self.stats["last_update"] = datetime.now()

            # Optional: if price is still unset, use mid-price once
            if self._last_price <= 0:
                mid = (bids[0][0] + asks[0][0]) / 2.0
                ts_ms = int(time.time() * 1000)
                self._append_price(ts_ms, mid)

        except Exception as e:
            logger.error(f"Error in _on_orderbook: {e}", exc_info=True)

    def _on_trade(self, data: Dict) -> None:
        """Handle trades (updates last price every trade)."""
        try:
            if not isinstance(data, dict):
                logger.error(f"Trade callback non-dict: {type(data)}")
                return

            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))
            ts_ms = int(data.get("T", 0))
            is_buyer_maker = bool(data.get("m", False))

            if price <= 0 or qty <= 0 or ts_ms <= 0:
                return

            self._recent_trades.append(
                {
                    "price": price,
                    "qty": qty,
                    "ts_ms": ts_ms,
                    "isBuyerMaker": is_buyer_maker,
                }
            )

            # Update last price from every trade
            self._append_price(ts_ms, price)
            self.stats["trades_received"] += 1

        except Exception as e:
            logger.error(f"Error in _on_trade: {e}", exc_info=True)

    def _on_candlestick_1m(self, data) -> None:
        """Handle 1‑minute candlestick updates (for EMA/ATR and LTF LSTM)."""
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str:
                    return
                try:
                    price = float(close_str)
                except Exception:
                    return
                if price <= 0:
                    return

                ts_ms = int(
                    data.get("ts")
                    or data.get("T")
                    or data.get("t")
                    or int(time.time() * 1000)
                )

                self._append_price(ts_ms, price)
                self._append_ltf_1m_close(ts_ms, price)
                self.stats["candles_received"] += 1

        except Exception as e:
            logger.error(f"Error in _on_candlestick_1m: {e}", exc_info=True)

    def _on_candlestick_5m(self, data) -> None:
        """Handle 5‑minute candlestick updates (native HTF stream)."""
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str:
                    return
                try:
                    price = float(close_str)
                except Exception:
                    return
                if price <= 0:
                    return

                ts_ms = int(
                    data.get("ts")
                    or data.get("T")
                    or data.get("t")
                    or int(time.time() * 1000)
                )

                self._append_htf_5m_close(ts_ms, price)

        except Exception as e:
            logger.error(f"Error in _on_candlestick_5m: {e}", exc_info=True)

    def _on_candlestick_15m(self, data) -> None:
        """Handle 15‑minute candlestick updates (BOS / structure)."""
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str:
                    return
                try:
                    price = float(close_str)
                except Exception:
                    return
                if price <= 0:
                    return

                ts_ms = int(
                    data.get("ts")
                    or data.get("T")
                    or data.get("t")
                    or int(time.time() * 1000)
                )

                self._append_bos_15m_close(ts_ms, price)

        except Exception as e:
            logger.error(f"Error in _on_candlestick_15m: {e}", exc_info=True)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _append_price(self, ts_ms: int, price: float) -> None:
        """Append price into the 1m/tick price window (EMA/ATR/etc.)."""
        self._last_price = price
        self._price_window.append((ts_ms, price))
        self.stats["prices_recorded"] += 1
        self.stats["last_update"] = datetime.now()

        # Keep slightly more than MAX history to allow for late data / buffer
        cutoff_price = ts_ms - (self._MAX_PRICE_HISTORY_SEC + 120) * 1000
        while self._price_window and self._price_window[0][0] < cutoff_price:
            self._price_window.popleft()

        cutoff_trades = ts_ms - (self._MAX_TRADES_HISTORY_SEC + 60) * 1000
        while self._recent_trades and self._recent_trades[0]["ts_ms"] < cutoff_trades:
            self._recent_trades.popleft()

    def _append_htf_5m_close(self, ts_ms: int, close_price: float) -> None:
        """Append a native 5‑minute close into the HTF buffer."""
        self._htf_5m_closes.append((ts_ms, close_price))

        # Keep only enough HTF history for robust LSTM training
        htf_interval_min = getattr(config, "HTF_TREND_INTERVAL", 5)
        htf_span = getattr(config, "HTF_EMA_SPAN", 80)
        htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
        max_htf_sec = (htf_span + htf_lookback + 3) * htf_interval_min * 60
        cutoff = ts_ms - max_htf_sec * 1000

        while self._htf_5m_closes and self._htf_5m_closes[0][0] < cutoff:
            self._htf_5m_closes.popleft()

    def _append_bos_15m_close(self, ts_ms: int, close_price: float) -> None:
        """Append a native 15‑minute close into the BOS buffer."""
        self._bos_15m_closes.append((ts_ms, close_price))

        bos_interval_min = 15
        htf_span = getattr(config, "HTF_EMA_SPAN", 80)
        htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
        max_bos_sec = (htf_span + htf_lookback + 3) * bos_interval_min * 60
        cutoff = ts_ms - max_bos_sec * 1000

        while self._bos_15m_closes and self._bos_15m_closes[0][0] < cutoff:
            self._bos_15m_closes.popleft()

    def _append_ltf_1m_close(self, ts_ms: int, close_price: float) -> None:
        """Append a native 1‑minute close into the LTF buffer for 1m trend."""
        self._ltf_1m_closes.append((ts_ms, close_price))

        ltf_span = getattr(config, "LTF_EMA_SPAN", getattr(config, "EMA_PERIOD", 20))
        ltf_lookback = getattr(config, "LTF_LOOKBACK_BARS", 60)
        max_ltf_sec = (ltf_span + ltf_lookback + 3) * 60  # 1m bars
        cutoff = ts_ms - max_ltf_sec * 1000

        while self._ltf_1m_closes and self._ltf_1m_closes[0][0] < cutoff:
            self._ltf_1m_closes.popleft()

    # ======================================================================
    # Public accessors
    # ======================================================================

    def get_orderbook_snapshot(
        self,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        return list(self._bids), list(self._asks)

    def get_last_price(self) -> float:
        return float(self._last_price) if self._last_price > 0 else 0.0

    def get_price_window(self, window_seconds: int = 480) -> List[Tuple[int, float]]:
        """
        Get price window for the last N seconds.
        FIXED: Uses a snapshot of the deque to avoid mutation/race
        issues while iterating.
        """
        if not self._price_window:
            return []

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_seconds * 1000

        # Snapshot
        snapshot = list(self._price_window)
        return [(ts, p) for (ts, p) in snapshot if ts >= cutoff_ms]

    def get_recent_trades(self, window_seconds: int) -> List[Dict]:
        """
        Get recent trades within a window.
        FIXED: Uses a snapshot of the deque to avoid mutation/race
        issues while iterating.
        """
        if not self._recent_trades:
            return []

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_seconds * 1000

        snapshot = list(self._recent_trades)
        return [t for t in snapshot if t["ts_ms"] >= cutoff_ms]

    # ======================================================================
    # Derived metrics: EMA + ATR (for volatility / legacy trend)
    # ======================================================================

    def get_ema(self, period: int = 20, window_minutes: int = 480) -> Optional[float]:
        """
        Calculate EMA from 1‑minute tick price window.
        Returns None if insufficient data.
        """
        try:
            pw = self.get_price_window(window_seconds=window_minutes * 60)
            if not pw or len(pw) < period:
                return None

            prices = np.asarray([p for (_, p) in pw], dtype=np.float64)
            if len(prices) < period:
                return None

            ema_series = (
                pd.Series(prices).ewm(span=period, adjust=False).mean()
            )
            return float(ema_series.iloc[-1])

        except Exception as e:
            logger.error(f"Error calculating EMA: {e}", exc_info=True)
            return None

    def get_atr_percent(self, window_minutes: int = 10) -> Optional[float]:
        """
        Calculate ATR as a percentage of current price using tick data.
        Returns ATR% (e.g., 0.008 = 0.8%).
        """
        try:
            pw = self.get_price_window(window_seconds=window_minutes * 60)
            if not pw or len(pw) < 2:
                return None

            prices = [p for (_, p) in pw]
            if len(prices) < 2:
                return None

            ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            if not ranges:
                return None

            atr = float(np.mean(ranges))
            current_price = self.get_last_price()

            if current_price <= 0:
                return None

            atr_pct = atr / current_price
            return atr_pct

        except Exception as e:
            logger.error(f"Error calculating ATR%: {e}", exc_info=True)
            return None

    # ======================================================================
    # VOL-REGIME DETECTION (NEW)
    # ======================================================================

    def get_vol_regime(self) -> Tuple[str, Optional[float]]:
        """
        Classify current volatility regime based on ATR%.
        Returns: (regime_str, atr_pct)
        regime_str: "LOW", "HIGH", "NEUTRAL", or "UNKNOWN"
        atr_pct: actual ATR% value or None if unavailable
        """
        try:
            atr_pct = self.get_atr_percent(window_minutes=config.ATR_WINDOW_MINUTES)

            if atr_pct is None:
                return ("UNKNOWN", None)

            if atr_pct < config.VOL_REGIME_LOW_THRESHOLD:
                return ("LOW", atr_pct)
            elif atr_pct > config.VOL_REGIME_HIGH_THRESHOLD:
                return ("HIGH", atr_pct)
            else:
                return ("NEUTRAL", atr_pct)

        except Exception as e:
            logger.error(f"Error in get_vol_regime: {e}", exc_info=True)
            return ("UNKNOWN", None)

    # ======================================================================
    # HTF Trend (5m LSTM-based robust 3-state)
    # ======================================================================

    def get_htf_trend(self) -> Optional[str]:
        """
        Get HTF (5m) trend using LSTM + EMA-slope + consistency logic.
        Returns: "UP", "DOWN", "RANGE", or None if insufficient data.
        """
        try:
            if len(self._htf_5m_closes) < getattr(config, "HTF_LOOKBACK_BARS", 86):
                return None

            # Ensure LSTM is trained
            if not self._htf_lstm_trained:
                self._train_htf_lstm()

            # Compute trend
            trend = self._compute_htf_trend_lstm()

            # Hysteresis: require 2 consecutive confirmations before changing trend
            if trend != self._htf_pending_trend:
                self._htf_pending_trend = trend
                self._htf_confirm_count = 1
                return self._last_htf_trend  # Keep old trend

            self._htf_confirm_count += 1
            if self._htf_confirm_count >= 2:
                self._last_htf_trend = trend
                return trend
            else:
                return self._last_htf_trend

        except Exception as e:
            logger.error(f"Error in get_htf_trend: {e}", exc_info=True)
            return None

    def _train_htf_lstm(self) -> None:
        """Train HTF LSTM on synthetic labels from EMA-slope + consistency."""
        try:
            if len(self._htf_5m_closes) < getattr(config, "HTF_LOOKBACK_BARS", 86):
                return

            closes = np.asarray([c for (_, c) in self._htf_5m_closes], dtype=np.float64)

            if len(closes) < 50:
                return

            # Synthetic labels using EMA-slope logic
            span = getattr(config, "HTF_EMA_SPAN", 34)
            lookback = getattr(config, "HTF_LOOKBACK_BARS", 24)
            min_slope = getattr(config, "MIN_TREND_SLOPE", 0.0003)
            consistency = getattr(config, "CONSISTENCY_THRESHOLD", 0.60)

            labels = []
            for i in range(len(closes)):
                if i < span + lookback:
                    labels.append(2)  # RANGE (not enough history)
                    continue

                window = closes[i - lookback : i]
                ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                slope = (ema - closes[i - lookback]) / closes[i - lookback]
                above = np.sum(window > ema) / len(window)

                if slope > min_slope and above >= consistency:
                    labels.append(0)  # UPTREND
                elif slope < -min_slope and (1 - above) >= consistency:
                    labels.append(1)  # DOWNTREND
                else:
                    labels.append(2)  # RANGE

            labels = np.asarray(labels, dtype=np.int64)

            # Build features: returns + volume proxy
            returns = np.diff(closes) / closes[:-1]
            returns = np.insert(returns, 0, 0.0)

            # Normalize
            self._htf_norm_mean = float(np.mean(returns))
            self._htf_norm_std = float(np.std(returns)) + 1e-8
            returns_norm = (returns - self._htf_norm_mean) / self._htf_norm_std

            # Prepare sequences
            seq_len = 10
            X_list = []
            y_list = []

            for i in range(seq_len, len(returns_norm)):
                X_list.append(returns_norm[i - seq_len : i].reshape(-1, 1))
                y_list.append(labels[i])

            if len(X_list) == 0:
                return

            # Create tensors with proper gradient handling
            X = torch.tensor(np.asarray(X_list), dtype=torch.float32).requires_grad_(False)
            y = torch.tensor(np.asarray(y_list), dtype=torch.long)

            # Initialize model
            self._htf_lstm = TrendLSTM(
                input_dim=1, hidden_dim=16, num_layers=1, num_classes=3
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self._htf_lstm.parameters(), lr=0.001)

            # Train - use detached copy in each iteration
            self._htf_lstm.train()
            for epoch in range(20):
                optimizer.zero_grad()
                # Create fresh detached copy for each iteration
                X_batch = X.detach().clone()
                outputs = self._htf_lstm(X_batch)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            self._htf_lstm.eval()
            self._htf_lstm_trained = True

            logger.info("HTF LSTM trained successfully")

        except Exception as e:
            if "gradient computation" not in str(e):
                    logger.error(f"Error training HTF LSTM: {e}", exc_info=True)
            else:
                    logger.debug(f"LSTM gradient warning (non-critical): {e}")

    def _compute_htf_trend_lstm(self) -> Optional[str]:
        """Compute HTF trend using trained LSTM."""
        try:
            if not self._htf_lstm_trained or self._htf_lstm is None:
                return None

            closes = np.asarray([c for (_, c) in self._htf_5m_closes], dtype=np.float64)
            if len(closes) < 11:
                return None

            returns = np.diff(closes[-11:]) / closes[-11:-1]
            returns_norm = (returns - self._htf_norm_mean) / self._htf_norm_std

            X = torch.tensor(returns_norm.reshape(1, -1, 1), dtype=torch.float32)

            with torch.no_grad():
                outputs = self._htf_lstm(X)
                pred = int(torch.argmax(outputs, dim=1).item())

            if pred == 0:
                return "UP"
            elif pred == 1:
                return "DOWN"
            else:
                return "RANGE"

        except Exception as e:
            logger.error(f"Error computing HTF LSTM trend: {e}", exc_info=True)
            return None

    # ======================================================================
    # LTF Trend (1m LSTM-based)
    # ======================================================================

    def get_ltf_trend(self) -> Optional[str]:
        """
        Get LTF (1m) trend using LSTM.
        Returns: "UP", "DOWN", "RANGE", or None.
        """
        try:
            if len(self._ltf_1m_closes) < getattr(config, "LTF_LOOKBACK_BARS", 30):
                return None

            if not self._ltf_lstm_trained:
                self._train_ltf_lstm()

            trend = self._compute_ltf_trend_lstm()

            # Hysteresis
            if trend != self._ltf_pending_trend:
                self._ltf_pending_trend = trend
                self._ltf_confirm_count = 1
                return self._last_ltf_trend

            self._ltf_confirm_count += 1
            if self._ltf_confirm_count >= 2:
                self._last_ltf_trend = trend
                return trend
            else:
                return self._last_ltf_trend

        except Exception as e:
            logger.error(f"Error in get_ltf_trend: {e}", exc_info=True)
            return None

    def _train_ltf_lstm(self) -> None:
        """Train LTF LSTM."""
        try:
            if len(self._ltf_1m_closes) < getattr(config, "LTF_LOOKBACK_BARS", 30):
                return

            closes = np.asarray([c for (_, c) in self._ltf_1m_closes], dtype=np.float64)

            if len(closes) < 50:
                return

            span = getattr(config, "LTF_EMA_SPAN", 12)
            lookback = getattr(config, "LTF_LOOKBACK_BARS", 30)
            min_slope = getattr(config, "LTF_MIN_TREND_SLOPE", 0.0002)
            consistency = getattr(config, "LTF_CONSISTENCY_THRESHOLD", 0.52)

            labels = []
            for i in range(len(closes)):
                if i < span + lookback:
                    labels.append(2)
                    continue

                window = closes[i - lookback : i]
                ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                slope = (ema - closes[i - lookback]) / closes[i - lookback]
                above = np.sum(window > ema) / len(window)

                if slope > min_slope and above >= consistency:
                    labels.append(0)
                elif slope < -min_slope and (1 - above) >= consistency:
                    labels.append(1)
                else:
                    labels.append(2)

            labels = np.asarray(labels, dtype=np.int64)

            returns = np.diff(closes) / closes[:-1]
            returns = np.insert(returns, 0, 0.0)

            self._ltf_norm_mean = float(np.mean(returns))
            self._ltf_norm_std = float(np.std(returns)) + 1e-8
            returns_norm = (returns - self._ltf_norm_mean) / self._ltf_norm_std

            seq_len = 10
            X_list = []
            y_list = []

            for i in range(seq_len, len(returns_norm)):
                X_list.append(returns_norm[i - seq_len : i].reshape(-1, 1))
                y_list.append(labels[i])

            if len(X_list) == 0:
                return

            # Create tensors with proper gradient handling
            X = torch.tensor(np.asarray(X_list), dtype=torch.float32).requires_grad_(False)
            y = torch.tensor(np.asarray(y_list), dtype=torch.long)

            self._ltf_lstm = TrendLSTM(
                input_dim=1, hidden_dim=16, num_layers=1, num_classes=3
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self._ltf_lstm.parameters(), lr=0.001)

            self._ltf_lstm.train()
            for epoch in range(20):
                optimizer.zero_grad()
                # Create fresh detached copy for each iteration
                X_batch = X.detach().clone()
                outputs = self._ltf_lstm(X_batch)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            self._ltf_lstm.eval()
            self._ltf_lstm_trained = True

            logger.info("LTF LSTM trained successfully")

        except Exception as e:
            logger.error(f"Error training LTF LSTM: {e}", exc_info=True)

    def _compute_ltf_trend_lstm(self) -> Optional[str]:
        """Compute LTF trend using trained LSTM."""
        try:
            if not self._ltf_lstm_trained or self._ltf_lstm is None:
                return None

            closes = np.asarray([c for (_, c) in self._ltf_1m_closes], dtype=np.float64)
            if len(closes) < 11:
                return None

            returns = np.diff(closes[-11:]) / closes[-11:-1]
            returns_norm = (returns - self._ltf_norm_mean) / self._ltf_norm_std

            X = torch.tensor(returns_norm.reshape(1, -1, 1), dtype=torch.float32)

            with torch.no_grad():
                outputs = self._ltf_lstm(X)
                pred = int(torch.argmax(outputs, dim=1).item())

            if pred == 0:
                return "UP"
            elif pred == 1:
                return "DOWN"
            else:
                return "RANGE"

        except Exception as e:
            logger.error(f"Error computing LTF LSTM trend: {e}", exc_info=True)
            return None

    # ======================================================================
    # Oracle Metric Wrappers (Called by Strategy)
    # ======================================================================

    def compute_liquidity_velocity_multi_tf(self) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """
        Wrapper for oracle's LV computation.
        Returns: (lv_1m, lv_5m, lv_15m, micro_trap)
        """
        try:
            return self._oracle.compute_liquidity_velocity_multi_tf(self)
        except Exception as e:
            logger.error(f"Error computing LV multi-TF: {e}", exc_info=True)
            return (None, None, None, False)

    def compute_norm_cvd(self, window_sec: int = 10) -> Optional[float]:
        """
        Wrapper for oracle's normalized CVD computation.
        Returns: Normalized CVD in [-1, 1] or None
        """
        try:
            return self._oracle.compute_norm_cvd(self, window_sec=window_sec)
        except Exception as e:
            logger.error(f"Error computing norm CVD: {e}", exc_info=True)
            return None

    def compute_hurst_exponent(self, window_ticks: int = 20) -> Optional[float]:
        """
        Wrapper for oracle's Hurst exponent computation.
        Returns: Hurst value or None
        """
        try:
            return self._oracle.compute_hurst_exponent(self, window_ticks=window_ticks)
        except Exception as e:
            logger.error(f"Error computing Hurst: {e}", exc_info=True)
            return None

    def compute_bos_alignment(self, current_price: float) -> Optional[float]:
        """
        Wrapper for oracle's BOS alignment computation.
        Returns: BOS alignment score (0-1) or None
        """
        try:
            return self._oracle.compute_bos_alignment(self, current_price)
        except Exception as e:
            logger.error(f"Error computing BOS alignment: {e}", exc_info=True)
            return None
