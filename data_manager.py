"""
Z-Score Data Manager - Depth / Trades / Price Window

CORRECTED: Matches official CoinSwitch WebSocket formats
UPDATED: Robust Volatility (ATR/Realized) calculation to prevent MISSING data
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime

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
        logger.info(f"Symbol : {config.SYMBOL}")
        logger.info(f"Exchange : {config.EXCHANGE}")
        logger.info("Streams : ORDERBOOK, TRADES, CANDLESTICK")
        logger.info("=" * 80)

        self.api: FuturesAPI = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )
        self.ws: Optional[FuturesWebSocket] = None

        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []
        self._recent_trades: deque = deque(maxlen=50000)
        self._price_window: deque = deque(maxlen=50000)
        self._last_price: float = 0.0

        # Native 5â€‘minute HTF candles: list of (ts_ms, close)
        self._htf_5m_closes: deque = deque(maxlen=1000)
        # Native 1â€‘minute LTF candles: list of (ts_ms, close)
        self._ltf_1m_closes: deque = deque(maxlen=3000)
        # Native 15â€‘minute BOS candles (for BOS alignment / 15m structure)
        self._bos_15m_closes: deque = deque(maxlen=1000)

        # LSTM models + training state
        self._htf_lstm: Optional[TrendLSTM] = None
        self._htf_lstm_trained: bool = False
        self._htf_norm_mean: float = 0.0
        self._htf_norm_std: float = 1.0

        self._ltf_lstm: Optional[TrendLSTM] = None
        self._ltf_lstm_trained: bool = False
        self._ltf_norm_mean: float = 0.0
        self._ltf_norm_std: float = 1.0

        # Hysteresis state for HTF trend
        self._last_htf_trend: Optional[str] = None
        self._htf_confirm_count: int = 0
        self._htf_pending_trend: Optional[str] = None

        # Hysteresis state for LTF trend
        self._last_ltf_trend: Optional[str] = None
        self._ltf_confirm_count: int = 0
        self._ltf_pending_trend: Optional[str] = None

        self.is_streaming: bool = False
        self.stats: Dict = {
            "orderbook_updates": 0,
            "trades_received": 0,
            "candles_received": 0,  # 1â€‘minute candles
            "prices_recorded": 0,
            "last_update": None,
        }

    # ======================================================================
    # Lifecycle
    # ======================================================================

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

            # 1â€‘minute CANDLESTICKS (for EMA/ATR + LTF LSTM)
            logger.info(
                f"Subscribing CANDLESTICKS 1m: {config.SYMBOL}_{config.CANDLE_INTERVAL}"
            )
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=config.CANDLE_INTERVAL,  # typically 1
                callback=self._on_candlestick_1m,
            )

            # 5â€‘minute CANDLESTICKS (HTF LSTM)
            logger.info(f"Subscribing CANDLESTICKS 5m: {config.SYMBOL}_5")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=5,
                callback=self._on_candlestick_5m,
            )

            # 15â€‘minute CANDLESTICKS (BOS / structure)
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

    def _warmup_from_klines_1m(
        self,
        minutes: Optional[int] = None,
        interval: str = "1",
    ) -> None:
        """
        Warm up 1â€‘minute price history using REST /trade/api/v2/futures/klines.
        This feeds _price_window for EMA/ATR and _ltf_1m_closes for LSTM.
        """
        try:
            if minutes is None:
                htf_interval_min = getattr(config, "HTF_TREND_INTERVAL", 5)
                htf_span = getattr(config, "HTF_EMA_SPAN", 80)
                htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
                atr_window = getattr(config, "ATR_WINDOW_MINUTES", 10)
                minutes = max(
                    10,
                    htf_interval_min * (htf_span + htf_lookback + 1),
                    atr_window + 1,
                )

            end_ms = int(time.time() * 1000)
            start_ms = end_ms - minutes * 60 * 1000

            params = {
                "symbol": config.SYMBOL,
                "exchange": config.EXCHANGE,
                "interval": interval,
                "start_time": start_ms,
                "end_time": end_ms,
                "limit": minutes,
            }
            logger.info(
                f"Warmup (1m) from REST klines: symbol={config.SYMBOL}, "
                f"interval={interval}, minutes={minutes}"
            )

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(
                    f"Klines warmup 1m: unexpected response type {type(resp)}"
                )
                return

            data = resp.get("data", [])
            if not data:
                logger.warning("Klines warmup 1m: no data returned")
                return

            def _ts(k):
                return int(k.get("close_time") or k.get("start_time") or 0)

            data_sorted = sorted(data, key=_ts)

            seeded = 0
            for k in data_sorted:
                try:
                    ts_ms = int(k.get("close_time") or k.get("start_time") or 0)
                    close_str = k.get("c") or k.get("close")
                    if ts_ms <= 0 or not close_str:
                        continue
                    close_price = float(close_str)
                    if close_price <= 0:
                        continue

                    self._append_price(ts_ms, close_price)
                    self._append_ltf_1m_close(ts_ms, close_price)
                    seeded += 1
                except Exception:
                    continue

            if seeded > 0:
                logger.info(
                    f"Klines warmup 1m complete: seeded {seeded} candle closes, "
                    f"last_price={self._last_price:.2f}"
                )
            else:
                logger.warning("Klines warmup 1m: no valid candles parsed")

        except Exception as e:
            logger.error(f"Error in klines 1m warmup: {e}", exc_info=True)

    def _warmup_htf_klines_5m(self) -> None:
        """
        Warm up native 5â€‘minute HTF candles from REST.
        Populates _htf_5m_closes with enough bars for LSTM training.
        """
        try:
            htf_interval_min = getattr(config, "HTF_TREND_INTERVAL", 5)
            htf_span = getattr(config, "HTF_EMA_SPAN", 80)
            htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
            minutes = htf_interval_min * (htf_span + htf_lookback + 2)  # small buffer

            end_ms = int(time.time() * 1000)
            start_ms = end_ms - minutes * 60 * 1000

            params = {
                "symbol": config.SYMBOL,
                "exchange": config.EXCHANGE,
                "interval": "5",
                "start_time": start_ms,
                "end_time": end_ms,
                "limit": minutes,  # generous; API will cap
            }
            logger.info(
                f"Warmup (5m HTF) from REST klines: symbol={config.SYMBOL}, "
                f"interval=5, minutes={minutes}"
            )

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(
                    f"HTF klines 5m warmup: unexpected response type {type(resp)}"
                )
                return

            data = resp.get("data", [])
            if not data:
                logger.warning("HTF klines 5m warmup: no data returned")
                return

            def _ts(k):
                return int(k.get("close_time") or k.get("start_time") or 0)

            data_sorted = sorted(data, key=_ts)

            seeded = 0
            for k in data_sorted:
                try:
                    ts_ms = int(k.get("close_time") or k.get("start_time") or 0)
                    close_str = k.get("c") or k.get("close")
                    if ts_ms <= 0 or not close_str:
                        continue
                    close_price = float(close_str)
                    if close_price <= 0:
                        continue

                    self._append_htf_5m_close(ts_ms, close_price)
                    seeded += 1
                except Exception:
                    continue

            if seeded > 0:
                logger.info(
                    f"HTF klines 5m warmup complete: seeded {seeded} candles"
                )
            else:
                logger.warning("HTF klines 5m warmup: no valid candles parsed")

        except Exception as e:
            logger.error(f"Error in HTF 5m klines warmup: {e}", exc_info=True)

    def _warmup_bos_klines_15m(self) -> None:
        """
        Warm up native 15â€‘minute BOS candles from REST for BOS alignment.
        Populates _bos_15m_closes with sufficient history.
        """
        try:
            htf_interval_min = getattr(config, "HTF_TREND_INTERVAL", 5)
            htf_span = getattr(config, "HTF_EMA_SPAN", 80)
            htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
            minutes = htf_interval_min * (htf_span + htf_lookback + 2)

            end_ms = int(time.time() * 1000)
            start_ms = end_ms - minutes * 60 * 1000

            params = {
                "symbol": config.SYMBOL,
                "exchange": config.EXCHANGE,
                "interval": "15",
                "start_time": start_ms,
                "end_time": end_ms,
                "limit": minutes,  # generous; API will cap
            }
            logger.info(
                f"Warmup (15m BOS) from REST klines: symbol={config.SYMBOL}, "
                f"interval=15, minutes={minutes}"
            )

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(
                    f"BOS klines 15m warmup: unexpected response type {type(resp)}"
                )
                return

            data = resp.get("data", [])
            if not data:
                logger.warning("BOS klines 15m warmup: no data returned")
                return

            def _ts(k):
                return int(k.get("close_time") or k.get("start_time") or 0)

            data_sorted = sorted(data, key=_ts)

            seeded = 0
            for k in data_sorted:
                try:
                    ts_ms = int(k.get("close_time") or k.get("start_time") or 0)
                    close_str = k.get("c") or k.get("close")
                    if ts_ms <= 0 or not close_str:
                        continue
                    close_price = float(close_str)
                    if close_price <= 0:
                        continue

                    self._append_bos_15m_close(ts_ms, close_price)
                    seeded += 1
                except Exception:
                    continue

            if seeded > 0:
                logger.info(
                    f"BOS klines 15m warmup complete: seeded {seeded} candles"
                )
            else:
                logger.warning("BOS klines 15m warmup: no valid candles parsed")

        except Exception as e:
            logger.error(f"Error in BOS 15m klines warmup: {e}", exc_info=True)

    # ======================================================================
    # WebSocket callbacks
    # ======================================================================

    def _on_orderbook(self, data: Dict) -> None:
        """Handle orderbook updates."""
        try:
            if not isinstance(data, dict):
                logger.error(f"Orderbook callback non-dict: {type(data)}")
                return

            bids_raw = data.get("b", []) or []
            asks_raw = data.get("a", []) or []

            bids: List[Tuple[float, float]] = []
            asks: List[Tuple[float, float]] = []

            for level in bids_raw:
                if len(level) >= 2:
                    bids.append((float(level[0]), float(level[1])))
            for level in asks_raw:
                if len(level) >= 2:
                    asks.append((float(level[0]), float(level[1])))

            if not bids or not asks:
                return

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            self._bids = bids
            self._asks = asks
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
        """Handle 1â€‘minute candlestick updates (for EMA/ATR and LTF LSTM)."""
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
        """Handle 5â€‘minute candlestick updates (native HTF stream)."""
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
        """Handle 15â€‘minute candlestick updates (BOS / structure)."""
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
        """Append a native 5â€‘minute close into the HTF buffer."""
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
        """Append a native 15â€‘minute close into the BOS buffer."""
        self._bos_15m_closes.append((ts_ms, close_price))

        bos_interval_min = 15
        htf_span = getattr(config, "HTF_EMA_SPAN", 80)
        htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
        max_bos_sec = (htf_span + htf_lookback + 3) * bos_interval_min * 60

        cutoff = ts_ms - max_bos_sec * 1000
        while self._bos_15m_closes and self._bos_15m_closes[0][0] < cutoff:
            self._bos_15m_closes.popleft()

    def _append_ltf_1m_close(self, ts_ms: int, close_price: float) -> None:
        """Append a native 1â€‘minute close into the LTF buffer for 1m trend."""
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

    def get_ema(self, period: int = None) -> Optional[float]:
        """
        Compute EMA over the most recent prices.
        Uses all prices currently in the internal window; requires at least
        `period` samples. Returns None if not enough data.
        """
        if period is None:
            period = getattr(config, "EMA_PERIOD", 20)

        if not self._price_window or len(self._price_window) < period:
            return None

        snapshot = list(self._price_window)
        prices = [p for (_, p) in snapshot]

        lookback = max(period * 3, period)
        prices = prices[-lookback:]

        try:
            s = pd.Series(prices, dtype="float64")
            ema = s.ewm(span=period, adjust=False).mean().iloc[-1]
            return float(ema)
        except Exception as e:
            logger.error(f"Error computing EMA({period}): {e}", exc_info=True)
            return None

    def get_realized_volatility_percent(
        self, window_seconds: int = 600
    ) -> Optional[float]:
        """
        Compute realized volatility (std dev of returns) from raw tick data
        as a robust backup if OHLC resampling fails.
        """
        price_window = self.get_price_window(window_seconds)
        if not price_window or len(price_window) < 10:
            return None

        try:
            prices = np.array([p for _, p in price_window], dtype=np.float64)
            returns = np.diff(prices) / prices[:-1]
            std_dev = np.std(returns)

            duration_sec = (price_window[-1][0] - price_window[0][0]) / 1000.0
            if duration_sec <= 0:
                return float(std_dev)

            ticks_per_min = (len(prices) / duration_sec) * 60.0
            vol_1min = std_dev * np.sqrt(ticks_per_min)
            return float(vol_1min)
        except Exception as e:
            logger.error(f"Error computing realized volatility: {e}", exc_info=True)
            return None

    def get_atr_percent(self, window_minutes: int = None) -> Optional[float]:
        """
        Compute ATR over 1â€‘minute bars for the last `window_minutes`.
        If OHLC is too sparse, falls back to realized volatility percent.
        """
        if window_minutes is None:
            window_minutes = getattr(config, "ATR_WINDOW_MINUTES", 10)

        window_seconds = window_minutes * 2 * 60
        price_window = self.get_price_window(window_seconds=window_seconds)
        if not price_window:
            return None

        try:
            df = pd.DataFrame(price_window, columns=["ts_ms", "price"])
            if df.empty:
                return self.get_realized_volatility_percent(window_seconds)

            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms")
            df.set_index("ts", inplace=True)

            ohlc = df["price"].resample("1min").agg(["first", "max", "min", "last"])
            ohlc.dropna(inplace=True)

            if ohlc.empty or len(ohlc) < window_minutes + 1:
                return self.get_realized_volatility_percent(window_seconds)

            ohlc.columns = ["open", "high", "low", "close"]
            high = ohlc["high"]
            low = ohlc["low"]
            close = ohlc["close"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()

            if tr.empty:
                return self.get_realized_volatility_percent(window_seconds)

            if len(tr) >= window_minutes:
                tr_last = tr.tail(window_minutes)
            else:
                tr_last = tr

            atr = tr_last.mean()
            last_close = close.iloc[-1]
            if last_close <= 0:
                return self.get_realized_volatility_percent(window_seconds)

            atr_pct = float(atr / last_close)
            return atr_pct

        except Exception as e:
            logger.error(
                f"Error computing ATR {window_minutes}m percent: {e}",
                exc_info=True,
            )
            return self.get_realized_volatility_percent(window_seconds)

    # ======================================================================
    # LSTM dataset builders and trainers
    # ======================================================================

    def _build_lstm_dataset(
        self,
        closes: List[float],
        seq_len: int,
        horizon: int,
        up_thresh: float,
        down_thresh: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Build supervised dataset for LSTM from close prices.
        - Normalizes closes with z-score.
        - Labels sequences based on average future return over `horizon` bars.
        """
        prices = np.asarray(closes, dtype=np.float64)
        if len(prices) <= seq_len + horizon:
            raise ValueError("Not enough price history for LSTM dataset")

        mean = float(prices.mean())
        std = float(prices.std() or 1.0)
        norm = (prices - mean) / std

        xs: List[np.ndarray] = []
        ys: List[int] = []

        for i in range(seq_len, len(prices) - horizon):
            seq = norm[i - seq_len : i]
            future_mean = prices[i : i + horizon].mean()
            current = prices[i - 1]
            ret = (future_mean - current) / current

            if ret > up_thresh:
                label = 0  # UPTREND
            elif ret < -down_thresh:
                label = 1  # DOWNTREND
            else:
                label = 2  # RANGEBOUND

            xs.append(seq[:, None])
            ys.append(label)

        if not xs:
            raise ValueError("No labeled samples for LSTM dataset")

        X = np.stack(xs, axis=0)
        y = np.asarray(ys, dtype=np.int64)
        return X, y, mean, std

    def _train_lstm_model(
        self,
        closes: List[float],
        seq_len: int,
        horizon: int,
        up_thresh: float,
        down_thresh: float,
        hidden_dim: int,
        num_layers: int,
        epochs: int,
        lr: float,
    ) -> Tuple[TrendLSTM, float, float]:
        """
        Generic trainer for TrendLSTM on closing prices.
        Returns: model, mean, std
        """
        X, y, mean, std = self._build_lstm_dataset(
            closes=closes,
            seq_len=seq_len,
            horizon=horizon,
            up_thresh=up_thresh,
            down_thresh=down_thresh,
        )

        device = torch.device("cpu")
        model = TrendLSTM(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
        model.to(device)

        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).long().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        return model, mean, std

    def _lstm_predict_trend(
        self,
        model: TrendLSTM,
        closes: List[float],
        mean: float,
        std: float,
        seq_len: int,
        prob_threshold: float,
    ) -> Optional[str]:
        """
        Predict trend from the latest `seq_len` closes using a trained LSTM.
        """
        if len(closes) < seq_len:
            return None

        prices = np.asarray(closes, dtype=np.float64)
        norm = (prices - mean) / (std or 1.0)
        seq = norm[-seq_len:]
        x = torch.from_numpy(seq[None, :, None]).float()

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            max_p = float(probs[idx])

            if max_p < prob_threshold:
                return "RANGEBOUND"

            if idx == 0:
                return "UPTREND"
            elif idx == 1:
                return "DOWNTREND"
            else:
                return "RANGEBOUND"

    # ======================================================================
    # Higher Timeframe (HTF) Trend Filter using 5m LSTM + hysteresis
    # ======================================================================

    def get_htf_trend(self) -> Optional[str]:
        """
        Compute higher timeframe (e.g. 5-minute) trend using an LSTM over
        native 5m closes with 3-state classification (UP/DOWN/RANGE) +
        hysteresis. Hysteresis confirmations reduced to 1 for faster flips.
        """
        if not self._htf_5m_closes:
            return self._last_htf_trend

        # Train once on warm data
        if not self._htf_lstm_trained:
            try:
                closes = [p for (_, p) in self._htf_5m_closes]
                (
                    self._htf_lstm,
                    self._htf_norm_mean,
                    self._htf_norm_std,
                ) = self._train_lstm_model(
                    closes=closes,
                    seq_len=20,
                    horizon=3,
                    up_thresh=0.0005,
                    down_thresh=0.0005,
                    hidden_dim=64,
                    num_layers=2,
                    epochs=30,
                    lr=0.001,
                )
                self._htf_lstm_trained = True
                logger.info("HTF LSTM model trained successfully")
            except Exception as e:
                logger.error(f"Error training HTF LSTM: {e}", exc_info=True)
                return self._last_htf_trend

        # Volume-weighted responsiveness: lower prob_threshold under spikes
        try:
            recent_trades = self.get_recent_trades(window_seconds=60)
            recent_qty = sum(
                float(t.get("qty", 0.0))
                for t in recent_trades
                if float(t.get("qty", 0.0)) > 0.0
            )
            recent_per_sec = recent_qty / 60.0

            baseline_window = min(self._MAX_TRADES_HISTORY_SEC, 900)
            baseline_trades = self.get_recent_trades(window_seconds=baseline_window)
            baseline_qty = sum(
                float(t.get("qty", 0.0))
                for t in baseline_trades
                if float(t.get("qty", 0.0)) > 0.0
            )
            baseline_per_sec = (
                baseline_qty / float(baseline_window) if baseline_window > 0 else 0.0
            )

            if baseline_per_sec > 0.0:
                vol_factor = min(2.0, max(0.5, recent_per_sec / baseline_per_sec))
            else:
                vol_factor = 1.0
        except Exception as e:
            logger.error(f"Error computing HTF volume factor: {e}", exc_info=True)
            vol_factor = 1.0

        base_prob = 0.6
        prob_threshold = max(0.4, base_prob / max(1.0, vol_factor))

        closes = [p for (_, p) in self._htf_5m_closes]
        raw_signal = self._lstm_predict_trend(
            model=self._htf_lstm,
            closes=closes,
            mean=self._htf_norm_mean,
            std=self._htf_norm_std,
            seq_len=20,
            prob_threshold=prob_threshold,
        )

        if raw_signal is None:
            return self._last_htf_trend

        # Hysteresis: 1 confirmation to flip
        if raw_signal == self._last_htf_trend or self._last_htf_trend is None:
            self._htf_confirm_count = 0
            self._htf_pending_trend = None
            self._last_htf_trend = raw_signal
        else:
            if self._htf_pending_trend == raw_signal:
                self._htf_confirm_count += 1
            else:
                self._htf_pending_trend = raw_signal
                self._htf_confirm_count = 1

            if self._htf_confirm_count >= 1:
                self._last_htf_trend = raw_signal
                self._htf_pending_trend = None
                self._htf_confirm_count = 0

        return self._last_htf_trend

    # ======================================================================
    # 1-minute (LTF) Trend Filter using 1m LSTM + hysteresis
    # with volume-weighted responsiveness (mirrors HTF logic)
    # ======================================================================

    def get_ltf_trend(self) -> Optional[str]:
        """
        Compute 1-minute trend using an LSTM over native 1m closes with the
        same 3-state classification + hysteresis. Volume-weighted responsiveness
        is applied: when recent trade volume spikes, probability threshold is
        lowered slightly to flip faster without changing the model itself.
        """
        if not self._ltf_1m_closes:
            return self._last_ltf_trend

        # Train once on warm data
        if not self._ltf_lstm_trained:
            try:
                closes = [p for (_, p) in self._ltf_1m_closes]
                (
                    self._ltf_lstm,
                    self._ltf_norm_mean,
                    self._ltf_norm_std,
                ) = self._train_lstm_model(
                    closes=closes,
                    seq_len=20,
                    horizon=5,
                    up_thresh=0.0004,
                    down_thresh=0.0004,
                    hidden_dim=32,
                    num_layers=1,
                    epochs=30,
                    lr=0.001,
                )
                self._ltf_lstm_trained = True
                logger.info("LTF (1m) LSTM model trained successfully")
            except Exception as e:
                logger.error(f"Error training LTF LSTM: {e}", exc_info=True)
                return self._last_ltf_trend

        # Volume-weighted responsiveness (LTF)
        try:
            recent_trades = self.get_recent_trades(window_seconds=60)
            recent_qty = sum(
                float(t.get("qty", 0.0))
                for t in recent_trades
                if float(t.get("qty", 0.0)) > 0.0
            )
            recent_per_sec = recent_qty / 60.0

            baseline_window = min(self._MAX_TRADES_HISTORY_SEC, 900)
            baseline_trades = self.get_recent_trades(window_seconds=baseline_window)
            baseline_qty = sum(
                float(t.get("qty", 0.0))
                for t in baseline_trades
                if float(t.get("qty", 0.0)) > 0.0
            )
            baseline_per_sec = (
                baseline_qty / float(baseline_window) if baseline_window > 0 else 0.0
            )

            if baseline_per_sec > 0.0:
                vol_factor = min(2.0, max(0.5, recent_per_sec / baseline_per_sec))
            else:
                vol_factor = 1.0
        except Exception as e:
            logger.error(f"Error computing LTF volume factor: {e}", exc_info=True)
            vol_factor = 1.0

        base_prob = 0.6
        prob_threshold = max(0.4, base_prob / max(1.0, vol_factor))

        closes = [p for (_, p) in self._ltf_1m_closes]
        raw_signal = self._lstm_predict_trend(
            model=self._ltf_lstm,
            closes=closes,
            mean=self._ltf_norm_mean,
            std=self._ltf_norm_std,
            seq_len=20,
            prob_threshold=prob_threshold,
        )

        if raw_signal is None:
            return self._last_ltf_trend

        # Hysteresis: 2 confirmations to flip (unchanged for LTF)
        if raw_signal == self._last_ltf_trend or self._last_ltf_trend is None:
            self._ltf_confirm_count = 0
            self._ltf_pending_trend = None
            self._last_ltf_trend = raw_signal
        else:
            if self._ltf_pending_trend == raw_signal:
                self._ltf_confirm_count += 1
            else:
                self._ltf_pending_trend = raw_signal
                self._ltf_confirm_count = 1

            if self._ltf_confirm_count >= 2:
                self._last_ltf_trend = raw_signal
                self._ltf_pending_trend = None
                self._ltf_confirm_count = 0

        return self._last_ltf_trend


    def get_vol_regime(self, atr_pct: Optional[float] = None) -> str:
        """Volatility regime classifier: LOW <0.15%, HIGH >0.30%, NEUTRAL else."""
        if atr_pct is None:
            atr_pct = self.get_atr_percent()
        if atr_pct is None:
            return "NEUTRAL"  # Default neutral if no ATR data
        
        if atr_pct < config.VOL_REGIME_LOW_ATR_PCT:
            return "LOW"
        elif atr_pct > config.VOL_REGIME_HIGH_ATR_PCT:
            return "HIGH"
        else:
            return "NEUTRAL"
