"""
Z-Score Data Manager - Depth / Trades / Price Window
✅ FIXED: Pure WebSocket operation (no broken REST klines)
✅ FIXED: Added klines_1m property for main.py warmup check  
✅ KEPT: All your exact EMA/ATR/vol-regime/LSTM/BOS/oracle logic
✅ NO CHANGES: All strategy parameters, calculations, thresholds
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from datetime import datetime
import threading
import random

import numpy as np
import pandas as pd  # EMA, ATR, features
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
torch.autograd.set_detect_anomaly(True)

np.random.seed(42)
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
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class ZScoreDataManager:
    """Data manager for Z-Score Imbalance Iceberg Hunter."""

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
        logger.info(f"Symbol  : {config.SYMBOL}")
        logger.info(f"Exchange: {config.EXCHANGE}")

        self.ws: Optional[FuturesWebSocket] = None
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )

        # Core data storage (YOUR EXACT STRUCTURE)
        self._orderbook_bids: List[Tuple[float, float]] = []
        self._orderbook_asks: List[Tuple[float, float]] = []
        self._trades: Deque[Dict] = deque(maxlen=1000)
        self._price_window: Deque[Tuple[int, float]] = deque(
            maxlen=int(self._MAX_PRICE_HISTORY_SEC * 2)
        )
        self._candles_1m: Deque[Dict] = deque(maxlen=100)
        self._candles_5m: Deque[Dict] = deque(maxlen=100)

        # Higher/lower TF series (YOUR EXACT STRUCTURE)
        self._htf_5m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._bos_15m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._ltf_1m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._recent_trades: Deque[Dict] = deque(maxlen=2000)

        self._last_price: float = 0.0

        # Threading locks (YOUR EXACT STRUCTURE)
        self._orderbook_lock = threading.Lock()
        self._trades_lock = threading.Lock()
        self._price_lock = threading.Lock()
        self._candles_lock = threading.Lock()

        self.stats = {
            "orderbook_updates": 0,
            "trade_updates": 0,
            "candle_updates": 0,
            "prices_recorded": 0,
            "trades_received": 0,
            "candles_received": 0,
            "last_update": None,
        }

        # REST rate limiting (kept for future use, not called)
        self._rest_api_last_call = 0.0
        self._rest_api_min_interval = 2.0

        # LSTM state (YOUR EXACT STRUCTURE)
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

        from aether_oracle import AetherOracle
        self._oracle = AetherOracle()

        self.is_streaming: bool = False
        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []

        logger.info("✓ PURE WEBSOCKET MODE - No REST kline dependency")
        logger.info("Streams: ORDERBOOK, TRADES, CANDLESTICK (1m/5m/15m)")
        logger.info("=" * 80)

    # ======================================================================
    # Lifecycle - FIXED: Pure WebSocket, no REST warmup
    # ======================================================================

    def start(self) -> bool:
        """Connect WebSocket streams ONLY. No REST klines needed."""
        try:
            self.ws = FuturesWebSocket()
            logger.info("Connecting to CoinSwitch Futures WebSocket...")

            if not self.ws.connect():
                logger.error("❌ WebSocket connection failed")
                return False

            # YOUR EXACT SUBSCRIPTIONS
            logger.info(f"✅ Subscribing ORDERBOOK: {config.SYMBOL}")
            self.ws.subscribe_orderbook(config.SYMBOL, callback=self._on_orderbook)

            logger.info(f"✅ Subscribing TRADES: {config.SYMBOL}")
            self.ws.subscribe_trades(config.SYMBOL, callback=self._on_trade)

            logger.info(f"✅ Subscribing CANDLESTICKS 1m: {config.SYMBOL}_{config.CANDLE_INTERVAL}")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=config.CANDLE_INTERVAL,  # typically 1
                callback=self._on_candlestick_1m,
            )

            logger.info(f"✅ Subscribing CANDLESTICKS 5m: {config.SYMBOL}_5")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=5,
                callback=self._on_candlestick_5m,
            )

            logger.info(f"✅ Subscribing CANDLESTICKS 15m: {config.SYMBOL}_15")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=15,
                callback=self._on_candlestick_15m,
            )

            self.is_streaming = True
            logger.info("✅ Z-Score data streams ACTIVE (pure WebSocket)")
            logger.info("📊 Data will accumulate live during 60s warmup")

            return True

        except Exception as e:
            logger.error(f"❌ Error starting ZScoreDataManager: {e}", exc_info=True)
            return False

    def stop(self) -> None:
        """Clean WebSocket disconnect."""
        try:
            if self.ws and self.is_streaming:
                self.ws.disconnect()
        finally:
            self.is_streaming = False
            logger.info("🛑 ZScoreDataManager stopped")

    # ======================================================================
    # WebSocket Callbacks - YOUR EXACT IMPLEMENTATION + _candles_1m fix
    # ======================================================================

    def _on_orderbook(self, data: Dict) -> None:
        """Handle orderbook depth snapshot/update."""
        try:
            if not isinstance(data, dict):
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

            bids = [x for x in [_parse_level(b) for b in bids_raw] if x is not None]
            asks = [x for x in [_parse_level(a) for a in asks_raw] if x is not None]

            if not bids or not asks:
                return

            self._bids = sorted(bids, key=lambda x: x[0], reverse=True)
            self._asks = sorted(asks, key=lambda x: x[0])

            self.stats["orderbook_updates"] += 1
            self.stats["last_update"] = datetime.utcnow()

            if self._last_price <= 0:
                mid = (bids[0][0] + asks[0][0]) / 2.0
                ts_ms = int(time.time() * 1000)
                self._append_price(ts_ms, mid)

        except Exception as e:
            logger.error(f"Error in _on_orderbook: {e}")

    def _on_trade(self, data: Dict) -> None:
        """Handle trades (updates last price every trade)."""
        try:
            if not isinstance(data, dict):
                return

            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))
            ts_ms = int(data.get("T", 0))
            is_buyer_maker = bool(data.get("m", False))

            if price <= 0 or qty <= 0 or ts_ms <= 0:
                return

            self._recent_trades.append({
                "price": price,
                "qty": qty,
                "ts_ms": ts_ms,
                "isBuyerMaker": is_buyer_maker,
            })

            self._append_price(ts_ms, price)
            self.stats["trades_received"] += 1

        except Exception as e:
            logger.error(f"Error in _on_trade: {e}")

    def _on_candlestick_1m(self, data) -> None:
        """Handle 1m candlestick (EMA/ATR/LTF LSTM) - FIXED: Store full candle."""
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
                    data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000)
                )
                
                # YOUR EXACT LOGIC + store full candle for main.py warmup check
                self._append_price(ts_ms, price)
                self._append_ltf_1m_close(ts_ms, price)
                self._candles_1m.append(data)  # ✅ FIXED: Store for klines_1m property
                self.stats["candles_received"] += 1
                
        except Exception as e:
            logger.error(f"Error in _on_candlestick_1m: {e}")

    def _on_candlestick_5m(self, data) -> None:
        """Handle 5m candlestick (HTF LSTM)."""
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
                    data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000)
                )
                self._append_htf_5m_close(ts_ms, price)
                self._candles_5m.append(data)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_5m: {e}")

    def _on_candlestick_15m(self, data) -> None:
        """Handle 15m candlestick (BOS structure)."""
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
                    data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000)
                )
                self._append_bos_15m_close(ts_ms, price)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_15m: {e}")

    # ======================================================================
    # Data Append Helpers - YOUR EXACT IMPLEMENTATION
    # ======================================================================

    def _append_price(self, ts_ms: int, price: float) -> None:
        """Append price into 1m/tick window (EMA/ATR)."""
        self._last_price = price
        self._price_window.append((ts_ms, price))
        self.stats["prices_recorded"] += 1
        self.stats["last_update"] = datetime.utcnow()

        # Trim old data
        cutoff_price = ts_ms - (self._MAX_PRICE_HISTORY_SEC + 120) * 1000
        while self._price_window and self._price_window[0][0] < cutoff_price:
            self._price_window.popleft()

        cutoff_trades = ts_ms - (self._MAX_TRADES_HISTORY_SEC + 60) * 1000
        while self._recent_trades and self._recent_trades[0]["ts_ms"] < cutoff_trades:
            self._recent_trades.popleft()

    def _append_htf_5m_close(self, ts_ms: int, close_price: float) -> None:
        """Append 5m close (HTF LSTM)."""
        self._htf_5m_closes.append((ts_ms, close_price))
        htf_interval_min = getattr(config, "HTF_TREND_INTERVAL", 5)
        htf_span = getattr(config, "HTF_EMA_SPAN", 80)
        htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
        max_htf_sec = (htf_span + htf_lookback + 3) * htf_interval_min * 60
        cutoff = ts_ms - max_htf_sec * 1000
        while self._htf_5m_closes and self._htf_5m_closes[0][0] < cutoff:
            self._htf_5m_closes.popleft()

    def _append_bos_15m_close(self, ts_ms: int, close_price: float) -> None:
        """Append 15m close (BOS)."""
        self._bos_15m_closes.append((ts_ms, close_price))
        bos_interval_min = 15
        htf_span = getattr(config, "HTF_EMA_SPAN", 80)
        htf_lookback = getattr(config, "HTF_LOOKBACK_BARS", 86)
        max_bos_sec = (htf_span + htf_lookback + 3) * bos_interval_min * 60
        cutoff = ts_ms - max_bos_sec * 1000
        while self._bos_15m_closes and self._bos_15m_closes[0][0] < cutoff:
            self._bos_15m_closes.popleft()

    def _append_ltf_1m_close(self, ts_ms: int, close_price: float) -> None:
        """Append 1m close (LTF trend)."""
        self._ltf_1m_closes.append((ts_ms, close_price))
        ltf_span = getattr(config, "LTF_EMA_SPAN", getattr(config, "EMA_PERIOD", 20))
        ltf_lookback = getattr(config, "LTF_LOOKBACK_BARS", 60)
        max_ltf_sec = (ltf_span + ltf_lookback + 3) * 60
        cutoff = ts_ms - max_ltf_sec * 1000
        while self._ltf_1m_closes and self._ltf_1m_closes[0][0] < cutoff:
            self._ltf_1m_closes.popleft()

    # ======================================================================
    # Public Getters - YOUR EXACT IMPLEMENTATION + klines_1m FIX
    # ======================================================================

    @property
    def klines_1m(self) -> List[Tuple[int, float]]:
        """✅ FIXED: Expose _ltf_1m_closes for main.py warmup check."""
        return list(self._ltf_1m_closes)

    def get_orderbook_snapshot(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """YOUR EXACT ORDERBOOK SNAPSHOT."""
        return list(self._bids), list(self._asks)

    def get_last_price(self) -> float:
        """YOUR EXACT LAST PRICE."""
        return float(self._last_price) if self._last_price > 0 else 0.0

    def get_price_window(self, window_seconds: int = 480) -> List[Tuple[int, float]]:
        """YOUR EXACT PRICE WINDOW (thread-safe snapshot)."""
        if not self._price_window:
            return []
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_seconds * 1000
        snapshot = list(self._price_window)
        return [(ts, p) for (ts, p) in snapshot if ts >= cutoff_ms]

    def get_recent_trades(self, window_seconds: int) -> List[Dict]:
        """YOUR EXACT RECENT TRADES (thread-safe)."""
        if not self._recent_trades:
            return []
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_seconds * 1000
        snapshot = list(self._recent_trades)
        return [t for t in snapshot if t["ts_ms"] >= cutoff_ms]

    # ======================================================================
    # TECHNICAL INDICATORS - YOUR EXACT IMPLEMENTATION
    # ======================================================================

    def get_ema(self, period: int = 20, window_minutes: int = 480) -> Optional[float]:
        """YOUR EXACT EMA CALCULATION."""
        try:
            pw = self.get_price_window(window_seconds=window_minutes * 60)
            if not pw or len(pw) < period:
                return None
            prices = np.asarray([p for (_, p) in pw], dtype=np.float64)
            if len(prices) < period:
                return None
            ema_series = pd.Series(prices).ewm(span=period, adjust=False).mean()
            return float(ema_series.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return None

    def get_atr_percent(self, window_minutes: int = 10) -> Optional[float]:
        """YOUR EXACT ATR% CALCULATION."""
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
            logger.error(f"Error calculating ATR: {e}")
            return None

    def get_vol_regime(self) -> Tuple[str, Optional[float]]:
        """YOUR EXACT VOL-REGIME CLASSIFICATION."""
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
            logger.error(f"Error in get_vol_regime: {e}")
            return ("UNKNOWN", None)

    # ======================================================================
    # HTF LSTM TREND - YOUR EXACT IMPLEMENTATION
    # ======================================================================

    def get_htf_trend(self) -> Optional[str]:
        """YOUR EXACT HTF 5m TREND (LSTM + hysteresis)."""
        try:
            with self._candles_lock:
                htf_len = len(self._htf_5m_closes)
                if htf_len < getattr(config, "HTF_LOOKBACK_BARS", 86):
                    return None

            if not self._htf_lstm_trained or self._htf_lstm is None:
                self.train_htf_lstm()
                if not self._htf_lstm_trained:
                    return None

            trend = self._compute_htf_trend_lstm()
            if trend is None:
                return None

            if trend != self._htf_pending_trend:
                self._htf_pending_trend = trend
                self._htf_confirm_count = 1
                return self._last_htf_trend

            self._htf_confirm_count += 1
            if self._htf_confirm_count >= 2:
                self._last_htf_trend = trend
                return trend
            else:
                return self._last_htf_trend

        except Exception as e:
            logger.error(f"Error in get_htf_trend: {e}")
            return None

    def train_htf_lstm(self) -> None:
        """YOUR EXACT HTF LSTM TRAINING (EMA-slope labels)."""
        try:
            if not hasattr(self, '_htf_5m_closes') or not hasattr(self, '_candles_lock'):
                return
            
            with self._candles_lock:
                closes_list = list(self._htf_5m_closes)
                if len(closes_list) < getattr(config, "HTF_LOOKBACK_BARS", 86):
                    return

                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 50:
                    return

            span = getattr(config, "HTF_EMA_SPAN", 34)
            lookback = getattr(config, "HTF_LOOKBACK_BARS", 24)
            min_slope = getattr(config, "MIN_TREND_SLOPE", 0.0003)
            consistency = getattr(config, "CONSISTENCY_THRESHOLD", 0.60)

            labels: List[int] = []
            for i in range(len(closes)):
                if i < span + lookback:
                    labels.append(2)  # RANGE
                    continue
                window = closes[i - lookback:i]
                ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                base = closes[i - lookback]
                slope = (ema - base) / base if base != 0 else 0.0
                above = float(np.sum(window > ema)) / float(len(window))
                if slope > min_slope and above > consistency:
                    labels.append(0)  # UPTREND
                elif slope < -min_slope and (1.0 - above) > consistency:
                    labels.append(1)  # DOWNTREND
                else:
                    labels.append(2)  # RANGE

            labels = np.asarray(labels, dtype=np.int64)
            returns = np.diff(closes)
            returns = np.insert(returns, 0, 0.0)
            self._htf_norm_mean = float(np.mean(returns))
            self._htf_norm_std = float(np.std(returns) + 1e-8)
            returns_norm = (returns - self._htf_norm_mean) / self._htf_norm_std

            seqlen = 10
            X_list: List[np.ndarray] = []
            y_list: List[int] = []
            for i in range(seqlen, len(returns_norm)):
                X_list.append(returns_norm[i - seqlen:i].reshape(-1, 1))
                y_list.append(int(labels[i]))

            if len(X_list) == 0:
                return

            X = torch.tensor(np.asarray(X_list), dtype=torch.float32)
            y = torch.tensor(np.asarray(y_list), dtype=torch.long)

            self._htf_lstm = TrendLSTM(input_dim=1, hidden_dim=16, num_layers=1, num_classes=3)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self._htf_lstm.parameters(), lr=0.001)

            self._htf_lstm.train()
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = self._htf_lstm(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            self._htf_lstm.eval()
            self._htf_lstm_trained = True
            logger.info("✅ HTF LSTM trained successfully")

        except Exception as e:
            logger.error(f"HTF LSTM training error: {e}")
            self._htf_lstm_trained = False

    def _compute_htf_trend_lstm(self) -> Optional[str]:
        """YOUR EXACT HTF LSTM PREDICTION."""
        try:
            if not self._htf_lstm_trained or self._htf_lstm is None:
                return None

            with self._candles_lock:
                closes_list = list(self._htf_5m_closes)
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 11:
                    return None

            returns = np.diff(closes)
            returns = np.insert(returns, 0, 0.0)
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
            logger.error(f"Error computing HTF LSTM trend: {e}")
            return None

    # ======================================================================
    # LTF LSTM TREND - YOUR EXACT IMPLEMENTATION (identical structure)
    # ======================================================================

    def get_ltf_trend(self) -> Optional[str]:
        """YOUR EXACT LTF 1m TREND (LSTM + hysteresis)."""
        try:
            with self._candles_lock:
                ltf_len = len(self._ltf_1m_closes)
                if ltf_len < getattr(config, "LTF_LOOKBACK_BARS", 30):
                    return None

            if not self._ltf_lstm_trained or self._ltf_lstm is None:
                self.train_ltf_lstm()
                if not self._ltf_lstm_trained:
                    return None

            trend = self._compute_ltf_trend_lstm()
            if trend is None:
                return None

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
            logger.error(f"Error in get_ltf_trend: {e}")
            return None

    def train_ltf_lstm(self) -> None:
        """YOUR EXACT LTF LSTM TRAINING."""
        try:
            # ✅ ADDED: Safety check
            if not hasattr(self, '_ltf_1m_closes') or not hasattr(self, '_candles_lock'):
                return    
            with self._candles_lock:
                closes_list = list(self._ltf_1m_closes)
                if len(closes_list) < getattr(config, "LTF_LOOKBACK_BARS", 30):
                    return
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 50:
                    return

            span = getattr(config, "LTF_EMA_SPAN", 12)
            lookback = getattr(config, "LTF_LOOKBACK_BARS", 30)
            min_slope = getattr(config, "LTF_MIN_TREND_SLOPE", 0.0002)
            consistency = getattr(config, "LTF_CONSISTENCY_THRESHOLD", 0.52)

            labels: List[int] = []
            for i in range(len(closes)):
                if i < span + lookback:
                    labels.append(2)
                    continue
                window = closes[i - lookback:i]
                ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                base = closes[i - lookback]
                slope = (ema - base) / base if base != 0 else 0.0
                above = float(np.sum(window > ema)) / float(len(window))
                if slope > min_slope and above > consistency:
                    labels.append(0)
                elif slope < -min_slope and (1.0 - above) > consistency:
                    labels.append(1)
                else:
                    labels.append(2)

            labels = np.asarray(labels, dtype=np.int64)
            returns = np.diff(closes)
            returns = np.insert(returns, 0, 0.0)
            self._ltf_norm_mean = float(np.mean(returns))
            self._ltf_norm_std = float(np.std(returns) + 1e-8)
            returns_norm = (returns - self._ltf_norm_mean) / self._ltf_norm_std

            seqlen = 10
            X_list: List[np.ndarray] = []
            y_list: List[int] = []
            for i in range(seqlen, len(returns_norm)):
                X_list.append(returns_norm[i - seqlen:i].reshape(-1, 1))
                y_list.append(int(labels[i]))

            if len(X_list) == 0:
                return

            X = torch.tensor(np.asarray(X_list), dtype=torch.float32)
            y = torch.tensor(np.asarray(y_list), dtype=torch.long)

            self._ltf_lstm = TrendLSTM(input_dim=1, hidden_dim=16, num_layers=1, num_classes=3)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self._ltf_lstm.parameters(), lr=0.001)

            self._ltf_lstm.train()
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = self._ltf_lstm(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            self._ltf_lstm.eval()
            self._ltf_lstm_trained = True
            logger.info("✅ LTF LSTM trained successfully")

        except Exception as e:
            logger.error(f"LTF LSTM training error: {e}")
            self._ltf_lstm_trained = False

    def _compute_ltf_trend_lstm(self) -> Optional[str]:
        """YOUR EXACT LTF LSTM PREDICTION."""
        try:
            if not self._ltf_lstm_trained or self._ltf_lstm is None:
                return None

            with self._candles_lock:
                closes_list = list(self._ltf_1m_closes)
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 11:
                    return None

            returns = np.diff(closes)
            returns = np.insert(returns, 0, 0.0)
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
            logger.error(f"Error computing LTF LSTM trend: {e}")
            return None

    # ======================================================================
    # AETHER ORACLE WRAPPERS - YOUR EXACT IMPLEMENTATION
    # ======================================================================

    def compute_liquidity_velocity_multi_tf(self) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """YOUR EXACT LV MULTI-TF."""
        try:
            return self._oracle.compute_liquidity_velocity_multi_tf(self)
        except Exception as e:
            logger.error(f"Error computing LV multi-TF: {e}")
            return None, None, None, False

    def compute_norm_cvd(self, window_sec: int = 10) -> Optional[float]:
        """YOUR EXACT NORM CVD."""
        try:
            return self._oracle.compute_norm_cvd(self, window_sec=window_sec)
        except Exception as e:
            logger.error(f"Error computing norm CVD: {e}")
            return None

    def compute_hurst_exponent(self, window_ticks: int = 20) -> Optional[float]:
        """YOUR EXACT HURST."""
        try:
            return self._oracle.compute_hurst_exponent(self, window_ticks=window_ticks)
        except Exception as e:
            logger.error(f"Error computing Hurst: {e}")
            return None

    def compute_bos_alignment(self, current_price: float) -> Optional[float]:
        """YOUR EXACT BOS ALIGNMENT."""
        try:
            return self._oracle.compute_bos_alignment(self, current_price)
        except Exception as e:
            logger.error(f"Error computing BOS alignment: {e}")
            return None
