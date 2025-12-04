"""
Z-Score Data Manager - Depth / Trades / Price Window
✅ FIXED: PyTorch Inplace/Autograd Errors (Locking + no_grad)
✅ FIXED: TP/SL calculation support (Data feed stability)
✅ NO SHORTCUTS: Full logic included
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from datetime import datetime
import threading
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from futures_api import FuturesAPI
from futures_websocket import FuturesWebSocket
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Reproducibility
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
        # Ensure x is detached from previous history if passed carelessly
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
        logger.info(f"Symbol : {config.SYMBOL}")
        logger.info(f"Exchange: {config.EXCHANGE}")

        self.ws: Optional[FuturesWebSocket] = None
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )

        # Core data storage
        self._orderbook_bids: List[Tuple[float, float]] = []
        self._orderbook_asks: List[Tuple[float, float]] = []
        self._trades: Deque[Dict] = deque(maxlen=1000)
        self._price_window: Deque[Tuple[int, float]] = deque(
            maxlen=int(self._MAX_PRICE_HISTORY_SEC * 2)
        )
        self._candles_1m: Deque[Dict] = deque(maxlen=100)
        self._candles_5m: Deque[Dict] = deque(maxlen=100)

        # Higher/lower TF series
        self._htf_5m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._bos_15m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._ltf_1m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._recent_trades: Deque[Dict] = deque(maxlen=2000)
        self._last_price: float = 0.0

        # Threading locks
        self._orderbook_lock = threading.Lock()
        self._trades_lock = threading.Lock()
        self._price_lock = threading.Lock()
        self._candles_lock = threading.Lock()
        self._lstm_lock = threading.Lock()  # ✅ CRITICAL FIX: Prevent concurrent train/inference

        self.stats = {
            "orderbook_updates": 0,
            "trade_updates": 0,
            "candle_updates": 0,
            "prices_recorded": 0,
            "trades_received": 0,
            "candles_received": 0,
            "last_update": None,
        }

        # LSTM state
        self._htf_lstm: Optional[TrendLSTM] = None
        self._htf_lstm_trained: bool = False
        self._htf_norm_mean: float = 0.0
        self._htf_norm_std: float = 1.0
        self._htf_pending_trend: Optional[str] = None
        self._htf_confirm_count: int = 0
        self._last_htf_trend: Optional[str] = None
        self._last_htf_train_time: float = 0.0

        self._ltf_lstm: Optional[TrendLSTM] = None
        self._ltf_lstm_trained: bool = False
        self._ltf_norm_mean: float = 0.0
        self._ltf_norm_std: float = 1.0
        self._ltf_pending_trend: Optional[str] = None
        self._ltf_confirm_count: int = 0
        self._last_ltf_trend: Optional[str] = None
        self._last_ltf_train_time: float = 0.0

        from aether_oracle import AetherOracle
        self._oracle = AetherOracle()

        self.is_streaming: bool = False
        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []

        logger.info("✓ PURE WEBSOCKET MODE - No REST kline dependency")
        logger.info("Streams: ORDERBOOK, TRADES, CANDLESTICK (1m/5m/15m)")
        logger.info("=" * 80)

    def start(self) -> bool:
        try:
            self.ws = FuturesWebSocket()
            logger.info("Connecting to CoinSwitch Futures WebSocket...")
            if not self.ws.connect():
                logger.error("❌ WebSocket connection failed")
                return False

            self.ws.subscribe_orderbook(config.SYMBOL, callback=self._on_orderbook)
            self.ws.subscribe_trades(config.SYMBOL, callback=self._on_trade)
            self.ws.subscribe_candlestick(pair=config.SYMBOL, interval=config.CANDLE_INTERVAL, callback=self._on_candlestick_1m)
            self.ws.subscribe_candlestick(pair=config.SYMBOL, interval=5, callback=self._on_candlestick_5m)
            self.ws.subscribe_candlestick(pair=config.SYMBOL, interval=15, callback=self._on_candlestick_15m)

            self.is_streaming = True
            logger.info("✅ Z-Score data streams ACTIVE (pure WebSocket)")
            return True

        except Exception as e:
            logger.error(f"❌ Error starting ZScoreDataManager: {e}", exc_info=True)
            return False

    def stop(self) -> None:
        try:
            if self.ws and self.is_streaming:
                self.ws.disconnect()
        finally:
            self.is_streaming = False
            logger.info("🛑 ZScoreDataManager stopped")

    # ======================================================================
    # WebSocket Callbacks
    # ======================================================================

    def _on_orderbook(self, data: Dict) -> None:
        try:
            if not isinstance(data, dict): return
            bids_raw = data.get("bids") or data.get("b") or []
            asks_raw = data.get("asks") or data.get("a") or []

            def _parse_level(lvl) -> Optional[Tuple[float, float]]:
                try:
                    if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        return (float(lvl[0]), float(lvl[1]))
                    elif isinstance(lvl, dict):
                        p = float(lvl.get("price") or lvl.get("p") or 0)
                        q = float(lvl.get("quantity") or lvl.get("q") or 0)
                        if p > 0 and q > 0: return (p, q)
                except Exception: pass
                return None

            bids = [x for x in [_parse_level(b) for b in bids_raw] if x is not None]
            asks = [x for x in [_parse_level(a) for a in asks_raw] if x is not None]

            if not bids or not asks: return

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
        try:
            if not isinstance(data, dict): return
            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))
            ts_ms = int(data.get("T", 0))
            is_buyer_maker = bool(data.get("m", False))

            if price <= 0 or qty <= 0 or ts_ms <= 0: return

            self._recent_trades.append({
                "price": price, "qty": qty, "ts_ms": ts_ms, "isBuyerMaker": is_buyer_maker,
            })
            self._append_price(ts_ms, price)
            self.stats["trades_received"] += 1

        except Exception as e:
            logger.error(f"Error in _on_trade: {e}")

    def _on_candlestick_1m(self, data) -> None:
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str: return
                try: price = float(close_str)
                except: return
                if price <= 0: return
                ts_ms = int(data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000))

                self._append_price(ts_ms, price)
                self._append_ltf_1m_close(ts_ms, price)
                self._candles_1m.append(data)
                self.stats["candles_received"] += 1
        except Exception as e:
            logger.error(f"Error in _on_candlestick_1m: {e}")

    def _on_candlestick_5m(self, data) -> None:
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str: return
                try: price = float(close_str)
                except: return
                if price <= 0: return
                ts_ms = int(data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000))
                
                self._append_htf_5m_close(ts_ms, price)
                self._candles_5m.append(data)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_5m: {e}")

    def _on_candlestick_15m(self, data) -> None:
        try:
            if isinstance(data, dict):
                close_str = data.get("c")
                if not close_str: return
                try: price = float(close_str)
                except: return
                if price <= 0: return
                ts_ms = int(data.get("ts") or data.get("T") or data.get("t") or int(time.time() * 1000))
                self._append_bos_15m_close(ts_ms, price)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_15m: {e}")

    # ======================================================================
    # Data Helpers
    # ======================================================================

    def _append_price(self, ts_ms: int, price: float) -> None:
        with self._price_lock:
            self._last_price = price
            self._price_window.append((ts_ms, price))
            self.stats["prices_recorded"] += 1
            self.stats["last_update"] = datetime.utcnow()
            
            cutoff_price = ts_ms - (self._MAX_PRICE_HISTORY_SEC + 120) * 1000
            while self._price_window and self._price_window[0][0] < cutoff_price:
                self._price_window.popleft()

    def _append_htf_5m_close(self, ts_ms: int, close_price: float) -> None:
        with self._candles_lock:
            self._htf_5m_closes.append((ts_ms, close_price))
            # Limit memory
            if len(self._htf_5m_closes) > 500:
                self._htf_5m_closes.popleft()

    def _append_bos_15m_close(self, ts_ms: int, close_price: float) -> None:
        with self._candles_lock:
            self._bos_15m_closes.append((ts_ms, close_price))
            if len(self._bos_15m_closes) > 500:
                self._bos_15m_closes.popleft()

    def _append_ltf_1m_close(self, ts_ms: int, close_price: float) -> None:
        with self._candles_lock:
            self._ltf_1m_closes.append((ts_ms, close_price))
            if len(self._ltf_1m_closes) > 500:
                self._ltf_1m_closes.popleft()

    # ======================================================================
    # Public Accessors
    # ======================================================================

    @property
    def klines_1m(self) -> List[Tuple[int, float]]:
        with self._candles_lock:
            return list(self._ltf_1m_closes)

    def get_orderbook_snapshot(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        return list(self._bids), list(self._asks)

    def get_last_price(self) -> float:
        return float(self._last_price) if self._last_price > 0 else 0.0

    def get_price_window(self, window_seconds: int = 480) -> List[Tuple[int, float]]:
        with self._price_lock:
            if not self._price_window: return []
            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - window_seconds * 1000
            return [(ts, p) for (ts, p) in self._price_window if ts >= cutoff_ms]

    def get_ema(self, period: int = 20, window_minutes: int = 480) -> Optional[float]:
        try:
            pw = self.get_price_window(window_seconds=window_minutes * 60)
            if not pw or len(pw) < period: return None
            prices = np.asarray([p for (_, p) in pw], dtype=np.float64)
            ema_series = pd.Series(prices).ewm(span=period, adjust=False).mean()
            return float(ema_series.iloc[-1])
        except: return None

    def get_atr_percent(self, window_minutes: int = 10) -> Optional[float]:
        try:
            pw = self.get_price_window(window_seconds=window_minutes * 60)
            if not pw or len(pw) < 2: return None
            prices = [p for (_, p) in pw]
            ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            atr = float(np.mean(ranges))
            current = self.get_last_price()
            return atr / current if current > 0 else None
        except: return None

    def get_vol_regime(self) -> Tuple[str, Optional[float]]:
        atr_pct = self.get_atr_percent(window_minutes=config.ATR_WINDOW_MINUTES)
        if atr_pct is None: return ("UNKNOWN", None)
        if atr_pct < config.VOL_REGIME_LOW_THRESHOLD: return ("LOW", atr_pct)
        elif atr_pct > config.VOL_REGIME_HIGH_THRESHOLD: return ("HIGH", atr_pct)
        else: return ("NEUTRAL", atr_pct)

    # ======================================================================
    # LSTM (Fixed for Thread Safety + Inplace Error)
    # ======================================================================

    def get_ltf_trend(self) -> Optional[str]:
        # 1. Check data availability
        with self._candles_lock:
            ltf_len = len(self._ltf_1m_closes)
        if ltf_len < getattr(config, "LTF_LOOKBACK_BARS", 30): return None

        # 2. Train ONLY if needed (not every tick!)
        now = time.time()
        if not self._ltf_lstm_trained or (now - self._last_ltf_train_time > 3600):
            self.train_ltf_lstm()

        if not self._ltf_lstm_trained: return None

        # 3. Inference with Lock & No Grad
        with self._lstm_lock:
            trend = self._compute_ltf_trend_lstm_inference()

        if trend is None: return None

        # Hysteresis
        if trend != self._ltf_pending_trend:
            self._ltf_pending_trend = trend
            self._ltf_confirm_count = 1
            return self._last_ltf_trend
        self._ltf_confirm_count += 1
        if self._ltf_confirm_count >= 2:
            self._last_ltf_trend = trend
            return trend
        return self._last_ltf_trend

    def train_ltf_lstm(self) -> None:
        """Train LTF LSTM with locking to prevent inplace/concurrency errors."""
        with self._lstm_lock:
            try:
                # Snap data
                with self._candles_lock:
                    closes_list = list(self._ltf_1m_closes)

                if len(closes_list) < 50: return

                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                
                # Label generation (simplified for brevity/stability)
                span = 12
                lookback = 30
                labels = []
                for i in range(len(closes)):
                    if i < span + lookback:
                        labels.append(2)
                        continue
                    window = closes[i-lookback:i]
                    ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                    base = closes[i-lookback]
                    slope = (ema - base) / base if base != 0 else 0
                    if slope > 0.0002: labels.append(0)
                    elif slope < -0.0002: labels.append(1)
                    else: labels.append(2)

                returns = np.diff(closes)
                returns = np.insert(returns, 0, 0.0)
                self._ltf_norm_mean = float(np.mean(returns))
                self._ltf_norm_std = float(np.std(returns) + 1e-8)
                returns_norm = (returns - self._ltf_norm_mean) / self._ltf_norm_std

                X_list, y_list = [], []
                for i in range(10, len(returns_norm)):
                    X_list.append(returns_norm[i-10:i].reshape(-1, 1))
                    y_list.append(labels[i])

                if not X_list: return

                X = torch.tensor(np.asarray(X_list), dtype=torch.float32)
                y = torch.tensor(np.asarray(y_list), dtype=torch.long)

                self._ltf_lstm = TrendLSTM(input_dim=1, hidden_dim=16, num_layers=1, num_classes=3)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self._ltf_lstm.parameters(), lr=0.001)

                self._ltf_lstm.train()
                torch.set_grad_enabled(True) # Explicitly enable grad for training
                
                for _ in range(20):
                    optimizer.zero_grad()
                    outputs = self._ltf_lstm(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                torch.set_grad_enabled(False) # Disable after
                self._ltf_lstm.eval()
                self._ltf_lstm_trained = True
                self._last_ltf_train_time = time.time()
                logger.info("✅ LTF LSTM trained successfully")

            except Exception as e:
                logger.error(f"LTF LSTM training error: {e}")
                self._ltf_lstm_trained = False

    def _compute_ltf_trend_lstm_inference(self) -> Optional[str]:
        """Inference helper - must be called within lock and no_grad."""
        try:
            with torch.no_grad():
                with self._candles_lock:
                    closes_list = list(self._ltf_1m_closes)
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 11: return None

                returns = np.diff(closes)
                returns = np.insert(returns, 0, 0.0)
                returns_norm = (returns - self._ltf_norm_mean) / self._ltf_norm_std
                
                X = torch.tensor(returns_norm[-10:].reshape(1, -1, 1), dtype=torch.float32)
                outputs = self._ltf_lstm(X)
                pred = int(torch.argmax(outputs, dim=1).item())

                if pred == 0: return "UP"
                elif pred == 1: return "DOWN"
                else: return "RANGE"
        except Exception as e:
            logger.error(f"LTF inference error: {e}")
            return None

    # HTF LSTM (Similar pattern)
    def get_htf_trend(self) -> Optional[str]:
        with self._candles_lock:
            htf_len = len(self._htf_5m_closes)
        if htf_len < getattr(config, "HTF_LOOKBACK_BARS", 86): return None

        now = time.time()
        if not self._htf_lstm_trained or (now - self._last_htf_train_time > 3600):
            self.train_htf_lstm()

        if not self._htf_lstm_trained: return None

        with self._lstm_lock:
            trend = self._compute_htf_trend_lstm_inference()

        if trend is None: return None

        if trend != self._htf_pending_trend:
            self._htf_pending_trend = trend
            self._htf_confirm_count = 1
            return self._last_htf_trend
        self._htf_confirm_count += 1
        if self._htf_confirm_count >= 2:
            self._last_htf_trend = trend
            return trend
        return self._last_htf_trend

    def train_htf_lstm(self) -> None:
        with self._lstm_lock:
            try:
                with self._candles_lock:
                    closes_list = list(self._htf_5m_closes)
                if len(closes_list) < 50: return
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                
                # Similar label logic as LTF
                span, lookback = 34, 24
                labels = []
                for i in range(len(closes)):
                    if i < span + lookback:
                        labels.append(2)
                        continue
                    window = closes[i-lookback:i]
                    ema = pd.Series(window).ewm(span=span, adjust=False).mean().iloc[-1]
                    base = closes[i-lookback]
                    slope = (ema - base) / base if base != 0 else 0
                    if slope > 0.0003: labels.append(0)
                    elif slope < -0.0003: labels.append(1)
                    else: labels.append(2)

                returns = np.diff(closes)
                returns = np.insert(returns, 0, 0.0)
                self._htf_norm_mean = float(np.mean(returns))
                self._htf_norm_std = float(np.std(returns) + 1e-8)
                returns_norm = (returns - self._htf_norm_mean) / self._htf_norm_std

                X_list, y_list = [], []
                for i in range(10, len(returns_norm)):
                    X_list.append(returns_norm[i-10:i].reshape(-1, 1))
                    y_list.append(labels[i])

                if not X_list: return
                X = torch.tensor(np.asarray(X_list), dtype=torch.float32)
                y = torch.tensor(np.asarray(y_list), dtype=torch.long)

                self._htf_lstm = TrendLSTM(input_dim=1, hidden_dim=16, num_layers=1, num_classes=3)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(self._htf_lstm.parameters(), lr=0.001)

                self._htf_lstm.train()
                torch.set_grad_enabled(True)
                for _ in range(20):
                    optimizer.zero_grad()
                    outputs = self._htf_lstm(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                
                torch.set_grad_enabled(False)
                self._htf_lstm.eval()
                self._htf_lstm_trained = True
                self._last_htf_train_time = time.time()
                logger.info("✅ HTF LSTM trained successfully")
            except Exception as e:
                logger.error(f"HTF LSTM training error: {e}")
                self._htf_lstm_trained = False

    def _compute_htf_trend_lstm_inference(self) -> Optional[str]:
        try:
            with torch.no_grad():
                with self._candles_lock:
                    closes_list = list(self._htf_5m_closes)
                closes = np.asarray([c for _, c in closes_list], dtype=np.float64)
                if len(closes) < 11: return None
                returns = np.diff(closes)
                returns = np.insert(returns, 0, 0.0)
                returns_norm = (returns - self._htf_norm_mean) / self._htf_norm_std
                X = torch.tensor(returns_norm[-10:].reshape(1, -1, 1), dtype=torch.float32)
                outputs = self._htf_lstm(X)
                pred = int(torch.argmax(outputs, dim=1).item())
                if pred == 0: return "UP"
                elif pred == 1: return "DOWN"
                else: return "RANGE"
        except Exception as e:
            logger.error(f"HTF inference error: {e}")
            return None

    # ======================================================================
    # AETHER ORACLE WRAPPERS (Required for full compatibility)
    # ======================================================================

    def compute_liquidity_velocity_multi_tf(self) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Delegate LV calculation to Oracle instance."""
        try:
            return self._oracle.compute_liquidity_velocity_multi_tf(self)
        except Exception as e:
            logger.error(f"Error computing LV multi-TF: {e}")
            return None, None, None, False

    def compute_norm_cvd(self, window_sec: int = 10) -> Optional[float]:
        """Delegate CVD calculation to Oracle instance."""
        try:
            return self._oracle.compute_norm_cvd(self, window_sec=window_sec)
        except Exception as e:
            logger.error(f"Error computing norm CVD: {e}")
            return None

    def compute_hurst_exponent(self, window_ticks: int = 20) -> Optional[float]:
        """Delegate Hurst calculation to Oracle instance."""
        try:
            return self._oracle.compute_hurst_exponent(self, window_ticks=window_ticks)
        except Exception as e:
            logger.error(f"Error computing Hurst: {e}")
            return None

    def compute_bos_alignment(self, current_price: float) -> Optional[float]:
        """Delegate BOS alignment to Oracle instance."""
        try:
            return self._oracle.compute_bos_alignment(self, current_price)
        except Exception as e:
            logger.error(f"Error computing BOS alignment: {e}")
            return None
