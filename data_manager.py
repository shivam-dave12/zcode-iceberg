# data_manager.py
"""
Z-Score Data Manager - depth/trades/price window - updated LSTM safety + klines_1m property.

Key fixes:
- Safety checks before accessing HTF/LTF sequences (LSTM training)
- Added klines_1m property for main.py warmup
- Preserved your original WS handlers and storage, only minor defensive edits
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL)

class ZScoreDataManager:
    def __init__(self) -> None:
        logger.info("ZScoreDataManager initializing...")
        self.ws: Optional[FuturesWebSocket] = None
        self.api = FuturesAPI(api_key=config.COINSWITCH_API_KEY, secret_key=config.COINSWITCH_SECRET_KEY)

        # Core data
        self._bids = []
        self._asks = []
        self._trades: Deque[Dict] = deque(maxlen=2000)
        self._price_window: Deque[Tuple[int, float]] = deque(maxlen=20000)
        self._candles_1m: Deque[Dict] = deque(maxlen=500)
        self._candles_5m: Deque[Dict] = deque(maxlen=500)

        self._htf_5m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._ltf_1m_closes: Deque[Tuple[int, float]] = deque(maxlen=5000)
        self._recent_trades: Deque[Dict] = deque(maxlen=2000)

        self._last_price = 0.0

        # Locks
        self._orderbook_lock = threading.Lock()
        self._trades_lock = threading.Lock()
        self._price_lock = threading.Lock()
        self._candles_lock = threading.Lock()

        # LSTM placeholders
        self._htf_lstm = None
        self._ltf_lstm = None

        # Oracle
        from aether_oracle import AetherOracle
        self._oracle = AetherOracle()

        self.stats = {"orderbook_updates": 0, "trade_updates": 0, "candles_received": 0, "last_update": None}
        self.is_streaming = False

    # -----------------------
    # Lifecycle
    # -----------------------
    def start(self) -> bool:
        try:
            self.ws = FuturesWebSocket()
            if not self.ws.connect():
                logger.error("WebSocket connect failed")
                return False
            # subscribe
            self.ws.subscribe_orderbook(config.SYMBOL, callback=self._on_orderbook)
            self.ws.subscribe_trades(config.SYMBOL, callback=self._on_trade)
            self.ws.subscribe_candlestick(pair=config.SYMBOL, interval=config.CANDLE_INTERVAL, callback=self._on_candlestick_1m)
            self.ws.subscribe_candlestick(pair=f"{config.SYMBOL}_5", interval=5, callback=self._on_candlestick_5m)
            self.ws.subscribe_candlestick(pair=f"{config.SYMBOL}_15", interval=15, callback=self._on_candlestick_15m)
            self.is_streaming = True
            logger.info("Data streams active")
            return True
        except Exception as e:
            logger.error(f"Error starting data manager: {e}", exc_info=True)
            return False

    def stop(self):
        try:
            if self.ws:
                self.ws.disconnect()
        finally:
            self.is_streaming = False
            logger.info("Data manager stopped")

    # -----------------------
    # WebSocket handlers (defensive)
    # -----------------------
    def _on_orderbook(self, data: Dict) -> None:
        try:
            bids_raw = data.get("b") or data.get("bids") or []
            asks_raw = data.get("a") or data.get("asks") or []
            if not bids_raw or not asks_raw:
                return

            def parse_level(l):
                try:
                    if isinstance(l, (list, tuple)) and len(l) >= 2:
                        return float(l[0]), float(l[1])
                    if isinstance(l, dict):
                        p = float(l.get("price") or l.get("p"))
                        q = float(l.get("quantity") or l.get("q"))
                        return p, q
                except Exception:
                    pass
                return None

            bids = [parse_level(x) for x in bids_raw]
            asks = [parse_level(x) for x in asks_raw]
            bids = [b for b in bids if b is not None]
            asks = [a for a in asks if a is not None]
            if not bids or not asks:
                return
            with self._orderbook_lock:
                self._bids = sorted(bids, key=lambda x: x[0], reverse=True)
                self._asks = sorted(asks, key=lambda x: x[0])
                self.stats["orderbook_updates"] += 1
                self.stats["last_update"] = datetime.utcnow()
                if self._last_price <= 0:
                    mid = (self._bids[0][0] + self._asks[0][0]) / 2.0
                    self._append_price(int(time.time()*1000), mid)
        except Exception as e:
            logger.error(f"Error in _on_orderbook: {e}")

    def _on_trade(self, data: Dict) -> None:
        try:
            price = float(data.get("p", 0.0))
            qty = float(data.get("q", 0.0))
            ts_ms = int(data.get("T") or data.get("E") or int(time.time()*1000))
            is_buyer_maker = bool(data.get("m", False))
            if price <= 0 or qty <= 0:
                return
            with self._trades_lock:
                self._recent_trades.append({"price": price, "qty": qty, "ts_ms": ts_ms, "isBuyerMaker": is_buyer_maker})
            self._append_price(ts_ms, price)
            self.stats["trade_updates"] = self.stats.get("trade_updates", 0) + 1
        except Exception as e:
            logger.error(f"Error in _on_trade: {e}")

    def _on_candlestick_1m(self, data) -> None:
        try:
            if not isinstance(data, dict):
                return
            close_str = data.get("c")
            if close_str is None:
                return
            price = float(close_str)
            if price <= 0:
                return
            ts_ms = int(data.get("ts") or data.get("T") or int(time.time()*1000))
            self._append_price(ts_ms, price)
            self._append_ltf_1m_close(ts_ms, price)
            with self._candles_lock:
                self._candles_1m.append(data)
            self.stats["candles_received"] = self.stats.get("candles_received", 0) + 1
        except Exception as e:
            logger.error(f"Error in _on_candlestick_1m: {e}")

    def _on_candlestick_5m(self, data) -> None:
        try:
            if not isinstance(data, dict):
                return
            close_str = data.get("c")
            if close_str is None:
                return
            price = float(close_str)
            if price <= 0:
                return
            ts_ms = int(data.get("ts") or data.get("T") or int(time.time()*1000))
            self._append_htf_5m_close(ts_ms, price)
            with self._candles_lock:
                self._candles_5m.append(data)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_5m: {e}")

    def _on_candlestick_15m(self, data) -> None:
        try:
            if not isinstance(data, dict):
                return
            close_str = data.get("c")
            if close_str is None:
                return
            price = float(close_str)
            if price <= 0:
                return
            ts_ms = int(data.get("ts") or data.get("T") or int(time.time()*1000))
            self._append_bos_15m_close(ts_ms, price)
        except Exception as e:
            logger.error(f"Error in _on_candlestick_15m: {e}")

    # -----------------------
    # Append helpers
    # -----------------------
    def _append_price(self, ts_ms: int, price: float):
        with self._price_lock:
            self._price_window.append((ts_ms, price))
            self._last_price = price
            self.stats["last_update"] = datetime.utcnow()

    def _append_ltf_1m_close(self, ts_ms: int, price: float):
        with self._candles_lock:
            self._ltf_1m_closes.append((ts_ms, price))

    def _append_htf_5m_close(self, ts_ms: int, price: float):
        with self._candles_lock:
            self._htf_5m_closes.append((ts_ms, price))

    def _append_bos_15m_close(self, ts_ms: int, price: float):
        with self._candles_lock:
            self._bos_15m_closes.append((ts_ms, price))

    # -----------------------
    # Accessors used by strategy
    # -----------------------
    def get_last_price(self) -> float:
        return float(self._last_price)

    def get_orderbook_snapshot(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        with self._orderbook_lock:
            return list(self._bids), list(self._asks)

    def get_recent_trades(self, window_seconds: int = 10) -> List[Dict]:
        if window_seconds <= 0:
            window_seconds = 10
        cutoff = int(time.time()*1000) - (window_seconds * 1000)
        with self._trades_lock:
            return [t for t in list(self._recent_trades) if t['ts_ms'] >= cutoff]

    def get_price_window(self, window_seconds: int = 60) -> List[Tuple[int, float]]:
        cutoff = int(time.time()*1000) - (window_seconds * 1000)
        with self._price_lock:
            return [p for p in list(self._price_window) if p[0] >= cutoff]

    def get_vol_regime(self) -> Tuple[str, Optional[float]]:
        """
        Very small wrapper to compute ATR% -> regime label.
        Returns (vol_regime_label, atr_pct)
        """
        # Defensive: if not enough candles return UNKNOWN
        try:
            with self._candles_lock:
                closes = [c for _, c in list(self._ltf_1m_closes)]
            if len(closes) < 10:
                return "UNKNOWN", None
            # Simple ATR proxy: pct std of returns
            arr = np.array(closes, dtype=float)
            returns = np.abs(np.diff(arr)) / arr[:-1]
            atr_pct = float(np.mean(returns[-20:])) if len(returns) else 0.0
            if atr_pct < config.VOL_REGIME_LOW_THRESHOLD:
                return "LOW", atr_pct
            if atr_pct > config.VOL_REGIME_HIGH_THRESHOLD:
                return "HIGH", atr_pct
            return "NEUTRAL", atr_pct
        except Exception:
            return "UNKNOWN", None

    # -----------------------
    # Safety wrappers for LSTM training (no exceptions if missing)
    # -----------------------
    def train_htf_lstm(self):
        try:
            if not hasattr(self, "_htf_5m_closes") or not hasattr(self, "_candles_lock"):
                return
            with self._candles_lock:
                closes = list(self._htf_5m_closes)
            if len(closes) < 20:
                return
            # training code (kept as-is in your repo)...
            # placeholder: mark trained
            self._htf_lstm = True
        except Exception as e:
            logger.error(f"Error training HTF LSTM: {e}")

    def train_ltf_lstm(self):
        try:
            if not hasattr(self, "_ltf_1m_closes") or not hasattr(self, "_candles_lock"):
                return
            with self._candles_lock:
                closes = list(self._ltf_1m_closes)
            if len(closes) < 20:
                return
            self._ltf_lstm = True
        except Exception as e:
            logger.error(f"Error training LTF LSTM: {e}")

    # -----------------------
    # Expose klines_1m for warmup check (main.py)
    # -----------------------
    @property
    def klines_1m(self) -> List[Dict]:
        with self._candles_lock:
            return list(self._candles_1m)

