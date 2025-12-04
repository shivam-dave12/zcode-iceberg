"""
Z-Score Data Manager - Enhanced Volume-Aware Edition

NEW FEATURES:
- Volume regime detection (LOW/HIGH/NEUTRAL)
- Event-driven callbacks for strategy updates
- Normalized signal scoring for probabilistic gauntlet
- LTF trend removed per spec (HTF only)
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from datetime import datetime
from scipy import stats as scipy_stats

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

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
import random
random.seed(42)


class TrendLSTM(nn.Module):
    """LSTM for 3-state trend: UP/DOWN/RANGE"""
    
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
    """
    Enhanced data manager with volume regime detection and event-driven updates.
    """

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
        logger.info("Z-SCORE DATA MANAGER - VOLUME-AWARE ENHANCED EDITION")
        logger.info("=" * 80)
        logger.info(f"Symbol : {config.SYMBOL}")
        logger.info(f"Exchange : {config.EXCHANGE}")
        logger.info("Streams : ORDERBOOK, TRADES, CANDLESTICK (1m, 5m, 15m)")
        logger.info("Features: VOL REGIME DETECTION, EVENT-DRIVEN CALLBACKS")
        logger.info("=" * 80)

        self.api: FuturesAPI = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )
        self.ws: Optional[FuturesWebSocket] = None

        # Core data structures
        self._bids: List[Tuple[float, float]] = []
        self._asks: List[Tuple[float, float]] = []
        self._recent_trades: deque = deque(maxlen=50000)
        self._price_window: deque = deque(maxlen=50000)
        self._last_price: float = 0.0

        # Candle buffers
        self._htf_5m_closes: deque = deque(maxlen=1000)
        self._bos_15m_closes: deque = deque(maxlen=1000)
        self._ltf_1m_closes: deque = deque(maxlen=3000)  # Keep for EMA/ATR

        # LSTM models (HTF only, LTF disabled per spec)
        self._htf_lstm: Optional[TrendLSTM] = None
        self._htf_lstm_trained: bool = False
        self._htf_norm_mean: float = 0.0
        self._htf_norm_std: float = 1.0

        # HTF hysteresis state
        self._last_htf_trend: Optional[str] = None
        self._htf_confirm_count: int = 0
        self._htf_pending_trend: Optional[str] = None

        # Volume regime state (NEW)
        self._current_vol_regime: str = config.VOL_REGIME_NEUTRAL
        self._last_atr_pct: Optional[float] = None

        # Event-driven callbacks (NEW)
        self._strategy_update_callback: Optional[Callable] = None

        self.is_streaming: bool = False
        self.stats: Dict = {
            "orderbook_updates": 0,
            "trades_received": 0,
            "candles_received": 0,
            "prices_recorded": 0,
            "last_update": None,
        }

    # ======================================================================
    # Event-driven callback registration (NEW)
    # ======================================================================

    def register_strategy_callback(self, callback: Callable) -> None:
        """
        Register strategy update callback for event-driven execution.
        Called on every orderbook/trade update with fresh data.
        """
        self._strategy_update_callback = callback
        logger.info("✓ Strategy callback registered for event-driven updates")

    def _trigger_strategy_update(self) -> None:
        """Fire strategy callback if registered (non-blocking)."""
        if self._strategy_update_callback is not None:
            try:
                self._strategy_update_callback()
            except Exception as e:
                logger.error(f"Error in strategy callback: {e}", exc_info=True)

    # ======================================================================
    # Volume Regime Detection (NEW)
    # ======================================================================

    def get_vol_regime(self, atr_pct: Optional[float] = None) -> str:
        """
        Classify current volatility regime: LOW, HIGH, or NEUTRAL.
        
        Args:
            atr_pct: Optional pre-computed ATR%; if None, computed fresh
            
        Returns:
            One of config.VOL_REGIME_LOW, VOL_REGIME_HIGH, VOL_REGIME_NEUTRAL
        """
        if atr_pct is None:
            try:
                atr_pct = self.get_atr_percent()
            except Exception as e:
                logger.error(f"Error computing ATR for regime: {e}", exc_info=True)
                return self._current_vol_regime  # Return cached

        if atr_pct is None:
            return self._current_vol_regime

        self._last_atr_pct = atr_pct

        if atr_pct < config.VOL_REGIME_LOW_THRESHOLD:
            regime = config.VOL_REGIME_LOW
        elif atr_pct > config.VOL_REGIME_HIGH_THRESHOLD:
            regime = config.VOL_REGIME_HIGH
        else:
            regime = config.VOL_REGIME_NEUTRAL

        if regime != self._current_vol_regime:
            logger.info(
                f"Volume regime transition: {self._current_vol_regime} → {regime} "
                f"(ATR={atr_pct*100:.3f}%)"
            )
            self._current_vol_regime = regime

        return regime

    def get_dynamic_z_threshold(self, atr_pct: Optional[float] = None) -> float:
        """
        Compute dynamic Z-score threshold based on ATR%.
        
        Formula: BASE + SCALE * ((atr_pct - LOW_THRESH) / LOW_THRESH)
        Clamped to [Z_SCORE_MIN, Z_SCORE_MAX]
        
        Returns:
            Adaptive Z-score threshold for current volatility
        """
        if atr_pct is None:
            regime = self.get_vol_regime()
            if regime == config.VOL_REGIME_LOW:
                return config.LOW_VOL_DELTA_Z_THRESHOLD
            elif regime == config.VOL_REGIME_HIGH:
                return config.HIGH_VOL_DELTA_Z_THRESHOLD
            else:
                return config.BASE_DELTA_Z_THRESHOLD

        # Continuous formula
        base = config.BASE_DELTA_Z_THRESHOLD
        scale = config.Z_SCORE_SCALE_FACTOR
        low_thresh = config.VOL_REGIME_LOW_THRESHOLD

        if atr_pct <= low_thresh:
            z_thresh = config.LOW_VOL_DELTA_Z_THRESHOLD
        else:
            delta = (atr_pct - low_thresh) / low_thresh
            z_thresh = base + scale * delta

        z_thresh = max(config.Z_SCORE_MIN, min(config.Z_SCORE_MAX, z_thresh))
        return z_thresh

    def get_dynamic_wall_mult(self, regime: Optional[str] = None) -> float:
        """Get wall volume multiplier for current regime."""
        if regime is None:
            regime = self._current_vol_regime

        if regime == config.VOL_REGIME_LOW:
            return config.LOW_VOL_WALL_VOLUME_MULT
        elif regime == config.VOL_REGIME_HIGH:
            return config.HIGH_VOL_WALL_VOLUME_MULT
        else:
            return config.BASE_WALL_VOLUME_MULT

    # ======================================================================
    # Normalized Signal Scoring (NEW) - For Probabilistic Gauntlet
    # ======================================================================

    def normalize_signal_cdf(
        self, 
        value: float, 
        threshold: float, 
        std_dev: float = 1.0
    ) -> float:
        """
        Normalize signal value to [0, 1] using CDF of normal distribution.
        
        Args:
            value: Raw signal value
            threshold: Decision threshold
            std_dev: Standard deviation for scaling
            
        Returns:
            Probability score [0, 1]
        """
        try:
            z = (value - threshold) / std_dev
            prob = float(scipy_stats.norm.cdf(z))
            return max(0.0, min(1.0, prob))
        except Exception as e:
            logger.error(f"Error in CDF normalization: {e}", exc_info=True)
            return 0.5  # Neutral on error

    def compute_signal_scores(
        self,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        htf_trend: Optional[str],
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute normalized scores [0, 1] for all base signals.
        
        Returns:
            Dict with keys: 'imbalance', 'wall', 'delta_z', 'touch', 'trend'
            Each value is a probability score for LONG entry (invert for SHORT)
        """
        if regime is None:
            regime = self._current_vol_regime

        scores = {
            'imbalance': 0.0,
            'wall': 0.0,
            'delta_z': 0.0,
            'touch': 0.0,
            'trend': 0.0,
        }

        # Imbalance score
        if imbalance_data is not None:
            imb_val = float(imbalance_data.get("imbalance", 0.0))
            imb_thresh = config.BASE_IMBALANCE_THRESHOLD
            # Use std=0.15 (empirical spread of imbalance values)
            scores['imbalance'] = self.normalize_signal_cdf(
                value=imb_val,
                threshold=imb_thresh,
                std_dev=0.15
            )

        # Wall score
        if wall_data is not None:
            wall_val = float(wall_data.get("bid_wall_strength", 0.0))
            wall_thresh = self.get_dynamic_wall_mult(regime)
            # Use std=1.0 (wall strength units)
            scores['wall'] = self.normalize_signal_cdf(
                value=wall_val,
                threshold=wall_thresh,
                std_dev=1.0
            )

        # Delta Z score
        if delta_data is not None:
            z_val = float(delta_data.get("z_score", 0.0))
            z_thresh = self.get_dynamic_z_threshold()
            # Use std=0.5 (Z-score distribution)
            scores['delta_z'] = self.normalize_signal_cdf(
                value=z_val,
                threshold=z_thresh,
                std_dev=0.5
            )

        # Touch score (inverted: closer = better)
        if touch_data is not None:
            touch_val = float(touch_data.get("bid_distance_ticks", 999.0))
            touch_thresh = config.BASE_TOUCH_THRESHOLD_TICKS
            # Invert: lower distance = higher score
            # Use std=2.0 ticks
            scores['touch'] = 1.0 - self.normalize_signal_cdf(
                value=touch_val,
                threshold=touch_thresh,
                std_dev=2.0
            )

        # Trend score (HTF alignment)
        if htf_trend is not None:
            htf_norm = htf_trend.upper()
            if htf_norm in ("UP", "UPTREND"):
                scores['trend'] = 1.0  # Full alignment for LONG
            elif htf_norm in ("DOWN", "DOWNTREND"):
                scores['trend'] = 0.0  # No alignment for LONG
            elif htf_norm in ("RANGE", "RANGEBOUND"):
                # Apply regime-specific RANGE bonus
                if regime == config.VOL_REGIME_LOW:
                    scores['trend'] = config.LOW_VOL_RANGE_BONUS
                elif regime == config.VOL_REGIME_HIGH:
                    scores['trend'] = config.HIGH_VOL_RANGE_BONUS
                else:
                    scores['trend'] = 0.65  # Neutral bonus
            else:
                scores['trend'] = 0.5  # Unknown = neutral

        return scores

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def start(self) -> bool:
        """Connect WebSocket with event-driven callbacks."""
        try:
            self.ws = FuturesWebSocket()
            logger.info("Connecting to CoinSwitch Futures WebSocket...")
            if not self.ws.connect():
                logger.error("WebSocket connection failed")
                return False

            # Subscribe with event-driven callbacks
            logger.info(f"Subscribing ORDERBOOK: {config.SYMBOL}")
            self.ws.subscribe_orderbook(config.SYMBOL, callback=self._on_orderbook)

            logger.info(f"Subscribing TRADES: {config.SYMBOL}")
            self.ws.subscribe_trades(config.SYMBOL, callback=self._on_trade)

            logger.info(f"Subscribing CANDLESTICKS 1m: {config.SYMBOL}_{config.CANDLE_INTERVAL}")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=config.CANDLE_INTERVAL,
                callback=self._on_candlestick_1m,
            )

            logger.info(f"Subscribing CANDLESTICKS 5m: {config.SYMBOL}_5")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=5,
                callback=self._on_candlestick_5m,
            )

            logger.info(f"Subscribing CANDLESTICKS 15m: {config.SYMBOL}_15")
            self.ws.subscribe_candlestick(
                pair=config.SYMBOL,
                interval=15,
                callback=self._on_candlestick_15m,
            )

            self.is_streaming = True
            logger.info("✓ Z-Score data streams started (event-driven mode)")

            # Warmup from REST
            self._warmup_from_klines_1m()
            self._warmup_htf_klines_5m()
            self._warmup_bos_klines_15m()

            # Initial regime detection
            self.get_vol_regime()

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
    # Warmup methods (unchanged from original)
    # ======================================================================

    def _warmup_from_klines_1m(
        self,
        minutes: Optional[int] = None,
        interval: str = "1",
    ) -> None:
        """Warm up 1m price history for EMA/ATR."""
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

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(f"Klines warmup 1m: unexpected response type {type(resp)}")
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
                    f"Klines warmup 1m complete: seeded {seeded} candles, "
                    f"last_price={self._last_price:.2f}"
                )

        except Exception as e:
            logger.error(f"Error in klines 1m warmup: {e}", exc_info=True)

    def _warmup_htf_klines_5m(self) -> None:
        """Warm up 5m HTF candles."""
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
                "interval": "5",
                "start_time": start_ms,
                "end_time": end_ms,
                "limit": minutes,
            }

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(f"HTF klines 5m warmup: unexpected response type {type(resp)}")
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
                logger.info(f"HTF klines 5m warmup complete: seeded {seeded} candles")

        except Exception as e:
            logger.error(f"Error in HTF 5m klines warmup: {e}", exc_info=True)

    def _warmup_bos_klines_15m(self) -> None:
        """Warm up 15m BOS candles."""
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
                "limit": minutes,
            }

            resp = self.api._make_request(
                method="GET",
                endpoint="/trade/api/v2/futures/klines",
                params=params,
                payload={},
            )

            if not isinstance(resp, dict):
                logger.error(f"BOS klines 15m warmup: unexpected response type {type(resp)}")
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
                logger.info(f"BOS klines 15m warmup complete: seeded {seeded} candles")

        except Exception as e:
            logger.error(f"Error in BOS 15m klines warmup: {e}", exc_info=True)

    # ======================================================================
    # WebSocket callbacks (ENHANCED with event-driven trigger)
    # ======================================================================

    def _on_orderbook(self, data: Dict) -> None:
        """Handle orderbook updates + trigger strategy."""
        try:
            if not isinstance(data, dict):
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

            if self._last_price <= 0:
                mid = (bids[0][0] + asks[0][0]) / 2.0
                ts_ms = int(time.time() * 1000)
                self._append_price(ts_ms, mid)

            # EVENT-DRIVEN: Trigger strategy update on fresh orderbook
            self._trigger_strategy_update()

        except Exception as e:
            logger.error(f"Error in _on_orderbook: {e}", exc_info=True)

    def _on_trade(self, data: Dict) -> None:
        """Handle trades + trigger strategy."""
        try:
            if not isinstance(data, dict):
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

            self._append_price(ts_ms, price)
            self.stats["trades_received"] += 1

            # EVENT-DRIVEN: Trigger strategy update on fresh trade
            self._trigger_strategy_update()

        except Exception as e:
            logger.error(f"Error in _on_trade: {e}", exc_info=True)

    def _on_candlestick_1m(self, data) -> None:
        """Handle 1m candles (for EMA/ATR)."""
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

                # Update volume regime on new candle
                self.get_vol_regime()

        except Exception as e:
            logger.error(f"Error in _on_candlestick_1m: {e}", exc_info=True)

    def _on_candlestick_5m(self, data) -> None:
        """Handle 5m candles (HTF LSTM)."""
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
        """Handle 15m candles (BOS)."""
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
        Returns per-minute volatility as a fraction of price.
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
        Compute ATR over 1‑minute bars for the last `window_minutes`.
        If OHLC is too sparse, falls back to realized volatility percent.
        Also stores the latest ATR% snapshot for volatility regime logic.
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

            # Store snapshot for regime logic
            self._latest_atr_pct = atr_pct
            self._latest_atr_ts_ms = int(df["ts_ms"].iloc[-1])

            return atr_pct
        except Exception as e:
            logger.error(
                f"Error computing ATR {window_minutes}m percent: {e}",
                exc_info=True,
            )
            return self.get_realized_volatility_percent(window_seconds)

    # ======================================================================
    # Volatility regime helpers (NEW)
    # ======================================================================

    def get_latest_atr_snapshot(self) -> Tuple[Optional[float], Optional[int]]:
        """
        Return the latest cached ATR percent and its timestamp (ms).
        This avoids recomputing ATR on every tick for regime checks.
        """
        return self._latest_atr_pct, self._latest_atr_ts_ms

    @staticmethod
    def get_vol_regime_from_value(atr_pct: Optional[float]) -> str:
        """
        Classify volatility regime based on ATR% and config thresholds.

        Returns one of: "LOW", "NEUTRAL", "HIGH".
        If atr_pct is None or non-positive, treats as LOW-vol (conservative).
        """
        if not getattr(config, "ENABLE_VOL_REGIME_LOGIC", False):
            return "NEUTRAL"

        if atr_pct is None or atr_pct <= 0:
            return "LOW"

        low_th = getattr(config, "ATR_LOW_THRESHOLD", 0.0015)
        high_th = getattr(config, "ATR_HIGH_THRESHOLD", 0.0030)

        if atr_pct < low_th:
            return "LOW"
        if atr_pct > high_th:
            return "HIGH"
        return "NEUTRAL"

    def get_vol_regime(self) -> Tuple[str, Optional[float]]:
        """
        Primary public API: get current volatility regime and the ATR%.

        - Uses cached ATR% if recent; otherwise triggers a recomputation.
        - Returns: (regime, atr_pct)
        """
        # If we have a snapshot newer than 60 seconds, reuse it.
        now_ms = int(time.time() * 1000)
        if (
            self._latest_atr_pct is not None
            and self._latest_atr_ts_ms is not None
            and now_ms - self._latest_atr_ts_ms < 60_000
        ):
            regime = self.get_vol_regime_from_value(self._latest_atr_pct)
            return regime, self._latest_atr_pct

        atr_pct = self.get_atr_percent()
        regime = self.get_vol_regime_from_value(atr_pct)
        return regime, atr_pct

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

        closes = [p for (_, p) in self._htf_5m_closes]
        trend = self._lstm_predict_trend(
            model=self._htf_lstm,
            closes=closes,
            mean=self._htf_norm_mean,
            std=self._htf_norm_std,
            seq_len=20,
            prob_threshold=0.55,
        )

        if trend is None:
            return self._last_htf_trend

        # Hysteresis: require just 1 confirmation (fast adaptation)
        if self._last_htf_trend is None:
            self._last_htf_trend = trend
            self._htf_confirm_count = 0
            self._htf_pending_trend = None
            return trend

        if trend == self._last_htf_trend:
            self._htf_confirm_count = 0
            self._htf_pending_trend = None
            return self._last_htf_trend

        if self._htf_pending_trend is None or self._htf_pending_trend != trend:
            self._htf_pending_trend = trend
            self._htf_confirm_count = 1
            return self._last_htf_trend

        self._htf_confirm_count += 1
        if self._htf_confirm_count >= 1:
            self._last_htf_trend = self._htf_pending_trend
            self._htf_pending_trend = None
            self._htf_confirm_count = 0

        return self._last_htf_trend

    # ======================================================================
    # LTF (1m) Trend Filter using 1m closes + LSTM + hysteresis
    # ======================================================================

    def get_ltf_trend(self) -> Optional[str]:
        """
        Compute lower timeframe (1-minute) trend using LSTM over native 1m closes
        with hysteresis, mirroring HTF logic.
        """
        if not self._ltf_1m_closes:
            return self._last_ltf_trend

        if not self._ltf_lstm_trained:
            try:
                closes = [p for (_, p) in self._ltf_1m_closes]
                (
                    self._ltf_lstm,
                    self._ltf_norm_mean,
                    self._ltf_norm_std,
                ) = self._train_lstm_model(
                    closes=closes,
                    seq_len=30,
                    horizon=5,
                    up_thresh=0.0004,
                    down_thresh=0.0004,
                    hidden_dim=64,
                    num_layers=2,
                    epochs=25,
                    lr=0.001,
                )
                self._ltf_lstm_trained = True
                logger.info("LTF LSTM model trained successfully")
            except Exception as e:
                logger.error(f"Error training LTF LSTM: {e}", exc_info=True)
                return self._last_ltf_trend

        closes = [p for (_, p) in self._ltf_1m_closes]
        trend = self._lstm_predict_trend(
            model=self._ltf_lstm,
            closes=closes,
            mean=self._ltf_norm_mean,
            std=self._ltf_norm_std,
            seq_len=30,
            prob_threshold=0.55,
        )

        if trend is None:
            return self._last_ltf_trend

        if self._last_ltf_trend is None:
            self._last_ltf_trend = trend
            self._ltf_confirm_count = 0
            self._ltf_pending_trend = None
            return trend

        if trend == self._last_ltf_trend:
            self._ltf_confirm_count = 0
            self._ltf_pending_trend = None
            return self._last_ltf_trend

        if self._ltf_pending_trend is None or self._ltf_pending_trend != trend:
            self._ltf_pending_trend = trend
            self._ltf_confirm_count = 1
            return self._last_ltf_trend

        self._ltf_confirm_count += 1
        if self._ltf_confirm_count >= 1:
            self._last_ltf_trend = self._ltf_pending_trend
            self._ltf_pending_trend = None
            self._ltf_confirm_count = 0

        return self._last_ltf_trend
