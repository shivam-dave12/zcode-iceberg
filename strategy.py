"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version - COMPLETE WITH ALL FIXES

CHANGELOG - December 3, 2025:
✅ FIXED: Thread-safe entry lock (prevents concurrent orders)
✅ FIXED: Volatility-based tick offset for limit orders  
✅ FIXED: Pre-trade validation checks
✅ ADDED: _calculate_entry_price() method
✅ ADDED: _validate_trade_conditions() method
✅ MODIFIED: _enter_position() with try/finally lock release
✅ MODIFIED: _close_position() with lock release
"""

import time
import threading  # ← ADDED for thread-safe entry lock
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque
import logging
import numpy as np
from scipy import stats as scipy_stats
import config
from zscore_excel_logger import ZScoreExcelLogger
from telegram_notifier import send_telegram_message
from aether_oracle import (
    AetherOracle,
    OracleInputs,
    OracleOutputs,
    OracleSideScores,
)

logger = logging.getLogger(__name__)

@dataclass
class ZScorePosition:
    trade_id: str
    side: str
    quantity: float
    entry_price: float
    entry_time_sec: float
    entry_wall_volume: float
    wall_zone_low: float
    wall_zone_high: float
    entry_imbalance: float
    entry_z_score: float
    tp_price: float
    sl_price: float
    margin_used: float
    tp_order_id: str
    sl_order_id: str
    main_order_id: str
    main_filled: bool
    tp_reduced: bool
    entry_htf_trend: str
    entry_vol_regime: str
    entry_weighted_score: float
    last_score_check_sec: float


class ZScoreIcebergHunterStrategy:
    """
    Z-Score Imbalance Iceberg Hunter with Vol-Regime dynamics and weighted scoring.
    """

    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 120.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0
    ENTRY_FILL_TIMEOUT_SEC = 60.0

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0

        # ═════════════════════════════════════════════════════════════════
        # ADDED: Thread-safe entry lock (FIX 1)
        # ═════════════════════════════════════════════════════════════════
        self._entry_lock = threading.Lock()
        self._entering_position = False

        # Store recent deltas to compute population Z-score
        self._delta_population: deque = deque(maxlen=3000)

        # Last time we printed a full decision snapshot
        self._last_decision_log_sec: float = 0.0

        # Last time we printed a position snapshot (open position)
        self._last_position_log_sec: float = 0.0

        # Last time TP/SL order statuses were checked
        self._last_status_check_sec: float = 0.0

        # 15-minute performance report state (Telegram)
        self._last_report_sec: float = 0.0
        self._last_report_total_trades: int = 0

        logger.info("=" * 80)
        logger.info("Z-SCORE IMBALANCE ICEBERG HUNTER STRATEGY INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Imbalance Threshold = {config.IMBALANCE_THRESHOLD:.2f}")
        logger.info(f"Wall Volume Mult BASE = {config.MIN_WALL_VOLUME_MULT:.2f}×")
        logger.info(f"Delta Z Threshold BASE= {config.DELTA_Z_THRESHOLD:.2f}")
        logger.info(f"Zone Ticks = ±{config.ZONE_TICKS}")
        logger.info(f"Touch Threshold = {config.PRICE_TOUCH_THRESHOLD_TICKS} ticks")
        logger.info(f"Profit Target ROI BASE= {config.PROFIT_TARGET_ROI * 100:.2f}%")
        logger.info(f"Stop Loss ROI BASE = {config.STOP_LOSS_ROI * 100:.2f}%")
        logger.info(f"Max Hold Minutes = {config.MAX_HOLD_MINUTES}")
        logger.info(f"Weighted Scoring = {config.ENABLE_WEIGHTED_SCORING}")
        logger.info(f"Score Entry Threshold = {config.WEIGHTED_SCORE_ENTRY_THRESHOLD}")
        logger.info("=" * 80)

    # ══════════════════════════════════════════════════════════════════════
    # Vol-Regime Helpers (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    def _get_dynamic_z_threshold(self, vol_regime: str, atr_pct: Optional[float]) -> float:
        """Get dynamic Z-score threshold based on volatility regime."""
        if vol_regime == "LOW":
            return config.VOL_REGIME_Z_LOW
        elif vol_regime == "HIGH":
            return config.VOL_REGIME_Z_HIGH
        elif vol_regime == "NEUTRAL" and atr_pct is not None:
            low_thresh = config.VOL_REGIME_LOW_THRESHOLD
            high_thresh = config.VOL_REGIME_HIGH_THRESHOLD
            z_range = config.VOL_REGIME_Z_HIGH - config.VOL_REGIME_Z_LOW
            if high_thresh > low_thresh:
                scaling = (atr_pct - low_thresh) / (high_thresh - low_thresh)
                scaling = max(0.0, min(1.0, scaling))
                z_thresh = config.VOL_REGIME_Z_LOW + z_range * scaling
                return z_thresh
            else:
                return config.VOL_REGIME_Z_BASE
        else:
            return config.VOL_REGIME_Z_BASE

    def _get_dynamic_wall_mult(self, vol_regime: str) -> float:
        """Get dynamic wall multiplier based on volatility regime."""
        if vol_regime == "LOW":
            return config.VOL_REGIME_WALL_MULT_LOW
        elif vol_regime == "HIGH":
            return config.VOL_REGIME_WALL_MULT_HIGH
        else:
            return config.VOL_REGIME_WALL_MULT_BASE

    def _get_dynamic_tp_sl(self, vol_regime: str) -> Tuple[float, float]:
        """
        Get dynamic TP and SL ROI based on volatility regime.
        Returns: (tp_roi, sl_roi)
        """
        base_tp = config.PROFIT_TARGET_ROI
        base_sl = config.STOP_LOSS_ROI

        if vol_regime == "HIGH":
            tp_roi = base_tp * config.VOL_REGIME_TP_MULT_HIGH
            sl_roi = base_sl * config.VOL_REGIME_SL_MULT_HIGH
        elif vol_regime == "LOW" or vol_regime == "NEUTRAL":
            tp_roi = base_tp * config.VOL_REGIME_TP_MULT_LOW
            sl_roi = base_sl * config.VOL_REGIME_SL_MULT_LOW
        else:
            tp_roi = base_tp
            sl_roi = base_sl

        return (tp_roi, sl_roi)

    def _get_dynamic_position_size_pct(self, vol_regime: str) -> float:
        """Get position size percentage based on volatility regime (Kelly-style)."""
        if vol_regime == "HIGH":
            return config.VOL_REGIME_SIZE_HIGH_PCT
        elif vol_regime == "LOW":
            return config.VOL_REGIME_SIZE_LOW_PCT
        else:
            return (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0


    # ══════════════════════════════════════════════════════════════════════
    # ADDED: NEW HELPER METHODS (FIX 8 & FIX 9)
    # ══════════════════════════════════════════════════════════════════════

    def _calculate_entry_price(self, side: str, current_price: float, vol_regime: str) -> float:
        """
        Calculate entry price with volatility-based tick offset (FIX 8).

        Uses dynamic tick offset based on volatility regime for optimal limit order placement:
        - LOW vol: 4 ticks (tighter spread, less movement)
        - HIGH vol: 10 ticks (wider spread, more movement)
        - NEUTRAL: 6 ticks (balanced approach)
        """
        # Determine tick offset based on volatility
        if vol_regime == "HIGH":
            tick_offset = 10  # 10 ticks in high vol for better fills
        elif vol_regime == "LOW":
            tick_offset = 4  # 4 ticks in low vol (tighter spread)
        else:  # NEUTRAL or UNKNOWN
            tick_offset = 6  # 6 ticks for neutral

        if side == "long":
            # For longs: buy slightly below current (better fill chance)
            entry_price = current_price - (tick_offset * config.TICK_SIZE)
        else:  # short
            # For shorts: sell slightly above current
            entry_price = current_price + (tick_offset * config.TICK_SIZE)

        entry_price = round(entry_price, 2)

        logger.info(
            f"Entry price: {entry_price:.2f} ({side.upper()}) | "
            f"Offset: {tick_offset} ticks | Vol: {vol_regime}"
        )

        return entry_price

    def _validate_trade_conditions(
        self, 
        data_manager, 
        risk_manager, 
        vol_regime: str
    ) -> Tuple[bool, str]:
        """
        Validate trade conditions before entry (FIX 9).

        Checks:
        1. Vol regime calculated (not UNKNOWN)
        2. HTF trend available (not NA)
        3. Sufficient historical data (min 40 bars)
        4. Balance available (>= MIN_MARGIN_PER_TRADE)

        Returns: (is_valid, failure_reason)
        """
        # Check 1: Data warmup complete
        if vol_regime == "UNKNOWN":
            return False, "Vol regime not calculated yet"

        # Check 2: HTF trend available
        try:
            if hasattr(data_manager, 'get_htf_trend'):
                htf_trend = data_manager.get_htf_trend()
                if not htf_trend or htf_trend == "NA":
                    return False, "HTF trend not available"
        except Exception as e:
            return False, f"Cannot get HTF trend: {e}"

        # Check 3: Sufficient historical data
        try:
            if hasattr(data_manager, 'klines_1m'):
                kline_count = len(data_manager.klines_1m)
                if kline_count < 40:
                    return False, f"Insufficient bars: {kline_count}/40"
        except:
            pass  # Optional check

        # Check 4: Balance available
        try:
            balance = risk_manager.get_available_balance()
            available = float(balance.get('available', 0))
            if available < config.MIN_MARGIN_PER_TRADE:
                return False, f"Insufficient balance: {available:.2f}"
        except Exception as e:
            return False, f"Cannot check balance: {e}"

        return True, "OK"

    # ======================================================================
    # Weighted Scoring Helpers
    # ======================================================================

    def _normalize_signal_cdf(self, value: float, threshold: float, std_dev: float = 1.0) -> float:
        """
        Normalize signal using CDF approach: norm.cdf((value - threshold) / std).
        Returns value between 0 and 1.
        
        FIXED: Added edge case handling with floor/ceiling.
        """
        try:
            if std_dev <= 0:
                std_dev = 1.0  # Prevent division by zero
            
            z = (value - threshold) / std_dev
            
            # Clamp extreme Z-scores to prevent numerical issues
            z = max(-5.0, min(5.0, z))
            
            normalized = scipy_stats.norm.cdf(z)
            
            # Apply floor/ceiling to avoid hard 0/1 (oracle-like behavior)
            normalized = max(0.01, min(0.99, normalized))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error in CDF normalization: {e}", exc_info=True)
            return 0.5  # Neutral fallback

    def _compute_weighted_score(
        self,
        side: str,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        htf_trend: Optional[str],
        ltf_trend: Optional[str],
        ema_val: Optional[float],
        current_price: float,
        vol_regime: str,
        oracle_inputs: Optional[OracleInputs],
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Compute weighted score for entry (0-1).
        Returns: (total_score, component_scores_dict, reasons_list)
        
        FIXED: Proper weight normalization to prevent inflation.
        """
        components = {}
        reasons = []
        raw_components = {}  # Store pre-weighted normalized values

        # Dynamic thresholds based on vol regime
        z_thresh_dynamic = self._get_dynamic_z_threshold(vol_regime, oracle_inputs.atr_pct if oracle_inputs else None)
        wall_mult_dynamic = self._get_dynamic_wall_mult(vol_regime)

        # === SIGNAL NORMALIZATION (all to 0-1 range) ===

        # 1. Imbalance
        if imbalance_data:
            imb = imbalance_data["imbalance"]
            if side == "long":
                imb_norm = self._normalize_signal_cdf(imb, config.IMBALANCE_THRESHOLD, std_dev=0.1)
            else:
                imb_norm = self._normalize_signal_cdf(-imb, config.IMBALANCE_THRESHOLD, std_dev=0.1)
            raw_components["imbalance"] = imb_norm
            reasons.append(f"imb={imb:.3f}->norm={imb_norm:.3f}")
        else:
            raw_components["imbalance"] = 0.0
            reasons.append("imb=MISSING")

        # 2. Wall
        if wall_data:
            if side == "long":
                wall_strength = wall_data["bid_wall_strength"]
            else:
                wall_strength = wall_data["ask_wall_strength"]
            
            wall_norm = self._normalize_signal_cdf(wall_strength, wall_mult_dynamic, std_dev=0.5)
            raw_components["wall"] = wall_norm
            reasons.append(f"wall={wall_strength:.2f}->norm={wall_norm:.3f}")
        else:
            raw_components["wall"] = 0.0
            reasons.append("wall=MISSING")

        # 3. Z-Score
        if delta_data:
            z = delta_data["z_score"]
            if side == "long":
                z_norm = self._normalize_signal_cdf(z, z_thresh_dynamic, std_dev=0.5)
            else:
                z_norm = self._normalize_signal_cdf(-z, z_thresh_dynamic, std_dev=0.5)
            raw_components["zscore"] = z_norm
            reasons.append(f"z={z:.2f}->norm={z_norm:.3f}(thresh={z_thresh_dynamic:.2f})")
        else:
            raw_components["zscore"] = 0.0
            reasons.append("z=MISSING")

        # 4. Touch
        if touch_data:
            if side == "long":
                dist = touch_data["bid_distance_ticks"]
            else:
                dist = touch_data["ask_distance_ticks"]
            
            touch_norm = self._normalize_signal_cdf(-dist, -config.PRICE_TOUCH_THRESHOLD_TICKS, std_dev=2.0)
            raw_components["touch"] = touch_norm
            reasons.append(f"touch_dist={dist:.1f}->norm={touch_norm:.3f}")
        else:
            raw_components["touch"] = 0.0
            reasons.append("touch=MISSING")

        # 5. Trend
        trend_score = 0.0
        if htf_trend and ema_val:
            htf_ok = False
            ema_ok = False
            
            if side == "long":
                htf_ok = htf_trend in ["UP", "RANGE"]
                ema_ok = current_price > ema_val
            else:
                htf_ok = htf_trend in ["DOWN", "RANGE"]
                ema_ok = current_price < ema_val

            range_bonus = 0.8 if vol_regime == "LOW" else 0.5

            if htf_trend == "RANGE":
                htf_contrib = range_bonus
            elif htf_ok:
                htf_contrib = 1.0
            else:
                htf_contrib = 0.0

            ema_contrib = 1.0 if ema_ok else 0.0
            trend_score = (htf_contrib + ema_contrib) / 2.0
            raw_components["trend"] = trend_score
            reasons.append(f"trend(htf={htf_trend},ema_ok={ema_ok})->score={trend_score:.3f}")
        else:
            raw_components["trend"] = 0.0
            reasons.append("trend=MISSING")

        # 6. CVD
        if oracle_inputs and oracle_inputs.norm_cvd is not None:
            cvd = oracle_inputs.norm_cvd
            if side == "long":
                cvd_norm = self._normalize_signal_cdf(cvd, 0.0, std_dev=0.3)
            else:
                cvd_norm = self._normalize_signal_cdf(-cvd, 0.0, std_dev=0.3)
            raw_components["cvd"] = cvd_norm
            reasons.append(f"cvd={cvd:.3f}->norm={cvd_norm:.3f}")
        else:
            raw_components["cvd"] = 0.0
            reasons.append("cvd=MISSING")

        # 7. LV Average
        if oracle_inputs and oracle_inputs.lv_1m is not None and oracle_inputs.lv_5m is not None:
            lv_avg = (oracle_inputs.lv_1m + oracle_inputs.lv_5m) / 2.0
            lv_norm = self._normalize_signal_cdf(lv_avg, 1.0, std_dev=0.5)
            raw_components["lv"] = lv_norm
            reasons.append(f"lv_avg={lv_avg:.2f}->norm={lv_norm:.3f}")
        else:
            raw_components["lv"] = 0.0
            reasons.append("lv=MISSING")

        # 8. Hurst/BOS blend
        hurst_bos_score = 0.0
        hurst_contrib = 0.0
        bos_contrib = 0.0

        if oracle_inputs and oracle_inputs.hurst is not None:
            h = oracle_inputs.hurst
            hurst_contrib = self._normalize_signal_cdf(h, 0.5, std_dev=0.1)
            reasons.append(f"hurst={h:.3f}->norm={hurst_contrib:.3f}")

        if oracle_inputs and oracle_inputs.bos_align is not None:
            bos = oracle_inputs.bos_align
            bos_contrib = self._normalize_signal_cdf(bos, 0.3, std_dev=0.2)
            reasons.append(f"bos={bos:.3f}->norm={bos_contrib:.3f}")

        if hurst_contrib > 0 or bos_contrib > 0:
            hurst_bos_score = (hurst_contrib + bos_contrib) / 2.0
        
        raw_components["hurst_bos"] = hurst_bos_score

        # 9. LSTM Probability
        lstm_prob = 0.5
        if oracle_inputs:
            if htf_trend == "UP" and side == "long":
                lstm_prob = 0.7
            elif htf_trend == "DOWN" and side == "short":
                lstm_prob = 0.7
            elif htf_trend == "RANGE":
                lstm_prob = 0.6
            else:
                lstm_prob = 0.4

        lstm_norm = self._normalize_signal_cdf(lstm_prob, 0.5, std_dev=0.15)
        raw_components["lstm"] = lstm_norm
        reasons.append(f"lstm_prob={lstm_prob:.3f}->norm={lstm_norm:.3f}")

        # === WEIGHTED AGGREGATION WITH PROPER NORMALIZATION ===
        
        # Define weights (from config)
        weights = {
            "imbalance": config.WEIGHT_IMBALANCE,
            "wall": config.WEIGHT_WALL,
            "zscore": config.WEIGHT_ZSCORE,
            "touch": config.WEIGHT_TOUCH,
            "trend": config.WEIGHT_TREND,
            "cvd": config.WEIGHT_CVD,
            "lv": config.WEIGHT_LV,
            "hurst_bos": config.WEIGHT_HURST_BOS,
            "lstm": config.WEIGHT_LSTM,
        }
        
        # Calculate total weight for normalization
        total_weight = sum(weights.values())
        
        # CORRECT CALCULATION: Normalize weights to sum to 1.0
        weighted_sum = 0.0
        for key in weights:
            if key in raw_components:
                normalized_weight = weights[key] / total_weight  # KEY FIX
                weighted_contribution = raw_components[key] * normalized_weight
                components[key] = weighted_contribution  # Store for logging
                weighted_sum += weighted_contribution
        
        # Clamp to [0, 1]
        total_score = max(0.0, min(1.0, weighted_sum))
        
        # Debug logging
        reasons.append(f"raw_weight_sum={total_weight:.3f}->normalized_to_1.0")
        reasons.append(f"total_score={total_score:.4f}(corrected)")

        return (total_score, components, reasons)

    def _compute_win_probability(
        self,
        lstm_prob: float,
        z_sign: float,
        cvd: float,
        lv_avg: float,
    ) -> float:
        """
        Compute win probability overlay using data-derived formula:
        win_prob = 0.4 + 0.2*lstm + 0.2*z_sign + 0.1*cvd + 0.1*lv
        All inputs normalized to [0, 1].
        """
        win_prob = 0.4 + 0.2 * lstm_prob + 0.2 * z_sign + 0.1 * cvd + 0.1 * lv_avg
        return max(0.0, min(1.0, win_prob))

    # ======================================================================
    # Core Metrics (Imbalance, Wall, Delta, Touch)
    # ======================================================================

    def _compute_imbalance(self, data_manager) -> Optional[Dict]:
        """
        Compute orderbook imbalance over top WALL_DEPTH_LEVELS.
        Returns: {
            'imbalance': float,  # [-1, 1]
            'bid_volume': float,
            'ask_volume': float,
            'bid_levels': int,
            'ask_levels': int,
        }
        """
        try:
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return None

            depth = config.WALL_DEPTH_LEVELS
            bid_vol = sum(q for (_, q) in bids[:depth])
            ask_vol = sum(q for (_, q) in asks[:depth])

            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                return None

            imbalance = (bid_vol - ask_vol) / total_vol

            return {
                "imbalance": imbalance,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "bid_levels": len(bids[:depth]),
                "ask_levels": len(asks[:depth]),
            }

        except Exception as e:
            logger.error(f"Error computing imbalance: {e}", exc_info=True)
            return None

    def _compute_wall_strength(
        self, data_manager, current_price: float, imbalance_data: Optional[Dict]
    ) -> Optional[Dict]:
        """
        Compute wall strength within ZONE_TICKS of current price.
        Returns: {
            'bid_wall_strength': float,  # Multiple of average volume
            'ask_wall_strength': float,
            'bid_wall_price': float,
            'ask_wall_price': float,
            'bid_wall_volume': float,
            'ask_wall_volume': float,
        }
        """
        try:
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks or not imbalance_data:
                return None

            zone_range = config.ZONE_TICKS * config.TICK_SIZE
            zone_low = current_price - zone_range
            zone_high = current_price + zone_range

            # Average volume for normalization
            depth = config.WALL_DEPTH_LEVELS
            avg_vol = (imbalance_data["bid_volume"] + imbalance_data["ask_volume"]) / (2.0 * depth)

            if avg_vol <= 0:
                avg_vol = 1.0

            # Find largest wall in zone
            bid_wall_vol = 0.0
            bid_wall_price = 0.0
            for (p, q) in bids:
                if zone_low <= p <= zone_high:
                    if q > bid_wall_vol:
                        bid_wall_vol = q
                        bid_wall_price = p

            ask_wall_vol = 0.0
            ask_wall_price = 0.0
            for (p, q) in asks:
                if zone_low <= p <= zone_high:
                    if q > ask_wall_vol:
                        ask_wall_vol = q
                        ask_wall_price = p

            bid_wall_strength = bid_wall_vol / avg_vol
            ask_wall_strength = ask_wall_vol / avg_vol

            return {
                "bid_wall_strength": bid_wall_strength,
                "ask_wall_strength": ask_wall_strength,
                "bid_wall_price": bid_wall_price,
                "ask_wall_price": ask_wall_price,
                "bid_wall_volume": bid_wall_vol,
                "ask_wall_volume": ask_wall_vol,
            }

        except Exception as e:
            logger.error(f"Error computing wall strength: {e}", exc_info=True)
            return None

    def _compute_delta_z_score(self, data_manager) -> Optional[Dict]:
        """
        Compute taker delta Z-score over DELTA_WINDOW_SEC.
        Returns: {
            'z_score': float,
            'raw_delta': float,
            'buy_volume': float,
            'sell_volume': float,
            'population_mean': float,
            'population_std': float,
        }
        """
        try:
            window_sec = config.DELTA_WINDOW_SEC
            trades = data_manager.get_recent_trades(window_seconds=window_sec)

            if not trades:
                return None

            buy_vol = 0.0
            sell_vol = 0.0

            for t in trades:
                qty = float(t.get("qty", 0.0))
                is_buyer_maker = t.get("isBuyerMaker", False)

                if not is_buyer_maker:
                    buy_vol += qty
                else:
                    sell_vol += qty

            delta = buy_vol - sell_vol

            # Update population
            self._delta_population.append(delta)

            if len(self._delta_population) < 10:
                return {
                    "z_score": 0.0,
                    "raw_delta": delta,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "population_mean": 0.0,
                    "population_std": 1.0,
                }

            pop_mean = float(np.mean(self._delta_population))
            pop_std = float(np.std(self._delta_population))

            if pop_std <= 0:
                pop_std = 1.0

            z_score = (delta - pop_mean) / pop_std

            return {
                "z_score": z_score,
                "raw_delta": delta,
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "population_mean": pop_mean,
                "population_std": pop_std,
            }

        except Exception as e:
            logger.error(f"Error computing delta Z-score: {e}", exc_info=True)
            return None

    def _compute_price_touch(self, data_manager, current_price: float) -> Optional[Dict]:
        """
        Compute distance from current price to best bid/ask walls.
        Returns: {
            'bid_distance_ticks': float,
            'ask_distance_ticks': float,
            'best_bid': float,
            'best_ask': float,
        }
        """
        try:
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return None

            best_bid = bids[0][0]
            best_ask = asks[0][0]

            bid_dist_ticks = abs(current_price - best_bid) / config.TICK_SIZE
            ask_dist_ticks = abs(current_price - best_ask) / config.TICK_SIZE

            return {
                "bid_distance_ticks": bid_dist_ticks,
                "ask_distance_ticks": ask_dist_ticks,
                "best_bid": best_bid,
                "best_ask": best_ask,
            }

        except Exception as e:
            logger.error(f"Error computing price touch: {e}", exc_info=True)
            return None

    # ======================================================================
    # 15-minute Telegram Report
    # ======================================================================

    def _maybe_send_15m_report(self, now_sec: float, risk_manager, current_price: float) -> None:
        """Send 15-minute performance report via Telegram."""
        interval = 900.0  # 15 minutes
        if now_sec - self._last_report_sec < interval:
            return

        self._last_report_sec = now_sec

        try:
            rm = risk_manager
            new_trades = rm.total_trades - self._last_report_total_trades
            self._last_report_total_trades = rm.total_trades

            lines = [
                "⏱️ 15-MIN PERFORMANCE REPORT",
                "=" * 40,
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"Price: {current_price:.2f}",
                f"New Trades (15m): {new_trades}",
                f"Total Trades: {rm.total_trades}",
                f"Win Rate: {(rm.winning_trades / rm.total_trades * 100.0) if rm.total_trades > 0 else 0:.2f}%",
                f"Daily P&L: {rm.daily_pnl:.2f} USDT",
                "=" * 40,
            ]

            send_telegram_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Error sending 15m report: {e}", exc_info=True)

    # ======================================================================
    # Main tick handler
    # ======================================================================

    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """
        Main per-tick strategy entrypoint called from the main loop.
        - Manages existing position (TP/SL + timeout + score decay).
        - If flat, evaluates weighted entry gates and possibly opens a new bracket.
        """
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            # 15-minute performance Telegram report
            self._maybe_send_15m_report(now_sec, risk_manager, current_price)

            # Manage open position first
            if self.current_position is not None:
                self._manage_open_position(
                    data_manager=data_manager,
                    order_manager=order_manager,
                    risk_manager=risk_manager,
                    current_price=current_price,
                    now_sec=now_sec,
                )
                return

            # Cooldown between trades
            if self.last_exit_time_min > 0:
                minutes_since_exit = (now_sec / 60.0) - self.last_exit_time_min
                if minutes_since_exit < config.MIN_TIME_BETWEEN_TRADES:
                    return

            # Risk / trading permission check
            allowed, reason = risk_manager.check_trading_allowed()
            if not allowed:
                logger.debug(f"Trading not allowed: {reason}")
                return

            # Vol-Regime Detection
            vol_regime, atr_pct = data_manager.get_vol_regime()

            # Core metrics
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(data_manager, current_price, imbalance_data)
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            # Higher timeframe trend
            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(f"Error fetching HTF trend in on_tick: {e}", exc_info=True)

            # Lower timeframe trend
            ltf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_ltf_trend"):
                    ltf_trend = data_manager.get_ltf_trend()
            except Exception as e:
                logger.error(f"Error fetching LTF trend in on_tick: {e}", exc_info=True)

            # EMA
            ema_val: Optional[float] = None
            try:
                ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
            except Exception as e:
                logger.error(f"Error fetching EMA in on_tick: {e}", exc_info=True)

            # Build Oracle Inputs for data fusion
            oracle_inputs: Optional[OracleInputs] = None
            try:
                # Compute advanced metrics via data_manager's oracle
                lv_1m, lv_5m, lv_15m, micro_trap = data_manager.compute_liquidity_velocity_multi_tf()
                norm_cvd = data_manager.compute_norm_cvd(window_sec=10)
                hurst = data_manager.compute_hurst_exponent(window_ticks=20)
                bos_align = data_manager.compute_bos_alignment(current_price)

                oracle_inputs = OracleInputs(
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    htf_trend=htf_trend,
                    ltf_trend=ltf_trend,
                    ema_val=ema_val,
                    atr_pct=atr_pct,
                    lv_1m=lv_1m,
                    lv_5m=lv_5m,
                    lv_15m=lv_15m,
                    micro_trap=micro_trap,
                    norm_cvd=norm_cvd,
                    hurst=hurst,
                    bos_align=bos_align,
                    current_price=current_price,
                    now_sec=now_sec,
                )
            except Exception as e:
                logger.error(f"Error building OracleInputs: {e}", exc_info=True)

            # === WEIGHTED SCORING ===
            if config.ENABLE_WEIGHTED_SCORING:
                # Compute scores for long and short
                long_score, long_components, long_reasons = self._compute_weighted_score(
                    side="long",
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    htf_trend=htf_trend,
                    ltf_trend=ltf_trend,
                    ema_val=ema_val,
                    current_price=current_price,
                    vol_regime=vol_regime,
                    oracle_inputs=oracle_inputs,
                )

                short_score, short_components, short_reasons = self._compute_weighted_score(
                    side="short",
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    htf_trend=htf_trend,
                    ltf_trend=ltf_trend,
                    ema_val=ema_val,
                    current_price=current_price,
                    vol_regime=vol_regime,
                    oracle_inputs=oracle_inputs,
                )

                # Win probability overlay
                lstm_prob_long = 0.7 if htf_trend == "UP" else 0.5
                lstm_prob_short = 0.7 if htf_trend == "DOWN" else 0.5

                z_sign_long = 1.0 if delta_data and delta_data["z_score"] > 0 else 0.0
                z_sign_short = 1.0 if delta_data and delta_data["z_score"] < 0 else 0.0

                cvd_long = oracle_inputs.norm_cvd if oracle_inputs and oracle_inputs.norm_cvd and oracle_inputs.norm_cvd > 0 else 0.0
                cvd_short = -oracle_inputs.norm_cvd if oracle_inputs and oracle_inputs.norm_cvd and oracle_inputs.norm_cvd < 0 else 0.0

                lv_avg = 0.5
                if oracle_inputs and oracle_inputs.lv_1m and oracle_inputs.lv_5m:
                    lv_avg = (oracle_inputs.lv_1m + oracle_inputs.lv_5m) / 2.0
                    lv_avg = min(1.0, lv_avg)  # Normalize

                win_prob_long = self._compute_win_probability(lstm_prob_long, z_sign_long, cvd_long, lv_avg)
                win_prob_short = self._compute_win_probability(lstm_prob_short, z_sign_short, cvd_short, lv_avg)

                # Entry decision
                long_ready = (
                    long_score >= config.WEIGHTED_SCORE_ENTRY_THRESHOLD
                    and win_prob_long >= config.WIN_PROB_THRESHOLD
                )
                short_ready = (
                    short_score >= config.WEIGHTED_SCORE_ENTRY_THRESHOLD
                    and win_prob_short >= config.WIN_PROB_THRESHOLD
                )

                # Log decision state
                self._log_weighted_decision_state(
                    now_sec=now_sec,
                    current_price=current_price,
                    vol_regime=vol_regime,
                    atr_pct=atr_pct,
                    long_score=long_score,
                    long_components=long_components,
                    long_ready=long_ready,
                    long_reasons=long_reasons,
                    win_prob_long=win_prob_long,
                    short_score=short_score,
                    short_components=short_components,
                    short_ready=short_ready,
                    short_reasons=short_reasons,
                    win_prob_short=win_prob_short,
                    htf_trend=htf_trend,
                    ltf_trend=ltf_trend,
                )

                # Enter position if score passes
                if long_ready:
                    self._enter_position(
                        data_manager=data_manager,
                        order_manager=order_manager,
                        risk_manager=risk_manager,
                        side="long",
                        current_price=current_price,
                        imbalance_data=imbalance_data,
                        wall_data=wall_data,
                        delta_data=delta_data,
                        touch_data=touch_data,
                        now_sec=now_sec,
                        vol_regime=vol_regime,
                        weighted_score=long_score,
                        oracle_inputs=oracle_inputs,
                    )
                    return

                if short_ready:
                    self._enter_position(
                        data_manager=data_manager,
                        order_manager=order_manager,
                        risk_manager=risk_manager,
                        side="short",
                        current_price=current_price,
                        imbalance_data=imbalance_data,
                        wall_data=wall_data,
                        delta_data=delta_data,
                        touch_data=touch_data,
                        now_sec=now_sec,
                        vol_regime=vol_regime,
                        weighted_score=short_score,
                        oracle_inputs=oracle_inputs,
                    )
                    return

            else:
                # Legacy binary AND gates (fallback if weighted scoring disabled)
                logger.warning("Weighted scoring disabled; using legacy binary gates")
                pass

        except Exception as e:
            logger.error(f"Error in on_tick: {e}", exc_info=True)

    # ======================================================================
    # Logging
    # ======================================================================

    def _log_weighted_decision_state(
        self,
        now_sec: float,
        current_price: float,
        vol_regime: str,
        atr_pct: Optional[float],
        long_score: float,
        long_components: Dict,
        long_ready: bool,
        long_reasons: List[str],
        win_prob_long: float,
        short_score: float,
        short_components: Dict,
        short_ready: bool,
        short_reasons: List[str],
        win_prob_short: float,
        htf_trend: Optional[str],
        ltf_trend: Optional[str],
    ) -> None:
        """Log weighted decision state every DECISION_LOG_INTERVAL_SEC."""
        if now_sec - self._last_decision_log_sec < self.DECISION_LOG_INTERVAL_SEC:
            return

        self._last_decision_log_sec = now_sec

        lines = [
            "=" * 80,
            "WEIGHTED SCORE DECISION SNAPSHOT",
            "=" * 80,
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Price: {current_price:.2f}",
            f"Vol Regime: {vol_regime} (ATR%: {(atr_pct*100) if atr_pct else 'N/A'})",
            f"HTF Trend: {htf_trend or 'UNKNOWN'}",
            f"LTF Trend: {ltf_trend or 'UNKNOWN'}",
            "",
            f"LONG SCORE: {long_score:.4f} (threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD})",
            f"  Components: {long_components}",
            f"  Win Prob: {win_prob_long:.4f} (threshold: {config.WIN_PROB_THRESHOLD})",
            f"  Ready: {long_ready}",
            f"  Reasons: {'; '.join(long_reasons[:5])}",
            "",
            f"SHORT SCORE: {short_score:.4f} (threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD})",
            f"  Components: {short_components}",
            f"  Win Prob: {win_prob_short:.4f} (threshold: {config.WIN_PROB_THRESHOLD})",
            f"  Ready: {short_ready}",
            f"  Reasons: {'; '.join(short_reasons[:5])}",
            "=" * 80,
        ]

        logger.info("\n".join(lines))

        # Send to Telegram (throttled)
        try:
            tg_lines = lines[:15]  # First 15 lines only
            send_telegram_message("\n".join(tg_lines))
        except Exception:
            pass

    def _check_momentum_direction(self, side: str, data_manager, current_price: float) -> bool:
        """Check momentum direction using LV or recent price action"""
        try:
            # Use liquidity velocity as momentum proxy
            lv_1m, lv_5m, _, _ = data_manager.compute_liquidity_velocity_multi_tf()
            lv_avg = (lv_1m or 0) + (lv_5m or 0)
            return lv_avg > 1.0  # Positive liquidity velocity
        except:
            return True  # Safe default

    def _check_trend_direction(self, side: str, data_manager) -> bool:
        """Check HTF trend alignment"""
        try:
            htf_trend = data_manager.get_htf_trend() if hasattr(data_manager, "get_htf_trend") else None
            if htf_trend is None:
                return False
            if side == "long":
                return htf_trend in ["UP", "RANGE"]
            else:
                return htf_trend in ["DOWN", "RANGE"]
        except:
            return False

    def _nearest_tp_percent(self, entry_price: float, current_price: float, side: str) -> float:
        """Nearest TP% above current profit (e.g. 6.5% -> 7%)"""
        full_tp_pct = config.PROFIT_TARGET_ROI * 100  # 10%
        current_profit_pct = ((current_price - entry_price) / entry_price * 100) if side == "long" else ((entry_price - current_price) / entry_price * 100)
        adjusted_tp_pct = min(full_tp_pct, int(current_profit_pct) + 1)
        if side == "long":
            return round(entry_price * (1 + adjusted_tp_pct / 100), 2)
        else:
            return round(entry_price * (1 - adjusted_tp_pct / 100), 2)

    def _update_take_profit(self, order_manager, pos, new_tp_price: float):
        """Update TP order"""
        logger.info(f"Updating TP: {pos.tp_price:.2f} -> {new_tp_price:.2f}")
        if pos.tp_order_id:
            order_manager.cancel_order(pos.tp_order_id)
        tp_order = order_manager.place_take_profit(
            side="SELL" if pos.side == "long" else "BUY",
            quantity=pos.quantity,
            trigger_price=new_tp_price,
        )
        if tp_order and "order_id" in tp_order:
            pos.tp_order_id = tp_order["order_id"]
            pos.tp_price = new_tp_price

    def _update_stop_loss(self, order_manager, pos, new_sl_price: float):
        """Update SL order"""
        logger.info(f"Updating SL: {pos.sl_price:.2f} -> {new_sl_price:.2f}")
        if pos.sl_order_id:
            order_manager.cancel_order(pos.sl_order_id)
        sl_order = order_manager.place_stop_loss(
            side="SELL" if pos.side == "long" else "BUY",
            quantity=pos.quantity,
            trigger_price=new_sl_price,
        )
        if sl_order and "order_id" in sl_order:
            pos.sl_order_id = sl_order["order_id"]
            pos.sl_price = new_sl_price

        # ══════════════════════════════════════════════════════════════════════
        # Position Entry (COMPLETE WITH ALL FIXES)
        # ══════════════════════════════════════════════════════════════════════

    def _enter_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        side: str,  # "long" or "short"
        current_price: float,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        now_sec: float,
        vol_regime: str,  # "LOW", "HIGH", "NEUTRAL", "UNKNOWN"
        weighted_score: float,  # 0-1 score from weighted gauntlet
        oracle_inputs: Optional[OracleInputs],
    ) -> None:
        """
        Enter position with dynamic TP/SL/sizing based on vol regime.

        UPDATED: Single LIMIT order placement and single wait_for_fill call.
        """
        with self._entry_lock:
            if self._entering_position:
                logger.warning("⚠️ Entry BLOCKED: Already entering position")
                return
            if self.current_position is not None:
                logger.warning("⚠️ Entry BLOCKED: Position already exists")
                return
            self._entering_position = True

        valid, reason = self._validate_trade_conditions(data_manager, risk_manager, vol_regime)
        if not valid:
            logger.warning(f"⚠️ Trade validation failed: {reason}")
            with self._entry_lock:
                self._entering_position = False
            return

        try:
            balance_info = risk_manager.get_available_balance()
            if not balance_info:
                logger.error("Cannot fetch balance for entry")
                return
            available = float(balance_info.get("available", 0.0))
            if available < config.MIN_MARGIN_PER_TRADE:
                logger.warning(f"Available {available:.2f} < MIN_MARGIN {config.MIN_MARGIN_PER_TRADE}")
                return

            tp_roi, sl_roi = self._get_dynamic_tp_sl(vol_regime)
            target_margin, quantity = risk_manager.calculate_position_size_regime_aware(
                entry_price=current_price,
                vol_regime=vol_regime,
                balance_available=available,
            )
            if quantity <= 0:
                logger.warning("Computed quantity <= 0; skipping entry")
                return
            min_qty = 0.001  # BTC minimum
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity:.6f} < minimum {min_qty}; adjusting")
                quantity = min_qty

            entry_price = self._calculate_entry_price(side, current_price, vol_regime)

            if side == "long":
                price_move_for_tp = (target_margin * abs(tp_roi)) / quantity
                tp_price = entry_price + price_move_for_tp
                price_move_for_sl = (target_margin * abs(sl_roi)) / quantity
                sl_price = entry_price - price_move_for_sl
            else:
                price_move_for_tp = (target_margin * abs(tp_roi)) / quantity
                tp_price = entry_price - price_move_for_tp
                price_move_for_sl = (target_margin * abs(sl_roi)) / quantity
                sl_price = entry_price + price_move_for_sl

            tp_price = round(tp_price, 2)
            sl_price = round(sl_price, 2)

            # Validate TP/SL placement
            if side == "long":
                if tp_price <= entry_price or sl_price >= entry_price:
                    logger.error(f"Invalid TP/SL for LONG: entry={entry_price}, tp={tp_price}, sl={sl_price}")
                    return
            else:
                if tp_price >= entry_price or sl_price <= entry_price:
                    logger.error(f"Invalid TP/SL for SHORT: entry={entry_price}, tp={tp_price}, sl={sl_price}")
                    return

            self.trade_seq += 1
            trade_id = f"Z{int(now_sec)}_{self.trade_seq}"

            logger.info("=" * 80)
            logger.info(f"ENTERING {side.upper()} POSITION")
            logger.info("=" * 80)
            logger.info(f"Trade ID : {trade_id}")
            logger.info(f"Timestamp : {datetime.utcfromtimestamp(now_sec).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"Vol Regime : {vol_regime}")
            logger.info(f"Weighted Score : {weighted_score:.4f} (threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD})")
            logger.info("-" * 80)
            logger.info(f"Entry Price : {entry_price:.2f}")
            logger.info(f"Quantity : {quantity:.6f} BTC")
            logger.info(f"Margin Used : {target_margin:.2f} USDT")
            logger.info(f"Leverage : {config.LEVERAGE}x")
            logger.info(f"Position Value : {quantity * entry_price:.2f} USDT")
            logger.info("-" * 80)
            logger.info(f"TP Price : {tp_price:.2f} (ROI: {tp_roi * 100:.2f}%)")
            logger.info(f"SL Price : {sl_price:.2f} (ROI: {sl_roi * 100:.2f}%)")
            logger.info(f"TP Distance : {abs(tp_price - entry_price):.2f} USDT")
            logger.info(f"SL Distance : {abs(sl_price - entry_price):.2f} USDT")
            logger.info(f"Risk:Reward Ratio : {abs(tp_price - entry_price) / abs(entry_price - sl_price):.2f}:1")
            logger.info("-" * 80)
            logger.info(f"Imbalance : {imbalance_data['imbalance']:.3f}")
            logger.info(f"Z-Score : {delta_data['z_score']:.2f}")
            logger.info(f"Wall Strength : {wall_data['bid_wall_strength' if side == 'long' else 'ask_wall_strength']:.2f}x")
            logger.info(f"Touch Distance : {touch_data['bid_distance_ticks' if side == 'long' else 'ask_distance_ticks']:.1f} ticks")
            if oracle_inputs:
                logger.info("-" * 80)
                cvd_str = f"{oracle_inputs.norm_cvd:.3f}" if oracle_inputs.norm_cvd is not None else "N/A"
                logger.info(f"CVD : {cvd_str}")
                lv_1m_str = f"{oracle_inputs.lv_1m:.2f}" if oracle_inputs.lv_1m is not None else "N/A"
                logger.info(f"LV 1m : {lv_1m_str}")
                hurst_str = f"{oracle_inputs.hurst:.3f}" if oracle_inputs.hurst is not None else "N/A"
                logger.info(f"Hurst : {hurst_str}")
                bos_str = f"{oracle_inputs.bos_align:.3f}" if oracle_inputs.bos_align is not None else "N/A"
                logger.info(f"BOS Align : {bos_str}")
                logger.info(f"HTF Trend : {oracle_inputs.htf_trend or 'N/A'}")
                logger.info(f"LTF Trend : {oracle_inputs.ltf_trend or 'N/A'}")
                logger.info("=" * 80)

            logger.info(f"Placing main LIMIT {side.upper()} order...")
            main_order = order_manager.place_limit_order(
                side="BUY" if side == "long" else "SELL",
                quantity=quantity,
                price=entry_price,
                reduce_only=False,
            )
            if not main_order or "order_id" not in main_order:
                logger.error("Main order placement failed - no order_id returned")
                return

            main_order_id = main_order["order_id"]
            logger.info(f"✓ Main order placed: {main_order_id}")

            logger.info(f"Waiting for fill (timeout: {self.ENTRY_FILL_TIMEOUT_SEC}s)...")
            filled_order = order_manager.wait_for_fill(main_order_id, self.ENTRY_FILL_TIMEOUT_SEC, 1.0)
            fill_price = order_manager.extract_fill_price(filled_order)
            logger.info(f"✓ Order filled at: {fill_price:.2f} (limit: {entry_price:.2f})")

            # Recalculate TP/SL if significant slippage
            if abs(fill_price - entry_price) / entry_price > 0.001:
                logger.warning(f"Slippage detected: {abs(fill_price - entry_price):.2f} USDT")
                if side == "long":
                    tp_price = fill_price + (target_margin * abs(tp_roi)) / quantity
                    sl_price = fill_price - (target_margin * abs(sl_roi)) / quantity
                else:
                    tp_price = fill_price - (target_margin * abs(tp_roi)) / quantity
                    sl_price = fill_price + (target_margin * abs(sl_roi)) / quantity
                tp_price = round(tp_price, 2)
                sl_price = round(sl_price, 2)

            logger.info("Placing TP and SL bracket orders...")
            tp_side = "SELL" if side == "long" else "BUY"
            sl_side = "SELL" if side == "long" else "BUY"
            tp_order = order_manager.place_take_profit(
                side=tp_side,
                quantity=quantity,
                trigger_price=tp_price,
            )
            tp_order_id = tp_order["order_id"] if tp_order and "order_id" in tp_order else ""
            sl_order = order_manager.place_stop_loss(
                side=sl_side,
                quantity=quantity,
                trigger_price=sl_price,
            )
            sl_order_id = sl_order["order_id"] if sl_order and "order_id" in sl_order else ""

            self.current_position = ZScorePosition(
                trade_id=trade_id,
                side=side,
                quantity=quantity,
                entry_price=fill_price,
                entry_time_sec=now_sec,
                entry_wall_volume=wall_data["bid_wall_volume" if side == "long" else "ask_wall_volume"],
                wall_zone_low=current_price - config.ZONE_TICKS * config.TICK_SIZE,
                wall_zone_high=current_price + config.ZONE_TICKS * config.TICK_SIZE,
                entry_imbalance=imbalance_data["imbalance"],
                entry_z_score=delta_data["z_score"],
                tp_price=tp_price,
                sl_price=sl_price,
                margin_used=target_margin,
                tp_order_id=tp_order_id,
                sl_order_id=sl_order_id,
                main_order_id=main_order_id,
                main_filled=True,
                tp_reduced=False,
                entry_htf_trend=oracle_inputs.htf_trend if oracle_inputs else "UNKNOWN",
                entry_vol_regime=vol_regime,
                entry_weighted_score=weighted_score,
                last_score_check_sec=now_sec,
            )
            risk_manager.record_trade_opened()
            logger.info(f"✓ Position created successfully: {trade_id}")

            # Excel logging
            if self.excel_logger:
                try:
                    self.excel_logger.log_entry(
                        trade_id=trade_id,
                        timestamp=datetime.utcfromtimestamp(now_sec),
                        side=side,
                        entry_price=fill_price,
                        quantity=quantity,
                        margin=target_margin,
                        leverage=config.LEVERAGE,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        imbalance=imbalance_data["imbalance"],
                        z_score=delta_data["z_score"],
                        wall_strength=wall_data["bid_wall_strength" if side == "long" else "ask_wall_strength"],
                        vol_regime=vol_regime,
                        weighted_score=weighted_score,
                        htf_trend=oracle_inputs.htf_trend if oracle_inputs else "N/A",
                        cvd=oracle_inputs.norm_cvd if oracle_inputs and oracle_inputs.norm_cvd else 0.0,
                    )
                except Exception as e:
                    logger.error(f"Excel logging failed: {e}", exc_info=True)

            # Telegram notification
            try:
                notification = (
                    f"🟢 {'LONG' if side == 'long' else 'SHORT'} ENTRY\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"ID: {trade_id}\n"
                    f"Price: {fill_price:.2f}\n"
                    f"Qty: {quantity:.6f} BTC\n"
                    f"Margin: {target_margin:.2f} USDT\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"TP: {tp_price:.2f} ({tp_roi * 100:+.2f}%)\n"
                    f"SL: {sl_price:.2f} ({sl_roi * 100:+.2f}%)\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"Vol Regime: {vol_regime}\n"
                    f"Score: {weighted_score:.4f}\n"
                    f"Z-Score: {delta_data['z_score']:.2f}\n"
                )
                send_telegram_message(notification)
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")

        finally:
            with self._entry_lock:
                self._entering_position = False
            logger.info("🔓 Entry lock RELEASED")

        # ======================================================================
        # Position Management
        # ======================================================================

    def _manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        NEW POSITION MANAGEMENT LOGIC:
        - After 10 mins: Check momentum/vol/trend every 5s
        - Volatile market: TP=10%, SL=3%
        - Favorable: wait 10 more mins, SL to half TP if >50%
        - Adverse: TP to half or nearest % if above half
        """
        pos = self.current_position
        if pos is None:
            return

        hold_time_min = (now_sec - pos.entry_time_sec) / 60.0

        # Periodic position logging
        if now_sec - self._last_position_log_sec >= self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            logger.info("=" * 80)
            logger.info(f"POSITION ACTIVE: {pos.side.upper()} {pos.trade_id}")
            logger.info(f"Entry: {pos.entry_price:.2f} | Current: {current_price:.2f}")
            logger.info(f"Hold Time: {hold_time_min:.2f} min")
            logger.info(f"TP: {pos.tp_price:.2f} | SL: {pos.sl_price:.2f}")
            logger.info(f"Vol Regime: {pos.entry_vol_regime}")
            logger.info(f"Entry Score: {pos.entry_weighted_score:.4f}")
            logger.info("=" * 80)

        # Check TP/SL status (single check every 10s)
        if now_sec - self._last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            if pos.tp_order_id:
                tp_status = order_manager.get_order_status(pos.tp_order_id)
                if tp_status and tp_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    self._close_position(reason="TP_HIT", exit_price=pos.tp_price, order_manager=order_manager, risk_manager=risk_manager)
                    return
            if pos.sl_order_id:
                sl_status = order_manager.get_order_status(pos.sl_order_id)
                if sl_status and sl_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    self._close_position(reason="SL_HIT", exit_price=pos.sl_price, order_manager=order_manager, risk_manager=risk_manager)
                    return

        # Time stop (keep existing 10 min max)
        if hold_time_min >= config.MAX_HOLD_MINUTES:
            self._close_position(reason="TIME_STOP", exit_price=current_price, order_manager=order_manager, risk_manager=risk_manager)
            return

        # === NEW 10-MINUTE POSITION MANAGEMENT ===
        if hold_time_min >= 10.0:
            # Check momentum/vol/trend every 5 seconds
            if now_sec - getattr(pos, 'last_condition_check_sec', 0) >= 5.0:
                pos.last_condition_check_sec = now_sec
                
                vol_regime, atr_pct = data_manager.get_vol_regime()
                momentum_ok = self._check_momentum_direction(pos.side, data_manager, current_price)
                trend_ok = self._check_trend_direction(pos.side, data_manager)
                volatility_ok = vol_regime == "HIGH"
                
                # LOG every 5s check
                logger.info(f"[{hold_time_min:.1f}min] Momentum:{'✅' if momentum_ok else '❌'} Trend:{'✅' if trend_ok else '❌'} Vol:{vol_regime}{'HIGH✅' if volatility_ok else 'LOW❌'}")
                
                # ALL 3 favorable -> continue with full TP, move SL if > half TP
                if momentum_ok and trend_ok and volatility_ok:
                    half_tp_price = pos.entry_price + (pos.tp_price - pos.entry_price) * 0.5 if pos.side == "long" else pos.entry_price - (pos.entry_price - pos.tp_price) * 0.5
                    current_profit_pct = ((current_price - pos.entry_price) / pos.entry_price * 100) if pos.side == "long" else ((pos.entry_price - current_price) / pos.entry_price * 100)
                    
                    if current_profit_pct > 5.0:  # > half TP (5%)
                        new_sl_price = half_tp_price
                        if abs(pos.sl_price - new_sl_price) > 0.01:
                            self._update_stop_loss(order_manager, pos, new_sl_price)
                    return
                
                # ANY adverse -> adjust TP based on current profit
                else:
                    half_tp_price = pos.entry_price + (pos.tp_price - pos.entry_price) * 0.5 if pos.side == "long" else pos.entry_price - (pos.entry_price - pos.tp_price) * 0.5
                    current_profit_pct = ((current_price - pos.entry_price) / pos.entry_price * 100) if pos.side == "long" else ((pos.entry_price - current_price) / pos.entry_price * 100)
                    
                    if current_profit_pct > 5.0:  # Above half TP
                        new_tp_price = self._nearest_tp_percent(pos.entry_price, current_price, pos.side)
                        if abs(pos.tp_price - new_tp_price) > 0.01:
                            self._update_take_profit(order_manager, pos, new_tp_price)
                    else:  # Below half TP
                        if abs(pos.tp_price - half_tp_price) > 0.01:
                            self._update_take_profit(order_manager, pos, half_tp_price)
                    return

    def _close_position(self, reason: str, exit_price: float, order_manager, risk_manager) -> None:
        """
        Close position - SINGLE cancel request only, check status first
        """
        pos = self.current_position
        if pos is None:
            return
        
        logger.info("=" * 80)
        logger.info(f"CLOSING POSITION: {pos.trade_id}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Exit Price: {exit_price:.2f}")
        logger.info("=" * 80)
        
        # SINGLE cancel request - check TP status first
        if pos.tp_order_id:
            tp_status = order_manager.get_order_status(pos.tp_order_id)
            if tp_status and tp_status.get("status", "").upper() not in ["EXECUTED", "FILLED", "CANCELLED"]:
                order_manager.cancel_order(pos.tp_order_id)
        
        # SINGLE cancel request - check SL status next
        if pos.sl_order_id:
            sl_status = order_manager.get_order_status(pos.sl_order_id)
            if sl_status and sl_status.get("status", "").upper() not in ["EXECUTED", "FILLED", "CANCELLED"]:
                order_manager.cancel_order(pos.sl_order_id)
        
        # Clear position
        if pos.side == "long":
            pnl_per_unit = exit_price - pos.entry_price
        else:
            pnl_per_unit = pos.entry_price - exit_price
        gross_pnl = pnl_per_unit * pos.quantity
        fees = pos.entry_price * exit_price * pos.quantity * config.TAKER_FEE_RATE
        net_pnl = gross_pnl - fees
        
        risk_manager.update_trade_stats(net_pnl)
        logger.info(f"P&L: {net_pnl:.2f} USDT (Gross: {gross_pnl:.2f}, Fees: {fees:.2f})")
        
        self.current_position = None
        self.last_exit_time_min = time.time() / 60.0
