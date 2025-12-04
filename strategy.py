"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version - PRODUCTION READY
✅ FIXED: TP/SL Calculation matches Excel Sheet (Margin-based ROI / Leverage)
✅ FIXED: Clean entry/exit flow with proper position management
✅ NO SHORTCUTS: Full logic included
"""

import time
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

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger

        self.trade_seq = 0
        self.total_trades = 0

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
        logger.info(f"Imbalance Threshold   = {config.IMBALANCE_THRESHOLD:.2f}")
        logger.info(f"Wall Volume Mult BASE = {config.MIN_WALL_VOLUME_MULT:.2f}x")
        logger.info(f"Delta Z Threshold BASE= {config.DELTA_Z_THRESHOLD:.2f}")
        logger.info(f"Zone Ticks            = ±{config.ZONE_TICKS}")
        logger.info(f"Touch Threshold       = {config.PRICE_TOUCH_THRESHOLD_TICKS} ticks")
        logger.info(f"Profit Target ROI BASE= {config.PROFIT_TARGET_ROI * 100:.2f}%")
        logger.info(f"Stop Loss ROI BASE    = {config.STOP_LOSS_ROI * 100:.2f}%")
        logger.info(f"Max Hold Minutes      = {config.MAX_HOLD_MINUTES}")
        logger.info(f"Weighted Scoring      = {config.ENABLE_WEIGHTED_SCORING}")
        logger.info(f"Score Entry Threshold = {config.WEIGHTED_SCORE_ENTRY_THRESHOLD}")
        logger.info("=" * 80)

    # ══════════════════════════════════════════════════════════════════════
    # Vol-Regime Helpers
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
    # Weighted Scoring Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _normalize_signal_cdf(self, value: float, threshold: float, std_dev: float = 1.0) -> float:
        """
        Normalize signal using CDF approach: norm.cdf((value - threshold) / std).
        Returns value between 0 and 1.
        """
        try:
            if std_dev <= 0:
                std_dev = 1.0
            z = (value - threshold) / std_dev
            z = max(-5.0, min(5.0, z))
            normalized = scipy_stats.norm.cdf(z)
            normalized = max(0.01, min(0.99, normalized))
            return normalized
        except Exception as e:
            logger.error(f"Error in CDF normalization: {e}", exc_info=True)
            return 0.5

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
        """
        components = {}
        reasons = []
        raw_components = {}

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

        # === WEIGHTED AGGREGATION ===
        # Base weights from config
        weights = {
            "imbalance": config.WEIGHT_IMBALANCE,
            "wall": config.WEIGHT_WALL,
            "zscore":    config.WEIGHT_ZSCORE,
            "touch":     config.WEIGHT_TOUCH,
            "trend":     config.WEIGHT_TREND,
            "cvd":       config.WEIGHT_CVD,
            "lv":        config.WEIGHT_LV,
            "hurst_bos": config.WEIGHT_HURST_BOS,
        }

        # Only use available components
        available_components = {k: v for k, v in raw_components.items() if v > 0}
        
        if not available_components:
            return 0.0, {}, ["NO_SIGNALS_AVAILABLE"]

        # Normalize weights to sum to 1.0 for available components only
        total_weight = sum(weights[k] for k in available_components)
        
        if total_weight <= 0:
            return 0.0, {}, ["ZERO_TOTAL_WEIGHT"]
        
        # Compute weighted score
        weighted_score = 0.0
        for key in available_components:
            normalized_weight = weights[key] / total_weight
            weighted_score += available_components[key] * normalized_weight
            components[key] = available_components[key]

        # Clamp to [0, 1]
        weighted_score = max(0.0, min(1.0, weighted_score))
        
        return weighted_score, components, reasons

    def _compute_win_probability(
        self, lstm_prob: float, z_sign: float, cvd: float, lv_avg: float
    ) -> float:
        """
        Compute win probability overlay from data fusion signals.
        Returns: win_prob (0-1)
        """
        try:
            # Simple weighted average
            prob = (0.4 * lstm_prob) + (0.3 * z_sign) + (0.2 * cvd) + (0.1 * lv_avg)
            return max(0.0, min(1.0, prob))
        except Exception as e:
            logger.error(f"Error computing win probability: {e}")
            return 0.5

    # ══════════════════════════════════════════════════════════════════════
    # Core Strategy Logic
    # ══════════════════════════════════════════════════════════════════════

    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """
        Main Strategy Loop - Called on every WebSocket update.
        1. Updates Oracle & Z-Scores
        2. Manages Existing Position
        3. Checks Entry Conditions (if flat)
        """
        try:
            # 1. Basic Data & Time Checks
            now_sec = time.time()
            current_price = data_manager.get_last_price()
            
            if current_price <= 0:
                return

            # 2. Manage Open Position
            if self.current_position:
                self._manage_open_position(
                    data_manager, order_manager, risk_manager, current_price, now_sec
                )
                return  # Exit early if position exists

            # 3. Validate basic trading conditions
            if not config.TRADING_ENABLED:
                return
                
            # Cooldown check
            if now_sec - self.last_exit_time_min * 60 < config.COOLDOWN_MINUTES * 60:
                return

            # 4. Calculate Indicators
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return

            vol_regime, atr_pct = data_manager.get_vol_regime()
            htf_trend = data_manager.get_htf_trend()
            
            # 5. Calculate Oracle Signals (Iceberg Hunter Logic)
            # Oracle Inputs
            oracle_inputs = AetherOracle.get_oracle_inputs(data_manager)
            
            # Z-Score Delta
            delta_data = AetherOracle.compute_z_score_delta(bids, asks)
            if not delta_data:
                return
                
            # Add delta to population for Z-score history
            self._delta_population.append(delta_data["raw_delta"])
            
            # Orderbook Imbalance
            imbalance_data = AetherOracle.compute_orderbook_imbalance(bids, asks)
            
            # Wall Detection
            wall_mult_dynamic = self._get_dynamic_wall_mult(vol_regime)
            wall_data = AetherOracle.detect_walls(
                bids, asks, current_price, 
                min_wall_mult=wall_mult_dynamic
            )
            
            # Price Touches
            touch_data = AetherOracle.check_price_touches(
                bids, asks, current_price, 
                threshold_ticks=config.PRICE_TOUCH_THRESHOLD_TICKS
            )

            # 6. Scoring & Entry Logic
            if config.ENABLE_WEIGHTED_SCORING:
                # Compute Scores
                ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
                ltf_trend = data_manager.get_ltf_trend() if hasattr(data_manager, "get_ltf_trend") else None
                
                long_score, long_comps, long_reasons = self._compute_weighted_score(
                    "long", imbalance_data, wall_data, delta_data, touch_data, 
                    htf_trend, ltf_trend, ema_val, current_price, vol_regime, oracle_inputs
                )
                
                short_score, short_comps, short_reasons = self._compute_weighted_score(
                    "short", imbalance_data, wall_data, delta_data, touch_data, 
                    htf_trend, ltf_trend, ema_val, current_price, vol_regime, oracle_inputs
                )

                # Decision Logging
                if now_sec - self._last_decision_log_sec > self.DECISION_LOG_INTERVAL_SEC:
                    self._last_decision_log_sec = now_sec
                    logger.info("=" * 80)
                    logger.info(f"DECISION SNAPSHOT @ ${current_price:.2f}")
                    logger.info(f"Vol: {vol_regime} | HTF: {htf_trend}")
                    logger.info(f"LONG Score: {long_score:.3f} | SHORT Score: {short_score:.3f}")
                    logger.info(f"Threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD}")
                    logger.info("=" * 80)

                # Entry Execution
                threshold = config.WEIGHTED_SCORE_ENTRY_THRESHOLD
                
                if long_score >= threshold:
                    logger.info(f"🚀 LONG SIGNAL (Score: {long_score:.3f})")
                    logger.info(f"Reasons: {long_reasons}")
                    self._enter_position(
                        "long", current_price, data_manager, order_manager, risk_manager,
                        imbalance_data, wall_data, delta_data, 
                        weighted_score=long_score, vol_regime=vol_regime
                    )
                elif short_score >= threshold:
                    logger.info(f"🚀 SHORT SIGNAL (Score: {short_score:.3f})")
                    logger.info(f"Reasons: {short_reasons}")
                    self._enter_position(
                        "short", current_price, data_manager, order_manager, risk_manager,
                        imbalance_data, wall_data, delta_data, 
                        weighted_score=short_score, vol_regime=vol_regime
                    )

        except Exception as e:
            logger.error(f"Error in strategy on_tick: {e}", exc_info=True)

    # ══════════════════════════════════════════════════════════════════════
    # Position Entry - EXCEL-BASED TP/SL METHOD
    # ══════════════════════════════════════════════════════════════════════

    def _enter_position(
        self,
        side: str,
        current_price: float,
        data_manager,
        order_manager,
        risk_manager,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        weighted_score: float,
        vol_regime: str
    ) -> None:
        """
        Execute Entry with Excel-based TP/SL Calculation.
        Formula: Move = Price * (ROI_on_Margin / Leverage)
        """
        try:
            now_sec = time.time()

            # 1. Validate conditions
            is_valid, reason = self._validate_trade_conditions(data_manager, risk_manager, vol_regime)
            if not is_valid:
                logger.warning(f"Trade rejected: {reason}")
                return

            # 2. Calculate Entry Price (Limit Order Logic)
            entry_price = self._calculate_entry_price(side, current_price, vol_regime)

            # 3. Calculate Position Size
            size_pct = self._get_dynamic_position_size_pct(vol_regime)
            quantity = risk_manager.calculate_position_size(entry_price, size_pct)
            
            if quantity <= 0:
                logger.warning("Calculated quantity is 0")
                return

            # 4. Calculate TP/SL using EXCEL METHOD (Margin-based)
            # Get ROI targets (e.g., 0.05 for 5%, -0.01 for -1%)
            tp_roi_margin, sl_roi_margin = self._get_dynamic_tp_sl(vol_regime)
            
            # Apply Excel Formula: Move = Entry * (ROI / Leverage)
            # This assumes ROI is on MARGIN, not on Notional
            leverage = config.LEVERAGE
            
            move_tp = entry_price * (tp_roi_margin / leverage)
            move_sl = entry_price * (abs(sl_roi_margin) / leverage)
            
            if side == "long":
                tp_price = entry_price + move_tp
                sl_price = entry_price - move_sl
            else:
                tp_price = entry_price - move_tp
                sl_price = entry_price + move_sl
                
            # Round to 2 decimals
            tp_price = round(tp_price, 2)
            sl_price = round(sl_price, 2)

            logger.info("-" * 50)
            logger.info(f"📐 TP/SL CALCULATION (Excel Method)")
            logger.info(f"Entry: {entry_price}")
            logger.info(f"Lev: {leverage}x")
            logger.info(f"TP ROI (Margin): {tp_roi_margin*100:.1f}% -> Move: ${move_tp:.2f}")
            logger.info(f"SL ROI (Margin): {sl_roi_margin*100:.1f}% -> Move: ${move_sl:.2f}")
            logger.info(f"Targets -> TP: {tp_price} | SL: {sl_price}")
            logger.info("-" * 50)

            # 5. Execute Bracket Order
            result = order_manager.place_bracket_order(
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price
            )

            if not result:
                logger.error("Bracket order execution failed")
                return

            filled_order, tp_order, sl_order = result
            
            # 6. Record Position
            self.trade_seq += 1
            self.total_trades += 1
            trade_id = f"T-{int(now_sec)}-{self.trade_seq}"
            
            margin_used = (quantity * entry_price) / leverage
            
            # Extract actual fill price
            actual_fill_price = order_manager.extract_fill_price(filled_order)
            
            htf_trend = data_manager.get_htf_trend() if hasattr(data_manager, "get_htf_trend") else "NA"
            
            self.current_position = ZScorePosition(
                trade_id=trade_id,
                side=side,
                quantity=quantity,
                entry_price=actual_fill_price,
                entry_time_sec=now_sec,
                entry_wall_volume=wall_data["bid_wall_vol"] if side == "long" else wall_data["ask_wall_vol"],
                wall_zone_low=wall_data["zone_low"],
                wall_zone_high=wall_data["zone_high"],
                entry_imbalance=imbalance_data["imbalance"],
                entry_z_score=delta_data["z_score"],
                tp_price=tp_price,
                sl_price=sl_price,
                margin_used=margin_used,
                tp_order_id=tp_order["order_id"],
                sl_order_id=sl_order["order_id"],
                main_order_id=filled_order["order_id"],
                entry_htf_trend=htf_trend,
                entry_vol_regime=vol_regime,
                entry_weighted_score=weighted_score,
                last_score_check_sec=now_sec,
            )

            # Update risk manager
            risk_manager.record_trade_opened()

            # Log to Excel
            if self.excel_logger:
                try:
                    self.excel_logger.log_entry(
                        trade_id=trade_id,
                        side=side,
                        entry_price=actual_fill_price,
                        quantity=quantity,
                        margin=margin_used,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        imbalance=imbalance_data["imbalance"],
                        wall_strength=wall_data["bid_wall_strength"] if side == "long" else wall_data["ask_wall_strength"],
                        delta_z=delta_data["z_score"],
                        htf_trend=htf_trend,
                        vol_regime=vol_regime,
                        weighted_score=weighted_score,
                    )
                except Exception as e:
                    logger.error(f"Excel logging error: {e}")

            # Telegram notification
            try:
                send_telegram_message(
                    f"📈 {side.upper()} ENTRY\n"
                    f"Trade: {trade_id}\n"
                    f"Price: ${actual_fill_price:.2f}\n"
                    f"Qty: {quantity:.6f}\n"
                    f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f}\n"
                    f"Score: {weighted_score:.3f} | Vol: {vol_regime}"
                )
            except Exception:
                pass

            logger.info("=" * 80)
            logger.info(f"✅ POSITION OPENED: {trade_id}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Error entering position: {e}", exc_info=True)

    def _calculate_entry_price(self, side: str, current_price: float, vol_regime: str) -> float:
        """
        Calculate entry price with volatility-based tick offset.
        """
        if vol_regime == "HIGH":
            tick_offset = 10 
        elif vol_regime == "LOW":
            tick_offset = 4
        else:
            tick_offset = 6
        
        if side == "long":
            entry_price = current_price - (tick_offset * config.TICK_SIZE)
        else:
            entry_price = current_price + (tick_offset * config.TICK_SIZE)
            
        entry_price = round(entry_price, 2)
        return entry_price

    def _validate_trade_conditions(
        self,
        data_manager,
        risk_manager,
        vol_regime: str
    ) -> Tuple[bool, str]:
        """
        Validate trade conditions before entry.
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
            pass

        # Check 4: Balance available
        try:
            balance = risk_manager.get_available_balance()
            if not balance:
                return False, "Cannot fetch balance"
            
            available = float(balance.get('available', 0))
            if available < config.MIN_MARGIN_PER_TRADE:
                return False, f"Insufficient balance: {available:.2f}"
        except Exception as e:
            return False, f"Cannot check balance: {e}"

        return True, "OK"

    # ══════════════════════════════════════════════════════════════════════
    # Position Management
    # ══════════════════════════════════════════════════════════════════════

    def _manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        Manage open position:
        1. Check if TP/SL triggered
        2. Check time stop
        3. Log position state
        """
        pos = self.current_position
        if not pos:
            return

        # 1. Check TP/SL order status
        if now_sec - self._last_status_check_sec > self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            try:
                tp_status = order_manager.get_order_status(pos.tp_order_id)
                sl_status = order_manager.get_order_status(pos.sl_order_id)
                
                if tp_status and tp_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    logger.info("✅ TP triggered")
                    self._close_position(
                        order_manager, risk_manager, "TP_HIT", now_sec, current_price
                    )
                    return

                if sl_status and sl_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    logger.info("❌ SL triggered")
                    self._close_position(
                        order_manager, risk_manager, "SL_HIT", now_sec, current_price
                    )
                    return

            except Exception as e:
                logger.error(f"Error checking TP/SL status: {e}")

        # 2. Time stop
        hold_minutes = (now_sec - pos.entry_time_sec) / 60.0
        if hold_minutes > config.MAX_HOLD_MINUTES:
            logger.info(f"⏰ Time stop triggered ({hold_minutes:.1f}min)")
            self._close_position(
                order_manager, risk_manager, "TIME_STOP", now_sec, current_price
            )
            return

        # 3. Periodic position logging
        if now_sec - self._last_position_log_sec > self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            pnl = self._calculate_unrealized_pnl(pos, current_price)
            roi = (pnl / pos.margin_used) * 100.0 if pos.margin_used > 0 else 0.0
            
            logger.info("=" * 80)
            logger.info(f"POSITION UPDATE: {pos.trade_id}")
            logger.info(f"Side: {pos.side.upper()} | Qty: {pos.quantity:.6f}")
            logger.info(f"Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f}")
            logger.info(f"TP: ${pos.tp_price:.2f} | SL: ${pos.sl_price:.2f}")
            logger.info(f"Unrealized P&L: ${pnl:.2f} ({roi:.2f}%)")
            logger.info(f"Hold Time: {hold_minutes:.1f}min / {config.MAX_HOLD_MINUTES}min")
            logger.info("=" * 80)

    def _close_position(
        self,
        order_manager,
        risk_manager,
        exit_reason: str,
        now_sec: float,
        current_price: float,
    ) -> None:
        """
        Close position and cleanup.
        """
        pos = self.current_position
        if not pos:
            return

        try:
            logger.info("=" * 80)
            logger.info(f"CLOSING POSITION: {pos.trade_id} ({exit_reason})")
            logger.info("=" * 80)

            # 1. Cancel remaining orders
            order_manager.cancel_order(pos.tp_order_id)
            order_manager.cancel_order(pos.sl_order_id)
            
            # 2. Force Close (if time stop)
            if exit_reason == "TIME_STOP":
                exit_side = "SELL" if pos.side == "long" else "BUY"
                order_manager.place_market_order(side=exit_side, quantity=pos.quantity, reduce_only=True)

            # 3. Calculate P&L
            realized_pnl = self._calculate_unrealized_pnl(pos, current_price)
            roi = (realized_pnl / pos.margin_used) * 100.0 if pos.margin_used > 0 else 0.0

            # 4. Log Exit
            if self.excel_logger:
                try:
                    self.excel_logger.log_exit(
                        trade_id=pos.trade_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl=realized_pnl,
                        roi=roi
                    )
                except Exception:
                    pass

            try:
                send_telegram_message(
                    f"🏁 POSITION CLOSED\n"
                    f"ID: {pos.trade_id}\n"
                    f"Reason: {exit_reason}\n"
                    f"PnL: ${realized_pnl:.2f} ({roi:.2f}%)"
                )
            except Exception:
                pass

            # 5. Cleanup
            risk_manager.record_trade_closed(realized_pnl)
            self.current_position = None
            self.last_exit_time_min = now_sec / 60.0

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)

    def _calculate_unrealized_pnl(self, pos: ZScorePosition, current_price: float) -> float:
        if pos.side == "long":
            return (current_price - pos.entry_price) * pos.quantity
        else:
            return (pos.entry_price - current_price) * pos.quantity
