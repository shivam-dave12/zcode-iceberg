"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version - PRODUCTION READY
✅ FIXED: Removed entry lock (single bracket order system)
✅ FIXED: TP/SL calculation matches Excel methodology exactly
✅ FIXED: Score decay exit with trend/volatility/momentum checks
✅ FIXED: Trailing stop-loss to halfway when profit > 50% of TP
✅ STREAMLINED: Clean entry/exit flow
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
    trailing_sl_active: bool = False  # ✅ NEW: Track if trailing SL activated

class ZScoreIcebergHunterStrategy:
    """Z-Score Imbalance Iceberg Hunter with Vol-Regime dynamics and weighted scoring."""
    
    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 120.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0
        self.total_trades = 0
        
        # ✅ REMOVED: _entry_lock and _entering_position (single bracket system)
        
        self._delta_population: deque = deque(maxlen=3000)
        self._last_decision_log_sec: float = 0.0
        self._last_position_log_sec: float = 0.0
        self._last_status_check_sec: float = 0.0
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
        ✅ EXACT EXCEL METHODOLOGY: Get dynamic TP and SL ROI
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

        # Dynamic thresholds
        z_thresh_dynamic = self._get_dynamic_z_threshold(vol_regime, oracle_inputs.atr_pct if oracle_inputs else None)
        wall_mult_dynamic = self._get_dynamic_wall_mult(vol_regime)

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

        # Weighted aggregation
        weights = {
            "imbalance": config.WEIGHT_IMBALANCE,
            "wall": config.WEIGHT_WALL,
            "zscore": config.WEIGHT_ZSCORE,
            "touch": config.WEIGHT_TOUCH,
            "trend": config.WEIGHT_TREND,
            "cvd": config.WEIGHT_CVD,
            "lv": config.WEIGHT_LV,
            "hurst_bos": config.WEIGHT_HURST_BOS,
        }

        available_components = {k: v for k, v in raw_components.items() if v > 0}
        if not available_components:
            return 0.0, {}, ["NO_SIGNALS_AVAILABLE"]

        total_weight = sum(weights[k] for k in available_components)
        if total_weight <= 0:
            return 0.0, {}, ["ZERO_TOTAL_WEIGHT"]

        weighted_score = 0.0
        for key in available_components:
            normalized_weight = weights[key] / total_weight
            weighted_score += available_components[key] * normalized_weight
            components[key] = available_components[key]

        weighted_score = max(0.0, min(1.0, weighted_score))
        return weighted_score, components, reasons

    def _compute_win_probability(
        self, lstm_prob: float, z_sign: float, cvd: float, lv_avg: float
    ) -> float:
        """Compute win probability overlay from data fusion signals."""
        try:
            prob = (0.4 * lstm_prob) + (0.3 * z_sign) + (0.2 * cvd) + (0.1 * lv_avg)
            return max(0.0, min(1.0, prob))
        except Exception as e:
            logger.error(f"Error computing win probability: {e}")
            return 0.5

    # ══════════════════════════════════════════════════════════════════════
    # Core Strategy Metrics
    # ══════════════════════════════════════════════════════════════════════
    
    def _compute_imbalance(self, data_manager) -> Optional[Dict]:
        """Compute orderbook imbalance."""
        try:
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return None

            depth_levels = min(config.WALL_DEPTH_LEVELS, len(bids), len(asks))
            if depth_levels < 5:
                return None

            bid_vol = sum(q for _, q in bids[:depth_levels])
            ask_vol = sum(q for _, q in asks[:depth_levels])
            total = bid_vol + ask_vol
            if total <= 0:
                return None

            imbalance = (bid_vol - ask_vol) / total
            return {
                "imbalance": imbalance,
                "bid_vol": bid_vol,
                "ask_vol": ask_vol,
            }
        except Exception as e:
            logger.error(f"Error computing imbalance: {e}")
            return None

    def _compute_wall_strength(
        self, data_manager, current_price: float, imbalance_data: Optional[Dict]
    ) -> Optional[Dict]:
        """Compute wall strength in zone around current price."""
        try:
            if not imbalance_data:
                return None

            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return None

            zone_ticks = config.ZONE_TICKS
            tick_size = config.TICK_SIZE
            zone_low = current_price - (zone_ticks * tick_size)
            zone_high = current_price + (zone_ticks * tick_size)

            bid_wall_vol = sum(q for p, q in bids if zone_low <= p <= current_price)
            ask_wall_vol = sum(q for p, q in asks if current_price <= p <= zone_high)

            depth_levels = min(config.WALL_DEPTH_LEVELS, len(bids), len(asks))
            avg_depth_vol = (imbalance_data["bid_vol"] + imbalance_data["ask_vol"]) / (2.0 * depth_levels)
            if avg_depth_vol <= 0:
                return None

            bid_wall_strength = bid_wall_vol / avg_depth_vol
            ask_wall_strength = ask_wall_vol / avg_depth_vol

            return {
                "bid_wall_strength": bid_wall_strength,
                "ask_wall_strength": ask_wall_strength,
                "bid_wall_vol": bid_wall_vol,
                "ask_wall_vol": ask_wall_vol,
                "zone_low": zone_low,
                "zone_high": zone_high,
            }
        except Exception as e:
            logger.error(f"Error computing wall strength: {e}")
            return None

    def _compute_delta_z_score(self, data_manager) -> Optional[Dict]:
        """Compute taker delta Z-score from recent trades."""
        try:
            window_sec = config.DELTA_WINDOW_SEC
            trades = data_manager.get_recent_trades(window_seconds=window_sec)
            if not trades:
                return None

            buy_vol = 0.0
            sell_vol = 0.0
            for t in trades:
                try:
                    qty = float(t.get("qty", 0.0))
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    continue
                is_buyer_maker = bool(t.get("isBuyerMaker", False))
                if not is_buyer_maker:
                    buy_vol += qty
                else:
                    sell_vol += qty

            delta = buy_vol - sell_vol
            self._delta_population.append(delta)
            if len(self._delta_population) < 10:
                return None

            pop_mean = float(np.mean(self._delta_population))
            pop_std = float(np.std(self._delta_population))
            if pop_std <= 0:
                return None

            z_score = (delta - pop_mean) / pop_std
            return {
                "delta": delta,
                "z_score": z_score,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
            }
        except Exception as e:
            logger.error(f"Error computing delta Z-score: {e}")
            return None

    def _compute_price_touch(self, data_manager, current_price: float) -> Optional[Dict]:
        """Compute distance from price to nearest strong wall."""
        try:
            bids, asks = data_manager.get_orderbook_snapshot()
            if not bids or not asks:
                return None

            tick_size = config.TICK_SIZE
            zone_ticks = config.ZONE_TICKS
            zone_low = current_price - (zone_ticks * tick_size)
            zone_high = current_price + (zone_ticks * tick_size)

            bid_levels = [(p, q) for p, q in bids if zone_low <= p <= current_price]
            ask_levels = [(p, q) for p, q in asks if current_price <= p <= zone_high]

            if not bid_levels or not ask_levels:
                return None

            bid_max_vol = max(q for _, q in bid_levels)
            ask_max_vol = max(q for _, q in ask_levels)
            bid_wall_price = next(p for p, q in bid_levels if q == bid_max_vol)
            ask_wall_price = next(p for p, q in ask_levels if q == ask_max_vol)

            bid_distance_ticks = abs(current_price - bid_wall_price) / tick_size
            ask_distance_ticks = abs(ask_wall_price - current_price) / tick_size

            return {
                "bid_distance_ticks": bid_distance_ticks,
                "ask_distance_ticks": ask_distance_ticks,
                "bid_wall_price": bid_wall_price,
                "ask_wall_price": ask_wall_price,
            }
        except Exception as e:
            logger.error(f"Error computing price touch: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════
    # ✅ STREAMLINED ENTRY FLOW
    # ══════════════════════════════════════════════════════════════════════
    
    def _calculate_entry_price(self, side: str, base_price: float, vol_regime: str) -> float:
        """
        ✅ EXCEL EXACT: Calculate entry limit price with vol-adjusted slippage
        """
        slippage_ticks = config.SLIPPAGE_TICKS_ASSUMED
        
        # Vol regime adjustment
        if vol_regime == "HIGH":
            slippage_ticks *= 1.5
        elif vol_regime == "LOW":
            slippage_ticks *= 0.8
        
        tick_size = config.TICK_SIZE
        if side == "long":
            price = base_price + (slippage_ticks * tick_size)
        else:
            price = base_price - (slippage_ticks * tick_size)
        
        return round(price, 2)

    def _enter_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        side: str,
        current_price: float,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        now_sec: float,
        vol_regime: str,
        weighted_score: float,
        oracle_inputs: Optional[OracleInputs],
    ) -> None:
        """
        ✅ STREAMLINED: Enter position with single bracket order
        
        Steps:
        1. Validate conditions
        2. Fetch balance ONCE for trade evaluation
        3. Calculate position sizing
        4. Calculate entry price (vol-adjusted)
        5. Calculate TP/SL prices (Excel exact)
        6. Place bracket order (limit + TP + SL)
        7. Record position
        """
        if self.current_position is not None:
            logger.debug("Position already open, skipping")
            return

        # Step 1: Pre-check trading allowed
        allowed, reason = risk_manager.check_trading_allowed()
        if not allowed:
            logger.debug(f"Trade not allowed: {reason}")
            return

        # ✅ Step 2: Fetch balance ONCE for this trade evaluation
        balance_info = risk_manager.get_balance_for_trade_evaluation()
        if not balance_info:
            logger.warning("Cannot fetch balance, aborting entry")
            return
        
        balance_available = balance_info["available"]
        logger.info(f"💰 Balance for trade: {balance_available:.2f} USDT")

        # Step 3: Calculate position sizing (regime-aware)
        margin_used, quantity = risk_manager.calculate_position_size_regime_aware(
            entry_price=current_price,
            vol_regime=vol_regime,
            balance_available=balance_available,
        )

        if quantity <= 0:
            logger.warning("Invalid quantity calculated")
            return

        # Step 4: Calculate entry price
        entry_price = self._calculate_entry_price(side, current_price, vol_regime)

        # ✅ Step 5: Calculate TP/SL (Excel exact methodology)
        tp_roi, sl_roi = self._get_dynamic_tp_sl(vol_regime)
        
        if side == "long":
            tp_price = round(entry_price * (1.0 + tp_roi), 2)
            sl_price = round(entry_price * (1.0 + sl_roi), 2)
        else:
            tp_price = round(entry_price * (1.0 - tp_roi), 2)
            sl_price = round(entry_price * (1.0 - sl_roi), 2)

        # Step 6: Place bracket order
        logger.info("=" * 80)
        logger.info(f"ENTERING {side.upper()} POSITION")
        logger.info(f"Score: {weighted_score:.3f} | Vol: {vol_regime}")
        logger.info(f"Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}")
        logger.info(f"Quantity: {quantity:.6f} | Margin: ${margin_used:.2f}")
        logger.info("=" * 80)

        bracket_result = order_manager.place_bracket_order(
            side="BUY" if side == "long" else "SELL",
            quantity=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )

        if not bracket_result:
            logger.error("❌ Bracket order failed")
            return

        filled_order, tp_order, sl_order = bracket_result

        # Step 7: Record position
        self.trade_seq += 1
        trade_id = f"Z{int(now_sec)}{self.trade_seq:03d}"

        self.current_position = ZScorePosition(
            trade_id=trade_id,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time_sec=now_sec,
            entry_wall_volume=wall_data.get("bid_wall_vol" if side == "long" else "ask_wall_vol", 0.0),
            wall_zone_low=wall_data.get("zone_low", 0.0),
            wall_zone_high=wall_data.get("zone_high", 0.0),
            entry_imbalance=imbalance_data.get("imbalance", 0.0),
            entry_z_score=delta_data.get("z_score", 0.0),
            tp_price=tp_price,
            sl_price=sl_price,
            margin_used=margin_used,
            tp_order_id=tp_order.get("order_id", ""),
            sl_order_id=sl_order.get("order_id", ""),
            main_order_id=filled_order.get("order_id", ""),
            entry_htf_trend=oracle_inputs.htf_trend if oracle_inputs else "UNKNOWN",
            entry_vol_regime=vol_regime,
            entry_weighted_score=weighted_score,
            last_score_check_sec=now_sec,
            trailing_sl_active=False,
        )

        risk_manager.record_trade_opened()
        self.total_trades += 1

        # Excel logging
        if self.excel_logger:
            try:
                self.excel_logger.log_entry(
                    trade_id=trade_id,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    margin_used=margin_used,
                    imbalance=imbalance_data.get("imbalance", 0.0),
                    wall_strength=wall_data.get("bid_wall_strength" if side == "long" else "ask_wall_strength", 0.0),
                    z_score=delta_data.get("z_score", 0.0),
                    htf_trend=oracle_inputs.htf_trend if oracle_inputs else "UNKNOWN",
                    vol_regime=vol_regime,
                    weighted_score=weighted_score,
                )
            except Exception as e:
                logger.error(f"Excel logging error: {e}")

        # Telegram notification
        try:
            send_telegram_message(
                f"🚀 {side.upper()} ENTRY\n"
                f"Trade: {trade_id}\n"
                f"Entry: ${entry_price:.2f}\n"
                f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f}\n"
                f"Qty: {quantity:.6f} | Margin: ${margin_used:.2f}\n"
                f"Score: {weighted_score:.3f} | Vol: {vol_regime}"
            )
        except Exception:
            pass

        logger.info(f"✅ Position opened: {trade_id}")

    # ══════════════════════════════════════════════════════════════════════
    # ✅ MANAGE OPEN POSITION (TP/SL Check + Score Decay + Trailing SL)
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
        Manage existing open position:
        1. Check TP/SL hit
        2. Check timeout
        3. ✅ NEW: Trailing stop-loss when profit > 50% of TP
        4. Score decay exit (if trend/vol/momentum deteriorated)
        """
        pos = self.current_position
        if not pos:
            return

        # Periodic status check (every 10s)
        if now_sec - self._last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            
            # Check TP hit
            tp_status = order_manager.get_order_status(pos.tp_order_id)
            if tp_status:
                status = str(tp_status.get("status", "")).upper()
                if status in ("EXECUTED", "FILLED", "PARTIALLY_EXECUTED"):
                    logger.info(f"✅ TP HIT for {pos.trade_id}")
                    self._close_position(
                        order_manager, risk_manager, current_price, now_sec, exit_reason="TP_HIT"
                    )
                    return
            
            # Check SL hit
            sl_status = order_manager.get_order_status(pos.sl_order_id)
            if sl_status:
                status = str(sl_status.get("status", "")).upper()
                if status in ("EXECUTED", "FILLED", "PARTIALLY_EXECUTED"):
                    logger.info(f"❌ SL HIT for {pos.trade_id}")
                    self._close_position(
                        order_manager, risk_manager, current_price, now_sec, exit_reason="SL_HIT"
                    )
                    return

        # ✅ NEW: Trailing stop-loss when profit > 50% of TP
        if not pos.trailing_sl_active:
            unrealized_pnl = self._calculate_unrealized_pnl(pos, current_price)
            max_possible_pnl = self._calculate_unrealized_pnl(pos, pos.tp_price)
            
            if max_possible_pnl > 0 and unrealized_pnl >= 0.5 * max_possible_pnl:
                # Move SL to halfway between entry and TP
                new_sl_price = (pos.entry_price + pos.tp_price) / 2.0
                new_sl_price = round(new_sl_price, 2)
                
                logger.info(f"🎯 TRAILING SL: Profit > 50% TP, moving SL ${pos.sl_price:.2f} → ${new_sl_price:.2f}")
                
                try:
                    # Cancel old SL
                    order_manager.cancel_order(pos.sl_order_id)
                    
                    # Place new SL
                    exit_side = "SELL" if pos.side == "long" else "BUY"
                    new_sl_order = order_manager.place_stop_loss(
                        side=exit_side,
                        quantity=pos.quantity,
                        trigger_price=new_sl_price,
                    )
                    
                    if new_sl_order and "order_id" in new_sl_order:
                        pos.sl_order_id = new_sl_order["order_id"]
                        pos.sl_price = new_sl_price
                        pos.trailing_sl_active = True
                        logger.info(f"✅ Trailing SL activated: {pos.sl_order_id}")
                    else:
                        logger.error("Failed to place trailing SL")
                        
                except Exception as e:
                    logger.error(f"Error setting trailing SL: {e}")

        # Timeout exit
        hold_duration_min = (now_sec - pos.entry_time_sec) / 60.0
        if hold_duration_min >= config.MAX_HOLD_MINUTES:
            logger.info(f"⏰ TIMEOUT: {hold_duration_min:.1f}min >= {config.MAX_HOLD_MINUTES}min")
            self._close_position(
                order_manager, risk_manager, current_price, now_sec, exit_reason="TIMEOUT"
            )
            return

        # ✅ Score decay exit (check every 2 minutes)
        if now_sec - pos.last_score_check_sec >= config.SCORE_DECAY_CHECK_INTERVAL_SEC:
            pos.last_score_check_sec = now_sec
            
            # Recalculate score
            vol_regime, atr_pct = data_manager.get_vol_regime()
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = self._compute_wall_strength(data_manager, current_price, imbalance_data) if imbalance_data else None
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)
            htf_trend = data_manager.get_htf_trend() if hasattr(data_manager, "get_htf_trend") else None
            ltf_trend = data_manager.get_ltf_trend() if hasattr(data_manager, "get_ltf_trend") else None
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD) if hasattr(data_manager, "get_ema") else None
            
            oracle_inputs = None
            try:
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
            except Exception:
                pass
            
            current_score, _, _ = self._compute_weighted_score(
                side=pos.side,
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
            
            logger.debug(f"Score decay check: entry={pos.entry_weighted_score:.3f}, current={current_score:.3f}")
            
            # ✅ Exit if score decayed below threshold AND conditions deteriorated
            if current_score < config.SCORE_DECAY_EXIT_THRESHOLD:
                # Additional checks: trend/vol/momentum must be unfavorable
                trend_reversed = False
                if htf_trend:
                    if pos.side == "long" and htf_trend == "DOWN":
                        trend_reversed = True
                    elif pos.side == "short" and htf_trend == "UP":
                        trend_reversed = True
                
                vol_spike = False
                if vol_regime == "HIGH" and pos.entry_vol_regime in ["LOW", "NEUTRAL"]:
                    vol_spike = True
                
                momentum_weak = False
                if delta_data:
                    z = delta_data["z_score"]
                    if pos.side == "long" and z < -1.0:
                        momentum_weak = True
                    elif pos.side == "short" and z > 1.0:
                        momentum_weak = True
                
                # Exit only if conditions truly deteriorated
                if trend_reversed or vol_spike or momentum_weak:
                    logger.info(
                        f"📉 SCORE DECAY EXIT: score={current_score:.3f} < {config.SCORE_DECAY_EXIT_THRESHOLD} "
                        f"(trend_rev={trend_reversed}, vol_spike={vol_spike}, momentum_weak={momentum_weak})"
                    )
                    self._close_position(
                        order_manager, risk_manager, current_price, now_sec, exit_reason="SCORE_DECAY"
                    )
                    return

        # Periodic position logging
        if now_sec - self._last_position_log_sec >= self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            unrealized_pnl = self._calculate_unrealized_pnl(pos, current_price)
            roi = (unrealized_pnl / pos.margin_used * 100) if pos.margin_used > 0 else 0
            hold_min = (now_sec - pos.entry_time_sec) / 60.0
            logger.info(
                f"📊 Position {pos.trade_id}: {pos.side.upper()} @ ${pos.entry_price:.2f}, "
                f"Current: ${current_price:.2f}, P&L: ${unrealized_pnl:.2f} ({roi:.2f}%), "
                f"Hold: {hold_min:.1f}min, TrailingSL: {pos.trailing_sl_active}"
            )

    def _close_position(
        self,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
        exit_reason: str,
    ) -> None:
        """
        Close position and clean up orders
        """
        pos = self.current_position
        if not pos:
            return

        logger.info("=" * 80)
        logger.info(f"CLOSING POSITION: {pos.trade_id}")
        logger.info(f"Reason: {exit_reason}")
        logger.info("=" * 80)

        try:
            # Cancel remaining orders
            if exit_reason == "TP_HIT":
                order_manager.cancel_order(pos.sl_order_id)
            elif exit_reason == "SL_HIT":
                order_manager.cancel_order(pos.tp_order_id)
            else:
                # Manual exit - cancel both and place market order
                order_manager.cancel_order(pos.tp_order_id)
                order_manager.cancel_order(pos.sl_order_id)
                
                exit_side = "SELL" if pos.side == "long" else "BUY"
                order_manager.place_market_order(
                    side=exit_side,
                    quantity=pos.quantity,
                    reduce_only=True
                )

        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

        # Calculate P&L
        if exit_reason == "TP_HIT":
            exit_price = pos.tp_price
        elif exit_reason == "SL_HIT":
            exit_price = pos.sl_price
        else:
            exit_price = current_price

        pnl = self._calculate_realized_pnl(pos, exit_price)
        roi = (pnl / pos.margin_used * 100.0) if pos.margin_used > 0 else 0.0
        hold_duration_min = (now_sec - pos.entry_time_sec) / 60.0

        # Update stats
        risk_manager.update_trade_stats(pnl)
        risk_manager.record_trade_closed()

        # Excel logging
        if self.excel_logger:
            try:
                self.excel_logger.log_exit(
                    trade_id=pos.trade_id,
                    exit_reason=exit_reason,
                    exit_price=exit_price,
                    pnl=pnl,
                    roi=roi,
                    hold_duration_min=hold_duration_min,
                )
            except Exception as e:
                logger.error(f"Excel logging error: {e}")

        # Telegram notification
        try:
            emoji = "✅" if pnl > 0 else "❌"
            send_telegram_message(
                f"{emoji} {pos.side.upper()} EXIT\n"
                f"Trade: {pos.trade_id}\n"
                f"Reason: {exit_reason}\n"
                f"Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}\n"
                f"P&L: ${pnl:.2f} ({roi:.2f}%)\n"
                f"Duration: {hold_duration_min:.1f}min"
            )
        except Exception:
            pass

        logger.info(f"Exit Price: ${exit_price:.2f}")
        logger.info(f"Realized P&L: ${pnl:.2f} ({roi:.2f}%)")
        logger.info(f"Hold Duration: {hold_duration_min:.1f}min")
        logger.info("=" * 80)

        # Clear position
        self.current_position = None
        self.last_exit_time_min = now_sec / 60.0

    def _calculate_unrealized_pnl(self, pos: ZScorePosition, current_price: float) -> float:
        """Calculate unrealized P&L"""
        try:
            if pos.side == "long":
                price_change = current_price - pos.entry_price
            else:
                price_change = pos.entry_price - current_price
            
            pnl = price_change * pos.quantity
            return pnl
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0

    def _calculate_realized_pnl(self, pos: ZScorePosition, exit_price: float) -> float:
        """Calculate realized P&L"""
        try:
            if pos.side == "long":
                price_change = exit_price - pos.entry_price
            else:
                price_change = pos.entry_price - exit_price
            
            pnl = price_change * pos.quantity
            return pnl
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0

    # ══════════════════════════════════════════════════════════════════════
    # Main tick handler
    # ══════════════════════════════════════════════════════════════════════
    
    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """
        Main per-tick strategy entrypoint.
        Called from main loop or WebSocket callbacks.
        """
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            # Periodic 15-min report
            self._maybe_send_15m_report(now_sec, risk_manager, current_price)

            # Manage existing position first
            if self.current_position is not None:
                self._manage_open_position(
                    data_manager, order_manager, risk_manager, current_price, now_sec
                )
                return

            # Cooldown between trades
            if self.last_exit_time_min > 0:
                minutes_since_exit = (now_sec / 60.0) - self.last_exit_time_min
                if minutes_since_exit < config.MIN_TIME_BETWEEN_TRADES:
                    return

            # Check trading allowed
            allowed, reason = risk_manager.check_trading_allowed()
            if not allowed:
                logger.debug(f"Trading not allowed: {reason}")
                return

            # Gather signals
            vol_regime, atr_pct = data_manager.get_vol_regime()
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = self._compute_wall_strength(data_manager, current_price, imbalance_data) if imbalance_data else None
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            htf_trend = data_manager.get_htf_trend() if hasattr(data_manager, "get_htf_trend") else None
            ltf_trend = data_manager.get_ltf_trend() if hasattr(data_manager, "get_ltf_trend") else None
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD) if hasattr(data_manager, "get_ema") else None

            # Build Oracle inputs
            oracle_inputs = None
            try:
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
                logger.error(f"Error building OracleInputs: {e}")

            # Compute weighted score for LONG
            weighted_score_long, components_long, reasons_long = self._compute_weighted_score(
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

            # Compute weighted score for SHORT
            weighted_score_short, components_short, reasons_short = self._compute_weighted_score(
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

            # Periodic decision logging (every 60s)
            if now_sec - self._last_decision_log_sec >= self.DECISION_LOG_INTERVAL_SEC:
                self._last_decision_log_sec = now_sec
                logger.info("=" * 80)
                logger.info(f"DECISION SNAPSHOT @ ${current_price:.2f}")
                logger.info(f"Vol: {vol_regime} | HTF: {htf_trend or 'NA'}")
                logger.info(f"LONG Score: {weighted_score_long:.3f} | SHORT Score: {weighted_score_short:.3f}")
                logger.info(f"Threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD}")
                logger.info("=" * 80)

            # Entry logic: highest score wins if above threshold
            if weighted_score_long >= config.WEIGHTED_SCORE_ENTRY_THRESHOLD or weighted_score_short >= config.WEIGHTED_SCORE_ENTRY_THRESHOLD:
                if weighted_score_long > weighted_score_short:
                    self._enter_position(
                        data_manager, order_manager, risk_manager,
                        side="long",
                        current_price=current_price,
                        imbalance_data=imbalance_data or {},
                        wall_data=wall_data or {},
                        delta_data=delta_data or {},
                        touch_data=touch_data or {},
                        now_sec=now_sec,
                        vol_regime=vol_regime,
                        weighted_score=weighted_score_long,
                        oracle_inputs=oracle_inputs,
                    )
                else:
                    self._enter_position(
                        data_manager, order_manager, risk_manager,
                        side="short",
                        current_price=current_price,
                        imbalance_data=imbalance_data or {},
                        wall_data=wall_data or {},
                        delta_data=delta_data or {},
                        touch_data=touch_data or {},
                        now_sec=now_sec,
                        vol_regime=vol_regime,
                        weighted_score=weighted_score_short,
                        oracle_inputs=oracle_inputs,
                    )

        except Exception as e:
            logger.error(f"Error in on_tick: {e}", exc_info=True)

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
                f"Price: ${current_price:.2f}",
                f"New Trades (15m): {new_trades}",
                f"Total Trades: {rm.total_trades}",
                f"Win Rate: {(rm.winning_trades / rm.total_trades * 100.0) if rm.total_trades > 0 else 0:.2f}%",
                f"Daily P&L: ${rm.daily_pnl:.2f} USDT",
                "=" * 40,
            ]

            send_telegram_message("\n".join(lines))
        except Exception as e:
            logger.error(f"Error sending 15m report: {e}")
