"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Vol-Regime + Weighted Score
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque
import logging
import numpy as np
from scipy.stats import norm
import config
from zscore_excel_logger import ZScoreExcelLogger
from telegram_notifier import send_telegram_message
from aether_oracle import AetherOracle, OracleInputs, OracleOutputs, OracleSideScores

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
    vol_regime: str  # NEW
    entry_score: float  # NEW


class ZScoreIcebergHunterStrategy:
    """Vol-Regime + Weighted Score + Event-Driven Strategy"""
    
    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 120.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0
    ENTRY_FILL_TIMEOUT_SEC = 60.0

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0
        self._delta_population: deque = deque(maxlen=3000)
        self._last_decision_log_sec: float = 0.0
        self._last_position_log_sec: float = 0.0
        self._last_status_check_sec: float = 0.0
        self._oracle = AetherOracle()
        self._last_report_sec: float = 0.0
        self._last_report_total_trades: int = 0
        
        logger.info("=" * 80)
        logger.info("Z-SCORE VOL-REGIME + WEIGHTED SCORE STRATEGY INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Score Entry Threshold: {config.SCORE_ENTRY_THRESHOLD}")
        logger.info(f"Score Exit Threshold: {config.SCORE_EXIT_THRESHOLD}")
        logger.info(f"Vol-Regime: LOW<{config.VOL_REGIME_LOW_ATR_PCT*100:.2f}% HIGH>{config.VOL_REGIME_HIGH_ATR_PCT*100:.2f}%")
        logger.info("=" * 80)

    # ======================================================================
    # VOLATILITY REGIME & DYNAMIC PARAMS
    # ======================================================================
    
    def _get_regime_params(self, data_manager, current_price: float) -> Dict:
        """
        Get dynamic params based on vol regime.
        Returns: z_thresh, wall_mult, tp_mult, sl_mult, size_pct, trail_enabled, regime
        """
        atr_pct = None
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception as e:
            logger.error(f"Error getting ATR%: {e}")
        
        regime = data_manager.get_vol_regime(atr_pct) if hasattr(data_manager, 'get_vol_regime') else "NEUTRAL"
        
        # Scale Z-threshold
        if regime == "LOW":
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH - 0.3  # 1.8
        elif regime == "HIGH":
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH + 0.3  # 2.4 (clamped formula)
        else:
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH
        
        # Wall multiplier
        if regime == "HIGH":
            wall_mult = config.VOL_REGIME_HIGH_WALL_MULT  # 3.8x
        else:
            wall_mult = config.VOL_REGIME_LOW_WALL_MULT  # 4.2x
        
        # TP/SL multipliers
        if regime == "HIGH":
            tp_mult = config.VOL_REGIME_HIGH_TP_MULT  # 1.40 (+40%)
            sl_mult = config.VOL_REGIME_HIGH_SL_MULT  # 0.90 (-10%)
            trail_enabled = True
        else:
            tp_mult = config.VOL_REGIME_LOW_TP_MULT  # 1.10 (+10%)
            sl_mult = config.VOL_REGIME_LOW_SL_MULT  # 0.97 (-3%)
            trail_enabled = False
        
        # Position sizing %
        if regime == "HIGH":
            size_pct = config.VOL_REGIME_HIGH_SIZE_PCT  # 15%
        else:
            size_pct = config.VOL_REGIME_LOW_SIZE_PCT  # 20%
        
        logger.info(f"[VOL-REGIME] {regime}: z_thresh={z_thresh:.2f}, wall_mult={wall_mult:.2f}, "
                    f"tp_mult={tp_mult:.2f}, sl_mult={sl_mult:.2f}, size_pct={size_pct:.1f}%, trail={trail_enabled}")
        
        return {
            "regime": regime,
            "z_thresh": z_thresh,
            "wall_mult": wall_mult,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "size_pct": size_pct,
            "trail_enabled": trail_enabled,
            "atr_pct": atr_pct,
        }

    # ======================================================================
    # WEIGHTED SCORE CALCULATION
    # ======================================================================
    
    def _compute_signal_score(self, value: float, threshold: float) -> float:
        """
        Normalize signal to [0, 1] using CDF.
        score = norm.cdf((value - threshold) / std)
        Simplified: assume std = threshold / 2
        """
        if threshold == 0:
            return 1.0 if value > 0 else 0.0
        std = abs(threshold) / 2.0
        z = (value - threshold) / std
        score = norm.cdf(z)
        return max(0.0, min(1.0, score))
    
    def _compute_weighted_score(
        self,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        trend_ok: bool,
        side: str,
        regime: str,
        z_thresh: float,
        wall_mult: float,
    ) -> Tuple[float, List[str]]:
        """
        Compute 5-signal weighted core score (65% total).
        Returns: (core_score, reasons)
        """
        reasons = []
        
        # 1. Imbalance score (25%)
        imb_val = imbalance_data["imbalance"]
        if side == "long":
            imb_score = self._compute_signal_score(imb_val, config.IMBALANCE_THRESHOLD)
        else:
            imb_score = self._compute_signal_score(-imb_val, config.IMBALANCE_THRESHOLD)
        reasons.append(f"imb={imb_score:.3f}")
        
        # 2. Wall score (20%)
        if side == "long":
            wall_val = wall_data["bid_wall_strength"]
        else:
            wall_val = wall_data["ask_wall_strength"]
        wall_score = self._compute_signal_score(wall_val, wall_mult)
        reasons.append(f"wall={wall_score:.3f}")
        
        # 3. Z-score (30%)
        z_val = delta_data["z_score"]
        if side == "long":
            z_score = self._compute_signal_score(z_val, z_thresh)
        else:
            z_score = self._compute_signal_score(-z_val, z_thresh)
        reasons.append(f"z={z_score:.3f}")
        
        # 4. Touch score (10%)
        if side == "long":
            touch_dist = touch_data["bid_distance_ticks"]
        else:
            touch_dist = touch_data["ask_distance_ticks"]
        touch_score = 1.0 if touch_dist <= config.PRICE_TOUCH_THRESHOLD_TICKS else 0.0
        reasons.append(f"touch={touch_score:.3f}")
        
        # 5. Trend score (15%) with RANGE bonus
        trend_score = 1.0 if trend_ok else 0.0
        # Apply RANGE bonus if HTF is RANGE
        # (Will be applied in main logic based on htf_trend)
        reasons.append(f"trend={trend_score:.3f}")
        
        # Weighted sum
        core_score = (
            imb_score * config.SCORE_IMB_WEIGHT +
            wall_score * config.SCORE_WALL_WEIGHT +
            z_score * config.SCORE_Z_WEIGHT +
            touch_score * config.SCORE_TOUCH_WEIGHT +
            trend_score * config.SCORE_TREND_WEIGHT
        )
        
        return core_score, reasons

    # ======================================================================
    # AETHER FUSION (9-signal)
    # ======================================================================
    
    def _compute_aether_fusion_score(
        self,
        oracle_inputs: Optional[OracleInputs],
        oracle_outputs: Optional[OracleOutputs],
        side: str,
    ) -> Tuple[float, List[str]]:
        """
        Compute Aether 9-signal fusion (35% weight).
        Signals: CVD (10%), LV avg (5%), Hurst/BOS (10%), LSTM (10%)
        Returns: (aether_score, reasons)
        """
        if oracle_inputs is None or oracle_outputs is None:
            return 0.0, ["aether=MISSING"]
        
        reasons = []
        components = []
        
        # CVD score (10%)
        if oracle_inputs.norm_cvd is not None:
            cvd_val = oracle_inputs.norm_cvd
            if side == "long":
                cvd_score = max(0.0, min(1.0, (cvd_val + 1.0) / 2.0))  # Map [-1,1] to [0,1]
            else:
                cvd_score = max(0.0, min(1.0, (-cvd_val + 1.0) / 2.0))
            components.append((cvd_score, config.AETHER_CVD_WEIGHT))
            reasons.append(f"cvd={cvd_score:.3f}")
        else:
            reasons.append("cvd=MISS")
        
        # LV average (5%)
        lv_vals = [oracle_inputs.lv_1m, oracle_inputs.lv_5m, oracle_inputs.lv_15m]
        lv_vals = [v for v in lv_vals if v is not None]
        if lv_vals:
            lv_avg = sum(lv_vals) / len(lv_vals)
            # Normalize: higher LV = better liquidity (arbitrary scale)
            lv_score = min(1.0, lv_avg / 100.0)  # Assume 100 is max
            components.append((lv_score, config.AETHER_LV_WEIGHT))
            reasons.append(f"lv={lv_score:.3f}")
        else:
            reasons.append("lv=MISS")
        
        # Hurst/BOS blend (10%)
        hurst_bos_vals = []
        if oracle_inputs.hurst is not None:
            # Hurst: mean-reversion favors wall scalping (H < 0.5 better)
            hurst_score = 1.0 - oracle_inputs.hurst  # Invert: lower H = higher score
            hurst_bos_vals.append(hurst_score)
        if oracle_inputs.bos_align is not None:
            hurst_bos_vals.append(oracle_inputs.bos_align)
        if hurst_bos_vals:
            hurst_bos_score = sum(hurst_bos_vals) / len(hurst_bos_vals)
            components.append((hurst_bos_score, config.AETHER_HURST_BOS_WEIGHT))
            reasons.append(f"hurst_bos={hurst_bos_score:.3f}")
        else:
            reasons.append("hurst_bos=MISS")
        
        # LSTM up-prob (10%)
        if oracle_outputs:
            if side == "long" and oracle_outputs.long_scores.fused is not None:
                lstm_score = oracle_outputs.long_scores.fused
            elif side == "short" and oracle_outputs.short_scores.fused is not None:
                lstm_score = oracle_outputs.short_scores.fused
            else:
                lstm_score = None
            
            if lstm_score is not None:
                components.append((lstm_score, config.AETHER_LSTM_WEIGHT))
                reasons.append(f"lstm={lstm_score:.3f}")
            else:
                reasons.append("lstm=MISS")
        else:
            reasons.append("lstm=MISS")
        
        # Fused score
        if not components:
            return 0.0, reasons
        
        num = sum(v * w for v, w in components)
        den = sum(w for _, w in components)
        aether_score = num / den if den > 0 else 0.0
        
        return aether_score, reasons

    # ======================================================================
    # WIN PROBABILITY OVERLAY
    # ======================================================================
    
    def _compute_win_prob(
        self,
        lstm_score: float,
        z_score_norm: float,
        cvd_norm: float,
        lv_norm: float,
    ) -> float:
        """
        Win probability overlay:
        wp = 0.4 + 0.2*lstm + 0.2*z_sign + 0.1*cvd + 0.1*lv
        """
        wp = (
            config.WINPROB_BASE +
            config.WINPROB_LSTM_WEIGHT * lstm_score +
            config.WINPROB_Z_WEIGHT * z_score_norm +
            config.WINPROB_CVD_WEIGHT * cvd_norm +
            config.WINPROB_LV_WEIGHT * lv_norm
        )
        return max(0.0, min(1.0, wp))

    # ======================================================================
    # MAIN TICK HANDLER (EVENT-DRIVEN)
    # ======================================================================
    
    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """
        Main per-tick strategy (called on WS callbacks).
        """
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return
            
            now_sec = time.time()
            
            # 15-min report
            self._maybe_send_15m_report(now_sec, risk_manager, current_price)
            
            # Manage open position (includes score decay exit)
            if self.current_position is not None:
                self._manage_open_position(
                    data_manager=data_manager,
                    order_manager=order_manager,
                    risk_manager=risk_manager,
                    current_price=current_price,
                    now_sec=now_sec,
                )
                return
            
            # Cooldown
            if self.last_exit_time_min > 0:
                minutes_since_exit = (now_sec / 60.0) - self.last_exit_time_min
                if minutes_since_exit < config.MIN_TIME_BETWEEN_TRADES:
                    return
            
            # Risk check
            allowed, reason = risk_manager.check_trading_allowed()
            if not allowed:
                return
            
            # Core metrics
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(data_manager, current_price, imbalance_data)
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)
            
            # HTF trend (LTF disabled per request)
            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(f"Error fetching HTF trend: {e}", exc_info=True)
            
            # Oracle
            oracle_inputs: Optional[OracleInputs] = None
            oracle_outputs: Optional[OracleOutputs] = None
            try:
                oracle_inputs = self._oracle.build_inputs(
                    data_manager=data_manager,
                    current_price=current_price,
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    htf_trend=htf_trend,
                    ltf_trend=None,  # Disabled
                )
                oracle_outputs = self._oracle.decide(oracle_inputs, risk_manager)
            except Exception as e:
                logger.error(f"Oracle error: {e}", exc_info=True)
            
            # Excel log
            if (
                self.excel_logger
                and imbalance_data is not None
                and wall_data is not None
                and delta_data is not None
                and touch_data is not None
            ):
                self.excel_logger.log_parameters(
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    current_price=current_price,
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    decision="CHECK",
                    reason="Core metrics computed",
                    htf_trend=htf_trend,
                )
            
            # Decision + entry
            self._try_entries_and_log(
                data_manager=data_manager,
                order_manager=order_manager,
                risk_manager=risk_manager,
                current_price=current_price,
                imbalance_data=imbalance_data,
                wall_data=wall_data,
                delta_data=delta_data,
                touch_data=touch_data,
                htf_trend=htf_trend,
                now_sec=now_sec,
                oracle_inputs=oracle_inputs,
                oracle_outputs=oracle_outputs,
            )
        
        except Exception as e:
            logger.error(f"Error in on_tick: {e}", exc_info=True)

    # ======================================================================
    # ENTRY DECISION WITH WEIGHTED SCORE
    # ======================================================================
    
    def _try_entries_and_log(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        htf_trend: Optional[str],
        now_sec: float,
        oracle_inputs: Optional[OracleInputs],
        oracle_outputs: Optional[OracleOutputs],
    ) -> None:
        """
        Evaluate LONG/SHORT with weighted score gauntlet.
        """
        if (
            imbalance_data is None
            or wall_data is None
            or delta_data is None
            or touch_data is None
        ):
            return
        
        # Get regime params
        regime_params = self._get_regime_params(data_manager, current_price)
        regime = regime_params["regime"]
        z_thresh = regime_params["z_thresh"]
        wall_mult = regime_params["wall_mult"]
        
        # EMA trend
        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception:
            pass
        
        trend_long_ok = (ema_val is not None and current_price > ema_val)
        trend_short_ok = (ema_val is not None and current_price < ema_val)
        
        # RANGE bonus
        range_bonus = 1.0
        if htf_trend and htf_trend.upper() in ("RANGE", "RANGEBOUND"):
            if regime == "LOW":
                range_bonus = config.RANGE_BONUS_LOW  # 0.8
            else:
                range_bonus = config.RANGE_BONUS_HIGH  # 0.5
        
        # Compute core scores
        long_core, long_core_reasons = self._compute_weighted_score(
            imbalance_data, wall_data, delta_data, touch_data,
            trend_long_ok, "long", regime, z_thresh, wall_mult
        )
        long_core *= range_bonus
        
        short_core, short_core_reasons = self._compute_weighted_score(
            imbalance_data, wall_data, delta_data, touch_data,
            trend_short_ok, "short", regime, z_thresh, wall_mult
        )
        short_core *= range_bonus
        
        # Aether fusion
        long_aether, long_aether_reasons = self._compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "long"
        )
        short_aether, short_aether_reasons = self._compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "short"
        )
        
        # Total scores (65% core + 35% aether)
        long_total = long_core * 0.65 + long_aether * 0.35
        short_total = short_core * 0.65 + short_aether * 0.35
        
        # Win prob overlay
        lstm_long = oracle_outputs.long_scores.fused if (oracle_outputs and oracle_outputs.long_scores.fused) else 0.5
        lstm_short = oracle_outputs.short_scores.fused if (oracle_outputs and oracle_outputs.short_scores.fused) else 0.5
        
        z_norm_long = max(0.0, delta_data["z_score"] / z_thresh) if delta_data["z_score"] > 0 else 0.0
        z_norm_short = max(0.0, -delta_data["z_score"] / z_thresh) if delta_data["z_score"] < 0 else 0.0
        
        cvd_norm_long = (oracle_inputs.norm_cvd + 1.0) / 2.0 if (oracle_inputs and oracle_inputs.norm_cvd is not None) else 0.5
        cvd_norm_short = 1.0 - cvd_norm_long
        
        lv_norm = 0.5  # Placeholder
        
        winprob_long = self._compute_win_prob(lstm_long, z_norm_long, cvd_norm_long, lv_norm)
        winprob_short = self._compute_win_prob(lstm_short, z_norm_short, cvd_norm_short, lv_norm)
        
        # Entry decision
        long_entry = (long_total > config.SCORE_ENTRY_THRESHOLD and 
                      winprob_long > config.WINPROB_ENTRY_THRESHOLD)
        short_entry = (short_total > config.SCORE_ENTRY_THRESHOLD and 
                       winprob_short > config.WINPROB_ENTRY_THRESHOLD)
        
        # Reasons
        long_reasons = [f"core={long_core:.3f}"] + long_core_reasons + [f"aether={long_aether:.3f}"] + long_aether_reasons + [f"total={long_total:.3f}", f"wp={winprob_long:.3f}"]
        short_reasons = [f"core={short_core:.3f}"] + short_core_reasons + [f"aether={short_aether:.3f}"] + short_aether_reasons + [f"total={short_total:.3f}", f"wp={winprob_short:.3f}"]
        
        # Log
        self._log_decision_state(
            now_sec, current_price, data_manager,
            imbalance_data, wall_data, delta_data, touch_data,
            long_entry, short_entry, long_reasons, short_reasons,
            htf_trend, None, oracle_inputs, oracle_outputs, regime_params
        )
        
        # Enter
        if long_entry:
            self._enter_position(
                data_manager, order_manager, risk_manager, "long", current_price,
                imbalance_data, wall_data, delta_data, touch_data, now_sec,
                regime_params, long_total
            )
        elif short_entry:
            self._enter_position(
                data_manager, order_manager, risk_manager, "short", current_price,
                imbalance_data, wall_data, delta_data, touch_data, now_sec,
                regime_params, short_total
            )

    # ======================================================================
    # POSITION ENTRY
    # ======================================================================
    
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
        regime_params: Dict,
        entry_score: float,
    ) -> None:
        """Enter position with vol-regime sizing and TP/SL."""
        regime = regime_params["regime"]
        tp_mult = regime_params["tp_mult"]
        sl_mult = regime_params["sl_mult"]
        
        # Position sizing
        quantity, margin_used = risk_manager.calculate_position_size_vol_regime(
            entry_price=current_price,
            regime=regime,
        )
        
        if quantity <= 0:
            logger.error("Position sizing returned 0 quantity")
            return
        
        # TP/SL prices
        base_tp_roi = config.PROFIT_TARGET_ROI
        base_sl_roi = config.STOP_LOSS_ROI
        
        tp_roi = base_tp_roi * tp_mult
        sl_roi = base_sl_roi * sl_mult
        
        if side == "long":
            tp_price = current_price * (1 + tp_roi)
            sl_price = current_price * (1 + sl_roi)
        else:
            tp_price = current_price * (1 - tp_roi)
            sl_price = current_price * (1 - sl_roi)
        
        logger.info("=" * 80)
        logger.info(f"ENTERING {side.upper()} POSITION | REGIME={regime}")
        logger.info(f"Price={current_price:.2f}, Qty={quantity:.6f}, Margin={margin_used:.2f}")
        logger.info(f"TP={tp_price:.2f} (ROI={tp_roi*100:.2f}%), SL={sl_price:.2f} (ROI={sl_roi*100:.2f}%)")
        logger.info(f"Entry Score={entry_score:.3f}")
        logger.info("=" * 80)
        
        # Place main order
        market_side = "BUY" if side == "long" else "SELL"
        main_order = order_manager.place_market_order(
            side=market_side,
            quantity=quantity,
            reduce_only=False,
        )
        
        if not main_order:
            logger.error("Main order placement failed")
            return
        
        main_order_id = main_order.get("order_id", "")
        
        # Wait for fill
        try:
            filled_order = order_manager.wait_for_fill(main_order_id, timeout_sec=5.0)
            entry_price = order_manager.extract_fill_price(filled_order)
        except Exception as e:
            logger.error(f"Main order fill failed: {e}")
            return
        
        # Place TP/SL
        tp_side = "SELL" if side == "long" else "BUY"
        tp_order = order_manager.place_take_profit(
            side=tp_side,
            quantity=quantity,
            trigger_price=tp_price,
        )
        sl_order = order_manager.place_stop_loss(
            side=tp_side,
            quantity=quantity,
            trigger_price=sl_price,
        )
        
        if not tp_order or not sl_order:
            logger.error("TP/SL placement failed - flattening")
            order_manager.place_market_order(side=tp_side, quantity=quantity, reduce_only=True)
            return
        
        # Store position
        self.trade_seq += 1
        htf_trend = "UNKNOWN"
        try:
            if hasattr(data_manager, "get_htf_trend"):
                htf_trend = data_manager.get_htf_trend() or "UNKNOWN"
        except:
            pass
        
        self.current_position = ZScorePosition(
            trade_id=f"ZS{self.trade_seq:04d}",
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time_sec=now_sec,
            entry_wall_volume=wall_data["bid_vol_zone"] if side == "long" else wall_data["ask_vol_zone"],
            wall_zone_low=wall_data["zone_low"],
            wall_zone_high=wall_data["zone_high"],
            entry_imbalance=imbalance_data["imbalance"],
            entry_z_score=delta_data["z_score"],
            tp_price=tp_price,
            sl_price=sl_price,
            margin_used=margin_used,
            tp_order_id=tp_order.get("order_id", ""),
            sl_order_id=sl_order.get("order_id", ""),
            main_order_id=main_order_id,
            main_filled=True,
            tp_reduced=False,
            entry_htf_trend=htf_trend,
            vol_regime=regime,
            entry_score=entry_score,
        )
        
        risk_manager.record_trade_opened()
        
        msg = (
            f"🚀 {side.upper()} ENTRY | {regime}\n"
            f"Price: {entry_price:.2f}\n"
            f"Qty: {quantity:.6f} BTC\n"
            f"TP: {tp_price:.2f} | SL: {sl_price:.2f}\n"
            f"Score: {entry_score:.3f}"
        )
        try:
            send_telegram_message(msg)
        except:
            pass
        
        logger.info(f"✓ Position opened: {self.current_position.trade_id}")

    # ======================================================================
    # POSITION MANAGEMENT (SCORE DECAY + TRAILING SL)
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
        Manage open position:
        - Score decay exit
        - Trailing SL in HIGH vol
        - TP/SL/time exits
        """
        pos = self.current_position
        hold_sec = now_sec - pos.entry_time_sec
        hold_min = hold_sec / 60.0
        
        # Check TP/SL status
        if now_sec - self._last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            self._check_bracket_exits(order_manager, risk_manager, current_price, now_sec)
        
        # Score decay exit
        if hold_min >= 2.0:  # After 2 min, start rescoring
            self._check_score_decay_exit(
                data_manager, order_manager, risk_manager, current_price, now_sec
            )
        
        # Trailing SL in HIGH vol
        if pos.vol_regime == "HIGH":
            self._check_trailing_sl(order_manager, current_price)
        
        # Time stop
        if hold_min >= config.MAX_HOLD_MINUTES:
            logger.warning(f"Time stop at {hold_min:.1f} min")
            self._exit_position(
                order_manager, risk_manager, current_price, "TIME_STOP", now_sec
            )

    def _check_score_decay_exit(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        Rescore position; exit if score < threshold.
        """
        pos = self.current_position
        if pos is None:
            return
        
        # Recompute core metrics
        imbalance_data = self._compute_imbalance(data_manager)
        wall_data = (
            self._compute_wall_strength(data_manager, current_price, imbalance_data)
            if imbalance_data is not None
            else None
        )
        delta_data = self._compute_delta_z_score(data_manager)
        touch_data = self._compute_price_touch(data_manager, current_price)
        
        if not all([imbalance_data, wall_data, delta_data, touch_data]):
            return
        
        # Get regime params
        regime_params = self._get_regime_params(data_manager, current_price)
        z_thresh = regime_params["z_thresh"]
        wall_mult = regime_params["wall_mult"]
        
        # EMA trend
        ema_val = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except:
            pass
        
        trend_ok = False
        if pos.side == "long":
            trend_ok = (ema_val is not None and current_price > ema_val)
        else:
            trend_ok = (ema_val is not None and current_price < ema_val)
        
        # Compute score
        core_score, _ = self._compute_weighted_score(
            imbalance_data, wall_data, delta_data, touch_data,
            trend_ok, pos.side, regime_params["regime"], z_thresh, wall_mult
        )
        
        # Simple: use core only for decay (65%)
        rescore = core_score * 0.65
        
        logger.info(f"[SCORE DECAY] rescore={rescore:.3f} (threshold={config.SCORE_EXIT_THRESHOLD})")
        
        if rescore < config.SCORE_EXIT_THRESHOLD:
            logger.warning(f"Score decay exit: {rescore:.3f} < {config.SCORE_EXIT_THRESHOLD}")
            self._exit_position(
                order_manager, risk_manager, current_price, "SCORE_DECAY", now_sec
            )

    def _check_trailing_sl(self, order_manager, current_price: float) -> None:
        """
        Trailing SL in HIGH vol: after +10% profit, move SL to entry + 0.05%.
        """
        pos = self.current_position
        if pos is None or pos.vol_regime != "HIGH":
            return
        
        direction = 1.0 if pos.side == "long" else -1.0
        pnl_roi = (current_price - pos.entry_price) * direction / pos.entry_price
        
        if pnl_roi >= config.HIGH_VOL_TRAIL_PROFIT_PCT and not pos.tp_reduced:
            # Move SL to entry + buffer
            new_sl = pos.entry_price * (1 + config.HIGH_VOL_TRAIL_BUFFER_PCT * direction)
            logger.info(f"[TRAIL SL] Moving SL to {new_sl:.2f} (entry+buffer)")
            
            # Cancel old SL
            order_manager.cancel_order(pos.sl_order_id)
            
            # Place new SL
            tp_side = "SELL" if pos.side == "long" else "BUY"
            new_sl_order = order_manager.place_stop_loss(
                side=tp_side,
                quantity=pos.quantity,
                trigger_price=new_sl,
            )
            
            if new_sl_order:
                pos.sl_order_id = new_sl_order.get("order_id", "")
                pos.sl_price = new_sl
                pos.tp_reduced = True  # Flag to avoid repeated updates

    def _check_bracket_exits(
        self,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """Check TP/SL order status."""
        pos = self.current_position
        if pos is None:
            return
        
        tp_status = order_manager.get_order_status(pos.tp_order_id)
        sl_status = order_manager.get_order_status(pos.sl_order_id)
        
        if tp_status and tp_status.get("status", "").upper() in ("EXECUTED", "FILLED"):
            logger.info("TP hit")
            self._exit_position(order_manager, risk_manager, current_price, "TP_HIT", now_sec)
        elif sl_status and sl_status.get("status", "").upper() in ("EXECUTED", "FILLED"):
            logger.info("SL hit")
            self._exit_position(order_manager, risk_manager, current_price, "SL_HIT", now_sec)

    def _exit_position(
        self,
        order_manager,
        risk_manager,
        exit_price: float,
        reason: str,
        now_sec: float,
    ) -> None:
        """Exit position and cleanup."""
        pos = self.current_position
        if pos is None:
            return
        
        # Cancel TP/SL
        order_manager.cancel_order(pos.tp_order_id)
        order_manager.cancel_order(pos.sl_order_id)
        
        # Flatten
        exit_side = "SELL" if pos.side == "long" else "BUY"
        exit_order = order_manager.place_market_order(
            side=exit_side,
            quantity=pos.quantity,
            reduce_only=True,
        )
        
        if exit_order:
            try:
                filled = order_manager.wait_for_fill(exit_order.get("order_id", ""), timeout_sec=3.0)
                exit_price = order_manager.extract_fill_price(filled)
            except:
                pass
        
        # Calculate P&L
        direction = 1.0 if pos.side == "long" else -1.0
        pnl = (exit_price - pos.entry_price) * direction * pos.quantity
        roi = pnl / pos.margin_used if pos.margin_used > 0 else 0.0
        
        risk_manager.update_trade_stats(pnl)
        
        logger.info("=" * 80)
        logger.info(f"POSITION CLOSED: {pos.trade_id} | {reason}")
        logger.info(f"Entry: {pos.entry_price:.2f} | Exit: {exit_price:.2f}")
        logger.info(f"P&L: {pnl:.2f} USDT | ROI: {roi*100:.2f}%")
        logger.info("=" * 80)
        
        msg = (
            f"🛑 EXIT {pos.side.upper()} | {reason}\n"
            f"Entry: {pos.entry_price:.2f} → Exit: {exit_price:.2f}\n"
            f"P&L: {pnl:.2f} USDT ({roi*100:.2f}%)\n"
            f"Hold: {(now_sec - pos.entry_time_sec)/60.0:.1f} min"
        )
        try:
            send_telegram_message(msg)
        except:
            pass
        
        self.last_exit_time_min = now_sec / 60.0
        self.current_position = None

    # ======================================================================
    # CORE METRIC CALCULATORS (unchanged)
    # ======================================================================
    
    def _compute_imbalance(self, data_manager) -> Optional[Dict]:
        """Compute orderbook imbalance."""
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks or len(bids) < config.WALL_DEPTH_LEVELS or len(asks) < config.WALL_DEPTH_LEVELS:
            return None
        
        depth_bids = bids[:config.WALL_DEPTH_LEVELS]
        depth_asks = asks[:config.WALL_DEPTH_LEVELS]
        
        total_bid = sum(vol for _, vol in depth_bids)
        total_ask = sum(vol for _, vol in depth_asks)
        
        if total_bid + total_ask == 0:
            return None
        
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        long_ok = imbalance >= config.IMBALANCE_THRESHOLD
        short_ok = imbalance <= -config.IMBALANCE_THRESHOLD
        
        return {
            "imbalance": imbalance,
            "total_bid": total_bid,
            "total_ask": total_ask,
            "long_ok": long_ok,
            "short_ok": short_ok,
        }

    def _compute_wall_strength(
        self, data_manager, current_price: float, imbalance_data: Dict
    ) -> Optional[Dict]:
        """Compute wall strength."""
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None
        
        zone_low = current_price - config.ZONE_TICKS * config.TICK_SIZE
        zone_high = current_price + config.ZONE_TICKS * config.TICK_SIZE
        
        bid_vol_zone = sum(vol for px, vol in bids if zone_low <= px <= zone_high)
        ask_vol_zone = sum(vol for px, vol in asks if zone_low <= px <= zone_high)
        
        avg_bid = imbalance_data["total_bid"] / config.WALL_DEPTH_LEVELS
        avg_ask = imbalance_data["total_ask"] / config.WALL_DEPTH_LEVELS
        
        bid_wall_strength = bid_vol_zone / avg_bid if avg_bid > 0 else 0.0
        ask_wall_strength = ask_vol_zone / avg_ask if avg_ask > 0 else 0.0
        
        long_wall_ok = bid_wall_strength >= config.MIN_WALL_VOLUME_MULT
        short_wall_ok = ask_wall_strength >= config.MIN_WALL_VOLUME_MULT
        
        return {
            "bid_wall_strength": bid_wall_strength,
            "ask_wall_strength": ask_wall_strength,
            "bid_vol_zone": bid_vol_zone,
            "ask_vol_zone": ask_vol_zone,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "long_wall_ok": long_wall_ok,
            "short_wall_ok": short_wall_ok,
        }

    def _compute_delta_z_score(self, data_manager) -> Optional[Dict]:
        """Compute taker delta Z-score."""
        trades = data_manager.get_recent_trades(window_seconds=config.DELTA_WINDOW_SEC)
        if not trades:
            return None
        
        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            try:
                qty = float(t.get("qty", 0.0))
            except:
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
        
        if len(self._delta_population) < 30:
            return None
        
        pop = list(self._delta_population)
        mean = sum(pop) / len(pop)
        var = sum((x - mean) ** 2 for x in pop) / len(pop)
        std = var ** 0.5
        
        z_score = (delta - mean) / std if std > 0 else 0.0
        
        long_ok = z_score >= config.DELTA_Z_THRESHOLD
        short_ok = z_score <= -config.DELTA_Z_THRESHOLD
        
        return {
            "delta": delta,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "z_score": z_score,
            "long_ok": long_ok,
            "short_ok": short_ok,
        }

    def _compute_price_touch(self, data_manager, current_price: float) -> Optional[Dict]:
        """Compute price touch distance."""
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None
        
        nearest_bid = bids[0][0]
        nearest_ask = asks[0][0]
        
        bid_distance_ticks = abs(current_price - nearest_bid) / config.TICK_SIZE
        ask_distance_ticks = abs(current_price - nearest_ask) / config.TICK_SIZE
        
        long_touch_ok = bid_distance_ticks <= config.PRICE_TOUCH_THRESHOLD_TICKS
        short_touch_ok = ask_distance_ticks <= config.PRICE_TOUCH_THRESHOLD_TICKS
        
        return {
            "nearest_bid": nearest_bid,
            "nearest_ask": nearest_ask,
            "bid_distance_ticks": bid_distance_ticks,
            "ask_distance_ticks": ask_distance_ticks,
            "long_touch_ok": long_touch_ok,
            "short_touch_ok": short_touch_ok,
        }

    # ======================================================================
    # LOGGING
    # ======================================================================
    
    def _log_decision_state(
        self,
        now_sec: float,
        current_price: float,
        data_manager,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        long_ready: bool,
        short_ready: bool,
        long_reasons: List[str],
        short_reasons: List[str],
        htf_trend: Optional[str],
        ltf_trend: Optional[str],
        oracle_inputs: Optional[OracleInputs],
        oracle_outputs: Optional[OracleOutputs],
        regime_params: Optional[Dict],
    ) -> None:
        """Log decision snapshot."""
        if now_sec - self._last_decision_log_sec < self.DECISION_LOG_INTERVAL_SEC:
            return
        
        self._last_decision_log_sec = now_sec
        
        logger.info("-" * 80)
        logger.info(f"DECISION @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | price={current_price:.2f}")
        
        if regime_params:
            logger.info(f" VOL-REGIME: {regime_params['regime']} | ATR%={regime_params.get('atr_pct', 0)*100:.3f}%")
            logger.info(f"  z_thresh={regime_params['z_thresh']:.2f}, wall_mult={regime_params['wall_mult']:.2f}")
        
        if imbalance_data:
            logger.info(f" Imbalance: {imbalance_data['imbalance']:.3f}")
        if wall_data:
            logger.info(f" Wall: bid={wall_data['bid_wall_strength']:.2f}, ask={wall_data['ask_wall_strength']:.2f}")
        if delta_data:
            logger.info(f" Z-score: {delta_data['z_score']:.2f}")
        if touch_data:
            logger.info(f" Touch: bid={touch_data['bid_distance_ticks']:.1f}, ask={touch_data['ask_distance_ticks']:.1f}")
        
        if htf_trend:
            logger.info(f" HTF Trend: {htf_trend}")
        
        if long_ready:
            logger.info(f" LONG READY: {' | '.join(long_reasons)}")
        else:
            logger.info(f" LONG BLOCKED: {' | '.join(long_reasons)}")
        
        if short_ready:
            logger.info(f" SHORT READY: {' | '.join(short_reasons)}")
        else:
            logger.info(f" SHORT BLOCKED: {' | '.join(short_reasons)}")
        
        logger.info("-" * 80)

    def _maybe_send_15m_report(
        self, now_sec: float, risk_manager, current_price: float
    ) -> None:
        """15-min Telegram report."""
        if self._last_report_sec == 0.0:
            self._last_report_sec = now_sec
            self._last_report_total_trades = int(getattr(risk_manager, "total_trades", 0))
            return
        
        if now_sec - self._last_report_sec < 900.0:
            return
        
        self._last_report_sec = now_sec
        
        total_trades = int(getattr(risk_manager, "total_trades", 0))
        winning_trades = int(getattr(risk_manager, "winning_trades", 0))
        realized_pnl = float(getattr(risk_manager, "realized_pnl", 0.0))
        
        trades_since = max(0, total_trades - self._last_report_total_trades)
        self._last_report_total_trades = total_trades
        
        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        pos_summary = "None"
        if self.current_position:
            pos = self.current_position
            direction = 1.0 if pos.side == "long" else -1.0
            u_pnl = (current_price - pos.entry_price) * direction * pos.quantity
            pos_summary = f"{pos.side.upper()} {pos.quantity:.6f} @ {pos.entry_price:.2f}, uPnL≈{u_pnl:.2f}"
        
        msg = (
            f"📊 Z-Score 15m Report\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Price: {current_price:.2f}\n"
            f"Trades: {total_trades} | WR: {win_rate:.1f}%\n"
            f"Realized P&L: {realized_pnl:.2f}\n"
            f"Position: {pos_summary}"
        )
        try:
            send_telegram_message(msg)
        except:
            pass
