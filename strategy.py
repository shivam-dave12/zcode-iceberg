"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version - PRODUCTION READY

- Single bracket order per entry (LIMIT + TP + SL)
- Margin/quantity-based TP/SL calculation (Excel-style)
- Clean entry/exit flow with proper position management
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

import math

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
    ENTRY_FILL_TIMEOUT_SEC = 60.0

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

        # Oracle instance
        self._oracle = AetherOracle()

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

    # =====================================================================
    # Vol-Regime Helpers
    # =====================================================================

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
        return config.VOL_REGIME_WALL_MULT_BASE

    def _get_dynamic_tp_sl_multipliers(self, vol_regime: str) -> Tuple[float, float]:
        """Get dynamic TP/SL multipliers based on regime."""
        if vol_regime == "HIGH":
            return 1.1, 0.9  # Slightly wider TP, tighter SL in high vol
        elif vol_regime == "LOW":
            return 0.8, 1.1  # Tighter TP, wider SL in low vol
        return 1.0, 1.0  # Neutral

    # =====================================================================
    # Raw Gate Computation Methods
    # =====================================================================

    def _compute_imbalance(self, data_manager) -> Optional[float]:
        """Compute orderbook imbalance (bid - ask) / total."""
        try:
            with data_manager._orderbook_lock:
                bids = data_manager._orderbook_bids[:config.WALL_DEPTH_LEVELS]
                asks = data_manager._orderbook_asks[:config.WALL_DEPTH_LEVELS]

            if not bids or not asks:
                return None

            bid_vol = sum(qty for _, qty in bids)
            ask_vol = sum(qty for _, qty in asks)
            total_vol = bid_vol + ask_vol

            if total_vol == 0:
                return None

            imbalance = (bid_vol - ask_vol) / total_vol
            return imbalance
        except Exception as e:
            logger.error(f"Error computing imbalance: {e}")
            return None

    def _compute_wall_strength(self, data_manager, current_price: float, side: str) -> Optional[float]:
        """Compute wall strength in zone around price."""
        try:
            tick_size = config.TICK_SIZE
            zone_low = current_price - config.ZONE_TICKS * tick_size
            zone_high = current_price + config.ZONE_TICKS * tick_size

            with data_manager._orderbook_lock:
                if side == "long":
                    relevant_levels = [(p, q) for p, q in data_manager._orderbook_bids if zone_low <= p <= zone_high]
                else:
                    relevant_levels = [(p, q) for p, q in data_manager._orderbook_asks if zone_low <= p <= zone_high]

            if not relevant_levels:
                return None

            wall_vol = sum(qty for _, qty in relevant_levels)

            # Average depth volume (top 5 levels bid + ask)
            avg_bids = sum(qty for _, qty in data_manager._orderbook_bids[:5])
            avg_asks = sum(qty for _, qty in data_manager._orderbook_asks[:5])
            avg_depth_vol = (avg_bids + avg_asks) / 10.0 if (avg_bids + avg_asks) > 0 else 1e-6

            strength = wall_vol / avg_depth_vol
            return strength
        except Exception as e:
            logger.error(f"Error computing wall strength: {e}")
            return None

    def _compute_delta_z_score(self, data_manager, window_sec: int = config.DELTA_WINDOW_SEC) -> Optional[float]:
        """Compute Z-score of taker delta in window."""
        try:
            trades = data_manager.get_recent_trades(window_seconds=window_sec)
            if not trades:
                return None

            deltas = []
            last_price = data_manager.get_last_price()
            for t in trades:
                qty = float(t.get("qty", 0.0))
                price = float(t.get("price", last_price))
                # Simple taker delta: qty * sign(price change)
                sign = 1 if price > last_price else -1 if price < last_price else 0
                deltas.append(qty * sign)

            if not deltas:
                return None

            # Update population
            self._delta_population.extend(deltas)
            if len(self._delta_population) < 30:
                return None

            # Recent delta mean
            recent_delta = np.mean(deltas[-10:]) if len(deltas) >= 10 else np.mean(deltas)

            # Z-score from population
            pop_mean = np.mean(self._delta_population)
            pop_std = np.std(self._delta_population)
            if pop_std == 0:
                return None

            z_score = (recent_delta - pop_mean) / pop_std
            return abs(z_score)
        except Exception as e:
            logger.error(f"Error computing delta Z-score: {e}")
            return None

    def _check_price_touch(self, data_manager, wall_price: float) -> bool:
        """Check if current price touches wall within threshold."""
        current_price = data_manager.get_last_price()
        if current_price <= 0 or wall_price <= 0:
            return False

        tick_size = config.TICK_SIZE
        threshold = config.PRICE_TOUCH_THRESHOLD_TICKS * tick_size
        return abs(current_price - wall_price) <= threshold

    def _get_vol_regime(self, data_manager) -> str:
        """Determine vol regime from ATR %."""
        atr_pct = data_manager.get_atr_percent(window_minutes=config.ATR_WINDOW_MINUTES)
        if atr_pct is None:
            return "NEUTRAL"
        if atr_pct < config.VOL_REGIME_LOW_THRESHOLD:
            return "LOW"
        elif atr_pct > config.VOL_REGIME_HIGH_THRESHOLD:
            return "HIGH"
        return "NEUTRAL"

    # =====================================================================
    # Weighted Scoring
    # =====================================================================

    def _compute_weighted_score(self, imbalance: float, wall_strength: float, z_score: float, touch: bool, trend_align: bool) -> float:
        """Compute normalized weighted score for entry."""
        if not config.ENABLE_WEIGHTED_SCORING:
            # Binary mode: all or nothing
            return 1.0 if all([imbalance is not None, wall_strength is not None, z_score is not None, touch, trend_align]) else 0.0

        score = 0.0
        total_weight = 0.0

        # Imbalance contribution (normalized 0-1)
        if imbalance is not None:
            norm_imb = max(0.0, min(1.0, (imbalance + 1.0) / 2.0))  # -1 to 1 -> 0 to 1
            score += norm_imb * config.WEIGHT_IMBALANCE
            total_weight += config.WEIGHT_IMBALANCE

        # Wall strength (threshold-normalized)
        if wall_strength is not None:
            thresh = self._get_dynamic_wall_mult(self._get_vol_regime(None))
            norm_wall = max(0.0, min(1.0, (wall_strength - thresh) / (10.0 - thresh)))  # Assume max 10x
            score += norm_wall * config.WEIGHT_WALL
            total_weight += config.WEIGHT_WALL

        # Z-score (threshold-normalized)
        if z_score is not None:
            thresh = self._get_dynamic_z_threshold(self._get_vol_regime(None), None)
            norm_z = max(0.0, min(1.0, (z_score - thresh) / (5.0 - thresh)))  # Assume max Z=5
            score += norm_z * config.WEIGHT_ZSCORE
            total_weight += config.WEIGHT_ZSCORE

        # Touch (binary)
        score += float(touch) * config.WEIGHT_TOUCH
        total_weight += config.WEIGHT_TOUCH

        # Trend align (binary)
        score += float(trend_align) * config.WEIGHT_TREND
        total_weight += config.WEIGHT_TREND

        return score / total_weight if total_weight > 0 else 0.0

    # =====================================================================
    # Oracle Integration
    # =====================================================================

    def _integrate_oracle(self, data_manager, side: str) -> Optional[OracleSideScores]:
        """Fuse Oracle for additional probability score."""
        try:
            current_price = data_manager.get_last_price()
            now_sec = time.time()

            # Build inputs from data_manager (exact from oracle spec)
            inputs = OracleInputs(
                imbalance_data={"value": self._compute_imbalance(data_manager)},
                wall_data={"strength": self._compute_wall_strength(data_manager, current_price, side)},
                delta_data={"zscore": self._compute_delta_z_score(data_manager)},
                touch_data={"touched": self._check_price_touch(data_manager, current_price)},  # Simplified wall_price
                htf_trend=data_manager.get_htf_trend(),
                ltf_trend=data_manager.get_ltf_trend(),
                ema_val=data_manager.get_ema(config.EMA_PERIOD),
                atr_pct=data_manager.get_atr_percent(config.ATR_WINDOW_MINUTES),
                lv_1m=data_manager.compute_liquidity_velocity_multi_tf()[0],
                lv_5m=data_manager.compute_liquidity_velocity_multi_tf()[1],
                lv_15m=data_manager.compute_liquidity_velocity_multi_tf()[2],
                micro_trap=data_manager.compute_liquidity_velocity_multi_tf()[3],
                norm_cvd=data_manager.compute_norm_cvd(),
                hurst=data_manager.compute_hurst_exponent(),
                bos_align=data_manager.compute_bos_alignment(current_price),
                current_price=current_price,
                now_sec=now_sec,
            )

            # Dummy risk_manager for oracle (oracle uses it for Kelly)
            class DummyRiskManager:
                def get_available_balance(self):
                    return {"available": 10000.0}

            rm = DummyRiskManager()

            outputs = self._oracle.decide(inputs, rm)
            if outputs and hasattr(outputs, side + '_scores'):
                return getattr(outputs, side + '_scores')
            return None
        except Exception as e:
            logger.error(f"Oracle integration error for {side}: {e}")
            return None

    # =====================================================================
    # Full Gate Check
    # =====================================================================

    def _check_all_gates(self, data_manager, side: str) -> Tuple[bool, float, str]:
        """Check all gates: binary AND + weighted score + oracle fusion."""
        current_price = data_manager.get_last_price()
        if current_price <= 0:
            return False, 0.0, "NO_PRICE"

        vol_regime = self._get_vol_regime(data_manager)
        atr_pct = data_manager.get_atr_percent(config.ATR_WINDOW_MINUTES)
        z_thresh = self._get_dynamic_z_threshold(vol_regime, atr_pct)
        wall_mult = self._get_dynamic_wall_mult(vol_regime)

        # 1. Imbalance gate
        imbalance = self._compute_imbalance(data_manager)
        imb_pass = imbalance is not None and (
            (side == "long" and imbalance >= config.IMBALANCE_THRESHOLD) or
            (side == "short" and imbalance <= -config.IMBALANCE_THRESHOLD)
        )
        if not imb_pass:
            return False, 0.0, "IMBALANCE_FAIL"

        # 2. Wall strength gate
        wall_strength = self._compute_wall_strength(data_manager, current_price, side)
        wall_pass = wall_strength is not None and wall_strength >= wall_mult
        if not wall_pass:
            return False, 0.0, "WALL_FAIL"

        # 3. Delta Z-score gate
        z_score = self._compute_delta_z_score(data_manager)
        z_pass = z_score is not None and z_score >= z_thresh
        if not z_pass:
            return False, 0.0, "ZSCORE_FAIL"

        # 4. Price touch gate (find strongest wall in zone)
        tick_size = config.TICK_SIZE
        zone_low = current_price - config.ZONE_TICKS * tick_size
        zone_high = current_price + config.ZONE_TICKS * tick_size
        wall_price = None
        with data_manager._orderbook_lock:
            if side == "long":
                relevant = [(p, q) for p, q in data_manager._orderbook_bids if zone_low <= p <= zone_high]
                if relevant:
                    wall_price = max(relevant, key=lambda x: x[1])[0]  # Strongest bid
            else:
                relevant = [(p, q) for p, q in data_manager._orderbook_asks if zone_low <= p <= zone_high]
                if relevant:
                    wall_price = min(relevant, key=lambda x: x[1])[0]  # Strongest ask

        touch = self._check_price_touch(data_manager, wall_price) if wall_price else False
        if not touch:
            return False, 0.0, "TOUCH_FAIL"

        # 5. Trend alignment gate
        htf_trend = data_manager.get_htf_trend() or "RANGE"
        ltf_trend = data_manager.get_ltf_trend() or "RANGE"
        trend_align = (
            (side == "long" and htf_trend != "DOWN" and ltf_trend != "DOWN") or
            (side == "short" and htf_trend != "UP" and ltf_trend != "UP")
        )
        if not trend_align:
            return False, 0.0, "TREND_FAIL"

        # 6. Weighted score
        weighted_score = self._compute_weighted_score(imbalance, wall_strength, z_score, touch, trend_align)
        if weighted_score < config.WEIGHTED_SCORE_ENTRY_THRESHOLD:
            return False, weighted_score, "SCORE_LOW"

        # 7. Oracle fusion (optional, boosts score)
        oracle_scores = self._integrate_oracle(data_manager, side)
        if oracle_scores and oracle_scores.fused is not None and oracle_scores.fused < self._oracle.entry_prob_threshold:
            return False, weighted_score, "ORACLE_LOW"

        return True, weighted_score, vol_regime

    # =====================================================================
    # Main Tick Handler
    # =====================================================================

    def on_tick(self, data_manager, order_manager, risk_manager):
        """Process tick: manage position or evaluate entry."""
        now_sec = time.time()

        # Cooldown after exit
        if (now_sec / 60.0) - self.last_exit_time_min < config.MIN_TIME_BETWEEN_TRADES:
            return

        if self.current_position:
            self._manage_position(data_manager, order_manager, risk_manager, now_sec)
            return

        # Evaluate entry for both sides, pick best
        long_pass, long_score, long_regime = self._check_all_gates(data_manager, "long")
        short_pass, short_score, short_regime = self._check_all_gates(data_manager, "short")

        side = None
        score = 0.0
        regime = "NEUTRAL"
        if long_pass and (not short_pass or long_score >= short_score):
            side = "long"
            score = long_score
            regime = long_regime
        elif short_pass:
            side = "short"
            score = short_score
            regime = short_regime

        if side:
            self._enter_position(data_manager, order_manager, risk_manager, side, score, regime)

        # Periodic decision logging
        if now_sec - self._last_decision_log_sec > self.DECISION_LOG_INTERVAL_SEC:
            self._log_decision(data_manager, long_pass, long_score, short_pass, short_score)
            self._last_decision_log_sec = now_sec

    # =====================================================================
    # Entry Flow
    # =====================================================================

    def _enter_position(self, data_manager, order_manager, risk_manager, side: str, score: float, regime: str):
        """Enter with single bracket order (atomic)."""
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                logger.warning("Cannot enter: invalid price")
                return

            # ✅ FIXED: Fetch balance ONCE before trade
            balance_info = risk_manager.get_available_balance()
            if not balance_info:
                logger.warning("Cannot enter: no balance info")
                return

            available = float(balance_info.get("available", 0.0))
            if available < config.MIN_MARGIN_PER_TRADE:
                logger.warning(f"Cannot enter: low balance {available:.2f}")
                return

            # Position sizing (regime-aware)
            margin_used, quantity = risk_manager.calculate_position_size_regime_aware(current_price, regime)
            if quantity <= 0 or margin_used <= 0:
                logger.warning("Cannot enter: invalid sizing")
                return

            # ✅ FIXED: Excel-exact TP/SL calculation
            tp_price, sl_price = self._calculate_tp_sl_prices(side, current_price, margin_used, quantity, regime)

            # Place bracket (limit entry + TP + SL)
            bracket_result = order_manager.place_bracket_order(
                side=side.upper() if side == "buy" else "SELL" if side == "short" else "BUY",
                quantity=quantity,
                entry_price=current_price,  # Limit at current (or adjust for direction)
                tp_price=tp_price,
                sl_price=sl_price,
            )

            if not bracket_result:
                logger.warning("Bracket order failed")
                return

            main_order, tp_order, sl_order = bracket_result
            main_id = main_order.get("order_id")
            tp_id = tp_order.get("order_id")
            sl_id = sl_order.get("order_id")

            # Wait for entry fill
            try:
                fill_data = order_manager.wait_for_fill(main_id, timeout_sec=self.ENTRY_FILL_TIMEOUT_SEC)
                actual_entry_price = order_manager.extract_fill_price(fill_data)
                actual_quantity = float(fill_data.get("exec_quantity", quantity))
            except RuntimeError as e:
                logger.error(f"Entry fill timeout/cancel: {e}")
                # Cancel bracket
                order_manager.cancel_order(tp_id)
                order_manager.cancel_order(sl_id)
                return

            # Create position object (populate from gates)
            imbalance = self._compute_imbalance(data_manager)
            z_score = self._compute_delta_z_score(data_manager)
            wall_strength = self._compute_wall_strength(data_manager, actual_entry_price, side)
            wall_vol = wall_strength * (sum(q for _, q in data_manager._orderbook_bids[:5]) + sum(q for _, q in data_manager._orderbook_asks[:5])) / 10 if wall_strength else 0.0
            htf_trend = data_manager.get_htf_trend() or "RANGE"

            self.trade_seq += 1
            trade_id = f"Z{self.trade_seq:04d}_{int(time.time())}"

            self.current_position = ZScorePosition(
                trade_id=trade_id,
                side=side,
                quantity=actual_quantity,
                entry_price=actual_entry_price,
                entry_time_sec=time.time(),
                entry_wall_volume=wall_vol,
                wall_zone_low=actual_entry_price - config.ZONE_TICKS * config.TICK_SIZE,
                wall_zone_high=actual_entry_price + config.ZONE_TICKS * config.TICK_SIZE,
                entry_imbalance=imbalance or 0.0,
                entry_z_score=z_score or 0.0,
                tp_price=tp_price,
                sl_price=sl_price,
                margin_used=margin_used,
                tp_order_id=tp_id,
                sl_order_id=sl_id,
                main_order_id=main_id,
                entry_htf_trend=htf_trend,
                entry_vol_regime=regime,
                entry_weighted_score=score,
                last_score_check_sec=time.time(),
            )

            risk_manager.record_trade_opened()
            logger.info(f"✓ LONG/SHORT ENTRY: {side.upper()} {actual_quantity:.6f} @ {actual_entry_price:.2f} | Margin: {margin_used:.2f} | Regime: {regime} | Score: {score:.3f}")

            if self.excel_logger:
                self.excel_logger.log_entry(self.current_position)

            # Telegram notify
            send_telegram_message(
                f"🚀 NEW TRADE #{self.trade_seq}\n"
                f"Side: {side.upper()}\n"
                f"Qty: {actual_quantity:.6f} BTC\n"
                f"Entry: ${actual_entry_price:.2f}\n"
                f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f}\n"
                f"Regime: {regime} | Score: {score:.3f}"
            )

        except Exception as e:
            logger.error(f"Entry error: {e}", exc_info=True)
            # Emergency cancel
            try:
                order_manager.cancel_all_orders()
            except:
                pass

    def _calculate_tp_sl_prices(self, side: str, entry_price: float, margin_used: float, quantity: float, vol_regime: str) -> Tuple[float, float]:
        """✅ FIXED: EXACT Excel implementation."""
        # Regime adjustments
        tp_mult, sl_mult = self._get_dynamic_tp_sl_multipliers(vol_regime)
        desired_tp_roi = config.PROFIT_TARGET_ROI * tp_mult  # e.g., 0.05
        desired_sl_roi = abs(config.STOP_LOSS_ROI) * sl_mult  # e.g., 0.01

        leverage = config.LEVERAGE

        # Excel: Increase/Decrease in Balance = Margin * Leverage * ROI %
        increase_balance_tp = margin_used * leverage * desired_tp_roi
        decrease_balance_sl = margin_used * leverage * desired_sl_roi

        # Price movement = Balance change / Qty (with leverage)
        price_movement_tp = increase_balance_tp / quantity
        price_movement_sl = decrease_balance_sl / quantity

        # Apply direction
        if side == "long":
            tp_price = entry_price + price_movement_tp
            sl_price = entry_price - price_movement_sl
        else:  # short
            tp_price = entry_price - price_movement_tp
            sl_price = entry_price + price_movement_sl

        # Round to tick size (Excel-like precision)
        tick_size = config.TICK_SIZE
        tp_price = round(tp_price / tick_size) * tick_size
        sl_price = round(sl_price / tick_size) * tick_size

        logger.debug(
            f"TP/SL Calc: Side={side}, Entry={entry_price:.2f}, Margin={margin_used:.2f}, "
            f"Qty={quantity:.6f}, TP_Mov={price_movement_tp:.2f}, SL_Mov={price_movement_sl:.2f}, "
            f"TP={tp_price:.2f}, SL={sl_price:.2f}"
        )

        return tp_price, sl_price

    # =====================================================================
    # Position Management State Machine
    # =====================================================================

    def _manage_position(self, data_manager, order_manager, risk_manager, now_sec: float):
        """Manage open position: check orders, time stop, log."""
        pos = self.current_position
        if not pos:
            return

        # Time stop
        age_min = (now_sec - pos.entry_time_sec) / 60.0
        if age_min > config.MAX_HOLD_MINUTES:
            self._exit_position(data_manager, order_manager, risk_manager, "TIME_STOP", now_sec)
            return

        # Periodic order status checks
        if now_sec - self._last_status_check_sec > self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            # Check TP
            tp_status = order_manager.get_order_status(pos.tp_order_id)
            if tp_status and float(tp_status.get("exec_quantity", 0)) > 0:
                self._exit_position(data_manager, order_manager, risk_manager, "TP_HIT", now_sec)
                return

            # Check SL
            sl_status = order_manager.get_order_status(pos.sl_order_id)
            if sl_status and float(sl_status.get("exec_quantity", 0)) > 0:
                self._exit_position(data_manager, order_manager, risk_manager, "SL_HIT", now_sec)
                return

            self._last_status_check_sec = now_sec

        # Periodic unrealized P&L log
        if now_sec - self._last_position_log_sec > self.POSITION_LOG_INTERVAL_SEC:
            current_price = data_manager.get_last_price()
            unreal_pnl = self._calculate_unrealized_pnl(pos, current_price)
            age_min = (now_sec - pos.entry_time_sec) / 60.0
            logger.info(
                f"POS UPDATE: {pos.side.upper()} {pos.quantity:.6f} @ {pos.entry_price:.2f} "
                f"(Age: {age_min:.1f}min | Unreal: ${unreal_pnl:.2f} | Regime: {pos.entry_vol_regime})"
            )
            self._last_position_log_sec = now_sec

    # =====================================================================
    # Exit Flow with Realized P&L
    # =====================================================================

    def _exit_position(self, data_manager, order_manager, risk_manager, reason: str, now_sec: float):
        """Exit position: determine price, calc P&L, cleanup."""
        pos = self.current_position
        if not pos:
            return

        try:
            # Determine exit price
            if reason == "TP_HIT":
                exit_price = pos.tp_price
            elif reason == "SL_HIT":
                exit_price = pos.sl_price
            else:  # TIME_STOP
                # Market close
                close_side = "SELL" if pos.side == "long" else "BUY"
                close_result = order_manager.place_market_order(
                    side=close_side,
                    quantity=pos.quantity,
                )
                if close_result:
                    close_id = close_result.get("order_id")
                    fill_data = order_manager.wait_for_fill(close_id, timeout_sec=30.0)
                    exit_price = order_manager.extract_fill_price(fill_data)
                else:
                    exit_price = data_manager.get_last_price()
                    if exit_price <= 0:
                        logger.error("Emergency exit failed: no price")
                        return

            # ✅ FIXED: Complete realized P&L calculation
            entry_notional = pos.quantity * pos.entry_price
            exit_notional = pos.quantity * exit_price
            gross_pnl = (
                (exit_notional - entry_notional) / entry_notional
                if pos.side == "long"
                else (entry_notional - exit_notional) / entry_notional
            )
            leverage_pnl = gross_pnl * config.LEVERAGE * pos.margin_used  # Margin-based

            # Fees (taker on entry/exit)
            entry_fee = entry_notional * config.TAKER_FEE_RATE
            exit_fee = exit_notional * config.TAKER_FEE_RATE
            net_pnl = leverage_pnl - entry_fee - exit_fee

            # Update stats
            risk_manager.update_trade_stats(net_pnl)
            self.total_trades += 1

            # Cancel remaining bracket orders
            order_manager.cancel_order(pos.tp_order_id)
            order_manager.cancel_order(pos.sl_order_id)

            # Logging & notify
            age_min = (now_sec - pos.entry_time_sec) / 60.0
            logger.info(
                f"EXIT {reason}: {pos.side.upper()} {pos.quantity:.6f} "
                f"{pos.entry_price:.2f} → {exit_price:.2f} "
                f"(Age: {age_min:.1f}min | P&L: ${net_pnl:.2f} | RR: {abs(net_pnl / (pos.margin_used * abs(config.STOP_LOSS_ROI))):.2f})"
            )

            send_telegram_message(
                f"🔔 TRADE #{pos.trade_id} CLOSED\n"
                f"Reason: {reason}\n"
                f"Exit: ${exit_price:.2f}\n"
                f"Duration: {age_min:.1f}min\n"
                f"P&L: ${net_pnl:.2f}\n"
                f"Total Trades: {self.total_trades}"
            )

            # Excel log
            if self.excel_logger:
                self.excel_logger.log_exit(pos, exit_price, net_pnl, reason)

            # Cleanup
            self.current_position = None
            self.last_exit_time_min = now_sec / 60.0
            risk_manager.record_trade_closed()

        except Exception as e:
            logger.error(f"Exit error: {e}", exc_info=True)
            # Force cancel all
            order_manager.cancel_all_orders()
            self.current_position = None

    def _calculate_unrealized_pnl(self, pos: ZScorePosition, current_price: float) -> float:
        """Unrealized P&L for logging."""
        if current_price <= 0:
            return 0.0

        entry_notional = pos.quantity * pos.entry_price
        current_notional = pos.quantity * current_price
        gross_unreal = (
            (current_notional - entry_notional) / entry_notional
            if pos.side == "long"
            else (entry_notional - current_notional) / entry_notional
        )
        leverage_unreal = gross_unreal * config.LEVERAGE * pos.margin_used
        return leverage_unreal  # Fees not deducted for unreal

    def _log_decision(self, data_manager, long_pass: bool, long_score: float, short_pass: bool, short_score: float):
        """Log tick decision snapshot."""
        current_price = data_manager.get_last_price()
        vol_regime = self._get_vol_regime(data_manager)
        atr_pct = data_manager.get_atr_percent(config.ATR_WINDOW_MINUTES)
        imbalance = self._compute_imbalance(data_manager)
        logger.info(
            f"TICK DECISION: Price={current_price:.2f} | Regime={vol_regime} (ATR%={atr_pct*100:.2f}) "
            f"| Imb={imbalance:.3f} | Long={long_pass}({long_score:.3f}) | Short={short_pass}({short_score:.3f})"
        )

    # =====================================================================
    # Event Callbacks (for WebSocket, no-op or extend as needed)
    # =====================================================================

    def on_orderbook_update(self, data: Dict):
        """Callback for orderbook update - no-op, handled in data_manager."""
        pass

    def on_trade_update(self, data: Dict):
        """Callback for trade update - no-op, handled in data_manager."""
        pass

    def on_candle_update(self, data: Dict):
        """Callback for candle update - no-op, handled in data_manager."""
        pass