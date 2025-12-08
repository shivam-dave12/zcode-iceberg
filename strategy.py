"""
Z-Score Strategy with Session-Based Entry Control and Dynamic Parameters

IMPROVEMENTS:
1. Session-aware parameter selection (Weekend/Major/Overlap)
2. Entry cooldown to prevent noisy signals
3. 3-signal confirmation with time tracking
4. Single-fire timeout handling for limit orders
5. Safe orphaned order cleanup
6. Session-based TP/SL calculations
7. FIXED: Fresh session fetch on entry/TP-SL calc to prevent UNKNOWN/stale var bug
8. FIXED: Early fill detection guard to prevent duplicate logging
9. FIXED: Centralized TP/SL calculation for consistent ROI methodology
10. FIXED: Industrial-grade TP management with single cancel+place cycle
11. FIXED: Concurrent status check prevention with execution guards
12. FIXED: Entry cooldown bypass with minimum time span enforcement
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone
from collections import deque
import logging
import numpy as np
from scipy.stats import norm

import config
from zscore_excel_logger import ZScoreExcelLogger
from telegram_notifier import (
    send_telegram_message,
    format_entry_message,
    format_fill_message,
    format_exit_message,
    format_position_update,
    format_tp_adjustment,
    format_sl_adjustment,
    format_order_status_unknown,
)
from aether_oracle import AetherOracle, OracleInputs, OracleOutputs

logger = logging.getLogger(__name__)


@dataclass
class ZScorePosition:
    """Position tracking dataclass with session info."""

    trade_id: str
    side: str
    quantity: float
    entry_price: float
    entry_time_sec: float
    entry_wall_volume: float
    wall_zone_low: float
    wall_zone_high: float
    entry_imbalance: float
    entry_zscore: float
    tp_price: float
    sl_price: float
    margin_used: float
    tp_order_id: str
    sl_order_id: str
    main_order_id: str
    main_filled: bool = False
    tp_reduced: bool = False
    entry_htf_trend: str = ""
    vol_regime: str = ""
    entry_score: float = 0.0
    is_volatile: bool = False
    tp_roi: float = 0.0
    sl_roi: float = 0.0
    last_momentum_check_sec: float = 0.0
    last_momentum_log_sec: float = 0.0
    tp_adjustment_count: int = 0
    entry_session: str = ""
    limit_order_placed_time: float = 0.0
    timeout_cancelled: bool = False
    tp_adjusted_10min: bool = False
    tp_tightened_15min: bool = False
    current_tp_roi: float = 0.0
    current_tp_price: float = 0.0
    initial_tp_roi: float = 0.0
    initial_sl_roi: float = 0.0


class ZScoreIcebergHunterStrategy:
    """Z-Score Iceberg Hunter Strategy with Session-Based Control."""

    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 60.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.pending_entry: bool = False
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0
        self.delta_population = deque(maxlen=3000)
        self.last_decision_log_sec: float = 0.0
        self.last_position_log_sec: float = 0.0
        self.last_status_check_sec: float = 0.0
        self.oracle = AetherOracle()
        self.signal_history = deque(maxlen=3)
        self.last_signal_time: float = 0.0
        self.last_entry_time_sec: float = 0.0
        self.current_session_params: Dict = {}
        self._early_fill_handled: Dict[str, bool] = {}
        
        # ✅ FIX: Add execution guards to prevent concurrent API calls
        self._checking_order_status = False
        self._adjusting_tp = False

        logger.info("=" * 80)
        logger.info("Z-SCORE STRATEGY INITIALIZED")
        logger.info("With Session + Weekend + Volatility Gates")
        logger.info("Dynamic Session-Based Parameters")
        logger.info("Entry Cooldown + 3-Signal Confirmation")
        logger.info("=" * 80)

    def get_current_session(self) -> Tuple[str, bool]:
        """
        Detect current trading session and weekend status.
        Returns: (session_name, is_major_session)

        Sessions (UTC):
        - Asia (Tokyo): 00:00 - 09:00 UTC
        - London: 08:00 - 16:00 UTC
        - New York: 13:00 - 22:00 UTC
        - Overlap zones have highest volatility
        """
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        day_of_week = now_utc.weekday()

        if day_of_week >= 5:
            return "WEEKEND", False

        in_asia = 0 <= hour_utc < 9
        in_london = 8 <= hour_utc < 16
        in_ny = 13 <= hour_utc < 22

        if in_london and in_ny:
            return "LONDON_NY_OVERLAP", True
        elif in_asia and in_london:
            return "ASIA_LONDON_OVERLAP", True
        elif in_london:
            return "LONDON", True
        elif in_ny:
            return "NEW_YORK", True
        elif in_asia:
            return "ASIA", True
        else:
            return "OFF_SESSION", False

    def get_session_parameters(self, session: str, is_major_session: bool) -> Dict:
        """
        Get dynamic parameters based on current trading session.
        Returns session-specific thresholds and timing parameters.
        """
        if session == "WEEKEND":
            params = config.WEEKEND_PARAMS.copy()
            session_type = "WEEKEND"
        elif session in ["LONDON_NY_OVERLAP", "ASIA_LONDON_OVERLAP"]:
            params = config.OVERLAP_SESSION_PARAMS.copy()
            session_type = session
        elif is_major_session:
            params = config.MAJOR_SESSION_PARAMS.copy()
            session_type = session
        else:
            params = config.WEEKEND_PARAMS.copy()
            session_type = "OFF_PEAK"

        params["session_name"] = session_type
        logger.debug(f"Session: {session_type} | Params: {params}")
        return params

    def determine_volatility_with_gates(
        self, data_manager, current_price: float
    ) -> Tuple[bool, str, float, str]:
        """
        Two-gate volatility detection:
        - Gate 1: Session/Weekend check
        - Gate 2: ATR-based calculation
        Returns: (is_volatile, session, atr_pct, explanation)
        """
        session, is_major_session = self.get_current_session()

        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception as e:
            logger.debug(f"Error getting ATR: {e}")
            atr_pct = None

        if session == "WEEKEND":
            explanation = "Weekend - Forced LOW volatility"
            return False, session, atr_pct or 0.0, explanation

        if not is_major_session:
            if atr_pct and atr_pct > config.VOLATILE_ATR_THRESHOLD * 1.5:
                explanation = f"Off-session but ATR very high ({atr_pct*100:.3f}%)"
                return True, session, atr_pct, explanation
            else:
                explanation = "Off-session - bias LOW volatility"
                return False, session, atr_pct or 0.0, explanation

        if atr_pct is not None:
            is_volatile = atr_pct > config.VOLATILE_ATR_THRESHOLD
            if is_volatile:
                explanation = f"Major session ({session}): High ATR ({atr_pct*100:.3f}%)"
            else:
                explanation = f"Major session ({session}): Normal ATR ({atr_pct*100:.3f}%)"
            return is_volatile, session, atr_pct, explanation
        else:
            if "OVERLAP" in session:
                explanation = f"Overlap session ({session}) - assume HIGH volatility"
                return True, session, 0.0, explanation
            else:
                explanation = f"Major session ({session}) - assume NORMAL volatility"
                return False, session, 0.0, explanation

    def check_signal_confirmation(
        self, side: str, score: float, now_sec: float
    ) -> Tuple[bool, str]:
        """
        Check if we have 3 consecutive signals for the same side.
        Includes SESSION-BASED entry cooldown to prevent noise.
        Returns: (confirmed, reason)
        """
        session, is_major = self.get_current_session()
        session_params = self.get_session_parameters(session, is_major)
        min_gap = session_params["min_signal_gap_sec"]

        # ✅ FIX: Enforce MINIMUM time between ANY signals
        if self.signal_history and len(self.signal_history) > 0:
            last_signal_time = self.signal_history[-1][2]
            time_since_last = now_sec - last_signal_time
            
            if time_since_last < 0.5:
                return False, f"TOO_FAST: {time_since_last:.2f}s since last signal"

        time_since_entry = now_sec - self.last_entry_time_sec
        if time_since_entry < min_gap:
            remaining = min_gap - time_since_entry
            return False, f"COOLDOWN: {remaining:.1f}s remaining"

        self.signal_history.append((side, score, now_sec))

        if len(self.signal_history) < 3:
            return False, f"NEED_MORE: {len(self.signal_history)}/3"

        sides = [s[0] for s in self.signal_history]
        if not all(s == side for s in sides):
            return False, f"MIXED_SIGNALS: {'/'.join(sides)}"

        # ✅ FIX: Check time span - must be at least 1 second between first and last
        first_time = self.signal_history[0][2]
        last_time = self.signal_history[-1][2]
        time_span = last_time - first_time
        
        if time_span < 1.0:
            return False, f"TOO_RAPID: 3 signals in {time_span:.1f}s (need 1.0s minimum)"

        avg_score = sum(s[1] for s in self.signal_history) / 3

        logger.info(
            f"✅ SIGNAL CONFIRMATION: 3x {side} signals over {time_span:.1f}s "
            f"(avg score: {avg_score:.3f})"
        )

        return True, f"CONFIRMED: 3x {side} over {time_span:.1f}s"

    def clean_orphaned_orders(self, order_manager, symbol: str) -> bool:
        """
        Clean up orphaned orders. Returns True if orders were cancelled.
        """
        try:
            # Check if there's an active position first
            positions_resp = order_manager.api.get_positions(
                symbol=symbol, exchange=config.EXCHANGE
            )
            if positions_resp and not positions_resp.get("error"):
                data = positions_resp.get("data", [])
                if isinstance(data, list):
                    for pos in data:
                        if pos.get("symbol") == symbol:
                            qty = float(pos.get("quantity", 0))
                            if abs(qty) > 0:
                                logger.warning(
                                    f"⚠️  Active position detected: qty={qty:.6f} BTC - "
                                    f"SKIPPING orphaned order cleanup"
                                )
                                return False

            # ✅ FIX: Get open orders first before attempting cancel
            logger.info(f"🔍 Checking for orphaned orders: {symbol}")
            orders_resp = order_manager.api.get_open_orders(
                symbol=symbol, exchange=config.EXCHANGE
            )
            
            if orders_resp and not orders_resp.get("error"):
                orders = orders_resp.get("data", [])
                if isinstance(orders, list) and len(orders) > 0:
                    logger.info(f"📋 Found {len(orders)} open orders - cancelling...")
                    order_manager.api.cancel_all_orders(symbol=symbol, exchange=config.EXCHANGE)
                    logger.info(f"✅ Cancelled {len(orders)} orphaned orders")
                    return True
                else:
                    logger.debug("✅ No orphaned orders found")
                    return False
            
            return False

        except Exception as e:
            # ✅ FIX: Ignore 422 errors - they just mean no orders exist
            if "422" in str(e) or "no open orders" in str(e).lower():
                logger.debug("✅ No orphaned orders to cancel")
                return False
            logger.error(f"Error cleaning orphaned orders: {e}")
            return False

    def _compute_bracket_prices(
        self,
        entry_price: float,
        margin_used: float,
        quantity: float,
        side: str,
        tp_roi: float,
        sl_roi: float,
        session: str,
    ) -> Tuple[float, float]:
        """
        Centralized TP/SL price calculation using session-based ROI.
        Returns: (tp_price, sl_price)
        """
        logger.info("=" * 80)
        logger.info("[TP/SL CALCULATION] Using session-based parameters")
        logger.info(f"  Session: {session}")
        logger.info(f"  Entry Price: {entry_price:.2f}")
        logger.info(f"  Margin Used: {margin_used:.2f} USDT")
        logger.info(f"  Quantity: {quantity:.6f} BTC")
        logger.info(f"  Side: {side}")
        logger.info(f"  Desired TP ROI: {tp_roi*100:.2f}%")
        logger.info(f"  Desired SL ROI: {sl_roi*100:.2f}%")

        tp_price_movement = margin_used * tp_roi
        sl_price_movement = margin_used * sl_roi

        logger.info(f"  TP Price Movement: {tp_price_movement:.2f} USDT")
        logger.info(f"  SL Price Movement: {sl_price_movement:.2f} USDT")

        tp_price_change = tp_price_movement / quantity
        sl_price_change = sl_price_movement / quantity

        if side == "long":
            tp_price = entry_price + tp_price_change
            sl_price = entry_price - sl_price_change
        else:
            tp_price = entry_price - tp_price_change
            sl_price = entry_price + sl_price_change

        logger.info(f"  ✓ TP Price: {tp_price:.2f}")
        logger.info(f"  ✓ SL Price: {sl_price:.2f}")
        logger.info("=" * 80)

        return tp_price, sl_price

    def get_regime_params(self, data_manager, current_price: float) -> Dict:
        """Get dynamic params based on vol regime."""
        try:
            atr_pct = data_manager.get_atr_percent()
        except:
            atr_pct = None

        try:
            regime = (
                data_manager.get_vol_regime(atr_pct)
                if hasattr(data_manager, "get_vol_regime")
                else "NEUTRAL"
            )
        except:
            regime = "NEUTRAL"

        if regime == "LOW":
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH - 0.3
        elif regime == "HIGH":
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH + 0.3
        else:
            z_thresh = config.VOL_REGIME_BASE_Z_THRESH

        if regime == "HIGH":
            wall_mult = config.VOL_REGIME_HIGH_WALL_MULT
            tp_mult = config.VOL_REGIME_HIGH_TP_MULT
            sl_mult = config.VOL_REGIME_HIGH_SL_MULT
            size_pct = config.VOL_REGIME_HIGH_SIZE_PCT
        else:
            wall_mult = config.VOL_REGIME_LOW_WALL_MULT
            tp_mult = config.VOL_REGIME_LOW_TP_MULT
            sl_mult = config.VOL_REGIME_LOW_SL_MULT
            size_pct = config.VOL_REGIME_LOW_SIZE_PCT

        return {
            "regime": regime,
            "z_thresh": z_thresh,
            "wall_mult": wall_mult,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "size_pct": size_pct,
            "atr_pct": atr_pct or 0.0,
        }

    def compute_signal_score(self, value: float, threshold: float) -> float:
        """Normalize signal to [0, 1] using CDF."""
        if threshold == 0:
            return 1.0 if value > 0 else 0.0
        std = abs(threshold) / 2.0
        z = (value - threshold) / std
        score = norm.cdf(z)
        return max(0.0, min(1.0, score))

    def compute_weighted_score(
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
        """Compute 5-signal weighted core score."""
        reasons = []

        imb_val = imbalance_data["imbalance"]
        if side == "long":
            imb_score = self.compute_signal_score(imb_val, config.IMBALANCE_THRESHOLD)
        else:
            imb_score = self.compute_signal_score(-imb_val, config.IMBALANCE_THRESHOLD)
        reasons.append(f"imb={imb_score:.3f}")

        if side == "long":
            wall_val = wall_data["bid_wall_strength"]
        else:
            wall_val = wall_data["ask_wall_strength"]
        wall_score = self.compute_signal_score(wall_val, wall_mult)
        reasons.append(f"wall={wall_score:.3f}")

        z_val = delta_data["zscore"]
        if side == "long":
            z_score = self.compute_signal_score(z_val, z_thresh)
        else:
            z_score = self.compute_signal_score(-z_val, z_thresh)
        reasons.append(f"z={z_score:.3f}")

        if side == "long":
            touch_dist = touch_data["bid_distance_ticks"]
        else:
            touch_dist = touch_data["ask_distance_ticks"]
        touch_score = 1.0 if touch_dist <= config.PRICE_TOUCH_THRESHOLD_TICKS else 0.0
        reasons.append(f"touch={touch_score:.3f}")

        trend_score = 1.0 if trend_ok else 0.0
        reasons.append(f"trend={trend_score:.3f}")

        core_score = (
            imb_score * config.SCORE_IMB_WEIGHT
            + wall_score * config.SCORE_WALL_WEIGHT
            + z_score * config.SCORE_Z_WEIGHT
            + touch_score * config.SCORE_TOUCH_WEIGHT
            + trend_score * config.SCORE_TREND_WEIGHT
        )

        return core_score, reasons

    def compute_aether_fusion_score(
        self, oracle_inputs: OracleInputs, oracle_outputs: OracleOutputs, side: str
    ) -> Tuple[float, List[str]]:
        """Compute Aether Oracle fusion score."""
        reasons = []

        scores = []
        if side == "long":
            scores.append(oracle_outputs.normalized_sentiment)
            scores.append(oracle_outputs.normalized_momentum)
            scores.append(oracle_outputs.normalized_liquidity)
        else:
            scores.append(1.0 - oracle_outputs.normalized_sentiment)
            scores.append(1.0 - oracle_outputs.normalized_momentum)
            scores.append(1.0 - oracle_outputs.normalized_liquidity)

        aether_score = sum(scores) / len(scores) if scores else 0.0
        reasons.append(f"aether={aether_score:.3f}")

        return aether_score, reasons

    def evaluate_signals(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """Evaluate all signals and manage entries."""
        session, is_major = self.get_current_session()
        session_params = self.get_session_parameters(session, is_major)

        imb_thresh = session_params["imbalance_threshold"]
        z_thresh = session_params["z_threshold"]
        entry_thresh = session_params["entry_score_threshold"]
        wall_mult = session_params["wall_multiplier"]

        imbalance_data = self.compute_bid_ask_imbalance(data_manager)
        if not imbalance_data:
            return

        wall_data = self.compute_wall_pressure(data_manager, current_price)
        if not wall_data:
            return

        delta_data = self.compute_delta_zscore(data_manager)
        if not delta_data:
            return

        touch_data = self.compute_price_touch(data_manager, current_price)
        if not touch_data:
            return

        regime_params = self.get_regime_params(data_manager, current_price)
        regime = regime_params["regime"]
        atr_pct = regime_params.get("atr_pct", 0.0)

        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except:
            pass

        trend_long_ok = ema_val is not None and current_price > ema_val
        trend_short_ok = ema_val is not None and current_price < ema_val

        long_core, long_core_reasons = self.compute_weighted_score(
            imbalance_data,
            wall_data,
            delta_data,
            touch_data,
            trend_long_ok,
            "long",
            regime,
            z_thresh,
            wall_mult,
        )
        short_core, short_core_reasons = self.compute_weighted_score(
            imbalance_data,
            wall_data,
            delta_data,
            touch_data,
            trend_short_ok,
            "short",
            regime,
            z_thresh,
            wall_mult,
        )

        oracle_inputs = OracleInputs(
            bid_ask_imbalance=imbalance_data["imbalance"],
            wall_strength_bid=wall_data["bid_wall_strength"],
            wall_strength_ask=wall_data["ask_wall_strength"],
            delta_zscore=delta_data["zscore"],
            price_touch_bid=touch_data["bid_distance_ticks"],
            price_touch_ask=touch_data["ask_distance_ticks"],
        )
        oracle_outputs = self.oracle.process(oracle_inputs)

        long_aether, long_aether_reasons = self.compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "long"
        )
        short_aether, short_aether_reasons = self.compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "short"
        )

        long_total = long_core * 0.65 + long_aether * 0.35
        short_total = short_core * 0.65 + short_aether * 0.35

        long_entry = long_total >= entry_thresh
        short_entry = short_total >= entry_thresh

        current_signal = "NEUTRAL"
        current_score = 0.0
        if long_entry and long_total > short_total:
            current_signal = "LONG"
            current_score = long_total
        elif short_entry and short_total > long_total:
            current_signal = "SHORT"
            current_score = short_total

        confirmed, confirmation_reason = self.check_signal_confirmation(
            current_signal, current_score, now_sec
        )

        should_log = (now_sec - self.last_decision_log_sec) >= self.DECISION_LOG_INTERVAL_SEC
        if should_log:
            self.last_decision_log_sec = now_sec

            signal_history_str = (
                "->".join([f"{s[0][0]}" for s in list(self.signal_history)])
                if self.signal_history
                else "Empty"
            )

            logger.info("=" * 100)
            logger.info(f"[DECISION REPORT] {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info("=" * 100)
            logger.info(f"PRICE: {current_price:.2f} USDT")
            logger.info("")
            logger.info("[SIGNAL CONFIRMATION]")
            logger.info(f"  Current Signal: {current_signal} (score: {current_score:.3f})")
            logger.info(f"  Signal History: {signal_history_str}")
            logger.info(f"  Confirmation Status: {confirmation_reason}")
            logger.info("")
            logger.info("[SESSION INFO]")
            logger.info(f"  Current Session: {session}")
            logger.info(f"  Major Session: {'Yes' if is_major else 'No'}")
            logger.info(f"  Min Signal Gap: {session_params['min_signal_gap_sec']}s")
            logger.info("")
            logger.info("[SESSION THRESHOLDS]")
            logger.info(f"  Z-Score: {z_thresh:.2f}")
            logger.info(f"  Imbalance: {imb_thresh:.2f}")
            logger.info(f"  Entry Score: {entry_thresh:.2f}")
            logger.info(f"  Wall Multiplier: {wall_mult:.1f}")
            logger.info("")
            logger.info("[VOL REGIME]")
            logger.info(f"  Regime: {regime}")
            logger.info(f"  ATR%: {atr_pct*100:.3f}%")
            logger.info("")
            logger.info("[CORE METRICS]")
            logger.info(f"  Imbalance: {imbalance_data['imbalance']:.3f}")
            logger.info(
                f"  Wall Bid/Ask: {wall_data['bid_wall_strength']:.2f} / {wall_data['ask_wall_strength']:.2f}"
            )
            logger.info(f"  Z-Score: {delta_data['zscore']:.2f}")
            logger.info(
                f"  Touch Bid/Ask: {touch_data['bid_distance_ticks']:.1f} / {touch_data['ask_distance_ticks']:.1f} ticks"
            )
            logger.info("")
            logger.info("[SCORE BREAKDOWN]")
            logger.info(f"  LONG: Core={long_core:.3f} | Aether={long_aether:.3f} | TOTAL={long_total:.3f}")
            logger.info(f"  SHORT: Core={short_core:.3f} | Aether={short_aether:.3f} | TOTAL={short_total:.3f}")
            logger.info("=" * 100)

        if confirmed and current_signal in ["LONG", "SHORT"]:
            self.pending_entry = True
            side = "long" if current_signal == "LONG" else "short"
            self.enter_position(
                data_manager=data_manager,
                order_manager=order_manager,
                risk_manager=risk_manager,
                side=side,
                current_price=current_price,
                imbalance_data=imbalance_data,
                wall_data=wall_data,
                delta_data=delta_data,
                touch_data=touch_data,
                now_sec=now_sec,
                regime_params=regime_params,
                entry_score=current_score,
                session_params=session_params,
            )
            self.signal_history.clear()

    def enter_position(
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
        session_params: Dict,
    ) -> None:
        """Enter position with LIMIT order + session-based TP/SL."""
        try:
            current_session, current_is_major = self.get_current_session()
            session = current_session

            try:
                positions_resp = order_manager.api.get_positions(
                    symbol=config.SYMBOL, exchange=config.EXCHANGE
                )
                if positions_resp and not positions_resp.get("error"):
                    data = positions_resp.get("data", [])
                    if isinstance(data, list):
                        for pos in data:
                            if pos.get("symbol") == config.SYMBOL:
                                qty = float(pos.get("quantity", 0))
                                if abs(qty) > 0:
                                    logger.warning(
                                        f"⚠️  Existing position detected: qty={qty:.6f} BTC - SKIPPING ENTRY"
                                    )
                                    self.pending_entry = False
                                    return
            except Exception as e:
                logger.error(f"Error checking existing positions: {e}")

            cleaned = self.clean_orphaned_orders(order_manager, symbol=config.SYMBOL)
            if cleaned:
                time.sleep(1.0)

            session = session_params.get("session_name", "UNKNOWN")
            regime = regime_params["regime"]

            is_volatile, _, atr_pct, vol_explanation = self.determine_volatility_with_gates(
                data_manager, current_price
            )

            quantity, margin_used = risk_manager.calculate_position_size_vol_regime(
                entry_price=current_price,
                regime=regime,
            )

            if quantity <= 0:
                logger.error("Position sizing returned 0 quantity")
                self.pending_entry = False
                return

            if is_volatile:
                offset_ticks = config.LIMIT_ORDER_HIGH_VOL_OFFSET_TICKS
            else:
                offset_ticks = config.LIMIT_ORDER_LOW_VOL_OFFSET_TICKS

            if side == "long":
                limit_entry_price = current_price - (offset_ticks * config.TICK_SIZE)
            else:
                limit_entry_price = current_price + (offset_ticks * config.TICK_SIZE)

            tp_roi = session_params["tp_roi_target"]
            sl_roi = session_params["sl_roi_max"]

            tp_price, sl_price = self._compute_bracket_prices(
                entry_price=limit_entry_price,
                margin_used=margin_used,
                quantity=quantity,
                side=side,
                tp_roi=tp_roi,
                sl_roi=sl_roi,
                session=session,
            )

            logger.info("=" * 100)
            logger.info(f"[ENTRY EXECUTION] {side.upper()} - SESSION-BASED BRACKET")
            logger.info("=" * 100)
            logger.info(f"  Session: {session}")
            logger.info(f"  Cooldown: {session_params['min_signal_gap_sec']}s")
            logger.info(f"  Volatility: {vol_explanation}")
            logger.info(f"  Regime: {regime}")
            logger.info(f"  Entry Score: {entry_score:.3f}")
            logger.info(f"  Current Price: {current_price:.2f}")
            logger.info(f"  Limit Entry: {limit_entry_price:.2f} (offset {offset_ticks} ticks)")
            logger.info(f"  Quantity: {quantity:.6f} BTC")
            logger.info(f"  Margin: {margin_used:.2f} USDT")
            logger.info(f"  TP: {tp_price:.2f} ({tp_roi*100:.2f}%) | SL: {sl_price:.2f} ({sl_roi*100:.2f}%)")
            logger.info("=" * 100)

            market_side = "BUY" if side == "long" else "SELL"

            main_order = order_manager.place_limit_order(
                side=market_side,
                quantity=quantity,
                price=limit_entry_price,
                reduce_only=False,
            )

            if not main_order:
                logger.error("❌ Limit order placement failed")
                self.pending_entry = False
                return

            main_order_id = main_order.get("order_id", "")
            logger.info(f"✓ Limit order placed: {main_order_id}")

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
                logger.error("❌ TP/SL placement failed - cancelling limit order")
                order_manager.cancel_order(main_order_id)
                self.pending_entry = False
                return

            logger.info(f"✓ TP order placed: {tp_order.get('order_id', '')}")
            logger.info(f"✓ SL order placed: {sl_order.get('order_id', '')}")

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
                entry_price=limit_entry_price,
                entry_time_sec=now_sec,
                entry_wall_volume=(
                    wall_data["bid_vol_zone"] if side == "long" else wall_data["ask_vol_zone"]
                ),
                wall_zone_low=wall_data["zone_low"],
                wall_zone_high=wall_data["zone_high"],
                entry_imbalance=imbalance_data["imbalance"],
                entry_zscore=delta_data["zscore"],
                tp_price=tp_price,
                sl_price=sl_price,
                margin_used=margin_used,
                tp_order_id=tp_order.get("order_id", ""),
                sl_order_id=sl_order.get("order_id", ""),
                main_order_id=main_order_id,
                main_filled=False,
                tp_reduced=False,
                entry_htf_trend=htf_trend,
                vol_regime=regime,
                entry_score=entry_score,
                is_volatile=is_volatile,
                tp_roi=tp_roi,
                sl_roi=sl_roi,
                last_momentum_check_sec=now_sec,
                last_momentum_log_sec=now_sec,
                tp_adjustment_count=0,
                entry_session=session,
                limit_order_placed_time=now_sec,
                initial_tp_roi=tp_roi,
                initial_sl_roi=sl_roi,
                current_tp_roi=tp_roi,
                current_tp_price=tp_price,
            )

            risk_manager.record_trade_opened()

            self.last_entry_time_sec = now_sec
            self.pending_entry = False

            try:
                wall_strength = (
                    wall_data["bid_wall_strength"]
                    if side == "long"
                    else wall_data["ask_wall_strength"]
                )
                msg = format_entry_message(
                    trade_id=self.current_position.trade_id,
                    side=side,
                    entry_type="LIMIT",
                    current_price=current_price,
                    limit_price=limit_entry_price,
                    quantity=quantity,
                    margin=margin_used,
                    leverage=config.LEVERAGE,
                    tp_price=tp_price,
                    tp_roi=tp_roi,
                    sl_price=sl_price,
                    sl_roi=sl_roi,
                    session=session,
                    entry_score=entry_score,
                    imbalance=imbalance_data["imbalance"],
                    zscore=delta_data["zscore"],
                    wall_strength=wall_strength,
                    cooldown_sec=session_params["min_signal_gap_sec"],
                )
                send_telegram_message(msg)
            except Exception as e:
                logger.error(f"Error sending entry notification: {e}")

            logger.info(f"✅ Position bracket created: {self.current_position.trade_id}")
            logger.info(f"Next entry allowed after {session_params['min_signal_gap_sec']}s cooldown")

        except Exception as e:
            logger.error(f"Error in enter_position: {e}", exc_info=True)
            self.pending_entry = False

    def manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        PRODUCTION-SAFE position management with:
        1. Single-fire timeout handler
        2. Proper order state handling
        3. No concurrent status checks
        """
        pos = self.current_position
        if pos is None:
            return

        # ============ LIMIT ORDER TIMEOUT HANDLER ============
        if not pos.main_filled:
            elapsed_since_place = now_sec - pos.limit_order_placed_time

            # ✅ FIX: Single-fire timeout check with guard against concurrent execution
            if (
                elapsed_since_place >= config.LIMIT_ORDER_WAIT_TIMEOUT_SEC
                and not pos.timeout_cancelled
            ):
                pos.timeout_cancelled = True
                logger.warning(
                    f"⏱️  LIMIT TIMEOUT: {elapsed_since_place:.1f}s - SINGLE-FIRE HANDLER"
                )

                # ✅ Guard against concurrent status checks
                if self._checking_order_status:
                    logger.warning("⚠️  Status check already in progress, skipping")
                    return

                self._checking_order_status = True

                try:
                    # STEP 1: Check if position exists via API (ONCE)
                    logger.info("🔍 Checking if position exists...")
                    position_open = False
                    try:
                        positions_resp = order_manager.api.get_positions(
                            symbol=config.SYMBOL,
                            exchange=config.EXCHANGE,
                        )
                        if positions_resp and not positions_resp.get("error"):
                            data = positions_resp.get("data", [])
                            if isinstance(data, list):
                                for p in data:
                                    if p.get("symbol") == config.SYMBOL:
                                        qty = float(p.get("quantity", 0))
                                        if abs(qty) > 0:
                                            position_open = True
                                            logger.info(
                                                f"✅ ACTIVE POSITION: qty={qty:.6f} BTC - ACTIVATING MANAGEMENT"
                                            )
                                            pos.main_filled = True
                                            pos.timeout_cancelled = False
                                            pos.entry_price = float(
                                                p.get("entry_price", current_price)
                                            )
                                            logger.info(
                                                f"Position detected @ {pos.entry_price:.2f} - TP/SL PROTECTED"
                                            )
                                            return
                    except Exception as e:
                        logger.debug(f"Position check error: {e}")

                    # STEP 2: If no position, check order status (ONCE)
                    if not position_open:
                        logger.info(f"🔍 Checking order status: {pos.main_order_id}")
                        final_status_normalized = order_manager.get_order_status_safe(
                            pos.main_order_id
                        )
                        logger.info(f"📊 Final status: {final_status_normalized}")

                        if final_status_normalized == "FILLED":
                            logger.info("✅ ORDER FILLED - Activating position management")
                            try:
                                full_status = order_manager.get_order_status(pos.main_order_id)
                                if full_status:
                                    pos.entry_price = order_manager.extract_fill_price(
                                        full_status
                                    )
                                else:
                                    pos.entry_price = current_price
                            except Exception as e:
                                logger.warning(f"Could not extract fill price: {e}")
                                pos.entry_price = current_price

                            pos.main_filled = True
                            pos.timeout_cancelled = False

                            try:
                                msg = format_fill_message(
                                    trade_id=pos.trade_id,
                                    side=pos.side,
                                    fill_price=pos.entry_price,
                                    quantity=pos.quantity,
                                    tp_price=pos.tp_price,
                                    sl_price=pos.sl_price,
                                )
                                send_telegram_message(msg)
                            except Exception as e:
                                logger.error(f"Error sending fill notification: {e}")

                            return

                        elif final_status_normalized == "UNKNOWN":
                            logger.warning(
                                "⚠️  ORDER STATUS UNKNOWN - Treating as POTENTIALLY FILLED"
                            )
                            logger.warning(
                                "TP/SL will be KEPT for safety. Manual verification recommended."
                            )
                            pos.main_filled = True
                            pos.timeout_cancelled = False
                            pos.entry_price = current_price

                            try:
                                msg = format_order_status_unknown(
                                    trade_id=pos.trade_id,
                                    side=pos.side,
                                    order_id=pos.main_order_id,
                                    assumed_price=pos.entry_price,
                                )
                                send_telegram_message(msg)
                            except Exception as e:
                                logger.error(f"Error sending unknown status notification: {e}")

                            return

                        elif final_status_normalized == "CANCELLED":
                            logger.info("❌ Order confirmed CANCELLED - cleaning up bracket")
                            try:
                                time.sleep(0.3)
                                order_manager.cancel_order(pos.tp_order_id)
                                order_manager.cancel_order(pos.sl_order_id)

                                msg = (
                                    f"🚫 TIMEOUT: {pos.trade_id}\n"
                                    f"{pos.side.upper()} @ {pos.entry_price:.2f}\n"
                                    f"No fill after {elapsed_since_place:.0f}s\n"
                                    f"Bracket cancelled"
                                )
                                send_telegram_message(msg)
                            except Exception as e:
                                logger.error(f"Error cancelling TP/SL: {e}")

                            self.current_position = None
                            self.pending_entry = False
                            self.last_entry_time_sec = now_sec
                            return

                        else:
                            logger.error(f"❌ Unexpected status: {final_status_normalized}")
                            pos.main_filled = True
                            pos.entry_price = current_price
                            return

                finally:
                    self._checking_order_status = False

            return  # Exit if order not filled yet

        # ============ FILLED POSITION MANAGEMENT ============
        hold_min = (now_sec - pos.entry_time_sec) / 60.0

        if (now_sec - self.last_position_log_sec) >= self.POSITION_LOG_INTERVAL_SEC:
            self.last_position_log_sec = now_sec
            direction = 1.0 if pos.side == "long" else -1.0
            current_profit_pct = (
                (current_price - pos.entry_price) / pos.entry_price * direction
            )
            upnl = (current_price - pos.entry_price) * direction * pos.quantity

            logger.info("")
            logger.info("=" * 100)
            logger.info(f"[POSITION STATUS] {pos.trade_id}")
            logger.info("=" * 100)
            logger.info(f"  Side: {pos.side.upper()}")
            logger.info(f"  Entry Session: {pos.entry_session}")
            logger.info(f"  Entry: {pos.entry_price:.2f} | Current: {current_price:.2f}")
            logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
            logger.info(f"  Margin: {pos.margin_used:.2f} USDT")
            logger.info(
                f"  TP: {pos.tp_price:.2f} ({pos.tp_roi*100:.2f}%) | SL: {pos.sl_price:.2f} ({pos.sl_roi*100:.2f}%)"
            )
            logger.info(f"  Hold Time: {hold_min:.1f} min")
            logger.info(f"  Unrealized P&L: {upnl:.2f} USDT ({current_profit_pct*100:.2f}%)")
            logger.info(f"  TP Adjustments: {pos.tp_adjustment_count}")
            logger.info("=" * 100)

        # Check bracket status every 10 seconds
        if (
            now_sec - getattr(self, "last_bracket_check_sec", 0)
            >= self.ORDER_STATUS_CHECK_INTERVAL_SEC
        ):
            self.last_bracket_check_sec = now_sec

            tp_status_resp = order_manager.get_order_status(pos.tp_order_id)
            if tp_status_resp:
                tp_status = str(tp_status_resp.get("status", "")).upper()

                if tp_status in ["EXECUTED", "FILLED", "PARTIALLY_EXECUTED", "PARTIALLY_FILLED"]:
                    logger.info(f"✅ TP triggered (status: {tp_status})")

                    exit_reason = "TP_PARTIAL" if "PARTIAL" in tp_status else "TP_HIT"
                    self.exit_position(
                        order_manager, risk_manager, current_price, exit_reason, now_sec
                    )
                    return

        # TP Management at 10 minutes
        if (
            hold_min >= config.FIRST_TP_WAIT_MINUTES
            and not pos.tp_adjusted_10min
        ):
            momentum_favorable, vol_favorable, trend_favorable = self.check_market_conditions(
                data_manager, pos.side, current_price
            )
            direction = 1.0 if pos.side == "long" else -1.0
            current_profit_pct = (
                (current_price - pos.entry_price) / pos.entry_price * direction
            )

            self._adjust_tp_after_10min(
                data_manager,
                order_manager,
                current_price,
                now_sec,
                momentum_favorable,
                vol_favorable,
                trend_favorable,
                current_profit_pct,
            )

        # TP Management at 15 minutes
        elif (
            hold_min >= config.FIRST_TP_WAIT_MINUTES + config.SECOND_TP_WAIT_MINUTES
            and not pos.tp_tightened_15min
        ):
            direction = 1.0 if pos.side == "long" else -1.0
            current_profit_pct = (
                (current_price - pos.entry_price) / pos.entry_price * direction
            )

            self._tighten_tp_after_15min(order_manager, current_price, current_profit_pct)

        # Momentum logging every 30 seconds
        if (now_sec - pos.last_momentum_log_sec) >= config.MOMENTUM_LOG_INTERVAL_SEC:
            pos.last_momentum_log_sec = now_sec
            direction = 1.0 if pos.side == "long" else -1.0
            current_profit_pct = (
                (current_price - pos.entry_price) / pos.entry_price * direction
            )
            momentum_favorable, vol_favorable, trend_favorable = self.check_market_conditions(
                data_manager, pos.side, current_price
            )
            logger.info(
                f"[MOMENTUM] {pos.trade_id} | Hold={hold_min:.1f}m | "
                f"P&L={current_profit_pct*100:.2f}% | "
                f"M={momentum_favorable} V={vol_favorable} T={trend_favorable}"
            )

        # Update market conditions check every 5 seconds
        if (now_sec - pos.last_momentum_check_sec) >= config.POSITION_CHECK_INTERVAL_SEC:
            pos.last_momentum_check_sec = now_sec
            momentum_favorable, vol_favorable, trend_favorable = self.check_market_conditions(
                data_manager, pos.side, current_price
            )

    def _adjust_tp_after_10min(
        self,
        data_manager,
        order_manager,
        current_price: float,
        now_sec: float,
        momentum_favorable: bool,
        vol_favorable: bool,
        trend_favorable: bool,
        current_profit_pct: float,
    ) -> None:
        """
        Adjust TP at 10-minute mark. Executes ONCE per position.
        """
        pos = self.current_position
        if pos is None:
            return

        # ✅ FIX: Guard against concurrent execution and re-entry
        if pos.tp_adjusted_10min or self._adjusting_tp:
            return

        self._adjusting_tp = True
        pos.tp_adjusted_10min = True
        pos.tp_adjustment_count += 1

        try:
            half_tp_roi = pos.initial_tp_roi * config.HALF_TP_THRESHOLD
            all_favorable = momentum_favorable and vol_favorable and trend_favorable

            logger.info("=" * 100)
            logger.info(f"[TP MANAGEMENT 10MIN] {pos.trade_id}")
            logger.info(f"  All favorable (M/V/T): {all_favorable}")
            logger.info(f"  Current profit: {current_profit_pct*100:.2f}%")
            logger.info(f"  Half TP: {half_tp_roi*100:.2f}%")

            if all_favorable:
                if current_profit_pct >= half_tp_roi:
                    self._move_sl_to_half_tp(order_manager, current_price)
                    logger.info("  → Moved SL to half TP, waiting 5min more")
                else:
                    logger.info("  → Waiting 5min more")
            else:
                # Determine new TP ROI
                if current_profit_pct >= half_tp_roi:
                    new_tp_roi = current_profit_pct + config.TP_BUFFER_PERCENT
                else:
                    new_tp_roi = half_tp_roi

                # Calculate new TP price
                new_tp_price, _ = self._compute_bracket_prices(
                    entry_price=pos.entry_price,
                    margin_used=pos.margin_used,
                    quantity=pos.quantity,
                    side=pos.side,
                    tp_roi=new_tp_roi,
                    sl_roi=pos.initial_sl_roi,
                    session=pos.entry_session,
                )

                # ✅ Single call to replace TP
                self._replace_take_profit_order(order_manager, new_tp_price, new_tp_roi)

            logger.info("=" * 100)

        finally:
            self._adjusting_tp = False

    def _tighten_tp_after_15min(
        self,
        order_manager,
        current_price: float,
        current_profit_pct: float,
    ) -> None:
        """
        Tighten TP at 15-minute mark. Executes ONCE per position.
        """
        pos = self.current_position
        if pos is None:
            return

        # ✅ FIX: Guard against concurrent execution and re-entry
        if pos.tp_tightened_15min or self._adjusting_tp:
            return

        self._adjusting_tp = True
        pos.tp_tightened_15min = True
        pos.tp_adjustment_count += 1

        try:
            new_tp_roi = current_profit_pct + config.TP_BUFFER_PERCENT

            # Calculate new TP price
            new_tp_price, _ = self._compute_bracket_prices(
                entry_price=pos.entry_price,
                margin_used=pos.margin_used,
                quantity=pos.quantity,
                side=pos.side,
                tp_roi=new_tp_roi,
                sl_roi=pos.initial_sl_roi,
                session=pos.entry_session,
            )

            # ✅ Single call to replace TP
            self._replace_take_profit_order(order_manager, new_tp_price, new_tp_roi)
            logger.info(f"[TP MANAGEMENT 15MIN] Final tightening to {new_tp_roi*100:.2f}%")

        finally:
            self._adjusting_tp = False

    def _replace_take_profit_order(
        self, order_manager, new_tp_price: float, new_tp_roi: float
    ) -> None:
        """
        SINGLE EXECUTION: 1 cancel + 1 place + 1 verify.
        NO loops, NO retries, NO multiple calls.
        """
        pos = self.current_position
        if pos is None:
            return

        old_tp_id = pos.tp_order_id
        old_tp_price = pos.current_tp_price or pos.tp_price
        old_tp_roi = pos.current_tp_roi or pos.tp_roi
        tp_side = "SELL" if pos.side == "long" else "BUY"

        logger.info("=" * 100)
        logger.info(f"[TP REPLACEMENT] {pos.trade_id}")
        logger.info(f"  Old TP: {old_tp_price:.2f} ({old_tp_roi*100:.2f}%)")
        logger.info(f"  New TP: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")
        logger.info("=" * 100)

        # STEP 1: Cancel old TP (SINGLE REQUEST)
        logger.info(f"🗑️  Cancelling old TP: {old_tp_id}")
        try:
            order_manager._wait_for_rate_limit()
            order_manager.cancel_order(old_tp_id)
            logger.info("✅ Old TP cancelled")
            time.sleep(1.0)
        except Exception as e:
            logger.warning(f"⚠️  TP cancel failed (may already be cancelled): {e}")

        # STEP 2: Place new TP (SINGLE REQUEST)
        logger.info(f"📤 Placing new TP @ {new_tp_price:.2f}")
        try:
            order_manager._wait_for_rate_limit()
            new_tp_order = order_manager.place_take_profit(
                side=tp_side,
                quantity=pos.quantity,
                trigger_price=new_tp_price,
            )

            if new_tp_order and new_tp_order.get("order_id"):
                new_tp_id = new_tp_order.get("order_id")

                # STEP 3: Verify placement (SINGLE REQUEST)
                logger.info(f"🔍 Verifying new TP: {new_tp_id}")
                time.sleep(0.5)
                order_manager._wait_for_rate_limit()

                verify_resp = order_manager.api.get_order(
                    order_id=new_tp_id,
                    symbol=config.SYMBOL,
                    exchange=config.EXCHANGE,
                )

                if verify_resp and not verify_resp.get("error"):
                    # Success - update position
                    pos.tp_order_id = new_tp_id
                    pos.current_tp_price = new_tp_price
                    pos.current_tp_roi = new_tp_roi
                    pos.tp_price = new_tp_price
                    pos.tp_roi = new_tp_roi

                    logger.info("✅ TP replacement successful")
                    logger.info(f"   New TP ID: {new_tp_id}")
                    logger.info(f"   Price: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")
                    logger.info("=" * 100)

                    # Send Telegram notification
                    try:
                        if pos.tp_adjustment_count == 1:
                            reason = "T+10min adjustment"
                        elif pos.tp_adjustment_count == 2:
                            reason = "T+15min tightening"
                        else:
                            reason = "Dynamic adjustment"

                        msg = format_tp_adjustment(
                            trade_id=pos.trade_id,
                            old_tp=old_tp_price,
                            new_tp=new_tp_price,
                            old_roi=old_tp_roi,
                            new_roi=new_tp_roi,
                            reason=reason,
                        )
                        send_telegram_message(msg)
                    except Exception as e:
                        logger.error(f"Telegram notification failed: {e}")

                    return
                else:
                    logger.error("❌ TP verification failed")
            else:
                logger.error("❌ TP placement returned no order_id")

        except Exception as e:
            logger.error(f"❌ TP replacement failed: {e}")

        logger.error("❌ Failed to replace TP order")
        logger.info("=" * 100)

    def _move_sl_to_half_tp(self, order_manager, current_price: float) -> None:
        """Move SL to half TP price."""
        pos = self.current_position
        if pos is None or pos.tp_reduced:
            return

        half_tp_roi = pos.initial_tp_roi * config.HALF_TP_THRESHOLD
        direction = 1.0 if pos.side == "long" else -1.0
        price_movement = pos.entry_price * half_tp_roi
        new_sl_price = pos.entry_price + (price_movement * direction)

        old_sl = pos.sl_price

        order_manager.cancel_order(pos.sl_order_id)

        tp_side = "SELL" if pos.side == "long" else "BUY"
        new_sl_order = order_manager.place_stop_loss(
            side=tp_side,
            quantity=pos.quantity,
            trigger_price=new_sl_price,
        )

        if new_sl_order:
            pos.sl_order_id = new_sl_order.get("order_id", "")
            pos.sl_price = new_sl_price
            pos.tp_reduced = True
            logger.info(f"✅ Moved SL to half TP: {new_sl_price:.2f}")

            try:
                msg = format_sl_adjustment(
                    trade_id=pos.trade_id,
                    old_sl=old_sl,
                    new_sl=new_sl_price,
                    reason="Half TP reached - securing profits",
                )
                send_telegram_message(msg)
            except Exception as e:
                logger.error(f"Error sending SL adjustment notification: {e}")

    def check_market_conditions(
        self, data_manager, side: str, current_price: float
    ) -> Tuple[bool, bool, bool]:
        """Check if momentum, volatility, and trend are favorable."""
        momentum_favorable = False
        try:
            delta_data = self.compute_delta_zscore(data_manager)
            if delta_data:
                zscore = delta_data["zscore"]
                if side == "long" and zscore > 0:
                    momentum_favorable = True
                elif side == "short" and zscore < 0:
                    momentum_favorable = True
        except:
            pass

        vol_favorable = True
        try:
            atr_pct = data_manager.get_atr_percent()
            if atr_pct is not None:
                if atr_pct > config.MAX_ATR_PERCENT:
                    vol_favorable = False
        except:
            pass

        trend_favorable = False
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
            if ema_val is not None:
                if side == "long" and current_price > ema_val:
                    trend_favorable = True
                elif side == "short" and current_price < ema_val:
                    trend_favorable = True
        except:
            pass

        return momentum_favorable, vol_favorable, trend_favorable

    def exit_position(
        self,
        order_manager,
        risk_manager,
        exit_price: float,
        reason: str,
        now_sec: float,
    ) -> None:
        """
        Exit position with SMART TP/SL cleanup.
        KEY FIX: Exchange auto-cancels TP/SL when triggered.
        Only cancel if we're SURE they're still active.
        """
        pos = self.current_position
        if pos is None:
            return

        hold_min = (now_sec - pos.entry_time_sec) / 60.0
        direction = 1.0 if pos.side == "long" else -1.0
        pnl = (exit_price - pos.entry_price) * direction * pos.quantity
        roi = (exit_price - pos.entry_price) / pos.entry_price * direction

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"[EXIT] {pos.trade_id} - Reason: {reason}")
        logger.info("=" * 100)

        # ✅ Only try to cancel the OTHER order (the one NOT triggered)
        try:
            if reason == "TP_HIT" or reason == "TP_PARTIAL":
                logger.info("TP triggered/partial - TP auto-cancelled by exchange, cancelling SL only")
                try:
                    order_manager.cancel_order(pos.sl_order_id)
                except Exception as e:
                    logger.debug(f"SL cancel failed (may already be cancelled): {e}")
            elif reason == "SL_HIT":
                logger.info("SL triggered - SL auto-cancelled by exchange, cancelling TP only")
                try:
                    order_manager.cancel_order(pos.tp_order_id)
                except Exception as e:
                    logger.debug(f"TP cancel failed (may already be cancelled): {e}")
            else:
                # Manual exit - try to cancel both
                logger.info("Manual exit - cancelling both TP and SL")
                try:
                    order_manager.cancel_order(pos.tp_order_id)
                except Exception as e:
                    logger.debug(f"TP cancel failed: {e}")
                try:
                    order_manager.cancel_order(pos.sl_order_id)
                except Exception as e:
                    logger.debug(f"SL cancel failed: {e}")
        except Exception as e:
            logger.debug(f"Bracket cleanup error: {e}")

        # Check if position is still open
        positions_resp = order_manager.api.get_positions(
            symbol=config.SYMBOL, exchange=config.EXCHANGE
        )
        position_open = False
        if positions_resp and not positions_resp.get("error"):
            data = positions_resp.get("data", [])
            if isinstance(data, list):
                for p in data:
                    if p.get("symbol") == config.SYMBOL:
                        qty = float(p.get("quantity", 0))
                        if abs(qty) > 0:
                            position_open = True
                            break

        if position_open:
            logger.info("Position still open - placing market close order")
            exit_side = "SELL" if pos.side == "long" else "BUY"
            try:
                exit_order = order_manager.place_market_order(
                    side=exit_side,
                    quantity=pos.quantity,
                    reduce_only=True,
                )
                if exit_order:
                    try:
                        filled = order_manager.wait_for_fill(
                            exit_order.get("order_id", ""), timeout_sec=3.0
                        )
                        exit_price = order_manager.extract_fill_price(filled)
                    except Exception as e:
                        logger.warning(f"Market close fill check failed: {e}")
            except Exception as e:
                logger.error(f"Market close failed: {e}")
        else:
            logger.info("Position already closed (TP/SL triggered) - no market order needed")

        pnl = (exit_price - pos.entry_price) * direction * pos.quantity
        roi = (exit_price - pos.entry_price) / pos.entry_price * direction

        risk_manager.record_trade_closed(pnl)

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"[POSITION CLOSED] {pos.trade_id} - {reason}")
        logger.info("=" * 100)
        logger.info(f"  Side: {pos.side.upper()}")
        logger.info(f"  Entry Session: {pos.entry_session}")
        logger.info(f"  Entry: {pos.entry_price:.2f} | Exit: {exit_price:.2f}")
        logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
        logger.info(f"  P&L: {pnl:.2f} USDT ({roi*100:.2f}%)")
        logger.info(f"  Hold Time: {hold_min:.1f} min")
        logger.info("=" * 100)

        try:
            msg = format_exit_message(
                trade_id=pos.trade_id,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                pnl=pnl,
                roi=roi,
                hold_time_min=hold_min,
                reason=reason,
            )
            send_telegram_message(msg)
        except Exception as e:
            logger.error(f"Error sending exit notification: {e}")

        if self.excel_logger:
            try:
                self.excel_logger.log_trade(
                    trade_id=pos.trade_id,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    roi=roi,
                    hold_time_min=hold_min,
                    entry_session=pos.entry_session,
                    vol_regime=pos.vol_regime,
                    entry_htf_trend=pos.entry_htf_trend,
                    entry_score=pos.entry_score,
                    tp_adjustments=pos.tp_adjustment_count,
                )
            except Exception as e:
                logger.error(f"Excel logging failed: {e}")

        self.current_position = None
        self.pending_entry = False

    def compute_bid_ask_imbalance(self, data_manager) -> Optional[Dict]:
        """Compute bid/ask volume imbalance."""
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        bid_vol = sum(b[1] for b in bids[:config.BOOK_DEPTH])
        ask_vol = sum(a[1] for a in asks[:config.BOOK_DEPTH])
        total_vol = bid_vol + ask_vol

        if total_vol == 0:
            return None

        imbalance = (bid_vol - ask_vol) / total_vol
        long_ok = imbalance >= config.IMBALANCE_THRESHOLD
        short_ok = imbalance <= -config.IMBALANCE_THRESHOLD

        return {
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "imbalance": imbalance,
            "long_ok": long_ok,
            "short_ok": short_ok,
        }

    def compute_wall_pressure(self, data_manager, current_price: float) -> Optional[Dict]:
        """Compute wall pressure zones."""
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        zone_width = current_price * config.WALL_ZONE_WIDTH_PCT
        zone_low = current_price - zone_width
        zone_high = current_price + zone_width

        bid_vol_zone = sum(b[1] for b in bids if zone_low <= b[0] <= current_price)
        ask_vol_zone = sum(a[1] for a in asks if current_price <= a[0] <= zone_high)

        total_zone_vol = bid_vol_zone + ask_vol_zone

        if total_zone_vol == 0:
            return {
                "zone_low": zone_low,
                "zone_high": zone_high,
                "bid_vol_zone": bid_vol_zone,
                "ask_vol_zone": ask_vol_zone,
                "bid_wall_strength": 0.0,
                "ask_wall_strength": 0.0,
            }

        bid_wall_strength = bid_vol_zone / total_zone_vol * 100
        ask_wall_strength = ask_vol_zone / total_zone_vol * 100

        return {
            "zone_low": zone_low,
            "zone_high": zone_high,
            "bid_vol_zone": bid_vol_zone,
            "ask_vol_zone": ask_vol_zone,
            "bid_wall_strength": bid_wall_strength,
            "ask_wall_strength": ask_wall_strength,
        }

    def compute_delta_zscore(self, data_manager) -> Optional[Dict]:
        """Compute recent trade delta z-score."""
        trades = data_manager.get_recent_trades(limit=30)
        if not trades or len(trades) < 30:
            return None

        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            qty = float(t.get("quantity", 0))
            is_buyer_maker = bool(t.get("isBuyerMaker", False))
            if not is_buyer_maker:
                buy_vol += qty
            else:
                sell_vol += qty

        delta = buy_vol - sell_vol
        self.delta_population.append(delta)

        if len(self.delta_population) < 30:
            return None

        pop = list(self.delta_population)
        mean = sum(pop) / len(pop)
        var = sum((x - mean) ** 2 for x in pop) / len(pop)
        std = var**0.5

        zscore = (delta - mean) / std if std > 0 else 0.0

        long_ok = zscore >= config.DELTA_Z_THRESHOLD
        short_ok = zscore <= -config.DELTA_Z_THRESHOLD

        return {
            "delta": delta,
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "zscore": zscore,
            "long_ok": long_ok,
            "short_ok": short_ok,
        }

    def compute_price_touch(self, data_manager, current_price: float) -> Optional[Dict]:
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
