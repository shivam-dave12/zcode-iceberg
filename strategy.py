"""
Z-Score Strategy with Session-Based Entry Control and Dynamic Parameters

IMPROVEMENTS:
1. Session-aware parameter selection (Weekend/Major/Overlap)
2. Entry cooldown to prevent noisy signals
3. 3-signal confirmation with time tracking
4. Single-fire timeout handling for limit orders
5. Safe orphaned order cleanup
6. Session-based TP/SL calculations
7. FIXED: Fresh session fetch on entry/TP-SL calc to prevent "UNKNOWN" (stale var bug)
8. FIXED: Early fill detection guard to prevent duplicate logging
9. FIXED: Centralized TP/SL calculation for consistent ROI methodology
10. FIXED: Industrial-grade TP management with single cancel+place cycle
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone
from collections import deque
import logging
import numpy as np
from scipy.stats import norm
import threading
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
    """Position tracking dataclass with session info"""
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
    # NEW: Track TP adjustments
    tp_adjusted_10min: bool = False
    tp_tightened_15min: bool = False
    current_tp_roi: float = 0.0
    current_tp_price: float = 0.0
    initial_tp_roi: float = 0.0
    initial_sl_roi: float = 0.0


class ZScoreIcebergHunterStrategy:
    """
    Z-Score Iceberg Hunter Strategy with Session-Based Control
    """

    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 60.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.pending_entry: bool = False
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0
        self._delta_population: deque = deque(maxlen=3000)
        self._last_decision_log_sec: float = 0.0
        self._last_position_log_sec: float = 0.0
        self._last_status_check_sec: float = 0.0
        self._oracle = AetherOracle()

        # Signal confirmation tracking (stores tuples of (side, score, timestamp))
        self._signal_history: deque = deque(maxlen=3)
        self._last_signal_time: float = 0.0

        # NEW: Entry cooldown tracking
        self.last_entry_time_sec: float = 0.0
        self.current_session_params: Dict = {}

        # FIX ISSUE 2: Early fill detection guard
        self.early_fill_handled: Dict[str, bool] = {}
        self._checking_order_status = False
        self._adjusting_tp = False  # ✅ ADDED: Prevent concurrent TP adjustments
        self._tp_lock = threading.RLock()
        self._adjusting_tp = False
        
        logger.info("Strategy initialized with thread-safe TP management")
        logger.info("=" * 80)
        logger.info("Z-SCORE STRATEGY INITIALIZED")
        logger.info("With Session + Weekend Volatility Gates")
        logger.info("Dynamic Session-Based Parameters")
        logger.info("Entry Cooldown & 3-Signal Confirmation")
        logger.info("=" * 80)

    # ======================================================================
    # SESSION DETECTION & PARAMETER SELECTION
    # ======================================================================

    def _get_current_session(self) -> Tuple[str, bool]:
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
        day_of_week = now_utc.weekday()  # Monday=0, Sunday=6

        # Weekend check (Saturday=5, Sunday=6)
        if day_of_week >= 5:
            return "WEEKEND", False

        # Session detection
        in_asia = 0 <= hour_utc < 9
        in_london = 8 <= hour_utc < 16
        in_ny = 13 <= hour_utc < 22

        # Overlap zones (highest volatility)
        if in_london and in_ny:  # London-NY overlap: 13:00-16:00 UTC
            return "LONDON_NY_OVERLAP", True
        elif in_asia and in_london:  # Asia-London overlap: 08:00-09:00 UTC
            return "ASIA_LONDON_OVERLAP", True
        elif in_london:
            return "LONDON", True
        elif in_ny:
            return "NEW_YORK", True
        elif in_asia:
            return "ASIA", True
        else:
            return "OFF_SESSION", False

    def _get_session_parameters(self, session: str, is_major_session: bool) -> Dict:
        """
        Get dynamic parameters based on current trading session.
        Returns session-specific thresholds and timing parameters.
        """
        if session == "WEEKEND":
            params = config.WEEKEND_PARAMS.copy()
            session_type = "WEEKEND"
        elif session in ("LONDON_NY_OVERLAP", "ASIA_LONDON_OVERLAP"):
            params = config.OVERLAP_SESSION_PARAMS.copy()
            session_type = session
        elif is_major_session:
            params = config.MAJOR_SESSION_PARAMS.copy()
            session_type = session
        else:
            # Off-peak hours
            params = config.WEEKEND_PARAMS.copy()
            session_type = "OFF_PEAK"

        # CRITICAL FIX: Add session_name to params dict
        params['session_name'] = session_type
        logger.debug(f"Session: {session_type} | Params: {params}")
        return params

    def _determine_volatility_with_gates(
        self,
        data_manager,
        current_price: float
    ) -> Tuple[bool, str, float, str]:
        """
        Two-gate volatility detection:
        Gate 1: Session + Weekend check
        Gate 2: ATR-based calculation
        Returns: (is_volatile, session, atr_pct, explanation)
        """
        # GATE 1: Session + Weekend
        session, is_major_session = self._get_current_session()

        # Get ATR for Gate 2
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception as e:
            logger.debug(f"Error getting ATR%: {e}")
            atr_pct = None

        # Weekend → always LOW volatility (override ATR)
        if session == "WEEKEND":
            explanation = "Weekend - Forced LOW volatility"
            return False, session, atr_pct or 0.0, explanation

        # Off-session → bias towards LOW unless ATR is extremely high
        if not is_major_session:
            if atr_pct and atr_pct > config.VOLATILE_ATR_THRESHOLD * 1.5:
                explanation = f"Off-session but ATR very high ({atr_pct*100:.3f}%)"
                return True, session, atr_pct, explanation
            else:
                explanation = f"Off-session - bias LOW volatility"
                return False, session, atr_pct or 0.0, explanation

        # GATE 2: Major session → use ATR threshold
        if atr_pct is not None:
            is_volatile = atr_pct > config.VOLATILE_ATR_THRESHOLD
            if is_volatile:
                explanation = f"Major session ({session}) + High ATR ({atr_pct*100:.3f}%)"
            else:
                explanation = f"Major session ({session}) + Normal ATR ({atr_pct*100:.3f}%)"
            return is_volatile, session, atr_pct, explanation
        else:
            # No ATR data, use session as indicator
            if "OVERLAP" in session:
                explanation = f"Overlap session ({session}) - assume HIGH volatility"
                return True, session, 0.0, explanation
            else:
                explanation = f"Major session ({session}) - assume NORMAL volatility"
                return False, session, 0.0, explanation

    # ======================================================================
    # SIGNAL CONFIRMATION WITH COOLDOWN
    # ======================================================================

    def _check_signal_confirmation(
        self,
        side: str,
        score: float,
        now_sec: float
    ) -> Tuple[bool, str]:
        """
        Check if we have 3 consecutive signals for the same side.
        Includes SESSION-BASED entry cooldown to prevent noise.
        Returns: (confirmed, reason)
        """
        # Get current session parameters
        session, is_major = self._get_current_session()
        session_params = self._get_session_parameters(session, is_major)
        min_gap = session_params["min_signal_gap_sec"]

        # ✅ FIXED: Enforce minimum time between signals
        if self._signal_history and len(self._signal_history) > 0:
            last_signal_time = self._signal_history[-1][2]
            time_since_last = now_sec - last_signal_time
            
            if time_since_last < 0.5:  # Minimum 500ms between signals
                return False, f"TOO_FAST ({time_since_last:.2f}s since last)"

        # Check if we're in cooldown period
        if now_sec - self.last_entry_time_sec < min_gap:
            remaining = min_gap - (now_sec - self.last_entry_time_sec)
            return False, f"COOLDOWN ({remaining:.0f}s remaining)"

        # Add signal to history (side, score, timestamp)
        self._signal_history.append((side, score, now_sec))

        # Need 3 consecutive signals
        if len(self._signal_history) < 3:
            return False, f"NEED_MORE ({len(self._signal_history)}/3)"

        # Check if all 3 are the same side (and not NEUTRAL)
        recent_3 = list(self._signal_history)[-3:]
        sides = [s[0] for s in recent_3]

        if sides[0] == sides[1] == sides[2] == side and side != "NEUTRAL":
            # ✅ FIXED: Check time span - must be at least 1 second
            time_span = now_sec - recent_3[0][2]
            
            if time_span < 1.0:
                return False, f"TOO_RAPID (3 signals in {time_span:.1f}s, need 1.0s)"
            
            # Calculate average score
            avg_score = sum(s[1] for s in recent_3) / 3
            logger.info(
                f"[SIGNAL CONFIRMATION] ✓ 3x {side.upper()} signals "
                f"over {time_span:.1f}s (avg score: {avg_score:.3f})"
            )
            return True, "CONFIRMED"

        return False, "NOT_CONSECUTIVE"

    # ======================================================================
    # FIX ISSUE 3: CENTRALIZED TP/SL CALCULATION
    # ======================================================================

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
        Centralized TP/SL calculation - MUST be used for initial and all adjustments.
        This ensures consistent methodology across all TP/SL computations.
        Returns: (tp_price, sl_price)
        """
        try:
            logger.info("=" * 80)
            logger.info("[TP/SL CALCULATION] Using session-based parameters")
            logger.info(f"  Session: {session}")
            logger.info(f"  Entry Price: {entry_price:.2f}")
            logger.info(f"  Margin Used: {margin_used:.2f} USDT")
            logger.info(f"  Quantity: {quantity:.6f} BTC")
            logger.info(f"  Side: {side.upper()}")
            logger.info(f"  Desired TP ROI: {tp_roi*100:.2f}%")
            logger.info(f"  Desired SL ROI: {sl_roi*100:.2f}%")

            if entry_price <= 0 or margin_used <= 0 or quantity <= 0:
                raise ValueError("Invalid input: prices/margin/quantity must be positive")

            # Calculate price movements based on ROI
            tp_price_movement = (margin_used * tp_roi) / quantity
            sl_price_movement = (margin_used * sl_roi) / quantity

            logger.info(f"  TP Price Movement: {tp_price_movement:.2f} USDT")
            logger.info(f"  SL Price Movement: {sl_price_movement:.2f} USDT")

            if side == "long":
                tp_price = entry_price + tp_price_movement
                sl_price = entry_price - sl_price_movement
            else:
                tp_price = entry_price - tp_price_movement
                sl_price = entry_price + sl_price_movement

            if tp_price <= 0 or sl_price <= 0:
                raise ValueError("Invalid TP/SL: prices must be positive")

            logger.info(f"  ✓ TP Price: {tp_price:.2f}")
            logger.info(f"  ✓ SL Price: {sl_price:.2f}")
            logger.info("=" * 80)

            return tp_price, sl_price

        except Exception as e:
            logger.error(f"Error in _compute_bracket_prices: {e}")
            # Fallback to safe defaults
            if side == "long":
                return entry_price * 1.01, entry_price * 0.99
            else:
                return entry_price * 0.99, entry_price * 1.01

    # ======================================================================
    # MAIN TICK HANDLER
    # ======================================================================

    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """Main per-tick strategy with session-aware logic."""
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            # If position exists, manage it
            if self.current_position is not None:
                self._manage_open_position(
                    data_manager=data_manager,
                    order_manager=order_manager,
                    risk_manager=risk_manager,
                    current_price=current_price,
                    now_sec=now_sec,
                )
                return

            # Don't enter if already pending
            if self.pending_entry:
                return

            # Check minimum time between trades
            if self.last_exit_time_min > 0:
                minutes_since_exit = (now_sec / 60.0) - self.last_exit_time_min
                if minutes_since_exit < config.MIN_TIME_BETWEEN_TRADES:
                    return

            # Check if trading is allowed
            allowed, reason = risk_manager.check_trading_allowed()
            if not allowed:
                return

            # Compute metrics
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(data_manager, current_price, imbalance_data)
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            # Get HTF trend
            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(f"Error fetching HTF trend: {e}")

            # Get Oracle inputs/outputs
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
                    ltf_trend=None,
                )
                oracle_outputs = self._oracle.decide(oracle_inputs, risk_manager)
            except Exception as e:
                logger.error(f"Oracle error: {e}")

            # Try entries with comprehensive logging
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
    # ENTRY DECISION WITH SESSION-BASED THRESHOLDS
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
        Evaluate LONG/SHORT with SESSION-BASED thresholds.
        Requires 3 consecutive identical signals + cooldown before entering.
        """
        if not all([imbalance_data, wall_data, delta_data, touch_data]):
            return

        if self.pending_entry:
            return

        # Get session and parameters
        session, is_major = self._get_current_session()
        session_params = self._get_session_parameters(session, is_major)

        # Use session-specific thresholds
        z_thresh = session_params["z_threshold"]
        imb_thresh = session_params["imbalance_threshold"]
        entry_thresh = session_params["entry_score_threshold"]
        wall_mult = session_params["wall_multiplier"]

        # Get volatility regime
        regime_params = self._get_regime_params(data_manager, current_price)
        regime = regime_params["regime"]
        atr_pct = regime_params.get("atr_pct", 0.0)

        # Get EMA for trend check
        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except:
            pass

        trend_long_ok = (ema_val is not None and current_price > ema_val)
        trend_short_ok = (ema_val is not None and current_price < ema_val)

        # Calculate scores with session thresholds
        long_core, long_core_reasons = self._compute_weighted_score(
            imbalance_data, wall_data, delta_data, touch_data,
            trend_long_ok, "long", regime, z_thresh, wall_mult
        )
        short_core, short_core_reasons = self._compute_weighted_score(
            imbalance_data, wall_data, delta_data, touch_data,
            trend_short_ok, "short", regime, z_thresh, wall_mult
        )

        long_aether, long_aether_reasons = self._compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "long"
        )
        short_aether, short_aether_reasons = self._compute_aether_fusion_score(
            oracle_inputs, oracle_outputs, "short"
        )

        long_total = long_core * 0.65 + long_aether * 0.35
        short_total = short_core * 0.65 + short_aether * 0.35

        # Use session-specific entry threshold
        long_entry = (long_total > entry_thresh)
        short_entry = (short_total > entry_thresh)

        # Determine current signal
        current_signal = "NEUTRAL"
        current_score = 0.0
        if long_entry and long_total > short_total:
            current_signal = "LONG"
            current_score = long_total
        elif short_entry and short_total > long_total:
            current_signal = "SHORT"
            current_score = short_total

        # Check for 3-signal confirmation with cooldown
        confirmed, confirmation_reason = self._check_signal_confirmation(
            current_signal, current_score, now_sec
        )

        # Periodic logging
        should_log = (now_sec - self._last_decision_log_sec >= self.DECISION_LOG_INTERVAL_SEC)
        if should_log:
            self._last_decision_log_sec = now_sec

            # Show signal history
            signal_history_str = " → ".join([
                f"{s[0][0]}" for s in list(self._signal_history)
            ]) if self._signal_history else "Empty"

            logger.info("\n" + "=" * 100)
            logger.info(f"[DECISION REPORT] {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info("=" * 100)
            logger.info(f"PRICE: {current_price:.2f} USDT")
            logger.info(f"")
            logger.info(f"[SIGNAL CONFIRMATION]")
            logger.info(f"  Current Signal: {current_signal} (score: {current_score:.3f})")
            logger.info(f"  Signal History: {signal_history_str}")
            logger.info(f"  Confirmation Status: {confirmation_reason}")
            logger.info(f"")
            logger.info(f"[SESSION INFO]")
            logger.info(f"  Current Session: {session}")
            logger.info(f"  Major Session: {'Yes' if is_major else 'No'}")
            logger.info(f"  Min Signal Gap: {session_params['min_signal_gap_sec']}s")
            logger.info(f"")
            logger.info(f"[SESSION THRESHOLDS]")
            logger.info(f"  Z-Score: {z_thresh:.2f}")
            logger.info(f"  Imbalance: {imb_thresh:.2f}")
            logger.info(f"  Entry Score: {entry_thresh:.2f}")
            logger.info(f"  Wall Multiplier: {wall_mult:.1f}")
            logger.info(f"")
            logger.info(f"[VOL REGIME]")
            logger.info(f"  Regime: {regime}")
            logger.info(f"  ATR%: {atr_pct*100:.3f}%")
            logger.info(f"")
            logger.info(f"[CORE METRICS]")
            logger.info(f"  Imbalance: {imbalance_data['imbalance']:.3f}")
            logger.info(f"  Wall Bid/Ask: {wall_data['bid_wall_strength']:.2f} / {wall_data['ask_wall_strength']:.2f}")
            logger.info(f"  Z-Score: {delta_data['z_score']:.2f}")
            logger.info(f"  Touch Bid/Ask: {touch_data['bid_distance_ticks']:.1f} / {touch_data['ask_distance_ticks']:.1f} ticks")
            logger.info(f"")
            logger.info(f"[SCORE BREAKDOWN]")
            logger.info(f"  LONG: Core={long_core:.3f} | Aether={long_aether:.3f} | TOTAL={long_total:.3f}")
            logger.info(f"  SHORT: Core={short_core:.3f} | Aether={short_aether:.3f} | TOTAL={short_total:.3f}")
            logger.info("=" * 100 + "\n")

        # Only enter if confirmed
        if confirmed and current_signal in ("LONG", "SHORT"):
            self.pending_entry = True
            side = "long" if current_signal == "LONG" else "short"

            self._enter_position(
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
            # Clear signal history after entry
            self._signal_history.clear()

    # ======================================================================
    # POSITION ENTRY WITH SESSION PARAMETERS
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
        session_params: Dict,
    ) -> None:
        """Enter position with LIMIT order + session-based TP/SL."""
        try:
            # CRITICAL FIX: Get fresh session (don't use stale variable)
            current_session, current_is_major = self._get_current_session()
            session = current_session

            # Check for existing position first
            try:
                positions_resp = order_manager.api.get_positions(
                    symbol=config.SYMBOL,
                    exchange=config.EXCHANGE
                )
                if positions_resp and not positions_resp.get("error"):
                    data = positions_resp.get("data", [])
                    if isinstance(data, list):
                        for pos in data:
                            if pos.get("symbol") == config.SYMBOL:
                                qty = float(pos.get("quantity", 0))
                                if abs(qty) > 0:
                                    logger.warning(
                                        f"⚠️ Existing position detected: {qty:.6f} BTC - SKIPPING ENTRY"
                                    )
                                    self.pending_entry = False
                                    return
            except Exception as e:
                logger.error(f"Error checking existing positions: {e}")

            # Clean up orphaned orders
            cleaned = self._clean_orphaned_orders(order_manager, symbol=config.SYMBOL)
            if cleaned:
                time.sleep(1.0)

            # Get session and volatility info
            session = session_params.get('session_name', 'UNKNOWN')
            regime = regime_params["regime"]

            is_volatile, _, atr_pct, vol_explanation = self._determine_volatility_with_gates(
                data_manager, current_price
            )

            # Position sizing
            quantity, margin_used = risk_manager.calculate_position_size_vol_regime(
                entry_price=current_price,
                regime=regime,
            )

            if quantity <= 0:
                logger.error("Position sizing returned 0 quantity")
                self.pending_entry = False
                return

            # Volatility-based entry offset
            if is_volatile:
                offset_ticks = config.LIMIT_ORDER_HIGH_VOL_OFFSET_TICKS
            else:
                offset_ticks = config.LIMIT_ORDER_LOW_VOL_OFFSET_TICKS

            if side == "long":
                limit_entry_price = current_price - (offset_ticks * config.TICK_SIZE)
            else:
                limit_entry_price = current_price + (offset_ticks * config.TICK_SIZE)

            # FIX ISSUE 3: Use centralized TP/SL calculation
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
            logger.info(f"  Limit Entry: {limit_entry_price:.2f} (offset: {offset_ticks} ticks)")
            logger.info(f"  Quantity: {quantity:.6f} BTC")
            logger.info(f"  Margin: {margin_used:.2f} USDT")
            logger.info(f"  TP: {tp_price:.2f} ({tp_roi*100:.2f}%)")
            logger.info(f"  SL: {sl_price:.2f} ({sl_roi*100:.2f}%)")
            logger.info("=" * 100)

            market_side = "BUY" if side == "long" else "SELL"

            # Place LIMIT entry
            main_order = order_manager.place_limit_order(
                side=market_side,
                quantity=quantity,
                price=limit_entry_price,
                reduce_only=False,
            )

            if not main_order:
                logger.error("Limit order placement failed")
                self.pending_entry = False
                return

            main_order_id = main_order.get("order_id", "")
            logger.info(f"✓ Limit order placed: {main_order_id}")

            # Place TP and SL immediately
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
                logger.error("TP/SL placement failed - cancelling limit order")
                order_manager.cancel_order(main_order_id)
                self.pending_entry = False
                return

            logger.info(f"✓ TP order placed: {tp_order.get('order_id', '')}")
            logger.info(f"✓ SL order placed: {sl_order.get('order_id', '')}")

            # Create position object
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
                entry_z_score=delta_data["z_score"],
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
                # FIX ISSUE 3: Track initial ROI values
                initial_tp_roi=tp_roi,
                initial_sl_roi=sl_roi,
                current_tp_roi=tp_roi,
                current_tp_price=tp_price,
            )

            risk_manager.record_trade_opened()

            # Update last entry time for cooldown
            self.last_entry_time_sec = now_sec
            self.pending_entry = False

            # Send formatted Telegram notification
            try:
                wall_strength = (
                    wall_data["bid_wall_strength"] if side == "long"
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
                    z_score=delta_data["z_score"],
                    wall_strength=wall_strength,
                    cooldown_sec=session_params['min_signal_gap_sec'],
                )
                send_telegram_message(msg)
            except Exception as e:
                logger.error(f"Error sending entry notification: {e}")

            logger.info(f"✓ Position bracket created: {self.current_position.trade_id}")
            logger.info(f"  Next entry allowed after {session_params['min_signal_gap_sec']}s cooldown")

        except Exception as e:
            logger.error(f"Error in _enter_position: {e}", exc_info=True)
            self.pending_entry = False

    # ======================================================================
    # POSITION MANAGEMENT WITH SAFE TIMEOUT
    # ======================================================================

    def _manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float
    ) -> None:
        """
        ✅ FIXED: Position management with proper rate limiting and SL adjustment
        
        Flow:
        1. Wait for limit fill (with timeout handling)
        2. T+10min: Check if profit > half TP, adjust TP and SL
        3. T+15min: Tighten TP to current profit + buffer
        """
        pos = self.current_position
        if pos is None:
            return
        
        # ========================================
        # PHASE 1: Wait for Limit Fill (120s timeout)
        # ========================================
    # ========================================
    # PHASE 1: Wait for Limit Fill with Enhanced Timeout
    # ========================================
    if not pos.main_filled:
        elapsed_since_place = now_sec - pos.limit_order_placed_time

        # Initialize timeout tracking attributes if not exist
        if not hasattr(pos, '_first_timeout_check_time'):
            pos._first_timeout_check_time = 0.0
            pos._second_timeout_check_time = 0.0
            pos._price_proximity_extension_granted = False

        # ===================================================================
        # FIRST TIMEOUT CHECK @ 60s
        # ===================================================================
        if (
            elapsed_since_place >= config.LIMIT_ORDER_WAIT_TIMEOUT_SEC
            and not pos.timeout_cancelled
            and pos._first_timeout_check_time == 0.0
        ):
            pos._first_timeout_check_time = now_sec
            logger.warning(
                f"⏱️ FIRST TIMEOUT CHECK @ {elapsed_since_place:.1f}s"
            )

            if self._checking_order_status:
                logger.warning("⚠️ Status check already in progress, skipping")
                return

            self._checking_order_status = True

            try:
                # Step 1: Check current price proximity to limit order (NO API CALL)
                current_price = data_manager.get_last_price()  # From websocket stream
                limit_price = pos.entry_price
                price_diff_ticks = abs(current_price - limit_price) / config.TICK_SIZE
                proximity_threshold = 50  # 50 ticks proximity threshold

                logger.info(f"📊 Price Proximity Check:")
                logger.info(f"   Current Price: {current_price:.2f}")
                logger.info(f"   Limit Price: {limit_price:.2f}")
                logger.info(f"   Distance: {price_diff_ticks:.1f} ticks")
                logger.info(f"   Threshold: {proximity_threshold} ticks")

                # If price is NEAR limit order, extend timeout by 60s
                if price_diff_ticks <= proximity_threshold:
                    logger.warning(
                        f"💡 PRICE PROXIMITY DETECTED: {price_diff_ticks:.1f} ticks from limit\n"
                        f"   → EXTENDING timeout by 60s (potential fill imminent)"
                    )
                    pos._price_proximity_extension_granted = True
                    return  # Exit and wait another 60s

                # If price is FAR, check order status immediately
                logger.info(f"✓ Price is FAR from limit ({price_diff_ticks:.1f} ticks) - checking order status")

                # Step 2: Check order status with single API call
                from order_manager import GlobalRateLimiter
                GlobalRateLimiter.wait()

                status_normalized = order_manager.get_order_status_safe(pos.main_order_id)
                logger.info(f"📊 Order status result: {status_normalized}")

                if status_normalized == "FILLED":
                    logger.info("✅ ORDER FILLED - Activating position management")
                    try:
                        full_status = order_manager.get_order_status(pos.main_order_id)
                        if full_status:
                            pos.entry_price = order_manager.extract_fill_price(full_status)
                        else:
                            pos.entry_price = current_price
                    except Exception as e:
                        logger.warning(f"Could not extract fill price: {e}")
                        pos.entry_price = current_price

                    pos.main_filled = True
                    pos.timeout_cancelled = False

                    try:
                        from telegram_notifier import format_fill_message, send_telegram_message
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

                # If status is UNKNOWN or PENDING, extend timeout
                elif status_normalized in ("UNKNOWN", "PENDING"):
                    logger.warning(
                        f"⚠️ ORDER STATUS: {status_normalized}\n"
                        f"   → EXTENDING timeout by 60s for status clarity"
                    )
                    pos._price_proximity_extension_granted = True
                    return  # Wait another 60s

                # If order is CANCELLED or REJECTED, proceed to cleanup
                else:
                    logger.info(f"🧹 Order status: {status_normalized} - proceeding to cleanup")
                    # Will handle cleanup in second timeout check

            finally:
                self._checking_order_status = False

        # ===================================================================
        # SECOND TIMEOUT CHECK @ 120s (after extension)
        # ===================================================================
        elif (
            elapsed_since_place >= (config.LIMIT_ORDER_WAIT_TIMEOUT_SEC * 2)
            and pos._first_timeout_check_time > 0.0
            and pos._second_timeout_check_time == 0.0
        ):
            pos._second_timeout_check_time = now_sec
            logger.warning(
                f"⏱️ SECOND TIMEOUT CHECK @ {elapsed_since_place:.1f}s"
            )

            if self._checking_order_status:
                logger.warning("⚠️ Status check already in progress, skipping")
                return

            self._checking_order_status = True

            try:
                # Step 1: Re-check price proximity (NO API CALL)
                current_price = data_manager.get_last_price()
                limit_price = pos.entry_price
                price_diff_ticks = abs(current_price - limit_price) / config.TICK_SIZE
                proximity_threshold = 50

                logger.info(f"📊 Second Price Proximity Check:")
                logger.info(f"   Current Price: {current_price:.2f}")
                logger.info(f"   Limit Price: {limit_price:.2f}")
                logger.info(f"   Distance: {price_diff_ticks:.1f} ticks")

                # Step 2: Check order status (single API call)
                from order_manager import GlobalRateLimiter
                GlobalRateLimiter.wait()

                status_normalized = order_manager.get_order_status_safe(pos.main_order_id)
                logger.info(f"📊 Final order status: {status_normalized}")

                # If FILLED, activate position
                if status_normalized == "FILLED":
                    logger.info("✅ ORDER FILLED at second check - Activating position")
                    try:
                        full_status = order_manager.get_order_status(pos.main_order_id)
                        if full_status:
                            pos.entry_price = order_manager.extract_fill_price(full_status)
                        else:
                            pos.entry_price = current_price
                    except Exception as e:
                        logger.warning(f"Could not extract fill price: {e}")
                        pos.entry_price = current_price

                    pos.main_filled = True
                    pos.timeout_cancelled = False

                    try:
                        from telegram_notifier import format_fill_message, send_telegram_message
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

                # If still price is near AND status is PENDING/UNKNOWN, do FINAL position check
                if price_diff_ticks <= proximity_threshold and status_normalized in ("PENDING", "UNKNOWN"):
                    logger.warning(
                        f"⚠️ CRITICAL: Price still near limit ({price_diff_ticks:.1f} ticks) AND status {status_normalized}\n"
                        f"   → Performing FINAL position check before any cancellations"
                    )

                    # Step 3: Final position check (single API call)
                    GlobalRateLimiter.wait()
                    positions_resp = order_manager.api.get_positions(
                        symbol=config.SYMBOL,
                        exchange=config.EXCHANGE,
                    )

                    if positions_resp and not positions_resp.get("error"):
                        data = positions_resp.get("data", [])
                        if isinstance(data, list):
                            for p in data:
                                if p.get("symbol") == config.SYMBOL:
                                    qty = abs(float(p.get("quantity", 0)))
                                    if qty > 0:
                                        logger.info(
                                            f"🛡️ ACTIVE POSITION DETECTED: {qty:.6f} BTC\n"
                                            f"   → Position is FILLED! Activating management with TP/SL PROTECTED"
                                        )
                                        pos.main_filled = True
                                        pos.timeout_cancelled = False
                                        pos.entry_price = float(p.get("entry_price", current_price))
                                        logger.info(f"✅ Position activated @ {pos.entry_price:.2f}")
                                        return

                # ===================================================================
                # SMART CANCELLATION: Cancel LIMIT first, preserve TP/SL if needed
                # ===================================================================
                logger.warning(
                    f"⚠️ FINAL TIMEOUT @ {elapsed_since_place:.0f}s\n"
                    f"   Order Status: {status_normalized}\n"
                    f"   Proceeding with smart cancellation..."
                )

                # Step 4: Cancel LIMIT order FIRST (single API call)
                logger.info(f"[1/3] Cancelling LIMIT order: {pos.main_order_id}")
                GlobalRateLimiter.wait()

                limit_cancel_resp = order_manager.api.cancel_order(
                    order_id=pos.main_order_id,
                    exchange=config.EXCHANGE,
                )

                limit_cancelled = False
                if limit_cancel_resp.get("error"):
                    error_msg = str(limit_cancel_resp.get("message", ""))
                    if "cannot be cancelled" in error_msg.lower() or "filled" in error_msg.lower():
                        logger.warning(
                            f"⚠️ LIMIT order already FILLED during cancellation attempt!\n"
                            f"   → ABORTING all cancellations - KEEPING TP/SL intact"
                        )
                        # Re-check position one more time
                        GlobalRateLimiter.wait()
                        positions_resp = order_manager.api.get_positions(
                            symbol=config.SYMBOL,
                            exchange=config.EXCHANGE,
                        )

                        if positions_resp and not positions_resp.get("error"):
                            data = positions_resp.get("data", [])
                            if isinstance(data, list):
                                for p in data:
                                    if p.get("symbol") == config.SYMBOL:
                                        qty = abs(float(p.get("quantity", 0)))
                                        if qty > 0:
                                            pos.main_filled = True
                                            pos.timeout_cancelled = False
                                            pos.entry_price = float(p.get("entry_price", current_price))
                                            logger.info(
                                                f"✅ LAST-SECOND FILL DETECTED @ {pos.entry_price:.2f}\n"
                                                f"   → TP/SL orders remain ACTIVE and PROTECTED"
                                            )
                                            return

                        # If no position found, something went wrong - keep TP/SL anyway
                        logger.error(
                            f"❌ LIMIT cancellation failed but no position found\n"
                            f"   → Playing safe: KEEPING TP/SL orders\n"
                            f"   → Manual verification required"
                        )
                        self.current_position = None
                        self.pending_entry = False
                        return

                    elif "not found" in error_msg.lower() or "cancelled" in error_msg.lower():
                        logger.info(f"✓ LIMIT order already cancelled/not found")
                        limit_cancelled = True
                    else:
                        logger.error(f"❌ LIMIT cancel error: {error_msg}")
                        # Proceed with TP/SL cancellation anyway
                        limit_cancelled = False
                else:
                    logger.info(f"✓ LIMIT order cancelled successfully")
                    limit_cancelled = True

                time.sleep(0.5)  # Brief delay for exchange processing

                # Step 5: Cancel TP order (single API call)
                logger.info(f"[2/3] Cancelling TP order: {pos.tp_order_id}")
                GlobalRateLimiter.wait()

                tp_cancel_resp = order_manager.api.cancel_order(
                    order_id=pos.tp_order_id,
                    exchange=config.EXCHANGE,
                )

                if tp_cancel_resp.get("error"):
                    logger.debug(f"TP cancel: {tp_cancel_resp.get('message', '')}")
                else:
                    logger.info(f"✓ TP order cancelled")

                time.sleep(0.3)

                # Step 6: Cancel SL order (single API call)
                logger.info(f"[3/3] Cancelling SL order: {pos.sl_order_id}")
                GlobalRateLimiter.wait()

                sl_cancel_resp = order_manager.api.cancel_order(
                    order_id=pos.sl_order_id,
                    exchange=config.EXCHANGE,
                )

                if sl_cancel_resp.get("error"):
                    logger.debug(f"SL cancel: {sl_cancel_resp.get('message', '')}")
                else:
                    logger.info(f"✓ SL order cancelled")

                # Cleanup
                logger.info(
                    f"✅ TIMEOUT CLEANUP COMPLETE\n"
                    f"   → All orders cancelled after {elapsed_since_place:.0f}s\n"
                    f"   → Strategy reset, ready for next signal"
                )

                try:
                    msg = (
                        f"⏱️ TIMEOUT #{pos.trade_id}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"  {pos.side.upper()} @ ${pos.entry_price:.2f}\n"
                        f"  No fill after {elapsed_since_place:.0f}s\n"
                        f"  All orders cancelled\n"
                        f"  Final Status: {status_normalized}"
                    )
                    from telegram_notifier import send_telegram_message
                    send_telegram_message(msg)
                except Exception as e:
                    logger.error(f"Error sending timeout notification: {e}")

                self.current_position = None
                self.pending_entry = False
                self.last_entry_time_sec = now_sec
                return

            finally:
                self._checking_order_status = False

        # ===================================================================
        # EARLY FILL DETECTION (between timeout checks)
        # ===================================================================
        if not hasattr(pos, "_last_early_check_time"):
            pos._last_early_check_time = 0.0

        if now_sec - pos._last_early_check_time >= config.EARLY_FILL_CHECK_INTERVAL_SEC:
            pos._last_early_check_time = now_sec

            if not self.early_fill_handled.get(pos.main_order_id, False):
                status_normalized = order_manager.get_order_status_safe(pos.main_order_id)

                if status_normalized == "FILLED":
                    self.early_fill_handled[pos.main_order_id] = True
                    logger.info(f"✅ EARLY FILL detected: {pos.main_order_id}")

                    try:
                        full_status = order_manager.get_order_status(pos.main_order_id)
                        if full_status:
                            pos.entry_price = order_manager.extract_fill_price(full_status)
                        else:
                            pos.entry_price = current_price
                    except Exception:
                        pos.entry_price = current_price

                    pos.main_filled = True
                    logger.info(f"Position activated @ {pos.entry_price:.2f}")

                    try:
                        from telegram_notifier import format_fill_message, send_telegram_message
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

            return  # Continue waiting for fill

    # ========================================
    # PHASE 2: FILLED POSITION MANAGEMENT
    # ========================================
        hold_min = (now_sec - pos.entry_time_sec) / 60.0
        direction = 1.0 if pos.side == "long" else -1.0
        # ✅ FIX: Use margin-based ROI (same methodology as _compute_bracket_prices)
        current_profit_roi = ((current_price - pos.entry_price) * pos.quantity / pos.margin_used) * direction

        
        # ========================================
        # Periodic Position Status Logging (60s)
        # ========================================
        if now_sec - self._last_position_log_sec >= self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            upnl = (current_price - pos.entry_price) * direction * pos.quantity
            
            logger.info("\n" + "=" * 100)
            logger.info(f"[POSITION STATUS] {pos.trade_id}")
            logger.info("=" * 100)
            logger.info(f"  Side: {pos.side.upper()}")
            logger.info(f"  Entry Session: {pos.entry_session}")
            logger.info(f"  Entry: {pos.entry_price:.2f} | Current: {current_price:.2f}")
            logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
            logger.info(f"  Margin: {pos.margin_used:.2f} USDT")
            logger.info(f"  TP: {pos.tp_price:.2f} ({pos.tp_roi*100:.2f}%) | SL: {pos.sl_price:.2f} ({pos.sl_roi*100:.2f}%)")
            logger.info(f"  Hold Time: {hold_min:.1f} min")
            logger.info(f"  Unrealized P&L: {upnl:.2f} USDT ({current_profit_roi*100:.2f}%)")
            logger.info(f"  TP Adjustments: {pos.tp_adjustment_count}")
            logger.info("=" * 100 + "\n")
        
        # ========================================
        # ✅ FIX: Bracket Exit Check (30s interval with rate limiting)
        # ========================================
        if not hasattr(self, '_last_bracket_check_sec'):
            self._last_bracket_check_sec = 0
        
        if now_sec - self._last_bracket_check_sec >= config.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_bracket_check_sec = now_sec
            
            # Check TP status
            tp_status_resp = order_manager.get_order_status(pos.tp_order_id)
            if tp_status_resp:
                tp_status = str(tp_status_resp.get("status", "")).upper()
                
                if tp_status in ("EXECUTED", "FILLED", "PARTIALLY_EXECUTED", "PARTIALLY_FILLED"):
                    logger.info(f"TP triggered (status: {tp_status})")
                    exit_reason = "TP_PARTIAL" if "PARTIAL" in tp_status else "TP_HIT"
                    self._exit_position(order_manager, risk_manager, current_price, exit_reason, now_sec)
                    return
            
            # Check SL status
            sl_status_resp = order_manager.get_order_status(pos.sl_order_id)
            if sl_status_resp:
                sl_status = str(sl_status_resp.get("status", "")).upper()
                if sl_status in ("EXECUTED", "FILLED", "PARTIALLY_EXECUTED", "PARTIALLY_FILLED"):
                    logger.info(f"SL triggered (status: {sl_status})")
                    self._exit_position(order_manager, risk_manager, current_price, "SL_HIT", now_sec)
                    return
        
        # ========================================
        # Market Conditions Check (5s interval)
        # ========================================
        momentum_favorable = False
        vol_favorable = True
        trend_favorable = False
        
        if now_sec - pos.last_momentum_check_sec >= config.POSITION_CHECK_INTERVAL_SEC:
            pos.last_momentum_check_sec = now_sec
            momentum_favorable, vol_favorable, trend_favorable = self._check_market_conditions(
                data_manager, pos.side, current_price
            )
        
        # Momentum logging (30s)
        if now_sec - pos.last_momentum_log_sec >= config.MOMENTUM_LOG_INTERVAL_SEC:
            pos.last_momentum_log_sec = now_sec
            logger.info(
                f"[MOMENTUM] {pos.trade_id} | Hold={hold_min:.1f}m | P&L={current_profit_roi*100:.2f}% | "
                f"M={momentum_favorable} V={vol_favorable} T={trend_favorable}"
            )
        
        # ========================================
        # ✅ FIXED: SESSION-BASED TP/SL MANAGEMENT TIMELINE
        # ========================================
        
        # T+10min: Half-TP and Breakeven SL Logic
        if (hold_min >= config.FIRST_TP_WAIT_MINUTES and not pos.tp_adjusted_10min):
            self._adjust_tp_and_sl_after_10min(
                data_manager, order_manager, current_price, now_sec,
                momentum_favorable, vol_favorable, trend_favorable,
                current_profit_roi
            )
        
        # T+15min: Final TP Tightening
        elif (hold_min >= (config.FIRST_TP_WAIT_MINUTES + config.SECOND_TP_WAIT_MINUTES) and not pos.tp_tightened_15min):
            self._tighten_tp_after_15min(
                order_manager, current_price, current_profit_roi
            )

    # ======================================================================
    # FIX ISSUE 3 & 4: INDUSTRIAL-GRADE TP MANAGEMENT
    # ======================================================================

    def _adjust_tp_and_sl_after_10min(
        self,
        data_manager,
        order_manager,
        current_price: float,
        now_sec: float,
        momentum_favorable: bool,
        vol_favorable: bool,
        trend_favorable: bool,
        current_profit_roi: float,
    ) -> None:
        """
        ✅ FIXED: T+10min management following Excel flow

        Excel Flow:
        - Case 1 (profit < half of original TP): Keep SL as-is, set TP to half of original TP
        - Case 2 (profit >= half of original TP): Keep TP as-is, move SL to breakeven (fees + GST)
        """
        pos = self.current_position
        if pos is None:
            return

        # Store data manager reference for price safety checks
        self._data_manager_ref = data_manager

        with self._tp_lock:
            if self._adjusting_tp:
                logger.warning("⚠️ TP/SL adjustment already in progress, skipping 10min adjust")
                return

            self._adjusting_tp = True

            try:
                pos.tp_adjusted_10min = True

                half_tp_roi = pos.initial_tp_roi * config.HALF_TP_THRESHOLD

                logger.info("=" * 100)
                logger.info(f"[TP/SL MANAGEMENT 10MIN] {pos.trade_id}")
                logger.info(f"  Current profit: {current_profit_roi*100:.2f}%")
                logger.info(f"  Half TP target: {half_tp_roi*100:.2f}%")

                # Case 2: Profit >= half of original TP -> Move SL to breakeven, TP unchanged
                if current_profit_roi >= half_tp_roi:
                    logger.info("  → Excel Case 2: Profit >= half TP → SL to breakeven, TP unchanged")

                    success = self._move_sl_to_breakeven(order_manager, current_price, pos.entry_price)

                    if success:
                        logger.info("  ✅ T+10min adjustment complete (SL to breakeven)")
                    else:
                        logger.error("  ❌ SL adjustment failed - keeping existing bracket")

                    logger.info("=" * 100)
                    return

                # Case 1: Profit < half of original TP -> TP to half, SL unchanged
                logger.info("  → Excel Case 1: Profit < half TP → TP to half, SL unchanged")

                new_tp_price, _ = self._compute_bracket_prices(
                    entry_price=pos.entry_price,
                    margin_used=pos.margin_used,
                    quantity=pos.quantity,
                    side=pos.side,
                    tp_roi=half_tp_roi,
                    sl_roi=pos.initial_sl_roi,
                    session=pos.entry_session,
                )

                success = self._replace_take_profit_order(order_manager, new_tp_price, half_tp_roi)

                if success:
                    logger.info(f"  ✅ TP reduced to {half_tp_roi*100:.2f}%, SL unchanged")
                    pos.tp_adjustment_count += 1
                else:
                    logger.error("  ❌ TP adjustment failed - keeping existing bracket")

                logger.info("=" * 100)

            finally:
                self._adjusting_tp = False

    def _move_sl_to_breakeven(
        self,
        order_manager,
        current_price: float,
        entry_price: float
    ) -> bool:
        """
        ✅ FIXED: Move SL to breakeven with safety checks
        Returns True if successful, False otherwise
        """
        pos = self.current_position
        if pos is None:
            return False
        
        try:
            logger.info("=" * 80)
            logger.info(f"[SL TO BREAKEVEN] {pos.trade_id}")
            logger.info(f"  Entry Price: {entry_price:.2f}")
            logger.info(f"  Breakeven Fee Buffer: {config.BREAKEVEN_FEE_BUFFER_PCT*100:.4f}%")
            
            # Calculate breakeven SL using same methodology
            _, new_sl_price = self._compute_bracket_prices(
                entry_price=entry_price,
                margin_used=pos.margin_used,
                quantity=pos.quantity,
                side=pos.side,
                tp_roi=pos.current_tp_roi,
                sl_roi=config.BREAKEVEN_FEE_BUFFER_PCT,
                session=pos.entry_session,
            )
            
            success = self._replace_stop_loss_order(order_manager, new_sl_price, config.BREAKEVEN_FEE_BUFFER_PCT)
            
            if success:
                logger.info(f"  ✓ SL moved to breakeven: {new_sl_price:.2f}")
            else:
                logger.error(f"  ✗ Failed to move SL to breakeven")
            
            logger.info("=" * 80)
            return success
            
        except Exception as e:
            logger.error(f"Error moving SL to breakeven: {e}", exc_info=True)
            return False

    def _replace_stop_loss_order(
        self,
        order_manager,
        new_sl_price: float,
        new_sl_roi: float
    ) -> bool:
        """
        ✅ FIXED: Atomically replace SL with PRICE SAFETY CHECK
        
        Flow:
        1. Check if new SL price is safe (far enough from current price)
        2. If UNSAFE → Abort and keep existing SL
        3. If SAFE → Cancel old SL (1 API call)
        4. Place new SL (1 API call)
        5. Verify placement (1 API call)
        
        Total: 3 API calls (with rate limiting)
        """
        pos = self.current_position
        if pos is None:
            logger.error("No active position for SL replacement")
            return False
        
        try:
            old_sl_id = pos.sl_order_id
            old_sl_price = pos.sl_price
            old_sl_roi = pos.sl_roi
            
            # ========================================
            # STEP 1: PRICE SAFETY CHECK (NO API CALL)
            # ========================================
            current_price = pos.entry_price  # Use cached price
            try:
                # Get real-time price from data manager
                if hasattr(self, '_data_manager_ref'):
                    current_price = self._data_manager_ref.get_last_price()
            except:
                pass
            
            # Calculate distance from current price to new SL
            if pos.side == "long":
                distance = current_price - new_sl_price
            else:
                distance = new_sl_price - current_price
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[SL REPLACEMENT SAFETY CHECK] {pos.trade_id}")
            logger.info(f"  Current Price: {current_price:.2f}")
            logger.info(f"  Old SL: {old_sl_price:.2f} ({old_sl_roi*100:.2f}%)")
            logger.info(f"  New SL: {new_sl_price:.2f} ({new_sl_roi*100:.2f}%)")
            logger.info(f"  Distance: {distance:.2f} USDT ({distance/config.TICK_SIZE:.1f} ticks)")
            logger.info(f"  Required: {config.MIN_SL_DISTANCE_PRICE:.2f} USDT ({config.MIN_SL_DISTANCE_TICKS} ticks)")
            
            # Safety check: New SL must be far enough from current price
            if distance < config.MIN_SL_DISTANCE_PRICE:
                logger.error(f"  ❌ UNSAFE: New SL too close to current price!")
                logger.error(f"     Distance {distance:.2f} < Required {config.MIN_SL_DISTANCE_PRICE:.2f}")
                logger.error(f"     ABORTING SL replacement to prevent order rejection")
                logger.error(f"     Keeping existing SL @ {old_sl_price:.2f}")
                logger.info("=" * 80)
                return False
            
            logger.info(f"  ✓ SAFE: Distance check passed ({distance:.2f} >= {config.MIN_SL_DISTANCE_PRICE:.2f})")
            
            # Check if new SL is better than old SL (tighter stop = better protection)
            sl_improvement = (new_sl_price - old_sl_price) if pos.side == "long" else (old_sl_price - new_sl_price)
            if sl_improvement > 0:
                logger.info(f"  ✓ IMPROVEMENT: SL tightened by {sl_improvement:.2f} USDT (better protection)")
            else:
                logger.warning(f"  ⚠️ WARNING: New SL is looser by {abs(sl_improvement):.2f} USDT")
            
            logger.info("")
            logger.info(f"[SL REPLACEMENT] {pos.trade_id}")
            logger.info(f"  Old SL: {old_sl_price:.2f} ({old_sl_roi*100:.2f}%)")
            logger.info(f"  New SL: {new_sl_price:.2f} ({new_sl_roi*100:.2f}%)")
            logger.info("")
            
            # ========================================
            # STEP 2: CANCEL OLD SL (1 API CALL)
            # ========================================
            logger.info(f"  [1/3] Cancelling old SL: {old_sl_id}")
            
            from order_manager import GlobalRateLimiter
            GlobalRateLimiter.wait()  # Rate limit protection
            
            cancel_resp = order_manager.api.cancel_order(
                order_id=old_sl_id,
                exchange=config.EXCHANGE,
            )
            
            if cancel_resp.get("error"):
                error_msg = str(cancel_resp.get("message", "Unknown error"))
                
                # Check if already cancelled/executed
                if "cannot be cancelled" in error_msg.lower() or "CANCELLED" in error_msg:
                    logger.warning(f"  ⚠️ Old SL already cancelled or executed: {error_msg}")
                elif "not found" in error_msg.lower():
                    logger.warning(f"  ⚠️ Old SL not found (may have triggered): {error_msg}")
                else:
                    logger.error(f"  ❌ SL cancel failed: {error_msg}")
                    logger.error(f"     ABORTING to keep existing SL protection")
                    logger.info("=" * 80)
                    return False
            else:
                logger.info(f"  ✓ Old SL cancelled successfully")
            
            time.sleep(1.0)  # Wait for cancellation to propagate
            
            # ========================================
            # STEP 3: PLACE NEW SL (1 API CALL)
            # ========================================
            logger.info(f"  [2/3] Placing new SL: {new_sl_price:.2f}")
            
            GlobalRateLimiter.wait()  # Rate limit protection
            
            sl_side = "SELL" if pos.side == "long" else "BUY"
            new_sl_order = order_manager.place_stop_loss(
                side=sl_side,
                quantity=pos.quantity,
                trigger_price=new_sl_price,
            )
            
            if not new_sl_order or "order_id" not in new_sl_order:
                logger.error(f"  ❌ SL placement failed: {new_sl_order}")
                logger.error(f"     ⚠️ CRITICAL: Position now has NO SL protection!")
                logger.error(f"     Attempting to restore old SL...")
                
                # Emergency: Try to restore old SL
                try:
                    GlobalRateLimiter.wait()
                    restore_sl = order_manager.place_stop_loss(
                        side=sl_side,
                        quantity=pos.quantity,
                        trigger_price=old_sl_price,
                    )
                    if restore_sl and "order_id" in restore_sl:
                        logger.info(f"     ✓ Old SL restored: {restore_sl['order_id']}")
                        pos.sl_order_id = restore_sl["order_id"]
                    else:
                        logger.error(f"     ❌ Could not restore old SL!")
                except Exception as restore_err:
                    logger.error(f"     ❌ Restore failed: {restore_err}")
                
                logger.info("=" * 80)
                return False
            
            new_sl_id = new_sl_order["order_id"]
            logger.info(f"  ✓ New SL placed: {new_sl_id}")
            
            # ========================================
            # STEP 4: VERIFY PLACEMENT (1 API CALL)
            # ========================================
            logger.info(f"  [3/3] Verifying new SL: {new_sl_id}")
            
            time.sleep(2.0)  # Wait for order to register
            GlobalRateLimiter.wait()  # Rate limit protection
            
            verify_resp = order_manager.get_order_status(new_sl_id)
            
            if not verify_resp:
                logger.error(f"  ❌ Could not verify new SL {new_sl_id}")
                logger.error(f"     Status check failed - manual verification required")
                logger.info("=" * 80)
                return False
            
            verify_status = str(verify_resp.get("status", "")).upper()
            
            if verify_status in ("CANCELLED", "REJECTED", "EXPIRED"):
                logger.error(f"  ❌ New SL rejected with status: {verify_status}")
                logger.error(f"     ⚠️ Position may have NO SL protection!")
                logger.info("=" * 80)
                return False
            
            if verify_status in ("PENDING", "NEW", "OPEN"):
                logger.info(f"  ✓ SL replacement verified (status: {verify_status})")
            else:
                logger.warning(f"  ⚠️ Unexpected SL status: {verify_status}")
            
            logger.info("")
            logger.info(f"  ✅ SL REPLACEMENT SUCCESSFUL")
            logger.info(f"     New SL ID: {new_sl_id}")
            logger.info(f"     Price: {new_sl_price:.2f} ({new_sl_roi*100:.2f}%)")
            logger.info("=" * 80)
            
            # ========================================
            # STEP 5: UPDATE POSITION TRACKING
            # ========================================
            pos.sl_order_id = new_sl_id
            pos.sl_price = new_sl_price
            pos.sl_roi = new_sl_roi
            
            # ========================================
            # STEP 6: SEND TELEGRAM NOTIFICATION
            # ========================================
            try:
                from telegram_notifier import send_telegram_message
                is_breakeven = abs(new_sl_roi) < 0.005  # Within 0.5% = breakeven
                
                msg = (
                    f"{'🛡️ SL TO BREAKEVEN' if is_breakeven else '📊 SL ADJUSTED'} #{pos.trade_id}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"  {pos.side.upper()} @ ${pos.entry_price:.2f}\n"
                    f"  Old SL: ${old_sl_price:.2f} ({old_sl_roi*100:.2f}%)\n"
                    f"  New SL: ${new_sl_price:.2f} ({new_sl_roi*100:.2f}%)\n"
                    f"  Distance: {distance:.2f} USDT ({distance/config.TICK_SIZE:.1f} ticks)\n"
                    f"  Status: ✅ ACTIVE"
                )
                send_telegram_message(msg)
            except Exception as e:
                logger.error(f"Error sending SL adjustment notification: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error replacing SL order: {e}", exc_info=True)
            logger.error(f"   ⚠️ Position protection status UNKNOWN - manual check required")
            return False

    def _tighten_tp_after_15min(
        self,
        order_manager,
        current_price: float,
        current_profit_roi: float,
    ) -> None:
        """
        ✅ FIXED T+15min management:
        
        **ALWAYS** tighten TP to current profit + buffer (with breakeven floor).
        **ALWAYS** set SL to half of the NEW tightened TP (profit-lock when in profit).
        
        Example: If current profit is 8%, set TP at 9% (8% + 1% buffer).
        """
        pos = self.current_position
        if pos is None:
            return

        with self._tp_lock:
            if self._adjusting_tp:
                logger.warning("⚠️ TP adjustment already in progress, skipping 15min tighten")
                return

            self._adjusting_tp = True

            try:
                pos.tp_tightened_15min = True

                half_original_tp_roi = pos.initial_tp_roi * config.HALF_TP_THRESHOLD

                logger.info("=" * 100)
                logger.info(f"[TP/SL MANAGEMENT 15MIN] {pos.trade_id}")
                logger.info(f"  Current profit ROI: {current_profit_roi*100:.2f}%")
                logger.info(f"  Half original TP threshold: {half_original_tp_roi*100:.2f}%")

                # ✅ ALWAYS tighten TP to current profit + buffer
                new_tp_roi = current_profit_roi + config.TP_BUFFER_PERCENT

                # Breakeven clamp: TP never goes below fees+GST (+10% cushion)
                breakeven_floor = config.BREAKEVEN_FEE_BUFFER_PCT * 1.10
                if new_tp_roi < breakeven_floor:
                    logger.info(
                        f"  ⚠️ TP breakeven clamp: requested {new_tp_roi*100:.2f}% "
                        f"< floor {breakeven_floor*100:.2f}%, using floor."
                    )
                    new_tp_roi = breakeven_floor

                # SL should be HALF of the NEW tightened TP (not half of original TP)
                # Use negative ROI to convert SL into profit-lock when in profit
                # (SL above entry for LONG / below entry for SHORT)
                half_new_tp = abs(new_tp_roi) * config.HALF_TP_THRESHOLD
                new_sl_roi = -half_new_tp if current_profit_roi > 0 else pos.sl_roi

                logger.info(f"  New TP target ROI: {new_tp_roi*100:.2f}%")
                logger.info(
                    f"  New SL target ROI: {abs(new_sl_roi)*100:.2f}% "
                    f"({'PROFIT-LOCK' if new_sl_roi < 0 else 'UNCHANGED'})"
                )

                new_tp_price, new_sl_price = self._compute_bracket_prices(
                    entry_price=pos.entry_price,
                    margin_used=pos.margin_used,
                    quantity=pos.quantity,
                    side=pos.side,
                    tp_roi=new_tp_roi,
                    sl_roi=new_sl_roi,
                    session=pos.entry_session,
                )

                tp_success = self._replace_take_profit_order(order_manager, new_tp_price, new_tp_roi)

                # Only attempt SL replace if it needs to change
                sl_success = False
                if new_sl_roi != pos.sl_roi:
                    sl_success = self._replace_stop_loss_order(order_manager, new_sl_price, new_sl_roi)
                else:
                    logger.info("  SL unchanged (already optimal)")
                    sl_success = True  # Consider it success if no change needed

                if tp_success or sl_success:
                    pos.tp_adjustment_count += 1

                if tp_success and sl_success:
                    logger.info("  ✅ TP and SL updated at 15min")
                elif tp_success:
                    logger.warning("  ⚠️ TP updated but SL update skipped/failed")
                elif sl_success:
                    logger.warning("  ⚠️ SL updated but TP update failed")
                else:
                    logger.error("  ❌ TP and SL update failed - keeping existing bracket")

                logger.info("=" * 100)

            finally:
                self._adjusting_tp = False


    def _replace_take_profit_order(
        self,
        order_manager,
        new_tp_price: float,
        new_tp_roi: float
    ) -> bool:
        """
        ✅ FIXED: Atomically replace TP with PRICE SAFETY CHECK
        
        Flow:
        1. Check if new TP price is safe (far enough from current price)
        2. If UNSAFE → Abort and keep existing TP
        3. If SAFE → Cancel old TP (1 API call)
        4. Place new TP (1 API call)
        5. Verify placement (1 API call)
        
        Total: 3 API calls (with rate limiting)
        """
        pos = self.current_position
        if pos is None:
            logger.error("No active position for TP replacement")
            return False
        
        try:
            old_tp_id = pos.tp_order_id
            old_tp_price = pos.tp_price
            old_tp_roi = pos.tp_roi
            
            # ========================================
            # STEP 1: PRICE SAFETY CHECK (NO API CALL)
            # ========================================
            current_price = pos.entry_price  # Use cached price from data stream
            try:
                # Get real-time price from data manager
                import data_manager
                if hasattr(self, '_data_manager_ref'):
                    current_price = self._data_manager_ref.get_last_price()
            except:
                pass
            
            # Calculate distance from current price to new TP
            if pos.side == "long":
                distance = new_tp_price - current_price
            else:
                distance = current_price - new_tp_price
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[TP REPLACEMENT SAFETY CHECK] {pos.trade_id}")
            logger.info(f"  Current Price: {current_price:.2f}")
            logger.info(f"  Old TP: {old_tp_price:.2f} ({old_tp_roi*100:.2f}%)")
            logger.info(f"  New TP: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")
            logger.info(f"  Distance: {distance:.2f} USDT ({distance/config.TICK_SIZE:.1f} ticks)")
            logger.info(f"  Required: {config.MIN_TP_DISTANCE_PRICE:.2f} USDT ({config.MIN_TP_DISTANCE_TICKS} ticks)")
            
            # Safety check: New TP must be far enough from current price
            if distance < config.MIN_TP_DISTANCE_PRICE:
                logger.error(f"  ❌ UNSAFE: New TP too close to current price!")
                logger.error(f"     Distance {distance:.2f} < Required {config.MIN_TP_DISTANCE_PRICE:.2f}")
                logger.error(f"     ABORTING TP replacement to prevent order rejection")
                logger.error(f"     Keeping existing TP @ {old_tp_price:.2f}")
                logger.info("=" * 80)
                return False
            
            logger.info(f"  ✓ SAFE: Distance check passed ({distance:.2f} >= {config.MIN_TP_DISTANCE_PRICE:.2f})")
            
            # Check if new TP is better than old TP (for LONG: lower is better, SHORT: higher is better)
            tp_improvement = (old_tp_price - new_tp_price) if pos.side == "long" else (new_tp_price - old_tp_price)
            if tp_improvement <= 0:
                logger.warning(f"  ⚠️ WARNING: New TP is not an improvement")
                logger.warning(f"     Old TP better by {abs(tp_improvement):.2f} USDT")
                logger.warning(f"     Proceeding anyway (tightening strategy)")
            
            logger.info("")
            logger.info(f"[TP REPLACEMENT] {pos.trade_id}")
            logger.info(f"  Old TP: {old_tp_price:.2f} ({old_tp_roi*100:.2f}%)")
            logger.info(f"  New TP: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")
            logger.info("")
            
            # ========================================
            # STEP 2: CANCEL OLD TP (1 API CALL)
            # ========================================
            logger.info(f"  [1/3] Cancelling old TP: {old_tp_id}")
            
            from order_manager import GlobalRateLimiter
            GlobalRateLimiter.wait()  # Rate limit protection
            
            cancel_resp = order_manager.api.cancel_order(
                order_id=old_tp_id,
                exchange=config.EXCHANGE,
            )
            
            if cancel_resp.get("error"):
                error_msg = str(cancel_resp.get("message", "Unknown error"))
                
                # Check if already cancelled/executed
                if "cannot be cancelled" in error_msg.lower() or "CANCELLED" in error_msg:
                    logger.warning(f"  ⚠️ Old TP already cancelled or executed: {error_msg}")
                elif "not found" in error_msg.lower():
                    logger.warning(f"  ⚠️ Old TP not found (may have triggered): {error_msg}")
                else:
                    logger.error(f"  ❌ TP cancel failed: {error_msg}")
                    logger.error(f"     ABORTING to keep existing TP protection")
                    logger.info("=" * 80)
                    return False
            else:
                logger.info(f"  ✓ Old TP cancelled successfully")
            
            time.sleep(1.0)  # Wait for cancellation to propagate
            
            # ========================================
            # STEP 3: PLACE NEW TP (1 API CALL)
            # ========================================
            logger.info(f"  [2/3] Placing new TP: {new_tp_price:.2f}")
            
            GlobalRateLimiter.wait()  # Rate limit protection
            
            tp_side = "SELL" if pos.side == "long" else "BUY"
            new_tp_order = order_manager.place_take_profit(
                side=tp_side,
                quantity=pos.quantity,
                trigger_price=new_tp_price,
            )
            
            if not new_tp_order or "order_id" not in new_tp_order:
                logger.error(f"  ❌ TP placement failed: {new_tp_order}")
                logger.error(f"     ⚠️ CRITICAL: Position now has NO TP protection!")
                logger.error(f"     Attempting to restore old TP...")
                
                # Emergency: Try to restore old TP
                try:
                    GlobalRateLimiter.wait()
                    restore_tp = order_manager.place_take_profit(
                        side=tp_side,
                        quantity=pos.quantity,
                        trigger_price=old_tp_price,
                    )
                    if restore_tp and "order_id" in restore_tp:
                        logger.info(f"     ✓ Old TP restored: {restore_tp['order_id']}")
                        pos.tp_order_id = restore_tp["order_id"]
                    else:
                        logger.error(f"     ❌ Could not restore old TP!")
                except Exception as restore_err:
                    logger.error(f"     ❌ Restore failed: {restore_err}")
                
                logger.info("=" * 80)
                return False
            
            new_tp_id = new_tp_order["order_id"]
            logger.info(f"  ✓ New TP placed: {new_tp_id}")
            
            # ========================================
            # STEP 4: VERIFY PLACEMENT (1 API CALL)
            # ========================================
            logger.info(f"  [3/3] Verifying new TP: {new_tp_id}")
            
            time.sleep(2.0)  # Wait for order to register
            GlobalRateLimiter.wait()  # Rate limit protection
            
            verify_resp = order_manager.get_order_status(new_tp_id)
            
            if not verify_resp:
                logger.error(f"  ❌ Could not verify new TP {new_tp_id}")
                logger.error(f"     Status check failed - manual verification required")
                logger.info("=" * 80)
                return False
            
            verify_status = str(verify_resp.get("status", "")).upper()
            
            if verify_status in ("CANCELLED", "REJECTED", "EXPIRED"):
                logger.error(f"  ❌ New TP rejected with status: {verify_status}")
                logger.error(f"     ⚠️ Position may have NO TP protection!")
                logger.info("=" * 80)
                return False
            
            if verify_status in ("PENDING", "NEW", "OPEN"):
                logger.info(f"  ✓ TP replacement verified (status: {verify_status})")
            else:
                logger.warning(f"  ⚠️ Unexpected TP status: {verify_status}")
            
            logger.info("")
            logger.info(f"  ✅ TP REPLACEMENT SUCCESSFUL")
            logger.info(f"     New TP ID: {new_tp_id}")
            logger.info(f"     Price: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")
            logger.info("=" * 80)
            
            # ========================================
            # STEP 5: UPDATE POSITION TRACKING
            # ========================================
            pos.tp_order_id = new_tp_id
            pos.tp_price = new_tp_price
            pos.tp_roi = new_tp_roi
            pos.current_tp_roi = new_tp_roi
            
            # ========================================
            # STEP 6: SEND TELEGRAM NOTIFICATION
            # ========================================
            try:
                from telegram_notifier import send_telegram_message
                msg = (
                    f"🎯 TP ADJUSTED #{pos.trade_id}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"  {pos.side.upper()} @ ${pos.entry_price:.2f}\n"
                    f"  Old TP: ${old_tp_price:.2f} ({old_tp_roi*100:.2f}%)\n"
                    f"  New TP: ${new_tp_price:.2f} ({new_tp_roi*100:.2f}%)\n"
                    f"  Distance: {distance:.2f} USDT ({distance/config.TICK_SIZE:.1f} ticks)\n"
                    f"  Status: ✅ ACTIVE"
                )
                send_telegram_message(msg)
            except Exception as e:
                logger.error(f"Error sending TP adjustment notification: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error replacing TP order: {e}", exc_info=True)
            logger.error(f"   ⚠️ Position protection status UNKNOWN - manual check required")
            return False


    def _move_sl_to_half_tp(self, order_manager, current_price: float) -> None:
        """
        Move SL to half TP price OR breakeven+fees, whichever is better.
        ATOMIC: Only executes ONCE per position with single-fire guard.
        """
        pos = self.current_position
        if pos is None or pos.tp_reduced:
            return
        
        # ✅ CRITICAL: Set flag IMMEDIATELY to prevent re-entry
        pos.tp_reduced = True

        half_tp_roi = pos.initial_tp_roi * config.HALF_TP_THRESHOLD
        direction = 1.0 if pos.side == "long" else -1.0
        
        # Calculate half TP price
        price_movement = pos.entry_price * half_tp_roi
        half_tp_sl_price = pos.entry_price + (price_movement * direction)
        
        # Calculate breakeven + fees price
        TAKER_FEE = 0.0006  # 0.06%
        GST = 1.18
        ROUND_TRIP_FEE = (TAKER_FEE * GST) * 2
        fee_buffer_price = pos.entry_price * ROUND_TRIP_FEE
        breakeven_sl_price = pos.entry_price + (fee_buffer_price * direction)
        
        # Use the better of the two (further from entry in profit direction)
        if pos.side == "long":
            new_sl_price = max(half_tp_sl_price, breakeven_sl_price)
        else:
            new_sl_price = min(half_tp_sl_price, breakeven_sl_price)
        
        old_sl = pos.sl_price

        logger.info("=" * 100)
        logger.info(f"[SL ADJUSTMENT] Half TP with fee protection")
        logger.info(f"  Half TP SL: {half_tp_sl_price:.2f}")
        logger.info(f"  Breakeven+Fees SL: {breakeven_sl_price:.2f}")
        logger.info(f"  Final SL (best): {new_sl_price:.2f}")
        logger.info(f"  Old SL: {old_sl:.2f}")

        # ✅ SINGLE API CALL: Use replace instead of cancel+place
        tp_side = "SELL" if pos.side == "long" else "BUY"
        
        try:
            # Check if replace_stop_loss method exists (more efficient)
            if hasattr(order_manager, 'replace_stop_loss'):
                new_sl_order = order_manager.replace_stop_loss(
                    existing_order_id=pos.sl_order_id,
                    side=tp_side,
                    quantity=pos.quantity,
                    new_trigger_price=new_sl_price,
                )
                logger.info(f"✓ SL replaced in single API call")
            else:
                # Fallback: Cancel + Place (2 API calls)
                logger.info("  Cancel old SL...")
                order_manager.cancel_order(pos.sl_order_id)
                time.sleep(0.3)  # Small delay for exchange processing
                
                logger.info("  Place new SL...")
                new_sl_order = order_manager.place_stop_loss(
                    side=tp_side,
                    quantity=pos.quantity,
                    trigger_price=new_sl_price,
                )
            
            if new_sl_order:
                pos.sl_order_id = new_sl_order.get("order_id", "")
                pos.sl_price = new_sl_price
                logger.info(f"✓ SL moved to half TP (fee-protected): {new_sl_price:.2f}")
                logger.info("=" * 100)

                try:
                    from telegram_notifier import format_sl_adjustment, send_telegram_message
                    msg = format_sl_adjustment(
                        trade_id=pos.trade_id,
                        old_sl=old_sl,
                        new_sl=new_sl_price,
                        reason=f"Half TP reached ({half_tp_roi*100:.2f}%) - fees covered",
                    )
                    send_telegram_message(msg)
                except Exception as e:
                    logger.error(f"Telegram notification failed: {e}")
            else:
                logger.error("Failed to place new SL - rolling back flag")
                pos.tp_reduced = False  # Rollback if failed
                
        except Exception as e:
            logger.error(f"SL adjustment failed: {e}")
            pos.tp_reduced = False  # Rollback on error


    # ======================================================================
    # HELPER METHODS (existing implementations)
    # ======================================================================

    def _tighten_sl_to_breakeven(
        self,
        order_manager,
        current_price: float
    ) -> None:
        """
        ✅ FIXED: Tighten SL to breakeven using proper calculation
        """
        pos = self.current_position
        if pos is None:
            return
        
        logger.info(f"[SL TIGHTENING] Moving SL to breakeven + fees")
        
        self._move_sl_to_breakeven(order_manager, current_price, pos.entry_price)

    def _get_regime_params(self, data_manager, current_price: float) -> Dict:
        """Get dynamic params based on vol regime."""
        try:
            atr_pct = data_manager.get_atr_percent()
        except:
            atr_pct = None

        try:
            regime = data_manager.get_vol_regime(atr_pct) if hasattr(data_manager, 'get_vol_regime') else "NEUTRAL"
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

    def _compute_signal_score(self, value: float, threshold: float) -> float:
        """Normalize signal to [0, 1] using CDF."""
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
        """Compute 5-signal weighted core score."""
        reasons = []

        imb_val = imbalance_data["imbalance"]
        if side == "long":
            imb_score = self._compute_signal_score(imb_val, config.IMBALANCE_THRESHOLD)
        else:
            imb_score = self._compute_signal_score(-imb_val, config.IMBALANCE_THRESHOLD)
        reasons.append(f"imb={imb_score:.3f}")

        if side == "long":
            wall_val = wall_data["bid_wall_strength"]
        else:
            wall_val = wall_data["ask_wall_strength"]
        wall_score = self._compute_signal_score(wall_val, wall_mult)
        reasons.append(f"wall={wall_score:.3f}")

        z_val = delta_data["z_score"]
        if side == "long":
            z_score = self._compute_signal_score(z_val, z_thresh)
        else:
            z_score = self._compute_signal_score(-z_val, z_thresh)
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
            imb_score * config.SCORE_IMB_WEIGHT +
            wall_score * config.SCORE_WALL_WEIGHT +
            z_score * config.SCORE_Z_WEIGHT +
            touch_score * config.SCORE_TOUCH_WEIGHT +
            trend_score * config.SCORE_TREND_WEIGHT
        )

        return core_score, reasons

    def _compute_aether_fusion_score(
        self,
        oracle_inputs: Optional[OracleInputs],
        oracle_outputs: Optional[OracleOutputs],
        side: str,
    ) -> Tuple[float, List[str]]:
        """Compute Aether fusion score."""
        if oracle_inputs is None or oracle_outputs is None:
            return 0.0, ["aether=MISSING"]

        reasons = []
        components = []

        if oracle_inputs.norm_cvd is not None:
            cvd_val = oracle_inputs.norm_cvd
            if side == "long":
                cvd_score = max(0.0, min(1.0, (cvd_val + 1.0) / 2.0))
            else:
                cvd_score = max(0.0, min(1.0, (-cvd_val + 1.0) / 2.0))
            components.append((cvd_score, config.AETHER_CVD_WEIGHT))
            reasons.append(f"cvd={cvd_score:.3f}")
        else:
            reasons.append("cvd=N/A")

        lv_vals = [oracle_inputs.lv_1m, oracle_inputs.lv_5m, oracle_inputs.lv_15m]
        lv_vals = [v for v in lv_vals if v is not None]
        if lv_vals:
            lv_avg = sum(lv_vals) / len(lv_vals)
            lv_score = min(1.0, lv_avg / 100.0)
            components.append((lv_score, config.AETHER_LV_WEIGHT))
            reasons.append(f"lv={lv_score:.3f}")
        else:
            reasons.append("lv=N/A")

        if oracle_inputs.hurst is not None:
            hurst_score = abs(oracle_inputs.hurst - 0.5) * 2.0
            components.append((hurst_score, config.AETHER_HURST_BOS_WEIGHT * 0.5))
            reasons.append(f"hurst={hurst_score:.3f}")

        if oracle_inputs.bos_align is not None:
            if side == "long":
                bos_score = oracle_inputs.bos_align
            else:
                bos_score = 1.0 - oracle_inputs.bos_align
            components.append((bos_score, config.AETHER_HURST_BOS_WEIGHT * 0.5))
            reasons.append(f"bos={bos_score:.3f}")
        else:
            reasons.append("bos=N/A")

        lstm_vals = [oracle_inputs.lstm_1m, oracle_inputs.lstm_5m, oracle_inputs.lstm_15m]
        lstm_vals = [v for v in lstm_vals if v is not None]
        if lstm_vals:
            lstm_avg = sum(lstm_vals) / len(lstm_vals)
            if side == "long":
                lstm_score = max(0.0, min(1.0, (lstm_avg + 1.0) / 2.0))
            else:
                lstm_score = max(0.0, min(1.0, (-lstm_avg + 1.0) / 2.0))
            components.append((lstm_score, config.AETHER_LSTM_WEIGHT))
            reasons.append(f"lstm={lstm_score:.3f}")
        else:
            reasons.append("lstm=N/A")

        if not components:
            return 0.0, reasons

        num = sum(v * w for v, w in components)
        den = sum(w for _, w in components)
        aether_score = num / den if den > 0 else 0.0

        return aether_score, reasons

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

    def _check_market_conditions(
        self,
        data_manager,
        side: str,
        current_price: float,
    ) -> Tuple[bool, bool, bool]:
        """Check if momentum, volatility, and trend are favorable."""
        momentum_favorable = False
        try:
            delta_data = self._compute_delta_z_score(data_manager)
            if delta_data:
                z_score = delta_data["z_score"]
                if side == "long" and z_score > 0:
                    momentum_favorable = True
                elif side == "short" and z_score < 0:
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

    def _exit_position(
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
        
        # ========================================
        # SMART TP/SL CLEANUP LOGIC
        # ========================================
        # ISSUE 3: Exchange auto-cancels the bracket (TP/SL) when either triggers.
        # So do NOT cancel anything on TP/SL hits.
        if reason in ("TP_HIT", "TP_PARTIAL", "SL_HIT"):
            logger.info("TP/SL triggered - exchange auto-cancels bracket orders - no cancellations needed")
        else:
            # Keep existing/manual exit behavior (cancel TP/SL) as-is below this point
            pass

        
        # ========================================
        # CHECK IF POSITION STILL OPEN
        # ========================================
        positions_resp = order_manager.api.get_positions(
            symbol=config.SYMBOL,
            exchange=config.EXCHANGE
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
        
        # ========================================
        # MARKET CLOSE IF NEEDED
        # ========================================
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
                            exit_order.get("order_id", ""),
                            timeout_sec=3.0
                        )
                        exit_price = order_manager.extract_fill_price(filled)
                    except Exception as e:
                        logger.warning(f"Market close fill check failed: {e}")
            except Exception as e:
                logger.error(f"Market close failed: {e}")
        else:
            logger.info("Position already closed (TP/SL triggered) - no market order needed")
        
        # ========================================
        # P&L CALCULATION & LOGGING
        # ========================================
        direction = 1.0 if pos.side == "long" else -1.0
        pnl = (exit_price - pos.entry_price) * direction * pos.quantity
        roi = pnl / pos.margin_used if pos.margin_used > 0 else 0.0
        
        risk_manager.update_trade_stats(pnl)
        
        logger.info("\n" + "=" * 100)
        logger.info(f"[POSITION CLOSED] {pos.trade_id} | {reason}")
        logger.info("=" * 100)
        logger.info(f"  Side: {pos.side.upper()}")
        logger.info(f"  Entry Session: {pos.entry_session}")
        logger.info(f"  Entry: {pos.entry_price:.2f} | Exit: {exit_price:.2f}")
        logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
        logger.info(f"  P&L: {pnl:.2f} USDT ({roi*100:.2f}%)")
        logger.info(f"  Hold Time: {(now_sec - pos.entry_time_sec)/60.0:.1f} min")
        logger.info("=" * 100 + "\n")
        
        # Telegram notification
        try:
            from telegram_notifier import format_exit_message, send_telegram_message
            msg = format_exit_message(
                trade_id=pos.trade_id,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                margin=pos.margin_used,
                pnl=pnl,
                roi=roi,
                hold_min=(now_sec - pos.entry_time_sec) / 60.0,
                exit_reason=reason,
                session=pos.entry_session,
            )
            send_telegram_message(msg)
        except Exception as e:
            logger.error(f"Error sending exit notification: {e}")
        
        # Excel logging
        if self.excel_logger:
            try:
                from datetime import datetime
                self.excel_logger.log_trade(
                    trade_id=pos.trade_id,
                    entry_time=datetime.fromtimestamp(pos.entry_time_sec).strftime("%Y-%m-%d %H:%M:%S"),
                    exit_time=datetime.fromtimestamp(now_sec).strftime("%Y-%m-%d %H:%M:%S"),
                    duration_minutes=(now_sec - pos.entry_time_sec) / 60.0,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    margin_used=pos.margin_used,
                    leverage=config.LEVERAGE,
                    tp_price=pos.tp_price,
                    sl_price=pos.sl_price,
                    entry_imbalance=pos.entry_imbalance,
                    entry_z_score=pos.entry_z_score,
                    entry_wall_volume=pos.entry_wall_volume,
                    exit_reason=reason,
                    pnl_usdt=pnl,
                    entry_htf_trend=pos.entry_htf_trend,
                )
            except Exception as e:
                logger.error(f"Error logging to Excel: {e}")
        
        self.last_exit_time_min = now_sec / 60.0
        self.current_position = None
        self.pending_entry = False

    def _clean_orphaned_orders(self, order_manager, symbol: str = "BTCUSDT") -> bool:
        """
        Cancel ALL open orders for symbol ONLY if no filled position exists.
        Returns True if cleanup was performed, False if skipped.
        """
        try:
            # Check if there's an active position
            positions_resp = order_manager.api.get_positions(symbol=symbol, exchange=config.EXCHANGE)
            if positions_resp and not positions_resp.get("error"):
                data = positions_resp.get("data", [])
                if isinstance(data, list):
                    for pos in data:
                        if pos.get("symbol") == symbol:
                            qty = float(pos.get("quantity", 0))
                            if abs(qty) > 0:
                                logger.warning(
                                    f"⚠️ Active position detected ({qty:.6f} BTC) - "
                                    f"SKIPPING orphaned order cleanup"
                                )
                                return False
            
            # No active position - safe to clean up
            logger.info(f"🧹 No active position - cleaning up orphaned orders for {symbol}")
            result = order_manager.api.cancel_all_orders(symbol=symbol, exchange=config.EXCHANGE)
            
            if result and not result.get("error"):
                logger.info(f"✓ All orphaned orders cancelled for {symbol}")
                return True
            else:
                logger.debug(f"Cancel all response: {result}")
                return False
        
        except Exception as e:
            logger.error(f"Error cleaning orphaned orders: {e}")
            return False
