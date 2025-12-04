"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version

DETAILED LOGGING VERSION:

- Logs raw orderbook/trade/touch counts
- Logs all calculated values (imbalance, wall strength, delta, Z-score, touch distances)
- Logs gate evaluation with actual thresholds
- Logs every minute regardless of data availability
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque
import logging
import math

import numpy as np

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
    tp_price: float  # full TP price (based on full PROFIT_TARGET_ROI)
    sl_price: float
    margin_used: float
    tp_order_id: str
    sl_order_id: str
    main_order_id: str
    main_filled: bool
    tp_reduced: bool  # has TP been switched to full already?
    entry_htf_trend: str  # "UP" / "DOWN" / "RANGE" / "UNKNOWN"
    regime_at_entry: str  # "LOW" / "NEUTRAL" / "HIGH"
    score_at_entry: float  # fused score at entry (for decay exits)


class ZScoreIcebergHunterStrategy:
    """
    Z-Score Imbalance Iceberg Hunter with detailed calculation logging.
    """

    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 120.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0  # Throttle TP/SL status polling
    ENTRY_FILL_TIMEOUT_SEC = 60.0  # Cancel bracket if main not filled in 60s

    # Score-decay exit
    SCORE_DECAY_EXIT_THRESHOLD = 0.50

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0

        # Store recent deltas to compute population Z-score
        self._delta_population: deque = deque(maxlen=3000)

        # Last time we printed a full decision snapshot
        self._last_decision_log_sec: float = 0.0

        # Last time we printed a position snapshot (open position)
        self._last_position_log_sec: float = 0.0

        # Last time TP/SL order statuses were checked
        self._last_status_check_sec: float = 0.0

        # Aether Oracle instance
        self._oracle = AetherOracle()

        # 15-minute performance report state (Telegram)
        self._last_report_sec: float = 0.0
        self._last_report_total_trades: int = 0

        logger.info("=" * 80)
        logger.info("Z-SCORE IMBALANCE ICEBERG HUNTER STRATEGY INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Imbalance Threshold = {config.IMBALANCE_THRESHOLD:.2f}")
        logger.info(f"Wall Volume Mult (base) = {config.MIN_WALL_VOLUME_MULT:.2f}×")
        logger.info(f"Delta Z Threshold (base) = {config.DELTA_Z_THRESHOLD:.2f}")
        logger.info(f"Zone Ticks = ±{config.ZONE_TICKS}")
        logger.info(
            f"Touch Threshold = {config.PRICE_TOUCH_THRESHOLD_TICKS} ticks"
        )
        logger.info(f"Profit Target ROI (base) = {config.PROFIT_TARGET_ROI * 100:.2f}%")
        logger.info(f"Stop Loss ROI (base) = {config.STOP_LOSS_ROI * 100:.2f}%")
        logger.info(f"Max Hold Minutes = {config.MAX_HOLD_MINUTES}")
        logger.info(
            f"Trend Filter: EMA{config.EMA_PERIOD} "
            f"(long if price>EMA, short if price<EMA)"
        )
        logger.info("Volatility regime logic enabled: "
                    f"{getattr(config, 'ENABLE_VOL_REGIME_LOGIC', False)}")

    # ======================================================================
    # Session helpers (existing)
    # ======================================================================

    @staticmethod
    def _get_session_label() -> Tuple[str, bool, bool]:
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        weekday = now_utc.weekday()
        is_weekend = weekday >= 5

        asia_start, asia_end = config.ASIA_SESSION_UTC
        london_start, london_end = config.LONDON_SESSION_UTC
        ny_start, ny_end = config.NEW_YORK_SESSION_UTC

        if asia_start <= hour < asia_end:
            return "ASIA", True, is_weekend
        if london_start <= hour < london_end:
            return "LONDON", True, is_weekend
        if ny_start <= hour < ny_end:
            return "NEW_YORK", True, is_weekend
        return "OFF_SESSION", False, is_weekend

    def _get_dynamic_params(self, data_manager, current_price, now_sec):
        """
        Decide slippage_ticks, profit_roi, stop_roi based on:

        - Session (Asia/London/New York/off-session)
        - Weekend flag
        - Current ATR% from DataManager

        Returns dict with values + a short text reason string.

        NOTE: regime-specific TP/SL multipliers are applied *on top* of
        these base values when computing TP/SL for a concrete entry.
        """
        # Defaults: original config values
        slippage_ticks = config.SLIPPAGE_TICKS_ASSUMED
        profit_roi = config.PROFIT_TARGET_ROI
        stop_roi = config.STOP_LOSS_ROI
        reason_parts = []

        if not getattr(config, "ENABLE_SESSION_DYNAMIC_PARAMS", False):
            return {
                "slippage_ticks": slippage_ticks,
                "profit_roi": profit_roi,
                "stop_roi": stop_roi,
                "atr_pct": None,
                "session": "DISABLED",
                "is_major": False,
                "is_weekend": False,
                "reason": "session_dynamics_disabled",
            }

        session, is_major, is_weekend = self._get_session_label()

        # ATR as fraction of price (e.g. 0.008 == 0.8%)
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception:
            atr_pct = None

        # Major sessions: allow standard 10%/3% RR with 1–2 ticks slippage
        if is_major and not is_weekend:
            slippage_ticks = config.SESSION_SLIPPAGE_TICKS
            profit_roi = config.SESSION_PROFIT_TARGET_ROI
            stop_roi = config.SESSION_STOP_LOSS_ROI
            reason_parts.append("major_session_standard_rr")
        else:
            # Off-session or weekend: adjust by ATR
            reason_parts.append("off_session_or_weekend")

            if atr_pct is not None:
                if atr_pct >= config.MIN_ATR_PCT_FOR_FULL_TP:
                    # Enough realized volatility: keep full TP but tighten slippage a bit
                    slippage_ticks = config.OFFSESSION_SLIPPAGE_TICKS_BASE
                    profit_roi = config.OFFSESSION_FULL_TP_ROI
                    reason_parts.append("atr_ok_fullish_tp")
                elif atr_pct >= config.VERY_LOW_ATR_PCT:
                    # Low vol: near TP, still 1 tick slippage
                    slippage_ticks = config.OFFSESSION_SLIPPAGE_TICKS_BASE
                    profit_roi = config.OFFSESSION_NEAR_TP_ROI
                    reason_parts.append("low_atr_near_tp")
                else:
                    # Very low vol: minimal slippage and near TP
                    slippage_ticks = config.OFFSESSION_SLIPPAGE_TICKS_LOW_VOL
                    profit_roi = config.OFFSESSION_NEAR_TP_ROI
                    reason_parts.append("very_low_atr_min_slip_near_tp")
            else:
                # No ATR available; stay conservative off-session
                slippage_ticks = config.OFFSESSION_SLIPPAGE_TICKS_BASE
                profit_roi = config.OFFSESSION_NEAR_TP_ROI
                reason_parts.append("no_atr_fallback_near_tp")

        return {
            "slippage_ticks": slippage_ticks,
            "profit_roi": profit_roi,
            "stop_roi": stop_roi,
            "atr_pct": atr_pct,
            "session": session,
            "is_major": is_major,
            "is_weekend": is_weekend,
            "reason": ",".join(reason_parts),
        }

    # ======================================================================
    # Volatility-regime helpers
    # ======================================================================

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Numerically stable approximation of standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _get_regime_and_atr(self, data_manager) -> Tuple[str, Optional[float]]:
        """
        Wrapper around DataManager's regime classifier.
        """
        if not getattr(config, "ENABLE_VOL_REGIME_LOGIC", False):
            return "NEUTRAL", None
        try:
            regime, atr_pct = data_manager.get_vol_regime()
        except Exception:
            # Hard fallback to neutral if anything goes wrong – never crash the loop
            regime, atr_pct = "NEUTRAL", None
        return regime, atr_pct

    def _get_regime_scaled_z_threshold(self, atr_pct: Optional[float]) -> float:
        """
        Compute dynamic Z threshold based on ATR% using the formula:

        z_dynamic = base + 0.3 * (atr_pct - 0.0015) / 0.0015,
        clamped into [DELTA_Z_THRESHOLD_LOW, DELTA_Z_THRESHOLD_HIGH].

        If atr_pct is missing, returns base threshold.
        """
        base = config.DELTA_Z_THRESHOLD
        low = getattr(config, "DELTA_Z_THRESHOLD_LOW", 1.8)
        high = getattr(config, "DELTA_Z_THRESHOLD_HIGH", 2.3)

        if atr_pct is None:
            return base

        try:
            z = base + 0.3 * (atr_pct - config.ATR_LOW_THRESHOLD) / max(
                config.ATR_LOW_THRESHOLD, 1e-6
            )
        except Exception:
            return base

        return max(low, min(high, z))

    def _get_regime_wall_mult(self, regime: str) -> float:
        """
        Effective wall multiplier for the current regime:

        wall_mult_eff = MIN_WALL_VOLUME_MULT * REGIME_WALL_MULT[regime]
        """
        reg_table = getattr(config, "REGIME_WALL_MULT", None)
        if not reg_table or regime not in reg_table:
            return float(config.MIN_WALL_VOLUME_MULT)
        multiplier = float(reg_table[regime])
        return float(config.MIN_WALL_VOLUME_MULT) * multiplier

    def _get_regime_tp_sl(
        self,
        base_profit_roi: float,
        base_stop_roi: float,
        regime: str,
    ) -> Tuple[float, float]:
        """
        Apply regime-specific multipliers on top of base TP/SL ROI:

        - HIGH: TP +40%, SL -10%;
        - LOW/NEUTRAL: TP +10%, SL -3%.

        base_* come from session dynamics; this function does not alter
        core config.PROFIT_TARGET_ROI / STOP_LOSS_ROI.
        """
        tp_mult_table = getattr(config, "REGIME_TP_MULT", {})
        sl_mult_table = getattr(config, "REGIME_SL_MULT", {})

        tp_mult = float(tp_mult_table.get(regime, 1.0))
        sl_mult = float(sl_mult_table.get(regime, 1.0))

        profit_roi = base_profit_roi * tp_mult
        # STOP_LOSS_ROI is negative; multiplying by <1.0 tightens it
        stop_roi = base_stop_roi * sl_mult
        return profit_roi, stop_roi

    # ======================================================================
    # Main tick handler
    # ======================================================================

    def on_tick(
        self, data_manager, order_manager, risk_manager
    ) -> None:
        """
        Main per-tick strategy entrypoint called from the main loop.

        - Manages existing position (TP/SL + timeout + TP adjustment).
        - If flat, evaluates entry gates and possibly opens a new bracket.
        """
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            # 15-minute performance Telegram report (non-blocking, stats-only)
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

            # Core metrics
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(
                    data_manager, current_price, imbalance_data
                )
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            # Higher timeframe trend (5min EMA-based robust trend)
            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(
                    f"Error fetching HTF trend in on_tick: {e}", exc_info=True
                )

            # LTF trend (no longer hard gate; used only by oracle/meta)
            ltf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_ltf_trend"):
                    ltf_trend = data_manager.get_ltf_trend()
            except Exception as e:
                logger.error(
                    f"Error fetching LTF (1m) trend in on_tick: {e}",
                    exc_info=True,
                )

            # Build Aether Oracle inputs
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
                    ltf_trend=ltf_trend,
                )
                oracle_outputs = self._oracle.decide(
                    inputs=oracle_inputs,
                    risk_manager=risk_manager,
                )
            except Exception as e:
                logger.error(f"Error building Oracle inputs/outputs: {e}", exc_info=True)
                oracle_inputs = None
                oracle_outputs = None

            # Excel logging when all core metrics present
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
                    reason="All core metrics computed",
                    htf_trend=htf_trend,
                )

            # Decision + entries
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
                ltf_trend=ltf_trend,
                now_sec=now_sec,
                oracle_inputs=oracle_inputs,
                oracle_outputs=oracle_outputs,
            )
        except Exception as e:
            logger.error(f"Error in ZScore on_tick: {e}", exc_info=True)

    # ======================================================================
    # 15-minute Telegram performance report
    # ======================================================================

    def _maybe_send_15m_report(
        self, now_sec: float, risk_manager, current_price: float
    ) -> None:
        """
        Send a rich 15-minute performance report to Telegram:

        - Trades taken, wins, losses, win rate
        - Realized P&L
        - Open-position snapshot (if any)

        Does not affect trading logic.
        """
        # Initialize timer on first tick
        if self._last_report_sec == 0.0:
            self._last_report_sec = now_sec
            self._last_report_total_trades = int(
                getattr(risk_manager, "total_trades", 0)
            )
            return

        if now_sec - self._last_report_sec < 900.0:
            return

        self._last_report_sec = now_sec

        total_trades = int(getattr(risk_manager, "total_trades", 0))
        winning_trades = int(getattr(risk_manager, "winning_trades", 0))
        losing_trades = int(
            getattr(risk_manager, "losing_trades", max(0, total_trades - winning_trades))
        )
        realized_pnl = float(getattr(risk_manager, "realized_pnl", 0.0))

        trades_since = max(0, total_trades - self._last_report_total_trades)
        self._last_report_total_trades = total_trades

        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100.0
        else:
            win_rate = 0.0

        # Open position snapshot (approx unrealized P&L)
        if self.current_position is not None:
            pos = self.current_position
            direction = 1.0 if pos.side == "long" else -1.0
            u_pnl = (current_price - pos.entry_price) * direction * pos.quantity
            pos_summary = (
                f"{pos.side.upper()} {pos.quantity:.6f} BTC @ {pos.entry_price:.2f}, "
                f"cur={current_price:.2f}, uPnL≈{u_pnl:.2f} USDT"
            )
        else:
            pos_summary = "None"

        msg_lines = [
            "📊 Z-Score 15m Performance Report",
            f"Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Symbol: {config.SYMBOL}",
            f"Current price: {current_price:.2f}",
            "",
            f"Total trades: {total_trades}",
            f"Wins / Losses: {winning_trades} / {losing_trades}",
            f"Win rate: {win_rate:.2f}%",
            f"Realized P&L: {realized_pnl:.2f} USDT",
            f"Trades since last report: {trades_since}",
            "",
            f"Open position: {pos_summary}",
        ]

        try:
            send_telegram_message("\n".join(msg_lines))
        except Exception as e:
            logger.error(f"Failed to send 15m Telegram report: {e}", exc_info=True)

    # ======================================================================
    # Detailed decision logging
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
        long_reasons_block: List[str],
        short_reasons_block: List[str],
        htf_trend: Optional[str],
        ltf_trend: Optional[str],
        oracle_inputs: Optional[OracleInputs],
        oracle_outputs: Optional[OracleOutputs],
        regime: str,
        atr_pct: Optional[float],
    ) -> None:
        if now_sec - self._last_decision_log_sec < self.DECISION_LOG_INTERVAL_SEC:
            return

        self._last_decision_log_sec = now_sec

        logger.info("-" * 80)
        logger.info(
            "Z-DECISION SNAPSHOT @ {} | price={:.2f}".format(
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                current_price,
            )
        )

        # DataManager stats
        stats = data_manager.stats
        logger.info(
            " DataManager Stats: ob_updates={}, trades={}, candles={}, prices={}".format(
                stats.get("orderbook_updates", 0),
                stats.get("trades_received", 0),
                stats.get("candles_received", 0),
                stats.get("prices_recorded", 0),
            )
        )

        bids, asks = data_manager.get_orderbook_snapshot()
        logger.info(
            " Raw Orderbook: bid_levels={}, ask_levels={}".format(
                len(bids), len(asks)
            )
        )

        trades_20s = data_manager.get_recent_trades(window_seconds=20)
        logger.info(f" Raw Trades (20s window): count={len(trades_20s)}")

        # Imbalance
        if imbalance_data is None:
            logger.info(" Imbalance       : MISSING (no book or <20 levels)")
        else:
            logger.info(
                " Imbalance       : {:.3f} (total_bid={:.2f}, total_ask={:.2f})".format(
                    imbalance_data["imbalance"],
                    imbalance_data["total_bid"],
                    imbalance_data["total_ask"],
                )
            )
            logger.info(
                " → long_ok={} (need ≥{:.2f}), short_ok={} (need ≤{:.2f})".format(
                    imbalance_data["long_ok"],
                    config.IMBALANCE_THRESHOLD,
                    imbalance_data["short_ok"],
                    -config.IMBALANCE_THRESHOLD,
                )
            )

        # Wall strength
        if wall_data is None:
            logger.info(" Wall Strength   : MISSING (depends on imbalance/orderbook)")
        else:
            logger.info(
                " Wall Strength   : bid={:.2f} (vol={:.2f}), ask={:.2f} (vol={:.2f})".format(
                    wall_data["bid_wall_strength"],
                    wall_data["bid_vol_zone"],
                    wall_data["ask_wall_strength"],
                    wall_data["ask_vol_zone"],
                )
            )

        # Delta Z-score
        if delta_data is None:
            logger.info(
                f" Delta Z-score   : MISSING (no trades in last {config.DELTA_WINDOW_SEC}s)"
            )
        else:
            logger.info(
                " Delta Z-score   : delta={:.2f} (buy={:.2f}, sell={:.2f}), z={:.2f}".format(
                    delta_data["delta"],
                    delta_data["buy_vol"],
                    delta_data["sell_vol"],
                    delta_data["z_score"],
                )
            )
            logger.info(
                " → pop_size={}, long_z_ok={}, short_z_ok={}".format(
                    len(self._delta_population),
                    delta_data["long_ok"],
                    delta_data["short_ok"],
                )
            )

        # Touch distances
        if touch_data is None:
            logger.info(" Touch ticks     : MISSING (no orderbook snapshot)")
        else:
            logger.info(
                " Touch ticks     : bid_dist={:.2f} (nearest_bid={:.2f}), "
                "ask_dist={:.2f} (nearest_ask={:.2f})".format(
                    touch_data["bid_distance_ticks"],
                    touch_data["nearest_bid"],
                    touch_data["ask_distance_ticks"],
                    touch_data["nearest_ask"],
                )
            )

        # EMA snapshot
        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception as e:
            logger.error(f"Error fetching EMA for decision log: {e}", exc_info=True)

        if ema_val is None:
            logger.info(
                f" Trend EMA{config.EMA_PERIOD} : MISSING (not enough price history)"
            )
        else:
            long_trend_ok = current_price > ema_val
            short_trend_ok = current_price < ema_val
            logger.info(
                f" Trend EMA{config.EMA_PERIOD} : ema={ema_val:.2f}, "
                f"price={current_price:.2f}"
            )
            logger.info(
                f" → long_trend_ok={long_trend_ok}, short_trend_ok={short_trend_ok}"
            )

        # HTF / LTF trend
        if htf_trend is None:
            logger.info(
                f" HTF Trend {config.HTF_TREND_INTERVAL}min : MISSING (insufficient data)"
            )
        else:
            logger.info(f" HTF Trend {config.HTF_TREND_INTERVAL}min : {htf_trend}")

        if ltf_trend is None:
            logger.info(" LTF Trend 1min : MISSING (insufficient data)")
        else:
            logger.info(f" LTF Trend 1min : {ltf_trend}")

        # Session + volatility snapshot
        try:
            dyn = self._get_dynamic_params(data_manager, current_price, now_sec)
            logger.info(
                " Session/Vol     : session=%s major=%s weekend=%s atr_pct=%s "
                "slip_ticks=%s base_profit_roi=%.4f base_stop_roi=%.4f reason=%s",
                dyn["session"],
                dyn["is_major"],
                dyn["is_weekend"],
                f"{dyn['atr_pct']:.4f}" if dyn["atr_pct"] is not None else "None",
                dyn["slippage_ticks"],
                dyn["profit_roi"],
                dyn["stop_roi"],
                dyn["reason"],
            )
        except Exception as e:
            logger.error(f"Error logging session/vol snapshot: {e}", exc_info=True)

        # Regime snapshot
        logger.info(
            " Vol Regime      : regime=%s atr_pct=%s",
            regime,
            f"{atr_pct*100:.3f}%" if atr_pct is not None else "None",
        )

        # Oracle inputs/outputs
        if oracle_inputs is not None:
            logger.info(" AETHER ORACLE INPUTS:")
            logger.info(
                " LV 1m={:.4f}, LV 5m={:.4f}, LV 15m={:.4f}, micro_trap={}".format(
                    oracle_inputs.lv_1m
                    if oracle_inputs.lv_1m is not None
                    else float("nan"),
                    oracle_inputs.lv_5m
                    if oracle_inputs.lv_5m is not None
                    else float("nan"),
                    oracle_inputs.lv_15m
                    if oracle_inputs.lv_15m is not None
                    else float("nan"),
                    oracle_inputs.micro_trap,
                )
            )
            logger.info(
                " norm_cvd={:.4f} hurst={:.4f} bos_align={}".format(
                    oracle_inputs.norm_cvd
                    if oracle_inputs.norm_cvd is not None
                    else float("nan"),
                    oracle_inputs.hurst
                    if oracle_inputs.hurst is not None
                    else float("nan"),
                    "MISSING"
                    if oracle_inputs.bos_align is None
                    else f"{oracle_inputs.bos_align:.4f}",
                )
            )

        if oracle_outputs is not None:
            ls = oracle_outputs.long_scores
            ss = oracle_outputs.short_scores
            logger.info(
                " AETHER ORACLE LONG : mc={} bayes={} rl={} fused={} "
                "kelly_f={:.4f} rr={:.2f}".format(
                    "MISSING" if ls.mc is None else f"{ls.mc:.3f}",
                    "MISSING" if ls.bayes is None else f"{ls.bayes:.3f}",
                    "MISSING" if ls.rl is None else f"{ls.rl:.3f}",
                    "MISSING" if ls.fused is None else f"{ls.fused:.3f}",
                    ls.kelly_f,
                    ls.rr,
                )
            )
            logger.info(
                " AETHER ORACLE SHORT: mc={} bayes={} rl={} fused={} "
                "kelly_f={:.4f} rr={:.2f}".format(
                    "MISSING" if ss.mc is None else f"{ss.mc:.3f}",
                    "MISSING" if ss.bayes is None else f"{ss.bayes:.3f}",
                    "MISSING" if ss.rl is None else f"{ss.rl:.3f}",
                    "MISSING" if ss.fused is None else f"{ss.fused:.3f}",
                    ss.kelly_f,
                    ss.rr,
                )
            )

        # Final readiness
        if long_ready:
            logger.info(" LONG ENTRY STATE  : READY (score gauntlet + oracle passed)")
        else:
            logger.info(
                " LONG ENTRY STATE  : BLOCKED | " + " | ".join(long_reasons_block)
            )

        if short_ready:
            logger.info(" SHORT ENTRY STATE : READY (score gauntlet + oracle passed)")
        else:
            logger.info(
                " SHORT ENTRY STATE : BLOCKED | " + " | ".join(short_reasons_block)
            )

        logger.info("-" * 80)

    # ======================================================================
    # Core metric calculators
    # ======================================================================

    def _compute_imbalance(self, data_manager) -> Optional[Dict]:
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        depth = min(config.WALL_DEPTH_LEVELS, len(bids), len(asks))
        if depth < 20:
            return None

        total_bid = sum(q for (_, q) in bids[:depth])
        total_ask = sum(q for (_, q) in asks[:depth])

        if total_bid + total_ask <= 0:
            return None

        imbalance = (total_bid - total_ask) / (total_bid + total_ask)

        return {
            "imbalance": imbalance,
            "long_ok": imbalance >= config.IMBALANCE_THRESHOLD,
            "short_ok": imbalance <= -config.IMBALANCE_THRESHOLD,
            "total_bid": total_bid,
            "total_ask": total_ask,
        }

    def _compute_wall_strength(
        self,
        data_manager,
        current_price: float,
        imbalance_data: Optional[Dict],
    ) -> Optional[Dict]:
        if imbalance_data is None:
            return None

        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        zone_low = current_price - (config.TICK_SIZE * config.ZONE_TICKS)
        zone_high = current_price + (config.TICK_SIZE * config.ZONE_TICKS)

        bid_vol_zone = sum(q for (p, q) in bids if zone_low <= p <= zone_high)
        ask_vol_zone = sum(q for (p, q) in asks if zone_low <= p <= zone_high)

        total_bid = imbalance_data["total_bid"]
        total_ask = imbalance_data["total_ask"]

        avg_bid = total_bid / config.WALL_DEPTH_LEVELS if total_bid > 0 else 1e-8
        avg_ask = total_ask / config.WALL_DEPTH_LEVELS if total_ask > 0 else 1e-8

        bid_wall_strength = bid_vol_zone / avg_bid
        ask_wall_strength = ask_vol_zone / avg_ask

        return {
            "zone_low": zone_low,
            "zone_high": zone_high,
            "bid_vol_zone": bid_vol_zone,
            "ask_vol_zone": ask_vol_zone,
            "bid_wall_strength": bid_wall_strength,
            "ask_wall_strength": ask_wall_strength,
            # `long_wall_ok` / `short_wall_ok` are now evaluated against
            # regime-scaled thresholds inside _score_gauntlet.
        }

    def _compute_delta_z_score(self, data_manager) -> Optional[Dict]:
        """
        Delta Z-score over the last DELTA_WINDOW_SEC using ONLY real trades.

        - Uses all available historical deltas in _delta_population.
        - Z-pop guard: for pop_len < 50, sigma = max(0.3, std(pop)).
        - Returns None only if there are literally no trades in the window.
        """
        trades = data_manager.get_recent_trades(
            window_seconds=config.DELTA_WINDOW_SEC
        )
        if not trades:
            return None

        buy_vol = 0.0
        sell_vol = 0.0

        for t in trades:
            qty = float(t.get("qty", 0.0))
            if qty <= 0:
                continue

            if not t.get("isBuyerMaker", False):
                buy_vol += qty
            else:
                sell_vol += qty

        delta = buy_vol - sell_vol
        self._delta_population.append(delta)

        pop = list(self._delta_population)
        pop_len = len(pop)
        mean_delta = sum(pop) / pop_len

        if pop_len < 50:
            sigma = float(np.std(pop, ddof=0))
            sigma = max(0.3, sigma)
        else:
            variance = sum((x - mean_delta) ** 2 for x in pop) / pop_len
            sigma = variance ** 0.5 if variance > 0 else 1.0

        z_score = (delta - mean_delta) / sigma if sigma > 0 else 0.0

        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "delta": delta,
            "z_score": z_score,
        }

    def _compute_price_touch(
        self, data_manager, current_price: float
    ) -> Optional[Dict]:
        """
        Compute touch distances in ticks between current price and nearest bid/ask.

        Uses absolute distance so that both:
        - bid far below current price
        - bid mistakenly above current price (stale / crossed book)
        are treated as 'far' and will block the touch gate when |dist| > threshold.
        """
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        nearest_bid = bids[0][0] if bids else current_price
        nearest_ask = asks[0][0] if asks else current_price

        bid_distance_ticks = abs((current_price - nearest_bid) / config.TICK_SIZE)
        ask_distance_ticks = abs((nearest_ask - current_price) / config.TICK_SIZE)

        return {
            "nearest_bid": nearest_bid,
            "nearest_ask": nearest_ask,
            "bid_distance_ticks": bid_distance_ticks,
            "ask_distance_ticks": ask_distance_ticks,
        }

    # ======================================================================
    # Weighted score gauntlet + entries
    # ======================================================================

    def _score_gauntlet(
        self,
        *,
        side: str,
        current_price: float,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        htf_trend: Optional[str],
        regime: str,
        atr_pct: Optional[float],
    ) -> Tuple[float, List[str]]:
        """
        Probabilistic score gauntlet.

        Signals and weights:
        - Imbalance (25%)
        - Wall strength (20%)
        - Delta Z (30%)
        - Touch / proximity (10%)
        - Trend (HTF + EMA20) (15%)

        Each component is normalized via a pseudo-CDF around its regime-aware threshold.
        Returns: total_score in [0,1], plus list of string diagnostics.
        """
        side_sign = 1.0 if side == "long" else -1.0

        # Regime-aware thresholds
        z_thresh = self._get_regime_scaled_z_threshold(atr_pct)
        wall_mult_eff = self._get_regime_wall_mult(regime)
        touch_thresh = float(config.PRICE_TOUCH_THRESHOLD_TICKS)

        # -----------------
        # Imbalance score
        # -----------------
        imb_val = float(imbalance_data["imbalance"])
        imb_thresh = float(config.IMBALANCE_THRESHOLD) * side_sign
        imb_std = 0.10  # heuristic scale
        imb_z = (imb_val - imb_thresh) / imb_std
        imb_score = self._norm_cdf(imb_z)

        # -----------------
        # Wall strength
        # -----------------
        wall_strength = (
            wall_data["bid_wall_strength"]
            if side == "long"
            else wall_data["ask_wall_strength"]
        )
        wall_thresh = wall_mult_eff
        wall_std = max(wall_thresh * 0.25, 1e-3)
        wall_z = (wall_strength - wall_thresh) / wall_std
        wall_score = self._norm_cdf(wall_z)

        # -----------------
        # Delta Z
        # -----------------
        raw_z = float(delta_data["z_score"]) * side_sign
        z_std = 1.0
        z_norm = (raw_z - z_thresh) / z_std
        z_score = self._norm_cdf(z_norm)

        # -----------------
        # Touch proximity
        # -----------------
        dist = (
            touch_data["bid_distance_ticks"]
            if side == "long"
            else touch_data["ask_distance_ticks"]
        )
        # smaller distance is better; invert
        touch_std = max(touch_thresh * 0.5, 1e-3)
        touch_norm = (touch_thresh - dist) / touch_std
        touch_score = self._norm_cdf(touch_norm)

        # -----------------
        # Trend: HTF + EMA20
        # -----------------
        # HTF label
        def _norm_dir(label: Optional[str]) -> str:
            if not label:
                return "UNKNOWN"
            l = label.upper()
            if l in ("UP", "UPTREND"):
                return "UP"
            if l in ("DOWN", "DOWNTREND"):
                return "DOWN"
            if l in ("RANGEBOUND", "RANGE", "FLAT"):
                return "RANGE"
            return "UNKNOWN"

        htf_dir = _norm_dir(htf_trend)

        ema_val = None
        try:
            # EMA from DataManager directly for current snapshot
            ema_val = None
        except Exception:
            ema_val = None

        ema_score = 0.5
        if ema_val is not None and ema_val > 0:
            if side == "long":
                ema_score = 0.75 if current_price > ema_val else 0.25
            else:
                ema_score = 0.75 if current_price < ema_val else 0.25

        # HTF trend directional factor
        htf_score = 0.5
        if side == "long":
            if htf_dir == "UP":
                htf_score = 0.80
            elif htf_dir == "RANGE":
                htf_score = 0.60
            elif htf_dir == "DOWN":
                htf_score = 0.20
        else:
            if htf_dir == "DOWN":
                htf_score = 0.80
            elif htf_dir == "RANGE":
                htf_score = 0.60
            elif htf_dir == "UP":
                htf_score = 0.20

        trend_score = 0.6 * htf_score + 0.4 * ema_score

        # -----------------
        # Weighted sum
        # -----------------
        weights = {
            "imb": 0.25,
            "wall": 0.20,
            "z": 0.30,
            "touch": 0.10,
            "trend": 0.15,
        }

        total_score = (
            weights["imb"] * imb_score
            + weights["wall"] * wall_score
            + weights["z"] * z_score
            + weights["touch"] * touch_score
            + weights["trend"] * trend_score
        )

        diag = [
            f"side={side}",
            f"regime={regime}",
            f"atr_pct={atr_pct:.4f}" if atr_pct is not None else "atr_pct=None",
            f"imb={imb_val:.3f}, imb_thresh={imb_thresh:.3f}, imb_score={imb_score:.3f}",
            f"wall={wall_strength:.2f}, wall_eff={wall_thresh:.2f}, wall_score={wall_score:.3f}",
            f"z_raw={raw_z:.2f}, z_thresh={z_thresh:.2f}, z_score={z_score:.3f}",
            f"touch_dist={dist:.2f}, touch_thresh={touch_thresh:.2f}, touch_score={touch_score:.3f}",
            f"trend_score={trend_score:.3f} (htf_dir={htf_dir})",
            f"total_score={total_score:.3f}",
        ]

        return total_score, diag

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
        ltf_trend: Optional[str],
        now_sec: float,
        oracle_inputs: Optional[OracleInputs] = None,
        oracle_outputs: Optional[OracleOutputs] = None,
    ) -> None:
        missing: List[str] = []
        if imbalance_data is None:
            missing.append("imbalance")
        if wall_data is None:
            missing.append("wall")
        if delta_data is None:
            missing.append("delta")
        if touch_data is None:
            missing.append("touch")

        regime, atr_pct = self._get_regime_and_atr(data_manager)

        if missing:
            long_ready = False
            short_ready = False
            long_reasons_block = [f"missing_{m}" for m in missing]
            short_reasons_block = [f"missing_{m}" for m in missing]

            self._log_decision_state(
                now_sec=now_sec,
                current_price=current_price,
                data_manager=data_manager,
                imbalance_data=imbalance_data,
                wall_data=wall_data,
                delta_data=delta_data,
                touch_data=touch_data,
                long_ready=long_ready,
                short_ready=short_ready,
                long_reasons_block=long_reasons_block,
                short_reasons_block=short_reasons_block,
                htf_trend=htf_trend,
                ltf_trend=ltf_trend,
                oracle_inputs=oracle_inputs,
                oracle_outputs=oracle_outputs,
                regime=regime,
                atr_pct=atr_pct,
            )
            return

        # Score gauntlet per side
        long_score, long_diag = self._score_gauntlet(
            side="long",
            current_price=current_price,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            htf_trend=htf_trend,
            regime=regime,
            atr_pct=atr_pct,
        )
        short_score, short_diag = self._score_gauntlet(
            side="short",
            current_price=current_price,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            htf_trend=htf_trend,
            regime=regime,
            atr_pct=atr_pct,
        )

        score_threshold = 0.75
        long_ready = long_score >= score_threshold
        short_ready = short_score >= score_threshold

        long_reasons_block: List[str] = []
        short_reasons_block: List[str] = []

        if not long_ready:
            long_reasons_block.append(f"score_long={long_score:.3f}<thr={score_threshold}")
        if not short_ready:
            short_reasons_block.append(
                f"score_short={short_score:.3f}<thr={score_threshold}"
            )

        # Oracle fusion veto / overlay (win-prob overlay)
        if oracle_outputs is not None:
            ls: OracleSideScores = oracle_outputs.long_scores
            ss: OracleSideScores = oracle_outputs.short_scores

            # Win-prob overlay:
            # win_p = 0.4 + 0.2*lstm + 0.2*z_sign + 0.1*cvd + 0.1*lv
            # implemented distributed inside AetherOracle, here we just
            # check fused score vs threshold.
            win_thr = 0.60

            if ls.fused is None or ls.fused < win_thr:
                long_ready = False
                long_reasons_block.append(
                    f"oracle_long_fused={ls.fused if ls.fused is not None else 'None'}<win_thr={win_thr}"
                )
            if ss.fused is None or ss.fused < win_thr:
                short_ready = False
                short_reasons_block.append(
                    f"oracle_short_fused={ss.fused if ss.fused is not None else 'None'}<win_thr={win_thr}"
                )
        else:
            long_reasons_block.append("oracle_disabled")
            short_reasons_block.append("oracle_disabled")

        # Final logging
        self._log_decision_state(
            now_sec=now_sec,
            current_price=current_price,
            data_manager=data_manager,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            long_ready=long_ready,
            short_ready=short_ready,
            long_reasons_block=long_reasons_block + long_diag,
            short_reasons_block=short_reasons_block + short_diag,
            htf_trend=htf_trend,
            ltf_trend=ltf_trend,
            oracle_inputs=oracle_inputs,
            oracle_outputs=oracle_outputs,
            regime=regime,
            atr_pct=atr_pct,
        )

        # Take entries (one side at a time; prefer stronger score)
        if not long_ready and not short_ready:
            return

        if long_ready and short_ready:
            # choose side with larger fused score; fall back to gauntlet score
            chosen_side = "long" if long_score >= short_score else "short"
        elif long_ready:
            chosen_side = "long"
        else:
            chosen_side = "short"

        fused_score_at_entry = 0.0
        if oracle_outputs is not None:
            fused_score_at_entry = (
                oracle_outputs.long_scores.fused
                if chosen_side == "long"
                else oracle_outputs.short_scores.fused
            ) or 0.0

        self._enter_position(
            side=chosen_side,
            current_price=current_price,
            data_manager=data_manager,
            order_manager=order_manager,
            risk_manager=risk_manager,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            htf_trend=htf_trend,
            regime=regime,
            fused_score=fused_score_at_entry,
            now_sec=now_sec,
        )

    # ======================================================================
    # Position entry / management
    # ======================================================================

    def _enter_position(
        self,
        *,
        side: str,
        current_price: float,
        data_manager,
        order_manager,
        risk_manager,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        htf_trend: Optional[str],
        regime: str,
        fused_score: float,
        now_sec: float,
    ) -> None:
        """
        Create bracket order with TP/SL derived from:

        - Base TP/SL from session dynamics
        - Regime-aware TP/SL multipliers
        - Existing TP/SL price math in OrderManager (unchanged)
        """
        base_dyn = self._get_dynamic_params(
            data_manager=data_manager,
            current_price=current_price,
            now_sec=now_sec,
        )
        base_profit_roi = base_dyn["profit_roi"]
        base_stop_roi = base_dyn["stop_roi"]

        profit_roi, stop_roi = self._get_regime_tp_sl(
            base_profit_roi=base_profit_roi,
            base_stop_roi=base_stop_roi,
            regime=regime,
        )

        direction = 1.0 if side == "long" else -1.0

        # TP and SL prices – TP/SL percentage logic is unchanged in
        # OrderManager/RiskManager; here only ROI inputs are different.
        tp_roi_input = profit_roi
        sl_roi_input = stop_roi

        margin_info = risk_manager.calculate_margin_for_entry(
            current_price=current_price,
            side=side,
            desired_roi_tp=tp_roi_input,
            desired_roi_sl=sl_roi_input,
            regime=regime,
        )
        if margin_info is None:
            logger.debug("Margin calculation blocked entry")
            return

        qty = margin_info["quantity"]
        tp_price = margin_info["tp_price"]
        sl_price = margin_info["sl_price"]
        margin_used = margin_info["margin_used"]

        self.trade_seq += 1
        trade_id = f"Z{int(time.time())}-{self.trade_seq}"

        order_ids = order_manager.place_bracket_order(
            trade_id=trade_id,
            side=side,
            quantity=qty,
            entry_price=current_price,
            tp_price=tp_price,
            sl_price=sl_price,
            slippage_ticks=base_dyn["slippage_ticks"],
        )
        if order_ids is None:
            logger.error("Bracket order placement failed")
            return

        main_id, tp_id, sl_id = order_ids

        pos = ZScorePosition(
            trade_id=trade_id,
            side=side,
            quantity=qty,
            entry_price=current_price,
            entry_time_sec=now_sec,
            entry_wall_volume=(
                wall_data["bid_vol_zone"]
                if side == "long"
                else wall_data["ask_vol_zone"]
            ),
            wall_zone_low=wall_data["zone_low"],
            wall_zone_high=wall_data["zone_high"],
            entry_imbalance=float(imbalance_data["imbalance"]),
            entry_z_score=float(delta_data["z_score"]),
            tp_price=tp_price,
            sl_price=sl_price,
            margin_used=margin_used,
            tp_order_id=tp_id,
            sl_order_id=sl_id,
            main_order_id=main_id,
            main_filled=False,
            tp_reduced=False,
            entry_htf_trend=htf_trend or "UNKNOWN",
            regime_at_entry=regime,
            score_at_entry=fused_score,
        )
        self.current_position = pos

        logger.info(
            f"ENTER {side.upper()} trade_id={trade_id} qty={qty:.6f} "
            f"entry={current_price:.2f} tp={tp_price:.2f} sl={sl_price:.2f} "
            f"regime={regime} fused_score={fused_score:.3f} "
            f"base_tp={base_profit_roi:.4f} base_sl={base_stop_roi:.4f} "
            f"eff_tp={profit_roi:.4f} eff_sl={stop_roi:.4f}"
        )

        try:
            send_telegram_message(
                f"✅ ENTER {side.upper()} {qty:.6f} @ {current_price:.2f}\n"
                f"TP={tp_price:.2f}, SL={sl_price:.2f}, Regime={regime}, "
                f"Score={fused_score:.3f}"
            )
        except Exception:
            pass

    def _manage_open_position(
        self,
        *,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        Manage an existing position:

        - TP/SL status polling
        - Time stop
        - Dynamic TP tightening (existing)
        - High-vol trailing stop after +10% profit
        - Score-decay exit when fused score deteriorates (<0.5)
        """
        pos = self.current_position
        if pos is None:
            return

        # Position snapshot logging
        if now_sec - self._last_position_log_sec > self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            logger.info(
                f"OPEN POSITION SNAPSHOT: {pos.side.upper()} qty={pos.quantity:.6f} "
                f"entry={pos.entry_price:.2f} tp={pos.tp_price:.2f} "
                f"sl={pos.sl_price:.2f} cur={current_price:.2f}"
            )

        # Poll TP/SL order status with throttling
        if now_sec - self._last_status_check_sec > self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            filled, exit_side, exit_price, exit_reason = order_manager.check_position_status(
                pos
            )
            if filled:
                self._handle_position_closed(
                    risk_manager=risk_manager,
                    exit_side=exit_side,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    now_sec=now_sec,
                )
                return

        # Time stop
        elapsed_min = (now_sec - pos.entry_time_sec) / 60.0
        if elapsed_min >= config.MAX_HOLD_MINUTES:
            logger.info(
                f"Time stop triggered after {elapsed_min:.1f}min; closing {pos.trade_id}"
            )
            order_manager.close_position_market(pos)
            self._handle_position_closed(
                risk_manager=risk_manager,
                exit_side="time_stop",
                exit_price=current_price,
                exit_reason="MAX_HOLD",
                now_sec=now_sec,
            )
            return

        # High-vol trailing stop: after +10% profit, move SL to BE+0.05% in HIGH regime
        regime, atr_pct = self._get_regime_and_atr(data_manager)
        if regime == "HIGH":
            direction = 1.0 if pos.side == "long" else -1.0
            pnl = (current_price - pos.entry_price) * direction
            roi = pnl * pos.quantity / max(pos.margin_used, 1e-8)

            if roi >= 0.10:  # +10% profit
                be_price = pos.entry_price * (1.0 + direction * 0.0005)
                if (pos.side == "long" and be_price > pos.sl_price) or (
                    pos.side == "short" and be_price < pos.sl_price
                ):
                    updated = order_manager.update_stop_loss(
                        pos=pos, new_sl_price=be_price
                    )
                    if updated:
                        logger.info(
                            f"TRAIL SL (HIGH VOL): moved SL to BE+0.05% at {be_price:.2f}"
                        )
                        pos.sl_price = be_price

        # Score-decay exit: re-score and flatten if <0.5
        try:
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(
                    data_manager, current_price, imbalance_data
                )
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)
        except Exception as e:
            logger.error(f"Error recomputing metrics for decay exit: {e}", exc_info=True)
            return

        if (
            imbalance_data is None
            or wall_data is None
            or delta_data is None
            or touch_data is None
        ):
            return

        cur_score, _ = self._score_gauntlet(
            side=pos.side,
            current_price=current_price,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            htf_trend=pos.entry_htf_trend,
            regime=regime,
            atr_pct=atr_pct,
        )

        if cur_score < self.SCORE_DECAY_EXIT_THRESHOLD:
            logger.info(
                f"SCORE DECAY EXIT: trade_id={pos.trade_id} "
                f"current_score={cur_score:.3f} < {self.SCORE_DECAY_EXIT_THRESHOLD:.3f}"
            )
            order_manager.close_position_market(pos)
            self._handle_position_closed(
                risk_manager=risk_manager,
                exit_side="score_decay",
                exit_price=current_price,
                exit_reason="SCORE_DECAY",
                now_sec=now_sec,
            )

    def _handle_position_closed(
        self,
        *,
        risk_manager,
        exit_side: str,
        exit_price: float,
        exit_reason: str,
        now_sec: float,
    ) -> None:
        pos = self.current_position
        if pos is None:
            return

        self.current_position = None
        self.last_exit_time_min = now_sec / 60.0

        pnl = risk_manager.update_after_trade_close(
            position=pos,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

        logger.info(
            f"EXIT {pos.side.upper()} trade_id={pos.trade_id} "
            f"exit_price={exit_price:.2f} reason={exit_reason} pnl={pnl:.2f}"
        )

        try:
            send_telegram_message(
                f"❌ EXIT {pos.side.upper()} {pos.quantity:.6f} @ {exit_price:.2f}\n"
                f"Reason={exit_reason}, PnL={pnl:.2f} USDT"
            )
        except Exception:
            pass
    