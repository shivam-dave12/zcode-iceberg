"""
strategy.py

Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version
(DETAILED LOGGING VERSION)

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

import numpy as np
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
    entry_zscore: float
    tp_price: float  # full TP price based on full PROFIT_TARGET_ROI
    sl_price: float
    margin_used: float
    tp_order_id: str
    sl_order_id: str
    main_order_id: str
    main_filled: bool
    tp_reduced: bool  # has TP been switched to full already?
    entry_htf_trend: str  # "UP", "DOWN", "RANGE", "UNKNOWN"


class ZScoreIcebergHunterStrategy:
    """
    Z-Score Imbalance Iceberg Hunter with detailed calculation logging.
    """

    DECISION_LOG_INTERVAL_SEC = 60.0
    POSITION_LOG_INTERVAL_SEC = 120.0
    ORDER_STATUS_CHECK_INTERVAL_SEC = 10.0  # Throttle TP/SL status polling
    ENTRY_FILL_TIMEOUT_SEC = 60.0  # Cancel bracket if main not filled in 60s

    def __init__(self, excel_logger: Optional[ZScoreExcelLogger] = None) -> None:
        self.current_position: Optional[ZScorePosition] = None
        self.last_exit_time_min: float = 0.0
        self.excel_logger = excel_logger
        self.trade_seq = 0

        # Store recent deltas to compute population Z-score
        self.delta_population = deque(maxlen=3000)

        # Last time we printed a full decision snapshot
        self.last_decision_log_sec: float = 0.0

        # Last time we printed a position snapshot (open position)
        self.last_position_log_sec: float = 0.0

        # Last time TP/SL order statuses were checked
        self.last_status_check_sec: float = 0.0

        # Aether Oracle instance
        self.oracle = AetherOracle()

        # 15-minute performance report state (Telegram)
        self.last_report_sec: float = 0.0
        self.last_report_total_trades: int = 0

        logger.info("=" * 80)
        logger.info("Z-SCORE IMBALANCE ICEBERG HUNTER STRATEGY INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Imbalance Threshold = {config.IMBALANCE_THRESHOLD:.2f}")
        logger.info(f"Wall Volume Mult = {config.MIN_WALL_VOLUME_MULT:.2f}×")
        logger.info(f"Delta Z Threshold = {config.DELTA_Z_THRESHOLD:.2f}")
        logger.info(f"Zone Ticks = ±{config.ZONE_TICKS}")
        logger.info(f"Touch Threshold = {config.PRICE_TOUCH_THRESHOLD_TICKS} ticks")
        logger.info(f"Profit Target ROI = {config.PROFIT_TARGET_ROI * 100:.2f}%")
        logger.info(f"Stop Loss ROI = {config.STOP_LOSS_ROI * 100:.2f}%")
        logger.info(f"Max Hold Minutes = {config.MAX_HOLD_MINUTES}")
        logger.info(
            f"Trend Filter: EMA{config.EMA_PERIOD} "
            f"(long if price>EMA, short if price<EMA)"
        )
        logger.info("=" * 80)

    def get_session_label(
        self, now_utc=None
    ) -> Tuple[str, bool, bool]:
        """
        Return ("ASIA"/"LONDON"/"NEWYORK"/"OFFSESSION", is_major_session: bool, is_weekend: bool)
        based on UTC time and weekday.
        """
        if now_utc is None:
            now_utc = datetime.utcnow()
        hour = now_utc.hour
        weekday = now_utc.weekday()  # 0=Mon ... 6=Sun
        is_weekend = weekday >= 5

        asia_start, asia_end = config.ASIA_SESSION_UTC
        london_start, london_end = config.LONDON_SESSION_UTC
        ny_start, ny_end = config.NEWYORK_SESSION_UTC

        if asia_start <= hour < asia_end:
            return ("ASIA", True, is_weekend)
        if london_start <= hour < london_end:
            return ("LONDON", True, is_weekend)
        if ny_start <= hour < ny_end:
            return ("NEWYORK", True, is_weekend)
        return ("OFFSESSION", False, is_weekend)

    def get_dynamic_params(self, data_manager, current_price, now_sec):
        """
        Decide slippage_ticks, profit_roi, stop_roi based on:
         - Session (Asia/London/New York/off-session)
         - Weekend flag
         - Current ATR from DataManager

        Returns dict with values + a short text reason string.
        """
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

        session, is_major, is_weekend = self.get_session_label()

        # Defaults = original config values
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception:
            atr_pct = None  # ATR as fraction of price (e.g. 0.008 = 0.8%)

        if is_major and not is_weekend:
            # Major sessions allow standard 10:3 RR with 1–2 ticks slippage
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
                # No ATR available: stay conservative off-session
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

    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """
        Main per-tick strategy entrypoint called from the main loop.
         - Manages existing position (TP/SL, timeout, TP adjustment).
         - If flat, evaluates entry gates and possibly opens a new bracket.
        """
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            # 15-minute performance Telegram report (non-blocking, stats-only)
            self.maybe_send_15m_report(now_sec, risk_manager, current_price)

            # Manage open position first
            if self.current_position is not None:
                self.manage_open_position(
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
            imbalance_data = self.compute_imbalance(data_manager)
            wall_data = (
                self.compute_wall_strength(data_manager, current_price, imbalance_data)
                if imbalance_data is not None
                else None
            )
            delta_data = self.compute_delta_zscore(data_manager)
            touch_data = self.compute_price_touch(data_manager, current_price)

            # Higher timeframe trend (5min EMA-based robust trend)
            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(f"Error fetching HTF trend in on_tick: {e}", exc_info=True)

            # 1-minute trend (same robust 3-state logic as HTF)
            ltf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_ltf_trend"):
                    ltf_trend = data_manager.get_ltf_trend()
            except Exception as e:
                logger.error(
                    f"Error fetching LTF (1m) trend in on_tick: {e}", exc_info=True
                )

            # Build Aether Oracle inputs
            oracle_inputs: Optional[OracleInputs] = None
            try:
                oracle_inputs = self.oracle.build_inputs(
                    data_manager=data_manager,
                    current_price=current_price,
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    htf_trend=htf_trend,
                    ltf_trend=ltf_trend,
                )
            except Exception as e:
                logger.error(f"Error building Oracle inputs: {e}", exc_info=True)
                oracle_inputs = None

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

            # Decision & entries
            self.try_entries_and_log(
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
            )

        except Exception as e:
            logger.error(f"Error in ZScore on_tick: {e}", exc_info=True)

    def maybe_send_15m_report(
        self, now_sec: float, risk_manager, current_price: float
    ) -> None:
        """
        Send a rich 15-minute performance report to Telegram:
         - Trades taken, wins, losses, win rate
         - Realized P/L
         - Open-position snapshot if any

        Does not affect trading logic.
        """
        # Initialize timer on first tick
        if self.last_report_sec == 0.0:
            self.last_report_sec = now_sec
            self.last_report_total_trades = int(getattr(risk_manager, "total_trades", 0))
            return

        if now_sec - self.last_report_sec < 900.0:
            return

        self.last_report_sec = now_sec

        total_trades = int(getattr(risk_manager, "total_trades", 0))
        winning_trades = int(getattr(risk_manager, "winning_trades", 0))
        losing_trades = int(
            getattr(risk_manager, "losing_trades", max(0, total_trades - winning_trades))
        )
        realized_pnl = float(getattr(risk_manager, "realized_pnl", 0.0))

        trades_since = max(0, total_trades - self.last_report_total_trades)
        self.last_report_total_trades = total_trades

        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100.0
        else:
            win_rate = 0.0

        # Open position snapshot (approx unrealized P/L)
        if self.current_position is not None:
            pos = self.current_position
            direction = 1.0 if pos.side == "long" else -1.0
            upnl = (current_price - pos.entry_price) * direction * pos.quantity
            pos_summary = (
                f"{pos.side.upper()} {pos.quantity:.6f} BTC @ {pos.entry_price:.2f}, "
                f"cur={current_price:.2f}, uPnL≈{upnl:.2f} USDT"
            )
        else:
            pos_summary = None

        msg_lines = [
            "📊 Z-Score 15m Performance Report",
            f"Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Symbol: {config.SYMBOL}",
            f"Current price: {current_price:.2f}",
            "",
            f"Total trades: {total_trades}",
            f"Wins / Losses: {winning_trades} / {losing_trades}",
            f"Win rate: {win_rate:.2f}%",
            f"Realized P/L: {realized_pnl:.2f} USDT",
            f"Trades since last report: {trades_since}",
            "",
            f"Open position: {pos_summary}",
        ]

        try:
            send_telegram_message("\n".join(msg_lines))
        except Exception as e:
            logger.error(f"Failed to send 15m Telegram report: {e}", exc_info=True)

    def log_decision_state(
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
    ) -> None:
        if now_sec - self.last_decision_log_sec < self.DECISION_LOG_INTERVAL_SEC:
            return
        self.last_decision_log_sec = now_sec

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
            "  DataManager Stats: ob_updates={}, trades={}, candles={}, prices={}".format(
                stats.get("orderbook_updates", 0),
                stats.get("trades_received", 0),
                stats.get("candles_received", 0),
                stats.get("prices_recorded", 0),
            )
        )

        bids, asks = data_manager.get_orderbook_snapshot()
        logger.info(f"  Raw Orderbook: bid_levels={len(bids)}, ask_levels={len(asks)}")

        trades_20s = data_manager.get_recent_trades(window_seconds=20)
        logger.info(f"  Raw Trades (20s window): count={len(trades_20s)}")

        # Imbalance
        if imbalance_data is None:
            logger.info("  Imbalance       : MISSING (no book or <20 levels)")
        else:
            logger.info(
                "  Imbalance       : {:.3f} (total_bid={:.2f}, total_ask={:.2f})".format(
                    imbalance_data["imbalance"],
                    imbalance_data["total_bid"],
                    imbalance_data["total_ask"],
                )
            )
            logger.info(
                "    → long_ok={} (need ≥{:.2f}), short_ok={} (need ≤{:.2f})".format(
                    imbalance_data["long_ok"],
                    config.IMBALANCE_THRESHOLD,
                    imbalance_data["short_ok"],
                    -config.IMBALANCE_THRESHOLD,
                )
            )

        # Wall strength
        if wall_data is None:
            logger.info(
                "  Wall Strength   : MISSING (depends on imbalance/orderbook)"
            )
        else:
            logger.info(
                "  Wall Strength   : bid={:.2f} (vol={:.2f}), ask={:.2f} (vol={:.2f})".format(
                    wall_data["bid_wall_strength"],
                    wall_data["bid_vol_zone"],
                    wall_data["ask_wall_strength"],
                    wall_data["ask_vol_zone"],
                )
            )
            logger.info(
                "    → long_wall_ok={} (need ≥{:.2f}), short_wall_ok={} (need ≥{:.2f})".format(
                    wall_data["long_wall_ok"],
                    config.MIN_WALL_VOLUME_MULT,
                    wall_data["short_wall_ok"],
                    config.MIN_WALL_VOLUME_MULT,
                )
            )

        # Delta Z-score
        if delta_data is None:
            logger.info(
                f"  Delta Z-score   : MISSING (no trades in last {config.DELTA_WINDOW_SEC}s)"
            )
        else:
            logger.info(
                "  Delta Z-score   : delta={:.2f} (buy={:.2f}, sell={:.2f}), z={:.2f}".format(
                    delta_data["delta"],
                    delta_data["buy_vol"],
                    delta_data["sell_vol"],
                    delta_data["zscore"],
                )
            )
            logger.info(
                "    → pop_size={}, long_z_ok={} (need ≥{:.2f}), short_z_ok={} (need ≤{:.2f})".format(
                    len(self.delta_population),
                    delta_data["long_ok"],
                    config.DELTA_Z_THRESHOLD,
                    delta_data["short_ok"],
                    -config.DELTA_Z_THRESHOLD,
                )
            )

        # Touch distances
        if touch_data is None:
            logger.info("  Touch ticks     : MISSING (no orderbook snapshot)")
        else:
            logger.info(
                "  Touch ticks     : bid_dist={:.2f} (nearest_bid={:.2f}), "
                "ask_dist={:.2f} (nearest_ask={:.2f})".format(
                    touch_data["bid_distance_ticks"],
                    touch_data["nearest_bid"],
                    touch_data["ask_distance_ticks"],
                    touch_data["nearest_ask"],
                )
            )
            logger.info(
                "    → long_touch_ok={} (need ≤{} ticks), short_touch_ok={} (need ≤{} ticks)".format(
                    touch_data["long_touch_ok"],
                    config.PRICE_TOUCH_THRESHOLD_TICKS,
                    touch_data["short_touch_ok"],
                    config.PRICE_TOUCH_THRESHOLD_TICKS,
                )
            )

        # EMA snapshot
        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception as e:
            logger.error(f"Error fetching EMA for decision log: {e}", exc_info=True)

        if ema_val is None:
            logger.info(f"  Trend EMA{config.EMA_PERIOD}   : MISSING (not enough price history)")
        else:
            long_trend_ok = current_price > ema_val
            short_trend_ok = current_price < ema_val
            logger.info(
                f"  Trend EMA{config.EMA_PERIOD}   : ema={ema_val:.2f}, price={current_price:.2f}"
            )
            logger.info(
                f"    → long_trend_ok={long_trend_ok}, short_trend_ok={short_trend_ok}"
            )

        # HTF / LTF trend
        if htf_trend is None:
            logger.info(
                f"  HTF Trend {config.HTF_TREND_INTERVAL}min : MISSING (insufficient data)"
            )
        else:
            logger.info(f"  HTF Trend {config.HTF_TREND_INTERVAL}min : {htf_trend}")

        if ltf_trend is None:
            logger.info("  LTF Trend 1min  : MISSING (insufficient data)")
        else:
            logger.info(f"  LTF Trend 1min  : {ltf_trend}")

        # Session/volatility snapshot (NEW)
        try:
            dyn = self.get_dynamic_params(data_manager, current_price, now_sec)
            logger.info(
                "  Session/Vol: session={} major={} weekend={} atr_pct={} slip_ticks={} profit_roi={:.4f} stop_roi={:.4f} reason={}".format(
                    dyn["session"],
                    dyn["is_major"],
                    dyn["is_weekend"],
                    f"{dyn['atr_pct']:.4f}" if dyn["atr_pct"] is not None else "None",
                    dyn["slippage_ticks"],
                    dyn["profit_roi"],
                    dyn["stop_roi"],
                    dyn["reason"],
                )
            )
        except Exception as e:
            logger.error(f"Error logging session/vol snapshot: {e}", exc_info=True)

        # Oracle inputs
        if oracle_inputs is not None:
            logger.info("  AETHER ORACLE INPUTS:")
            logger.info(
                "    LV 1m={:.4f}, LV 5m={:.4f}, LV 15m={:.4f}, micro_trap={}".format(
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
                "    norm_cvd={:.4f} hurst={:.4f} bos_align={}".format(
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
            logger.info(
                "    ema_val={:.2f} atr_pct={}".format(
                    oracle_inputs.ema_val
                    if oracle_inputs.ema_val is not None
                    else float("nan"),
                    "MISSING"
                    if oracle_inputs.atr_pct is None
                    else f"{oracle_inputs.atr_pct * 100:.2f}%",
                )
            )

        if oracle_outputs is not None:
            ls = oracle_outputs.long_scores
            ss = oracle_outputs.short_scores

            logger.info(
                "  AETHER ORACLE LONG : mc={} bayes={} rl={} fused={} kelly_f={:.4f} rr={:.2f}".format(
                    "MISSING" if ls.mc is None else f"{ls.mc:.3f}",
                    "MISSING" if ls.bayes is None else f"{ls.bayes:.3f}",
                    "MISSING" if ls.rl is None else f"{ls.rl:.3f}",
                    "MISSING" if ls.fused is None else f"{ls.fused:.3f}",
                    ls.kelly_f,
                    ls.rr,
                )
            )
            logger.info(
                "  AETHER ORACLE SHORT: mc={} bayes={} rl={} fused={} kelly_f={:.4f} rr={:.2f}".format(
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
            logger.info("  LONG ENTRY STATE  : READY (all gates + oracle passed)")
        else:
            logger.info(
                "  LONG ENTRY STATE  : BLOCKED | " + " | ".join(long_reasons_block)
            )

        if short_ready:
            logger.info("  SHORT ENTRY STATE : READY (all gates + oracle passed)")
        else:
            logger.info(
                "  SHORT ENTRY STATE : BLOCKED | " + " | ".join(short_reasons_block)
            )

        logger.info("-" * 80)

    def compute_imbalance(self, data_manager) -> Optional[Dict]:
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None
        depth = min(config.WALL_DEPTH_LEVELS, len(bids), len(asks))
        if depth < 20:
            return None

        total_bid = sum(q for (_, q) in bids[:depth])
        total_ask = sum(q for (_, q) in asks[:depth])
        if total_bid + total_ask == 0:
            return None

        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        return {
            "imbalance": imbalance,
            "long_ok": imbalance >= config.IMBALANCE_THRESHOLD,
            "short_ok": imbalance <= -config.IMBALANCE_THRESHOLD,
            "total_bid": total_bid,
            "total_ask": total_ask,
        }

    def compute_wall_strength(
        self, data_manager, current_price: float, imbalance_data: Optional[Dict]
    ) -> Optional[Dict]:
        if imbalance_data is None:
            return None

        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        zone_low = current_price - config.TICKSIZE * config.ZONE_TICKS
        zone_high = current_price + config.TICKSIZE * config.ZONE_TICKS

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
            "long_wall_ok": bid_wall_strength >= config.MIN_WALL_VOLUME_MULT,
            "short_wall_ok": ask_wall_strength >= config.MIN_WALL_VOLUME_MULT,
        }

    def compute_delta_zscore(self, data_manager) -> Optional[Dict]:
        """
        Delta Z-score over the last DELTA_WINDOW_SEC using ONLY real trades.
         - Uses all available historical deltas in delta_population.
         - Z-pop guard: for pop_len < 50, sigma = max(0.3, std(pop)).
         - Returns None only if there are literally no trades in the window.
        """
        trades = data_manager.get_recent_trades(window_seconds=config.DELTA_WINDOW_SEC)
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
        self.delta_population.append(delta)

        pop = list(self.delta_population)
        pop_len = len(pop)
        mean_delta = sum(pop) / pop_len

        if pop_len < 50:
            sigma = float(np.std(pop, ddof=0))
            sigma = max(0.3, sigma)
        else:
            variance = sum((x - mean_delta) ** 2 for x in pop) / pop_len
            sigma = variance**0.5 if variance > 0 else 1.0

        zscore = (delta - mean_delta) / sigma if sigma > 0 else 0.0

        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "delta": delta,
            "zscore": zscore,
            "long_ok": zscore >= config.DELTA_Z_THRESHOLD,
            "short_ok": zscore <= -config.DELTA_Z_THRESHOLD,
        }

    def compute_price_touch(
        self, data_manager, current_price: float
    ) -> Optional[Dict]:
        """
        Compute touch distances in ticks between current price and nearest bid/ask.
        Uses absolute distance so that both:
         - bid far below current price
         - bid mistakenly above current price (stale / crossed book)
        are treated as "far" and will block the touch gate when dist > threshold.
        """
        bids, asks = data_manager.get_orderbook_snapshot()
        if not bids or not asks:
            return None

        nearest_bid = bids[0][0] if bids else current_price
        nearest_ask = asks[0][0] if asks else current_price

        bid_distance_ticks = abs(current_price - nearest_bid) / config.TICKSIZE
        ask_distance_ticks = abs(nearest_ask - current_price) / config.TICKSIZE

        return {
            "nearest_bid": nearest_bid,
            "nearest_ask": nearest_ask,
            "bid_distance_ticks": bid_distance_ticks,
            "ask_distance_ticks": ask_distance_ticks,
            "long_touch_ok": bid_distance_ticks <= config.PRICE_TOUCH_THRESHOLD_TICKS,
            "short_touch_ok": ask_distance_ticks <= config.PRICE_TOUCH_THRESHOLD_TICKS,
        }

    def try_entries_and_log(
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

        if missing:
            long_ready = False
            short_ready = False
            long_reasons_block = [f"missing={m}" for m in missing]
            short_reasons_block = [f"missing={m}" for m in missing]

            self.log_decision_state(
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
                oracle_outputs=None,
            )
            return

        # Core orderbook / flow conditions
        long_conds: Dict[str, bool] = {
            "imbalance": imbalance_data["long_ok"],
            "wall": wall_data["long_wall_ok"],
            "delta_z": delta_data["long_ok"],
            "touch": touch_data["long_touch_ok"],
        }
        short_conds: Dict[str, bool] = {
            "imbalance": imbalance_data["short_ok"],
            "wall": wall_data["short_wall_ok"],
            "delta_z": delta_data["short_ok"],
            "touch": touch_data["short_touch_ok"],
        }

        # 1m EMA trend filter (legacy)
        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception as e:
            logger.error(f"Error fetching EMA for entry gates: {e}", exc_info=True)

        if ema_val is None:
            trend_long_ok = False
            trend_short_ok = False
        else:
            trend_long_ok = current_price > ema_val
            trend_short_ok = current_price < ema_val

        long_conds["ema_trend"] = trend_long_ok
        short_conds["ema_trend"] = trend_short_ok

        # ------------------------------------------------------------------
        # Multi-timeframe trend sync logic
        # ------------------------------------------------------------------
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
        ltf_dir = _norm_dir(ltf_trend)

        multi_long_ok = False
        multi_short_ok = False

        if htf_dir == "UP":
            if ltf_dir == "UP":
                multi_long_ok = True
            elif ltf_dir == "RANGE":
                multi_long_ok = True  # Allow longs when LTF is range during HTF uptrend
        elif htf_dir == "DOWN":
            if ltf_dir == "DOWN":
                multi_short_ok = True
            elif ltf_dir == "RANGE":
                multi_short_ok = True  # Allow shorts when LTF is range during HTF downtrend
        elif htf_dir == "RANGE":
            # HTF rangebound: trade in direction of 1m trend or both if LTF RANGE
            if ltf_dir == "UP":
                multi_long_ok = True
            elif ltf_dir == "DOWN":
                multi_short_ok = True
            elif ltf_dir == "RANGE":
                multi_long_ok = True
                multi_short_ok = True
        else:
            # UNKNOWN HTF: block both (conservative)
            multi_long_ok = False
            multi_short_ok = False

        long_conds["mtf_trend"] = multi_long_ok
        short_conds["mtf_trend"] = multi_short_ok

        long_ready = all(long_conds.values())
        short_ready = all(short_conds.values())

        long_reasons_block: List[str] = []
        short_reasons_block: List[str] = []

        # Reasons for blocking (core)
        if not long_conds["imbalance"]:
            long_reasons_block.append(
                f"imbalance={imbalance_data['imbalance']:.3f} "
                f"< {config.IMBALANCE_THRESHOLD:.2f}"
            )
        if not long_conds["wall"]:
            long_reasons_block.append(
                f"bid_wall={wall_data['bid_wall_strength']:.2f} "
                f"< {config.MIN_WALL_VOLUME_MULT:.2f}"
            )
        if not long_conds["delta_z"]:
            long_reasons_block.append(
                f"z={delta_data['zscore']:.2f} < {config.DELTA_Z_THRESHOLD:.2f}"
            )
        if not long_conds["touch"]:
            long_reasons_block.append(
                f"bid_dist={touch_data['bid_distance_ticks']:.1f} "
                f"> {config.PRICE_TOUCH_THRESHOLD_TICKS}"
            )

        if not short_conds["imbalance"]:
            short_reasons_block.append(
                f"imbalance={imbalance_data['imbalance']:.3f} "
                f"> {-config.IMBALANCE_THRESHOLD:.2f}"
            )
        if not short_conds["wall"]:
            short_reasons_block.append(
                f"ask_wall={wall_data['ask_wall_strength']:.2f} "
                f"< {config.MIN_WALL_VOLUME_MULT:.2f}"
            )
        if not short_conds["delta_z"]:
            short_reasons_block.append(
                f"z={delta_data['zscore']:.2f} > {-config.DELTA_Z_THRESHOLD:.2f}"
            )
        if not short_conds["touch"]:
            short_reasons_block.append(
                f"ask_dist={touch_data['ask_distance_ticks']:.1f} "
                f"> {config.PRICE_TOUCH_THRESHOLD_TICKS}"
            )

        # EMA trend reasons
        if not long_conds["ema_trend"]:
            if ema_val is None:
                long_reasons_block.append("trend=ema_missing")
            else:
                long_reasons_block.append(
                    f"trend price={current_price:.2f} "
                    f"<= EMA{config.EMA_PERIOD}={ema_val:.2f}"
                )

        if not short_conds["ema_trend"]:
            if ema_val is None:
                short_reasons_block.append("trend=ema_missing")
            else:
                short_reasons_block.append(
                    f"trend price={current_price:.2f} "
                    f">= EMA{config.EMA_PERIOD}={ema_val:.2f}"
                )

        # MTF trend sync reasons
        if not long_conds["mtf_trend"]:
            if htf_trend is None or ltf_trend is None:
                long_reasons_block.append("mtf_trend_sync=missing")
            else:
                long_reasons_block.append(
                    f"mtf_trend_sync htf={htf_trend}, ltf={ltf_trend} "
                    f"(need both UP or HTF RANGE + LTF UP/RANGE)"
                )

        if not short_conds["mtf_trend"]:
            if htf_trend is None or ltf_trend is None:
                short_reasons_block.append("mtf_trend_sync=missing")
            else:
                short_reasons_block.append(
                    f"mtf_trend_sync htf={htf_trend}, ltf={ltf_trend} "
                    f"(need both DOWN or HTF RANGE + LTF DOWN/RANGE)"
                )

        # ------------------------------------------------------------------
        # Aether Oracle decision
        # ------------------------------------------------------------------
        oracle_outputs: Optional[OracleOutputs] = None
        if oracle_inputs is not None:
            try:
                oracle_outputs = self.oracle.decide(oracle_inputs, risk_manager)
            except Exception as e:
                logger.error(f"Error in Aether Oracle decide: {e}", exc_info=True)
                oracle_outputs = None

        if oracle_outputs is not None:
            # Long side
            if (
                oracle_outputs.long_scores.fused is not None
                and oracle_outputs.long_scores.fused < self.oracle.entry_prob_threshold
            ):
                if long_ready:
                    long_ready = False
                    long_reasons_block.append(
                        f"oracle_fused={oracle_outputs.long_scores.fused:.3f} "
                        f"< {self.oracle.entry_prob_threshold:.2f}"
                    )
                    long_reasons_block.extend(oracle_outputs.long_scores.reasons)

            # Short side
            if (
                oracle_outputs.short_scores.fused is not None
                and oracle_outputs.short_scores.fused < self.oracle.entry_prob_threshold
            ):
                if short_ready:
                    short_ready = False
                    short_reasons_block.append(
                        f"oracle_fused={oracle_outputs.short_scores.fused:.3f} "
                        f"< {self.oracle.entry_prob_threshold:.2f}"
                    )
                    short_reasons_block.extend(oracle_outputs.short_scores.reasons)

        # Log decision snapshot including oracle metrics
        self.log_decision_state(
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
        )

        # Entry decisions
        if oracle_outputs is None:
            # Fallback to legacy behavior if Oracle not available at all
            if long_ready:
                self.enter_position(
                    data_manager,
                    order_manager,
                    risk_manager,
                    side="long",
                    current_price=current_price,
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    now_sec=now_sec,
                    kelly_margin=None,
                    oracle_fused=None,
                )
                return
            if short_ready:
                self.enter_position(
                    data_manager,
                    order_manager,
                    risk_manager,
                    side="short",
                    current_price=current_price,
                    imbalance_data=imbalance_data,
                    wall_data=wall_data,
                    delta_data=delta_data,
                    touch_data=touch_data,
                    now_sec=now_sec,
                    kelly_margin=None,
                    oracle_fused=None,
                )
                return
            return

        # With Oracle active: use Kelly sizing and fused score
        if long_ready:
            self.enter_position(
                data_manager,
                order_manager,
                risk_manager,
                side="long",
                current_price=current_price,
                imbalance_data=imbalance_data,
                wall_data=wall_data,
                delta_data=delta_data,
                touch_data=touch_data,
                now_sec=now_sec,
                kelly_margin=self.compute_kelly_margin(
                    risk_manager, oracle_outputs.long_scores
                ),
                oracle_fused=oracle_outputs.long_scores.fused,
            )
            return

        if short_ready:
            self.enter_position(
                data_manager,
                order_manager,
                risk_manager,
                side="short",
                current_price=current_price,
                imbalance_data=imbalance_data,
                wall_data=wall_data,
                delta_data=delta_data,
                touch_data=touch_data,
                now_sec=now_sec,
                kelly_margin=self.compute_kelly_margin(
                    risk_manager, oracle_outputs.short_scores
                ),
                oracle_fused=oracle_outputs.short_scores.fused,
            )
            return

    def compute_kelly_margin(
        self, risk_manager, scores: OracleSideScores
    ) -> Optional[float]:
        """
        Compute Kelly-based margin target, capped at 2% of available balance
        and respecting MIN/MAX_MARGIN_PER_TRADE.

        Returns None if balance missing.
        """
        balance_info = risk_manager.get_available_balance()
        if not balance_info:
            logger.error("Cannot fetch balance for Kelly sizing")
            return None

        available = float(balance_info.get("available", 0.0))
        if available <= 0.0:
            logger.warning("No available balance for Kelly sizing")
            return None

        kelly_f = scores.kelly_f
        kelly_margin = available * kelly_f
        kelly_margin = max(config.MIN_MARGIN_PER_TRADE, kelly_margin)
        kelly_margin = min(config.MAX_MARGIN_PER_TRADE, kelly_margin)

        return kelly_margin

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
        kelly_margin: Optional[float],
        oracle_fused: Optional[float],
    ) -> None:
        """
        Bracket entry:
         - Place main LIMIT order slightly beyond current price (below for longs, above for shorts).
         - Compute full TP and SL from margin-based price move.
         - Place HALF TP initially; extend to FULL if still in trade after 5min.
         - If kelly_margin is provided, override legacy BALANCE_USAGE_PERCENTAGE sizing
           and use Kelly-based margin for this trade.
         - Uses session-aware dynamic slippage and TP/SL ROI.
        """
        if self.current_position is not None:
            return

        balance_info = risk_manager.get_available_balance()
        if not balance_info:
            logger.error("Cannot fetch balance for entry")
            return

        available = float(balance_info.get("available", 0.0))
        if available < config.MIN_MARGIN_PER_TRADE:
            logger.warning(
                f"Available {available:.2f} < MIN_MARGIN {config.MIN_MARGIN_PER_TRADE}"
            )
            return

        # Session-aware dynamic parameters
        dyn = self.get_dynamic_params(data_manager, current_price, now_sec)
        slippage_ticks = dyn["slippage_ticks"]
        profit_roi = dyn["profit_roi"]
        stop_roi = dyn["stop_roi"]

        # Position sizing: Kelly if provided, else % of balance
        if kelly_margin is not None:
            position_margin = kelly_margin
        else:
            position_margin = available * (config.BALANCE_USAGE_PERCENTAGE / 100.0)

        position_margin = max(config.MIN_MARGIN_PER_TRADE, position_margin)
        position_margin = min(config.MAX_MARGIN_PER_TRADE, position_margin)

        # Limit price slightly better than current (slippage-aware)
        tick = config.TICKSIZE
        if side == "long":
            limit_price = current_price - slippage_ticks * tick
        else:
            limit_price = current_price + slippage_ticks * tick

        if limit_price <= 0:
            logger.error(f"Computed invalid limit_price={limit_price:.2f}")
            return

        notional = position_margin * config.LEVERAGE
        raw_quantity = notional / limit_price

        # Enforce BTC step size 0.001
        quantity = round(raw_quantity / 0.001) * 0.001
        if quantity <= 0 or quantity < 0.001:
            logger.warning(
                f"Quantity raw={raw_quantity:.6f} too small after rounding "
                f"→ {quantity:.6f}"
            )
            return

        self.trade_seq += 1
        trade_id = f"ZS{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{self.trade_seq}"

        side_str = "BUY" if side == "long" else "SELL"

        htf_trend_at_entry = dyn.get("htf_trend", None)

        # ------------------------------------------------------------------
        # Target profit/loss in USDT based on margin ROE%
        half_profit_usdt = position_margin * (profit_roi / 2.0)  # 5% for half TP
        full_profit_usdt = position_margin * profit_roi  # 10% for full TP
        loss_usdt = position_margin * abs(stop_roi)  # 3% for SL

        if quantity <= 0:
            logger.error("Invalid quantity for TP/SL computation")
            return

        # Calculate price moves needed to hit ROE% targets
        half_tp_move = half_profit_usdt / quantity
        full_tp_move = full_profit_usdt / quantity
        sl_move = loss_usdt / quantity

        # Calculate TP/SL prices and round to tick size for exchange
        tick = config.TICKSIZE
        if side == "long":
            half_tp_price = round((limit_price + half_tp_move) / tick) * tick
            full_tp_price = round((limit_price + full_tp_move) / tick) * tick
            sl_price = round((limit_price - sl_move) / tick) * tick
        else:
            half_tp_price = round((limit_price - half_tp_move) / tick) * tick
            full_tp_price = round((limit_price - full_tp_move) / tick) * tick
            sl_price = round((limit_price + sl_move) / tick) * tick

        # Verify actual ROE% after rounding to tick size
        actual_half_profit = abs(half_tp_price - limit_price) * quantity
        actual_full_profit = abs(full_tp_price - limit_price) * quantity
        actual_loss = abs(sl_price - limit_price) * quantity

        half_roe = (actual_half_profit / position_margin) * 100.0
        full_roe = (actual_full_profit / position_margin) * 100.0
        sl_roe = (actual_loss / position_margin) * 100.0

        logger.info(
            f"TP/SL ROE verification: Half TP={half_roe:.2f}%, "
            f"Full TP={full_roe:.2f}%, SL={sl_roe:.2f}%"
        )

        logger.info("-" * 80)
        logger.info(f"Z-SCORE ICEBERG ENTRY BRACKET {trade_id}")
        logger.info("-" * 80)
        logger.info(f"Side         : {side.upper()}")
        logger.info(f"Current Price: {current_price:.2f}")
        logger.info(f"Limit Price  : {limit_price:.2f}")
        logger.info(
            f"Session      : {dyn['session']} major={dyn['is_major']}, weekend={dyn['is_weekend']}"
        )
        logger.info(
            f"Dynamic Params: slip={slippage_ticks} ticks, "
            f"TP_ROI={profit_roi*100:.2f}%, SL_ROI={stop_roi*100:.2f}%"
        )
        logger.info(f"Imbalance    : {imbalance_data['imbalance']:.3f}")
        logger.info(
            "Wall Strength: bid={:.2f}, ask={:.2f}".format(
                wall_data["bid_wall_strength"], wall_data["ask_wall_strength"]
            )
        )
        logger.info(f"Delta Z-Score: {delta_data['zscore']:.2f}")
        logger.info(
            "Touch Dist: bid={:.1f} ticks, ask={:.1f} ticks".format(
                touch_data["bid_distance_ticks"], touch_data["ask_distance_ticks"]
            )
        )
        if oracle_fused is not None:
            logger.info(f"Aether Fused: {oracle_fused:.3f}")

        # Reward/Risk logging with FULL TP
        reward = 0.0
        risk = 0.0
        if side == "long":
            reward = max(full_tp_price - limit_price, 0.0)
            risk = max(limit_price - sl_price, 0.0)
        else:
            reward = max(limit_price - full_tp_price, 0.0)
            risk = max(sl_price - limit_price, 0.0)

        if risk > 0 and reward > 0:
            rr = reward / risk
            logger.info(
                f"R:R (FULL TP): reward={reward:.2f}, risk={risk:.2f}, RR={rr:.2f}:1"
            )
        else:
            logger.info("R:R (FULL TP): N/A (check TP/SL configuration)")

        logger.info(f"Initial HALF TP: {half_tp_price:.2f} ({profit_roi * 50:.2f}% on margin)")
        logger.info(f"FULL TP target : {full_tp_price:.2f} ({profit_roi * 100:.2f}% on margin)")
        logger.info(f"SL Price       : {sl_price:.2f}")

        # 1) Place main LIMIT order
        main_order = order_manager.place_limit_order(
            side=side_str,
            quantity=quantity,
            price=limit_price,
            reduce_only=False,
        )
        if not main_order or "order_id" not in main_order:
            logger.error("Main LIMIT order placement failed")
            return
        main_order_id = main_order["order_id"]
        logger.info(f"Main LIMIT order id: {main_order_id}")

        # 2) Place HALF TP (initial take-profit)
        if side == "long":
            tp_side = "SELL"
        else:
            tp_side = "BUY"

        tp_order = order_manager.place_take_profit(
            side=tp_side,
            quantity=0,
            trigger_price=half_tp_price,
        )
        if not tp_order or "order_id" not in tp_order:
            logger.error("TP order placement failed; cancelling main order")
            try:
                order_manager.cancel_order(main_order_id)
            except Exception as e:
                logger.error(
                    f"Error cancelling main order {main_order_id}: {e}", exc_info=True
                )
            return
        tp_order_id = tp_order["order_id"]
        logger.info(f"Initial HALF TP order id: {tp_order_id} @ {half_tp_price:.2f}")

        # 3) Place SL (stop-loss)
        if side == "long":
            sl_side = "SELL"
        else:
            sl_side = "BUY"

        sl_order = order_manager.place_stop_loss(
            side=sl_side,
            quantity=0,
            trigger_price=sl_price,
        )
        if not sl_order or "order_id" not in sl_order:
            logger.error("SL order placement failed; cancelling main & TP")
            try:
                order_manager.cancel_order(tp_order_id)
                order_manager.cancel_order(main_order_id)
            except Exception as e:
                logger.error(
                    f"Error cancelling main/TP after SL failure: {e}", exc_info=True
                )
            return
        sl_order_id = sl_order["order_id"]
        logger.info(f"SL order id: {sl_order_id} @ {sl_price:.2f}")

        # Telegram trade-entry notification (rich)
        try:
            atr_str = f"{dyn['atr_pct']*100:.2f}%" if dyn["atr_pct"] else "N/A"
            msg_lines = [
                "🚀 Z-Score trade OPENED",
                f"Trade ID: {trade_id}",
                f"Symbol: {config.SYMBOL}",
                f"Side: {side.upper()} | Qty: {quantity:.6f} BTC",
                f"Limit price: {limit_price:.2f} | Current: {current_price:.2f}",
                f"Session: {dyn['session']} | ATR: {atr_str}",
                f"Margin: {position_margin:.2f} USDT | Lev: {config.LEVERAGE}x",
                f"TP (half): {half_tp_price:.2f} | TP (full): {full_tp_price:.2f}",
                f"SL: {sl_price:.2f}",
                f"Imbalance: {imbalance_data['imbalance']:.3f}",
                f"Z-score: {delta_data['zscore']:.2f}",
                f"Walls (bid/ask): {wall_data['bid_wall_strength']:.1f} / {wall_data['ask_wall_strength']:.1f}",
            ]
            if oracle_fused is not None:
                msg_lines.append(f"Aether fused: {oracle_fused:.3f}")
            if htf_trend_at_entry is not None:
                msg_lines.append(f"HTF trend: {htf_trend_at_entry}")
            send_telegram_message("\n".join(msg_lines))
        except Exception as e:
            logger.error(f"Failed to send Telegram entry notification: {e}", exc_info=True)

        # Store position
        htf_trend_at_entry_safe = (
            htf_trend_at_entry if htf_trend_at_entry is not None else "UNKNOWN"
        )

        self.current_position = ZScorePosition(
            trade_id=trade_id,
            side=side,
            quantity=quantity,
            entry_price=limit_price,
            entry_time_sec=now_sec,
            entry_wall_volume=(
                wall_data["bid_vol_zone"] if side == "long" else wall_data["ask_vol_zone"]
            ),
            wall_zone_low=wall_data["zone_low"],
            wall_zone_high=wall_data["zone_high"],
            entry_imbalance=imbalance_data["imbalance"],
            entry_zscore=delta_data["zscore"],
            tp_price=full_tp_price,
            sl_price=sl_price,
            margin_used=position_margin,
            tp_order_id=tp_order_id,
            sl_order_id=sl_order_id,
            main_order_id=main_order_id,
            main_filled=False,
            tp_reduced=False,
            entry_htf_trend=htf_trend_at_entry_safe,
        )

        risk_manager.record_trade_opened()
        logger.info("=" * 80)

    def is_filled(self, order_status: Optional[Dict]) -> bool:
        """
        Check if order status indicates filled/executed state.
        """
        if order_status is None:
            return False
        status = order_status.get("status", "").upper()
        return status in ("FILLED", "EXECUTED")

    def manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """
        Manage an open position by tracking main/TP/SL status.
         - If main LIMIT is not filled within ENTRY_FILL_TIMEOUT_SEC, cancel main, TP, SL.
         - Start with HALF TP at entry; after 5 minutes, if still in trade and TP not hit,
           extend TP to FULL target stored in pos.tp_price.
         - Dynamic TP tightening if trade stagnates in low volatility (session-aware).
         - No market close orders are sent; exits are via TP/SL.
        """
        pos = self.current_position
        if pos is None:
            return

        elapsed_min = (now_sec - pos.entry_time_sec) / 60.0

        if elapsed_min >= config.MAX_HOLD_MINUTES:
            if not hasattr(pos, "max_hold_logged"):
                pos.max_hold_logged = True
                logger.info(
                    f"Max hold {config.MAX_HOLD_MINUTES}min reached for {pos.trade_id} - "
                    f"leaving exit to TP/SL (no forced market close)."
                )

        # If main LIMIT not filled within ENTRY_FILL_TIMEOUT_SEC, cancel bracket
        if not pos.main_filled:
            elapsed = now_sec - pos.entry_time_sec
            if elapsed >= self.ENTRY_FILL_TIMEOUT_SEC:
                logger.info(
                    f"Main LIMIT not filled in {self.ENTRY_FILL_TIMEOUT_SEC:.0f}s "
                    f"for trade {pos.trade_id} - cancelling main, TP, and SL."
                )
                try:
                    if pos.tp_order_id:
                        order_manager.cancel_order(pos.tp_order_id)
                    if pos.sl_order_id:
                        order_manager.cancel_order(pos.sl_order_id)
                    if pos.main_order_id:
                        order_manager.cancel_order(pos.main_order_id)
                except Exception as e:
                    logger.error(
                        f"Error cancelling bracket for {pos.trade_id}: {e}",
                        exc_info=True,
                    )
                self.current_position = None
                self.last_exit_time_min = now_sec / 60.0
                return

        # Throttle TP/SL status checks (every 10 seconds)
        if now_sec - self.last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self.last_status_check_sec = now_sec

            main_status = order_manager.get_order_status(pos.main_order_id)
            tp_status = order_manager.get_order_status(pos.tp_order_id)
            sl_status = order_manager.get_order_status(pos.sl_order_id)

            # Check if main LIMIT was filled
            if not pos.main_filled and self.is_filled(main_status):
                pos.main_filled = True
                logger.info(f"Main LIMIT filled for {pos.trade_id}")

            # Exit logic: check if TP or SL was hit
            exit_reason: Optional[str] = None
            exit_order: Optional[Dict] = None

            if self.is_filled(tp_status):
                exit_reason = "TP_HIT"
                exit_order = tp_status
            elif self.is_filled(sl_status):
                exit_reason = "SL_HIT"
                exit_order = sl_status

            if exit_order is not None:
                try:
                    exit_price = order_manager.extract_fill_price(exit_order)
                except Exception as e:
                    logger.error(
                        f"Failed to extract exit price for {pos.trade_id}: {e}",
                        exc_info=True,
                    )
                    return

                self.finalize_exit(
                    order_manager=order_manager,
                    risk_manager=risk_manager,
                    exit_price_live=exit_price,
                    exit_order=exit_order,
                    entry_order=main_status,
                    reason=exit_reason,
                    now_sec=now_sec,
                )

        # TP extension: if main is filled and HALF TP not hit yet, extend to FULL TP after 5 min
        if pos.main_filled and not pos.tp_reduced:
            elapsed_min_for_tp = (now_sec - pos.entry_time_sec) / 60.0
            if elapsed_min_for_tp >= 5.0:
                logger.info(
                    f"Considering TP EXTENSION to FULL for {pos.trade_id} "
                    f"(elapsed={elapsed_min_for_tp:.1f} min, full TP={pos.tp_price:.2f})"
                )
                if pos.quantity <= 0:
                    logger.error(f"Invalid quantity for TP extension on {pos.trade_id}")
                    pos.tp_reduced = True
                    return

                if pos.side == "long":
                    tp_side = "SELL"
                else:
                    tp_side = "BUY"

                try:
                    if pos.tp_order_id:
                        order_manager.cancel_order(pos.tp_order_id)
                except Exception as e:
                    logger.error(
                        f"Error cancelling HALF TP {pos.tp_order_id}: {e}", exc_info=True
                    )

                try:
                    new_tp_order = order_manager.place_take_profit(
                        side=tp_side,
                        quantity=0,
                        trigger_price=pos.tp_price,
                    )
                    if new_tp_order and "order_id" in new_tp_order:
                        pos.tp_order_id = new_tp_order["order_id"]
                        pos.tp_reduced = True
                        logger.info(
                            f"✓ New FULL TP placed for {pos.trade_id} @ {pos.tp_price:.2f}, "
                            f"order_id={pos.tp_order_id}"
                        )
                    else:
                        logger.error(
                            f"Failed to place FULL TP for {pos.trade_id} - "
                            f"leaving position without active TP."
                        )
                        pos.tp_reduced = True
                except Exception as e:
                    logger.error(
                        f"Error placing FULL TP for {pos.trade_id}: {e}", exc_info=True
                    )
                    pos.tp_reduced = True

        # Dynamic TP tightening (only after full TP extension logic)
        if (
            getattr(config, "ENABLE_TP_TIGHTENING", False)
            and pos.main_filled
            and pos.tp_reduced
        ):
            elapsed_min_dyn = (now_sec - pos.entry_time_sec) / 60.0
            if elapsed_min_dyn >= config.DYNAMIC_TP_MINUTES:
                atr_pct = None
                try:
                    atr_pct = data_manager.get_atr_percent()
                except Exception:
                    pass

                direction = 1.0 if pos.side == "long" else -1.0
                upnl_price = (current_price - pos.entry_price) * direction * pos.quantity
                upnl_roi = upnl_price / pos.margin_used if pos.margin_used > 0 else 0.0
                full_tp_roi = config.PROFIT_TARGET_ROI

                if (
                    atr_pct is not None
                    and atr_pct < config.DYNAMIC_TP_MIN_ATR_PCT
                    and upnl_roi >= full_tp_roi * config.DYNAMIC_TP_REQUIRED_PROGRESS
                ):
                    new_tp_roi = config.DYNAMIC_TP_NEAR_ROI
                    new_profit_usdt = pos.margin_used * new_tp_roi
                    newmove_tp = (
                        new_profit_usdt / pos.quantity if pos.quantity > 0 else 0.0
                    )

                    if pos.side == "long":
                        new_tp_price = pos.entry_price + newmove_tp
                        tp_side_dyn = "SELL"
                    else:
                        new_tp_price = pos.entry_price - newmove_tp
                        tp_side_dyn = "BUY"

                    logger.info(
                        f"Dynamic TP tightening for {pos.trade_id}: "
                        f"elapsed={elapsed_min_dyn:.1f}m, atr={atr_pct}, "
                        f"uROI={upnl_roi:.4f}, new_tp_roi={new_tp_roi:.4f}, "
                        f"new_tp_price={new_tp_price:.2f}"
                    )

                    try:
                        if pos.tp_order_id:
                            order_manager.cancel_order(pos.tp_order_id)
                    except Exception as e:
                        logger.error(
                            f"Error cancelling old TP {pos.tp_order_id} for {pos.trade_id}: {e}",
                            exc_info=True,
                        )

                    try:
                        new_tp_order_dyn = order_manager.place_take_profit(
                            side=tp_side_dyn,
                            quantity=0,
                            trigger_price=new_tp_price,
                        )
                        if new_tp_order_dyn and "order_id" in new_tp_order_dyn:
                            pos.tp_order_id = new_tp_order_dyn["order_id"]
                            logger.info(
                                f"✓ New tightened TP placed for {pos.trade_id} @ {new_tp_price:.2f}, "
                                f"order_id={pos.tp_order_id}"
                            )
                        else:
                            logger.error(
                                f"Failed to place tightened TP for {pos.trade_id} "
                                f"at {new_tp_price:.2f}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error placing tightened TP for {pos.trade_id}: {e}",
                            exc_info=True,
                        )

        # Log position snapshot every 2 minutes
        if now_sec - self.last_position_log_sec >= self.POSITION_LOG_INTERVAL_SEC:
            self.last_position_log_sec = now_sec
            direction = 1.0 if pos.side == "long" else -1.0
            upnl = (current_price - pos.entry_price) * direction * pos.quantity
            elapsed_min_log = (now_sec - pos.entry_time_sec) / 60.0
            logger.info(
                f"POSITION {pos.trade_id} | {pos.side.upper()} qty={pos.quantity:.6f} "
                f"entry={pos.entry_price:.2f} cur={current_price:.2f} "
                f"uPnL≈{upnl:.2f} USDT, elapsed={elapsed_min_log:.1f} min"
            )

    def finalize_exit(
        self,
        order_manager,
        risk_manager,
        exit_price_live: float,
        exit_order: Optional[Dict],
        entry_order: Optional[Dict],
        reason: str,
        now_sec: float,
    ) -> None:
        """
        Finalize exit: calculate P/L, update RiskManager, log to Excel, send Telegram.
        """
        pos = self.current_position
        if pos is None:
            return

        direction = 1.0 if pos.side == "long" else -1.0

        # Fees from configured VIP rates (maker on entry, taker on exit)
        notional_entry = pos.entry_price * pos.quantity
        notional_exit = exit_price_live * pos.quantity
        fee_entry = config.MAKER_FEE_RATE * notional_entry
        fee_exit = config.TAKER_FEE_RATE * notional_exit
        total_fees = fee_entry + fee_exit

        pnl_gross = (exit_price_live - pos.entry_price) * direction * pos.quantity
        pnl_usdt_real = pnl_gross - total_fees

        duration_min = (now_sec - pos.entry_time_sec) / 60.0

        logger.info("=" * 80)
        logger.info(f"EXITING Z-SCORE POSITION {pos.trade_id}")
        logger.info(f"Reason   : {reason}")
        logger.info(f"Side     : {pos.side.upper()}")
        logger.info(f"Entry Price: {pos.entry_price:.2f}")
        logger.info(f"Exit Price : {exit_price_live:.2f}")
        logger.info(f"Quantity : {pos.quantity:.6f}")
        logger.info(f"Fees (USDT): {total_fees:.4f}")
        logger.info(f"P/L (USDT): {pnl_usdt_real:.2f}")
        logger.info("=" * 80)

        risk_manager.update_trade_stats(pnl_usdt_real)

        # Telegram trade-exit notification (rich)
        try:
            total_trades = int(getattr(risk_manager, "total_trades", 0))
            winning_trades = int(getattr(risk_manager, "winning_trades", 0))
            losing_trades = int(
                getattr(
                    risk_manager,
                    "losing_trades",
                    max(0, total_trades - winning_trades),
                )
            )
            realized_pnl = float(getattr(risk_manager, "realized_pnl", 0.0))

            if total_trades > 0:
                win_rate = (winning_trades / total_trades) * 100.0
            else:
                win_rate = 0.0

            outcome = (
                "WIN"
                if pnl_usdt_real > 0
                else "LOSS" if pnl_usdt_real < 0 else "BREAKEVEN"
            )

            msg_lines = [
                "🏁 Z-Score trade EXITED",
                f"Trade ID: {pos.trade_id}",
                f"Reason: {reason}",
                f"Side: {pos.side.upper()} | Qty: {pos.quantity:.6f} BTC",
                f"Entry: {pos.entry_price:.2f} | Exit: {exit_price_live:.2f}",
                f"Duration: {duration_min:.1f} min",
                f"Net P/L: {pnl_usdt_real:.2f} USDT ({outcome})",
                "",
                "Session stats:",
                f"  Trades: {total_trades}",
                f"  Wins / Losses: {winning_trades} / {losing_trades}",
                f"  Win rate: {win_rate:.2f}%",
                f"  Realized P/L: {realized_pnl:.2f} USDT",
            ]
            send_telegram_message("\n".join(msg_lines))
        except Exception as e:
            logger.error(
                f"Failed to send Telegram exit notification: {e}", exc_info=True
            )

        # Excel logging
        if self.excel_logger:
            self.excel_logger.log_trade(
                trade_id=pos.trade_id,
                entry_time=datetime.fromtimestamp(pos.entry_time_sec).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                exit_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                duration_minutes=duration_min,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price_live,
                quantity=pos.quantity,
                margin_used=pos.margin_used,
                leverage=config.LEVERAGE,
                tp_price=pos.tp_price,
                sl_price=pos.sl_price,
                entry_imbalance=pos.entry_imbalance,
                entry_zscore=pos.entry_zscore,
                entry_wall_volume=pos.entry_wall_volume,
                exit_reason=reason,
                pnl_usdt=pnl_usdt_real,
                entry_htf_trend=pos.entry_htf_trend,
            )

        self.current_position = None
        self.last_exit_time_min = now_sec / 60.0
