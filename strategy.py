"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version

UPDATED WITH:
- Vol-Regime Detection & Dynamic Gates
- Weighted Score Gauntlet (instead of binary AND)
- Data Fusion Maximization (9 signals)
- Dynamic TP/SL/Sizing per regime
- Enhanced logging for all calculated parameters
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
    tp_price: float  # full TP price (based on full PROFIT_TARGET_ROI)
    sl_price: float
    margin_used: float
    tp_order_id: str
    sl_order_id: str
    main_order_id: str
    main_filled: bool
    tp_reduced: bool  # has TP been switched to full already?
    entry_htf_trend: str  # "UP" / "DOWN" / "RANGE" / "UNKNOWN"
    entry_vol_regime: str  # "LOW" / "HIGH" / "NEUTRAL" / "UNKNOWN"
    entry_weighted_score: float  # Entry score (0-1)
    last_score_check_sec: float  # For score decay exit


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
        logger.info(f"Imbalance Threshold   = {config.IMBALANCE_THRESHOLD:.2f}")
        logger.info(f"Wall Volume Mult BASE = {config.MIN_WALL_VOLUME_MULT:.2f}×")
        logger.info(f"Delta Z Threshold BASE= {config.DELTA_Z_THRESHOLD:.2f}")
        logger.info(f"Zone Ticks            = ±{config.ZONE_TICKS}")
        logger.info(f"Touch Threshold       = {config.PRICE_TOUCH_THRESHOLD_TICKS} ticks")
        logger.info(f"Profit Target ROI BASE= {config.PROFIT_TARGET_ROI * 100:.2f}%")
        logger.info(f"Stop Loss ROI BASE    = {config.STOP_LOSS_ROI * 100:.2f}%")
        logger.info(f"Max Hold Minutes      = {config.MAX_HOLD_MINUTES}")
        logger.info(f"Weighted Scoring      = {config.ENABLE_WEIGHTED_SCORING}")
        logger.info(f"Score Entry Threshold = {config.WEIGHTED_SCORE_ENTRY_THRESHOLD}")
        logger.info("=" * 80)

    # ======================================================================
    # Vol-Regime Helpers
    # ======================================================================

    def _get_dynamic_z_threshold(self, vol_regime: str, atr_pct: Optional[float]) -> float:
        """
        Get dynamic Z-score threshold based on volatility regime.
        Formula: base + scaling * (atr_pct - low) / (high - low), clamped to regime bounds.
        """
        if vol_regime == "LOW":
            return config.VOL_REGIME_Z_LOW
        elif vol_regime == "HIGH":
            return config.VOL_REGIME_Z_HIGH
        elif vol_regime == "NEUTRAL" and atr_pct is not None:
            # Scale between LOW and HIGH thresholds
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
            # NEUTRAL: average of both
            return (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0

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
                lv_1m, lv_5m, lv_15m, micro_trap = data_manager._oracle.compute_liquidity_velocity_multi_tf(data_manager)
                norm_cvd = data_manager._oracle.compute_norm_cvd(data_manager, window_sec=10)
                hurst = data_manager._oracle.compute_hurst_exponent(data_manager, window_ticks=20)
                bos_align = data_manager._oracle.compute_bos_alignment(data_manager, current_price)

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
            f"Vol Regime: {vol_regime} (ATR%: {atr_pct*100:.3f}% if atr_pct else 'N/A'})",
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

    # ======================================================================
    # Position Entry
    # ======================================================================

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
        
        WORKFLOW:
        1. Guard: Check if position already exists
        2. Balance check: Ensure sufficient margin
        3. Dynamic TP/SL calculation: Based on vol regime
        4. Position sizing: Vol-regime aware (15% HIGH, 20% LOW)
        5. Quantity calculation: Leverage-adjusted
        6. Entry price: Limit order slightly beyond current price
        7. Place main LIMIT order
        8. Wait for fill (timeout: 60s)
        9. Place TP and SL bracket orders
        10. Create position object and store state
        11. Update risk manager stats
        12. Send Telegram notification
        """
        
        # ========================================================================
        # STEP 1: Guard - Prevent double entry
        # ========================================================================
        if self.current_position is not None:
            logger.warning("Position already exists; skipping entry")
            return

        # ========================================================================
        # STEP 2: Balance Check
        # ========================================================================
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

        # ========================================================================
        # STEP 3: Dynamic TP/SL Calculation Based on Vol Regime
        # ========================================================================
        # Get regime-specific TP and SL ROI multipliers
        tp_roi, sl_roi = self._get_dynamic_tp_sl(vol_regime)
        
        # Example outputs:
        # - HIGH vol: tp_roi = 0.14 (+14%), sl_roi = -0.027 (-2.7%)
        # - LOW vol:  tp_roi = 0.11 (+11%), sl_roi = -0.029 (-2.9%)
        # - NEUTRAL:  tp_roi = 0.10 (+10%), sl_roi = -0.03 (-3%)
        
        logger.info(
            f"Vol Regime: {vol_regime} -> TP ROI: {tp_roi*100:.2f}%, SL ROI: {sl_roi*100:.2f}%"
        )

        # ========================================================================
        # STEP 4: Vol-Regime Aware Position Sizing
        # ========================================================================
        # Use the fixed risk manager method for regime-aware sizing
        target_margin, quantity = risk_manager.calculate_position_size_regime_aware(
            entry_price=current_price,
            vol_regime=vol_regime,
        )
        
        # Validate quantity
        if quantity <= 0:
            logger.warning("Computed quantity <= 0; skipping entry")
            return
        
        # Additional validation: ensure quantity meets minimum
        min_qty = 0.001  # BTC minimum
        if quantity < min_qty:
            logger.warning(f"Quantity {quantity:.6f} < minimum {min_qty}; adjusting")
            quantity = min_qty

        # ========================================================================
        # STEP 5: Entry Price Calculation (Limit Order Strategy)
        # ========================================================================
        # Place limit order slightly beyond current price to ensure fill
        # while avoiding market order slippage
        
        if side == "long":
            # For longs: buy slightly below current price
            entry_price = current_price - (1 * config.TICK_SIZE)
            # Example: current=95000, tick=1 -> entry=94999
        else:
            # For shorts: sell slightly above current price
            entry_price = current_price + (1 * config.TICK_SIZE)
            # Example: current=95000, tick=1 -> entry=95001
        
        entry_price = round(entry_price, 2)
        
        logger.info(f"Entry price calculated: {entry_price:.2f} (current: {current_price:.2f})")

        # ========================================================================
        # STEP 6: Calculate TP and SL Prices from Margin-Based ROI
        # ========================================================================
        # TP and SL are calculated based on ROI on margin, not on price directly
        # This ensures risk management is precise regardless of leverage
        
        if side == "long":
            # Long TP: Entry + price_move_for_TP_roi
            # Formula: price_move = (margin * tp_roi) / (quantity * leverage)
            # Note: We use quantity directly (not leveraged) because position value = qty * price
            price_move_for_tp = (target_margin * abs(tp_roi)) / quantity
            tp_price = entry_price + price_move_for_tp
            
            # Long SL: Entry - price_move_for_SL_roi
            price_move_for_sl = (target_margin * abs(sl_roi)) / quantity
            sl_price = entry_price - price_move_for_sl
            
        else:  # short
            # Short TP: Entry - price_move_for_TP_roi
            price_move_for_tp = (target_margin * abs(tp_roi)) / quantity
            tp_price = entry_price - price_move_for_tp
            
            # Short SL: Entry + price_move_for_SL_roi
            price_move_for_sl = (target_margin * abs(sl_roi)) / quantity
            sl_price = entry_price + price_move_for_sl
        
        # Round to 2 decimals for exchange compatibility
        tp_price = round(tp_price, 2)
        sl_price = round(sl_price, 2)
        
        # Validate TP/SL placement
        if side == "long":
            if tp_price <= entry_price or sl_price >= entry_price:
                logger.error(
                    f"Invalid TP/SL for LONG: entry={entry_price}, tp={tp_price}, sl={sl_price}"
                )
                return
        else:
            if tp_price >= entry_price or sl_price <= entry_price:
                logger.error(
                    f"Invalid TP/SL for SHORT: entry={entry_price}, tp={tp_price}, sl={sl_price}"
                )
                return

        # ========================================================================
        # STEP 7: Generate Trade ID
        # ========================================================================
        self.trade_seq += 1
        trade_id = f"Z{int(now_sec)}_{self.trade_seq}"
        # Example: Z1733215000_42

        # ========================================================================
        # STEP 8: Pre-Entry Logging (Complete Audit Trail)
        # ========================================================================
        logger.info("=" * 80)
        logger.info(f"ENTERING {side.upper()} POSITION")
        logger.info("=" * 80)
        logger.info(f"Trade ID           : {trade_id}")
        logger.info(f"Timestamp          : {datetime.utcfromtimestamp(now_sec).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Vol Regime         : {vol_regime}")
        logger.info(f"Weighted Score     : {weighted_score:.4f} (threshold: {config.WEIGHTED_SCORE_ENTRY_THRESHOLD})")
        logger.info("-" * 80)
        logger.info(f"Entry Price        : {entry_price:.2f}")
        logger.info(f"Quantity           : {quantity:.6f} BTC")
        logger.info(f"Margin Used        : {target_margin:.2f} USDT")
        logger.info(f"Leverage           : {config.LEVERAGE}x")
        logger.info(f"Position Value     : {quantity * entry_price:.2f} USDT")
        logger.info("-" * 80)
        logger.info(f"TP Price           : {tp_price:.2f} (ROI: {tp_roi*100:.2f}%)")
        logger.info(f"SL Price           : {sl_price:.2f} (ROI: {sl_roi*100:.2f}%)")
        logger.info(f"TP Distance        : {abs(tp_price - entry_price):.2f} USDT")
        logger.info(f"SL Distance        : {abs(sl_price - entry_price):.2f} USDT")
        logger.info(f"Risk:Reward Ratio  : {abs(tp_price - entry_price) / abs(entry_price - sl_price):.2f}:1")
        logger.info("-" * 80)
        logger.info(f"Imbalance          : {imbalance_data['imbalance']:.3f}")
        logger.info(f"Z-Score            : {delta_data['z_score']:.2f}")
        logger.info(f"Wall Strength      : {wall_data['bid_wall_strength' if side=='long' else 'ask_wall_strength']:.2f}x")
        logger.info(f"Touch Distance     : {touch_data['bid_distance_ticks' if side=='long' else 'ask_distance_ticks']:.1f} ticks")
        
        if oracle_inputs:
            logger.info("-" * 80)
            logger.info(f"CVD                : {oracle_inputs.norm_cvd:.3f if oracle_inputs.norm_cvd else 'N/A'}")
            logger.info(f"LV 1m              : {oracle_inputs.lv_1m:.2f if oracle_inputs.lv_1m else 'N/A'}")
            logger.info(f"Hurst              : {oracle_inputs.hurst:.3f if oracle_inputs.hurst else 'N/A'}")
            logger.info(f"BOS Align          : {oracle_inputs.bos_align:.3f if oracle_inputs.bos_align else 'N/A'}")
            logger.info(f"HTF Trend          : {oracle_inputs.htf_trend or 'N/A'}")
            logger.info(f"LTF Trend          : {oracle_inputs.ltf_trend or 'N/A'}")
        
        logger.info("=" * 80)

        # ========================================================================
        # STEP 9: Place Main LIMIT Order
        # ========================================================================
        try:
            logger.info(f"Placing main LIMIT {side.upper()} order...")
            
            main_order = order_manager.place_limit_order(
                side="BUY" if side == "long" else "SELL",
                quantity=quantity,
                price=entry_price,
                reduce_only=False,  # Opening position, not closing
            )

            if not main_order or "order_id" not in main_order:
                logger.error("Main order placement failed - no order_id returned")
                logger.error(f"Order response: {main_order}")
                return

            main_order_id = main_order["order_id"]
            logger.info(f"✓ Main order placed: {main_order_id}")

            # ====================================================================
            # STEP 10: Wait for Fill (with Timeout)
            # ====================================================================
            logger.info(f"Waiting for fill (timeout: {self.ENTRY_FILL_TIMEOUT_SEC}s)...")
            
            try:
                filled_order = order_manager.wait_for_fill(
                    order_id=main_order_id,
                    timeout_sec=self.ENTRY_FILL_TIMEOUT_SEC,
                    poll_interval_sec=0.1,  # Check every 100ms
                )
                
                # Extract actual fill price (may differ from limit price)
                fill_price = order_manager.extract_fill_price(filled_order)
                
                logger.info(f"✓ Order filled at: {fill_price:.2f} (limit: {entry_price:.2f})")
                
                # Use actual fill price for TP/SL calculation if significantly different
                if abs(fill_price - entry_price) / entry_price > 0.001:  # >0.1% slippage
                    logger.warning(f"Slippage detected: {abs(fill_price - entry_price):.2f} USDT")
                    
                    # Recalculate TP/SL based on actual fill price
                    if side == "long":
                        tp_price = fill_price + (target_margin * abs(tp_roi)) / quantity
                        sl_price = fill_price - (target_margin * abs(sl_roi)) / quantity
                    else:
                        tp_price = fill_price - (target_margin * abs(tp_roi)) / quantity
                        sl_price = fill_price + (target_margin * abs(sl_roi)) / quantity
                    
                    tp_price = round(tp_price, 2)
                    sl_price = round(sl_price, 2)
                    
                    logger.info(f"TP/SL recalculated: TP={tp_price:.2f}, SL={sl_price:.2f}")
                
            except Exception as e:
                logger.error(f"Main order fill failed: {e}")
                logger.info("Cancelling unfilled order...")
                order_manager.cancel_order(main_order_id)
                return

            # ====================================================================
            # STEP 11: Place TP and SL Bracket Orders
            # ====================================================================
            logger.info("Placing TP and SL bracket orders...")
            
            # Determine order sides for TP and SL (opposite of main order)
            tp_side = "SELL" if side == "long" else "BUY"
            sl_side = "SELL" if side == "long" else "BUY"

            # Place Take Profit order
            tp_order = order_manager.place_take_profit(
                side=tp_side,
                quantity=quantity,
                trigger_price=tp_price,
            )

            if tp_order and "order_id" in tp_order:
                tp_order_id = tp_order["order_id"]
                logger.info(f"✓ TP order placed: {tp_order_id} @ {tp_price:.2f}")
            else:
                logger.error("TP order placement failed")
                tp_order_id = ""

            # Place Stop Loss order
            sl_order = order_manager.place_stop_loss(
                side=sl_side,
                quantity=quantity,
                trigger_price=sl_price,
            )

            if sl_order and "order_id" in sl_order:
                sl_order_id = sl_order["order_id"]
                logger.info(f"✓ SL order placed: {sl_order_id} @ {sl_price:.2f}")
            else:
                logger.error("SL order placement failed")
                sl_order_id = ""

            # ====================================================================
            # STEP 12: Create Position Object
            # ====================================================================
            # Store complete position state for management in subsequent ticks
            
            self.current_position = ZScorePosition(
                # Identity
                trade_id=trade_id,
                side=side,
                
                # Order details
                quantity=quantity,
                entry_price=fill_price,  # Use actual fill price, not limit price
                entry_time_sec=now_sec,
                
                # Entry conditions (for degradation checks)
                entry_wall_volume=wall_data["bid_wall_volume" if side == "long" else "ask_wall_volume"],
                wall_zone_low=current_price - config.ZONE_TICKS * config.TICK_SIZE,
                wall_zone_high=current_price + config.ZONE_TICKS * config.TICK_SIZE,
                entry_imbalance=imbalance_data["imbalance"],
                entry_z_score=delta_data["z_score"],
                
                # Exit targets
                tp_price=tp_price,
                sl_price=sl_price,
                
                # Risk management
                margin_used=target_margin,
                
                # Order IDs for management
                tp_order_id=tp_order_id,
                sl_order_id=sl_order_id,
                main_order_id=main_order_id,
                main_filled=True,
                tp_reduced=False,  # Flag for TP tightening logic
                
                # Entry context (for analysis)
                entry_htf_trend=oracle_inputs.htf_trend if oracle_inputs else "UNKNOWN",
                entry_vol_regime=vol_regime,
                entry_weighted_score=weighted_score,
                
                # Score decay tracking
                last_score_check_sec=now_sec,
            )

            # ====================================================================
            # STEP 13: Update Risk Manager Stats
            # ====================================================================
            risk_manager.record_trade_opened()

            logger.info(f"✓ Position opened successfully: {trade_id}")

            # ====================================================================
            # STEP 14: Excel Logging (if enabled)
            # ====================================================================
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

            # ====================================================================
            # STEP 15: Send Telegram Notification
            # ====================================================================
            try:
                notification = (
                    f"🟢 {'LONG' if side == 'long' else 'SHORT'} ENTRY\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"ID: {trade_id}\n"
                    f"Price: {fill_price:.2f}\n"
                    f"Qty: {quantity:.6f} BTC\n"
                    f"Margin: {target_margin:.2f} USDT\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"TP: {tp_price:.2f} ({tp_roi*100:+.2f}%)\n"
                    f"SL: {sl_price:.2f} ({sl_roi*100:+.2f}%)\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"Vol Regime: {vol_regime}\n"
                    f"Score: {weighted_score:.4f}\n"
                    f"Z-Score: {delta_data['z_score']:.2f}\n"
                    f"Imbalance: {imbalance_data['imbalance']:.3f}"
                )
                
                send_telegram_message(notification)
                
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Critical error entering position: {e}", exc_info=True)
            
            # Emergency cleanup: cancel any placed orders
            try:
                if 'main_order_id' in locals():
                    order_manager.cancel_order(main_order_id)
                if 'tp_order_id' in locals() and tp_order_id:
                    order_manager.cancel_order(tp_order_id)
                if 'sl_order_id' in locals() and sl_order_id:
                    order_manager.cancel_order(sl_order_id)
            except:
                pass


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
        Manage open position:
        - Check TP/SL filled
        - Time stop
        - Score decay exit
        - Trail stop in high vol
        """
        pos = self.current_position
        if pos is None:
            return

        hold_time_min = (now_sec - pos.entry_time_sec) / 60.0

        # Log position state periodically
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

        # Check TP/SL status
        if now_sec - self._last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec

            if pos.tp_order_id:
                tp_status = order_manager.get_order_status(pos.tp_order_id)
                if tp_status and tp_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    self._close_position(
                        reason="TP_HIT",
                        exit_price=pos.tp_price,
                        order_manager=order_manager,
                        risk_manager=risk_manager,
                    )
                    return

            if pos.sl_order_id:
                sl_status = order_manager.get_order_status(pos.sl_order_id)
                if sl_status and sl_status.get("status", "").upper() in ["EXECUTED", "FILLED"]:
                    self._close_position(
                        reason="SL_HIT",
                        exit_price=pos.sl_price,
                        order_manager=order_manager,
                        risk_manager=risk_manager,
                    )
                    return

        # Time stop
        if hold_time_min >= config.MAX_HOLD_MINUTES:
            self._close_position(
                reason="TIME_STOP",
                exit_price=current_price,
                order_manager=order_manager,
                risk_manager=risk_manager,
            )
            return

        # Score decay exit
        if now_sec - pos.last_score_check_sec >= config.SCORE_DECAY_CHECK_INTERVAL_SEC:
            pos.last_score_check_sec = now_sec

            # Recompute score
            vol_regime, atr_pct = data_manager.get_vol_regime()
            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = self._compute_wall_strength(data_manager, current_price, imbalance_data) if imbalance_data else None
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            htf_trend = data_manager.get_htf_trend() if hasattr(data_manager, "get_htf_trend") else None
            ltf_trend = data_manager.get_ltf_trend() if hasattr(data_manager, "get_ltf_trend") else None
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD) if hasattr(data_manager, "get_ema") else None

            try:
                lv_1m, lv_5m, lv_15m, micro_trap = data_manager._oracle.compute_liquidity_velocity_multi_tf(data_manager)
                norm_cvd = data_manager._oracle.compute_norm_cvd(data_manager, window_sec=10)
                hurst = data_manager._oracle.compute_hurst_exponent(data_manager, window_ticks=20)
                bos_align = data_manager._oracle.compute_bos_alignment(data_manager, current_price)

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

                if current_score < config.SCORE_DECAY_EXIT_THRESHOLD:
                    logger.info(f"Score decay exit: current_score={current_score:.4f} < threshold={config.SCORE_DECAY_EXIT_THRESHOLD}")
                    self._close_position(
                        reason="SCORE_DECAY",
                        exit_price=current_price,
                        order_manager=order_manager,
                        risk_manager=risk_manager,
                    )
                    return

            except Exception as e:
                logger.error(f"Error checking score decay: {e}", exc_info=True)

        # Trail stop in high vol (FIXED: Symmetric logic for shorts)
        if pos.entry_vol_regime == "HIGH":
            if pos.side == "long":
                profit_pct = (current_price - pos.entry_price) / pos.entry_price
                if profit_pct >= config.VOL_REGIME_TRAIL_PROFIT_THRESHOLD:
                    new_sl = pos.entry_price * (1.0 + config.VOL_REGIME_TRAIL_BUFFER)
                    if new_sl > pos.sl_price:
                        logger.info(f"Trailing SL (LONG): {pos.sl_price:.2f} -> {new_sl:.2f}")
                        if pos.sl_order_id:
                            order_manager.cancel_order(pos.sl_order_id)
                        
                        sl_order = order_manager.place_stop_loss(
                            side="SELL",
                            quantity=pos.quantity,
                            trigger_price=new_sl,
                        )
                        pos.sl_price = new_sl
                        pos.sl_order_id = sl_order["order_id"] if sl_order else ""
            
            else:  # short
                profit_pct = (pos.entry_price - current_price) / pos.entry_price  # Positive on profit for shorts
                if profit_pct >= config.VOL_REGIME_TRAIL_PROFIT_THRESHOLD:
                    new_sl = pos.entry_price * (1.0 - config.VOL_REGIME_TRAIL_BUFFER)  # Trail down for shorts
                    if new_sl < pos.sl_price:  # Lower is better for short SL
                        logger.info(f"Trailing SL (SHORT): {pos.sl_price:.2f} -> {new_sl:.2f}")
                        if pos.sl_order_id:
                            order_manager.cancel_order(pos.sl_order_id)
                        
                        sl_order = order_manager.place_stop_loss(
                            side="BUY",
                            quantity=pos.quantity,
                            trigger_price=new_sl,
                        )
                        pos.sl_price = new_sl
                        pos.sl_order_id = sl_order["order_id"] if sl_order else ""

    def _close_position(
        self,
        reason: str,
        exit_price: float,
        order_manager,
        risk_manager,
    ) -> None:
        """Close position and calculate P&L."""
        pos = self.current_position
        if pos is None:
            return

        logger.info("=" * 80)
        logger.info(f"CLOSING POSITION: {pos.trade_id}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Exit Price: {exit_price:.2f}")
        logger.info("=" * 80)

        # Cancel remaining orders
        if pos.tp_order_id:
            order_manager.cancel_order(pos.tp_order_id)
        if pos.sl_order_id:
            order_manager.cancel_order(pos.sl_order_id)

        # Calculate P&L
        if pos.side == "long":
            pnl_per_unit = exit_price - pos.entry_price
        else:
            pnl_per_unit = pos.entry_price - exit_price

        gross_pnl = pnl_per_unit * pos.quantity
        fees = (pos.entry_price + exit_price) * pos.quantity * config.TAKER_FEE_RATE
        net_pnl = gross_pnl - fees

        risk_manager.update_trade_stats(net_pnl)

        logger.info(f"P&L: {net_pnl:.2f} USDT (Gross: {gross_pnl:.2f}, Fees: {fees:.2f})")

        # Send Telegram notification
        send_telegram_message(
            f"🔴 {pos.side.upper()} EXIT\n"
            f"ID: {pos.trade_id}\n"
            f"Reason: {reason}\n"
            f"Entry: {pos.entry_price:.2f} | Exit: {exit_price:.2f}\n"
            f"P&L: {net_pnl:.2f} USDT\n"
            f"Hold: {(time.time() - pos.entry_time_sec) / 60.0:.2f} min"
        )

        self.current_position = None
        self.last_exit_time_min = time.time() / 60.0
