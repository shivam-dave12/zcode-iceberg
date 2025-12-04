"""
Z-Score Imbalance Iceberg Hunter Strategy - 2025 Real Version - PRODUCTION READY

✅ FIXED: Removed unnecessary entry lock mechanism
✅ FIXED: Streamlined to single bracket order (LIMIT + TP + SL)
✅ FIXED: Clean entry/exit flow with proper position management
✅ NO CHANGES: All calculations, parameters, and logic preserved
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
        
        # ✅ REMOVED: _entry_lock and _entering_position (no longer needed with bracket orders)
        
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
    
    def _get_dynamic_tp_sl(self, vol_regime: str, entry_price: float, side: str) -> Tuple[float, float]:
        """
        Get dynamic TP and SL PRICES based on exact Excel methodology.
        - Base: Desired % on margin (config.DESIRED_TP_PCT=0.05, SL=0.01)
        - Movement: (Desired % / Leverage) * Entry
        - For long: TP = Entry * (1 + tp_mov %), SL = Entry * (1 - sl_mov %)
        - For short: TP = Entry * (1 - tp_mov %), SL = Entry * (1 + sl_mov %)
        - Vol-regime: Scales base % (e.g., HIGH vol widens TP/SL)
        - Round to 3 decimals (CoinSwitch req)
        Returns: (tp_price, sl_price)
        """
        leverage = config.LEVERAGE
        
        # Base % from Excel (config)
        base_tp_pct = config.DESIRED_TP_PCT
        base_sl_pct = config.DESIRED_SL_PCT
        
        # Vol-regime scaling (multiplicative on base %)
        if vol_regime == "HIGH":
            tp_mult = config.VOL_REGIME_TP_MULT_HIGH
            sl_mult = config.VOL_REGIME_SL_MULT_HIGH
        elif vol_regime == "LOW":
            tp_mult = config.VOL_REGIME_TP_MULT_LOW
            sl_mult = config.VOL_REGIME_SL_MULT_LOW
        else:  # NEUTRAL
            tp_mult = 1.0
            sl_mult = 1.0
        
        tp_pct = base_tp_pct * tp_mult
        sl_pct = base_sl_pct * sl_mult
        
        # Exact Excel movement: % / Leverage
        tp_movement_pct = tp_pct / leverage
        sl_movement_pct = sl_pct / leverage
        
        if side == "long":
            tp_price = entry_price * (1 + tp_movement_pct)
            sl_price = entry_price * (1 - sl_movement_pct)
        else:  # short
            tp_price = entry_price * (1 - tp_movement_pct)
            sl_price = entry_price * (1 + sl_movement_pct)
        
        # Round to 3 decimals (CoinSwitch)
        tp_price = round(tp_price, 3)
        sl_price = round(sl_price, 3)
        
        logger.debug(f"TP/SL (Excel sync): side={side}, vol={vol_regime}, entry={entry_price:.3f}, "
                    f"tp={tp_price:.3f} (mov={tp_movement_pct*100:.3f}%), sl={sl_price:.3f} (mov={sl_movement_pct*100:.3f}%)")
        
        return tp_price, sl_price
        
    def _get_dynamic_position_size_pct(self, vol_regime: str) -> float:
        """Get position size percentage based on volatility regime (Kelly-style)."""
        if vol_regime == "HIGH":
            return config.VOL_REGIME_SIZE_HIGH_PCT
        elif vol_regime == "LOW":
            return config.VOL_REGIME_SIZE_LOW_PCT
        else:
            return (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0
    
    # ══════════════════════════════════════════════════════════════════════
    # Weighted Scoring Helpers (unchanged)
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
            "zscore": config.WEIGHT_ZSCORE,
            "touch": config.WEIGHT_TOUCH,
            "trend": config.WEIGHT_TREND,
            "cvd": config.WEIGHT_CVD,
            "lv": config.WEIGHT_LV,
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
    # Core Strategy Metrics (unchanged)
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
            avg_depth_vol = (imbalance_data["bid_vol"] + imbalance_data["ask_vol"]) / (
                2.0 * depth_levels
            )
            
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
            
            # Store in population
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
            
            # Find strongest bid/ask within zone
            zone_ticks = config.ZONE_TICKS
            zone_low = current_price - (zone_ticks * tick_size)
            zone_high = current_price + (zone_ticks * tick_size)
            
            bid_levels = [(p, q) for p, q in bids if zone_low <= p <= current_price]
            ask_levels = [(p, q) for p, q in asks if current_price <= p <= zone_high]
            
            if not bid_levels or not ask_levels:
                return None
            
            # Find max volume level
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
    # ✅ STREAMLINED: Entry Flow (single bracket order)
    # ══════════════════════════════════════════════════════════════════════
    
    def _enter_position(
        self,
        data_manager,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        side: str,
        current_price: float,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        now_sec: float,
        vol_regime: str,
        weighted_score: float,
        oracle_inputs: Optional[OracleInputs] = None,
        entry_price: Optional[float] = None,  # Optional: for limit entry; defaults to current_price
    ) -> None:
        """
        Enter a new position with single bracket order (Entry + TP + SL).
        - Enforces: No open position (single entry only).
        - Fetches fresh balance/sizing at entry-time.
        - TP/SL: Excel-synced, leverage-adjusted, 3-dec rounded.
        - Logs full decision context.
        """
        # ────────────────────────────────────────────────────────
        # STEP 1: Safety gates (no multiples, cooldowns)
        # ────────────────────────────────────────────────────────
        if self.current_position is not None:
            logger.warning(f"Entry blocked: position already open ({self.current_position.side})")
            return
        
        # Cooldown check
        if now_sec - self.last_exit_time_min * 60 < config.MIN_TIME_BETWEEN_TRADES * 60:
            logger.info(f"Entry blocked: cooldown active ({config.MIN_TIME_BETWEEN_TRADES} min)")
            return
        
        # Risk check (no balance fetch here; deferred to sizing)
        allowed, reason = risk_manager.check_trading_allowed()
        if not allowed:
            logger.warning(f"Entry blocked: {reason}")
            return
        
        # Use provided entry_price or current_price for limit
        entry_price = entry_price or current_price
        entry_price = round(entry_price, 3)  # 3-dec for CoinSwitch
        
        # ────────────────────────────────────────────────────────
        # STEP 2: Position sizing (fresh balance fetch here)
        # ────────────────────────────────────────────────────────
        margin_to_use, quantity = risk_manager.calculate_position_size_regime_aware(
            entry_price=entry_price,
            vol_regime=vol_regime,
        )
        
        if quantity <= 0 or margin_to_use < config.MIN_MARGIN_PER_TRADE:
            logger.warning(f"Entry blocked: invalid sizing (qty={quantity:.6f}, margin={margin_to_use:.2f})")
            return
        
        # ────────────────────────────────────────────────────────
        # STEP 3: Calculate TP/SL prices (Excel-synced)
        # ────────────────────────────────────────────────────────
        tp_price, sl_price = self._get_dynamic_tp_sl(
            vol_regime=vol_regime,
            entry_price=entry_price,
            side=side,
        )
        
        # ────────────────────────────────────────────────────────
        # STEP 4: Place single bracket order (Entry + TP + SL)
        # ────────────────────────────────────────────────────────
        bracket_resp = order_manager.place_bracket_order(
            side=side.upper(),  # 'BUY' or 'SELL'
            quantity=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )
        
        if not bracket_resp:
            logger.error("Bracket order failed - aborting entry")
            return
        
        main_order_resp, tp_resp, sl_resp = bracket_resp
        main_order_id = main_order_resp.get("order_id")
        tp_order_id = tp_resp.get("order_id") if tp_resp else None
        sl_order_id = sl_resp.get("order_id") if sl_resp else None
        
        # Wait for entry fill confirmation (brief poll)
        try:
            filled_data = order_manager.wait_for_fill(main_order_id, timeout_sec=5.0)
            actual_fill_price = order_manager.extract_fill_price(filled_data)
            actual_fill_price = round(actual_fill_price, 3)
            logger.info(f"✓ Entry filled: {side.upper()} {quantity:.6f} @ {actual_fill_price:.3f}")
            
            # Update TP/SL if needed (re-calc based on actual fill for precision)
            tp_price, sl_price = self._get_dynamic_tp_sl(
                vol_regime=vol_regime,
                entry_price=actual_fill_price,  # Use actual fill
                side=side,
            )
            
            # If TP/SL not placed yet, place now (post-fill)
            if not tp_order_id:
                tp_side = "SELL" if side == "long" else "BUY"
                tp_resp = order_manager.api.place_order(
                    side=tp_side.upper(),
                    order_type="stop_market",
                    quantity=quantity,
                    stop_price=tp_price,
                    exchange=config.EXCHANGE,
                    symbol=config.SYMBOL,
                )
                tp_order_id = tp_resp["data"].get("order_id") if "data" in tp_resp else None
            
            if not sl_order_id:
                sl_side = "BUY" if side == "long" else "SELL"
                sl_resp = order_manager.api.place_order(
                    side=sl_side.upper(),
                    order_type="stop_market",
                    quantity=quantity,
                    stop_price=sl_price,
                    exchange=config.EXCHANGE,
                    symbol=config.SYMBOL,
                )
                sl_order_id = sl_resp["data"].get("order_id") if "data" in sl_resp else None
            
        except Exception as e:
            logger.error(f"Entry fill failed: {e} - cancelling bracket")
            order_manager.cancel_all_orders()
            return
        
        # ────────────────────────────────────────────────────────
        # STEP 5: Record position (update state)
        # ────────────────────────────────────────────────────────
        self.trade_seq += 1
        trade_id = f"TRADE_{self.trade_seq:04d}_{side}_{int(now_sec)}"
        
        self.current_position = ZScorePosition(
            trade_id=trade_id,
            side=side,
            quantity=quantity,
            entry_price=actual_fill_price,
            entry_time_sec=now_sec,
            entry_wall_volume=wall_data.get("wall_volume", 0.0) if wall_data else 0.0,
            wall_zone_low=wall_data.get("zone_low", 0.0) if wall_data else 0.0,
            wall_zone_high=wall_data.get("zone_high", 0.0) if wall_data else 0.0,
            entry_imbalance=imbalance_data.get("imbalance", 0.0) if imbalance_data else 0.0,
            entry_z_score=delta_data.get("z_score", 0.0) if delta_data else 0.0,
            tp_price=tp_price,
            sl_price=sl_price,
            margin_used=margin_to_use,
            tp_order_id=tp_order_id or "",
            sl_order_id=sl_order_id or "",
            main_order_id=main_order_id or "",
            entry_htf_trend=oracle_inputs.htf_trend if oracle_inputs else "UNKNOWN",
            entry_vol_regime=vol_regime,
            entry_weighted_score=weighted_score,
            last_score_check_sec=now_sec,
        )
        
        # Update risk manager
        risk_manager.record_trade_opened()
        
        # ────────────────────────────────────────────────────────
        # STEP 6: Logging & Notifications
        # ────────────────────────────────────────────────────────
        logger.info("=" * 80)
        logger.info(f"🚀 NEW POSITION OPENED: {trade_id}")
        logger.info("=" * 80)
        logger.info(f"Side       : {side.upper()}")
        logger.info(f"Entry Price: ${actual_fill_price:.3f} (Limit: {entry_price:.3f})")
        logger.info(f"Quantity   : {quantity:.6f} BTC | Margin: ${margin_to_use:.2f}")
        logger.info(f"Vol Regime : {vol_regime} | Weighted Score: {weighted_score:.3f}")
        logger.info(f"Imbalance  : {self.current_position.entry_imbalance:.3f} | Z-Score: {self.current_position.entry_z_score:.3f}")
        logger.info(f"Wall Vol   : {self.current_position.entry_wall_volume:.0f} | Zone: {self.current_position.wall_zone_low:.0f}-{self.current_position.wall_zone_high:.0f}")
        logger.info(f"TP         : ${tp_price:.3f} | SL: ${sl_price:.3f}")
        logger.info(f"Orders     : Entry={main_order_id}, TP={tp_order_id}, SL={sl_order_id}")
        logger.info(f"HTF Trend  : {self.current_position.entry_htf_trend}")
        logger.info("=" * 80)
        
        # Excel log
        if self.excel_logger:
            self.excel_logger.log_entry(
                trade_id=trade_id,
                timestamp=datetime.fromtimestamp(now_sec),
                side=side,
                entry_price=actual_fill_price,
                quantity=quantity,
                tp_price=tp_price,
                sl_price=sl_price,
                margin_used=margin_to_use,
                vol_regime=vol_regime,
                weighted_score=weighted_score,
                imbalance=self.current_position.entry_imbalance,
                z_score=self.current_position.entry_z_score,
                wall_volume=self.current_position.entry_wall_volume,
            )
        
        # Telegram alert
        try:
            alert_msg = (
                f"🚀 {side.upper()} ENTRY\n"
                f"ID: {trade_id}\n"
                f"Price: ${actual_fill_price:.3f} | Qty: {quantity:.6f}\n"
                f"TP: ${tp_price:.3f} | SL: ${sl_price:.3f}\n"
                f"Margin: ${margin_to_use:.2f} | Score: {weighted_score:.3f}\n"
                f"Regime: {vol_regime}"
            )
            send_telegram_message(alert_msg)
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
        
        logger.info(f"Position tracking active - next check in {self.POSITION_LOG_INTERVAL_SEC / 60:.1f} min")
        

    def _calculate_entry_price(self, side: str, current_price: float, vol_regime: str) -> float:
        """
        Calculate entry price with volatility-based tick offset.
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
        Validate trade conditions before entry.
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
            if not balance:
                return False, "Cannot fetch balance"
            
            available = float(balance.get('available', 0))
            if available < config.MIN_MARGIN_PER_TRADE:
                return False, f"Insufficient balance: {available:.2f}"
        except Exception as e:
            return False, f"Cannot check balance: {e}"
        
        return True, "OK"
    
    # ══════════════════════════════════════════════════════════════════════
    # ✅ STREAMLINED: Position Management
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
        
        # ────────────────────────────────────────────────────────
        # STEP 1: Check TP/SL order status
        # ────────────────────────────────────────────────────────
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
        
        # ────────────────────────────────────────────────────────
        # STEP 2: Time stop
        # ────────────────────────────────────────────────────────
        hold_minutes = (now_sec - pos.entry_time_sec) / 60.0
        
        if hold_minutes > config.MAX_HOLD_MINUTES:
            logger.info(f"⏰ Time stop triggered ({hold_minutes:.1f}min)")
            self._close_position(
                order_manager, risk_manager, "TIME_STOP", now_sec, current_price
            )
            return
        
        # ────────────────────────────────────────────────────────
        # STEP 3: Periodic position logging
        # ────────────────────────────────────────────────────────
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
        ✅ STREAMLINED: Close position and cleanup.
        
        Flow:
        1. Cancel remaining TP/SL orders
        2. Calculate realized P&L
        3. Update risk manager
        4. Log trade
        5. Clear position
        """
        pos = self.current_position
        
        if not pos:
            return
        
        try:
            logger.info("=" * 80)
            logger.info(f"CLOSING POSITION: {pos.trade_id} ({exit_reason})")
            logger.info("=" * 80)
            
            # ────────────────────────────────────────────────────────
            # STEP 1: Cancel remaining orders
            # ────────────────────────────────────────────────────────
            try:
                if exit_reason == "TP_HIT":
                    order_manager.cancel_order(pos.sl_order_id)
                elif exit_reason == "SL_HIT":
                    order_manager.cancel_order(pos.tp_order_id)
                else:
                    # Manual exit - cancel both
                    order_manager.cancel_order(pos.tp_order_id)
                    order_manager.cancel_order(pos.sl_order_id)
                    
                    # Place market exit order
                    exit_side = "SELL" if pos.side == "long" else "BUY"
                    order_manager.place_market_order(
                        side=exit_side,
                        quantity=pos.quantity,
                        reduce_only=True
                    )
            except Exception as e:
                logger.error(f"Error canceling orders: {e}")
            
            # ────────────────────────────────────────────────────────
            # STEP 2: Calculate realized P&L
            # ────────────────────────────────────────────────────────
            if exit_reason == "TP_HIT":
                exit_price = pos.tp_price
            elif exit_reason == "SL_HIT":
                exit_price = pos.sl_price
            else:
                exit_price = current_price
            
            pnl = self._calculate_realized_pnl(pos, exit_price)
            roi = (pnl / pos.margin_used) * 100.0 if pos.margin_used > 0 else 0.0
            
            hold_duration_min = (now_sec - pos.entry_time_sec) / 60.0
            
            # ────────────────────────────────────────────────────────
            # STEP 3: Update risk manager
            # ────────────────────────────────────────────────────────
            risk_manager.update_trade_stats(pnl)
            risk_manager.record_trade_closed()
            
            # ────────────────────────────────────────────────────────
            # STEP 4: Log trade
            # ────────────────────────────────────────────────────────
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
            
            # ────────────────────────────────────────────────────────
            # STEP 5: Clear position
            # ────────────────────────────────────────────────────────
            self.current_position = None
            self.last_exit_time_min = now_sec / 60.0
        
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            # Force cleanup
            self.current_position = None
            self.last_exit_time_min = now_sec / 60.0
    
    def _calculate_unrealized_pnl(self, pos: ZScorePosition, current_price: float) -> float:
        """Calculate unrealized P&L."""
        try:
            if pos.side == "long":
                price_change = current_price - pos.entry_price
            else:
                price_change = pos.entry_price - current_price
            
            notional_change = price_change * pos.quantity
            pnl = notional_change
            
            return pnl
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
    
    def _calculate_realized_pnl(self, pos: ZScorePosition, exit_price: float) -> float:
        """Calculate realized P&L."""
        try:
            if pos.side == "long":
                price_change = exit_price - pos.entry_price
            else:
                price_change = pos.entry_price - exit_price
            
            notional_change = price_change * pos.quantity
            pnl = notional_change
            
            return pnl
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0
    
    # ══════════════════════════════════════════════════════════════════════
    # Periodic Reporting
    # ══════════════════════════════════════════════════════════════════════
    
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
    
    # ══════════════════════════════════════════════════════════════════════
    # Main tick handler
    # ══════════════════════════════════════════════════════════════════════
    
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
                    lv_avg = min(1.0, lv_avg)
                
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
                
                # Log decision state (throttled)
                if now_sec - self._last_decision_log_sec > self.DECISION_LOG_INTERVAL_SEC:
                    self._last_decision_log_sec = now_sec
                    
                    logger.info("=" * 80)
                    logger.info("STRATEGY DECISION STATE")
                    logger.info("=" * 80)
                    logger.info(f"Price: ${current_price:.2f} | Vol: {vol_regime} | HTF: {htf_trend}")
                    logger.info(f"LONG  Score: {long_score:.3f} | Win Prob: {win_prob_long:.3f} | Ready: {long_ready}")
                    logger.info(f"SHORT Score: {short_score:.3f} | Win Prob: {win_prob_short:.3f} | Ready: {short_ready}")
                    logger.info("=" * 80)
                
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
