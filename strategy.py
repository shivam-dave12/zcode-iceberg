"""
Z-Score Strategy with LIMIT order entry and new position management

CHANGES:
1. LIMIT order entry with volatility-based price offset
2. New position management logic (volatile vs non-volatile)
3. Calculations every 50ms, logging every 60s
4. Entry lock to prevent multiple simultaneous orders
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
    """Position tracking dataclass"""
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
    vol_regime: str
    entry_score: float
    is_volatile: bool
    tp_roi: float
    sl_roi: float
    last_momentum_check_sec: float
    last_momentum_log_sec: float
    tp_adjustment_count: int


class ZScoreIcebergHunterStrategy:
    """
    Z-Score Iceberg Hunter Strategy
    Industry-grade implementation with proper error handling
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

        logger.info("=" * 80)
        logger.info("Z-SCORE STRATEGY INITIALIZED")
        logger.info("=" * 80)

    # ======================================================================
    # MARGIN-BASED TP/SL CALCULATION
    # ======================================================================

    def _calculate_tp_sl_prices(
        self,
        entry_price: float,
        margin_used: float,
        quantity: float,
        side: str,
        desired_profit_roi: float,
        desired_sl_roi: float,
    ) -> Tuple[float, float]:
        """Calculate TP/SL based on margin methodology with validation"""
        try:
            logger.info("=" * 80)
            logger.info("[TP/SL CALCULATION] Using margin-based formula")
            logger.info(f"  Entry Price: {entry_price:.2f}")
            logger.info(f"  Margin Used: {margin_used:.2f} USDT")
            logger.info(f"  Quantity: {quantity:.6f} BTC")
            logger.info(f"  Side: {side.upper()}")
            logger.info(f"  Desired TP ROI: {desired_profit_roi*100:.2f}%")
            logger.info(f"  Desired SL ROI: {desired_sl_roi*100:.2f}%")

            if entry_price <= 0 or margin_used <= 0 or quantity <= 0:
                raise ValueError("Invalid input: prices/margin/quantity must be positive")

            tp_price_movement = (margin_used * desired_profit_roi) / quantity
            sl_price_movement = (margin_used * desired_sl_roi) / quantity

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
            logger.error(f"Error calculating TP/SL: {e}")
            if side == "long":
                return entry_price * 1.01, entry_price * 0.99
            else:
                return entry_price * 0.99, entry_price * 1.01

    # ======================================================================
    # VOLATILITY REGIME
    # ======================================================================

    def _get_regime_params(self, data_manager, current_price: float) -> Dict:
        """Get dynamic params based on vol regime with error handling"""
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception as e:
            logger.debug(f"Error getting ATR%: {e}")
            atr_pct = None

        try:
            regime = data_manager.get_vol_regime(atr_pct) if hasattr(data_manager, 'get_vol_regime') else "NEUTRAL"
        except Exception as e:
            logger.debug(f"Error getting vol regime: {e}")
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

    # ======================================================================
    # WEIGHTED SCORE CALCULATION
    # ======================================================================

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
            logger.debug(f"[AETHER] LV data unavailable: 1m={oracle_inputs.lv_1m}, 5m={oracle_inputs.lv_5m}, 15m={oracle_inputs.lv_15m}")
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

    # ======================================================================
    # MAIN TICK HANDLER
    # ======================================================================

    def on_tick(self, data_manager, order_manager, risk_manager) -> None:
        """Main per-tick strategy."""
        try:
            current_price = data_manager.get_last_price()
            if current_price <= 0:
                return

            now_sec = time.time()

            if self.current_position is not None:
                self._manage_open_position(
                    data_manager=data_manager,
                    order_manager=order_manager,
                    risk_manager=risk_manager,
                    current_price=current_price,
                    now_sec=now_sec,
                )
                return

            if self.pending_entry:
                return

            if self.last_exit_time_min > 0:
                minutes_since_exit = (now_sec / 60.0) - self.last_exit_time_min
                if minutes_since_exit < config.MIN_TIME_BETWEEN_TRADES:
                    return

            allowed, reason = risk_manager.check_trading_allowed()
            if not allowed:
                return

            imbalance_data = self._compute_imbalance(data_manager)
            wall_data = (
                self._compute_wall_strength(data_manager, current_price, imbalance_data)
                if imbalance_data is not None
                else None
            )
            delta_data = self._compute_delta_z_score(data_manager)
            touch_data = self._compute_price_touch(data_manager, current_price)

            htf_trend: Optional[str] = None
            try:
                if hasattr(data_manager, "get_htf_trend"):
                    htf_trend = data_manager.get_htf_trend()
            except Exception as e:
                logger.error(f"Error fetching HTF trend: {e}")

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
    # ENTRY DECISION WITH COMPREHENSIVE LOGGING
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
        """Evaluate LONG/SHORT every tick, log every minute."""
        if not all([imbalance_data, wall_data, delta_data, touch_data]):
            return

        if self.pending_entry:
            return

        regime_params = self._get_regime_params(data_manager, current_price)
        regime = regime_params["regime"]
        z_thresh = regime_params["z_thresh"]
        wall_mult = regime_params["wall_mult"]
        atr_pct = regime_params.get("atr_pct", 0.0)

        ema_val: Optional[float] = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except:
            pass

        trend_long_ok = (ema_val is not None and current_price > ema_val)
        trend_short_ok = (ema_val is not None and current_price < ema_val)

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

        long_entry = (long_total > config.SCORE_ENTRY_THRESHOLD)
        short_entry = (short_total > config.SCORE_ENTRY_THRESHOLD)

        should_log = (now_sec - self._last_decision_log_sec >= self.DECISION_LOG_INTERVAL_SEC)
        
        if should_log:
            self._last_decision_log_sec = now_sec
            
            logger.info("\n" + "=" * 100)
            logger.info(f"[DECISION REPORT] {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info("=" * 100)
            logger.info(f"PRICE: {current_price:.2f} USDT")
            logger.info(f"")
            logger.info(f"[VOL REGIME]")
            logger.info(f"  Regime: {regime}")
            logger.info(f"  ATR%: {atr_pct*100:.3f}% (LOW<{config.VOL_REGIME_LOW_ATR_PCT*100:.2f}%, HIGH>{config.VOL_REGIME_HIGH_ATR_PCT*100:.2f}%)")
            logger.info(f"  Z Threshold: {z_thresh:.2f}")
            logger.info(f"  Wall Multiplier: {wall_mult:.2f}")
            logger.info(f"")
            logger.info(f"[CORE METRICS]")
            logger.info(f"  Imbalance: {imbalance_data['imbalance']:.3f} (threshold: ±{config.IMBALANCE_THRESHOLD})")
            logger.info(f"    Total Bid: {imbalance_data['total_bid']:.2f} BTC")
            logger.info(f"    Total Ask: {imbalance_data['total_ask']:.2f} BTC")
            logger.info(f"  Wall Strength:")
            logger.info(f"    Bid: {wall_data['bid_wall_strength']:.2f}x (volume: {wall_data['bid_vol_zone']:.2f} BTC)")
            logger.info(f"    Ask: {wall_data['ask_wall_strength']:.2f}x (volume: {wall_data['ask_vol_zone']:.2f} BTC)")
            logger.info(f"  Z-Score: {delta_data['z_score']:.2f} (threshold: ±{z_thresh:.2f})")
            logger.info(f"    Buy Volume: {delta_data['buy_vol']:.4f} BTC")
            logger.info(f"    Sell Volume: {delta_data['sell_vol']:.4f} BTC")
            logger.info(f"    Delta: {delta_data['delta']:.4f} BTC")
            logger.info(f"  Price Touch:")
            logger.info(f"    Bid Distance: {touch_data['bid_distance_ticks']:.1f} ticks (threshold: {config.PRICE_TOUCH_THRESHOLD_TICKS})")
            logger.info(f"    Ask Distance: {touch_data['ask_distance_ticks']:.1f} ticks")
            logger.info(f"  Trend:")
            logger.info(f"    EMA{config.EMA_PERIOD}: {ema_val:.2f}" if ema_val else "    EMA: N/A")
            logger.info(f"    HTF Trend: {htf_trend or 'N/A'}")
            logger.info(f"    Long Trend OK: {trend_long_ok}")
            logger.info(f"    Short Trend OK: {trend_short_ok}")
            logger.info(f"")
            logger.info(f"[SCORE BREAKDOWN]")
            logger.info(f"  LONG:")
            logger.info(f"    Core (65%): {long_core:.3f} | {' | '.join(long_core_reasons)}")
            logger.info(f"    Aether (35%): {long_aether:.3f} | {' | '.join(long_aether_reasons)}")
            logger.info(f"    TOTAL: {long_total:.3f} (threshold: {config.SCORE_ENTRY_THRESHOLD})")
            logger.info(f"    → {'✓ ENTRY SIGNAL' if long_entry else '✗ NO ENTRY'}")
            logger.info(f"  SHORT:")
            logger.info(f"    Core (65%): {short_core:.3f} | {' | '.join(short_core_reasons)}")
            logger.info(f"    Aether (35%): {short_aether:.3f} | {' | '.join(short_aether_reasons)}")
            logger.info(f"    TOTAL: {short_total:.3f} (threshold: {config.SCORE_ENTRY_THRESHOLD})")
            logger.info(f"    → {'✓ ENTRY SIGNAL' if short_entry else '✗ NO ENTRY'}")
            logger.info("=" * 100 + "\n")

        if long_entry:
            self.pending_entry = True
            self._enter_position(
                data_manager, order_manager, risk_manager, "long", current_price,
                imbalance_data, wall_data, delta_data, touch_data, now_sec,
                regime_params, long_total
            )
        elif short_entry:
            self.pending_entry = True
            self._enter_position(
                data_manager, order_manager, risk_manager, "short", current_price,
                imbalance_data, wall_data, delta_data, touch_data, now_sec,
                regime_params, short_total
            )

    # ======================================================================
    # POSITION ENTRY (WITH LIMIT ORDER)
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
        """Enter position with LIMIT order and volatility-based entry price."""
        try:
            regime = regime_params["regime"]
            atr_pct = regime_params.get("atr_pct", 0.0)

            is_volatile = (atr_pct is not None and atr_pct > config.VOLATILE_ATR_THRESHOLD)

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

            if is_volatile:
                tp_roi = config.PROFIT_TARGET_ROI
                sl_roi = config.STOP_LOSS_ROI
            else:
                tp_mult = regime_params["tp_mult"]
                sl_mult = regime_params["sl_mult"]
                tp_roi = config.PROFIT_TARGET_ROI * tp_mult
                sl_roi = config.STOP_LOSS_ROI * sl_mult

            logger.info("\n" + "=" * 100)
            logger.info(f"[ENTRY EXECUTION] {side.upper()} - LIMIT ORDER")
            logger.info("=" * 100)
            logger.info(f"  Regime: {regime} | Volatile: {is_volatile}")
            logger.info(f"  Entry Score: {entry_score:.3f}")
            logger.info(f"  Current Price: {current_price:.2f}")
            logger.info(f"  Limit Entry Price: {limit_entry_price:.2f} (offset: {offset_ticks} ticks)")
            logger.info(f"  Quantity: {quantity:.6f} BTC")
            logger.info(f"  Margin: {margin_used:.2f} USDT")
            logger.info(f"  TP ROI: {tp_roi*100:.2f}%")
            logger.info(f"  SL ROI: {sl_roi*100:.2f}%")
            logger.info("=" * 100)

            market_side = "BUY" if side == "long" else "SELL"
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

            try:
                filled_order = order_manager.wait_for_fill(
                    main_order_id, 
                    timeout_sec=config.LIMIT_ORDER_WAIT_TIMEOUT_SEC
                )
                entry_price = order_manager.extract_fill_price(filled_order)
                logger.info(f"✓ Limit order filled at {entry_price:.2f}")
            except Exception as e:
                logger.error(f"Limit order fill failed or timeout: {e}")
                order_manager.cancel_order(main_order_id)
                logger.warning("Limit order cancelled due to timeout")
                self.pending_entry = False
                return

            tp_price, sl_price = self._calculate_tp_sl_prices(
                entry_price=entry_price,
                margin_used=margin_used,
                quantity=quantity,
                side=side,
                desired_profit_roi=tp_roi,
                desired_sl_roi=sl_roi,
            )

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
                self.pending_entry = False
                return

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
                is_volatile=is_volatile,
                tp_roi=tp_roi,
                sl_roi=sl_roi,
                last_momentum_check_sec=now_sec,
                last_momentum_log_sec=now_sec,
                tp_adjustment_count=0,
            )

            risk_manager.record_trade_opened()
            self.pending_entry = False

            msg = (
                f"🚀 {side.upper()} ENTRY (LIMIT)\n"
                f"Price: {entry_price:.2f} | Qty: {quantity:.6f}\n"
                f"TP: {tp_price:.2f} ({tp_roi*100:.1f}%) | SL: {sl_price:.2f} ({sl_roi*100:.1f}%)\n"
                f"Volatile: {is_volatile} | Score: {entry_score:.3f}"
            )

            try:
                send_telegram_message(msg)
            except:
                pass

            logger.info(f"✓ Position opened: {self.current_position.trade_id}\n")

        except Exception as e:
            logger.error(f"Error in _enter_position: {e}", exc_info=True)
            self.pending_entry = False

    # ======================================================================
    # POSITION MANAGEMENT (NEW LOGIC)
    # ======================================================================

    def _manage_open_position(
        self,
        data_manager,
        order_manager,
        risk_manager,
        current_price: float,
        now_sec: float,
    ) -> None:
        """Manage open position with NEW logic."""
        pos = self.current_position
        hold_sec = now_sec - pos.entry_time_sec
        hold_min = hold_sec / 60.0

        if now_sec - self._last_position_log_sec >= self.POSITION_LOG_INTERVAL_SEC:
            self._last_position_log_sec = now_sec
            direction = 1.0 if pos.side == "long" else -1.0
            current_profit_pct = ((current_price - pos.entry_price) / pos.entry_price) * direction
            upnl = (current_price - pos.entry_price) * direction * pos.quantity

            logger.info("\n" + "=" * 100)
            logger.info(f"[POSITION STATUS] {pos.trade_id}")
            logger.info("=" * 100)
            logger.info(f"  Side: {pos.side.upper()}")
            logger.info(f"  Entry: {pos.entry_price:.2f} | Current: {current_price:.2f}")
            logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
            logger.info(f"  Margin: {pos.margin_used:.2f} USDT")
            logger.info(f"  TP: {pos.tp_price:.2f} ({pos.tp_roi*100:.2f}%) | SL: {pos.sl_price:.2f} ({pos.sl_roi*100:.2f}%)")
            logger.info(f"  Hold Time: {hold_min:.1f} min")
            logger.info(f"  Unrealized P&L: {upnl:.2f} USDT ({current_profit_pct*100:.2f}%)")
            logger.info(f"  Volatile: {pos.is_volatile} | TP Adjustments: {pos.tp_adjustment_count}")
            logger.info("=" * 100 + "\n")

        if now_sec - self._last_status_check_sec >= self.ORDER_STATUS_CHECK_INTERVAL_SEC:
            self._last_status_check_sec = now_sec
            self._check_bracket_exits(order_manager, risk_manager, current_price, now_sec)

        direction = 1.0 if pos.side == "long" else -1.0
        current_profit_pct = ((current_price - pos.entry_price) / pos.entry_price) * direction

        if now_sec - pos.last_momentum_check_sec >= config.POSITION_CHECK_INTERVAL_SEC:
            pos.last_momentum_check_sec = now_sec

            momentum_favorable, vol_favorable, trend_favorable = self._check_market_conditions(
                data_manager, pos.side, current_price
            )

            if now_sec - pos.last_momentum_log_sec >= config.MOMENTUM_LOG_INTERVAL_SEC:
                pos.last_momentum_log_sec = now_sec
                logger.info(f"[MOMENTUM CHECK] {pos.trade_id} | Hold={hold_min:.1f}min | "
                           f"Profit={current_profit_pct*100:.2f}% | "
                           f"Momentum={momentum_favorable} Vol={vol_favorable} Trend={trend_favorable}")

            if hold_min >= config.FIRST_TP_WAIT_MINUTES and pos.tp_adjustment_count == 0:
                self._manage_tp_new_logic_10min(
                    data_manager, order_manager, current_price, now_sec,
                    momentum_favorable, vol_favorable, trend_favorable,
                    current_profit_pct
                )

            elif hold_min >= (config.FIRST_TP_WAIT_MINUTES + config.SECOND_TP_WAIT_MINUTES) and pos.tp_adjustment_count == 1:
                self._manage_tp_new_logic_15min(
                    order_manager, current_price, current_profit_pct
                )

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

    def _manage_tp_new_logic_10min(
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
        """NEW LOGIC: After 10 minutes, adjust TP based on conditions."""
        pos = self.current_position
        if pos is None:
            return

        half_tp = pos.tp_roi * config.HALF_TP_THRESHOLD
        all_favorable = momentum_favorable and vol_favorable and trend_favorable

        logger.info("=" * 100)
        logger.info(f"[TP MANAGEMENT 10MIN] {pos.trade_id}")
        logger.info(f"  All favorable (M/V/T): {all_favorable}")
        logger.info(f"  Current profit: {current_profit_pct*100:.2f}%")
        logger.info(f"  Half TP: {half_tp*100:.2f}%")

        if all_favorable:
            if current_profit_pct >= half_tp:
                self._move_sl_to_half_tp(order_manager, current_price)
                logger.info("  → Moved SL to half TP, waiting 5min more")
            else:
                logger.info("  → Waiting 5min more")
        else:
            if current_profit_pct >= half_tp:
                new_tp_roi = current_profit_pct + config.TP_BUFFER_PERCENT
                self._adjust_tp_order(order_manager, current_price, new_tp_roi)
                logger.info(f"  → New TP set to {new_tp_roi*100:.2f}% (near current profit)")
            else:
                new_tp_roi = half_tp
                self._adjust_tp_order(order_manager, current_price, new_tp_roi)
                logger.info(f"  → New TP set to half TP ({new_tp_roi*100:.2f}%)")

        logger.info("=" * 100)
        pos.tp_adjustment_count = 1

    def _manage_tp_new_logic_15min(
        self,
        order_manager,
        current_price: float,
        current_profit_pct: float,
    ) -> None:
        """NEW LOGIC: After 15 minutes (10+5), final TP tighten."""
        pos = self.current_position
        if pos is None:
            return

        new_tp_roi = current_profit_pct + config.TP_BUFFER_PERCENT
        self._adjust_tp_order(order_manager, current_price, new_tp_roi)
        logger.info(f"[TP MANAGEMENT 15MIN] Final tightening TP to {new_tp_roi*100:.2f}%")
        pos.tp_adjustment_count = 2

    def _move_sl_to_half_tp(self, order_manager, current_price: float) -> None:
        """Move SL to half TP price."""
        pos = self.current_position
        if pos is None or pos.tp_reduced:
            return

        half_tp_roi = pos.tp_roi * config.HALF_TP_THRESHOLD
        direction = 1.0 if pos.side == "long" else -1.0
        price_movement = pos.entry_price * half_tp_roi
        new_sl_price = pos.entry_price + (price_movement * direction)

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
            logger.info(f"✓ Moved SL to half TP: {new_sl_price:.2f}")

    def _adjust_tp_order(self, order_manager, current_price: float, new_tp_roi: float) -> None:
        """Adjust TP order to new ROI."""
        pos = self.current_position
        if pos is None:
            return

        direction = 1.0 if pos.side == "long" else -1.0
        price_movement = pos.entry_price * new_tp_roi
        new_tp_price = pos.entry_price + (price_movement * direction)

        order_manager.cancel_order(pos.tp_order_id)
        tp_side = "SELL" if pos.side == "long" else "BUY"

        new_tp_order = order_manager.place_take_profit(
            side=tp_side,
            quantity=pos.quantity,
            trigger_price=new_tp_price,
        )

        if new_tp_order:
            pos.tp_order_id = new_tp_order.get("order_id", "")
            pos.tp_price = new_tp_price
            pos.tp_roi = new_tp_roi
            logger.info(f"✓ Adjusted TP to: {new_tp_price:.2f} ({new_tp_roi*100:.2f}%)")

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

        order_manager.cancel_order(pos.tp_order_id)
        order_manager.cancel_order(pos.sl_order_id)

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

        direction = 1.0 if pos.side == "long" else -1.0
        pnl = (exit_price - pos.entry_price) * direction * pos.quantity
        roi = pnl / pos.margin_used if pos.margin_used > 0 else 0.0

        risk_manager.update_trade_stats(pnl)

        logger.info("\n" + "=" * 100)
        logger.info(f"[POSITION CLOSED] {pos.trade_id} | {reason}")
        logger.info("=" * 100)
        logger.info(f"  Side: {pos.side.upper()}")
        logger.info(f"  Entry: {pos.entry_price:.2f} | Exit: {exit_price:.2f}")
        logger.info(f"  Quantity: {pos.quantity:.6f} BTC")
        logger.info(f"  P&L: {pnl:.2f} USDT ({roi*100:.2f}%)")
        logger.info(f"  Hold Time: {(now_sec - pos.entry_time_sec)/60.0:.1f} min")
        logger.info("=" * 100 + "\n")

        msg = (
            f"🛑 EXIT {pos.side.upper()} | {reason}\n"
            f"Entry: {pos.entry_price:.2f} → Exit: {exit_price:.2f}\n"
            f"P&L: {pnl:.2f} USDT ({roi*100:.2f}%)\n"
            f"Hold: {(now_sec - pos.entry_time_sec)/60.0:.1f}min"
        )

        try:
            send_telegram_message(msg)
        except:
            pass

        self.last_exit_time_min = now_sec / 60.0
        self.current_position = None
        self.pending_entry = False

    # ======================================================================
    # CORE METRIC CALCULATORS
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
