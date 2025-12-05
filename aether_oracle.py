"""
Aether Oracle - 9-Signal Data Fusion for Z-Score Strategy
Industry-grade refactored version with proper error handling
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import config

logger = logging.getLogger(__name__)


@dataclass
class OracleInputs:
    """All inputs required for oracle decision making"""
    # Core gates
    imbalance_data: Optional[Dict] = None
    wall_data: Optional[Dict] = None
    delta_data: Optional[Dict] = None
    touch_data: Optional[Dict] = None
    htf_trend: Optional[str] = None
    ltf_trend: Optional[str] = None
    ema_val: Optional[float] = None
    atr_pct: Optional[float] = None
    
    # Advanced metrics (9-signal fusion)
    lv_1m: Optional[float] = None
    lv_5m: Optional[float] = None
    lv_15m: Optional[float] = None
    micro_trap: bool = False
    norm_cvd: Optional[float] = None
    hurst: Optional[float] = None
    bos_align: Optional[float] = None  # FIXED: was bos_signal
    
    # LSTM predictions
    lstm_1m: Optional[float] = None
    lstm_5m: Optional[float] = None
    lstm_15m: Optional[float] = None
    
    # Meta
    current_price: float = 0.0
    now_sec: float = 0.0


@dataclass
class OracleSideScores:
    """Scoring output for one side (long/short)"""
    side: str
    mc: Optional[float] = None
    bayes: Optional[float] = None
    rl: Optional[float] = None
    fused: Optional[float] = None
    kelly_f: float = 0.0
    rr: float = 3.33
    reasons: List[str] = field(default_factory=list)


@dataclass
class OracleOutputs:
    """Complete oracle decision output"""
    long_scores: OracleSideScores
    short_scores: OracleSideScores


class AetherOracle:
    """
    Aether Oracle: 9-signal data fusion.
    Signals: Imb, Wall, Z, Touch, Trend (core 5) + CVD, LV, Hurst/BOS, LSTM (aether 4).
    """

    def __init__(self) -> None:
        np.random.seed(42)
        self.entry_prob_threshold: float = 0.70
        self.rr_assumed: float = 3.33
        self.mc_paths: int = 100
        self._prob_min: float = 0.01
        self._prob_max: float = 0.99
        logger.info("AetherOracle initialized")

    # ======================================================================
    # UTILITY FUNCTIONS
    # ======================================================================

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function with overflow protection"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def _clamp_prob(self, p: float) -> float:
        """Clamp probability to valid range"""
        return max(self._prob_min, min(self._prob_max, p))

    # ======================================================================
    # FUSION COMPONENTS
    # ======================================================================

    def _compute_mc_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        atr_percent: Optional[float],
    ) -> Optional[float]:
        """
        Monte Carlo component: simulate paths ~ N(mu_cvd, sigma_atr).
        """
        if (
            imbalance_val is None
            or norm_cvd is None
            or atr_percent is None
            or atr_percent <= 0.0
        ):
            return None
        
        try:
            sign_side = 1.0 if side == "long" else -1.0
            mu = float(norm_cvd) * sign_side
            sigma = max(float(atr_percent), 1e-4)
            
            paths = np.random.normal(loc=mu, scale=sigma, size=self.mc_paths)
            sign_I = 1.0 if imbalance_val >= 0 else -1.0
            
            positives = float(np.sum(paths * sign_I > 0))
            mc = positives / float(self.mc_paths)
            
            return self._clamp_prob(mc)
        except Exception as e:
            logger.error(f"Error in MC component: {e}")
            return None

    def _compute_bayes_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        bos_align: Optional[float],
        hurst: Optional[float],
    ) -> Optional[float]:
        """
        Bayesian psy score via sigmoid.
        """
        if norm_cvd is None or imbalance_val is None:
            return None
        
        try:
            sign_side = 1.0 if side == "long" else -1.0
            f1 = float(norm_cvd) * sign_side
            f2 = float(imbalance_val) * sign_side
            
            if bos_align is not None:
                f3 = bos_align
                w3 = 1.0
            else:
                f3 = 0.0
                w3 = 0.0
            
            if hurst is not None:
                f4 = max(float(hurst) - 0.5, 0.0)
            else:
                f4 = 0.0
            
            psy = 2.5 * f1 + 1.5 * f2 + w3 * f3 + 1.0 * f4
            bayes_raw = self._sigmoid(psy)
            
            return self._clamp_prob(bayes_raw)
        except Exception as e:
            logger.error(f"Error in Bayes component: {e}")
            return None

    def _compute_rl_component(
        self, side: str, risk_manager, hurst: Optional[float]
    ) -> Optional[float]:
        """
        RL proxy: Hurst < 0.5 = mean reverting (good for walls).
        """
        if hurst is None:
            return None
        
        try:
            score = 0.5 + (0.5 - float(hurst))
            return max(0.01, min(0.99, score))
        except Exception as e:
            logger.error(f"Error in RL component: {e}")
            return None

    def _compute_kelly_fraction(self, p: float) -> float:
        """
        Kelly sizing: f = (p*b - q) / b, capped to 2%.
        """
        try:
            p = max(0.0, min(1.0, p))
            q = 1.0 - p
            b = self.rr_assumed
            edge = p * b - q
            f = edge / b
            return max(0.0, min(0.02, f))
        except Exception as e:
            logger.error(f"Error in Kelly fraction: {e}")
            return 0.0

    # ======================================================================
    # HIGH-LEVEL API
    # ======================================================================

    def build_inputs(
        self,
        data_manager,
        current_price: float,
        imbalance_data: Optional[Dict],
        wall_data: Optional[Dict],
        delta_data: Optional[Dict],
        touch_data: Optional[Dict],
        htf_trend: Optional[str],
        ltf_trend: Optional[str],
    ) -> OracleInputs:
        """Build oracle inputs with comprehensive error handling"""
        
        # Normalized CVD
        norm_cvd = None
        try:
            cvd = data_manager.get_cumulative_volume_delta()
            if cvd is not None:
                norm_cvd = max(-1.0, min(1.0, cvd / 100.0))
        except Exception as e:
            logger.debug(f"Error calculating CVD: {e}")
        
        # Liquidity Volatility
        lv_1m = self._safe_get_lv(data_manager, "1m")
        lv_5m = self._safe_get_lv(data_manager, "5m")
        lv_15m = self._safe_get_lv(data_manager, "15m")
        
        # Hurst Exponent
        hurst = None
        try:
            hurst = data_manager.get_hurst_exponent()
        except Exception as e:
            logger.debug(f"Error calculating Hurst: {e}")
        
        # BOS Alignment (FIXED: renamed from bos_signal to bos_align)
        bos_align = None
        try:
            bos_signal = data_manager.get_bos_signal()
            if bos_signal is not None:
                # Convert BOS signal (-1, 0, 1) to alignment score [0, 1]
                bos_align = (float(bos_signal) + 1.0) / 2.0
        except Exception as e:
            logger.debug(f"Error calculating BOS: {e}")
        
        # LSTM Predictions
        lstm_1m = self._safe_get_lstm(data_manager, "1m")
        lstm_5m = self._safe_get_lstm(data_manager, "5m")
        lstm_15m = self._safe_get_lstm(data_manager, "15m")
        
        # ATR
        atr_pct = None
        try:
            atr_pct = data_manager.get_atr_percent()
        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
        
        # EMA
        ema_val = None
        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception as e:
            logger.debug(f"Error calculating EMA: {e}")
        
        return OracleInputs(
            current_price=current_price,
            imbalance_data=imbalance_data,
            wall_data=wall_data,
            delta_data=delta_data,
            touch_data=touch_data,
            htf_trend=htf_trend,
            ltf_trend=ltf_trend,
            ema_val=ema_val,
            atr_pct=atr_pct,
            norm_cvd=norm_cvd,
            lv_1m=lv_1m,
            lv_5m=lv_5m,
            lv_15m=lv_15m,
            hurst=hurst,
            bos_align=bos_align,
            lstm_1m=lstm_1m,
            lstm_5m=lstm_5m,
            lstm_15m=lstm_15m,
            now_sec=time.time(),
        )

    def _safe_get_lv(self, data_manager, timeframe: str) -> Optional[float]:
        """Safely get liquidity volatility with error handling"""
        try:
            lv = data_manager.get_liquidity_volatility(timeframe=timeframe)
            logger.debug(f"[LV {timeframe}] = {lv}")
            return lv
        except Exception as e:
            logger.debug(f"Error getting LV {timeframe}: {e}")
            return None

    def _safe_get_lstm(self, data_manager, timeframe: str) -> Optional[float]:
        """Safely get LSTM prediction with error handling"""
        try:
            return data_manager.get_lstm_prediction(timeframe=timeframe)
        except Exception as e:
            logger.debug(f"Error getting LSTM {timeframe}: {e}")
            return None

    def compute_side_scores(
        self,
        side: str,
        inputs: OracleInputs,
        risk_manager,
    ) -> OracleSideScores:
        """
        Compute MC, Bayes, RL + fused score + Kelly for one side.
        """
        try:
            imbalance_val = None
            if inputs.imbalance_data is not None:
                imbalance_val = float(inputs.imbalance_data.get("imbalance", 0.0))
            
            mc = self._compute_mc_component(
                side=side,
                imbalance_val=imbalance_val,
                norm_cvd=inputs.norm_cvd,
                atr_percent=inputs.atr_pct,
            )
            
            bayes = self._compute_bayes_component(
                side=side,
                imbalance_val=imbalance_val,
                norm_cvd=inputs.norm_cvd,
                bos_align=inputs.bos_align,
                hurst=inputs.hurst,
            )
            
            rl = self._compute_rl_component(
                side=side,
                risk_manager=risk_manager,
                hurst=inputs.hurst
            )
            
            # Fusion
            components: List[Tuple[float, float]] = []
            if mc is not None:
                components.append((mc, 0.4))
            if bayes is not None:
                components.append((bayes, 0.3))
            if rl is not None:
                components.append((rl, 0.3))
            
            if not components:
                fused = None
                kelly_f = 0.0
            else:
                num = sum(v * w for (v, w) in components)
                den = sum(w for (_, w) in components)
                fused_val = num / den if den > 0 else 0.0
                fused_val = max(0.0, min(1.0, fused_val))
                fused = fused_val
                kelly_f = self._compute_kelly_fraction(p=fused_val)
            
            # Build reasons
            reasons: List[str] = []
            if inputs.hurst is not None:
                reasons.append(f"hurst={inputs.hurst:.2f}")
            if inputs.bos_align is not None:
                reasons.append(f"bos={inputs.bos_align:.2f}")
            if inputs.norm_cvd is not None:
                reasons.append(f"cvd={inputs.norm_cvd:.2f}")
            if mc is not None:
                reasons.append(f"mc={mc:.3f}")
            if bayes is not None:
                reasons.append(f"bayes={bayes:.3f}")
            if rl is not None:
                reasons.append(f"rl={rl:.3f}")
            if fused is not None:
                reasons.append(f"fused={fused:.3f}")
            reasons.append(f"kelly={kelly_f:.4f}")
            
            return OracleSideScores(
                side=side,
                mc=mc,
                bayes=bayes,
                rl=rl,
                fused=fused,
                kelly_f=kelly_f,
                rr=self.rr_assumed,
                reasons=reasons,
            )
        except Exception as e:
            logger.error(f"Error computing side scores for {side}: {e}")
            return OracleSideScores(side=side, reasons=["error"])

    def decide(self, inputs: OracleInputs, risk_manager) -> Optional[OracleOutputs]:
        """
        Compute scores for both sides.
        """
        try:
            long_scores = self.compute_side_scores("long", inputs, risk_manager)
            short_scores = self.compute_side_scores("short", inputs, risk_manager)
            
            # Disable oracle if no components available
            if (
                long_scores.mc is None
                and long_scores.bayes is None
                and long_scores.rl is None
                and short_scores.mc is None
                and short_scores.bayes is None
                and short_scores.rl is None
            ):
                logger.debug("Oracle disabled: no valid components")
                return None
            
            return OracleOutputs(long_scores=long_scores, short_scores=short_scores)
        except Exception as e:
            logger.error(f"Error in oracle decide: {e}")
            return None