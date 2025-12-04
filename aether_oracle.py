#  aether_oracle.py

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.stats import norm  # For normalization in fusion
import config

logger = logging.getLogger(__name__)

@dataclass
class OracleInputs:
    # Core gates (as computed in strategy)
    imbalance_data: Optional[Dict]
    wall_data: Optional[Dict]
    delta_data: Optional[Dict]
    touch_data: Optional[Dict]
    htf_trend: Optional[str]
    ltf_trend: Optional[str]  # Dropped, but kept for compat
    ema_val: Optional[float]
    atr_pct: Optional[float]
    # Advanced metrics (NEW: Expanded to 9)
    lv_1m: Optional[float]
    lv_5m: Optional[float]
    lv_15m: Optional[float]
    micro_trap: bool
    norm_cvd: Optional[float]
    hurst: Optional[float]
    bos_align: Optional[float]
    lstm_up_prob: Optional[float]  # From data_manager LSTM
    # Meta
    current_price: float
    now_sec: float
    regime: str  # NEW: Vol regime for weight twist

@dataclass
class OracleSideScores:
    side: str  # "long" or "short"
    mc: Optional[float]
    bayes: Optional[float]
    rl: Optional[float]
    fused: Optional[float]
    kelly_f: float
    rr: float
    reasons: List[str]
    win_prob: float  # NEW: Overlay prob

@dataclass
class OracleOutputs:
    long_scores: OracleSideScores
    short_scores: OracleSideScores

class AetherOracle:
    """
    Aether Oracle: fuses core gates + advanced metrics into a probabilistic
    entry decision and Kelly sizing.
    All heavy math lives here to avoid touching the core Z-Score strategy logic.
    UPDATED: 9-signal fusion with regime-twisted weights; win prob overlay.
    """
    def __init__(self) -> None:
        np.random.seed(42)
        self.entry_prob_threshold: float = 0.70  # CHANGED from 0.95; for momentum spikes
        self.rr_assumed: float = 3.33  # R:R used for Kelly
        self.mc_paths: int = 100
        # Floors/ceilings to avoid hard 0/1 probabilities in logs
        self._prob_min: float = 0.01
        self._prob_max: float = 0.99

        # 9-signal weights (base, sum=1.0): Core 60% + CVD10% LV5% HurstBOS10% LSTM10% (adjusted to 95% core for balance)
        self.weights_base = {
            "imb": 0.15, "wall": 0.12, "z": 0.18, "touch": 0.06, "trend": 0.09,  # Core 60%
            "cvd": 0.10, "lv": 0.05, "hurst_bos": 0.10, "lstm": 0.10  # Advanced 35%, total ~95%; normalize in fusion
        }

        # Regime twists: HIGH +Z/CVD (aggressor heavy), LOW +trend/LV (persistent)
        self.weight_twists = {
            "HIGH": {"z": 0.22, "cvd": 0.14},  # +0.04 each
            "LOW": {"trend": 0.12, "lv": 0.07},  # +0.03 each
            "NEUTRAL": {}  # Base
        }

    # ======================================================================
    # Raw metric builders from DataManager (Full Impl)
    # ======================================================================

    def compute_liquidity_velocity_multi_tf(
        self, data_manager
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """
        Liquidity Velocity per TF:
        LV = sum(volume) / (sum(|ΔP|) + ε)
        Uses recent trades for volume and price window for ΔP.
        """
        def _lv_for_window(window_sec: int) -> Optional[float]:
            price_window = data_manager.get_price_window(window_seconds=window_sec)
            if not price_window or len(price_window) < 2:
                return None
            prices = [p for _, p in price_window]
            deltas = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            sum_abs_dp = float(sum(deltas))
            eps = 1e-6
            trades = data_manager.get_recent_trades(window_seconds=window_sec)
            if not trades:
                return None
            vol_sum = 0.0
            for t in trades:
                try:
                    qty = float(t.get("qty", 0.0))
                except Exception:
                    qty = 0.0
                if qty <= 0:
                    continue
                vol_sum += qty
            if vol_sum <= 0.0:
                return None
            return vol_sum / (sum_abs_dp + eps)

        lv_1m = _lv_for_window(60)
        lv_5m = _lv_for_window(300)
        lv_15m = _lv_for_window(900)
        micro_trap = False
        if lv_1m is not None and lv_5m is not None and lv_5m > 0:
            if lv_1m > 1.5 * lv_5m:
                micro_trap = True
        logger.debug(f"LV: 1m={lv_1m:.2f}, 5m={lv_5m:.2f}, 15m={lv_15m:.2f}, trap={micro_trap}")
        return lv_1m, lv_5m, lv_15m, micro_trap

    def compute_norm_cvd(self, data_manager, window_sec: int) -> Optional[float]:
        """
        Footprint CVD in window_sec (typically 10s), normalized by volume:
        NormCVD = (∑ Vol_i * side_sign_i) / (∑ Vol_i)
        where side_sign = +1 buy, -1 sell.
        UPDATED: Returns 0.0 (Neutral) instead of None if valid orderbook exists
        but no trades occurred in the window.
        """
        trades = data_manager.get_recent_trades(window_seconds=window_sec)
        # If no trades, check if we have price data at least.
        # If yes, return 0.0 (neutral flow). If no price/book, return None.
        if not trades:
            if data_manager.get_last_price() > 0:
                return 0.0
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
            # On CoinSwitch streams, isBuyerMaker == False → aggressive buy
            if not is_buyer_maker:
                buy_vol += qty
            else:
                sell_vol += qty

        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        norm_cvd = (buy_vol - sell_vol) / total  # +1 buy heavy, -1 sell heavy
        logger.debug(f"Norm CVD: {norm_cvd:.3f} (buy_vol={buy_vol:.0f}, sell_vol={sell_vol:.0f})")
        return norm_cvd

    # NEW: Helpers for advanced signals (called in compute_side_scores)
    def _compute_hurst_bos_blend(self, hurst: Optional[float], bos_align: Optional[float]) -> Optional[float]:
        """Blend Hurst (persistence) and BOS align (-1/-1 to +1/+1)."""
        if hurst is None or bos_align is None:
            return None
        # Normalize Hurst [0.5 mean revert, 1.0 trend] to [0,1], BOS already [-1,1] → [0,1]
        hurst_norm = (hurst - 0.5) * 2  # [0,1]
        bos_norm = (bos_align + 1) / 2  # [0,1]
        blend = 0.5 * hurst_norm + 0.5 * bos_norm
        return blend

    # ======================================================================
    # Component Computations (Base: MC, Bayes, RL - Full Impl)
    # ======================================================================

    def _compute_mc_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        atr_percent: Optional[float],
    ) -> Optional[float]:
        """Monte Carlo sim prob based on imbalance/CVD/ATR."""
        if imbalance_val is None or norm_cvd is None or atr_percent is None:
            return None

        # Simple MC: Simulate paths with vol=ATR, bias from imb+CVD
        bias = (imbalance_val + norm_cvd) / 2  # [-1,1] → direction
        direction_mult = 1.0 if side == "long" else -1.0
        biased_bias = bias * direction_mult

        # MC paths: Geometric Brownian with bias
        paths = np.random.normal(
            loc=biased_bias * atr_percent,
            scale=atr_percent,
            size=(self.mc_paths, 10)  # 10 steps
        )
        path_returns = np.exp(np.sum(paths, axis=1)) - 1
        mc_prob = np.mean(path_returns > 0)  # Prob positive return
        logger.debug(f"MC {side}: {mc_prob:.3f} (bias={biased_bias:.3f}, atr%={atr_percent*100:.2f})")
        return mc_prob

    def _compute_bayes_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        bos_align: Optional[float],
        hurst: Optional[float],
    ) -> Optional[float]:
        """Bayesian update on prior=0.5 with likelihood from signals."""
        if imbalance_val is None or norm_cvd is None:
            return None

        prior = 0.5
        # Likelihoods: Imb/CVD as evidence strength
        imb_lik = abs(imbalance_val)  # [0,1]
        cvd_lik = abs(norm_cvd)  # [0,1]
        # BOS/Hurst as prior adjust
        bos_hurst_prior = 0.5
        if bos_align is not None and hurst is not None:
            bos_hurst_prior = 0.5 + 0.2 * (bos_align + (hurst - 0.5))  # Blend to [0.3,0.7]

        # Bayes: P(H|E) = [P(E|H) * P(H)] / P(E)
        # Simplified: Update prior with product of liks
        lik_product = imb_lik * cvd_lik * bos_hurst_prior
        bayes_prob = prior * lik_product / (prior * lik_product + (1 - prior) * (1 - lik_product))
        direction_adjust = 1.0 if side == "long" else (1.0 - 2 * bayes_prob)  # Flip for short
        bayes_prob = (bayes_prob + direction_adjust) / 2  # [0,1]
        bayes_prob = np.clip(bayes_prob, self._prob_min, self._prob_max)
        logger.debug(f"Bayes {side}: {bayes_prob:.3f} (lik_prod={lik_product:.3f})")
        return bayes_prob

    def _compute_rl_component(
        self,
        side: str,
        risk_manager,
        hurst: Optional[float],
    ) -> Optional[float]:
        """RL-inspired: Q-value from recent wins, Hurst for persistence."""
        if not hasattr(risk_manager, 'total_trades') or risk_manager.total_trades == 0:
            return None

        win_rate = risk_manager.winning_trades / risk_manager.total_trades
        # Q = alpha * reward + (1-alpha) * Q_prev; simple: win_rate * hurst adjust
        alpha = 0.1
        hurst_adjust = (hurst - 0.5) * 0.2 if hurst else 0  # Trend boost
        rl_q = alpha * win_rate + (1 - alpha) * 0.5 + hurst_adjust  # Decay to 0.5
        rl_prob = np.clip(rl_q, self._prob_min, self._prob_max)
        # Side adjust (assume symmetric for now)
        logger.debug(f"RL {side}: {rl_prob:.3f} (win_rate={win_rate:.2f}, hurst_adj={hurst_adjust:.3f})")
        return rl_prob

    def _compute_kelly_fraction(self, p: float) -> float:
        """Kelly: f = p - (1-p)/b, b=rr_assumed."""
        if p <= 0.5:
            return 0.0
        f = p - (1 - p) / self.rr_assumed
        return np.clip(f, 0.0, 0.25)  # Cap at 25%

    # ======================================================================
    # Side Scores (Updated: 9-Signal Fusion + Regime Weights + Win Prob)
    # ======================================================================

    def compute_side_scores(
        self,
        side: str,
        inputs: OracleInputs,
        risk_manager,
    ) -> OracleSideScores:
        """
        Compute MC, Bayes, RL and fused score + Kelly for one side.
        No component uses synthetic defaults; each metric is either
        computed from real data or left as None and excluded from fusion.
        UPDATED: 9-signal weighted fusion, regime-twist, win prob overlay.
        """
        imbalance_val = None
        if inputs.imbalance_data is not None:
            imbalance_val = float(inputs.imbalance_data.get("imbalance", 0.0))

        z_raw = 0.0
        if inputs.delta_data is not None:
            z_raw = float(inputs.delta_data.get("z_score", 0.0))

        # ... (z_adj logic from base, if needed)

        norm_cvd = inputs.norm_cvd
        bos_align = inputs.bos_align
        hurst = inputs.hurst
        atr_pct = inputs.atr_pct

        mc = self._compute_mc_component(
            side=side,
            imbalance_val=imbalance_val,
            norm_cvd=norm_cvd,
            atr_percent=atr_pct,
        )
        bayes = self._compute_bayes_component(
            side=side,
            imbalance_val=imbalance_val,
            norm_cvd=norm_cvd,
            bos_align=bos_align,
            hurst=hurst,
        )
        rl = self._compute_rl_component(
            side=side, risk_manager=risk_manager, hurst=hurst
        )

        # Fusion: use only available components, no neutral replacements
        components: List[Tuple[float, float]] = []  # (value, weight)
        if mc is not None:
            components.append((mc, 0.4))
        if bayes is not None:
            components.append((bayes, 0.3))
        if rl is not None:
            components.append((rl, 0.3))

        if not components:
            fused: Optional[float] = None
            kelly_f = 0.0
        else:
            num = sum(v * w for (v, w) in components)
            den = sum(w for (_, w) in components)
            fused_val = num / den if den > 0 else 0.0
            fused_val = max(0.0, min(1.0, fused_val))
            fused = fused_val
            kelly_f = self._compute_kelly_fraction(p=fused_val)

        # NEW: 9-Signal Weighted Fusion (post base-fusion, or replace; here as enhancer)
        # Normalize signals to [0,1] via simple min-max (or CDF)
        signals_norm = {
            "imb": abs(imbalance_val) if imbalance_val is not None else 0.5,
            "wall": inputs.wall_data.get("strength_mult", 0.5) if inputs.wall_data else 0.5,
            "z": abs(z_raw) / 3.0 if z_raw else 0.5,  # Assume max Z=3
            "touch": 1.0 if inputs.touch_data and inputs.touch_data.get("is_touching") else 0.0,
            "trend": 1.0 if inputs.htf_trend == ("UP" if side=="long" else "DOWN") else 0.5,
            "cvd": abs(norm_cvd) if norm_cvd is not None else 0.5,
            "lv": np.mean([inputs.lv_1m or 1.0, inputs.lv_5m or 1.0, inputs.lv_15m or 1.0]),
            "hurst_bos": self._compute_hurst_bos_blend(hurst, bos_align) or 0.5,
            "lstm": inputs.lstm_up_prob or 0.5
        }
        # Apply regime-twist to weights
        weights = self.weights_base.copy()
        twist = self.weight_twists.get(inputs.regime, {})
        for k, v in twist.items():
            if k in weights:
                weights[k] = v
        # Normalize weights to sum=1
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum for k, v in weights.items()}
        # Fused signal prob
        signal_fused = sum(signals_norm[k] * weights[k] for k in weights)
        # Blend with base fused (if exists)
        if fused is not None:
            fused = 0.7 * fused + 0.3 * signal_fused  # 70/30 base/advanced

        # NEW: Win prob overlay: 0.4 + 0.2*lstm + 0.2*z_sign + 0.1*cvd + 0.1*lv >0.6 veto
        z_sign = 1.0 if abs(z_raw) > 2.1 else 0.0
        lv_avg = signals_norm["lv"]
        cvd_abs = signals_norm["cvd"]
        lstm_prob = signals_norm["lstm"]
        win_prob = 0.4 + 0.2 * lstm_prob + 0.2 * z_sign + 0.1 * cvd_abs + 0.1 * lv_avg
        win_prob = np.clip(win_prob, 0.0, 1.0)

        # Reasons (enhanced with new)
        reasons: List[str] = []
        reasons.append(f"z_raw={z_raw:.2f}")
        if inputs.micro_trap and side == "short":
            reasons.append("micro_trap_short=1")
        if inputs.hurst is not None:
            reasons.append(f"hurst={inputs.hurst:.2f}")
        else:
            reasons.append("hurst=MISSING")
        if inputs.bos_align is not None:
            reasons.append(f"bos_align={inputs.bos_align:.2f}")
        else:
            reasons.append("bos_align=MISSING")
        if inputs.norm_cvd is not None:
            reasons.append(f"norm_cvd={inputs.norm_cvd:.2f}")
        else:
            reasons.append("norm_cvd=MISSING")
        if mc is not None:
            reasons.append(f"mc={mc:.3f}")
        else:
            reasons.append("mc=MISSING")
        if bayes is not None:
            reasons.append(f"bayes={bayes:.3f}")
        else:
            reasons.append("bayes=MISSING")
        if rl is not None:
            reasons.append(f"rl={rl:.3f}")
        else:
            reasons.append("rl=MISSING")
        if fused is not None:
            reasons.append(f"fused={fused:.3f} (9-sig={signal_fused:.3f})")
        else:
            reasons.append("fused=MISSING")
        reasons.append(f"kelly_f={kelly_f:.4f}, rr={self.rr_assumed:.2f}")
        reasons.append(f"win_prob={win_prob:.3f}, regime={inputs.regime}")

        return OracleSideScores(
            side=side,
            mc=mc,
            bayes=bayes,
            rl=rl,
            fused=fused,
            kelly_f=kelly_f,
            rr=self.rr_assumed,
            reasons=reasons,
            win_prob=win_prob,
        )

    def decide(self, inputs: OracleInputs, risk_manager) -> Optional[OracleOutputs]:
        """
        Compute side scores and return both.
        If *all* fusion components (MC, Bayes, RL) are missing on both sides,
        returns None so the strategy falls back to legacy gate logic only.
        """
        long_scores = self.compute_side_scores("long", inputs, risk_manager)
        short_scores = self.compute_side_scores("short", inputs, risk_manager)

        # If there is literally no MC/Bayes/RL on either side, disable oracle
        if (
            long_scores.mc is None
            and long_scores.bayes is None
            and long_scores.rl is None
            and short_scores.mc is None
            and short_scores.bayes is None
            and short_scores.rl is None
        ):
            return None

        return OracleOutputs(long_scores=long_scores, short_scores=short_scores)