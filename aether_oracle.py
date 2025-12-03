# aether_oracle.py

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
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
    ltf_trend: Optional[str]
    ema_val: Optional[float]
    atr_pct: Optional[float]

    # Advanced metrics
    lv_1m: Optional[float]
    lv_5m: Optional[float]
    lv_15m: Optional[float]
    micro_trap: bool
    norm_cvd: Optional[float]
    hurst: Optional[float]
    bos_align: Optional[float]

    # Meta
    current_price: float
    now_sec: float


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


@dataclass
class OracleOutputs:
    long_scores: OracleSideScores
    short_scores: OracleSideScores


class AetherOracle:
    """
    Aether Oracle: fuses core gates + advanced metrics into a probabilistic
    entry decision and Kelly sizing.
    All heavy math lives here to avoid touching the core Z-Score strategy logic.
    
    UPDATED: 9-signal data fusion maximization.
    """

    def __init__(self) -> None:
        np.random.seed(42)
        self.entry_prob_threshold: float = 0.70  # CHANGED from 0.95; for momentum spikes
        self.rr_assumed: float = 3.33  # R:R used for Kelly
        self.mc_paths: int = 100

        # Floors/ceilings to avoid hard 0/1 probabilities in logs
        self._prob_min: float = 0.01
        self._prob_max: float = 0.99

    # ======================================================================
    # Raw metric builders from DataManager
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

        return lv_1m, lv_5m, lv_15m, micro_trap

    def compute_norm_cvd(self, data_manager, window_sec: int) -> Optional[float]:
        """
        Footprint CVD in window_sec (typically 10s), normalized by volume:
        NormCVD = (Σ Vol_i * side_sign_i) / (Σ Vol_i)
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
        if total <= 0.0:
            # Trades existed but sum volume 0? Unlikely, but return neutral.
            return 0.0

        norm_cvd = (buy_vol - sell_vol) / total
        # Clamp to [-1,1] numerically
        return max(-1.0, min(1.0, norm_cvd))

    def compute_hurst_exponent(
        self, data_manager, window_ticks: int = 20
    ) -> Optional[float]:
        """
        Hurst exponent via classic R/S method on last N price ticks:
        H = log(R/S) / log(n), using cumulative deviations from mean.
        """
        window_sec = getattr(config, "DELTA_WINDOW_SEC", 10) * 3
        price_window = data_manager.get_price_window(window_seconds=window_sec)

        if not price_window or len(price_window) < window_ticks:
            return None

        prices = np.asarray([p for _, p in price_window], dtype=np.float64)
        series = prices[-window_ticks:]

        if len(series) < window_ticks:
            return None

        mean = float(series.mean())
        dev = series - mean
        cum_dev = np.cumsum(dev)

        r = float(np.max(cum_dev) - np.min(cum_dev))
        s = float(np.std(cum_dev))
        n = len(series)

        if n <= 1 or s == 0.0 or r <= 0.0:
            return None

        try:
            h = math.log(r / s) / math.log(n)
        except Exception:
            return None

        return h

    def compute_bos_alignment(
        self, data_manager, current_price: float
    ) -> Optional[float]:
        """
        Multi-TF SMC BOS alignment approximation.
        Builds synthetic OHLC from tick prices at 1m / 5m / 15m, then
        measures fraction of recent bars where price > H_i or price < L_i.
        Uses a deeper historical tick window for each TF (10×TF minutes),
        so BOS is available from real historical prices and then
        naturally rolls forward as new ticks arrive.
        
        Returns:
            None -> not enough data for a stable measure
            0–1 -> BOS frequency alignment across TFs
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        def _build_ohlc(window_min: int, rule: str) -> Optional["pd.DataFrame"]:
            # Look back over 10 bars worth of that TF to build BOS structure
            window_sec = window_min * 60 * 10
            price_window = data_manager.get_price_window(window_seconds=window_sec)

            if not price_window:
                return None

            df = pd.DataFrame(price_window, columns=["tsms", "price"])
            if df.empty:
                return None

            df["ts"] = pd.to_datetime(df["tsms"], unit="ms")
            df.set_index("ts", inplace=True)

            ohlc = df["price"].resample(rule).agg(["first", "max", "min", "last"])
            ohlc.dropna(inplace=True)

            if ohlc.empty:
                return None

            ohlc.columns = ["open", "high", "low", "close"]
            return ohlc

        bos_vals: List[float] = []
        valid_count = 0

        for window_min, rule in ((15, "15min"), (5, "5min"), (1, "1min")):
            ohlc = _build_ohlc(window_min, rule)
            if ohlc is None:
                continue

            # Use up to the last 10 bars of this TF
            tail = ohlc.tail(10)
            if len(tail) == 0:
                continue

            highs = tail["high"].values
            lows = tail["low"].values

            if len(highs) == 0:
                continue

            breaks = 0
            total = len(highs)

            for h, l in zip(highs, lows):
                if current_price > h or current_price < l:
                    breaks += 1

            bos_tf = breaks / float(total)
            bos_vals.append(bos_tf)
            valid_count += 1

        if valid_count == 0:
            return None

        align = float(sum(bos_vals) / valid_count)
        return align

    # ======================================================================
    # Monte Carlo, Bayesian, RL stubs (simplified for this implementation)
    # ======================================================================

    def _compute_monte_carlo_prob(
        self, side: str, inputs: OracleInputs
    ) -> Optional[float]:
        """
        Simplified Monte Carlo probability based on Z-score and imbalance.
        In full implementation, this would run MC simulations.
        """
        try:
            if not inputs.delta_data or not inputs.imbalance_data:
                return None

            z = inputs.delta_data["z_score"]
            imb = inputs.imbalance_data["imbalance"]

            if side == "long":
                z_contrib = max(0.0, min(1.0, (z + 3.0) / 6.0))
                imb_contrib = max(0.0, min(1.0, (imb + 1.0) / 2.0))
            else:
                z_contrib = max(0.0, min(1.0, (-z + 3.0) / 6.0))
                imb_contrib = max(0.0, min(1.0, (-imb + 1.0) / 2.0))

            mc_prob = (z_contrib + imb_contrib) / 2.0
            return max(self._prob_min, min(self._prob_max, mc_prob))

        except Exception as e:
            logger.error(f"Error in MC prob: {e}", exc_info=True)
            return None

    def _compute_bayesian_prob(
        self, side: str, inputs: OracleInputs
    ) -> Optional[float]:
        """
        Simplified Bayesian probability incorporating CVD and wall data.
        """
        try:
            if not inputs.wall_data:
                return None

            # Prior: 0.5
            prior = 0.5

            # Likelihood based on wall strength
            if side == "long":
                wall_strength = inputs.wall_data["bid_wall_strength"]
            else:
                wall_strength = inputs.wall_data["ask_wall_strength"]

            likelihood = max(0.0, min(1.0, wall_strength / 5.0))

            # CVD evidence
            cvd_evidence = 1.0
            if inputs.norm_cvd is not None:
                if side == "long":
                    cvd_evidence = max(0.5, min(1.5, 1.0 + inputs.norm_cvd))
                else:
                    cvd_evidence = max(0.5, min(1.5, 1.0 - inputs.norm_cvd))

            # Posterior (simplified Bayes)
            posterior = prior * likelihood * cvd_evidence
            posterior = max(self._prob_min, min(self._prob_max, posterior))

            return posterior

        except Exception as e:
            logger.error(f"Error in Bayesian prob: {e}", exc_info=True)
            return None

    def _compute_rl_prob(
        self, side: str, inputs: OracleInputs
    ) -> Optional[float]:
        """
        Simplified RL-style probability using Hurst and BOS.
        """
        try:
            score = 0.5

            if inputs.hurst is not None:
                h = inputs.hurst
                # Trending market (h > 0.5) favors directional trades
                if h > 0.5:
                    score += 0.1
                else:
                    score -= 0.1

            if inputs.bos_align is not None:
                bos = inputs.bos_align
                # Higher BOS alignment = more structure breaks
                score += bos * 0.2

            score = max(self._prob_min, min(self._prob_max, score))
            return score

        except Exception as e:
            logger.error(f"Error in RL prob: {e}", exc_info=True)
            return None

    # ======================================================================
    # Kelly fraction
    # ======================================================================

    def _compute_kelly_fraction(self, p: float) -> float:
        """
        Kelly criterion: f = (p*b - q) / b
        where p = win prob, q = 1-p, b = odds (R:R ratio - 1)
        
        Capped at 2% for safety.
        """
        if p <= 0 or p >= 1:
            return 0.01

        b = self.rr_assumed - 1.0
        q = 1.0 - p

        kelly_f = (p * b - q) / b
        kelly_f = max(0.01, min(0.02, kelly_f))

        return kelly_f

    # ======================================================================
    # Side score computation with 9-signal fusion
    # ======================================================================

    def compute_side_scores(
        self, side: str, inputs: OracleInputs, risk_manager
    ) -> OracleSideScores:
        """
        Compute fusion scores for one side (long or short).
        
        FIXED: Proper weight normalization.
        """
        # ... [existing code for vol_regime and individual probabilities] ...

        # Z-score contribution
        z_raw = inputs.delta_data["z_score"] if inputs.delta_data else 0.0
        if side == "long":
            z_contrib = max(0.0, min(1.0, (z_raw + 3.0) / 6.0))
        else:
            z_contrib = max(0.0, min(1.0, (-z_raw + 3.0) / 6.0))

        # Imbalance contribution
        imb = inputs.imbalance_data["imbalance"] if inputs.imbalance_data else 0.0
        if side == "long":
            imb_contrib = max(0.0, min(1.0, (imb + 1.0) / 2.0))
        else:
            imb_contrib = max(0.0, min(1.0, (-imb + 1.0) / 2.0))

        # Wall contribution
        if inputs.wall_data:
            if side == "long":
                wall_contrib = min(1.0, inputs.wall_data["bid_wall_strength"] / 5.0)
            else:
                wall_contrib = min(1.0, inputs.wall_data["ask_wall_strength"] / 5.0)
        else:
            wall_contrib = 0.0

        # CVD contribution
        cvd_contrib = 0.5
        if inputs.norm_cvd is not None:
            if side == "long":
                cvd_contrib = max(0.0, min(1.0, (inputs.norm_cvd + 1.0) / 2.0))
            else:
                cvd_contrib = max(0.0, min(1.0, (-inputs.norm_cvd + 1.0) / 2.0))

        # LV contribution
        lv_contrib = 0.5
        if inputs.lv_1m is not None and inputs.lv_5m is not None:
            lv_avg = (inputs.lv_1m + inputs.lv_5m) / 2.0
            lv_contrib = min(1.0, lv_avg / 2.0)

        # Hurst/BOS contribution
        hurst_bos_contrib = 0.5
        if inputs.hurst is not None:
            hurst_bos_contrib = inputs.hurst
        if inputs.bos_align is not None:
            hurst_bos_contrib = (hurst_bos_contrib + inputs.bos_align) / 2.0

        # === FUSION WITH REGIME-ADJUSTED WEIGHTS ===
        
        # Get vol regime from inputs
        vol_regime = "NEUTRAL"
        if inputs.atr_pct is not None:
            if inputs.atr_pct < 0.0015:
                vol_regime = "LOW"
            elif inputs.atr_pct > 0.003:
                vol_regime = "HIGH"

        # Define weights based on regime
        if vol_regime == "HIGH":
            # High vol: weight Z/CVD heavily
            weight_map = {
                "mc": 0.15,
                "bayes": 0.10,
                "rl": 0.10,
                "z": 0.25,
                "imb": 0.10,
                "wall": 0.05,
                "cvd": 0.15,
                "lv": 0.05,
                "hurst_bos": 0.05,
            }
        else:
            # LOW/NEUTRAL: balanced weights
            weight_map = {
                "mc": 0.20,
                "bayes": 0.15,
                "rl": 0.10,
                "z": 0.15,
                "imb": 0.15,
                "wall": 0.10,
                "cvd": 0.05,
                "lv": 0.05,
                "hurst_bos": 0.05,
            }

        # Build component dict with actual values
        component_values = {
            "mc": mc if mc is not None else None,
            "bayes": bayes if bayes is not None else None,
            "rl": rl if rl is not None else None,
            "z": z_contrib,
            "imb": imb_contrib,
            "wall": wall_contrib,
            "cvd": cvd_contrib,
            "lv": lv_contrib,
            "hurst_bos": hurst_bos_contrib,
        }

        # CORRECT CALCULATION: Only use available components, normalize weights
        available_components = {k: v for k, v in component_values.items() if v is not None}
        
        if not available_components:
            fused = None
            kelly_f = 0.01
        else:
            # Calculate total weight of available components
            available_weight_sum = sum(weight_map[k] for k in available_components)
            
            # NORMALIZED weighted sum
            fused_val = 0.0
            for key in available_components:
                normalized_weight = weight_map[key] / available_weight_sum  # KEY FIX
                fused_val += available_components[key] * normalized_weight
            
            fused = max(self._prob_min, min(self._prob_max, fused_val))
            kelly_f = self._compute_kelly_fraction(fused)

        # Build reasons
        reasons = []
        reasons.append(f"vol_regime={vol_regime}")
        reasons.append(f"available_weight_sum={available_weight_sum:.3f}->normalized")
        
        for k, v in available_components.items():
            normalized_w = weight_map[k] / available_weight_sum
            reasons.append(f"{k}={v:.3f}(w={normalized_w:.3f})")
        
        if fused is not None:
            reasons.append(f"fused={fused:.3f}(corrected)")

        reasons.append(f"kelly_f={kelly_f:.4f}, rr={self.rr_assumed:.2f}")

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
