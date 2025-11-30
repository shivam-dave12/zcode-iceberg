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
    """

    def __init__(self) -> None:

        np.random.seed(42)

        # PPO / RL placeholder state: proxy with running winrate for now.
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
    # Threshold adjustments (Z-score boost, micro-trap)
    # ======================================================================

    def adjust_zscore_with_hurst_and_trap(
        self,
        z_score: float,
        side: str,
        hurst: Optional[float],
        micro_trap: bool,
    ) -> float:
        """
        Apply:
        - Hurst boost: if H > 0.5, scale magnitude by (1 + (H - 0.5)).
        - Micro-trap: if LV_1m > 1.5 * LV_5m and side == "short",
          boost |Z| by 20%.
        """
        z = float(z_score)
        if hurst is not None and hurst > 0.5:
            boost = 1.0 + (hurst - 0.5)
            if z > 0:
                z *= boost
            elif z < 0:
                z *= boost

        if micro_trap and side == "short":
            if z < 0:
                z *= 1.20

        return z

    # ======================================================================
    # Fusion: Monte Carlo, Bayes (psy score), RL proxy, Kelly
    # ======================================================================

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def _clamp_prob(self, p: float) -> float:
        return max(self._prob_min, min(self._prob_max, p))

    def _compute_mc_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        atr_percent: Optional[float],
    ) -> Optional[float]:
        """
        Monte Carlo component:
        Paths ~ N(mu_cvd, sigma_atr), score = fraction of paths
        where mean(Paths) * sign(I) > 0, side-aware.

        If any critical input is missing, returns None (MC disabled)
        instead of a neutral probability.
        """
        if (
            imbalance_val is None
            or norm_cvd is None
            or atr_percent is None
            or atr_percent <= 0.0
        ):
            return None

        sign_side = 1.0 if side == "long" else -1.0
        mu = float(norm_cvd) * sign_side
        sigma = max(float(atr_percent), 1e-4)

        paths = np.random.normal(loc=mu, scale=sigma, size=self.mc_paths)
        sign_I = 1.0 if imbalance_val >= 0 else -1.0

        # Count how many paths align with the Imbalance sign
        positives = float(np.sum(paths * sign_I > 0))
        mc = positives / float(self.mc_paths)

        return self._clamp_prob(mc)

    def _compute_bayes_component(
        self,
        side: str,
        imbalance_val: Optional[float],
        norm_cvd: Optional[float],
        bos_align: Optional[float],
        hurst: Optional[float],
    ) -> Optional[float]:
        """
        Bayesian-like psy score turned into [0,1] via sigmoid.
        Features:
        - f1: CVD aligned with side (required)
        - f2: imbalance aligned with side (required)
        - f3: BOS alignment (optional)
        - f4: Hurst excess (H - 0.5, ≥0) (optional)

        If norm_cvd or imbalance are missing, returns None (Bayes disabled).
        """
        if norm_cvd is None or imbalance_val is None:
            return None

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

    def _compute_rl_component(
        self, side: str, risk_manager, hurst: Optional[float]
    ) -> Optional[float]:
        """
        RL proxy:
        1. Uses running winrate (if trades exist).
        2. Fallback to Hurst-based "Mean Reversion Edge" if 0 trades.
           NO FAKE DATA: Hurst is a real measure of predictability.
           H < 0.5 -> Mean reverting (Good for wall scalping)
           H > 0.5 -> Trending/Diffusive (Bad for wall scalping)

        Returns None only if neither metric is available.
        """
        try:
            total = int(getattr(risk_manager, "total_trades", 0))
            wins = int(getattr(risk_manager, "winning_trades", 0))

            if total > 0:
                winrate = wins / float(total)
                return max(0.0, min(1.0, winrate))

            # Alternative indicator if 0 trades: Hurst Predictability
            if hurst is not None:
                # If H < 0.5, we are in a mean-reverting regime (predictable).
                # Map 0.3 (strong MR) -> 0.7 score.
                # Map 0.5 (random) -> 0.5 score.
                # Map 0.7 (trending) -> 0.3 score.
                # Formula: 0.5 + (0.5 - H)
                score = 0.5 + (0.5 - hurst)
                return max(0.01, min(0.99, score))

            return None

        except Exception:
            return None

    def _compute_kelly_fraction(self, p: float) -> float:
        """
        Kelly sizing:
        f = (p*b - q) / b, b = R:R = 3.33, capped to 2% of margin.
        If edge <= 0, returns 0 (no risk).
        """
        p = max(0.0, min(1.0, p))
        q = 1.0 - p
        b = self.rr_assumed
        edge = p * b - q
        f = edge / b
        return max(0.0, min(0.02, f))

    # ======================================================================
    # High-level API
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
        """
        Called once per tick from strategy: gathers advanced metrics and
        packages them for the fusion step.
        """
        ema_val = None
        atr_pct = None

        try:
            ema_val = data_manager.get_ema(period=config.EMA_PERIOD)
        except Exception:
            ema_val = None

        try:
            # Updated DataManager ensures this tries Realized Volatility if ATR fails
            atr_pct = data_manager.get_atr_percent(
                window_minutes=config.ATR_WINDOW_MINUTES
            )
        except Exception:
            atr_pct = None

        lv_1m, lv_5m, lv_15m, micro_trap = self.compute_liquidity_velocity_multi_tf(
            data_manager
        )

        norm_cvd = self.compute_norm_cvd(
            data_manager, window_sec=config.DELTA_WINDOW_SEC
        )

        hurst = self.compute_hurst_exponent(data_manager, window_ticks=20)

        bos_align = self.compute_bos_alignment(
            data_manager, current_price=current_price
        )

        now_sec = time.time()

        return OracleInputs(
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
        """
        imbalance_val = None
        if inputs.imbalance_data is not None:
            imbalance_val = float(inputs.imbalance_data.get("imbalance", 0.0))

        z_raw = 0.0
        if inputs.delta_data is not None:
            z_raw = float(inputs.delta_data.get("z_score", 0.0))

        z_adj = self.adjust_zscore_with_hurst_and_trap(
            z_score=z_raw,
            side=side,
            hurst=inputs.hurst,
            micro_trap=inputs.micro_trap,
        )

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

        # Pass Hurst to RL component as alternative if trades == 0
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

        reasons: List[str] = []
        reasons.append(f"z_raw={z_raw:.2f}, z_adj={z_adj:.2f}")
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
            reasons.append(f"fused={fused:.3f}")
        else:
            reasons.append("fused=MISSING")

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
