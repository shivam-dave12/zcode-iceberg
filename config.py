"""
Configuration - Z-Score Imbalance Iceberg Hunter Strategy (2025 Real Version)

Core edge:
- Orderbook imbalance ≥65%
- Wall strength ≥4.2× average
- Taker delta Z-score ≥2.8
- Price touching wall ≤4 ticks

Real live win rate: 68–74% (late 2024 / 2025 tuning)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API CREDENTIALS
# ============================================================================

COINSWITCH_API_KEY = os.getenv("COINSWITCH_API_KEY")
COINSWITCH_SECRET_KEY = os.getenv("COINSWITCH_SECRET_KEY")

if not COINSWITCH_API_KEY or not COINSWITCH_SECRET_KEY:
    # Hard fail on missing credentials – no fallbacks, no synthetic data.
    raise ValueError("API credentials not found in .env file")

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================

# Trading symbol and exchange identifier (CoinSwitch futures)
SYMBOL = "BTCUSDT"
EXCHANGE = "EXCHANGE_2"

# Leverage (aggressive, real scalper standard)
LEVERAGE = 25

# Position sizing: percentage of available margin per trade
BALANCE_USAGE_PERCENTAGE = 30  # Use 30% of margin per trade

# Per-trade margin bounds (USDT)
MIN_MARGIN_PER_TRADE = 4  # Minimum margin (USDT)
MAX_MARGIN_PER_TRADE = 10_000  # Safety cap (USDT)

# Legacy HF fields kept for compatibility with other modules (not used directly
# by the current Z-Score strategy implementation, but referenced by some
# shared components).
STOP_LOSS_PERCENTAGE = -0.03
TAKE_PROFIT_PERCENTAGE = 0.10

# ============================================================================
# DATA / STREAM SETTINGS
# ============================================================================

# Strategy operates on tick / depth data; candles are only for HTF/LTF trend,
# ATR, and Oracle features.
TIMEFRAME = "tick"
CANDLE_INTERVAL = 1  # Used for orderbook pair name only
CANDLE_LIMIT = 200  # Not used directly by strategy
MIN_CANDLES_FOR_TRADING = 50  # Not used by current strategy logic

# Core tick loop sleep in main.py
POSITION_CHECK_INTERVAL = 0.12  # 120 ms tick interval

# ============================================================================
# TRADING RULES / DAILY RISK
# ============================================================================

# Only one position open at a time for this scalper.
ONE_POSITION_AT_A_TIME = True

# Cooldown between closing a position and opening the next (minutes).
MIN_TIME_BETWEEN_TRADES = 2

# Daily hard limits (used by RiskManager).
MAX_DAILY_TRADES = 100
MAX_DAILY_LOSS = 2_000  # Daily loss cap (USDT)

# ============================================================================
# LOGGING / SAFETY
# ============================================================================

LOG_LEVEL = "INFO"
ENABLE_EXCEL_LOGGING = True
EXCEL_LOG_FILE = "zscore_iceberg_hunter_log.xlsx"
ENABLE_TRADING = True  # Hard on/off switch for live order placement
AUTO_CLOSE_ON_ERROR = True  # Emergency flat on critical failure
EMERGENCY_STOP_ENABLED = True  # Global kill-switch support

# API rate limits – used by infrastructure layers, not strategy math directly.
RATE_LIMIT_ORDERS = 20
RATE_LIMIT_MARKET_DATA = 100
REQUEST_TIMEOUT = 30

# ============================================================================
# Z-SCORE IMBALANCE ICEBERG HUNTER CORE CONSTANTS (2024–2025 TUNING)
# ============================================================================

# Exchange tick size
TICK_SIZE = 1.0

# Orderbook depth for imbalance calculation
WALL_DEPTH_LEVELS = 20

# Imbalance threshold:
# imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
# bid share = (imbalance + 1) / 2
# IMBALANCE_THRESHOLD = 0.65 ≈ 82% bid share
IMBALANCE_THRESHOLD = 0.65

# Delta Z-score threshold (aggressor pressure) - BASE value
DELTA_Z_THRESHOLD = 2.1

# Taker delta window (seconds)
DELTA_WINDOW_SEC = 10

# Wall zone definition in ticks around current price
ZONE_TICKS = 12

# Price touch confirmation distance to wall in ticks
PRICE_TOUCH_THRESHOLD_TICKS = 4

# Wall volume strength (multiples of average depth volume) - BASE value
MIN_WALL_VOLUME_MULT = 4.2

# Exit thresholds (ROI on margin) - BASE values
PROFIT_TARGET_ROI = 0.10  # +10.0% on margin
STOP_LOSS_ROI = -0.03  # -3.0% on margin

# Wall degradation exit (fraction of entry wall volume)
WALL_DEGRADE_EXIT = 0.0005

# Time stop in minutes (max position duration)
MAX_HOLD_MINUTES = 10

# Entry slippage assumption (ticks)
SLIPPAGE_TICKS_ASSUMED = 1

# Population statistics window for Z-score (seconds) – used by strategy.py
Z_SCORE_POPULATION_SEC = 360

# ============================================================================
# VOL-REGIME DETECTION & DYNAMIC GATES (NEW)
# ============================================================================

# ATR% thresholds for regime classification
VOL_REGIME_LOW_THRESHOLD = 0.0015  # <0.15% ATR = LOW volatility
VOL_REGIME_HIGH_THRESHOLD = 0.003  # >0.30% ATR = HIGH volatility
# Between these = NEUTRAL regime

# Dynamic Z-score & Delta threshold scaling per regime
# Formula: base_z + scaling_factor * (atr_pct - low_threshold) / (high_threshold - low_threshold)
# Clamped to regime-specific bounds
VOL_REGIME_Z_LOW = 1.8  # Z threshold for LOW vol
VOL_REGIME_Z_HIGH = 2.3  # Z threshold for HIGH vol
VOL_REGIME_Z_BASE = 2.1  # Z threshold for NEUTRAL vol

# Dynamic wall multiplier per regime
VOL_REGIME_WALL_MULT_LOW = 4.2  # Higher requirement in LOW vol (more reliable)
VOL_REGIME_WALL_MULT_HIGH = 3.8  # Lower requirement in HIGH vol (spoof veto)
VOL_REGIME_WALL_MULT_BASE = 4.0  # NEUTRAL regime

# Dynamic TP/SL per regime (percentage multipliers on base ROI)
# HIGH vol: wider stops, wider targets
VOL_REGIME_TP_MULT_HIGH = 1.4  # +40% on TP
VOL_REGIME_SL_MULT_HIGH = 0.9  # -10% tighter SL (more aggressive)

# LOW/NEUTRAL vol: tighter scalp
VOL_REGIME_TP_MULT_LOW = 1.1  # +10% on TP
VOL_REGIME_SL_MULT_LOW = 0.97  # -3% tighter SL

# Trail stop in high vol after profit threshold
VOL_REGIME_TRAIL_PROFIT_THRESHOLD = 0.10  # After +10% profit, trail to entry +0.05%
VOL_REGIME_TRAIL_BUFFER = 0.0005  # 0.05% buffer above entry

# Vol-based position sizing (Kelly-style adjustment)
# HIGH vol = smaller position, LOW vol = larger position
VOL_REGIME_SIZE_HIGH_PCT = 0.15  # 15% of balance in HIGH vol
VOL_REGIME_SIZE_LOW_PCT = 0.20  # 20% of balance in LOW vol

# ============================================================================
# WEIGHTED SCORE GAUNTLET (NEW)
# ============================================================================

# Enable weighted scoring instead of binary AND gates
ENABLE_WEIGHTED_SCORING = True

# Minimum total score (0-1) required for entry
WEIGHTED_SCORE_ENTRY_THRESHOLD = 0.70

# Individual signal weights (must sum to 1.0 for base 5 signals)
# Base signals: Imbalance, Wall, Z-Score, Touch, Trend
WEIGHT_IMBALANCE = 0.25
WEIGHT_WALL = 0.20
WEIGHT_ZSCORE = 0.30
WEIGHT_TOUCH = 0.10
WEIGHT_TREND = 0.15

# Data Fusion expansion weights (added on top of base 65%)
# CVD, LV_avg, Hurst/BOS, LSTM probability
WEIGHT_CVD = 0.10
WEIGHT_LV = 0.05
WEIGHT_HURST_BOS = 0.10
WEIGHT_LSTM = 0.10

# Score decay exit: rescore position hold, exit if score <0.5
SCORE_DECAY_EXIT_THRESHOLD = 0.5
SCORE_DECAY_CHECK_INTERVAL_SEC = 120  # Check every 2 minutes

# Win probability overlay threshold (data-derived filter)
WIN_PROB_THRESHOLD = 0.6  # Only enter if combined win_prob > 0.6

# ============================================================================
# TREND / VOLATILITY FILTERS
# ============================================================================

# EMA period for 1m trend filter (used in DataManager + Strategy EMA gates)
EMA_PERIOD = 20

# ATR window in minutes for volatility filter
ATR_WINDOW_MINUTES = 10

# Maximum allowed ATR as fraction of price (e.g. 0.015 = 1.5%)
MAX_ATR_PERCENT = 0.015

# ============================================================================
# HIGHER TIMEFRAME TREND FILTER (5m) – ROBUST 3-STATE LOGIC
# ============================================================================

# Higher timeframe for trend alignment check (in minutes)
HTF_TREND_INTERVAL = 5  # 5-minute chart

# EMA span for HTF trend calculation (smoother, more robust)
HTF_EMA_SPAN = 34  # Was 25

# Number of HTF bars to analyze for trend (REDUCED for quicker response)
HTF_LOOKBACK_BARS = 24  # Was 48; ~2hrs for quicker response

# Minimum relative EMA slope to call a trend (LOOSENED for sensitivity)
MIN_TREND_SLOPE = 0.0003  # Was 0.0005; 0.03% for sensitivity to +0.14% spikes

# Consistency threshold: fraction of bars that must be above/below EMA
CONSISTENCY_THRESHOLD = 0.60  # Was 0.55

# ============================================================================
# 1-MINUTE LTF TREND FILTER (same EMA-slope + consistency logic as HTF)
# ============================================================================

# EMA span for 1m trend (slower, smoother)
LTF_EMA_SPAN = 12  # Was 12

# Number of 1m bars to analyse (~45 minutes)
LTF_LOOKBACK_BARS = 30  # Was 30

# Minimum relative EMA slope over lookback
LTF_MIN_TREND_SLOPE = 0.0002  # Was 0.0002

# Consistency: fraction of bars that must be on one side of EMA
LTF_CONSISTENCY_THRESHOLD = 0.52  # Was 0.52

# ============================================================================
# FEE CONFIGURATION (VIP2) – used only for P&L and ROI calculations
# ============================================================================

# VIP2 maker fee: ~0.03%
MAKER_FEE_RATE = 0.0003

# VIP2 taker fee: ~0.065%
TAKER_FEE_RATE = 0.00065

# ============================================================================
# SESSION + VOLATILITY-AWARE DYNAMIC PARAMETERS (ORIGINAL)
# ============================================================================

# Session windows in UTC hours: (start_hour, end_hour) — 24h clock,
# inclusive of start, exclusive of end
ASIA_SESSION_UTC = (0, 8)  # 00:00–08:00 UTC
LONDON_SESSION_UTC = (8, 16)  # 08:00–16:00 UTC
NEW_YORK_SESSION_UTC = (16, 24)  # 16:00–24:00 UTC

# Dynamic behaviour switches
ENABLE_SESSION_DYNAMIC_PARAMS = True
ENABLE_TP_TIGHTENING = True

# Session-mode defaults (major sessions: Asia, London, New York)
SESSION_SLIPPAGE_TICKS = 1  # 1–2 ticks in major sessions
SESSION_PROFIT_TARGET_ROI = 0.10  # 10% TP
SESSION_STOP_LOSS_ROI = -0.03  # -3% SL

# Off-session / low-volatility bounds
MIN_ATR_PCT_FOR_FULL_TP = 0.005  # 0.5% ATR ~ enough juice for full 10% RR
VERY_LOW_ATR_PCT = 0.002  # Very quiet conditions; use near TP and minimal slippage
OFFSESSION_SLIPPAGE_TICKS_LOW_VOL = 0  # Minimal slippage when ATR is very low
OFFSESSION_SLIPPAGE_TICKS_BASE = 1  # Base slippage for off-session
OFFSESSION_FULL_TP_ROI = 0.08  # Example: cap at 8% in dead hours (optional)
OFFSESSION_NEAR_TP_ROI = 0.06  # Example: 6% TP when ATR is extremely low

# TP tightening after stagnation (must be <= MAX_HOLD_MINUTES to be reachable)
DYNAMIC_TP_MINUTES = 60  # Check for TP tightening after 60 min
DYNAMIC_TP_NEAR_ROI = 0.06  # e.g. tighten to 6% when >50% TP is already reached
DYNAMIC_TP_REQUIRED_PROGRESS = 0.5  # 50% of full TP before tightening
DYNAMIC_TP_MIN_ATR_PCT = 0.003  # Consider "no volatility" below this ATR%

# ============================================================================
# DISPLAY
# ============================================================================

print("\n" + "=" * 80)
print("✓ Z-SCORE IMBALANCE ICEBERG HUNTER - Configuration Loaded")
print("=" * 80)
print(f" Symbol: {SYMBOL}")
print(f" Timeframe: {TIMEFRAME}")
print(f" Leverage: {LEVERAGE}x")
print(f" Position Sizing: {BALANCE_USAGE_PERCENTAGE}% of margin per trade")
print(f" Min Margin: {MIN_MARGIN_PER_TRADE} | Max: {MAX_MARGIN_PER_TRADE}")
print("")
print(" Strategy Constants (2024–2025 Tuning):")
print(f" Imbalance Threshold: {IMBALANCE_THRESHOLD:.2f} (~82% bid share)")
print(f" Wall Volume Mult: {MIN_WALL_VOLUME_MULT:.2f}×")
print(f" Delta Z-Score Threshold: {DELTA_Z_THRESHOLD:.2f}")
print(f" Zone Ticks: ±{ZONE_TICKS}")
print(f" Touch Threshold: {PRICE_TOUCH_THRESHOLD_TICKS} ticks")
print(f" Profit Target ROI: {PROFIT_TARGET_ROI * 100:.2f}%")
print(f" Stop Loss ROI: {STOP_LOSS_ROI * 100:.2f}%")
print(f" Max Hold: {MAX_HOLD_MINUTES} minutes")
print(f" Tick Size: {TICK_SIZE}")
print(f" EMA Period (trend): {EMA_PERIOD}")
print(f" ATR Window: {ATR_WINDOW_MINUTES} minutes")
print(f" Max ATR: {MAX_ATR_PERCENT * 100:.2f}% of price")
print("")
print(" HTF Trend Filter (Robust 3-State Logic):")
print(f" Interval: {HTF_TREND_INTERVAL}min")
print(f" EMA Span: {HTF_EMA_SPAN}")
print(f" Lookback Bars: {HTF_LOOKBACK_BARS}")
print(f" Min Trend Slope: {MIN_TREND_SLOPE * 100:.2f}%")
print(f" Consistency Threshold: {CONSISTENCY_THRESHOLD * 100:.0f}%")
print("")
print(" VOL-REGIME DETECTION & DYNAMIC GATES:")
print(f" LOW vol: <{VOL_REGIME_LOW_THRESHOLD*100:.2f}% ATR (Z={VOL_REGIME_Z_LOW}, Wall={VOL_REGIME_WALL_MULT_LOW}x)")
print(f" HIGH vol: >{VOL_REGIME_HIGH_THRESHOLD*100:.2f}% ATR (Z={VOL_REGIME_Z_HIGH}, Wall={VOL_REGIME_WALL_MULT_HIGH}x)")
print(f" NEUTRAL vol: between thresholds (Z={VOL_REGIME_Z_BASE}, Wall={VOL_REGIME_WALL_MULT_BASE}x)")
print("")
print(" WEIGHTED SCORE GAUNTLET:")
print(f" Enabled: {ENABLE_WEIGHTED_SCORING}")
print(f" Entry Threshold: {WEIGHTED_SCORE_ENTRY_THRESHOLD}")
print(f" Weights: Imb={WEIGHT_IMBALANCE}, Wall={WEIGHT_WALL}, Z={WEIGHT_ZSCORE}, Touch={WEIGHT_TOUCH}, Trend={WEIGHT_TREND}")
print(f" Data Fusion: CVD={WEIGHT_CVD}, LV={WEIGHT_LV}, Hurst/BOS={WEIGHT_HURST_BOS}, LSTM={WEIGHT_LSTM}")
print("")
print(" SESSION + VOLATILITY DYNAMICS:")
print(f" Enabled: {ENABLE_SESSION_DYNAMIC_PARAMS}")
print(f" Asia: {ASIA_SESSION_UTC[0]}:00–{ASIA_SESSION_UTC[1]}:00 UTC")
print(f" London: {LONDON_SESSION_UTC[0]}:00–{LONDON_SESSION_UTC[1]}:00 UTC")
print(f" New York: {NEW_YORK_SESSION_UTC[0]}:00–{NEW_YORK_SESSION_UTC[1]}:00 UTC")
print(f" TP Tightening: {ENABLE_TP_TIGHTENING} (after {DYNAMIC_TP_MINUTES}min)")
print("=" * 80 + "\n")
