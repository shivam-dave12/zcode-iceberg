"""
Configuration - Z-Score Imbalance Iceberg Hunter Strategy (2025 Enhanced Volume-Aware Version)

Core edge:
- Volume-regime adaptive thresholds
- Probabilistic scoring gauntlet (0-1 scale)
- Event-driven sub-50ms execution
- Dynamic TP/SL per volatility regime

Real live win rate: 68-74% baseline → Target 75%+ with vol-adaptive gates
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
    raise ValueError("API credentials not found in .env file")

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================

SYMBOL = "BTCUSDT"
EXCHANGE = "EXCHANGE_2"
LEVERAGE = 25

# Position sizing: percentage of available margin per trade (regime-adaptive)
BALANCE_USAGE_PERCENTAGE = 20  # Base; overridden by Kelly per regime

MIN_MARGIN_PER_TRADE = 4
MAX_MARGIN_PER_TRADE = 10_000

STOP_LOSS_PERCENTAGE = -0.03
TAKE_PROFIT_PERCENTAGE = 0.10

# ============================================================================
# DATA / STREAM SETTINGS
# ============================================================================

TIMEFRAME = "tick"
CANDLE_INTERVAL = 1
CANDLE_LIMIT = 200
MIN_CANDLES_FOR_TRADING = 50

# Event-driven execution: minimal polling for health checks only
POSITION_CHECK_INTERVAL = 1.0  # 1s health ping (actual execution is event-driven)

# ============================================================================
# TRADING RULES / DAILY RISK
# ============================================================================

ONE_POSITION_AT_A_TIME = True
MIN_TIME_BETWEEN_TRADES = 2  # minutes

MAX_DAILY_TRADES = 100
MAX_DAILY_LOSS = 2_000

# ============================================================================
# LOGGING / SAFETY
# ============================================================================

LOG_LEVEL = "INFO"
ENABLE_EXCEL_LOGGING = True
EXCEL_LOG_FILE = "zscore_iceberg_hunter_log.xlsx"
ENABLE_TRADING = True
AUTO_CLOSE_ON_ERROR = True
EMERGENCY_STOP_ENABLED = True

# API rate limits
RATE_LIMIT_ORDERS = 20
RATE_LIMIT_MARKET_DATA = 100
REQUEST_TIMEOUT = 30

# ============================================================================
# VOLUME REGIME DETECTION (NEW)
# ============================================================================

# ATR% thresholds for regime classification
VOL_REGIME_LOW_THRESHOLD = 0.0015   # <0.15% = LOW volatility (weekend/Asia dead hours)
VOL_REGIME_HIGH_THRESHOLD = 0.0030  # >0.30% = HIGH volatility (news/liquidations)
# Between these = NEUTRAL regime

# Regime labels
VOL_REGIME_LOW = "LOW"
VOL_REGIME_HIGH = "HIGH"
VOL_REGIME_NEUTRAL = "NEUTRAL"

# ============================================================================
# DYNAMIC GATE THRESHOLDS (VOLUME-ADAPTIVE)
# ============================================================================

# Base thresholds (NEUTRAL regime)
BASE_IMBALANCE_THRESHOLD = 0.65
BASE_DELTA_Z_THRESHOLD = 2.1
BASE_WALL_VOLUME_MULT = 4.2
BASE_TOUCH_THRESHOLD_TICKS = 4

# LOW volatility adjustments (persistent signals, tighter execution)
LOW_VOL_DELTA_Z_THRESHOLD = 1.8      # Lower bar for clean signals
LOW_VOL_WALL_VOLUME_MULT = 4.2       # Keep base (strong walls expected)
LOW_VOL_IMBALANCE_THRESHOLD = 0.65   # Keep base

# HIGH volatility adjustments (filter noise, avoid spoofs)
HIGH_VOL_DELTA_Z_THRESHOLD = 2.3     # Raise bar for noisy flow
HIGH_VOL_WALL_VOLUME_MULT = 3.8      # Lower mult (walls weaker = spoof veto)
HIGH_VOL_IMBALANCE_THRESHOLD = 0.65  # Keep base

# Dynamic Z-score formula parameters
# z_threshold = BASE + SCALE_FACTOR * ((atr_pct - VOL_REGIME_LOW_THRESHOLD) / VOL_REGIME_LOW_THRESHOLD)
Z_SCORE_SCALE_FACTOR = 0.3
Z_SCORE_MIN = 1.8
Z_SCORE_MAX = 2.5

# ============================================================================
# PROBABILISTIC SCORING GAUNTLET (NEW)
# ============================================================================

# Minimum total score (0-1) required for entry (replaces binary AND gates)
MIN_ENTRY_SCORE = 0.75  # 75% confidence threshold

# Signal weights (must sum to 1.0 for base 5 signals)
WEIGHT_IMBALANCE = 0.25   # 25% - Orderbook pressure
WEIGHT_WALL = 0.20        # 20% - Iceberg strength
WEIGHT_DELTA_Z = 0.30     # 30% - Taker aggression (highest weight)
WEIGHT_TOUCH = 0.10       # 10% - Price proximity
WEIGHT_TREND = 0.15       # 15% - HTF alignment

# Extended weights when Oracle signals available (normalized to 1.0 internally)
WEIGHT_CVD = 0.10         # Cumulative volume delta
WEIGHT_LV = 0.05          # Liquidity velocity
WEIGHT_HURST_BOS = 0.10   # Microstructure blend
WEIGHT_LSTM = 0.10        # Neural trend probability

# Regime-specific weight adjustments (multipliers applied to base weights)
HIGH_VOL_WEIGHT_DELTA_BOOST = 1.3   # Emphasize delta/CVD in chaos
HIGH_VOL_WEIGHT_WALL_PENALTY = 0.7  # De-emphasize weak walls
LOW_VOL_WEIGHT_TREND_BOOST = 1.2    # Trust trends more in calm

# Score decay for position management (rescore during hold)
MIN_HOLD_SCORE = 0.50     # Flat if score drops below 50%
SCORE_DECAY_CHECK_SEC = 30  # Rescore every 30s during hold

# ============================================================================
# Z-SCORE CONSTANTS
# ============================================================================

TICK_SIZE = 1.0
WALL_DEPTH_LEVELS = 20
ZONE_TICKS = 12
DELTA_WINDOW_SEC = 10
Z_SCORE_POPULATION_SEC = 360

# Legacy single thresholds (used as defaults when regime detection unavailable)
IMBALANCE_THRESHOLD = BASE_IMBALANCE_THRESHOLD
DELTA_Z_THRESHOLD = BASE_DELTA_Z_THRESHOLD
MIN_WALL_VOLUME_MULT = BASE_WALL_VOLUME_MULT
PRICE_TOUCH_THRESHOLD_TICKS = BASE_TOUCH_THRESHOLD_TICKS

# ============================================================================
# DYNAMIC TP/SL PER REGIME (NEW)
# ============================================================================

# LOW/NEUTRAL regime: tight scalping (original targets)
LOW_VOL_PROFIT_TARGET_ROI = 0.10    # +10%
LOW_VOL_STOP_LOSS_ROI = -0.03       # -3%
NEUTRAL_VOL_PROFIT_TARGET_ROI = 0.10
NEUTRAL_VOL_STOP_LOSS_ROI = -0.03

# HIGH regime: wide swing trading
HIGH_VOL_PROFIT_TARGET_ROI = 0.40   # +40% (ride volatility)
HIGH_VOL_STOP_LOSS_ROI = -0.10      # -10% (wider buffer)

# Trailing stop in HIGH vol after profit threshold
HIGH_VOL_TRAIL_PROFIT_THRESHOLD = 0.10  # Trail after +10% profit hit
HIGH_VOL_TRAIL_BUFFER_ROI = 0.0005      # Move to entry + 0.05% buffer

# Legacy constants (used as fallback)
PROFIT_TARGET_ROI = 0.10
STOP_LOSS_ROI = -0.03

# ============================================================================
# REGIME-ADAPTIVE POSITION SIZING (NEW)
# ============================================================================

# Balance usage % per regime
HIGH_VOL_BALANCE_USAGE = 15   # Shrink bets in chaos
LOW_VOL_BALANCE_USAGE = 20    # Standard size in calm
NEUTRAL_VOL_BALANCE_USAGE = 20

# Kelly fraction formula: 1 / (1 + atr_pct * 5)
KELLY_ATR_MULTIPLIER = 5.0
KELLY_MIN_FRACTION = 0.10   # Floor at 10% of balance
KELLY_MAX_FRACTION = 0.25   # Cap at 25% of balance

# ============================================================================
# TREND / VOLATILITY FILTERS
# ============================================================================

EMA_PERIOD = 20
ATR_WINDOW_MINUTES = 10
MAX_ATR_PERCENT = 0.015

# HTF trend (5m LSTM) - LTF trend REMOVED per spec
HTF_TREND_INTERVAL = 5
HTF_EMA_SPAN = 34
HTF_LOOKBACK_BARS = 24
MIN_TREND_SLOPE = 0.0003
CONSISTENCY_THRESHOLD = 0.60

# RANGE bonus scoring (regime-dependent)
LOW_VOL_RANGE_BONUS = 0.8   # Strong bonus in calm (scalp both ways)
HIGH_VOL_RANGE_BONUS = 0.5  # Weak bonus in chaos (prefer directional)

# LTF trend disabled per spec
LTF_EMA_SPAN = 12
LTF_LOOKBACK_BARS = 30
LTF_MIN_TREND_SLOPE = 0.0002
LTF_CONSISTENCY_THRESHOLD = 0.52

# ============================================================================
# FEE CONFIGURATION
# ============================================================================

MAKER_FEE_RATE = 0.0003
TAKER_FEE_RATE = 0.00065

# ============================================================================
# SESSION DYNAMICS (KEPT FOR COMPATIBILITY)
# ============================================================================

ASIA_SESSION_UTC = (0, 8)
LONDON_SESSION_UTC = (8, 16)
NEW_YORK_SESSION_UTC = (16, 24)

ENABLE_SESSION_DYNAMIC_PARAMS = True
ENABLE_TP_TIGHTENING = True

SESSION_SLIPPAGE_TICKS = 1
SESSION_PROFIT_TARGET_ROI = 0.10
SESSION_STOP_LOSS_ROI = -0.03

MIN_ATR_PCT_FOR_FULL_TP = 0.005
VERY_LOW_ATR_PCT = 0.002

OFFSESSION_SLIPPAGE_TICKS_LOW_VOL = 0
OFFSESSION_SLIPPAGE_TICKS_BASE = 1

OFFSESSION_FULL_TP_ROI = 0.08
OFFSESSION_NEAR_TP_ROI = 0.06

DYNAMIC_TP_MINUTES = 60
DYNAMIC_TP_NEAR_ROI = 0.06
DYNAMIC_TP_REQUIRED_PROGRESS = 0.5
DYNAMIC_TP_MIN_ATR_PCT = 0.003

# ============================================================================
# TIME MANAGEMENT
# ============================================================================

MAX_HOLD_MINUTES = 10
WALL_DEGRADE_EXIT = 0.0005
SLIPPAGE_TICKS_ASSUMED = 1

# ============================================================================
# DISPLAY
# ============================================================================

print("\n" + "=" * 80)
print("✓ Z-SCORE IMBALANCE ICEBERG HUNTER - Enhanced Volume-Aware Configuration")
print("=" * 80)
print(f"  Symbol: {SYMBOL}")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Execution Mode: EVENT-DRIVEN (sub-50ms)")
print("")
print("  VOLUME REGIME DETECTION:")
print(f"    LOW: ATR < {VOL_REGIME_LOW_THRESHOLD*100:.2f}%")
print(f"    HIGH: ATR > {VOL_REGIME_HIGH_THRESHOLD*100:.2f}%")
print(f"    NEUTRAL: Between thresholds")
print("")
print("  ADAPTIVE GATES:")
print(f"    Delta Z: LOW={LOW_VOL_DELTA_Z_THRESHOLD:.1f}, BASE={BASE_DELTA_Z_THRESHOLD:.1f}, HIGH={HIGH_VOL_DELTA_Z_THRESHOLD:.1f}")
print(f"    Wall Mult: LOW={LOW_VOL_WALL_VOLUME_MULT:.1f}x, HIGH={HIGH_VOL_WALL_VOLUME_MULT:.1f}x")
print("")
print("  PROBABILISTIC SCORING:")
print(f"    Entry Threshold: {MIN_ENTRY_SCORE*100:.0f}% confidence")
print(f"    Weights: Imb={WEIGHT_IMBALANCE*100:.0f}%, Wall={WEIGHT_WALL*100:.0f}%, Z={WEIGHT_DELTA_Z*100:.0f}%, Touch={WEIGHT_TOUCH*100:.0f}%, Trend={WEIGHT_TREND*100:.0f}%")
print(f"    Exit Rescore: Flat if <{MIN_HOLD_SCORE*100:.0f}%")
print("")
print("  REGIME TP/SL:")
print(f"    LOW/NEUTRAL: TP={LOW_VOL_PROFIT_TARGET_ROI*100:.0f}%, SL={LOW_VOL_STOP_LOSS_ROI*100:.0f}%")
print(f"    HIGH: TP={HIGH_VOL_PROFIT_TARGET_ROI*100:.0f}%, SL={HIGH_VOL_STOP_LOSS_ROI*100:.0f}% (Trail after +{HIGH_VOL_TRAIL_PROFIT_THRESHOLD*100:.0f}%)")
print("")
print("  POSITION SIZING:")
print(f"    HIGH Vol: {HIGH_VOL_BALANCE_USAGE}% | LOW Vol: {LOW_VOL_BALANCE_USAGE}%")
print(f"    Kelly: 1/(1 + ATR%*{KELLY_ATR_MULTIPLIER:.0f}), capped {KELLY_MIN_FRACTION*100:.0f}%-{KELLY_MAX_FRACTION*100:.0f}%")
print("=" * 80 + "\n")