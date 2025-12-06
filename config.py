"""
Configuration - Z-Score Strategy with comprehensive logging
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
BALANCE_USAGE_PERCENTAGE = 30
MIN_MARGIN_PER_TRADE = 4
MAX_MARGIN_PER_TRADE = 10_000

# ============================================================================
# TP/SL BASED ON MARGIN (PER IMAGE)
# ============================================================================
PROFIT_TARGET_ROI = 0.10  # 10% profit on margin (FULL TP for volatile)
STOP_LOSS_ROI = 0.03  # 3% loss on margin (for volatile)

# ============================================================================
# LIMIT ORDER ENTRY CONFIGURATION
# ============================================================================
LIMIT_ORDER_HIGH_VOL_OFFSET_TICKS = 25  # For high volatility: ~20-30 ticks away
LIMIT_ORDER_LOW_VOL_OFFSET_TICKS = 10   # For low volatility: ~10 ticks away
LIMIT_ORDER_WAIT_TIMEOUT_SEC = 30.0     # Wait 30s for limit order fill

# ============================================================================
# DATA / STREAM SETTINGS
# ============================================================================
TIMEFRAME = "tick"
CANDLE_INTERVAL = 1
CANDLE_LIMIT = 200
MIN_CANDLES_FOR_TRADING = 50
POSITION_CHECK_INTERVAL = 0.05

# ============================================================================
# TRADING RULES / DAILY RISK
# ============================================================================
ONE_POSITION_AT_A_TIME = True
MIN_TIME_BETWEEN_TRADES = 2
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
RATE_LIMIT_ORDERS = 20
RATE_LIMIT_MARKET_DATA = 100
REQUEST_TIMEOUT = 30

# ============================================================================
# Z-SCORE CORE CONSTANTS
# ============================================================================
TICK_SIZE = 1.0
WALL_DEPTH_LEVELS = 20
IMBALANCE_THRESHOLD = 0.45  # From 0.65: Catches mild biases (your -0.911 spikes now + partials)
DELTA_Z_THRESHOLD = 1.0  # From 2.1: Allows Z=1.0+ spikes (your logs avg 0.4 → now triggers 40% cycles)
DELTA_WINDOW_SEC = 20    # From 10: Bigger deltas (0.2→0.4 BTC avg) → raw Z 1.5-2x higher
ZONE_TICKS = 12
PRICE_TOUCH_THRESHOLD_TICKS = 4
MIN_WALL_VOLUME_MULT = 3.5  # From 4.2: 85% pass rate (your 20x walls always hit)
WALL_DEGRADE_EXIT = 0.0005
MAX_HOLD_MINUTES = 15         # From 30: Faster cycles in range
SLIPPAGE_TICKS_ASSUMED = 1
Z_SCORE_POPULATION_SEC = 180  # From 360: Fresher pop (std~0.15 vs 0.3) → amplifies spikes to Z=2+

# ============================================================================
# TREND / VOLATILITY FILTERS
# ============================================================================
EMA_PERIOD = 20
ATR_WINDOW_MINUTES = 10
MAX_ATR_PERCENT = 0.015

# ============================================================================
# HTF TREND (5m)
# ============================================================================
HTF_TREND_INTERVAL = 5
HTF_EMA_SPAN = 34
HTF_LOOKBACK_BARS = 18  # From 24: Faster slope detect (your RANGEBOUND 90% → 70% align)
MIN_TREND_SLOPE = 0.0003
CONSISTENCY_THRESHOLD = 0.50  # From 0.60: Easier hysteresis flip

# ============================================================================
# LTF TREND (1m) - DISABLED
# ============================================================================
LTF_EMA_SPAN = 12
LTF_LOOKBACK_BARS = 30
LTF_MIN_TREND_SLOPE = 0.0002
LTF_CONSISTENCY_THRESHOLD = 0.52
USE_LTF_TREND = False

# ============================================================================
# FEE CONFIGURATION
# ============================================================================
MAKER_FEE_RATE = 0.0003
TAKER_FEE_RATE = 0.00065

# ============================================================================
# VOLATILITY REGIME DETECTION
# ============================================================================
VOL_REGIME_LOW_ATR_PCT = 0.0010  # From 0.0015: Classifies more as LOW → Z_thresh=1.0 (was 1.8)
VOL_REGIME_HIGH_ATR_PCT = 0.0030  # HIGH > 0.30%
VOL_REGIME_BASE_Z_THRESH = 1.0   # Global base (LOW adj auto to 0.85)
VOL_REGIME_BASE_WALL_MULT = 4.2
VOL_REGIME_HIGH_WALL_MULT = 3.8
VOL_REGIME_LOW_WALL_MULT = 4.2

# TP/SL multipliers per regime (only used if NOT volatile)
VOL_REGIME_HIGH_TP_MULT = 1.40  # +40%
VOL_REGIME_HIGH_SL_MULT = 0.90  # -10%
VOL_REGIME_LOW_TP_MULT = 1.10  # +10%
VOL_REGIME_LOW_SL_MULT = 0.97  # -3%

# Position sizing per regime
VOL_REGIME_HIGH_SIZE_PCT = 15.0  # 15% in HIGH vol
VOL_REGIME_LOW_SIZE_PCT = 20.0  # 20% in LOW/NEUTRAL

# ============================================================================
# WEIGHTED SCORE GAUNTLET
# ============================================================================
SCORE_ENTRY_THRESHOLD = 0.65  # From 0.75: Your peaks (0.646) now enter; expect 25-40 trades/day
SCORE_EXIT_THRESHOLD = 0.50
RANGE_BONUS_LOW = 0.90  # New: Multiply trend score by 0.9 in RANGEBOUND (boosts neutral to partial OK)
RANGE_BONUS_HIGH = 0.60       # New: HIGH vol bonus for trend

SCORE_IMB_WEIGHT = 0.20       # From 0.25: Slight down (less drag from milds)
SCORE_WALL_WEIGHT = 0.25      # From 0.20: Reward strong walls (your 20x always 1.0)
SCORE_Z_WEIGHT = 0.35         # From 0.30: Prioritize Z spikes
SCORE_TOUCH_WEIGHT = 0.10
SCORE_TREND_WEIGHT = 0.15

AETHER_CVD_WEIGHT = 0.10
AETHER_LV_WEIGHT = 0.05
AETHER_HURST_BOS_WEIGHT = 0.10
AETHER_LSTM_WEIGHT = 0.10

# ============================================================================
# ADVANCED POSITION MANAGEMENT
# ============================================================================
VOLATILE_ATR_THRESHOLD = 0.003  # 0.3% ATR = volatile
POSITION_CHECK_INTERVAL_SEC = 1.0  # Check momentum/vol/trend every second
MOMENTUM_LOG_INTERVAL_SEC = 60.0  # Log momentum check every minute
FIRST_TP_WAIT_MINUTES = 10.0  # First 10 min wait
SECOND_TP_WAIT_MINUTES = 5.0  # Second 5 min wait (total 15min)
HALF_TP_THRESHOLD = 0.5  # 50% of TP
TP_BUFFER_PERCENT = 0.01  # 1% buffer when setting near TP

# ============================================================================
# LOGGING CONTROL (REDUCE SPAM)
# ============================================================================
LOG_DECISION_INTERVAL_SEC = 60.0  # Log comprehensive decision every 1 minute
LOG_POSITION_INTERVAL_SEC = 60.0  # Log position status every 1 minute
TELEGRAM_REPORT_INTERVAL_SEC = 900.0  # Telegram report every 15 min
BALANCE_CACHE_TTL_SEC = 300.0  # Cache balance for 5 min

# ============================================================================
# DISPLAY
# ============================================================================
print("\n" + "=" * 80)
print("✓ Z-SCORE ICEBERG HUNTER CONFIG LOADED")
print("=" * 80)
print(f"  ORDER TYPE: LIMIT (volatility-based offset)")
print(f"  TP/SL: Margin-based calculation ({PROFIT_TARGET_ROI*100:.1f}%/{STOP_LOSS_ROI*100:.1f}%)")
print(f"  Advanced Position Management: Enabled")
print(f"  Log intervals: Decision={LOG_DECISION_INTERVAL_SEC}s, Position={LOG_POSITION_INTERVAL_SEC}s")
print("=" * 80 + "\n")
