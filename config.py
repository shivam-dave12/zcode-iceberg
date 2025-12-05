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
STOP_LOSS_PERCENTAGE = -0.03
TAKE_PROFIT_PERCENTAGE = 0.10

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
IMBALANCE_THRESHOLD = 0.65
DELTA_Z_THRESHOLD = 2.1
DELTA_WINDOW_SEC = 10
ZONE_TICKS = 12
PRICE_TOUCH_THRESHOLD_TICKS = 4
MIN_WALL_VOLUME_MULT = 4.2
PROFIT_TARGET_ROI = 0.05  # 5% profit on margin
STOP_LOSS_ROI = 0.01  # 1% loss on margin
WALL_DEGRADE_EXIT = 0.0005
MAX_HOLD_MINUTES = 10
SLIPPAGE_TICKS_ASSUMED = 1
Z_SCORE_POPULATION_SEC = 360

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
HTF_LOOKBACK_BARS = 24
MIN_TREND_SLOPE = 0.0003
CONSISTENCY_THRESHOLD = 0.60

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
# SESSION DYNAMICS
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
# VOLATILITY REGIME DETECTION
# ============================================================================
VOL_REGIME_LOW_ATR_PCT = 0.0015
VOL_REGIME_HIGH_ATR_PCT = 0.0030
VOL_REGIME_BASE_Z_THRESH = 2.1
VOL_REGIME_Z_SCALE_FACTOR = 0.3
VOL_REGIME_Z_NORMALIZE_PCT = 0.0015
VOL_REGIME_BASE_WALL_MULT = 4.2
VOL_REGIME_HIGH_WALL_MULT = 3.8
VOL_REGIME_LOW_WALL_MULT = 4.2

VOL_REGIME_HIGH_TP_MULT = 1.40
VOL_REGIME_HIGH_SL_MULT = 0.90
VOL_REGIME_LOW_TP_MULT = 1.10
VOL_REGIME_LOW_SL_MULT = 0.97

VOL_REGIME_HIGH_SIZE_PCT = 15.0
VOL_REGIME_LOW_SIZE_PCT = 20.0

HIGH_VOL_TRAIL_PROFIT_PCT = 0.10
HIGH_VOL_TRAIL_BUFFER_PCT = 0.0005

# ============================================================================
# WEIGHTED SCORE GAUNTLET
# ============================================================================
SCORE_ENTRY_THRESHOLD = 0.75
SCORE_EXIT_THRESHOLD = 0.50
RANGE_BONUS_LOW = 0.8
RANGE_BONUS_HIGH = 0.5

SCORE_IMB_WEIGHT = 0.25
SCORE_WALL_WEIGHT = 0.20
SCORE_Z_WEIGHT = 0.30
SCORE_TOUCH_WEIGHT = 0.10
SCORE_TREND_WEIGHT = 0.15

AETHER_CVD_WEIGHT = 0.10
AETHER_LV_WEIGHT = 0.05
AETHER_HURST_BOS_WEIGHT = 0.10
AETHER_LSTM_WEIGHT = 0.10

WINPROB_BASE = 0.4
WINPROB_LSTM_WEIGHT = 0.2
WINPROB_Z_WEIGHT = 0.2
WINPROB_CVD_WEIGHT = 0.1
WINPROB_LV_WEIGHT = 0.1
WINPROB_ENTRY_THRESHOLD = 0.6

# ============================================================================
# ADVANCED POSITION MANAGEMENT
# ============================================================================
VOLATILE_ATR_THRESHOLD = 0.003  # 0.3% ATR = volatile
POSITION_CHECK_INTERVAL_SEC = 5.0  # Check momentum/vol/trend every 5s
FIRST_TP_WAIT_MINUTES = 10.0  # First 10 min wait
SECOND_TP_WAIT_MINUTES = 10.0  # Second 10 min wait if conditions favorable
HALF_TP_THRESHOLD = 0.5  # 50% of TP
TP_BUFFER_PERCENT = 0.005  # 0.5% buffer when setting near TP

# ============================================================================
# LOGGING CONTROL
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
print(f" TP/SL: Margin-based calculation ({PROFIT_TARGET_ROI*100:.1f}%/{STOP_LOSS_ROI*100:.1f}%)")
print(f" Advanced Position Management: Enabled")
print(f" Log intervals: Decision={LOG_DECISION_INTERVAL_SEC}s, Position={LOG_POSITION_INTERVAL_SEC}s")
print("=" * 80 + "\n")
