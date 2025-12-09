"""
Configuration - Z-Score Strategy with SESSION-BASED PARAMETERS
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
# TRADING PARAMETERS
# ============================================================================
SYMBOL = "BTCUSDT"
EXCHANGE = "EXCHANGE_2"
LEVERAGE = 25
BALANCE_USAGE_PERCENTAGE = 30
MIN_MARGIN_PER_TRADE = 4
MAX_MARGIN_PER_TRADE = 10000

# ============================================================================
# SESSION-BASED VOLATILITY THRESHOLDS
# ============================================================================
# Weekend / Off-Hours (Conservative)
WEEKEND_PARAMS = {
    "z_threshold": 0.80,
    "imbalance_threshold": 0.50,
    "entry_score_threshold": 0.70,
    "wall_multiplier": 5.0,
    "min_signal_gap_sec": 180.0,  # 3 minutes
    "tp_roi_target": 0.04,        # 4%
    "sl_roi_max": 0.02,           # 2%
}

# Major Sessions (Standard)
MAJOR_SESSION_PARAMS = {
    "z_threshold": 2.1,
    "imbalance_threshold": 0.65,
    "entry_score_threshold": 0.85,
    "wall_multiplier": 4.2,
    "min_signal_gap_sec": 60.0,   # 1 minute
    "tp_roi_target": 0.12,        # 12%
    "sl_roi_max": 0.03,          # 3%
}

# Overlap Sessions (Aggressive)
OVERLAP_SESSION_PARAMS = {
    "z_threshold": 2.5,
    "imbalance_threshold": 0.65,
    "entry_score_threshold": 0.65,
    "wall_multiplier": 4.0,
    "min_signal_gap_sec": 45.0,   # 45 seconds
    "tp_roi_target": 0.10,        # 10%
    "sl_roi_max": 0.02,          # 2%
}

# ============================================================================
# ORDER PARAMETERS
# ============================================================================
PROFIT_TARGET_ROI = 0.10
STOP_LOSS_ROI = 0.03
LIMIT_ORDER_HIGH_VOL_OFFSET_TICKS = 25
LIMIT_ORDER_LOW_VOL_OFFSET_TICKS = 10
LIMIT_ORDER_WAIT_TIMEOUT_SEC = 60.0

# ============================================================================
# DATA PARAMETERS
# ============================================================================
TIMEFRAME = "tick"
CANDLE_INTERVAL = 1
CANDLE_LIMIT = 200
MIN_CANDLES_FOR_TRADING = 50

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
ONE_POSITION_AT_A_TIME = True
MIN_TIME_BETWEEN_TRADES = 2
MAX_DAILY_TRADES = 100
MAX_DAILY_LOSS = 2000

# ============================================================================
# LOGGING & MONITORING
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
# TECHNICAL ANALYSIS PARAMETERS
# ============================================================================
TICK_SIZE = 1.0
WALL_DEPTH_LEVELS = 20
IMBALANCE_THRESHOLD = 0.45
DELTA_Z_THRESHOLD = 1.0
DELTA_WINDOW_SEC = 20
ZONE_TICKS = 12
PRICE_TOUCH_THRESHOLD_TICKS = 4
MIN_WALL_VOLUME_MULT = 3.5
WALL_DEGRADE_EXIT = 0.0005
MAX_HOLD_MINUTES = 15
SLIPPAGE_TICKS_ASSUMED = 1
ZSCORE_POPULATION_SEC = 180

# ============================================================================
# TREND PARAMETERS
# ============================================================================
EMA_PERIOD = 20
ATR_WINDOW_MINUTES = 10
MAX_ATR_PERCENT = 0.015
HTF_TREND_INTERVAL = 5
HTF_EMA_SPAN = 34
HTF_LOOKBACK_BARS = 18
MIN_TREND_SLOPE = 0.0003
CONSISTENCY_THRESHOLD = 0.50
LTF_EMA_SPAN = 12
LTF_LOOKBACK_BARS = 30
LTF_MIN_TREND_SLOPE = 0.0002
LTF_CONSISTENCY_THRESHOLD = 0.52
USE_LTF_TREND = False

# ============================================================================
# FEES
# ============================================================================
MAKER_FEE_RATE = 0.0003
TAKER_FEE_RATE = 0.00065

# ============================================================================
# VOLATILITY REGIMES
# ============================================================================
VOL_REGIME_LOW_ATR_PCT = 0.0010
VOL_REGIME_HIGH_ATR_PCT = 0.0030
VOL_REGIME_BASE_Z_THRESH = 1.0
VOL_REGIME_BASE_WALL_MULT = 4.2
VOL_REGIME_HIGH_WALL_MULT = 3.8
VOL_REGIME_LOW_WALL_MULT = 4.2
VOL_REGIME_HIGH_TP_MULT = 1.40
VOL_REGIME_HIGH_SL_MULT = 0.90
VOL_REGIME_LOW_TP_MULT = 1.10
VOL_REGIME_LOW_SL_MULT = 0.97
VOL_REGIME_HIGH_SIZE_PCT = 15.0
VOL_REGIME_LOW_SIZE_PCT = 20.0

# ============================================================================
# SCORING WEIGHTS
# ============================================================================
SCORE_ENTRY_THRESHOLD = 0.65
SCORE_EXIT_THRESHOLD = 0.50
RANGE_BONUS_LOW = 0.90
RANGE_BONUS_HIGH = 0.60
SCORE_IMB_WEIGHT = 0.20
SCORE_WALL_WEIGHT = 0.25
SCORE_Z_WEIGHT = 0.35
SCORE_TOUCH_WEIGHT = 0.10
SCORE_TREND_WEIGHT = 0.15
AETHER_CVD_WEIGHT = 0.10
AETHER_LV_WEIGHT = 0.05
AETHER_HURST_BOS_WEIGHT = 0.10
AETHER_LSTM_WEIGHT = 0.10

# ============================================================================
# VOLATILITY GATES
# ============================================================================
VOLATILE_ATR_THRESHOLD = 0.003
POSITION_CHECK_INTERVAL_SEC = 1.0
MOMENTUM_LOG_INTERVAL_SEC = 60.0
FIRST_TP_WAIT_MINUTES = 10.0
SECOND_TP_WAIT_MINUTES = 5.0
HALF_TP_THRESHOLD = 0.5
TP_BUFFER_PERCENT = 0.01

# ============================================================================
# LOGGING INTERVALS
# ============================================================================
LOG_DECISION_INTERVAL_SEC = 60.0
LOG_POSITION_INTERVAL_SEC = 60.0
TELEGRAM_REPORT_INTERVAL_SEC = 900.0
BALANCE_CACHE_TTL_SEC = 300.0

# ========================================
# ORDER STATUS CHECK INTERVALS (RATE LIMIT PROTECTION)
# ========================================
ORDER_STATUS_CHECK_INTERVAL_SEC = 30.0  # Check TP/SL status every 30s (was 10s)
POSITION_STATUS_CHECK_INTERVAL_SEC = 60.0  # Check position via API every 60s
EARLY_FILL_CHECK_INTERVAL_SEC = 5.0  # Check for early fill every 5s

# ========================================
# BREAKEVEN FEE CALCULATION
# ========================================
MAKER_FEE_PCT = 0.00024  # 0.024%
TAKER_FEE_PCT = 0.0006   # 0.06%
GST_ON_FEES_PCT = 0.18   # 18% GST on trading fees

# Add to your existing config.py

# ========================================
# TP/SL REPLACEMENT SAFETY BUFFERS
# ========================================
MIN_TP_DISTANCE_TICKS = 10  # Minimum 10 ticks away from current price
MIN_SL_DISTANCE_TICKS = 10  # Minimum 10 ticks away from current price
TICK_SIZE = 0.10  # BTC/USDT tick size

# Calculate minimum distance in price
MIN_TP_DISTANCE_PRICE = MIN_TP_DISTANCE_TICKS * TICK_SIZE  # $1.00
MIN_SL_DISTANCE_PRICE = MIN_SL_DISTANCE_TICKS * TICK_SIZE  # $1.00


# Breakeven buffer = entry fee + exit fee (assuming taker on both sides for safety)
BREAKEVEN_FEE_BUFFER_PCT = (TAKER_FEE_PCT * 2) * (1 + GST_ON_FEES_PCT)  # ~0.14%


if __name__ == "__main__":
    print("=" * 80)
    print("Z-SCORE ICEBERG HUNTER CONFIG LOADED")
    print("=" * 80)
    print(f"ORDER TYPE: LIMIT (volatility-based offset)")
    print(f"TP/SL: Margin-based calculation ({PROFIT_TARGET_ROI*100:.1f}%/{STOP_LOSS_ROI*100:.1f}%)")
    print("Advanced Position Management: Enabled")
    print(f"Log intervals: Decision={LOG_DECISION_INTERVAL_SEC}s, Position={LOG_POSITION_INTERVAL_SEC}s")
    print("=" * 80)
