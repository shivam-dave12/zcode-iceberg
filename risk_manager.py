# risk_manager.py
"""
Risk Manager - updated attributes and position sizing behavior.

Key fixes:
- Added missing total_trades and winning_trades attributes
- Balance caching TTL for external calls only
- calculate_position_size_regime_aware fetches balance internally
- update_trade_stats increments both daily and lifetime stats
"""
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from futures_api import FuturesAPI
import config

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.api = FuturesAPI(api_key=config.COINSWITCH_API_KEY, secret_key=config.COINSWITCH_SECRET_KEY)

        # Daily statistics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_reset_date = datetime.now().date()

        # ✅ ADDED
        self.total_trades = 0
        self.winning_trades = 0

        # Trade tracking
        self.open_trades = 0
        self.total_margin_used = 0.0

        # Balance cache (3s TTL for external calls)
        self._last_balance_time = 0.0
        self._cached_balance = None

        logger.info("✓ RiskManager initialized")

    def _reset_daily_stats_if_needed(self):
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"Resetting daily stats (prev trades {self.daily_trades}, pnl {self.daily_pnl:.2f})")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_reset_date = current_date

    def get_available_balance(self) -> Optional[Dict]:
        now = time.time()
        # throttle to 1 call per 3 seconds
        if now - self._last_balance_time < 3.0 and self._cached_balance is not None:
            logger.debug("Using cached balance")
            return self._cached_balance

        self._last_balance_time = now
        try:
            resp = self.api.get_wallet_balance()
            if "data" in resp and "base_asset_balances" in resp["data"]:
                balances = resp["data"]["base_asset_balances"]
                for asset in balances:
                    if asset.get("base_asset") == "USDT":
                        balances_dict = asset.get("balances", {})
                        available_usdt = float(balances_dict.get("total_available_balance", 0.0))
                        self._cached_balance = {
                            "available": available_usdt,
                            "total": float(balances_dict.get("total_balance", 0.0)),
                            "used": float(balances_dict.get("total_blocked_balance", 0.0)),
                            "timestamp": now
                        }
                        logger.info(f"✓ Fresh balance: {available_usdt:.2f} USDT")
                        return self._cached_balance
            logger.warning("Balance response had no USDT entry")
            return self._cached_balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self._cached_balance

    def check_trading_allowed(self) -> Tuple[bool, str]:
        self._reset_daily_stats_if_needed()
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"Daily trades limit reached ({self.daily_trades}/{config.MAX_DAILY_TRADES})"
        if self.daily_pnl < -config.MAX_DAILY_LOSS:
            return False, f"Daily loss cap reached (${self.daily_pnl:.2f} / -${config.MAX_DAILY_LOSS})"
        if self.open_trades >= config.MAX_CONCURRENT_POSITIONS:
            return False, f"Concurrent positions limit reached ({self.open_trades}/{config.MAX_CONCURRENT_POSITIONS})"
        balance_info = self.get_available_balance()
        if not balance_info:
            return False, "Cannot fetch balance"
        if float(balance_info.get("available", 0.0)) < config.MIN_MARGIN_PER_TRADE:
            return False, f"Insufficient balance (${balance_info.get('available',0):.2f})"
        return True, "OK"

    def calculate_position_size_regime_aware(self, entry_price: float, vol_regime: str) -> Tuple[float, float]:
        """
        Fetches balance internally and returns (margin_to_use, quantity_in_btc)
        Position size is based on BALANCE_USAGE_PERCENTAGE and vol regime size multipliers.
        """
        balance_info = self.get_available_balance()
        if not balance_info:
            logger.error("Cannot calculate position size without balance")
            return 0.0, 0.0

        available = float(balance_info.get("available", 0.0))
        pct = config.BALANCE_USAGE_PERCENTAGE / 100.0
        regime_pct = pct
        # Use vol-regime size multipliers if present in config (fallback to pct)
        try:
            if vol_regime == "HIGH":
                regime_pct = config.VOL_REGIME_SIZE_HIGH_PCT
            elif vol_regime == "LOW":
                regime_pct = config.VOL_REGIME_SIZE_LOW_PCT
            else:
                regime_pct = (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0
        except Exception:
            regime_pct = pct

        margin_to_use = max(config.MIN_MARGIN_PER_TRADE, min(config.MAX_MARGIN_PER_TRADE, available * regime_pct))
        # Convert margin to quantity using leverage:
        leverage = float(config.LEVERAGE)
        qty = (margin_to_use * leverage) / max(1.0, float(entry_price))
        return margin_to_use, qty

    def update_trade_stats(self, pnl: float):
        """Update daily and lifetime stats. Positive pnl increments wins."""
        self._reset_daily_stats_if_needed()
        self.daily_trades += 1
        self.total_trades += 1
        self.daily_pnl += pnl
        if pnl > 0:
            self.daily_wins += 1
            self.winning_trades += 1
        else:
            self.daily_losses += 1

