"""
Risk Manager - Handles position sizing, risk limits, and trade statistics
✅ FIXED: Balance fetched ONCE per trade cycle (NOT every 3 seconds)
✅ FIXED: Added missing total_trades, winning_trades attributes  
✅ PRODUCTION: Explicit lifecycle management
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
    """Manages risk parameters and position sizing"""

    def __init__(self):
        """Initialize risk manager"""
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )
        
        # Daily statistics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_reset_date = datetime.now().date()
        
        # ✅ FIXED: Added missing lifetime attributes
        self.total_trades = 0
        self.winning_trades = 0
        
        # Trade tracking
        self.open_trades = 0
        self.total_margin_used = 0.0
        
        # ✅ FIXED: Proper balance lifecycle (NOT every 3 seconds)
        self._balance_cache: Optional[Dict] = None
        self._balance_cache_time: float = 0.0
        self._balance_cache_ttl: float = 30.0  # 30s TTL for safety checks
        
        logger.info("✓ RiskManager initialized - PRODUCTION MODE")

    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"New day detected - resetting daily stats (prev: {self.daily_trades} trades, P&L: {self.daily_pnl:.2f})")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_reset_date = current_date

    def _fetch_balance_from_api(self) -> Optional[Dict]:
        """
        ✅ PRODUCTION: Single API call with proper CoinSwitch parsing
        Called ONCE per trade evaluation cycle by get_balance_for_trade_evaluation()
        """
        try:
            logger.debug("Fetching fresh balance from API...")
            response = self.api.get_wallet_balance()
            logger.debug(f"Raw wallet response: {response}")

            if "data" in response and "base_asset_balances" in response["data"]:
                balances = response["data"]["base_asset_balances"]
                for asset in balances:
                    if asset.get("base_asset") == "USDT":
                        balances_dict = asset.get("balances", {})
                        available_usdt = float(balances_dict.get("total_available_balance", 0.0))
                        
                        result = {
                            "available": available_usdt,
                            "total": float(balances_dict.get("total_balance", 0.0)),
                            "used": float(balances_dict.get("total_blocked_balance", 0.0)),
                            "timestamp": time.time()
                        }
                        
                        # Update cache for safety checks
                        self._balance_cache = result
                        self._balance_cache_time = time.time()
                        
                        logger.info(f"✓ Fresh balance: {available_usdt:.2f} USDT available")
                        return result

            logger.warning(f"Balance API no USDT data: {response}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}", exc_info=True)
            return None

    def get_balance_for_trade_evaluation(self) -> Optional[Dict]:
        """
        ✅ NEW PRIMARY METHOD: Call this ONCE at start of each trade opportunity
        Fetches fresh balance and caches for safety checks during that cycle
        """
        return self._fetch_balance_from_api()

    def get_cached_balance(self) -> Optional[Dict]:
        """
        ✅ SECONDARY METHOD: Use for periodic safety checks (30s TTL)
        Does NOT trigger new API call if cache valid
        """
        now = time.time()
        if self._balance_cache and (now - self._balance_cache_time < self._balance_cache_ttl):
            return self._balance_cache
        
        # Cache expired/missing - fetch fresh
        logger.debug("Balance cache expired - fetching fresh")
        return self._fetch_balance_from_api()

    def get_available_balance(self) -> Optional[Dict]:
        """
        ✅ BACKWARD COMPATIBLE: Existing code continues to work unchanged
        """
        return self.get_cached_balance()

    def check_trading_allowed(self, balance_available: Optional[float] = None) -> Tuple[bool, str]:
        """
        ✅ UPDATED: Accept pre-fetched balance to avoid redundant API calls
        """
        self._reset_daily_stats_if_needed()

        # Daily limits
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit ({self.daily_trades}/{config.MAX_DAILY_TRADES})"
        
        if self.daily_pnl < -config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit (${self.daily_pnl:.2f}/-${config.MAX_DAILY_LOSS})"
        
        if self.open_trades >= config.MAX_CONCURRENT_POSITIONS:
            return False, f"Max positions ({self.open_trades}/{config.MAX_CONCURRENT_POSITIONS})"

        # Balance check
        if balance_available is None:
            balance_info = self.get_cached_balance()
            if not balance_info:
                return False, "Cannot fetch balance"
            balance_available = float(balance_info.get("available", 0.0))
        
        if balance_available < config.MIN_MARGIN_PER_TRADE:
            return False, f"Low balance (${balance_available:.2f} < ${config.MIN_MARGIN_PER_TRADE})"
        
        return True, "OK"

    def calculate_position_size_regime_aware(
        self,
        entry_price: float,
        vol_regime: str,
        balance_available: float,  # ✅ REQUIRED: Pass pre-fetched balance
    ) -> Tuple[float, float]:
        """
        ✅ FIXED: Uses pre-fetched balance (NO API call here)
        """
        if balance_available <= 0:
            logger.error("Cannot size position - invalid balance")
            return 0.0, 0.0
        
        # Regime sizing
        if vol_regime == "HIGH":
            size_pct = config.VOL_REGIME_SIZE_HIGH_PCT
        elif vol_regime == "LOW":
            size_pct = config.VOL_REGIME_SIZE_LOW_PCT
        else:
            size_pct = (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0
        
        margin_to_use = balance_available * size_pct
        margin_to_use = max(config.MIN_MARGIN_PER_TRADE, min(margin_to_use, config.MAX_MARGIN_PER_TRADE))
        
        notional = margin_to_use * config.LEVERAGE
        quantity = notional / entry_price
        quantity = round(quantity, 6)
        
        logger.info(f"Regime sizing: {vol_regime}, {size_pct*100:.1f}%, margin=${margin_to_use:.2f}, qty={quantity:.6f}")
        return margin_to_use, quantity

    def record_trade_opened(self):
        """Record trade opened"""
        self.open_trades += 1
        logger.debug(f"Trade opened - open: {self.open_trades}")

    def record_trade_closed(self):
        """Record trade closed"""
        self.open_trades = max(0, self.open_trades - 1)
        logger.debug(f"Trade closed - open: {self.open_trades}")

    def update_trade_stats(self, pnl: float):
        """Update statistics"""
        self._reset_daily_stats_if_needed()
        self.daily_trades += 1
        self.total_trades += 1  # ✅ FIXED
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.daily_wins += 1
            self.winning_trades += 1  # ✅ FIXED
            logger.info(f"✓ Win: ${pnl:.2f}")
        else:
            self.daily_losses += 1
            logger.info(f"✗ Loss: ${pnl:.2f}")
        
        logger.info(f"Daily: {self.daily_trades} trades, P&L=${self.daily_pnl:.2f}")
        logger.info(f"Lifetime: {self.total_trades} trades, {self.winning_trades} wins")

    def get_daily_stats(self) -> Dict:
        """Get daily statistics"""
        self._reset_daily_stats_if_needed()
        win_rate = (self.daily_wins / self.daily_trades * 100) if self.daily_trades > 0 else 0
        return {
            "trades": self.daily_trades,
            "pnl": self.daily_pnl,
            "wins": self.daily_wins,
            "losses": self.daily_losses,
            "win_rate": win_rate,
            "open_trades": self.open_trades,
        }

if __name__ == "__main__":
    rm = RiskManager()
    print("✓ RiskManager test")
    balance = rm.get_balance_for_trade_evaluation()
    if balance:
        print(f"✓ Balance: {balance['available']:.2f} USDT")
    allowed, reason = rm.check_trading_allowed(balance_available=balance['available'] if balance else None)
    print(f"Trading: {allowed} - {reason}")
    print(f"Stats: {rm.get_daily_stats()}")
