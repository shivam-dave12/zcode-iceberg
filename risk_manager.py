"""
Risk Manager - Position sizing and risk controls
With balance caching to reduce API spam
"""

import time
import logging
from typing import Optional, Dict, Tuple
import config
from futures_api import FuturesAPI

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management with balance caching"""

    def __init__(self) -> None:
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.realized_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_daily_reset: float = time.time()
        
        # Balance caching (reduce API spam)
        self._cached_balance: Optional[Dict] = None
        self._balance_cache_time: float = 0.0
        self._balance_cache_ttl: float = config.BALANCE_CACHE_TTL_SEC
        
        logger.info("RiskManager initialized with balance caching")

    def get_available_balance(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get available balance with caching.
        Only fetches from API if cache expired or force_refresh=True.
        """
        now = time.time()
        
        # Return cached balance if still valid
        if not force_refresh and self._cached_balance is not None:
            if now - self._balance_cache_time < self._balance_cache_ttl:
                return self._cached_balance
        
        # Fetch fresh balance from API
        try:
            balance = self.api.get_balance(currency="USDT")
            if balance and "available" in balance:
                self._cached_balance = balance
                self._balance_cache_time = now
                logger.debug(f"Balance fetched: {balance['available']:.2f} USDT (cached for {self._balance_cache_ttl}s)")
                return balance
            else:
                logger.error("Invalid balance response from API")
                return self._cached_balance  # Return stale cache if API fails
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self._cached_balance  # Return stale cache on error

    def calculate_position_size_vol_regime(
        self, entry_price: float, regime: str, current_balance: float = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on vol regime.
        Only fetches balance if not provided.
        """
        if current_balance is None:
            balance_info = self.get_available_balance()
            if not balance_info:
                logger.error("Could not fetch balance for vol-regime sizing")
                return 0.001, 0.0
            current_balance = balance_info["available"]
        
        if regime == "HIGH":
            size_pct = config.VOL_REGIME_HIGH_SIZE_PCT / 100.0
        else:
            size_pct = config.VOL_REGIME_LOW_SIZE_PCT / 100.0
        
        margin_used = current_balance * size_pct
        margin_used = max(config.MIN_MARGIN_PER_TRADE, min(margin_used, config.MAX_MARGIN_PER_TRADE))
        
        quantity = (margin_used * config.LEVERAGE) / entry_price
        quantity = max(0.001, round(quantity, 6))
        
        logger.info(f"[VOL-SIZING] regime={regime}, size={size_pct*100:.1f}%, margin={margin_used:.2f}, qty={quantity:.6f}")
        return quantity, margin_used

    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk rules."""
        self._maybe_reset_daily_counters()
        
        if not config.ENABLE_TRADING:
            return False, "TRADING_DISABLED"
        
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"MAX_DAILY_TRADES_REACHED ({self.daily_trades}/{config.MAX_DAILY_TRADES})"
        
        if self.daily_pnl <= -config.MAX_DAILY_LOSS:
            return False, f"MAX_DAILY_LOSS_REACHED ({self.daily_pnl:.2f}/-{config.MAX_DAILY_LOSS})"
        
        return True, "OK"

    def record_trade_opened(self) -> None:
        """Record trade opening."""
        self.daily_trades += 1
        self.total_trades += 1
        # Invalidate balance cache when trade is opened
        self._cached_balance = None

    def update_trade_stats(self, pnl: float) -> None:
        """Update trade statistics."""
        self.realized_pnl += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        
        logger.info(f"[TRADE STATS] Total={self.total_trades}, WR={win_rate:.1f}%, PnL={self.realized_pnl:.2f}")
        
        # Invalidate balance cache after trade closes
        self._cached_balance = None

    def _maybe_reset_daily_counters(self) -> None:
        """Reset daily counters at midnight UTC."""
        now = time.time()
        if now - self.last_daily_reset >= 86400:
            logger.info("Resetting daily counters")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_daily_reset = now
