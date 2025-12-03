"""
Risk Manager - Handles position sizing, risk limits, and trade statistics
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
        
        # Trade tracking
        self.open_trades = 0
        self.total_margin_used = 0.0
        
        # BALANCE CACHE - 3 second TTL
        self._last_balance_time = 0
        self._cached_balance = None
        
        logger.info("✓ RiskManager initialized")
    
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
    
    def get_available_balance(self) -> Optional[Dict]:
        """
        Get available balance with CORRECT CoinSwitch API parsing (1 call per 3 seconds MAX)
        """
        now = time.time()
        
        # GLOBAL RATE LIMIT - 1 call per 3 seconds MAX
        if hasattr(self, '_last_balance_time') and now - self._last_balance_time < 3.0:
            logger.debug("Balance rate limited - using cache")
            return getattr(self, '_cached_balance', None)
        
        self._last_balance_time = now
        
        try:
            logger.debug("Fetching fresh balance...")
            response = self.api.get_wallet_balance()
            
            logger.debug(f"Raw wallet response: {response}")  # DEBUG
            
            if "data" in response and "base_asset_balances" in response["data"]:
                balances = response["data"]["base_asset_balances"]
                for asset in balances:
                    if asset.get("base_asset") == "USDT":
                        balances_dict = asset.get("balances", {})
                        available_usdt = float(balances_dict.get("total_available_balance", 0.0))
                        
                        # Cache result
                        self._cached_balance = {
                            "available": available_usdt,
                            "total": float(balances_dict.get("total_balance", 0.0)),
                            "used": float(balances_dict.get("total_blocked_balance", 0.0)),
                            "timestamp": now
                        }
                        
                        logger.info(f"✓ Fresh balance: {available_usdt:.2f} USDT available")  # Changed to INFO
                        return self._cached_balance
            
            logger.warning(f"Balance API no USDT data: {response}")
            return getattr(self, '_cached_balance', None)
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return getattr(self, '_cached_balance', None)

    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits"""
        self._reset_daily_stats_if_needed()
        
        # Check daily trade limit
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached ({self.daily_trades}/{config.MAX_DAILY_TRADES})"
        
        # Check daily loss limit
        if self.daily_pnl < -config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f}/-${config.MAX_DAILY_LOSS})"
        
        # Check concurrent position limit
        if self.open_trades >= config.MAX_CONCURRENT_POSITIONS:
            return False, f"Max concurrent positions reached ({self.open_trades}/{config.MAX_CONCURRENT_POSITIONS})"
        
        # Check available balance
        balance_info = self.get_available_balance()
        if not balance_info:
            return False, "Cannot fetch balance"
        
        available = float(balance_info.get("available", 0.0))
        if available < config.MIN_MARGIN_PER_TRADE:
            return False, f"Insufficient balance (${available:.2f} < ${config.MIN_MARGIN_PER_TRADE})"
        
        return True, "OK"
    
    def calculate_position_size_regime_aware(
        self,
        entry_price: float,
        vol_regime: str,
    ) -> Tuple[float, float]:
        """
        Calculate position size based on vol regime (NO balance_available param)
        
        Returns:
            (margin_to_use, quantity_in_btc)
        """
        balance_info = self.get_available_balance()
        if not balance_info:
            logger.error("Cannot calculate position size without balance")
            return 0.0, 0.0
        
        available = float(balance_info.get("available", 0.0))
        
        # Vol-regime sizing
        if vol_regime == "HIGH":
            size_pct = config.VOL_REGIME_SIZE_HIGH_PCT
        elif vol_regime == "LOW":
            size_pct = config.VOL_REGIME_SIZE_LOW_PCT
        else:
            size_pct = (config.VOL_REGIME_SIZE_HIGH_PCT + config.VOL_REGIME_SIZE_LOW_PCT) / 2.0
        
        margin_to_use = available * size_pct
        margin_to_use = max(config.MIN_MARGIN_PER_TRADE, min(margin_to_use, config.MAX_MARGIN_PER_TRADE))
        
        # Calculate quantity
        notional = margin_to_use * config.LEVERAGE
        quantity = notional / entry_price
        quantity = round(quantity, 6)
        
        logger.info(f"Regime-aware sizing: vol_regime={vol_regime}, usage={size_pct*100:.1f}%, margin={margin_to_use:.2f}, qty={quantity:.6f}")
        
        return margin_to_use, quantity
    
    def record_trade_opened(self):
        """Record that a trade was opened"""
        self.open_trades += 1
        logger.debug(f"Trade opened - open count: {self.open_trades}")
    
    def record_trade_closed(self):
        """Record that a trade was closed"""
        self.open_trades = max(0, self.open_trades - 1)
        logger.debug(f"Trade closed - open count: {self.open_trades}")
    
    def update_trade_stats(self, pnl: float):
        """Update daily trade statistics"""
        self._reset_daily_stats_if_needed()
        
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.daily_wins += 1
            logger.info(f"✓ Winning trade: ${pnl:.2f}")
        else:
            self.daily_losses += 1
            logger.info(f"✗ Losing trade: ${pnl:.2f}")
        
        logger.info(f"Daily stats: trades={self.daily_trades}, P&L={self.daily_pnl:.2f}, wins={self.daily_wins}, losses={self.daily_losses}")
    
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
    print("✓ RiskManager initialized")
    
    # Test balance
    balance = rm.get_available_balance()
    if balance:
        print(f"✓ Balance: {balance['available']:.2f} USDT available")
    
    # Test trading allowed
    allowed, reason = rm.check_trading_allowed()
    print(f"Trading allowed: {allowed} - {reason}")
    
    # Test stats
    stats = rm.get_daily_stats()
    print(f"Daily stats: {stats}")
