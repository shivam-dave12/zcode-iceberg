"""
Risk Manager - Manages position sizing and risk controls

Updated to use new CoinSwitch API plugins

+ Balance caching and 429-safe behaviour for Z-Score Imbalance Iceberg Hunter
"""

import logging
import time
from typing import Dict, Optional
from datetime import datetime

from futures_api import FuturesAPI
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk controls and position sizing"""

    def __init__(self):
        """Initialize risk manager with new API plugin"""

        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )

        # Risk tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.utcnow()

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Balance cache (to avoid hammering wallet_balance endpoint)
        self._balance_cache: Optional[Dict] = None
        self._last_balance_fetch: float = 0.0
        self._balance_cache_ttl_sec: float = 5.0
        self._last_429_log: float = 0.0

        logger.info("✓ RiskManager initialized")

    # ======================================================================
    # Trading permissions
    # ======================================================================

    def check_trading_allowed(self) -> tuple:
        """
        Check if trading is allowed based on risk limits.

        Returns:
            (allowed: bool, reason: str)
        """
        try:
            self._reset_daily_counters()

            if not config.ENABLE_TRADING:
                return False, "Trading is disabled in config"

            if self.daily_trades >= config.MAX_DAILY_TRADES:
                return False, f"Daily trade limit reached ({config.MAX_DAILY_TRADES})"

            if self.daily_pnl <= -config.MAX_DAILY_LOSS:
                return False, f"Daily loss limit reached (${abs(self.daily_pnl):.2f})"

            balance_info = self.get_available_balance()
            if not balance_info:
                return False, "No balance information available (API / rate-limit issue)"

            if balance_info["available"] <= 0:
                return False, "Insufficient balance"

            if balance_info["available"] < config.MIN_MARGIN_PER_TRADE:
                return False, (
                    f"Available balance {balance_info['available']:.2f} "
                    f"< MIN_MARGIN_PER_TRADE ({config.MIN_MARGIN_PER_TRADE})"
                )

            return True, "Trading allowed"

        except Exception as e:
            logger.error(f"Error checking trading permissions: {e}", exc_info=True)
            return False, f"Error: {e}"

    # ======================================================================
    # Balance management
    # ======================================================================

    def get_available_balance(self) -> Optional[Dict]:
        """
        Get available USDT futures wallet balance via new API.

        - Uses a short cache (5 seconds) to respect rate limits.
        - On 429 Too Many Requests, keeps and returns last cached value.
        """
        now = time.time()

        if (
            self._balance_cache is not None
            and (now - self._last_balance_fetch) < self._balance_cache_ttl_sec
        ):
            return self._balance_cache

        try:
            response = self.api.get_wallet_balance()

            if isinstance(response, dict) and response.get("status_code") == 429:
                if now - self._last_429_log > 5.0:
                    logger.error(
                        "Failed to get balance: 429 Too Many Requests. "
                        "Using cached balance if available."
                    )
                    self._last_429_log = now
                return self._balance_cache

            if "data" in response:
                data = response["data"]
                base_balances = data.get("base_asset_balances", [])
                usdt_balance = next(
                    (b for b in base_balances if b.get("base_asset") == "USDT"),
                    None,
                )

                if usdt_balance:
                    balances = usdt_balance.get("balances", {})
                    result = {
                        "total": float(balances.get("total_balance", 0)),
                        "available": float(balances.get("total_available_balance", 0)),
                        "blocked": float(balances.get("total_blocked_balance", 0)),
                        "currency": "USDT",
                    }

                    self._balance_cache = result
                    self._last_balance_fetch = now

                    logger.debug(
                        "Wallet balance fetched: "
                        f"total={result['total']:.4f}, "
                        f"available={result['available']:.4f}, "
                        f"blocked={result['blocked']:.4f}"
                    )
                    return result
                else:
                    logger.warning("USDT balance not found in wallet response")
                    return self._balance_cache

            logger.error(f"Failed to get balance: {response}")
            return self._balance_cache

        except Exception as e:
            logger.error(f"Error getting balance: {e}", exc_info=True)
            return self._balance_cache

    # ======================================================================
    # Legacy HF position sizing (kept for compatibility)
    # ======================================================================

    def calculate_position_size(
        self, entry_price: float, stop_loss_price: float, current_balance: float = None
    ) -> float:
        """
        Legacy HF method (kept for compatibility), not used by Z-Score strategy.
        """
        try:
            if current_balance is None:
                balance_info = self.get_available_balance()
                if not balance_info:
                    logger.error("Could not fetch balance for HF position sizing")
                    return 0.001
                current_balance = balance_info["available"]

            risk_amount = min(20, current_balance * 0.02)
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff == 0:
                logger.warning("Stop loss same as entry price in HF calc")
                return 0.001

            position_size = risk_amount / price_diff
            position_size = max(0.001, position_size)
            position_size = min(2.0, position_size)

            position_value = position_size * entry_price
            if position_value > 50000:
                position_size = 50000 / entry_price

            logger.info(f"HF calc position size: {position_size:.6f} BTC")
            return round(position_size, 6)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0.001

    # ======================================================================
    # Trade statistics
    # ======================================================================

    def update_trade_stats(self, pnl: float):
        """Update trade statistics."""
        self.daily_trades += 1
        self.total_trades += 1
        self.daily_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
            logger.info(f"✓ Winning trade: ${pnl:.2f}")
        else:
            self.losing_trades += 1
            logger.info(f"✗ Losing trade: ${pnl:.2f}")

        logger.info(
            f"Daily stats: trades={self.daily_trades}, "
            f"P&L={self.daily_pnl:.2f}, "
            f"wins={self.winning_trades}, losses={self.losing_trades}"
        )

    def record_trade_opened(self):
        """Record that a new trade has been opened."""
        logger.debug(
            f"Trade opened. Daily trade count (including this open) will be "
            f"{self.daily_trades + 1}"
        )

    # ======================================================================
    # Daily reset
    # ======================================================================

    def _reset_daily_counters(self):
        now = datetime.utcnow()
        if now.date() > self.last_reset.date():
            logger.info("Resetting daily counters")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = now
