"""
Risk Manager - Manages position sizing and risk controls

Updated to use new CoinSwitch API plugins
+ Balance caching and 429-safe behaviour for Z-Score Imbalance Iceberg Hunter
+ Volatility regime-aware Kelly sizing
+ Dynamic TP/SL calculation from ROI inputs
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
        self.realized_pnl = 0.0  # Total realized P&L across all trades
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
        Fetch available USDT balance with caching to avoid rate limits.

        Returns dict with keys: total, available, blocked, currency.
        """
        now = time.time()

        # Use cached balance if recent
        if (
            self._balance_cache is not None
            and (now - self._last_balance_fetch) < self._balance_cache_ttl_sec
        ):
            return self._balance_cache

        backoff = 0.5
        for attempt in range(4):
            try:
                response = self.api.get_wallet_balance()

                # Some _make_request wrappers return error dicts with status_code
                if isinstance(response, dict) and response.get("status_code") == 429:
                    if now - self._last_429_log > 5.0:
                        logger.error(
                            "Failed to get balance: 429 Too Many Requests. "
                            "Using cached balance if available."
                        )
                        self._last_429_log = now
                    time.sleep(backoff)
                    backoff *= 2
                    continue

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
                            "available": float(
                                balances.get("total_available_balance", 0)
                            ),
                            "blocked": float(balances.get("total_blocked_balance", 0)),
                            "currency": "USDT",
                        }
                        self._balance_cache = result
                        self._last_balance_fetch = now
                        return result
                    else:
                        logger.warning("USDT balance not found in wallet response")
                        return self._balance_cache

                logger.error(f"Failed to get balance: {response}")
                return self._balance_cache

            except Exception as e:
                logger.error(
                    f"Error getting balance (attempt {attempt+1}): {e}", exc_info=True
                )
                time.sleep(backoff)
                backoff *= 2

        # After retries, return cached or None
        return self._balance_cache

    # ======================================================================
    # NEW: Volatility regime-aware Kelly sizing + TP/SL calculation
    # ======================================================================

    def calculate_margin_for_entry(
        self,
        current_price: float,
        side: str,
        desired_roi_tp: float,
        desired_roi_sl: float,
        regime: str,
    ) -> Optional[Dict]:
        """
        Calculate position sizing and TP/SL prices using:

        1. Kelly sizing: kelly_raw = 1 / (1 + atr_pct_5m), bounded into [0, 1].
        2. Regime-specific caps: VOL_POSITION_SIZE_CAP[regime].
        3. Margin = available_balance * capped_fraction.
        4. Quantity = (margin * leverage) / entry_price.
        5. TP/SL prices based on desired ROI inputs.

        Args:
            current_price: Entry price.
            side: "long" or "short".
            desired_roi_tp: Desired ROI for take-profit (e.g. 0.10 = 10%).
            desired_roi_sl: Desired ROI for stop-loss (e.g. -0.03 = -3%).
            regime: Volatility regime ("LOW", "NEUTRAL", "HIGH").

        Returns:
            Dict with keys: quantity, tp_price, sl_price, margin_used, kelly_fraction.
            None if insufficient balance or computation error.
        """
        try:
            balance_info = self.get_available_balance()
            if not balance_info or balance_info["available"] <= 0:
                logger.warning("Insufficient balance for entry sizing")
                return None

            available = float(balance_info["available"])

            # Kelly sizing overlay
            # For simplicity, we assume atr_pct_5m ~ atr_pct_10m cached in data_manager.
            # Alternatively, pass atr_pct explicitly. Here we use a conservative default
            # if not available.
            # kelly_raw = 1 / (1 + atr_pct_5m)
            # For production, this should be passed or fetched. For now, assume 0.5% default.
            atr_pct_5m = 0.005  # placeholder; ideally fetched from data_manager

            kelly_raw = 1.0 / (1.0 + atr_pct_5m) if atr_pct_5m > 0 else 0.5
            kelly_raw = max(0.0, min(1.0, kelly_raw))

            # Map into regime-specific caps
            vol_caps = getattr(config, "VOL_POSITION_SIZE_CAP", {})
            regime_cap = float(vol_caps.get(regime, 0.18))  # default 18% if missing

            # Final sizing fraction
            sizing_fraction = min(kelly_raw, regime_cap)

            # Margin to use for this trade
            margin = available * sizing_fraction

            # Respect min/max bounds from config
            margin = max(float(config.MIN_MARGIN_PER_TRADE), margin)
            margin = min(float(config.MAX_MARGIN_PER_TRADE), margin)

            # Quantity (BTC) = (margin * leverage) / entry_price
            leverage = float(config.LEVERAGE)
            quantity = (margin * leverage) / current_price

            # Round to 6 decimals (CoinSwitch standard)
            quantity = round(quantity, 6)

            if quantity <= 0.0:
                logger.warning("Computed quantity <= 0 after rounding")
                return None

            # TP and SL prices
            # ROI is defined on margin:
            # profit_usdt = margin * roi_tp
            # For a long: profit_usdt = (tp_price - entry_price) * qty
            # => tp_price = entry_price + (margin * roi_tp) / qty
            # For a short: profit_usdt = (entry_price - tp_price) * qty
            # => tp_price = entry_price - (margin * roi_tp) / qty

            direction = 1.0 if side == "long" else -1.0

            tp_price = current_price + direction * (margin * desired_roi_tp) / quantity
            sl_price = current_price + direction * (margin * desired_roi_sl) / quantity

            # Round to 2 decimals (standard for BTC price)
            tp_price = round(tp_price, 2)
            sl_price = round(sl_price, 2)

            # Sanity checks
            if side == "long":
                if tp_price <= current_price or sl_price >= current_price:
                    logger.error(
                        f"Invalid TP/SL for long: entry={current_price}, "
                        f"tp={tp_price}, sl={sl_price}"
                    )
                    return None
            else:
                if tp_price >= current_price or sl_price <= current_price:
                    logger.error(
                        f"Invalid TP/SL for short: entry={current_price}, "
                        f"tp={tp_price}, sl={sl_price}"
                    )
                    return None

            result = {
                "quantity": quantity,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "margin_used": margin,
                "kelly_fraction": sizing_fraction,
            }

            logger.info(
                f"Margin calc: regime={regime} kelly_raw={kelly_raw:.4f} "
                f"cap={regime_cap:.2f} fraction={sizing_fraction:.4f} "
                f"margin={margin:.2f} qty={quantity:.6f} tp={tp_price:.2f} sl={sl_price:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in calculate_margin_for_entry: {e}", exc_info=True)
            return None

    # ======================================================================
    # Trade lifecycle
    # ======================================================================

    def update_after_trade_close(
        self,
        position,
        exit_price: float,
        exit_reason: str,
    ) -> float:
        """
        Update risk manager state after a position closes.

        Args:
            position: ZScorePosition dataclass.
            exit_price: Exit price.
            exit_reason: Reason string (TP, SL, TIME_STOP, etc.).

        Returns:
            Realized P&L (USDT).
        """
        try:
            direction = 1.0 if position.side == "long" else -1.0
            pnl_raw = (exit_price - position.entry_price) * direction * position.quantity

            # Subtract fees (approx taker fee on entry + exit)
            fee_rate = float(getattr(config, "TAKER_FEE_RATE", 0.00065))
            entry_value = position.entry_price * position.quantity
            exit_value = exit_price * position.quantity
            total_fees = (entry_value + exit_value) * fee_rate

            pnl = pnl_raw - total_fees

            # Update stats
            self.update_trade_stats(pnl)
            self.realized_pnl += pnl

            logger.info(
                f"Trade closed: {position.side.upper()} {position.quantity:.6f} BTC | "
                f"Entry={position.entry_price:.2f} Exit={exit_price:.2f} | "
                f"PnL={pnl:.2f} USDT (raw={pnl_raw:.2f}, fees={total_fees:.2f}) | "
                f"Reason={exit_reason}"
            )

            return pnl

        except Exception as e:
            logger.error(f"Error updating after trade close: {e}", exc_info=True)
            return 0.0

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
        """Reset daily trade count and P&L at midnight UTC."""
        now = datetime.utcnow()
        if now.date() > self.last_reset.date():
            logger.info("Resetting daily counters")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = now
