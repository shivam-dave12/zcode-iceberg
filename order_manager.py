"""
Order Manager - FIXED: Robust order status with rate limiting + partial fill support

Key improvements:
1. Thread-safe rate limiter for order status checks (max 1 call per 2 seconds)
2. Treats UNKNOWN status as "potentially filled" (keeps TP/SL)
3. Handles partial fills as full fills
4. Exponential backoff on API errors
5. Never drops TP/SL unless 100% certain order is cancelled
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime
import threading
from futures_api import FuturesAPI
import config

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL ORDER STATUS RATE LIMITER (2 seconds between ANY status check)
# ============================================================================
_ORDER_STATUS_LOCK = threading.Lock()
_LAST_STATUS_CHECK_TIME = 0.0
_MIN_STATUS_CHECK_INTERVAL = 2.0  # 2 seconds between status checks


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""
    def __init__(self, max_calls: int, period_sec: float):
        self.capacity = float(max_calls)
        self.tokens = float(max_calls)
        self.period_sec = float(period_sec)
        self.refill_rate = float(max_calls) / float(period_sec)
        self.lock = threading.Lock()
        self.last_ts = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_ts
        if elapsed <= 0:
            return
        add = elapsed * self.refill_rate
        if add > 0:
            self.tokens = min(self.capacity, self.tokens + add)
            self.last_ts = now

    def acquire(self) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def available_tokens(self) -> float:
        with self.lock:
            self._refill()
            return self.tokens


class OrderManager:
    """Manages order placement, cancellation, and tracking"""

    def __init__(self):
        """Initialize order manager with new API plugin"""
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )

        # Order tracking
        self.active_orders = {}
        self.order_history = []

        # Rate limiting
        self.last_order_time = 0
        self.order_count = 0
        self.rate_limit_window_start = time.time()

        logger.info("✓ OrderManager initialized with robust status checking")

    # ======================================================================
    # Rate limiting / helpers
    # ======================================================================

    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows new order."""
        current_time = time.time()

        if current_time - self.rate_limit_window_start > 60:
            self.order_count = 0
            self.rate_limit_window_start = current_time

        if self.order_count >= config.RATE_LIMIT_ORDERS:
            return False

        self.order_count += 1
        self.last_order_time = current_time
        return True

    def extract_fill_price(self, order_data: Dict) -> float:
        """Extract a valid fill price from API order dict or raise."""
        price_fields = ["avg_execution_price", "avg_price", "average_price", "price"]
        price = None

        for pf in price_fields:
            if pf in order_data and order_data[pf] is not None:
                try:
                    price = float(order_data[pf])
                    break
                except Exception:
                    continue

        if price is None or price <= 0 or str(price).lower() in ("nan", "inf", "-inf"):
            msg = str(order_data.get("message") or order_data.get("error") or "")
            status_code = order_data.get("status_code")
            if status_code == 429 or "too many requests" in msg.lower():
                raise RuntimeError(
                    "Order fill price not returned due to API rate limit (429 / Too Many Requests)."
                )

            raise RuntimeError(
                f"API did not return a realistic fill price! Full response: {order_data}"
            )

        return price

    def wait_for_fill(self, order_id: str, timeout_sec: float = 30.0) -> Dict:
        """
        Wait for order to fill with timeout.
        IMPROVED: Rate-limited polling with exponential backoff.
        """
        start_time = time.time()
        check_interval = 2.0  # Start with 2 second interval
        
        while time.time() - start_time < timeout_sec:
            try:
                status = self.get_order_status(order_id)
                
                if status:
                    order_status = status.get("status", "").upper()
                    
                    # Treat partial fills as complete
                    if order_status in ("EXECUTED", "FILLED", "PARTIALLY_FILLED", "PARTIALLY_EXECUTED"):
                        logger.info(f"✓ Order filled: {order_id} (status: {order_status})")
                        return status
                    elif order_status in ("CANCELLED", "REJECTED", "EXPIRED"):
                        raise Exception(f"Order {order_id} terminated: {order_status}")
                
                time.sleep(check_interval)
                # Exponential backoff (2s -> 3s -> 4s)
                check_interval = min(4.0, check_interval + 1.0)
                
            except Exception as e:
                logger.debug(f"Error checking order status: {e}")
                time.sleep(check_interval)
        
        # Timeout - get final status
        final_status = self.get_order_status(order_id)
        raise Exception(
            f"Timed out waiting for fill on order {order_id}. "
            f"Last status: {final_status}"
        )

    # ======================================================================
    # ROBUST ORDER STATUS CHECKING
    # ======================================================================

    def get_order_status(self, order_id: str, max_retries: int = 3) -> Optional[Dict]:
        """
        PRODUCTION-GRADE order status checking with:
        - Global rate limiting (2s minimum between calls)
        - Exponential backoff on failures
        - Returns None on persistent failures (caller must handle gracefully)
        
        CRITICAL: Never fails fast - always exhausts retries before giving up
        """
        global _LAST_STATUS_CHECK_TIME, _ORDER_STATUS_LOCK, _MIN_STATUS_CHECK_INTERVAL
        
        last_response = None
        
        for attempt in range(1, max_retries + 1):
            # ========================================
            # GLOBAL RATE LIMIT ENFORCEMENT
            # ========================================
            with _ORDER_STATUS_LOCK:
                now = time.time()
                elapsed = now - _LAST_STATUS_CHECK_TIME
                
                if elapsed < _MIN_STATUS_CHECK_INTERVAL:
                    wait_time = _MIN_STATUS_CHECK_INTERVAL - elapsed
                    time.sleep(wait_time)
                
                _LAST_STATUS_CHECK_TIME = time.time()
            
            # ========================================
            # ACTUAL API CALL WITH ERROR HANDLING
            # ========================================
            try:
                response = self.api.get_order_status(order_id)
                last_response = response
                
                # Successful response structure check
                if isinstance(response, dict):
                    # CoinSwitch v2: data.order.status OR direct status
                    if "data" in response:
                        order_data = response.get("data", {})
                        if isinstance(order_data, dict) and "order" in order_data:
                            return response  # Valid structure
                        elif "status" in order_data:
                            return response  # Alternative structure
                    elif "order" in response:
                        return response  # Direct order object
                    elif "status" in response:
                        return response  # Direct status
                    
                    # Check for error indicators
                    if "error" in response or response.get("status_code") == 429:
                        logger.warning(f"API error in status check: {response.get('error') or 'Rate limited'}")
                        time.sleep(1.0 * attempt)  # Exponential backoff
                        continue
                
                # Unexpected response type
                logger.warning(f"Unexpected response type: {type(response)}")
                time.sleep(1.0 * attempt)
                continue
                
            except Exception as exc:
                logger.warning(f"[STATUS CHECK #{attempt}] Exception: {exc}")
                time.sleep(1.0 * attempt)  # Exponential backoff
                continue
        
        # ========================================
        # EXHAUSTED ALL RETRIES
        # ========================================
        logger.error(
            f"❌ get_order_status FAILED after {max_retries} retries: {order_id}\n"
            f"Last response: {last_response}"
        )
        return None  # Caller MUST handle None gracefully

    def get_order_status_safe(self, order_id: str) -> str:
        """
        SAFE wrapper that NEVER raises exceptions.
        Returns normalized status string: FILLED, CANCELLED, UNKNOWN
        
        CRITICAL: Treats UNKNOWN (API failure) as "potentially filled" to prevent
        accidentally cancelling TP/SL on an active position.
        """
        try:
            response = self.get_order_status(order_id)
            
            if response is None:
                logger.warning(f"⚠️ Cannot determine status for {order_id} - treating as UNKNOWN (potentially filled)")
                return "UNKNOWN"
            
            # Extract status from various response structures
            order_data = response
            if "data" in response:
                order_data = response["data"]
                if isinstance(order_data, dict) and "order" in order_data:
                    order_data = order_data["order"]
            elif "order" in response:
                order_data = response["order"]
            
            status = str(order_data.get("status", "UNKNOWN")).upper()
            
            # Normalize status
            if status in ("EXECUTED", "FILLED", "PARTIALLY_FILLED", "PARTIALLY_EXECUTED"):
                return "FILLED"
            elif status in ("CANCELLED", "REJECTED", "EXPIRED"):
                return "CANCELLED"
            else:
                return "UNKNOWN"
                
        except Exception as e:
            logger.error(f"Exception in get_order_status_safe: {e}")
            return "UNKNOWN"

    # ======================================================================
    # Order placement
    # ======================================================================

    def place_market_order(
        self, side: str, quantity: float, reduce_only: bool = False
    ) -> Optional[Dict]:
        """Place a market order."""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded for orders; delaying by 2 seconds")
                time.sleep(2)

            logger.info(f"Placing MARKET {side} order: {quantity} {config.SYMBOL}")

            response = self.api.place_order(
                symbol=config.SYMBOL,
                side=side.upper(),
                order_type="MARKET",
                quantity=quantity,
                exchange=config.EXCHANGE,
                reduce_only=reduce_only,
            )

            if "data" in response and "order_id" in response["data"]:
                order_id = response["data"]["order_id"]
                order_details = response["data"]

                self.active_orders[order_id] = {
                    "order_id": order_id,
                    "symbol": config.SYMBOL,
                    "side": side,
                    "type": "MARKET",
                    "quantity": quantity,
                    "status": order_details.get("status", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                    "reduce_only": reduce_only,
                }
                self.order_history.append(self.active_orders[order_id].copy())

                logger.info(f"✓ Order placed successfully: {order_id}")
                logger.info(f"  Status: {order_details.get('status')}")
                return order_details
            else:
                error_msg = response.get("response", {}).get("message", "Unknown error")
                logger.error(f"✗ Order placement failed: {error_msg}")
                logger.error(f"  Full response: {response}")
                return None

        except Exception as e:
            logger.error(f"Error placing market order: {e}", exc_info=True)
            return None

    def place_limit_order(
        self,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Place a limit order."""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded for orders; delaying by 2 seconds")
                time.sleep(2)

            logger.info(
                f"Placing LIMIT {side} order: {quantity} {config.SYMBOL} @ ${price:,.2f}"
            )

            response = self.api.place_order(
                symbol=config.SYMBOL,
                side=side.upper(),
                order_type="LIMIT",
                quantity=quantity,
                price=price,
                exchange=config.EXCHANGE,
                reduce_only=reduce_only,
            )

            if "data" in response and "order_id" in response["data"]:
                order_id = response["data"]["order_id"]
                order_details = response["data"]

                self.active_orders[order_id] = {
                    "order_id": order_id,
                    "symbol": config.SYMBOL,
                    "side": side,
                    "type": "LIMIT",
                    "quantity": quantity,
                    "price": price,
                    "status": order_details.get("status", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                    "reduce_only": reduce_only,
                }
                self.order_history.append(self.active_orders[order_id].copy())

                logger.info(f"✓ Limit order placed: {order_id}")
                return order_details
            else:
                error_msg = response.get("response", {}).get("message", "Unknown error")
                logger.error(f"✗ Limit order failed: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error placing limit order: {e}", exc_info=True)
            return None

    def place_stop_loss(
        self, side: str, quantity: float, trigger_price: float
    ) -> Optional[Dict]:
        """Place a stop loss order."""
        try:
            logger.info(f"Placing STOP LOSS {side} @ ${trigger_price:,.2f}")

            response = self.api.place_order(
                symbol=config.SYMBOL,
                side=side.upper(),
                order_type="STOP_MARKET",
                quantity=0,
                trigger_price=trigger_price,
                exchange=config.EXCHANGE,
                reduce_only=True,
            )

            if "data" in response and "order_id" in response["data"]:
                order_id = response["data"]["order_id"]
                logger.info(f"✓ Stop loss order placed: {order_id}")

                self.active_orders[order_id] = {
                    "order_id": order_id,
                    "symbol": config.SYMBOL,
                    "side": side,
                    "type": "STOP_LOSS",
                    "quantity": 0,
                    "trigger_price": trigger_price,
                    "status": response["data"].get("status", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                }
                return response["data"]
            else:
                logger.error(f"✗ Stop loss order failed: {response}")
                return None

        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")
            return None

    def place_take_profit(
        self, side: str, quantity: float, trigger_price: float
    ) -> Optional[Dict]:
        """Place a take profit order."""
        try:
            logger.info(f"Placing TAKE PROFIT {side} @ ${trigger_price:,.2f}")

            response = self.api.place_order(
                symbol=config.SYMBOL,
                side=side.upper(),
                order_type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                trigger_price=trigger_price,
                exchange=config.EXCHANGE,
                reduce_only=True,
            )

            if "data" in response and "order_id" in response["data"]:
                order_id = response["data"]["order_id"]
                logger.info(f"✓ Take profit order placed: {order_id}")

                self.active_orders[order_id] = {
                    "order_id": order_id,
                    "symbol": config.SYMBOL,
                    "side": side,
                    "type": "TAKE_PROFIT",
                    "quantity": quantity,
                    "trigger_price": trigger_price,
                    "status": response["data"].get("status", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                }
                return response["data"]
            else:
                logger.error(f"✗ Take profit order failed: {response}")
                return None

        except Exception as e:
            logger.error(f"Error placing take profit: {e}")
            return None

    # ======================================================================
    # Cancellation
    # ======================================================================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            logger.info(f"Cancelling order: {order_id}")
            response = self.api.cancel_order(
                order_id=order_id,
                exchange=config.EXCHANGE,
            )

            if "data" in response:
                logger.info(f"✓ Order cancelled: {order_id}")
                if order_id in self.active_orders:
                    self.active_orders[order_id]["status"] = "CANCELLED"
                return True
            else:
                logger.error(f"✗ Cancel failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders for the symbol."""
        try:
            logger.info(f"Cancelling all orders for {config.SYMBOL}")
            response = self.api.cancel_all_orders(
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )

            if "data" in response or not response.get("error"):
                logger.info("✓ All orders cancelled")
                self.active_orders.clear()
                return True
            else:
                logger.error(f"✗ Cancel all failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def get_open_orders(self) -> list:
        """Get all open orders."""
        try:
            response = self.api.get_open_orders(
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )

            if "data" in response:
                orders = response["data"].get("orders", [])
                logger.debug(f"Found {len(orders)} open orders")
                return orders
            else:
                logger.warning("Could not fetch open orders")
                return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    # ======================================================================
    # Stats
    # ======================================================================

    def get_order_statistics(self) -> Dict:
        """Get order statistics."""
        total_orders = len(self.order_history)
        if total_orders == 0:
            return {
                "total_orders": 0,
                "active_orders": 0,
                "success_rate": 0,
            }

        successful = sum(
            1
            for order in self.order_history
            if order.get("status") in ["EXECUTED", "FILLED"]
        )

        return {
            "total_orders": total_orders,
            "active_orders": len(self.active_orders),
            "successful_orders": successful,
            "success_rate": (successful / total_orders) * 100 if total_orders > 0 else 0,
            "last_order_time": self.last_order_time,
        }


if __name__ == "__main__":
    om = OrderManager()
    print("Order Manager initialized")
    print(f"Statistics: {om.get_order_statistics()}")
    open_orders = om.get_open_orders()
    print(f"Open orders: {len(open_orders)}")