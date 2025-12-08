"""
Order Manager - FIXED: Aggressive rate limit protection
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime
from futures_api import FuturesAPI
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class OrderManager:
    """Manages order placement, cancellation, and tracking"""
    
    # CRITICAL: Rate limit protection
    _GLOBAL_API_LOCK = {}
    _MIN_REQUEST_INTERVAL = 2.0  # 2 seconds between ANY API calls
    _LAST_REQUEST_TIME = 0.0
    
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
        
        logger.info("✓ OrderManager initialized with aggressive rate limiting")
    
    @classmethod
    def _wait_for_rate_limit(cls):
        """GLOBAL rate limiter - ensures 2s between ANY API call"""
        now = time.time()
        elapsed = now - cls._LAST_REQUEST_TIME
        
        if elapsed < cls._MIN_REQUEST_INTERVAL:
            wait_time = cls._MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"[RATE LIMIT] Waiting {wait_time:.2f}s before API call")
            time.sleep(wait_time)
        
        cls._LAST_REQUEST_TIME = time.time()
    
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
    
    def get_order_status(self, order_id: str, retry_count: int = 3) -> Optional[Dict]:
        """
        Get order status with AGGRESSIVE rate limiting and retries.
        Returns None on failure after all retries.
        """
        for attempt in range(retry_count):
            try:
                # CRITICAL: Wait before EVERY API call
                self._wait_for_rate_limit()
                
                response = self.api.get_order(order_id)
                
                if "data" in response:
                    order_data = response["data"].get("order", response["data"])
                    if order_id in self.active_orders:
                        self.active_orders[order_id]["status"] = order_data.get(
                            "status", "UNKNOWN"
                        )
                    return order_data
                else:
                    # Check if 429 error
                    if response.get("status_code") == 429:
                        wait_time = 3.0 * (attempt + 1)  # Exponential backoff
                        logger.warning(f"[429] Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{retry_count})")
                        time.sleep(wait_time)
                        continue
                    
                    logger.warning(f"Could not get order status for {order_id} (attempt {attempt + 1}/{retry_count})")
                    return None
            
            except Exception as e:
                logger.error(f"Error getting order status (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None
        
        logger.error(f"Failed to get order status for {order_id} after {retry_count} attempts")
        return None
    
    def get_order_status_safe(self, order_id: str) -> str:
        """
        Safe wrapper for get_order_status that never raises exceptions.
        Returns: "FILLED", "CANCELLED", or "UNKNOWN"
        
        CRITICAL: Uses aggressive rate limiting
        """
        try:
            response = self.get_order_status(order_id, retry_count=2)  # Only 2 retries for safety checks
            if response is None:
                return "UNKNOWN"
            
            status = str(response.get("status", "UNKNOWN")).upper()
            
            if status in ("EXECUTED", "FILLED", "PARTIALLY_FILLED", "PARTIALLY_EXECUTED"):
                return "FILLED"
            elif status in ("CANCELLED", "REJECTED", "EXPIRED"):
                return "CANCELLED"
            else:
                return "UNKNOWN"
        except Exception as e:
            logger.error(f"Exception in get_order_status_safe: {e}")
            return "UNKNOWN"
    
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
    
    def wait_for_fill(
        self,
        order_id: str,
        timeout_sec: float = 3.0,
        poll_interval_sec: float = 2.0,  # INCREASED from 0.1s to 2s
    ) -> Dict:
        """
        Poll order status until a real fill is present or timeout.
        CRITICAL: Uses 2s polling interval to avoid rate limits.
        """
        start = time.time()
        last_data: Optional[Dict] = None
        
        while True:
            now = time.time()
            if now - start > timeout_sec:
                raise RuntimeError(
                    f"Timed out waiting for fill on order {order_id}. Last status: {last_data}"
                )
            
            status_resp = self.get_order_status(order_id, retry_count=1)  # Only 1 retry in polling
            if not status_resp:
                time.sleep(poll_interval_sec)
                continue
            
            last_data = status_resp
            status = str(status_resp.get("status", "")).upper()
            exec_qty_str = status_resp.get("exec_quantity") or status_resp.get(
                "executed_qty"
            )
            
            try:
                exec_qty = float(exec_qty_str) if exec_qty_str is not None else 0.0
            except Exception:
                exec_qty = 0.0
            
            if status in ("CANCELLED", "REJECTED"):
                raise RuntimeError(
                    f"Order {order_id} not filled. Status={status}, data={status_resp}"
                )
            
            if exec_qty > 0:
                _ = self.extract_fill_price(status_resp)
                return status_resp
            
            time.sleep(poll_interval_sec)
    
    # ======================================================================
    # Order placement (unchanged but with rate limiting)
    # ======================================================================
    
    def place_market_order(
        self, side: str, quantity: float, reduce_only: bool = False
    ) -> Optional[Dict]:
        """Place a market order."""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded for orders; delaying by 2 seconds")
                time.sleep(2)
            
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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
                return order_details
            else:
                error_msg = response.get("response", {}).get("message", "Unknown error")
                logger.error(f"✗ Order placement failed: {error_msg}")
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
            
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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
            # CRITICAL: Wait before API call...
            self._wait_for_rate_limit()
            logger.info(f"Cancelling all orders for {config.SYMBOL}")
            response = self.api.cancel_all_orders(
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )
            if response.get("data") is not None or not response.get("error"):
                logger.info("All orders cancelled")
                self.active_orders.clear()
                return True
            else:
                logger.error(f"Cancel all failed: {response}")
                return False
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False
   
    def get_open_orders(self) -> list:
        """Get all open orders."""
        try:
            # CRITICAL: Wait before API call
            self._wait_for_rate_limit()
            
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

    def replace_take_profit(
        self,
        existing_tp_order_id: Optional[str],
        side: str,
        quantity: float,
        new_trigger_price: float,
    ) -> Optional[Dict]:
        """
        Atomically replace a take-profit order:
        - If we know the existing TP order id, attempt a single cancel.
        - Then place a new TP.
        - No loops – higher-level strategy must not spam this method.
        """
        try:
            # Cancel existing TP once, if provided
            if existing_tp_order_id:
                self.wait_for_rate_limit()
                logger.info(f"Cancelling TP order {existing_tp_order_id}")
                cancel_resp = self.api.cancel_order(
                    order_id=existing_tp_order_id,
                    exchange=config.EXCHANGE,
                )
                # If already cancelled or not cancellable, just log and continue
                if cancel_resp.get("error"):
                    logger.warning(f"TP cancel non-fatal: {cancel_resp}")
                else:
                    logger.info(f"TP order cancelled {existing_tp_order_id}")

            # Place new TP
            self.wait_for_rate_limit()
            logger.info(f"Placing TAKE PROFIT {side} {new_trigger_price:,.2f}")
            resp = self.api.place_order(
                symbol=config.SYMBOL,
                side=side.upper(),
                order_type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                trigger_price=new_trigger_price,
                exchange=config.EXCHANGE,
                reduce_only=True,
            )
            data = resp.get("data")
            if data and "order_id" in data:
                new_id = data["order_id"]
                logger.info(f"Take profit order placed {new_id}")
                self.active_orders[new_id] = {
                    "order_id": new_id,
                    "symbol": config.SYMBOL,
                    "side": side,
                    "type": "TAKE_PROFIT",
                    "quantity": quantity,
                    "trigger_price": new_trigger_price,
                    "status": data.get("status", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                }
                return data
            else:
                logger.error(f"Take profit order failed: {resp}")
                return None
        except Exception as e:
            logger.error(f"Error replacing take profit: {e}", exc_info=True)
            return None
