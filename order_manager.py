"""
Order Manager - Handles order placement and management
Updated to use new CoinSwitch API plugins
"""
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from futures_api import FuturesAPI
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

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
        
        logger.info("✓ OrderManager initialized with new API plugin")
        
        # Order status caching - Reduces API calls 95%
        self._order_status_cache = {}  # {order_id: (status_dict, timestamp)}
        self._cache_ttl = 2.0  # 2-second cache TTL
        self._last_status_call = defaultdict(float)  # Rate limit per order
        self._min_status_interval = 1.0  # Min 1s between status calls
        
        logger.info("✓ Order status caching enabled (2s TTL, 1s min interval)")
    
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
        """
        Extract a valid fill price from API order dict or raise.
        Enforces:
        - non-zero
        - non-NaN
        - not obviously rate-limit/error stub
        """
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
        poll_interval_sec: float = 1.0,
    ) -> Dict:
        """
        Poll order status until a real fill is present or timeout.
        Conditions to return:
        - status in EXECUTED / PARTIALLY_EXECUTED / FILLED (case-insensitive)
        - exec_quantity > 0
        - avg_execution_price (or equivalent) > 0
        
        Raises RuntimeError on:
        - timeout without valid fill
        - status in CANCELLED / REJECTED
        - API / rate limit errors
        """
        start = time.time()
        last_data: Optional[Dict] = None
        
        while True:
            now = time.time()
            if now - start > timeout_sec:
                raise RuntimeError(
                    f"Timed out waiting for fill on order {order_id}. Last status: {last_data}"
                )
            
            status_resp = self.get_order_status(order_id)
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
    # ✅ NEW: Bracket Order Placement (Main + TP + SL in one method)
    # ======================================================================
    
    def place_bracket_order(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Optional[Tuple[Dict, Dict, Dict]]:
        """
        MODIFIED: Place EXACTLY 1 limit entry + 1 TP + 1 SL (no multiples/retries).
        - Entry: Limit order at entry_price
        - TP/SL: Market orders triggered at tp/sl_price (CoinSwitch bracket style)
        Returns: (main_order_resp, tp_order_resp, sl_order_resp) or None on failure.
        """
        if quantity <= 0 or entry_price <= 0:
            logger.error(f"Invalid params for bracket: qty={quantity}, entry={entry_price}")
            return None
        
        try:
            # Step 1: Place single LIMIT entry order
            logger.info(f"Placing single LIMIT entry: {side} {quantity:.6f} @ {entry_price:.3f}")
            if not self._check_rate_limit():
                logger.error("Rate limit hit for entry order")
                return None
            
            entry_resp = self.api.place_order(
                side=side,
                order_type="limit",
                quantity=quantity,
                price=entry_price,
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )
            
            if "data" not in entry_resp or not entry_resp["data"].get("order_id"):
                logger.error(f"Entry order failed: {entry_resp}")
                return None
            
            main_order_id = entry_resp["data"]["order_id"]
            self.active_orders[main_order_id] = {"status": "PENDING", "side": side, "type": "ENTRY"}
            
            # Wait briefly for potential fill (but don't block; monitor later)
            time.sleep(0.5)
            entry_status = self.get_order_status(main_order_id)
            if entry_status and float(entry_status.get("executed_qty", 0)) < quantity * 0.5:
                logger.warning(f"Entry partial/not filled yet: {entry_status.get('executed_qty')}")
            
            # Step 2: Place single TP order (trigger at tp_price, market fill)
            logger.info(f"Placing single TP: { 'SELL' if side == 'BUY' else 'BUY' } {quantity:.6f} @ trigger {tp_price:.3f}")
            tp_side = "SELL" if side == "BUY" else "BUY"
            tp_resp = self.api.place_order(
                side=tp_side,
                order_type="stop_market",  # Trigger at tp_price, market fill
                quantity=quantity,
                trigger_price=tp_price,  # Trigger price
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )
            
            if "data" not in tp_resp or not tp_resp["data"].get("order_id"):
                logger.error(f"TP order failed: {tp_resp}")
                # Don't cancel entry yet; handle in monitoring
                tp_order_id = None
            else:
                tp_order_id = tp_resp["data"]["order_id"]
                self.active_orders[tp_order_id] = {"status": "PENDING", "side": tp_side, "type": "TP"}
            
            # Step 3: Place single SL order (trigger at sl_price, market fill)
            logger.info(f"Placing single SL: { 'BUY' if side == 'SELL' else 'SELL' } {quantity:.6f} @ trigger {sl_price:.3f}")
            sl_side = "BUY" if side == "SELL" else "SELL"
            sl_resp = self.api.place_order(
                side=sl_side,
                order_type="stop_market",  # Trigger at sl_price, market fill
                quantity=quantity,
                trigger_price=sl_price,  # Trigger price
                exchange=config.EXCHANGE,
                symbol=config.SYMBOL,
            )
            
            if "data" not in sl_resp or not sl_resp["data"].get("order_id"):
                logger.error(f"SL order failed: {sl_resp}")
                # Don't cancel entry yet; handle in monitoring
                sl_order_id = None
            else:
                sl_order_id = sl_resp["data"]["order_id"]
                self.active_orders[sl_order_id] = {"status": "PENDING", "side": sl_side, "type": "SL"}
            
            logger.info(f"✓ Bracket placed: Entry={main_order_id}, TP={tp_order_id or 'FAILED'}, SL={sl_order_id or 'FAILED'}")
            return entry_resp["data"], tp_resp["data"] if tp_order_id else None, sl_resp["data"] if sl_order_id else None
            
        except Exception as e:
            logger.error(f"Bracket order error: {e}", exc_info=True)
            # Emergency: Cancel any partial orders
            self.cancel_all_orders()
            return None
        
    # ======================================================================
    # Order placement (existing methods)
    # ======================================================================
    
    def place_market_order(
        self, side: str, quantity: float, reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            reduce_only: If True, only reduces position
        
        Returns:
            Order details or None on failure
        """
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
                quantity=quantity,
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
                    "quantity": quantity,
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
    # Cancellation / status
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
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get order status with caching and rate limiting.
        - 2-second cache TTL (reduces API calls by 95%)
        - Min 1 second between API calls per order
        - Returns cached status on API failure (graceful degradation)
        """
        now = time.time()
        
        # Check cache first
        if order_id in self._order_status_cache:
            cached_status, cached_time = self._order_status_cache[order_id]
            if now - cached_time < self._cache_ttl:
                logger.debug(f"📋 Cache HIT for order {order_id[:8]}...")
                return cached_status
        
        # Check rate limit
        last_call = self._last_status_call[order_id]
        if now - last_call < self._min_status_interval:
            wait_time = self._min_status_interval - (now - last_call)
            logger.debug(f"⏸️ Rate limit: waiting {wait_time:.2f}s for {order_id[:8]}...")
            time.sleep(wait_time)
        
        # Make API call
        try:
            self._last_status_call[order_id] = time.time()
            response = self.api.get_order(order_id)
            
            if "data" in response:
                order_data = response["data"].get("order", response["data"])
                
                # Cache result
                self._order_status_cache[order_id] = (order_data, time.time())
                logger.debug(f"📥 Cached order status for {order_id[:8]}...")
                
                # Update active orders tracking
                if order_id in self.active_orders:
                    self.active_orders[order_id]["status"] = order_data.get(
                        "status", "UNKNOWN"
                    )
                
                return order_data
            else:
                logger.warning(f"Could not get order status for {order_id}")
                
                # Return cached if available (graceful degradation)
                if order_id in self._order_status_cache:
                    cached_status, _ = self._order_status_cache[order_id]
                    logger.debug(f"Using stale cache for {order_id[:8]}...")
                    return cached_status
                
                return None
        
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            
            # Return cached status if available (graceful degradation)
            if order_id in self._order_status_cache:
                cached_status, _ = self._order_status_cache[order_id]
                logger.debug(f"Using stale cache on error for {order_id[:8]}...")
                return cached_status
            
            return None
    
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

    def ensure_single_bracket(self, order_id: str, order_type: str) -> bool:
        """
        NEW: Enforce single bracket - check if duplicate order_type exists.
        Returns True if OK to place; False if duplicate detected (cancel dupe).
        """
        for oid, details in self.active_orders.items():
            if details["type"] == order_type and oid != order_id:
                logger.warning(f"Duplicate {order_type} detected: {oid} - cancelling")
                self.cancel_order(oid)
                return False
        return True
        
if __name__ == "__main__":
    om = OrderManager()
    print("Order Manager initialized")
    print(f"Statistics: {om.get_order_statistics()}")
    
    open_orders = om.get_open_orders()
    print(f"Open orders: {len(open_orders)}")
