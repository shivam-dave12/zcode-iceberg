"""
Order Manager - PRODUCTION GRADE

Rate Limiting: Token bucket + adaptive backoff + circuit breaker
Error Handling: 429 detection, exponential backoff with jitter
Thread Safety: Request queue + operation deduplication
"""

import time
import logging
import random
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime
import threading
from collections import deque
from futures_api import FuturesAPI
import config

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL RATE LIMITER WITH ADAPTIVE BACKOFF
# ============================================================================

class AdaptiveRateLimiter:
    """
    Token bucket rate limiter with adaptive backoff and circuit breaker.
    """
    
    def __init__(self, base_interval: float = 2.5, max_tokens: int = 5):
        self.lock = threading.Lock()
        self.base_interval = base_interval
        self.current_interval = base_interval
        self.last_request_time = 0.0
        self.tokens = float(max_tokens)
        self.max_tokens = float(max_tokens)
        self.refill_rate = max_tokens / 60.0  # Refill over 60 seconds
        self.last_refill_time = time.time()
        
        # Circuit breaker
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0
        self.max_failures_before_circuit = 3
        self.circuit_cooldown = 10.0  # 10 seconds
        
        # 429 tracking
        self.last_429_time = 0.0
        self.penalty_until = 0.0
        
        logger.info(f"✓ AdaptiveRateLimiter initialized: base={base_interval}s, max_tokens={max_tokens}")
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        if elapsed > 0:
            new_tokens = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill_time = now
    
    def acquire(self, operation: str = "generic") -> Tuple[bool, float]:
        """
        Attempt to acquire permission to make a request.
        Returns: (allowed, wait_time_if_denied)
        """
        with self.lock:
            now = time.time()
            
            # Check circuit breaker
            if now < self.circuit_open_until:
                wait_time = self.circuit_open_until - now
                logger.warning(f"⛔ Circuit breaker OPEN for {operation}: wait {wait_time:.1f}s")
                return False, wait_time
            
            # Check 429 penalty
            if now < self.penalty_until:
                wait_time = self.penalty_until - now
                logger.debug(f"⏸️ 429 penalty active for {operation}: wait {wait_time:.1f}s")
                return False, wait_time
            
            # Refill tokens
            self._refill_tokens()
            
            # Check if we have tokens
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.refill_rate
                return False, wait_time
            
            # Check minimum interval
            elapsed_since_last = now - self.last_request_time
            if elapsed_since_last < self.current_interval:
                wait_time = self.current_interval - elapsed_since_last
                return False, wait_time
            
            # All checks passed - consume token
            self.tokens -= 1.0
            self.last_request_time = now
            return True, 0.0
    
    def report_success(self):
        """Report successful API call - reduce backoff."""
        with self.lock:
            self.consecutive_failures = 0
            # Gradually reduce interval back to base
            self.current_interval = max(
                self.base_interval,
                self.current_interval * 0.9
            )
    
    def report_failure(self, is_429: bool = False):
        """Report failed API call - increase backoff."""
        with self.lock:
            self.consecutive_failures += 1
            now = time.time()
            
            if is_429:
                # Aggressive penalty for rate limit
                self.last_429_time = now
                penalty_duration = min(30.0, 5.0 * (2 ** self.consecutive_failures))
                self.penalty_until = now + penalty_duration
                logger.warning(f"⚠️ 429 Rate Limit Hit - Penalty: {penalty_duration:.1f}s")
            
            # Exponential backoff on current interval
            self.current_interval = min(
                10.0,  # Max 10 seconds between requests
                self.current_interval * 1.5
            )
            
            # Open circuit breaker if too many failures
            if self.consecutive_failures >= self.max_failures_before_circuit:
                self.circuit_open_until = now + self.circuit_cooldown
                logger.error(
                    f"💥 Circuit breaker OPENED: {self.consecutive_failures} consecutive failures. "
                    f"Cooldown: {self.circuit_cooldown}s"
                )
    
    def wait_with_jitter(self, base_wait: float) -> None:
        """Sleep with jittered duration to prevent thundering herd."""
        jitter = random.uniform(0, base_wait * 0.2)  # Add up to 20% jitter
        actual_wait = base_wait + jitter
        logger.debug(f"Sleeping {actual_wait:.2f}s (base={base_wait:.2f}s + jitter={jitter:.2f}s)")
        time.sleep(actual_wait)

# Global rate limiter instance
_GLOBAL_RATE_LIMITER = AdaptiveRateLimiter(base_interval=3.5, max_tokens=20)

# ============================================================================
# OPERATION DEDUPLICATOR (prevents duplicate simultaneous requests)
# ============================================================================

class OperationDeduplicator:
    """Prevents duplicate concurrent operations (e.g., placing same TP order twice)."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.in_flight = {}  # operation_hash -> (timestamp, thread_id)
        self.recent_completions = deque(maxlen=100)  # Last 100 completed operations
    
    def _hash_operation(self, op_type: str, **kwargs) -> str:
        """Create hash of operation for deduplication."""
        key_str = f"{op_type}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def check_and_register(self, op_type: str, **kwargs) -> Tuple[bool, str]:
        """
        Check if operation is already in flight or recently completed.
        Returns: (is_duplicate, operation_hash)
        """
        op_hash = self._hash_operation(op_type, **kwargs)
        
        with self.lock:
            now = time.time()
            
            # Check if already in flight
            if op_hash in self.in_flight:
                start_time, thread_id = self.in_flight[op_hash]
                age = now - start_time
                logger.warning(
                    f"🔄 Duplicate {op_type} detected (hash={op_hash}, age={age:.1f}s, thread={thread_id})"
                )
                return True, op_hash
            
            # Check recent completions (within last 5 seconds)
            for completed_hash, completed_time in self.recent_completions:
                if completed_hash == op_hash and (now - completed_time) < 5.0:
                    logger.warning(f"⏭️ {op_type} recently completed (hash={op_hash})")
                    return True, op_hash
            
            # Register as in-flight
            self.in_flight[op_hash] = (now, threading.get_ident())
            return False, op_hash
    
    def mark_complete(self, op_hash: str):
        """Mark operation as complete."""
        with self.lock:
            if op_hash in self.in_flight:
                del self.in_flight[op_hash]
            self.recent_completions.append((op_hash, time.time()))
    
    def mark_failed(self, op_hash: str):
        """Mark operation as failed (remove from in-flight)."""
        with self.lock:
            if op_hash in self.in_flight:
                del self.in_flight[op_hash]

# Global deduplicator
_OPERATION_DEDUP = OperationDeduplicator()

# ============================================================================
# ORDER MANAGER
# ============================================================================

class OrderManager:
    """Manages order placement, cancellation, and tracking."""
    
    def __init__(self):
        """Initialize order manager."""
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
        
        logger.info("✓ OrderManager initialized with production-grade rate limiting")
    
    # ======================================================================
    # HELPER METHODS
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
    
    def _is_rate_limit_error(self, response: Dict) -> bool:
        """Check if response indicates rate limiting."""
        if not isinstance(response, dict):
            return False
        
        # Check status code
        if response.get("status_code") == 429:
            return True
        
        # Check error message
        error_msg = str(response.get("error", "")).lower()
        if "too many requests" in error_msg or "rate limit" in error_msg:
            return True
        
        # Check response message
        if "response" in response and isinstance(response["response"], dict):
            resp_msg = str(response["response"].get("message", "")).lower()
            if "too many requests" in resp_msg:
                return True
        
        return False
    
    # ======================================================================
    # PRODUCTION-GRADE ORDER STATUS CHECKING
    # ======================================================================
    
    def get_order_status(self, order_id: str, max_retries: int = 3) -> Optional[Dict]:
        """
        PRODUCTION-GRADE order status checking with:
        - Adaptive rate limiting with circuit breaker
        - Exponential backoff with jitter on failures
        - 429 detection and aggressive penalty
        - Returns None on persistent failures (caller must handle gracefully)
        """
        global _GLOBAL_RATE_LIMITER
        
        # Check for duplicate operation
        is_duplicate, op_hash = _OPERATION_DEDUP.check_and_register(
            "get_order_status",
            order_id=order_id
        )
        
        if is_duplicate:
            logger.warning(f"Blocking duplicate get_order_status for {order_id}")
            return None
        
        last_response = None
        attempt = 0
        
        try:
            while attempt < max_retries:
                attempt += 1
                
                # Wait for rate limiter permission
                max_wait_attempts = 10
                wait_attempt = 0
                while wait_attempt < max_wait_attempts:
                    allowed, wait_time = _GLOBAL_RATE_LIMITER.acquire("get_order_status")
                    if allowed:
                        break
                    
                    if wait_time > 0:
                        logger.debug(f"Rate limit: waiting {wait_time:.1f}s (attempt {wait_attempt+1}/{max_wait_attempts})")
                        _GLOBAL_RATE_LIMITER.wait_with_jitter(min(wait_time, 5.0))
                    
                    wait_attempt += 1
                
                if not allowed:
                    logger.error(f"❌ Could not acquire rate limit permission after {max_wait_attempts} attempts")
                    _GLOBAL_RATE_LIMITER.report_failure()
                    continue
                
                # Make API call
                try:
                    response = self.api.get_order_status(order_id)
                    last_response = response
                    
                    # Check for rate limit error
                    if self._is_rate_limit_error(response):
                        logger.warning(f"[STATUS CHECK #{attempt}] 429 Rate Limit Error")
                        _GLOBAL_RATE_LIMITER.report_failure(is_429=True)
                        
                        # Exponential backoff with jitter
                        backoff_time = min(30.0, 2.0 * (2 ** attempt))
                        _GLOBAL_RATE_LIMITER.wait_with_jitter(backoff_time)
                        continue
                    
                    # Check for other errors
                    if "error" in response:
                        logger.warning(f"[STATUS CHECK #{attempt}] API error: {response.get('error')}")
                        _GLOBAL_RATE_LIMITER.report_failure()
                        
                        backoff_time = 1.0 * attempt
                        _GLOBAL_RATE_LIMITER.wait_with_jitter(backoff_time)
                        continue
                    
                    # Successful response structure check
                    if isinstance(response, dict):
                        # CoinSwitch v2: data.order.status OR direct status
                        if "data" in response:
                            order_data = response.get("data", {})
                            if isinstance(order_data, dict) and ("order" in order_data or "status" in order_data):
                                _GLOBAL_RATE_LIMITER.report_success()
                                _OPERATION_DEDUP.mark_complete(op_hash)
                                return response
                        elif "order" in response or "status" in response:
                            _GLOBAL_RATE_LIMITER.report_success()
                            _OPERATION_DEDUP.mark_complete(op_hash)
                            return response
                    
                    # Unexpected response
                    logger.warning(f"[STATUS CHECK #{attempt}] Unexpected response structure")
                    _GLOBAL_RATE_LIMITER.report_failure()
                    backoff_time = 1.0 * attempt
                    _GLOBAL_RATE_LIMITER.wait_with_jitter(backoff_time)
                    
                except Exception as exc:
                    logger.warning(f"[STATUS CHECK #{attempt}] Exception: {exc}")
                    _GLOBAL_RATE_LIMITER.report_failure()
                    backoff_time = 1.0 * attempt
                    _GLOBAL_RATE_LIMITER.wait_with_jitter(backoff_time)
            
            # Exhausted all retries
            logger.error(
                f"❌ get_order_status FAILED after {max_retries} retries: {order_id}\n"
                f"Last response: {last_response}"
            )
            _OPERATION_DEDUP.mark_failed(op_hash)
            return None
            
        except Exception as e:
            logger.error(f"Fatal error in get_order_status: {e}", exc_info=True)
            _OPERATION_DEDUP.mark_failed(op_hash)
            return None
    
    def get_order_status_safe(self, order_id: str) -> str:
        """
        SAFE wrapper that NEVER raises exceptions.
        Returns normalized status string: FILLED, CANCELLED, UNKNOWN
        """
        try:
            response = self.get_order_status(order_id)
            
            if response is None:
                logger.warning(f"⚠️ Cannot determine status for {order_id} - treating as UNKNOWN")
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
    # ORDER PLACEMENT WITH DEDUPLICATION
    # ======================================================================
    
    def place_take_profit(
        self, side: str, quantity: float, trigger_price: float
    ) -> Optional[Dict]:
        """Place a take profit order with deduplication."""
        # Check for duplicate operation
        is_duplicate, op_hash = _OPERATION_DEDUP.check_and_register(
            "place_take_profit",
            side=side,
            quantity=quantity,
            trigger_price=trigger_price,
            symbol=config.SYMBOL
        )
        
        if is_duplicate:
            logger.warning(
                f"🔄 Blocking duplicate TP order: {side} {quantity} @ {trigger_price}"
            )
            return None
        
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
            
            # Check for "already exists" error
            if "error" in response or response.get("status_code") in (400, 500):
                error_msg = str(response.get("response", {}).get("message", "")).lower()
                if "already exists" in error_msg:
                    logger.warning(f"ℹ️ TP order already exists (skipping duplicate)")
                    _OPERATION_DEDUP.mark_complete(op_hash)
                    return None
                
                logger.error(f"✗ Take profit order failed: {response}")
                _OPERATION_DEDUP.mark_failed(op_hash)
                return None
            
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
                
                _OPERATION_DEDUP.mark_complete(op_hash)
                return response["data"]
            else:
                logger.error(f"✗ Take profit order failed: {response}")
                _OPERATION_DEDUP.mark_failed(op_hash)
                return None
        
        except Exception as e:
            logger.error(f"Error placing take profit: {e}")
            _OPERATION_DEDUP.mark_failed(op_hash)
            return None
    
    def place_stop_loss(
        self, side: str, quantity: float, trigger_price: float
    ) -> Optional[Dict]:
        """Place a stop loss order with deduplication."""
        # Check for duplicate operation
        is_duplicate, op_hash = _OPERATION_DEDUP.check_and_register(
            "place_stop_loss",
            side=side,
            quantity=quantity,
            trigger_price=trigger_price,
            symbol=config.SYMBOL
        )
        
        if is_duplicate:
            logger.warning(
                f"🔄 Blocking duplicate SL order: {side} {quantity} @ {trigger_price}"
            )
            return None
        
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
            
            # Check for "already exists" error
            if "error" in response or response.get("status_code") in (400, 500):
                error_msg = str(response.get("response", {}).get("message", "")).lower()
                if "already exists" in error_msg:
                    logger.warning(f"ℹ️ SL order already exists (skipping duplicate)")
                    _OPERATION_DEDUP.mark_complete(op_hash)
                    return None
                
                logger.error(f"✗ Stop loss order failed: {response}")
                _OPERATION_DEDUP.mark_failed(op_hash)
                return None
            
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
                
                _OPERATION_DEDUP.mark_complete(op_hash)
                return response["data"]
            else:
                logger.error(f"✗ Stop loss order failed: {response}")
                _OPERATION_DEDUP.mark_failed(op_hash)
                return None
        
        except Exception as e:
            logger.error(f"Error placing stop loss: {e}")
            _OPERATION_DEDUP.mark_failed(op_hash)
            return None
    
    # Continue with other methods (place_limit_order, cancel_order, etc.) using same pattern...
    
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
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with better error handling."""
        try:
            logger.info(f"Cancelling order: {order_id}")
            
            response = self.api.cancel_order(
                order_id=order_id,
                exchange=config.EXCHANGE,
            )
            
            # Check if already cancelled
            if response.get("status_code") == 500:
                error_msg = str(response.get("response", {}).get("message", "")).lower()
                if "already" in error_msg and "cancelled" in error_msg:
                    logger.info(f"ℹ️ Order already cancelled: {order_id}")
                    if order_id in self.active_orders:
                        self.active_orders[order_id]["status"] = "CANCELLED"
                    return True
            
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
