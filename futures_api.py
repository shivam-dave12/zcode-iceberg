"""
CoinSwitch Futures Trading API Plugin
CORRECTED: Proper signature generation matching official specification
"""

import os
import time
import json
import requests
import urllib.parse
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from urllib.parse import urlparse, urlencode
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class FuturesAPI:
    """CoinSwitch Futures Trading API Client"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Futures API client
        
        Args:
            api_key: CoinSwitch API key
            secret_key: CoinSwitch secret key
        """
        self.api_key = api_key or os.getenv('COINSWITCH_API_KEY')
        self.secret_key = secret_key or os.getenv('COINSWITCH_SECRET_KEY')
        self.base_url = "https://coinswitch.co"
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key required")
    
    def _generate_signature(self, method: str, endpoint: str, params: Dict = None, payload: Dict = None) -> str:
        """
        Generate ED25519 signature
        
        CRITICAL: Based on official documentation
        - GET: signature = METHOD + ENDPOINT + JSON_PAYLOAD (even if empty {})
        - POST/DELETE: signature = METHOD + ENDPOINT + JSON_PAYLOAD (sorted keys)
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters for GET
            payload: Body payload
            
        Returns:
            Signature as hex string
        """
        params = params or {}
        payload = payload or {}
        
        unquote_endpoint = endpoint
        
        # Add query parameters to endpoint if GET
        if method == "GET" and len(params) != 0:
            endpoint += ('&', '?')[urlparse(endpoint).query == ''] + urlencode(params)
            unquote_endpoint = urllib.parse.unquote_plus(endpoint)
        
        # CRITICAL: Always include JSON payload in signature (even empty {})
        payload_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        signature_msg = method + unquote_endpoint + payload_json
        
        request_string = bytes(signature_msg, 'utf-8')
        secret_key_bytes = bytes.fromhex(self.secret_key)
        secret_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
        signature_bytes = secret_key_obj.sign(request_string)
        
        return signature_bytes.hex()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, payload: Dict = None) -> Dict:
        """Make authenticated API request WITH EPOCH HEADER (2024 req)."""
        signature, epoch_ms = self._generate_signature(method, endpoint, params, payload)
        
        url = self.base_url + endpoint
        headers = {
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
            "X-AUTH-EPOCH": str(epoch_ms),  # Required header
            "Content-Type": "application/json",
        }
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, json=payload, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed ({method} {endpoint}): {e}"
            if 'response' in locals() and response:
                error_msg += f" | Status: {response.status_code} | Body: {response.text[:500]}..."  # Trunc raw body for debug
            logger.error(error_msg)
            return {
                "error": str(e),
                "status_code": getattr(response, 'status_code', None) if 'response' in locals() else None,
                "response_text": getattr(response, 'text', 'N/A')[:200] if 'response' in locals() else 'N/A'  # For debug
            }
    
    # ============ ORDER MANAGEMENT ============
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                   exchange: str = "EXCHANGE_2", price: float = None, 
                   trigger_price: float = None, reduce_only: bool = False) -> Dict:
        """
        Place a futures order
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            side: BUY or SELL
            order_type: MARKET, LIMIT, TAKE_PROFIT_MARKET, STOP_MARKET
            quantity: Order quantity in base asset
            exchange: Exchange identifier
            price: Limit price (required for LIMIT orders)
            trigger_price: Trigger price (for TAKE_PROFIT_MARKET/STOP_MARKET)
            reduce_only: Reduce only flag (for TP/SL orders)
            
        Returns:
            Order response with order_id
        """
        endpoint = "/trade/api/v2/futures/order"
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "side": side,
            "order_type": order_type,
            "quantity": quantity
        }
        
        if price is not None:
            payload["price"] = price
        if trigger_price is not None:
            payload["trigger_price"] = trigger_price
        if reduce_only:
            payload["reduce_only"] = reduce_only
        
        return self._make_request("POST", endpoint, payload=payload)
    
    def cancel_order(self, order_id: str, exchange: str = "EXCHANGE_2") -> Dict:
        """
        Cancel a futures order
        
        Args:
            order_id: Order ID to cancel
            exchange: Exchange identifier
        """
        endpoint = "/trade/api/v2/futures/order"
        payload = {
            "order_id": order_id,
            "exchange": exchange
        }
        return self._make_request("DELETE", endpoint, payload=payload)
    
    def get_order(self, order_id: str) -> Dict:
        """
        Get order status
        
        Args:
            order_id: Order ID to query
        """
        endpoint = "/trade/api/v2/futures/order"
        params = {"order_id": order_id}
        return self._make_request("GET", endpoint, params=params, payload={})
    
    def get_open_orders(self, exchange: str = "EXCHANGE_2", symbol: str = None,
                       limit: int = 50, from_time: int = None, to_time: int = None) -> Dict:
        """
        Get open orders
        
        Args:
            exchange: Exchange identifier
            symbol: Filter by symbol
            limit: Max orders to return (max 50)
            from_time: Start time in milliseconds
            to_time: End time in milliseconds
        """
        endpoint = "/trade/api/v2/futures/orders/open"
        payload = {"exchange": exchange}
        
        if symbol:
            payload["symbol"] = symbol
        if limit:
            payload["limit"] = min(limit, 50)
        if from_time:
            payload["from_time"] = from_time
        if to_time:
            payload["to_time"] = to_time
        
        return self._make_request("POST", endpoint, payload=payload)
    
    def get_closed_orders(self, exchange: str = "EXCHANGE_2", symbol: str = None,
                         limit: int = 50, from_time: int = None, to_time: int = None) -> Dict:
        """
        Get closed orders
        
        Args:
            exchange: Exchange identifier
            symbol: Filter by symbol
            limit: Max orders to return (max 50)
            from_time: Start time in milliseconds
            to_time: End time in milliseconds
        """
        endpoint = "/trade/api/v2/futures/orders/closed"
        payload = {"exchange": exchange}
        
        if symbol:
            payload["symbol"] = symbol
        if limit:
            payload["limit"] = min(limit, 50)
        if from_time:
            payload["from_time"] = from_time
        if to_time:
            payload["to_time"] = to_time
        
        return self._make_request("POST", endpoint, payload=payload)
    
    def cancel_all_orders(self, exchange: str = "EXCHANGE_2", symbol: str = None) -> Dict:
        """
        Cancel all open orders
        
        Args:
            exchange: Exchange identifier
            symbol: Cancel orders for specific symbol (optional)
        """
        endpoint = "/trade/api/v2/futures/cancel_all"
        payload = {"exchange": exchange}
        
        if symbol:
            payload["symbol"] = symbol
        
        return self._make_request("POST", endpoint, payload=payload)
    
    # ============ LEVERAGE & MARGIN ============
    
    def set_leverage(self, symbol: str, leverage: int, exchange: str = "EXCHANGE_2") -> Dict:
        """
        Set leverage for a symbol
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            leverage: Leverage value (1 to max_leverage)
            exchange: Exchange identifier
        """
        endpoint = "/trade/api/v2/futures/leverage"
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "leverage": leverage
        }
        return self._make_request("POST", endpoint, payload=payload)
    
    def get_leverage(self, symbol: str, exchange: str = "EXCHANGE_2") -> Dict:
        """
        Get current leverage for symbol
        
        Args:
            symbol: Trading symbol
            exchange: Exchange identifier
        """
        endpoint = "/trade/api/v2/futures/leverage"
        params = {
            "symbol": symbol,
            "exchange": exchange
        }
        return self._make_request("GET", endpoint, params=params, payload={})
    
    def add_margin(self, symbol: str, margin: float, exchange: str = "EXCHANGE_2") -> Dict:
        """
        Add margin to position
        
        Args:
            symbol: Trading symbol
            margin: Margin amount to add
            exchange: Exchange identifier
        """
        endpoint = "/trade/api/v2/futures/add_margin"
        payload = {
            "exchange": exchange,
            "symbol": symbol,
            "margin": margin
        }
        return self._make_request("POST", endpoint, payload=payload)
    
    # ============ POSITIONS & ACCOUNT ============
    
    def get_positions(self, exchange: str = "EXCHANGE_2", symbol: str = None) -> Dict:
        """
        Get open positions
        
        Args:
            exchange: Exchange identifier
            symbol: Filter by symbol
        """
        endpoint = "/trade/api/v2/futures/positions"
        params = {"exchange": exchange}
        
        if symbol:
            params["symbol"] = symbol
        
        return self._make_request("GET", endpoint, params=params, payload={})
    
    def get_wallet_balance(self) -> Dict:
        """Get futures wallet balance"""
        endpoint = "/trade/api/v2/futures/wallet_balance"
        return self._make_request("GET", endpoint, params={}, payload={})
    
    def get_transactions(self, exchange: str = "EXCHANGE_2", symbol: str = None,
                        transaction_type: str = None, transaction_id: str = None) -> Dict:
        """
        Get transaction history
        
        Args:
            exchange: Exchange identifier
            symbol: Filter by symbol
            transaction_type: Filter by type (commission, P&L, funding fee, liquidation fee)
            transaction_id: Specific transaction ID
        """
        endpoint = "/trade/api/v2/futures/transactions"
        params = {"exchange": exchange}
        
        if symbol:
            params["symbol"] = symbol
        if transaction_type:
            params["type"] = transaction_type
        if transaction_id:
            params["transaction_id"] = transaction_id
        
        return self._make_request("GET", endpoint, params=params, payload={})
    
    def get_instrument_info(self, exchange: str = "EXCHANGE_2") -> Dict:
        """
        Get instrument specifications
        
        Args:
            exchange: Exchange identifier
        """
        endpoint = "/trade/api/v2/futures/instrument_info"
        params = {"exchange": exchange}
        return self._make_request("GET", endpoint, params=params, payload={})


if __name__ == "__main__":
    # Quick test
    try:
        api = FuturesAPI()
        print("✓ Futures API initialized successfully")
        
        # Test getting positions
        positions = api.get_positions()
        print(f"✓ Positions retrieved")
        
        # Test getting wallet balance
        balance = api.get_wallet_balance()
        print(f"✓ Wallet balance retrieved")
        
    except Exception as e:
        print(f"✗ Error: {e}")
