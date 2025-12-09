#!/usr/bin/env python3
"""
CoinSwitch Futures Trading WebSocket Plugin
Real-time market data streaming for futures markets
INDUSTRY-GRADE with robust reconnection + subscription tracking
"""

import socketio
import time
import json
import logging
from typing import Callable, Dict, Optional, List
from threading import Event, RLock
from datetime import datetime
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesWebSocket:
    """CoinSwitch Futures Trading WebSocket Client with ROBUST reconnection"""
    
    BASE_URL = "https://ws.coinswitch.co"
    HANDSHAKE_PATH = "/pro/realtime-rates-socket/futures/exchange2"
    NAMESPACE = "/exchange2"
    
    # Event names
    EVENT_ORDERBOOK = "FETCH_ORDERBOOK_CS_PRO"
    EVENT_CANDLESTICK = "FETCH_CANDLESTICK_CS_PRO"
    EVENT_TRADES = "FETCH_TRADES_CS_PRO"
    EVENT_TICKER = "FETCH_TICKER_INFO_CS_PRO"
    
    # ✅ Reconnection settings
    MAX_RECONNECT_ATTEMPTS = 10
    RECONNECT_DELAY_BASE = 2  # seconds
    RECONNECT_DELAY_MAX = 60  # seconds

    def __init__(self):
        """Initialize Futures WebSocket client"""
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=0)  # Manual reconnection
        self.is_connected = False
        self.stop_event = Event()
        
        # Callback storage
        self.orderbook_callbacks: List[Callable] = []
        self.candlestick_callbacks: List[Callable] = []
        self.trades_callbacks: List[Callable] = []
        self.ticker_callbacks: List[Callable] = []
        
        # ✅ Track active subscriptions for reconnection
        self.active_subscriptions = {
            'orderbook': [],
            'candlestick': [],
            'trades': [],
            'ticker': []
        }
        
        # Thread safety
        self.callbacks_lock = RLock()
        
        self.message_count = 0
        self.reconnect_attempts = 0
        self.last_disconnect_time = 0
        self.last_message_time = None
        
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.event(namespace=self.NAMESPACE)
        def connect():
            self.is_connected = True
            self.reconnect_attempts = 0  # Reset on successful connect
            self.last_message_time = datetime.now()
            print("✓ Connected to futures market")
            logger.info(f"WebSocket connected to {self.NAMESPACE}")

        @self.sio.event(namespace=self.NAMESPACE)
        def disconnect():
            self.is_connected = False
            self.last_disconnect_time = time.time()
            print("✗ Disconnected from futures market")
            logger.warning("WebSocket disconnected")
            
            # ✅ Auto-reconnect if not intentional shutdown
            if not self.stop_event.is_set():
                self._attempt_reconnection()

        @self.sio.event(namespace=self.NAMESPACE)
        def connect_error(data):
            print(f"✗ Connection error: {data}")
            logger.error(f"Connection error: {data}")

        # Orderbook handler
        @self.sio.on(self.EVENT_ORDERBOOK, namespace=self.NAMESPACE)
        def on_orderbook(data):
            self.last_message_time = datetime.now()
            self.message_count += 1
            if isinstance(data, dict) and ("bids" in data or "b" in data):
                formatted = {
                    "b": data.get("bids", data.get("b", [])),
                    "a": data.get("asks", data.get("a", [])),
                    "timestamp": data.get("timestamp"),
                    "symbol": data.get("s"),
                }
                for callback in self.orderbook_callbacks:
                    try:
                        callback(formatted)
                    except Exception as e:
                        logger.error(f"Orderbook callback error: {e}")

        # Trades handler
        @self.sio.on(self.EVENT_TRADES, namespace=self.NAMESPACE)
        def on_trades(data):
            self.last_message_time = datetime.now()
            self.message_count += 1
            if isinstance(data, dict) and "p" in data:
                formatted = {
                    "p": data.get("p"),
                    "q": data.get("q"),
                    "T": data.get("E", data.get("T")),
                    "m": data.get("m"),
                    "s": data.get("s"),
                }
                for callback in self.trades_callbacks:
                    try:
                        callback(formatted)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")

        # Candlestick handler
        @self.sio.on(self.EVENT_CANDLESTICK, namespace=self.NAMESPACE)
        def on_candlestick(data):
            self.last_message_time = datetime.now()
            self.message_count += 1
            if isinstance(data, dict) and "c" in data:
                for callback in self.candlestick_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Candlestick callback error: {e}")

    def _attempt_reconnection(self):
        """
        ✅ ROBUST reconnection with exponential backoff
        """
        if self.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(f"❌ Max reconnection attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached")
            return

        self.reconnect_attempts += 1
        
        # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s (max)
        delay = min(
            self.RECONNECT_DELAY_BASE ** self.reconnect_attempts,
            self.RECONNECT_DELAY_MAX
        )
        
        logger.info(f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS} in {delay}s...")
        time.sleep(delay)
        
        try:
            # Disconnect old connection first
            if self.sio.connected:
                self.sio.disconnect()
            
            # Create new client
            self.sio = socketio.Client(reconnection=True, reconnection_attempts=0)
            self._setup_handlers()
            
            # Attempt connection
            success = self.connect(timeout=10)
            
            if success:
                logger.info(f"✅ Reconnection successful after {self.reconnect_attempts} attempts")
                # Re-subscribe to all channels
                self._resubscribe_all()
            else:
                logger.warning(f"⚠️  Reconnection attempt {self.reconnect_attempts} failed")
                self._attempt_reconnection()  # Retry
                
        except Exception as e:
            logger.error(f"Reconnection error: {e}")
            self._attempt_reconnection()  # Retry

    def _resubscribe_all(self):
        """Re-subscribe to all active channels after reconnection"""
        logger.info("🔄 Re-subscribing to all channels after reconnection...")
        
        try:
            # Resubscribe to orderbook
            for pair in self.active_subscriptions['orderbook']:
                subscribe_data = {"event": "subscribe", "pair": pair}
                self.sio.emit(self.EVENT_ORDERBOOK, subscribe_data, namespace=self.NAMESPACE)
                logger.info(f"  ✓ Re-subscribed to orderbook: {pair}")
                time.sleep(0.1)  # Small delay between subscriptions
            
            # Resubscribe to candlestick
            for pair_interval in self.active_subscriptions['candlestick']:
                subscribe_data = {"event": "subscribe", "pair": pair_interval}
                self.sio.emit(self.EVENT_CANDLESTICK, subscribe_data, namespace=self.NAMESPACE)
                logger.info(f"  ✓ Re-subscribed to candlestick: {pair_interval}")
                time.sleep(0.1)
            
            # Resubscribe to trades
            for pair in self.active_subscriptions['trades']:
                subscribe_data = {"event": "subscribe", "pair": pair}
                self.sio.emit(self.EVENT_TRADES, subscribe_data, namespace=self.NAMESPACE)
                logger.info(f"  ✓ Re-subscribed to trades: {pair}")
                time.sleep(0.1)
            
            # Resubscribe to ticker
            for pair in self.active_subscriptions['ticker']:
                subscribe_data = {"event": "subscribe", "pair": pair}
                self.sio.emit(self.EVENT_TICKER, subscribe_data, namespace=self.NAMESPACE)
                logger.info(f"  ✓ Re-subscribed to ticker: {pair}")
                time.sleep(0.1)
            
            logger.info("✅ All channels re-subscribed successfully")
        except Exception as e:
            logger.error(f"Error during re-subscription: {e}")

    def connect(self, timeout: int = 30) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.BASE_URL} with namespace {self.NAMESPACE}...")
            self.sio.connect(
                url=self.BASE_URL,
                namespaces=[self.NAMESPACE],
                transports=["websocket"],
                socketio_path=self.HANDSHAKE_PATH,
                wait=True,
                wait_timeout=timeout,
            )
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            print(f"✗ Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from WebSocket"""
        try:
            self.stop_event.set()  # Signal intentional disconnect
            if self.sio.connected:
                self.sio.disconnect()
            print("✓ Disconnected successfully")
            logger.info("Disconnected successfully")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            print(f"✗ Disconnect error: {e}")

    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """Check if connection is healthy (receiving data)."""
        if not self.is_connected:
            return False
        if self.last_message_time is None:
            return True  # Just connected, no messages yet
        time_since_last = (datetime.now() - self.last_message_time).total_seconds()
        return time_since_last <= timeout_seconds

    def subscribe_orderbook(self, pair: str, callback: Optional[Callable] = None):
        """Subscribe to order book updates (thread-safe)"""
        subscribe_data = {"event": "subscribe", "pair": pair}
        
        with self.callbacks_lock:
            if callback and callback not in self.orderbook_callbacks:
                self.orderbook_callbacks.append(callback)
        
            # Track subscription
            if pair not in self.active_subscriptions['orderbook']:
                self.active_subscriptions['orderbook'].append(pair)
        
        logger.info(f"Subscribing to orderbook: {pair}")
        self.sio.emit(self.EVENT_ORDERBOOK, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to orderbook: {pair}")

    def subscribe_candlestick(self, pair: str, interval: int = 5, callback: Optional[Callable] = None):
        """Subscribe to candlestick updates (thread-safe)"""
        pair_with_interval = f"{pair}{interval}"
        subscribe_data = {"event": "subscribe", "pair": pair_with_interval}
        
        with self.callbacks_lock:
            if callback and callback not in self.candlestick_callbacks:
                self.candlestick_callbacks.append(callback)
            
            # Track subscription
            if pair_with_interval not in self.active_subscriptions['candlestick']:
                self.active_subscriptions['candlestick'].append(pair_with_interval)
        
        logger.info(f"Subscribing to candlestick: {pair_with_interval}")
        self.sio.emit(self.EVENT_CANDLESTICK, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to candlestick: {pair_with_interval}")

    def subscribe_trades(self, pair: str, callback: Optional[Callable] = None):
        """Subscribe to trade updates (thread-safe)"""
        subscribe_data = {"event": "subscribe", "pair": pair}
        
        with self.callbacks_lock:
            if callback and callback not in self.trades_callbacks:
                self.trades_callbacks.append(callback)
            
            # Track subscription
            if pair not in self.active_subscriptions['trades']:
                self.active_subscriptions['trades'].append(pair)
        
        logger.info(f"Subscribing to trades: {pair}")
        self.sio.emit(self.EVENT_TRADES, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to trades: {pair}")

    def subscribe_ticker(self, pair: str, callback: Optional[Callable] = None):
        """Subscribe to ticker updates (thread-safe)"""
        subscribe_data = {"event": "subscribe", "pair": pair}
        
        with self.callbacks_lock:
            if callback and callback not in self.ticker_callbacks:
                self.ticker_callbacks.append(callback)
            
            # Track subscription
            if pair not in self.active_subscriptions['ticker']:
                self.active_subscriptions['ticker'].append(pair)
        
        logger.info(f"Subscribing to ticker: {pair}")
        self.sio.emit(self.EVENT_TICKER, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to ticker: {pair}")

    def wait(self):
        """Keep connection alive"""
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            print("Shutting down...")
            self.disconnect()
