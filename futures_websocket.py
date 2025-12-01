"""
CoinSwitch Futures Trading WebSocket Plugin

Real-time market data streaming for futures markets

FINAL DEBUG VERSION: Logs every incoming message and callback execution
"""

import socketio
import time
import json
import logging
from typing import Callable, Dict, Optional, List
from threading import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesWebSocket:
    """CoinSwitch Futures Trading WebSocket Client"""

    BASE_URL = "https://ws.coinswitch.co"
    HANDSHAKE_PATH = "/pro/realtime-rates-socket/futures/exchange_2"
    NAMESPACE = "/exchange_2"

    # Official event names from docs
    EVENT_ORDERBOOK = "FETCH_ORDER_BOOK_CS_PRO"
    EVENT_CANDLESTICK = "FETCH_CANDLESTICK_CS_PRO"
    EVENT_TRADES = "FETCH_TRADES_CS_PRO"
    EVENT_TICKER = "FETCH_TICKER_INFO_CS_PRO"

    def __init__(self):
        """Initialize Futures WebSocket client"""
        self.sio = socketio.Client()
        self.is_connected = False
        self.stop_event = Event()

        # Callback storage
        self.orderbook_callbacks = []
        self.candlestick_callbacks = []
        self.trades_callbacks = []
        self.ticker_callbacks = []

        self._message_count = 0
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup WebSocket event handlers"""

        @self.sio.event(namespace=self.NAMESPACE)
        def connect():
            self.is_connected = True
            print(f"✓ Connected to futures market")
            logger.info(f"WebSocket connected to {self.NAMESPACE}")

        @self.sio.event(namespace=self.NAMESPACE)
        def disconnect():
            self.is_connected = False
            print(f"✗ Disconnected from futures market")
            logger.info("WebSocket disconnected")

        @self.sio.event(namespace=self.NAMESPACE)
        def connect_error(data):
            print(f"✗ Connection error: {data}")
            logger.error(f"Connection error: {data}")

        # CATCH-ALL HANDLER to see every message
        @self.sio.on('*', namespace=self.NAMESPACE)
        def catch_all(event, *args):
            self._message_count += 1
            data = args[0] if args else None
            logger.info(
                f"[WS MSG #{self._message_count}] event='{event}' | "
                f"data_type={type(data).__name__} | "
                f"data_keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )
            
            # Log first 200 chars of data for debugging
            data_str = str(data)[:200]
            logger.debug(f"  data_sample: {data_str}")

        # Orderbook handler (production)
        @self.sio.on(self.EVENT_ORDERBOOK, namespace=self.NAMESPACE)
        def on_orderbook(data):
            # Ignore subscription acks
            if isinstance(data, dict) and ("bids" in data and "asks" in data):
                formatted = {
                    "b": data["bids"],
                    "a": data["asks"],
                    "timestamp": data.get("timestamp"),
                    "symbol": data.get("s"),
                }
                for callback in self.orderbook_callbacks:
                    try:
                        callback(formatted)
                    except Exception:
                        pass
            # else: ignore acks

        # Trades handler (production)
        @self.sio.on(self.EVENT_TRADES, namespace=self.NAMESPACE)
        def on_trades(data):
            # Ignore subscription acks
            if isinstance(data, dict) and all(k in data for k in ("E", "p", "q")):
                formatted = {
                    "p": data.get("p"),
                    "q": data.get("q"),
                    "T": data.get("E"),
                    "m": data.get("m"),
                    "s": data.get("s"),
                }
                for callback in self.trades_callbacks:
                    try:
                        callback(formatted)
                    except Exception:
                        pass

        # Candlestick handler (official format - direct object)
        @self.sio.on(self.EVENT_CANDLESTICK, namespace=self.NAMESPACE)
        def on_candlestick(data):
            logger.debug(f"[CANDLESTICK HANDLER] Callback fired! data_type={type(data).__name__}")
            
            try:
                # Official format: Direct dict with "o", "h", "l", "c", "v", etc.
                if isinstance(data, dict):
                    for callback in self.candlestick_callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in candlestick callback: {e}", exc_info=True)
                            
            except Exception as e:
                logger.error(f"Error in on_candlestick handler: {e}", exc_info=True)

        # Ticker handler (official format - dict with multiple symbols)
        @self.sio.on(self.EVENT_TICKER, namespace=self.NAMESPACE)
        def on_ticker(data):
            logger.debug(f"[TICKER HANDLER] Callback fired! data_type={type(data).__name__}")
            
            try:
                # Official format: {"BTCUSDT": {...}, "ETHUSDT": {...}}
                if isinstance(data, dict):
                    for symbol, ticker_data in data.items():
                        if isinstance(ticker_data, dict):
                            for callback in self.ticker_callbacks:
                                try:
                                    callback(ticker_data)
                                except Exception as e:
                                    logger.error(f"Error in ticker callback: {e}", exc_info=True)
                                    
            except Exception as e:
                logger.error(f"Error in on_ticker handler: {e}", exc_info=True)

    def connect(self, timeout: int = 30) -> bool:
        """Connect to WebSocket server"""
        try:
            logger.info(f"Connecting to {self.BASE_URL} with namespace {self.NAMESPACE}...")
            self.sio.connect(
                url=self.BASE_URL,
                namespaces=[self.NAMESPACE],
                transports='websocket',
                socketio_path=self.HANDSHAKE_PATH,
                wait=True,
                wait_timeout=timeout
            )
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from WebSocket"""
        try:
            self.stop_event.set()
            if self.sio.connected:
                self.sio.disconnect()
            print("✓ Disconnected successfully")
            logger.info("Disconnected successfully")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            print(f"Disconnect error: {e}")

    def subscribe_orderbook(self, pair: str, callback: Callable = None):
        """Subscribe to order book updates (pair: "BTCUSDT")"""
        subscribe_data = {'event': 'subscribe', 'pair': pair}
        if callback:
            self.orderbook_callbacks.append(callback)
        logger.info(f"Subscribing to orderbook: {pair}")
        self.sio.emit(self.EVENT_ORDERBOOK, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to orderbook: {pair}")

    def subscribe_candlestick(self, pair: str, interval: int = 5, callback: Callable = None):
        """Subscribe to candlestick updates (pair: "BTCUSDT_5")"""
        pair_with_interval = f"{pair}_{interval}"
        subscribe_data = {'event': 'subscribe', 'pair': pair_with_interval}
        if callback:
            self.candlestick_callbacks.append(callback)
        logger.info(f"Subscribing to candlestick: {pair_with_interval}")
        self.sio.emit(self.EVENT_CANDLESTICK, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to candlestick: {pair_with_interval}")

    def subscribe_trades(self, pair: str, callback: Callable = None):
        """Subscribe to trade updates (pair: "BTCUSDT")"""
        subscribe_data = {'event': 'subscribe', 'pair': pair}
        if callback:
            self.trades_callbacks.append(callback)
        logger.info(f"Subscribing to trades: {pair}")
        self.sio.emit(self.EVENT_TRADES, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to trades: {pair}")

    def subscribe_ticker(self, pair: str, callback: Callable = None):
        """Subscribe to ticker updates (pair: "BTCUSDT")"""
        subscribe_data = {'event': 'subscribe', 'pair': pair}
        if callback:
            self.ticker_callbacks.append(callback)
        logger.info(f"Subscribing to ticker: {pair}")
        self.sio.emit(self.EVENT_TICKER, subscribe_data, namespace=self.NAMESPACE)
        print(f"✓ Subscribed to ticker: {pair}")

    def wait(self):
        """Keep connection alive"""
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            print("\n✓ Shutting down...")
            self.disconnect()
