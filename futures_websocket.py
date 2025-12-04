"""
CoinSwitch Futures WebSocket Client
Fully updated: Event-driven callbacks, robust reconnect, auth, error handling
"""

import websocket
import json
import threading
import time
import logging
from typing import Optional, Callable, List, Dict, Tuple
import config

logger = logging.getLogger(__name__)

class FuturesWebSocket:
    """
    Production-grade WebSocket client with:
    - Callbacks for book/trade/candle
    - Exponential backoff reconnect
    - Subscription persistence
    - Auth support
    - Thread-safe
    """

    def __init__(self,
                 callback_book: Optional[Callable[[List[Tuple[float, float]], List[Tuple[float, float]]], None]] = None,
                 callback_trade: Optional[Callable[[List[Dict]], None]] = None,
                 callback_candle: Optional[Callable[[Dict], None]] = None):
        self.ws = None
        self.callback_book = callback_book
        self.callback_trade = callback_trade
        self.callback_candle = callback_candle
        self.running = False
        self.thread = None
        self.base_url = "wss://stream.coinswitch.co/ws"
        self.subscriptions = set()
        self.reconnect_attempts = 0
        self.max_reconnects = 5

    def connect(self) -> bool:
        try:
            def on_open(ws):
                logger.info("WebSocket CONNECTED – sending auth & subscriptions")
                self._on_open(ws)

            def on_message(ws, message):
                self._on_message(message)

            def on_error(ws, error):
                logger.error(f"WebSocket ERROR: {error}")

            def on_close(ws, code, msg):
                logger.warning(f"WebSocket CLOSED: {code} – {msg}")
                self.running = False
                self._attempt_reconnect()

            self.ws = websocket.WebSocketApp(
                self.base_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            self.thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={"ping_interval": 30, "ping_timeout": 10}
            )
            self.thread.daemon = True
            self.thread.start()

            time.sleep(2)  # Allow connect
            self.running = True
            self.reconnect_attempts = 0
            return True

        except Exception as e:
            logger.error(f"WS connect failed: {e}", exc_info=True)
            return False

    def _on_open(self, ws):
        # Auth
        auth_msg = {
            "method": "AUTH",
            "params": {"api_key": config.COINSWITCH_API_KEY},
            "id": 0
        }
        ws.send(json.dumps(auth_msg))
        time.sleep(0.5)

        # Resubscribe
        for sub in self.subscriptions:
            ws.send(json.dumps(sub))
        logger.info(f"Re-subscribed to {len(self.subscriptions)} streams")

    def _on_message(self, message: str):
        try:
            data = json.loads(message)

            if "stream" not in data:
                if "id" in data:
                    logger.debug(f"WS ACK/ID: {data.get('id')}")
                return

            stream = data["stream"].lower()

            if "depth" in stream or "orderbook" in stream:
                bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
                if self.callback_book:
                    self.callback_book(bids, asks)

            elif "trade" in stream:
                trades = data.get("trades", [data])
                if self.callback_trade:
                    self.callback_trade(trades)

            elif "kline" in stream or "candle" in stream:
                k = data.get("k", {})
                candle = {
                    "ts": float(k.get("t", time.time() * 1000)),
                    "close": float(k.get("c", 0.0))
                }
                if self.callback_candle:
                    self.callback_candle(candle)

        except Exception as e:
            logger.error(f"WS message parse error: {e} | Raw: {message[:200]}")

    def _attempt_reconnect(self):
        if self.reconnect_attempts >= self.max_reconnects:
            logger.error("Max reconnects reached – giving up")
            return

        self.reconnect_attempts += 1
        backoff = min(60, 2 ** self.reconnect_attempts)
        logger.warning(f"Reconnecting in {backoff}s (attempt {self.reconnect_attempts}/{self.max_reconnects})")
        time.sleep(backoff)
        self.connect()

    # ------------------------------------------------------------------ #
    # Subscription Methods
    # ------------------------------------------------------------------ #

    def subscribe_orderbook(self, symbol: str):
        sub = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@depth@100ms"],
            "id": int(time.time() * 1000)
        }
        self.subscriptions.add(json.dumps(sub))
        if self.ws and self.running:
            self.ws.send(json.dumps(sub))
        logger.info(f"Subscribed: orderbook {symbol}")

    def subscribe_trades(self, symbol: str):
        sub = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@trade"],
            "id": int(time.time() * 1000)
        }
        self.subscriptions.add(json.dumps(sub))
        if self.ws and self.running:
            self.ws.send(json.dumps(sub))
        logger.info(f"Subscribed: trades {symbol}")

    def subscribe_candles(self, symbol: str, interval: int):
        sub = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@kline_{interval}m"],
            "id": int(time.time() * 1000)
        }
        self.subscriptions.add(json.dumps(sub))
        if self.ws and self.running:
            self.ws.send(json.dumps(sub))
        logger.info(f"Subscribed: {interval}m candles {symbol}")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def disconnect(self):
        self.running = False
        if self.ws:
            self.ws.close()
            logger.info("WebSocket disconnected")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self.subscriptions.clear()