# telegram_command_handler.py
import logging
import threading
import time
import json
from typing import Optional, Callable
from urllib import request, parse
from datetime import datetime
import telegram_config

logger = logging.getLogger(__name__)

class TelegramCommandHandler:
    """
    Listens for incoming Telegram commands using long-polling.
    Handles Start, Stop, Status commands to control the bot.
    """
    
    def __init__(self, bot_instance) -> None:
        self._token: Optional[str] = telegram_config.TELEGRAM_BOT_TOKEN
        self._chat_id: Optional[str] = telegram_config.TELEGRAM_CHAT_ID
        self.enabled: bool = bool(self._token and self._chat_id)
        self.bot_instance = bot_instance
        
        if not self.enabled:
            logger.warning(
                "TelegramCommandHandler disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
            )
            return
        
        self._base_url: str = f"https://api.telegram.org/bot{self._token}"
        self._last_update_id: int = 0
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_polling: bool = False
        
        logger.info("TelegramCommandHandler initialized")
    
    def start_polling(self) -> None:
        """Start background thread to poll for commands"""
        if not self.enabled:
            return
        
        if self._polling_thread and self._polling_thread.is_alive():
            logger.warning("Polling thread already running")
            return
        
        self._stop_polling = False
        self._polling_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="TelegramCommandPoller"
        )
        self._polling_thread.start()
        logger.info("Telegram command polling started")
    
    def stop_polling(self) -> None:
        """Stop polling thread"""
        self._stop_polling = True
        if self._polling_thread:
            self._polling_thread.join(timeout=5.0)
        logger.info("Telegram command polling stopped")
    
    def _poll_loop(self) -> None:
        """Main polling loop running in background thread"""
        while not self._stop_polling:
            try:
                updates = self._get_updates()
                if updates:
                    for update in updates:
                        self._process_update(update)
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in Telegram polling loop: {e}", exc_info=True)
                time.sleep(5.0)
    
    def _get_updates(self) -> list:
        """Get updates from Telegram using long-polling"""
        try:
            url = f"{self._base_url}/getUpdates"
            params = {
                "offset": self._last_update_id + 1,
                "timeout": 30,
                "allowed_updates": json.dumps(["message"])
            }
            full_url = f"{url}?{parse.urlencode(params)}"
            
            req = request.Request(full_url, method="GET")
            with request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                if data.get("ok") and data.get("result"):
                    updates = data["result"]
                    if updates:
                        self._last_update_id = max(
                            u.get("update_id", 0) for u in updates
                        )
                    return updates
                return []
        except Exception as e:
            logger.error(f"Error getting Telegram updates: {e}", exc_info=True)
            return []
    
    def _process_update(self, update: dict) -> None:
        """Process a single update/message"""
        try:
            message = update.get("message", {})
            chat_id = str(message.get("chat", {}).get("id", ""))
            text = message.get("text", "").strip()
            
            # Only process messages from authorized chat
            if chat_id != self._chat_id:
                logger.warning(f"Ignoring message from unauthorized chat: {chat_id}")
                return
            
            if not text:
                return
            
            command = text.upper()
            
            if command == "START":
                self._handle_start_command()
            elif command == "STOP":
                self._handle_stop_command()
            elif command == "STATUS":
                self._handle_status_command()
            else:
                self._send_message(
                    f"Unknown command: {text}\n\n"
                    "Available commands:\n"
                    "START - Start the trading bot\n"
                    "STOP - Stop the trading bot\n"
                    "STATUS - Get current bot status"
                )
        except Exception as e:
            logger.error(f"Error processing Telegram update: {e}", exc_info=True)
    
    def _handle_start_command(self) -> None:
        """Handle START command"""
        try:
            if self.bot_instance.running:
                self._send_message("✅ Bot is already running")
                return
            
            self._send_message("🚀 Starting bot...")
            
            # Start bot in separate thread to avoid blocking
            start_thread = threading.Thread(
                target=self.bot_instance.start,
                daemon=False,
                name="BotStartThread"
            )
            start_thread.start()
            
            time.sleep(2.0)
            
            if self.bot_instance.running:
                self._send_message("✅ Bot started successfully")
            else:
                self._send_message("❌ Failed to start bot - check logs")
                
        except Exception as e:
            logger.error(f"Error handling START command: {e}", exc_info=True)
            self._send_message(f"❌ Error starting bot: {e}")
    
    def _handle_stop_command(self) -> None:
        """Handle STOP command - graceful shutdown"""
        try:
            if not self.bot_instance.running:
                self._send_message("⚠️ Bot is already stopped")
                return
            
            self._send_message("🛑 Stopping bot...")
            
            # Stop bot
            self.bot_instance.stop()
            
            self._send_message("✅ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error handling STOP command: {e}", exc_info=True)
            self._send_message(f"❌ Error stopping bot: {e}")
    
    def _handle_status_command(self) -> None:
        """Handle STATUS command - provide detailed status"""
        try:
            lines = ["📊 BOT STATUS REPORT", "=" * 40, ""]
            
            # Bot running state
            if self.bot_instance.running:
                lines.append("✅ Status: RUNNING")
            else:
                lines.append("⛔ Status: STOPPED")
            
            lines.append(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append("")
            
            # Market data
            try:
                last_price = self.bot_instance.data_manager.get_last_price()
                if last_price > 0:
                    lines.append(f"💹 Current Price: {last_price:.2f}")
                else:
                    lines.append("💹 Current Price: N/A")
            except:
                lines.append("💹 Current Price: ERROR")
            
            # DataManager stats (last 2 snapshots concept)
            try:
                stats = self.bot_instance.data_manager.stats
                lines.append("")
                lines.append("📈 Data Stream Stats:")
                lines.append(f"  Orderbook updates: {stats.get('orderbook_updates', 0)}")
                lines.append(f"  Trades received: {stats.get('trades_received', 0)}")
                lines.append(f"  Candles received: {stats.get('candles_received', 0)}")
                lines.append(f"  Prices recorded: {stats.get('prices_recorded', 0)}")
                
                last_update = stats.get('last_update')
                if last_update:
                    idle_sec = (datetime.utcnow() - last_update).total_seconds()
                    lines.append(f"  Last update: {idle_sec:.1f}s ago")
            except:
                lines.append("📈 Data Stream Stats: ERROR")
            
            # Balance
            try:
                balance_info = self.bot_instance.risk_manager.get_available_balance()
                if balance_info:
                    lines.append("")
                    lines.append("💰 Balance:")
                    lines.append(f"  Available: {balance_info.get('available', 0.0):.2f} {balance_info.get('currency', 'USDT')}")
                    lines.append(f"  Total: {balance_info.get('total', 0.0):.2f} {balance_info.get('currency', 'USDT')}")
            except:
                lines.append("💰 Balance: ERROR")
            
            # P&L and trade stats
            try:
                rm = self.bot_instance.risk_manager
                lines.append("")
                lines.append("📊 Performance:")
                lines.append(f"  Total trades: {rm.total_trades}")
                lines.append(f"  Winning trades: {rm.winning_trades}")
                lines.append(f"  Losing trades: {rm.losing_trades}")
                
                if rm.total_trades > 0:
                    win_rate = (rm.winning_trades / rm.total_trades) * 100.0
                    lines.append(f"  Win rate: {win_rate:.2f}%")
                
                lines.append(f"  Daily trades: {rm.daily_trades}")
                lines.append(f"  Daily P&L: {rm.daily_pnl:.2f} USDT")
            except:
                lines.append("📊 Performance: ERROR")
            
            # Current position
            try:
                pos = self.bot_instance.strategy.current_position
                lines.append("")
                if pos:
                    lines.append("📍 Open Position:")
                    lines.append(f"  Side: {pos.side.upper()}")
                    lines.append(f"  Quantity: {pos.quantity:.6f} BTC")
                    lines.append(f"  Entry Price: {pos.entry_price:.2f}")
                    lines.append(f"  TP Price: {pos.tp_price:.2f}")
                    lines.append(f"  SL Price: {pos.sl_price:.2f}")
                    
                    dur_min = (time.time() - pos.entry_time_sec) / 60.0
                    lines.append(f"  Duration: {dur_min:.1f} minutes")
                    
                    # Unrealized P&L estimate
                    try:
                        current_price = self.bot_instance.data_manager.get_last_price()
                        if current_price > 0:
                            direction = 1.0 if pos.side == "long" else -1.0
                            upnl = (current_price - pos.entry_price) * direction * pos.quantity
                            lines.append(f"  Unrealized P&L: ~{upnl:.2f} USDT")
                    except:
                        pass
                else:
                    lines.append("📍 Open Position: None")
            except:
                lines.append("📍 Open Position: ERROR")
            
            # WebSocket health
            try:
                lines.append("")
                if self.bot_instance.data_manager.is_streaming:
                    lines.append("🌐 WebSocket: CONNECTED")
                else:
                    lines.append("🌐 WebSocket: DISCONNECTED")
            except:
                lines.append("🌐 WebSocket: UNKNOWN")
            
            lines.append("")
            lines.append("=" * 40)
            
            self._send_message("\n".join(lines))
            
        except Exception as e:
            logger.error(f"Error handling STATUS command: {e}", exc_info=True)
            self._send_message(f"❌ Error getting status: {e}")
    
    def _send_message(self, text: str) -> None:
        """Send message to Telegram (helper)"""
        try:
            data = parse.urlencode({
                "chat_id": self._chat_id,
                "text": text,
                "disable_web_page_preview": "true",
            }).encode("utf-8")
            
            url = f"{self._base_url}/sendMessage"
            req = request.Request(url, data=data, method="POST")
            
            with request.urlopen(req, timeout=5) as resp:
                pass
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}", exc_info=True)
