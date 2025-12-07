# telegram_bot_controller.py
"""
Standalone Telegram Bot Controller
Runs independently of the trading bot and can start/stop it via Telegram commands.
"""

import logging
import threading
import time
import json
import sys
from typing import Optional
from urllib import request, parse
from datetime import datetime
import telegram_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TelegramBotController:
    """
    Independent Telegram controller that can start/stop the trading bot.
    Runs as a daemon and listens for commands even when bot is stopped.
    """
    
    def __init__(self) -> None:
        self._token: Optional[str] = telegram_config.TELEGRAM_BOT_TOKEN
        self._chat_id: Optional[str] = telegram_config.TELEGRAM_CHAT_ID
        self.enabled: bool = bool(self._token and self._chat_id)
        
        if not self.enabled:
            logger.error(
                "TelegramBotController disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set"
            )
            sys.exit(1)
        
        self._base_url: str = f"https://api.telegram.org/bot{self._token}"
        self._last_update_id: int = 0
        self._stop_polling: bool = False
        
        # Bot instance control
        self.bot_instance = None
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_running: bool = False
        
        logger.info("TelegramBotController initialized")
        
        # Clear old messages on startup
        self._clear_old_updates()
    
    def _clear_old_updates(self) -> None:
        """Clear all pending updates to avoid processing old commands"""
        try:
            logger.info("Clearing old Telegram messages...")
            updates = self._get_updates(timeout=1)
            if updates:
                count = len(updates)
                logger.info(f"Cleared {count} old message(s)")
            else:
                logger.info("No old messages to clear")
        except Exception as e:
            logger.error(f"Error clearing old updates: {e}")
    
    def start_daemon(self) -> None:
        """Start the daemon controller"""
        if not self.enabled:
            logger.error("Cannot start daemon - Telegram not configured")
            return
        
        logger.info("=" * 80)
        logger.info("TELEGRAM BOT CONTROLLER DAEMON STARTED")
        logger.info("=" * 80)
        logger.info("Waiting for Telegram commands...")
        logger.info("Send 'Start' to start the trading bot")
        logger.info("Send 'Stop' to stop the trading bot")
        logger.info("Send 'Status' to get bot status")
        logger.info("=" * 80)
        
        self._send_message(
            "🤖 <b>BOT CONTROLLER ONLINE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Commands:</b>\n"
            "  • START - Start trading\n"
            "  • STOP - Stop trading\n"
            "  • STATUS - Get stats"
        )

        
        self._stop_polling = False
        
        # Run polling in main thread
        while not self._stop_polling:
            try:
                updates = self._get_updates()
                if updates:
                    for update in updates:
                        self._process_update(update)
                time.sleep(0.5)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop_daemon()
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)
                time.sleep(5.0)
    
    def stop_daemon(self) -> None:
        """Stop the daemon and any running bot"""
        logger.info("Stopping Telegram Bot Controller...")
        self._stop_polling = True
        
        if self.bot_running:
            self._stop_bot()
        
        logger.info("Telegram Bot Controller stopped")
    
    def _get_updates(self, timeout: int = 30) -> list:
        """Get updates from Telegram using long-polling"""
        try:
            url = f"{self._base_url}/getUpdates"
            params = {
                "offset": self._last_update_id + 1,
                "timeout": timeout,
                "allowed_updates": json.dumps(["message"])
            }
            full_url = f"{url}?{parse.urlencode(params)}"
            
            req = request.Request(full_url, method="GET")
            with request.urlopen(req, timeout=timeout + 5) as resp:
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
            logger.debug(f"Error getting updates: {e}")
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
            logger.info(f"Received command: {command}")
            
            if command == "START":
                self._handle_start_command()
            elif command == "STOP":
                self._handle_stop_command()
            elif command == "STATUS":
                self._handle_status_command()
            else:
                self._send_message(
                    f"❌ <b>Unknown:</b> {text}\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    "<b>Commands:</b>\n"
                    "  • START\n"
                    "  • STOP\n"
                    "  • STATUS"
                )
        except Exception as e:
            logger.error(f"Error processing update: {e}", exc_info=True)
    
    def _handle_start_command(self) -> None:
        """Handle START command"""
        try:
            if self.bot_running:
                self._send_message("✅ <b>Already running</b>")
                return
            
            self._send_message("🚀 <b>Starting bot...</b>")
            logger.info("Starting trading bot from Telegram command...")
            
            # Start bot in separate thread
            self.bot_thread = threading.Thread(
                target=self._run_bot,
                daemon=False,
                name="TradingBotThread"
            )
            self.bot_thread.start()
            
            # Wait for bot to initialize with multiple checks
            max_wait = 15.0  # Maximum 15 seconds
            check_interval = 0.5  # Check every 0.5 seconds
            elapsed = 0.0
            
            while elapsed < max_wait:
                time.sleep(check_interval)
                elapsed += check_interval
                
                if self.bot_running:
                            self._send_message(
                                "✅ <b>BOT STARTED</b>\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "Trading engine active"
                            )
                    logger.info("Trading bot started successfully")
                    return
            
            # Timeout - but bot might still be starting
            if self.bot_thread.is_alive():
                self._send_message(
                    "⚠️ <b>Starting...</b>\n"
                    "Taking longer than expected.\n"
                    "Send STATUS to verify."
                )
                logger.warning("Bot thread alive but bot_running flag not set yet")
            else:
                self._send_message("❌ <b>Start failed</b> - check logs")
                logger.error("Failed to start trading bot - thread not alive")
                
        except Exception as e:
            logger.error(f"Error handling START command: {e}", exc_info=True)
            self._send_message(f"❌ Error starting bot: {e}")
    
    def _handle_stop_command(self) -> None:
        """Handle STOP command"""
        try:
            if not self.bot_running:
                self._send_message("⚪ <b>Already stopped</b>")
                return
            
            self._send_message("🛑 <b>Stopping bot...</b>")
            logger.info("Stopping trading bot from Telegram command...")
            
            self._stop_bot()
            
            self._send_message(
                "✅ <b>BOT STOPPED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Trading engine offline"
            )
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error handling STOP command: {e}", exc_info=True)
            self._send_message(f"❌ Error stopping bot: {e}")
    
    def _handle_status_command(self) -> None:
        """Status report with formatted output."""
        try:
            from telegram_notifier import format_status_report
            
            if not self.bot_running or not self.bot_instance:
                msg = (
                    "⚪ <b>BOT OFFLINE</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"Time: {datetime.utcnow().strftime('%H:%M:%S UTC')}"
                )
                self._send_message(msg)
                return
            
            # Get data from running bot
            try:
                last_price = self.bot_instance.data_manager.get_last_price()
                balance_info = self.bot_instance.risk_manager.get_available_balance()
                rm = self.bot_instance.risk_manager
                pos = self.bot_instance.strategy.current_position
                
                # Calculate win rate
                total = rm.total_trades
                win_rate = (rm.winning_trades / total * 100.0) if total > 0 else 0.0
                
                # Position data
                position_side = None
                position_qty = None
                position_entry = None
                position_upnl = None
                
                if pos:
                    position_side = pos.side
                    position_qty = pos.quantity
                    position_entry = pos.entry_price
                    
                    direction = 1.0 if pos.side == "long" else -1.0
                    position_upnl = (last_price - pos.entry_price) * direction * pos.quantity
                
                msg = format_status_report(
                    current_price=last_price,
                    balance=balance_info.get('available', 0.0) if balance_info else 0.0,
                    total_trades=total,
                    win_rate=win_rate,
                    daily_pnl=rm.daily_pnl,
                    total_pnl=rm.realized_pnl,
                    position_side=position_side,
                    position_qty=position_qty,
                    position_entry=position_entry,
                    position_upnl=position_upnl,
                )
                
                # Add WebSocket status
                stats = self.bot_instance.data_manager.stats
                last_update = stats.get('last_update')
                if last_update:
                    idle_sec = (datetime.utcnow() - last_update).total_seconds()
                    ws_status = f"🟢 {idle_sec:.0f}s" if idle_sec < 10 else f"🟡 {idle_sec:.0f}s"
                else:
                    ws_status = "🔴 No data"
                
                msg += f"\n\n<b>WebSocket:</b> {ws_status}"
                
                self._send_message(msg)
                
            except Exception as e:
                self._send_message(
                    f"❌ <b>Error getting status</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"{str(e)[:200]}"
                )
                
        except Exception as e:
            logger.error(f"Error handling STATUS command: {e}", exc_info=True)
            self._send_message(f"❌ <b>Status failed:</b> {e}")

    def _run_bot(self) -> None:
        """Run the trading bot (called in separate thread)"""
        try:
            # Import here to avoid circular imports
            from main import ZScoreIcebergBot
            
            # Set flag immediately so controller knows we're starting
            self.bot_running = True
            logger.info("Bot thread started, initializing bot instance...")
            
            self.bot_instance = ZScoreIcebergBot(controller=self)
            
            # Start the bot (this blocks until bot stops)
            self.bot_instance.start()
            
        except Exception as e:
            logger.error(f"Error running bot: {e}", exc_info=True)
            self._send_message(f"❌ Bot crashed: {e}")
        finally:
            self.bot_running = False
            self.bot_instance = None
            logger.info("Bot thread finished")
   
    def _stop_bot(self) -> None:
        """Stop the running bot"""
        try:
            if self.bot_instance:
                self.bot_instance.stop()
            
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=10.0)
            
            self.bot_running = False
            self.bot_instance = None
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}", exc_info=True)
    
    def _send_message(self, text: str) -> None:
        """Send message to Telegram with HTML formatting."""
        try:
            data = parse.urlencode({
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",  # Enable HTML formatting
                "disable_web_page_preview": "true",
            }).encode("utf-8")
            
            url = f"{self._base_url}/sendMessage"
            req = request.Request(url, data=data, method="POST")
            
            with request.urlopen(req, timeout=5) as resp:
                pass
                
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)


if __name__ == "__main__":
    controller = TelegramBotController()
    try:
        controller.start_daemon()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        controller.stop_daemon()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
