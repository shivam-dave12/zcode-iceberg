# telegram_bot_controller.py
"""
Standalone Telegram Bot Controller
Runs independently of the trading bot and can start/stop it via Telegram commands.

UPDATED: Added "logs" command to retrieve last 2 log entries.
"""

import logging
import threading
import time
import json
import sys
import os
from typing import Optional, List
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

        # Log storage for "logs" command
        self._log_buffer: List[str] = []
        self._log_buffer_max_size: int = 100
        self._log_file_handler: Optional[logging.FileHandler] = None

        logger.info("TelegramBotController initialized")

        # Clear old messages on startup
        self._clear_old_updates()

        # Setup log capture
        self._setup_log_capture()

    def _setup_log_capture(self) -> None:
        """Setup logging handler to capture logs into buffer."""
        try:
            # Create a custom handler that writes to our buffer
            class BufferHandler(logging.Handler):
                def __init__(self, buffer: List[str], max_size: int):
                    super().__init__()
                    self.buffer = buffer
                    self.max_size = max_size

                def emit(self, record):
                    try:
                        msg = self.format(record)
                        self.buffer.append(msg)
                        if len(self.buffer) > self.max_size:
                            self.buffer.pop(0)
                    except Exception:
                        pass

            buffer_handler = BufferHandler(self._log_buffer, self._log_buffer_max_size)
            buffer_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            buffer_handler.setLevel(logging.INFO)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(buffer_handler)

            logger.info("Log buffer capture initialized")

        except Exception as e:
            logger.error(f"Error setting up log capture: {e}", exc_info=True)

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
        logger.info("Send 'START' to start the trading bot")
        logger.info("Send 'STOP' to stop the trading bot")
        logger.info("Send 'STATUS' to get bot status")
        logger.info("Send 'LOGS' to get last 2 log entries")
        logger.info("=" * 80)

        self._send_message(
            "🤖 Telegram Bot Controller Online\n\n"
            "Available commands:\n"
            "• START - Start trading bot\n"
            "• STOP - Stop trading bot\n"
            "• STATUS - Get current status\n"
            "• LOGS - Get last 2 log entries"
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
            elif command == "LOGS":
                self._handle_logs_command()
            else:
                self._send_message(
                    f"❓ Unknown command: {text}\n\n"
                    "Available commands:\n"
                    "• START - Start trading bot\n"
                    "• STOP - Stop trading bot\n"
                    "• STATUS - Get current status\n"
                    "• LOGS - Get last 2 log entries"
                )

        except Exception as e:
            logger.error(f"Error processing update: {e}", exc_info=True)

    def _handle_start_command(self) -> None:
        """Handle START command"""
        try:
            if self.bot_running:
                self._send_message("✅ Trading bot is already running")
                return

            self._send_message("🚀 Starting trading bot...")
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
                    self._send_message("✅ Trading bot started successfully")
                    logger.info("Trading bot started successfully")
                    return

            # Timeout - but bot might still be starting
            if self.bot_thread.is_alive():
                self._send_message(
                    "⚠️ Bot is starting (taking longer than expected)...\n"
                    "Send 'Status' in a few seconds to verify"
                )
                logger.warning("Bot thread alive but bot_running flag not set yet")
            else:
                self._send_message("❌ Failed to start trading bot - check logs")
                logger.error("Failed to start trading bot - thread not alive")

        except Exception as e:
            logger.error(f"Error handling START command: {e}", exc_info=True)
            self._send_message(f"❌ Error starting bot: {e}")

    def _handle_stop_command(self) -> None:
        """Handle STOP command"""
        try:
            if not self.bot_running:
                self._send_message("⚠️ Trading bot is already stopped")
                return

            self._send_message("🛑 Stopping trading bot...")
            logger.info("Stopping trading bot from Telegram command...")

            self._stop_bot()

            self._send_message("✅ Trading bot stopped successfully")
            logger.info("Trading bot stopped successfully")

        except Exception as e:
            logger.error(f"Error handling STOP command: {e}", exc_info=True)
            self._send_message(f"❌ Error stopping bot: {e}")

    def _handle_status_command(self) -> None:
        """Handle STATUS command"""
        try:
            lines = ["📊 BOT STATUS REPORT", "=" * 40, ""]

            # Bot running state
            if self.bot_running and self.bot_instance:
                lines.append("✅ Status: RUNNING")

                try:
                    # Get detailed status from running bot
                    last_price = self.bot_instance.data_manager.get_last_price()
                    if last_price > 0:
                        lines.append(f"💹 Current Price: {last_price:.2f}")

                    balance_info = self.bot_instance.risk_manager.get_available_balance()
                    if balance_info:
                        lines.append(f"💰 Balance: {balance_info.get('available', 0.0):.2f} USDT")

                    rm = self.bot_instance.risk_manager
                    lines.append(f"📊 Total Trades: {rm.total_trades}")
                    lines.append(f"📊 Win Rate: {(rm.winning_trades / rm.total_trades * 100.0) if rm.total_trades > 0 else 0:.2f}%")
                    lines.append(f"📊 Daily P&L: {rm.daily_pnl:.2f} USDT")

                    pos = self.bot_instance.strategy.current_position
                    if pos:
                        lines.append(f"📍 Position: {pos.side.upper()} {pos.quantity:.6f} BTC @ {pos.entry_price:.2f}")
                    else:
                        lines.append("📍 Position: None")

                    stats = self.bot_instance.data_manager.stats
                    last_update = stats.get('last_update')
                    if last_update:
                        idle_sec = (datetime.utcnow() - last_update).total_seconds()
                        lines.append(f"🌐 WebSocket: {idle_sec:.1f}s ago")

                except Exception as e:
                    lines.append(f"⚠️ Error getting details: {e}")
            else:
                lines.append("⛔ Status: STOPPED")

            lines.append(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append("=" * 40)

            self._send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Error handling STATUS command: {e}", exc_info=True)
            self._send_message(f"❌ Error getting status: {e}")

    def _handle_logs_command(self) -> None:
        """Handle LOGS command - send last 2 log entries"""
        try:
            if len(self._log_buffer) == 0:
                self._send_message("📋 No logs available yet")
                return

            # Get last 2 log entries
            last_logs = self._log_buffer[-2:] if len(self._log_buffer) >= 2 else self._log_buffer

            lines = ["📋 LAST 2 LOG ENTRIES", "=" * 40, ""]
            for log_entry in last_logs:
                lines.append(log_entry)
                lines.append("")  # Empty line between entries

            lines.append("=" * 40)

            message = "\n".join(lines)

            # Telegram has 4096 char limit, truncate if needed
            if len(message) > 3800:
                message = message[:3800] + "\n... [truncated]"

            self._send_message(message)

        except Exception as e:
            logger.error(f"Error handling LOGS command: {e}", exc_info=True)
            self._send_message(f"❌ Error retrieving logs: {e}")

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
        """Send message to Telegram"""
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
