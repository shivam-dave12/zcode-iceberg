"""
Z-SCORE IMBALANCE ICEBERG HUNTER - Main Execution
EVENT-DRIVEN with reduced API spam
"""

import time
import signal
import sys
import logging
from datetime import datetime
from futures_api import FuturesAPI
from futures_websocket import FuturesWebSocket
import config
from data_manager import ZScoreDataManager
from order_manager import OrderManager
from risk_manager import RiskManager
from strategy import ZScoreIcebergHunterStrategy
from zscore_excel_logger import ZScoreExcelLogger
import telegram_config
from telegram_notifier import send_telegram_message, install_global_telegram_log_handler

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"zscore_iceberg_hunter_{datetime.utcnow().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

install_global_telegram_log_handler(level=logging.WARNING, throttle_seconds=5.0)
logger = logging.getLogger(__name__)

def _handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = _handle_uncaught_exception

WS_IDLE_RESTART_SEC = 20.0

class ZScoreIcebergBot:
    """Main bot - Event-Driven with minimal API calls"""

    def __init__(self, controller=None) -> None:
        logger.info("=" * 80)
        logger.info("Z-SCORE BOT INITIALIZING")
        logger.info("=" * 80)
        
        self.controller = controller
        self.api = FuturesAPI(
            api_key=config.COINSWITCH_API_KEY,
            secret_key=config.COINSWITCH_SECRET_KEY,
        )
        self.data_manager = ZScoreDataManager()
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()
        
        if config.ENABLE_EXCEL_LOGGING:
            excel_file = f"zscore_iceberg_hunter_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
            self.excel_logger = ZScoreExcelLogger(filepath=excel_file)
        else:
            self.excel_logger = None
        
        self.strategy = ZScoreIcebergHunterStrategy(excel_logger=self.excel_logger)
        self.running = False
        self._last_stream_check_sec: float = 0.0
        self._last_report_sec: float = 0.0
        self._last_strategy_update_sec: float = 0.0
        
        if controller is None:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers registered")
        
        logger.info("Bot initialized\n")

    def _signal_handler(self, signum, frame) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("\n" + "=" * 80)
        logger.info("SHUTDOWN SIGNAL RECEIVED")
        logger.info("=" * 80)
        self.stop()

    def _register_strategy_callbacks(self) -> None:
        """Register strategy callbacks on WebSocket streams."""
        logger.info("[EVENT-DRIVEN] Registering strategy callbacks")
        
        def on_book_update(data):
            try:
                self._trigger_strategy_update()
            except Exception as e:
                logger.error(f"Error in book callback: {e}")
        
        def on_trade_update(data):
            try:
                self._trigger_strategy_update()
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
        
        if self.data_manager.ws:
            self.data_manager.ws.orderbook_callbacks.append(on_book_update)
            self.data_manager.ws.trades_callbacks.append(on_trade_update)
            logger.info("✓ Strategy callbacks registered")

    def _trigger_strategy_update(self) -> None:
        """Trigger strategy update (throttled to 50ms)."""
        now_sec = time.time()
        if now_sec - self._last_strategy_update_sec < 0.05:
            return
        
        self._last_strategy_update_sec = now_sec
        
        try:
            self.strategy.on_tick(
                data_manager=self.data_manager,
                order_manager=self.order_manager,
                risk_manager=self.risk_manager,
            )
        except Exception as e:
            logger.error(f"Error in strategy update: {e}", exc_info=True)

    def start(self) -> None:
        """Start bot."""
        try:
            logger.info("=" * 80)
            logger.info("STARTING Z-SCORE BOT")
            logger.info("=" * 80)
            
            # Set leverage (ONE-TIME API call)
            logger.info(f"Setting leverage to {config.LEVERAGE}x...")
            lev_res = self.api.set_leverage(
                symbol=config.SYMBOL,
                exchange=config.EXCHANGE,
                leverage=config.LEVERAGE,
            )
            if lev_res:
                logger.info(f"✓ Leverage set to {config.LEVERAGE}x")
            else:
                logger.warning("Leverage may already be set")
            
            # Fetch initial balance (ONE-TIME API call, then cached)
            logger.info("Fetching initial balance...")
            balance_info = self.risk_manager.get_available_balance(force_refresh=True)
            if balance_info:
                logger.info(f"✓ Initial Balance: {balance_info['available']:.2f} USDT")
            else:
                logger.error("Could not fetch balance; aborting")
                return
            
            # Start data manager (WebSocket streams)
            logger.info("Starting data streams...")
            if not self.data_manager.start():
                logger.error("Data manager start failed; aborting")
                return
            
            # Register callbacks
            self._register_strategy_callbacks()
            
            # Wait for first price
            logger.info("Waiting for first price tick...")
            wait_start = time.time()
            while self.data_manager.get_last_price() <= 0:
                if time.time() - wait_start > 60:
                    logger.error("Timeout waiting for initial price")
                    break
                time.sleep(0.5)
            
            last_price = self.data_manager.get_last_price()
            if last_price > 0:
                logger.info(f"✓ First price: {last_price:.2f}")
            
            logger.info("=" * 80)
            logger.info("BOT RUNNING - EVENT-DRIVEN MODE")
            logger.info("Strategy reacts to WebSocket in <50ms")
            logger.info("=" * 80)
            
            self.running = True
            self._run_main_loop()
        
        except Exception as e:
            logger.error(f"Error starting bot: {e}", exc_info=True)
            self.stop()

    def _run_main_loop(self) -> None:
        """Event-driven main loop (only health checks)."""
        logger.info("Main loop active (minimal overhead)")
        last_health_sec = time.time()
        
        while self.running:
            try:
                now_sec = time.time()
                
                # Health check every 1s
                if now_sec - last_health_sec >= 1.0:
                    self._check_stream_health()
                    self._maybe_send_telegram_report()
                    last_health_sec = now_sec
                
                # Minimal sleep
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1.0)

    def _check_stream_health(self) -> None:
        """Monitor WebSocket health."""
        now_sec = time.time()
        if now_sec - self._last_stream_check_sec < 1.0:
            return
        
        self._last_stream_check_sec = now_sec
        
        try:
            last_update = self.data_manager.stats.get("last_update")
            if not last_update:
                return
            
            idle_sec = (datetime.utcnow() - last_update).total_seconds()
            if idle_sec <= WS_IDLE_RESTART_SEC:
                return
            
            logger.warning("=" * 80)
            logger.warning(f"WebSocket stale ({idle_sec:.1f}s) - restarting")
            logger.warning("=" * 80)
            
            try:
                self.data_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping data manager: {e}")
            
            time.sleep(2.0)
            
            for attempt in range(3):
                try:
                    logger.info(f"Restart attempt {attempt + 1}/3...")
                    if self.data_manager.start():
                        self._register_strategy_callbacks()
                        logger.info("✓ Data manager restarted")
                        break
                    else:
                        logger.warning(f"Restart attempt {attempt + 1} failed")
                        time.sleep(5.0)
                except Exception as e:
                    logger.error(f"Error in restart: {e}")
                    time.sleep(5.0)
        
        except Exception as e:
            logger.error(f"Error in stream health check: {e}")

    def stop(self) -> None:
        """Clean shutdown."""
        self.running = False
        try:
            if self.data_manager:
                self.data_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping data manager: {e}")
        
        try:
            if self.excel_logger:
                self.excel_logger.close()
        except Exception as e:
            logger.error(f"Error closing Excel logger: {e}")
        
        logger.info("=" * 80)
        logger.info("BOT STOPPED")
        logger.info("=" * 80)

    def _maybe_send_telegram_report(self) -> None:
        """Send periodic Telegram report (15 min intervals)."""
        interval = config.TELEGRAM_REPORT_INTERVAL_SEC
        if interval <= 0:
            return
        
        now = time.time()
        if now - self._last_report_sec < interval:
            return
        
        self._last_report_sec = now
        
        try:
            last_price = self.data_manager.get_last_price()
            
            # Use cached balance (no API call)
            balance_info = self.risk_manager.get_available_balance()
            pos = self.strategy.current_position
            
            lines = [
                "📊 Z-Score Bot Report",
                datetime.utcnow().strftime("%H:%M:%S UTC"),
                "",
            ]
            
            if last_price > 0:
                lines.append(f"Price: {last_price:.2f}")
            
            if balance_info:
                lines.append(f"Balance: {balance_info.get('available', 0.0):.2f} USDT")
            
            if pos:
                dur_min = (now - pos.entry_time_sec) / 60.0
                direction = 1.0 if pos.side == "long" else -1.0
                upnl = (last_price - pos.entry_price) * direction * pos.quantity
                lines.append(
                    f"Position: {pos.side.upper()} {pos.quantity:.3f} @ {pos.entry_price:.2f}\n"
                    f"uPnL: {upnl:.2f} ({upnl/pos.margin_used*100:.1f}%) | {dur_min:.1f}min"
                )
            
            send_telegram_message("\n".join(lines))
        except Exception:
            logger.exception("Failed to send Telegram report")


if __name__ == "__main__":
    try:
        bot = ZScoreIcebergBot()
        bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
