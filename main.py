"""
Z-Score Imbalance Iceberg Hunter - Main Execution

High-frequency BTCUSDT futures scalper on CoinSwitch PRO.
UPDATED: Event-driven WebSocket callbacks for sub-50ms latency.
✅ FIXED: Proper startup notification flow
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
from telegram_notifier import (
    send_telegram_message,
    install_global_telegram_log_handler,
)

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

install_global_telegram_log_handler(
    level=logging.WARNING,
    throttle_seconds=5.0,
)

logger = logging.getLogger(__name__)

def _handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )

sys.excepthook = _handle_uncaught_exception

WS_IDLE_RESTART_SEC = 20.0

class ZScoreIcebergBot:
    """Main bot class for Z-Score Imbalance Iceberg Hunter."""
    
    def __init__(self, controller=None) -> None:
        logger.info("=" * 80)
        logger.info("Z-SCORE IMBALANCE ICEBERG HUNTER BOT")
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
            excel_file = (
                f"zscore_iceberg_hunter_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )
            self.excel_logger = ZScoreExcelLogger(filepath=excel_file)
        else:
            self.excel_logger = None
        
        self.strategy = ZScoreIcebergHunterStrategy(
            excel_logger=self.excel_logger,
        )
        
        self.running = False
        self._last_stream_check_sec: float = 0.0
        self._last_report_sec: float = 0.0
        
        # Event-driven callbacks registered flag
        self._callbacks_registered: bool = False
        
        # Only set signal handlers if running standalone (not from controller)
        if controller is None:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers registered (standalone mode)")
        else:
            logger.info("Running under controller - signal handlers skipped")
        
        logger.info("Z-Score bot initialized\n")
    
    def _signal_handler(self, signum, frame) -> None:
        """Only used in standalone mode"""
        logger.info("\n" + "=" * 80)
        logger.info("SHUTDOWN SIGNAL RECEIVED")
        logger.info("=" * 80)
        self.stop()
    
    def start(self) -> None:
        """Set leverage, start data streams, minimal warmup, and enter main loop."""
        try:
            logger.info("=" * 80)
            logger.info("STARTING Z-SCORE IMBALANCE ICEBERG HUNTER BOT")
            logger.info("=" * 80)
            
            logger.info(f"Setting leverage to {config.LEVERAGE}x...")
            lev_res = self.api.set_leverage(
                symbol=config.SYMBOL,
                exchange=config.EXCHANGE,
                leverage=config.LEVERAGE,
            )
            
            if lev_res:
                logger.info(f"Leverage set to {config.LEVERAGE}x")
            else:
                logger.warning("Could not set leverage (may already be set)")
            
            logger.info("Fetching available USDT futures balance...")
            balance_info = self.risk_manager.get_available_balance()
            
            if balance_info:
                available = float(balance_info.get("available", 0.0))
                logger.info(f"Initial Balance: {available:.2f} USDT")
            else:
                logger.warning("Could not fetch balance - using fallback")
                logger.info("Initial Balance: UNKNOWN USDT")
            
            logger.info("Starting Z-Score data manager (WebSocket streams)...")
            if not self.data_manager.start():
                logger.error("Data manager start failed; aborting")
                return
            
            # Register event-driven callbacks
            self._register_event_callbacks()
            
            logger.info("Waiting for first live prices (minimal warmup)...")
            wait_start = time.time()
            
            while self.data_manager.get_last_price() <= 0:
                if time.time() - wait_start > 60:
                    logger.error("Timeout waiting for initial price ticks")
                    break
                time.sleep(0.5)
            
            last_price = self.data_manager.get_last_price()
            
            if last_price > 0:
                logger.info(f"First live price received: {last_price:.2f}")
            else:
                logger.warning(
                    "Proceeding without confirmed first price; "
                    "strategy will self‑gate entries."
                )
            
            # ═══════════════════════════════════════════════════════════
            # DATA WARMUP PERIOD - Prevents trading with bad data
            # ═══════════════════════════════════════════════════════════
            logger.info("=" * 80)
            logger.info("📊 DATA WARMUP - Accumulating market data...")
            # ✅ FIXED: 60s fixed warmup with klines check
            warmup_duration = 60
            start_time = time.time()
            
            while time.time() - start_time < warmup_duration:
                self.data_manager.maybe_train_lstm()  # Train if ready
                if len(self.data_manager.klines_1m) >= 10:  # At least 10 1m candles
                    logger.info(f"Warmup progress: {len(self.data_manager.klines_1m)} candles accumulated")
                time.sleep(5)
            
            logger.info("✅ WARMUP COMPLETE - Bot ready to trade")
            
            # ✅ FIXED: Notify controller AFTER warmup completes
            if self.controller is not None:
                try:
                    self.controller.notify_bot_started()
                    logger.info("✅ Notified controller of successful startup")
                except Exception as e:
                    logger.error(f"Error notifying controller: {e}")
            
            # Enter main loop (from original snippet logic)
            self.running = True
            logger.info("Entering main tick loop...")
            while self.running:
                self.strategy.on_tick(
                    data_manager=self.data_manager,
                    order_manager=self.order_manager,
                    risk_manager=self.risk_manager,
                )
                self._check_stream_health()
                self._maybe_send_telegram_report()
                time.sleep(config.POSITION_CHECK_INTERVAL)
            
            logger.info("Main loop exited normally")
        
        except Exception as e:
            logger.error(f"Fatal error in start(): {e}", exc_info=True)
            
            # ✅ FIXED: Notify controller of failure
            if self.controller is not None:
                try:
                    self.controller._bot_start_failed_event.set()
                    self.controller._bot_start_error = str(e)
                except:
                    pass
            
            self.stop()
    
    def _register_event_callbacks(self) -> None:
        """Register WebSocket event callbacks (from original)."""
        if self._callbacks_registered:
            return
        
        # Register callbacks to strategy
        self.data_manager.on_orderbook_update = lambda data: self.strategy.on_orderbook_update(data)
        self.data_manager.on_trade_update = lambda data: self.strategy.on_trade_update(data)
        self.data_manager.on_candle_update = lambda data: self.strategy.on_candle_update(data)
        
        self._callbacks_registered = True
        logger.info("Event callbacks registered")
    
    def _check_stream_health(self) -> None:
        """Check WebSocket stream health and restart if needed (verbatim from snippet)."""
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
            
            alert_msg = (
                f"⚠️ WEBSOCKET ISSUE DETECTED\n"
                f"No data for {idle_sec:.1f}s (threshold: {WS_IDLE_RESTART_SEC:.0f}s)\n"
                f"Attempting automatic restart..."
            )
            
            try:
                send_telegram_message(alert_msg)
            except:
                pass
            
            logger.warning(
                "=" * 80
                + "\n"
                + f"No data from WebSocket for {idle_sec:.1f}s "
                f"(threshold {WS_IDLE_RESTART_SEC:.0f}s) – restarting data manager...\n"
                + "=" * 80
            )
            
            try:
                self.data_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping data manager: {e}", exc_info=True)
            
            time.sleep(2.0)
            
            restart_success = False
            for attempt in range(3):
                try:
                    logger.info(f"WebSocket restart attempt {attempt + 1}/3...")
                    
                    if self.data_manager.start():
                        # Re-register callbacks after restart
                        self._callbacks_registered = False
                        self._register_event_callbacks()
                        
                        restart_success = True
                        logger.info("Data manager successfully restarted.")
                        
                        success_msg = (
                            f"✅ WEBSOCKET RESTORED\n"
                            f"Reconnected after {idle_sec:.1f}s downtime\n"
                            f"Attempt: {attempt + 1}/3"
                        )
                        
                        try:
                            send_telegram_message(success_msg)
                        except:
                            pass
                        
                        break
                    else:
                        logger.warning(f"Restart attempt {attempt + 1} failed")
                        time.sleep(5.0)
                
                except Exception as e:
                    logger.error(f"Error in restart attempt {attempt + 1}: {e}", exc_info=True)
                    time.sleep(5.0)
            
            if not restart_success:
                error_msg = (
                    f"❌ WEBSOCKET RESTART FAILED\n"
                    f"Could not restore connection after 3 attempts"
                )
                
                try:
                    send_telegram_message(error_msg)
                except:
                    pass
                
                logger.error("Failed to restart WebSocket after 3 attempts.")
        
        except Exception as e:
            logger.error(f"Error in stream health check: {e}", exc_info=True)
    
    def stop(self) -> None:
        """Clean shutdown (verbatim from snippet)."""
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
        logger.info("Z-SCORE BOT STOPPED")
        logger.info("=" * 80)
    
    def _maybe_send_telegram_report(self) -> None:
        """Send periodic Telegram report (verbatim from snippet)."""
        interval = getattr(telegram_config, "TELEGRAM_REPORT_INTERVAL_SEC", 900)
        
        if interval <= 0:
            return
        
        now = time.time()
        
        if now - self._last_report_sec < interval:
            return
        
        self._last_report_sec = now
        
        try:
            last_price = self.data_manager.get_last_price()
            ema = None
            atr = None
            
            try:
                ema = self.data_manager.get_ema(period=config.EMA_PERIOD)
            except:
                pass
            
            try:
                atr = self.data_manager.get_atr_percent(window_minutes=config.ATR_WINDOW_MINUTES)
            except:
                pass
            
            balance_info = self.risk_manager.get_available_balance()
            pos = self.strategy.current_position
            
            lines = [
                "Z-Score BOT 15m Report",
                datetime.utcnow().strftime("Time: %Y-%m-%d %H:%M:%S UTC"),
            ]
            
            if last_price > 0:
                price_line = f"Price: {last_price:.2f}"
                if ema is not None:
                    price_line += f" | EMA{config.EMA_PERIOD}: {ema:.2f}"
                lines.append(price_line)
            
            if balance_info:
                lines.append(f"Balance: {balance_info.get('available', 0.0):.2f} USDT")
            
            if pos:
                dur_min = (now - pos.entry_time_sec) / 60.0
                lines.append(
                    f"Position: {pos.side.upper()} {pos.quantity:.3f} @ {pos.entry_price:.2f} "
                    f"({dur_min:.1f}min)"
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