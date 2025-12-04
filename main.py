# main.py
"""
Main execution file for Z-Score bot.

Key fixes applied:
- Warmup flow robust: 60s warmup, safe kline count check via data_manager.klines_1m
- Controller notified AFTER warmup completes via controller.notify_bot_started()
- Event-driven callback registration preserved
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
        logging.FileHandler(f"zscore_iceberg_hunter_{datetime.utcnow().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

install_global_telegram_log_handler(level=logging.WARNING, throttle_seconds=5.0)
logger = logging.getLogger(__name__)

WS_IDLE_RESTART_SEC = 20.0

class ZScoreIcebergBot:
    def __init__(self, controller=None) -> None:
        logger.info("Z-SCORE ICEBERG BOT INIT")
        self.controller = controller
        self.api = FuturesAPI(api_key=config.COINSWITCH_API_KEY, secret_key=config.COINSWITCH_SECRET_KEY)
        self.data_manager = ZScoreDataManager()
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()
        self.excel_logger = ZScoreExcelLogger(filepath=f"zscore_iceberg_hunter_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx") if config.ENABLE_EXCEL_LOGGING else None
        self.strategy = ZScoreIcebergHunterStrategy(excel_logger=self.excel_logger)
        self.running = False
        self._callbacks_registered = False

        if controller is None:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info("SHUTDOWN SIGNAL")
        self.stop()

    def start(self) -> None:
        try:
            logger.info("STARTING BOT")
            # set leverage
            lev_res = self.api.set_leverage(symbol=config.SYMBOL, exchange=config.EXCHANGE, leverage=config.LEVERAGE)
            logger.info(f"Leverage set result: {lev_res}")

            balance_info = self.risk_manager.get_available_balance()
            if balance_info:
                logger.info(f"Initial Balance: {float(balance_info.get('available',0.0)):.2f}")
            else:
                logger.warning("Could not fetch initial balance")

            if not self.data_manager.start():
                logger.error("Data manager failed to start")
                return

            # register event callbacks for sub-50ms decisions
            self._register_event_callbacks()

            # Wait for first price
            logger.info("Waiting for first live price...")
            wait_start = time.time()
            while self.data_manager.get_last_price() <= 0:
                if time.time() - wait_start > 60:
                    logger.error("Timeout waiting for initial price ticks")
                    break
                time.sleep(0.5)
            last_price = self.data_manager.get_last_price()
            if last_price > 0:
                logger.info(f"First live price received: {last_price:.2f}")

            # Warmup period
            logger.info("DATA WARMUP - accumulating market data...")
            warmup_duration = 60
            start_time = time.time()
            while time.time() - start_time < warmup_duration:
                elapsed = time.time() - start_time
                vol_regime, atr_pct = self.data_manager.get_vol_regime()
                htf_trend = self.data_manager.get_htf_trend() if hasattr(self.data_manager, "get_htf_trend") else None
                kline_count = 0
                try:
                    kline_count = len(self.data_manager.klines_1m)
                except Exception:
                    kline_count = 0
                logger.info(f"Warmup {elapsed:.0f}/{warmup_duration}s | Bars: {kline_count} | Vol: {vol_regime} | HTF: {htf_trend or 'NA'}")
                if elapsed >= 40 and kline_count >= 40 and vol_regime != "UNKNOWN" and htf_trend:
                    logger.info("Data ready early - proceeding")
                    break
                time.sleep(5)

            logger.info("✅ WARMUP COMPLETE - Bot ready to trade")

            # Notify controller AFTER warmup completes
            if self.controller is not None:
                try:
                    self.controller.notify_bot_started()
                    logger.info("Notified controller of successful startup")
                except Exception as e:
                    logger.error(f"Error notifying controller: {e}")

            self.running = True
            self._run_main_loop()

        except Exception as e:
            logger.error(f"Error starting bot: {e}", exc_info=True)
            if self.controller is not None:
                try:
                    self.controller._bot_start_failed_event.set()
                    self.controller._bot_start_error = str(e)
                except Exception:
                    pass
            self.stop()

    def _register_event_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        try:
            ws = self.data_manager.ws
            if not ws:
                logger.warning("WS not ready; cannot register callbacks")
                return

            orig_ob = self.data_manager._on_orderbook
            orig_tr = self.data_manager._on_trade

            def enhanced_orderbook_callback(data):
                try:
                    orig_ob(data)
                    self._on_data_update()
                except Exception as e:
                    logger.error(f"enhanced_orderbook_callback error: {e}")

            def enhanced_trade_callback(data):
                try:
                    orig_tr(data)
                    self._on_data_update()
                except Exception as e:
                    logger.error(f"enhanced_trade_callback error: {e}")

            # replace callbacks and re-subscribe
            self.data_manager._on_orderbook = enhanced_orderbook_callback
            self.data_manager._on_trade = enhanced_trade_callback
            ws.subscribe_orderbook(config.SYMBOL, callback=enhanced_orderbook_callback)
            ws.subscribe_trades(config.SYMBOL, callback=enhanced_trade_callback)
            self._callbacks_registered = True
            logger.info("Event-driven callbacks registered")
        except Exception as e:
            logger.error(f"Error registering callbacks: {e}", exc_info=True)

    def _on_data_update(self) -> None:
        if not self.running:
            return
        try:
            self.strategy.on_tick(self.data_manager, self.order_manager, self.risk_manager)
        except Exception as e:
            logger.error(f"Error in strategy on_tick: {e}", exc_info=True)

    def _run_main_loop(self) -> None:
        logger.info("Bot running (health loop)")
        while self.running:
            try:
                # health checks
                last_update = self.data_manager.stats.get("last_update")
                if last_update:
                    idle_sec = (datetime.utcnow() - last_update).total_seconds()
                    if idle_sec > WS_IDLE_RESTART_SEC:
                        logger.warning("Websocket idle detected - restarting data manager")
                        self.data_manager.stop()
                        time.sleep(1.0)
                        self.data_manager.start()
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                time.sleep(1.0)

    def stop(self) -> None:
        logger.info("Stopping bot")
        try:
            self.running = False
            try:
                self.data_manager.stop()
            except Exception:
                pass
        finally:
            logger.info("Bot stopped")

