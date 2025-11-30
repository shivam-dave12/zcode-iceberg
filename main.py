"""
Z-SCORE IMBALANCE ICEBERG HUNTER - Main Execution

High-frequency BTCUSDT futures scalper on CoinSwitch PRO.

Components:
- FuturesAPI / FuturesWebSocket (existing plugins)
- ZScoreDataManager (depth + trades + ticker)
- ZScoreIcebergHunterStrategy (strategy.py)
- RiskManager / OrderManager (existing modules)
- ZScoreExcelLogger (Excel logging)
"""

import time
import signal
import sys
import logging
from datetime import datetime

from futures_api import FuturesAPI
from futures_websocket import FuturesWebSocket  # imported for completeness

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

# Install global Telegram log handler for all WARNING+ logs
install_global_telegram_log_handler(
    level=logging.WARNING,     # WARNING, ERROR, CRITICAL
    throttle_seconds=5.0,      # set to 0.0 if you want zero throttling
)

logger = logging.getLogger(__name__)


def _handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    """
    Global hook for uncaught exceptions so they also go through logging/Telegram.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Let KeyboardInterrupt behave normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


sys.excepthook = _handle_uncaught_exception

# Seconds without any data_manager update before we treat it as a disconnect
WS_IDLE_RESTART_SEC = 20.0  # you can tune this if needed


class ZScoreIcebergBot:
    """
    Main bot class for Z-Score Imbalance Iceberg Hunter.
    """

    def __init__(self) -> None:
        logger.info("=" * 80)
        logger.info("Z-SCORE IMBALANCE ICEBERG HUNTER BOT")
        logger.info("=" * 80)

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

        # Track last times to avoid thrashing
        self._last_stream_check_sec: float = 0.0
        self._last_report_sec: float = 0.0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Z-Score bot initialized\n")

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def _signal_handler(self, signum, frame) -> None:
        logger.info("\n" + "=" * 80)
        logger.info("SHUTDOWN SIGNAL RECEIVED")
        logger.info("=" * 80)
        self.stop()

    def start(self) -> None:
        """
        Set leverage, start data streams, minimal warmup, and enter main loop.
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING Z-SCORE IMBALANCE ICEBERG HUNTER BOT")
            logger.info("=" * 80)

            # 1) Set leverage
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

            # 2) Fetch initial balance
            logger.info("Fetching available USDT futures balance...")
            balance_info = self.risk_manager.get_available_balance()
            if balance_info:
                logger.info(
                    f"Initial Balance: {balance_info['available']:.2f} "
                    f"{balance_info['currency']}"
                )
            else:
                logger.error("Could not fetch balance; aborting")
                return

            # 3) Start data manager (WebSocket + REST warmup)
            logger.info("Starting Z-Score data manager (WebSocket streams)...")
            if not self.data_manager.start():
                logger.error("Data manager start failed; aborting")
                return

            # 4) Minimal warmup: wait for first price
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

            logger.info(
                "Entering Z-Score main loop. Strategy will only trade "
                "when imbalance, wall, delta, and touch conditions all align."
            )

            self.running = True
            self._run_main_loop()

        except Exception as e:
            logger.error(f"Error starting Z-Score bot: {e}", exc_info=True)
            self.stop()

    def _run_main_loop(self) -> None:
        """
        High-frequency loop calling strategy.on_tick and monitoring stream health.
        """
        logger.info("=" * 80)
        logger.info("BOT RUNNING - Z-SCORE IMBALANCE ICEBERG HUNTER ACTIVE")
        logger.info("=" * 80)

        while self.running:
            try:
                self.strategy.on_tick(
                    data_manager=self.data_manager,
                    order_manager=self.order_manager,
                    risk_manager=self.risk_manager,
                )

                self._check_stream_health()
                self._maybe_send_telegram_report()

                time.sleep(config.POSITION_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1.0)

    def _check_stream_health(self) -> None:
        """
        Monitor data_manager stats and restart streams (with full REST warmup)
        if we see no updates for WS_IDLE_RESTART_SEC seconds.

        Adds one extra safeguard: if data_manager.start() fails, attempts a raw
        WebSocket reconnect once via data_manager.ws.connect().
        """
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

            logger.warning(
                "=" * 80
                + "\n"
                + f"No data from WebSocket for {idle_sec:.1f}s "
                f"(threshold {WS_IDLE_RESTART_SEC:.0f}s) – restarting data manager "
                f"with fresh REST warmup for HTF trend and ATR...\n"
                + "=" * 80
            )

            try:
                self.data_manager.stop()
            except Exception as e:
                logger.error(
                    f"Error stopping data manager during restart: {e}",
                    exc_info=True,
                )

            time.sleep(1.0)

            # Restart streams + REST warmup
            if not self.data_manager.start():
                logger.error(
                    "Data manager restart failed; attempting raw WS reconnect..."
                )
                try:
                    if self.data_manager.ws and not self.data_manager.ws.connect():
                        time.sleep(10)
                except Exception as e:
                    logger.error(
                        f"Raw WS reconnect attempt failed: {e}",
                        exc_info=True,
                    )
                    return

            logger.info(
                "Data manager successfully restarted. "
                "Historical klines reloaded for HTF trend and ATR filters."
            )

        except Exception as e:
            logger.error(f"Error in WebSocket stream health check: {e}", exc_info=True)

    def stop(self) -> None:
        """
        Clean shutdown.
        """
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
        sys.exit(0)

    def _maybe_send_telegram_report(self) -> None:
        """
        Send a compact market + state snapshot to Telegram every
        TELEGRAM_REPORT_INTERVAL_SEC seconds (default 15 minutes).
        """
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
            except Exception:
                ema = None

            try:
                atr = self.data_manager.get_atr_percent(
                    window_minutes=config.ATR_WINDOW_MINUTES
                )
            except Exception:
                atr = None

            htf_trend = None
            ltf_trend = None

            try:
                if hasattr(self.data_manager, "get_htf_trend"):
                    htf_trend = self.data_manager.get_htf_trend()
            except Exception:
                htf_trend = None

            try:
                if hasattr(self.data_manager, "get_ltf_trend"):
                    ltf_trend = self.data_manager.get_ltf_trend()
            except Exception:
                ltf_trend = None

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

            if atr is not None:
                lines.append(
                    f"ATR{config.ATR_WINDOW_MINUTES}m: {atr * 100:.2f}%"
                )

            lines.append(
                f"HTF Trend: {htf_trend or 'N/A'} | "
                f"LTF Trend: {ltf_trend or 'N/A'}"
            )

            if balance_info:
                lines.append(
                    "Balance: "
                    f"{float(balance_info.get('available', 0.0)):.2f} "
                    f"{balance_info.get('currency', 'USDT')}"
                )

            if pos is not None:
                dur_min = (now - pos.entry_time_sec) / 60.0
                lines.append(
                    f"Open Position: {pos.side.upper()} "
                    f"qty={pos.quantity:.3f} "
                    f"entry={pos.entry_price:.2f} "
                    f"TP={pos.tp_price:.2f} SL={pos.sl_price:.2f} "
                    f"for {dur_min:.1f} min"
                )

            send_telegram_message("\n".join(lines))
        except Exception:
            logger.exception("Failed to send Telegram 15m report")


if __name__ == "__main__":
    try:
        bot = ZScoreIcebergBot()
        bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
