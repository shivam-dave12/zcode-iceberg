# telegram_notifier.py

import logging
import time
from typing import Optional
from urllib import request, parse

import telegram_config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Thin wrapper around Telegram Bot API sendMessage.

    - No external dependencies (uses stdlib urllib).
    - Completely no-op if token/chat_id are missing.
    """

    def __init__(self) -> None:
        self._token: Optional[str] = telegram_config.TELEGRAM_BOT_TOKEN
        self._chat_id: Optional[str] = telegram_config.TELEGRAM_CHAT_ID
        self.enabled: bool = bool(self._token and self._chat_id)

        if not self.enabled:
            logger.warning(
                "TelegramNotifier disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
                "not set in environment."
            )

        self._base_url: Optional[str] = (
            f"https://api.telegram.org/bot{self._token}/sendMessage"
            if self.enabled
            else None
        )

    def send_message(self, text: str) -> None:
        """
        Fire-and-forget send. Never raises up the stack.
        """
        if not self.enabled or not text:
            return

        try:
            data = parse.urlencode(
                {
                    "chat_id": self._chat_id,
                    "text": text,
                    # Plain text, no parse_mode to avoid escaping headaches
                    "disable_web_page_preview": "true",
                }
            ).encode("utf-8")

            req = request.Request(self._base_url, data=data, method="POST")
            with request.urlopen(req, timeout=5) as resp:
                status = getattr(resp, "status", None)
                if status is not None and status != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    logger.error(
                        "Telegram sendMessage failed: status=%s body=%s",
                        status,
                        body,
                    )
        except Exception as e:
            logger.error("Error sending Telegram message: %s", e, exc_info=True)


_notifier = TelegramNotifier()


def send_telegram_message(text: str) -> None:
    """
    Public helper used by other modules.
    """
    _notifier.send_message(text)


class TelegramLogHandler(logging.Handler):
    """
    Logging handler that forwards WARNING/ERROR/CRITICAL records to Telegram.

    - Skips logs from this module to avoid recursion if Telegram send fails.
    - Has a simple global throttle to prevent message floods.
    """

    def __init__(self, level: int = logging.WARNING, throttle_seconds: float = 5.0) -> None:
        super().__init__(level)
        self.throttle_seconds = max(throttle_seconds, 0.0)
        self._last_sent_ts: float = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        # Avoid recursion on our own internal errors
        if record.name.startswith(__name__):
            return

        now = time.time()
        if self.throttle_seconds and (now - self._last_sent_ts) < self.throttle_seconds:
            # Simple global throttle; set throttle_seconds=0.0 to disable
            return

        self._last_sent_ts = now

        try:
            msg = self.format(record)
            prefix = f"[{record.levelname}] {record.name}: "
            text = prefix + msg

            # Telegram hard limit is ~4096 chars; keep a margin
            max_len = 3800
            if len(text) > max_len:
                text = text[:max_len] + " ... [truncated]"

            send_telegram_message(text)
        except Exception:
            # Never let logging handler throw
            return


def install_global_telegram_log_handler(
    level: int = logging.WARNING,
    throttle_seconds: float = 5.0,
) -> logging.Handler:
    """
    Attach a TelegramLogHandler to the root logger.

    Args:
        level: Minimum level to forward (default WARNING).
        throttle_seconds: Min seconds between Telegram messages from this handler.
                          Use 0.0 to disable throttling completely.
    """
    handler = TelegramLogHandler(level=level, throttle_seconds=throttle_seconds)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    return handler
