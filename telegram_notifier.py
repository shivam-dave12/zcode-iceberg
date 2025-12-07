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
        """Fire-and-forget send. Never raises up the stack."""
        if not self.enabled or not text:
            return

        try:
            data = parse.urlencode(
                {
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",  # Enable HTML formatting
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
    """Public helper used by other modules."""
    _notifier.send_message(text)


# ============================================================================
# FORMATTED MESSAGE BUILDERS - No Bullshit, Pure Data
# ============================================================================

def format_entry_message(
    trade_id: str,
    side: str,
    entry_type: str,
    current_price: float,
    limit_price: float,
    quantity: float,
    margin: float,
    leverage: int,
    tp_price: float,
    tp_roi: float,
    sl_price: float,
    sl_roi: float,
    session: str,
    entry_score: float,
    imbalance: float,
    z_score: float,
    wall_strength: float,
    cooldown_sec: float,
) -> str:
    """
    Entry notification - pure data, no fluff.
    """
    direction = "🟢" if side == "long" else "🔴"
    
    msg = (
        f"{direction} <b>{side.upper()} {entry_type}</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>ENTRY</b>\n"
        f"  Current: ${current_price:,.2f}\n"
        f"  Limit: ${limit_price:,.2f}\n"
        f"  Qty: {quantity:.6f} BTC\n"
        f"  Margin: ${margin:.2f} ({leverage}x)\n"
        f"  Value: ${quantity * limit_price:,.2f}\n"
        f"\n"
        f"<b>TARGETS</b>\n"
        f"  TP: ${tp_price:,.2f} (+{tp_roi*100:.1f}%)\n"
        f"  SL: ${sl_price:,.2f} (-{sl_roi*100:.1f}%)\n"
        f"  R:R: {tp_roi/sl_roi:.2f}:1\n"
        f"\n"
        f"<b>SIGNALS</b>\n"
        f"  Score: {entry_score:.3f}\n"
        f"  Imb: {imbalance:+.3f}\n"
        f"  Z: {z_score:+.2f}\n"
        f"  Wall: {wall_strength:.1f}x\n"
        f"\n"
        f"<b>SESSION</b>\n"
        f"  {session}\n"
        f"  Next: {cooldown_sec:.0f}s cooldown"
    )
    
    return msg


def format_fill_message(
    trade_id: str,
    side: str,
    fill_price: float,
    quantity: float,
    tp_price: float,
    sl_price: float,
) -> str:
    """Order filled notification."""
    direction = "🟢" if side == "long" else "🔴"
    
    msg = (
        f"{direction} <b>FILLED</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"  Entry: ${fill_price:,.2f}\n"
        f"  Qty: {quantity:.6f} BTC\n"
        f"  TP: ${tp_price:,.2f}\n"
        f"  SL: ${sl_price:,.2f}"
    )
    
    return msg


def format_exit_message(
    trade_id: str,
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    margin: float,
    pnl: float,
    roi: float,
    hold_min: float,
    exit_reason: str,
    session: str,
) -> str:
    """Exit notification - focus on P&L and performance."""
    profit = pnl > 0
    icon = "✅" if profit else "❌"
    color = "🟢" if profit else "🔴"
    
    price_change = ((exit_price - entry_price) / entry_price) * 100
    if side == "short":
        price_change = -price_change
    
    msg = (
        f"{icon} <b>{exit_reason}</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>TRADE</b>\n"
        f"  {side.upper()} {quantity:.6f} BTC\n"
        f"  Entry: ${entry_price:,.2f}\n"
        f"  Exit: ${exit_price:,.2f}\n"
        f"  Move: {price_change:+.2f}%\n"
        f"\n"
        f"<b>P&L</b>\n"
        f"  {color} ${pnl:+.2f} ({roi:+.1f}%)\n"
        f"  Margin: ${margin:.2f}\n"
        f"\n"
        f"<b>TIME</b>\n"
        f"  Hold: {hold_min:.1f}m\n"
        f"  Session: {session}"
    )
    
    return msg


def format_position_update(
    trade_id: str,
    side: str,
    entry_price: float,
    current_price: float,
    quantity: float,
    upnl: float,
    upnl_pct: float,
    hold_min: float,
    tp_price: float,
    sl_price: float,
) -> str:
    """Position update notification."""
    direction = "🟢" if side == "long" else "🔴"
    profit = upnl > 0
    pnl_icon = "📈" if profit else "📉"
    
    msg = (
        f"{direction} <b>POSITION UPDATE</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>CURRENT</b>\n"
        f"  {side.upper()} {quantity:.6f} BTC\n"
        f"  Entry: ${entry_price:,.2f}\n"
        f"  Now: ${current_price:,.2f}\n"
        f"  Hold: {hold_min:.1f}m\n"
        f"\n"
        f"<b>P&L</b>\n"
        f"  {pnl_icon} ${upnl:+.2f} ({upnl_pct:+.1f}%)\n"
        f"\n"
        f"<b>TARGETS</b>\n"
        f"  TP: ${tp_price:,.2f}\n"
        f"  SL: ${sl_price:,.2f}"
    )
    
    return msg


def format_tp_adjustment(
    trade_id: str,
    old_tp: float,
    new_tp: float,
    old_roi: float,
    new_roi: float,
    reason: str,
) -> str:
    """TP adjustment notification."""
    msg = (
        f"🎯 <b>TP ADJUSTED</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"  Old: ${old_tp:,.2f} ({old_roi*100:.1f}%)\n"
        f"  New: ${new_tp:,.2f} ({new_roi*100:.1f}%)\n"
        f"\n"
        f"  Reason: {reason}"
    )
    
    return msg


def format_sl_adjustment(
    trade_id: str,
    old_sl: float,
    new_sl: float,
    reason: str,
) -> str:
    """SL adjustment notification."""
    msg = (
        f"🛡️ <b>SL ADJUSTED</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"  Old: ${old_sl:,.2f}\n"
        f"  New: ${new_sl:,.2f}\n"
        f"\n"
        f"  Reason: {reason}"
    )
    
    return msg


def format_status_report(
    current_price: float,
    balance: float,
    total_trades: int,
    win_rate: float,
    daily_pnl: float,
    total_pnl: float,
    position_side: Optional[str] = None,
    position_qty: Optional[float] = None,
    position_entry: Optional[float] = None,
    position_upnl: Optional[float] = None,
) -> str:
    """Status report - pure stats, no narrative."""
    pos_icon = "🟢" if position_side == "long" else "🔴" if position_side == "short" else "⚪"
    pnl_icon = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "➖"
    
    msg = (
        f"📊 <b>STATUS REPORT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>MARKET</b>\n"
        f"  BTC: ${current_price:,.2f}\n"
        f"  Balance: ${balance:.2f}\n"
        f"\n"
        f"<b>STATS</b>\n"
        f"  Trades: {total_trades}\n"
        f"  Win Rate: {win_rate:.1f}%\n"
        f"  Daily: {pnl_icon} ${daily_pnl:+.2f}\n"
        f"  Total: ${total_pnl:+.2f}\n"
    )
    
    if position_side:
        msg += (
            f"\n"
            f"<b>POSITION</b>\n"
            f"  {pos_icon} {position_side.upper()} {position_qty:.6f} BTC\n"
            f"  Entry: ${position_entry:,.2f}\n"
            f"  uPnL: ${position_upnl:+.2f}"
        )
    else:
        msg += "\n<b>POSITION</b>\n  None"
    
    return msg


def format_error_alert(error_type: str, details: str) -> str:
    """Error alert - critical issues only."""
    msg = (
        f"⚠️ <b>ERROR ALERT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Type:</b> {error_type}\n"
        f"<b>Details:</b> {details}"
    )
    
    return msg


def format_order_status_unknown(
    trade_id: str,
    side: str,
    order_id: str,
    assumed_price: float,
) -> str:
    """Order status unknown - requires manual verification."""
    msg = (
        f"⚠️ <b>STATUS UNKNOWN</b> #{trade_id}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"  Side: {side.upper()}\n"
        f"  Order: {order_id[-12:]}\n"
        f"\n"
        f"<b>ACTION TAKEN</b>\n"
        f"  Assumed filled @ ${assumed_price:,.2f}\n"
        f"  TP/SL kept active\n"
        f"\n"
        f"⚠️ <b>MANUAL CHECK REQUIRED</b>"
    )
    
    return msg


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
            return

        self._last_sent_ts = now

        try:
            msg = self.format(record)
            
            # Format as error alert
            error_msg = format_error_alert(
                error_type=f"{record.levelname}",
                details=f"{record.name}: {msg[:500]}"  # Limit length
            )
            
            send_telegram_message(error_msg)
        except Exception:
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