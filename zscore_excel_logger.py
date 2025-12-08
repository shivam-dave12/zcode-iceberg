"""
Z-Score Excel Logger - PRODUCTION GRADE

Features:
- Async non-blocking writes (queue-based)
- Thread-safe operations
- Graceful shutdown
- Memory-efficient batching
"""

import logging
import threading
import queue
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass

from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import config

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record for queued logging"""
    trade_id: str
    entry_time: str
    exit_time: str
    duration_minutes: float
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    margin_used: float
    leverage: int
    tp_price: float
    sl_price: float
    entry_imbalance: float
    entry_z_score: float
    entry_wall_volume: float
    exit_reason: str
    pnl_usdt: float
    entry_htf_trend: str = "UNKNOWN"


class ZScoreExcelLogger:
    """
    Async Excel logger with queue-based writes.
    
    All logging operations are queued and executed in background thread.
    Main thread never blocks on I/O.
    """

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = Path(filepath or config.EXCEL_LOG_FILE)
        
        # Queue for async operations
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Thread-safe workbook access
        self._wb_lock = threading.RLock()
        self.wb: Optional[Workbook] = None
        
        # Initialize workbook in main thread
        self._init_workbook()
        
        # Start background worker
        self._start_worker()
        
        logger.info(f"✓ Async Excel logger initialized: {self.filepath}")

    def _init_workbook(self) -> None:
        """Initialize workbook (called in main thread)"""
        try:
            if self.filepath.exists():
                logger.info(f"Opening existing Excel log: {self.filepath}")
                self.wb = load_workbook(str(self.filepath))
            else:
                logger.info(f"Creating new Excel log: {self.filepath}")
                self.wb = Workbook()
                if "Sheet" in self.wb.sheetnames:
                    del self.wb["Sheet"]

            if "Trades" not in self.wb.sheetnames:
                self._init_trades_sheet()
            if "Parameters" not in self.wb.sheetnames:
                self._init_parameters_sheet()
            if "Daily Summary" not in self.wb.sheetnames:
                self._init_summary_sheet()

            # Initial save
            with self._wb_lock:
                self.wb.save(str(self.filepath))
        
        except Exception as e:
            logger.error(f"Error initializing workbook: {e}", exc_info=True)

    def _start_worker(self) -> None:
        """Start background worker thread"""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="ExcelLoggerWorker"
        )
        self._worker_thread.start()
        logger.info("Excel logger worker thread started")

    def _worker_loop(self) -> None:
        """Background worker loop"""
        logger.info("Excel logger worker loop started")
        
        batch = []
        last_save_time = time.time()
        save_interval = 5.0  # Save every 5 seconds
        
        while not self._stop_event.is_set():
            try:
                # Get items from queue (timeout to check stop event)
                try:
                    item = self._queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                # Process batch if needed
                now = time.time()
                should_save = (
                    len(batch) >= 10 or  # Batch size
                    (batch and now - last_save_time >= save_interval) or  # Time interval
                    self._stop_event.is_set()  # Shutdown
                )
                
                if should_save and batch:
                    self._process_batch(batch)
                    batch.clear()
                    last_save_time = now
            
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(1.0)
        
        # Process remaining items on shutdown
        if batch:
            logger.info(f"Processing {len(batch)} remaining items...")
            self._process_batch(batch)
        
        logger.info("Excel logger worker loop stopped")

    def _process_batch(self, batch: List) -> None:
        """Process batch of queued operations"""
        try:
            with self._wb_lock:
                for operation, args in batch:
                    if operation == "log_trade":
                        self._log_trade_sync(*args)
                    elif operation == "log_parameters":
                        self._log_parameters_sync(*args)
                    elif operation == "update_summary":
                        self._update_daily_summary_sync()
                
                # Save workbook
                self.wb.save(str(self.filepath))
                logger.debug(f"Batch processed: {len(batch)} operations")
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

    # ======================================================================
    # PUBLIC API (Async - non-blocking)
    # ======================================================================

    def log_trade(
        self,
        trade_id: str,
        entry_time: str,
        exit_time: str,
        duration_minutes: float,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        margin_used: float,
        leverage: int,
        tp_price: float,
        sl_price: float,
        entry_imbalance: float,
        entry_z_score: float,
        entry_wall_volume: float,
        exit_reason: str,
        pnl_usdt: float,
        entry_htf_trend: str = "UNKNOWN",
    ) -> None:
        """
        Queue trade for logging (non-blocking).
        """
        try:
            record = TradeRecord(
                trade_id=trade_id,
                entry_time=entry_time,
                exit_time=exit_time,
                duration_minutes=duration_minutes,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                margin_used=margin_used,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                entry_imbalance=entry_imbalance,
                entry_z_score=entry_z_score,
                entry_wall_volume=entry_wall_volume,
                exit_reason=exit_reason,
                pnl_usdt=pnl_usdt,
                entry_htf_trend=entry_htf_trend,
            )
            
            self._queue.put_nowait(("log_trade", (record,)))
            logger.debug(f"Trade {trade_id} queued for logging")
        
        except queue.Full:
            logger.error(f"Excel logger queue full - dropping trade {trade_id}")
        except Exception as e:
            logger.error(f"Error queueing trade: {e}")

    def log_parameters(
        self,
        timestamp: str,
        current_price: float,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        decision: str,
        reason: str,
        htf_trend: Optional[str] = None,
    ) -> None:
        """
        Queue parameters for logging (non-blocking).
        """
        try:
            self._queue.put_nowait((
                "log_parameters",
                (timestamp, current_price, imbalance_data, wall_data,
                 delta_data, touch_data, decision, reason, htf_trend)
            ))
            logger.debug("Parameters queued for logging")
        
        except queue.Full:
            logger.warning("Excel logger queue full - dropping parameters")
        except Exception as e:
            logger.error(f"Error queueing parameters: {e}")

    def update_daily_summary(self) -> None:
        """
        Queue daily summary update (non-blocking).
        """
        try:
            self._queue.put_nowait(("update_summary", ()))
        except queue.Full:
            logger.warning("Excel logger queue full - dropping summary update")
        except Exception as e:
            logger.error(f"Error queueing summary: {e}")

    def close(self) -> None:
        """
        Gracefully shutdown logger.
        Waits for queue to drain.
        """
        logger.info("Shutting down Excel logger...")
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for queue to drain (max 10 seconds)
        start = time.time()
        while not self._queue.empty() and time.time() - start < 10:
            time.sleep(0.1)
        
        # Wait for worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        # Final save
        try:
            with self._wb_lock:
                if self.wb:
                    self._update_daily_summary_sync()
                    self.wb.save(str(self.filepath))
                    logger.info(f"Excel log saved: {self.filepath}")
        except Exception as e:
            logger.error(f"Error in final save: {e}")

    # ======================================================================
    # INTERNAL SYNC METHODS (called by worker thread)
    # ======================================================================

    def _log_trade_sync(self, record: TradeRecord) -> None:
        """Internal sync trade logging"""
        try:
            ws = self.wb["Trades"]

            notional = record.entry_price * record.quantity
            pnl_pct = (record.pnl_usdt / notional * 100.0) if notional > 0 else 0.0
            roi_pct = (record.pnl_usdt / record.margin_used * 100.0) if record.margin_used > 0 else 0.0
            cumulative = self._get_cumulative_pnl() + record.pnl_usdt

            row = [
                record.trade_id,
                record.entry_time,
                record.exit_time,
                f"{record.duration_minutes:.1f}",
                record.side.upper(),
                f"{record.entry_price:.2f}",
                f"{record.exit_price:.2f}",
                f"{record.quantity:.6f}",
                f"{record.margin_used:.4f}",
                record.leverage,
                f"{notional:.2f}",
                f"{record.tp_price:.2f}",
                f"{record.sl_price:.2f}",
                f"{record.entry_imbalance:.3f}",
                f"{record.entry_z_score:.2f}",
                f"{record.entry_wall_volume:.4f}",
                record.entry_htf_trend,
                record.exit_reason,
                f"{record.pnl_usdt:.2f}",
                f"{pnl_pct:.2f}",
                f"{roi_pct:.2f}",
                f"{cumulative:.2f}",
            ]

            ws.append(row)
            self._style_data_row(ws, ws.max_row, len(row), is_winning=(record.pnl_usdt > 0))

        except Exception as e:
            logger.error(f"Error in _log_trade_sync: {e}", exc_info=True)

    def _log_parameters_sync(
        self,
        timestamp: str,
        current_price: float,
        imbalance_data: Dict,
        wall_data: Dict,
        delta_data: Dict,
        touch_data: Dict,
        decision: str,
        reason: str,
        htf_trend: Optional[str],
    ) -> None:
        """Internal sync parameters logging"""
        try:
            ws = self.wb["Parameters"]

            row = [
                timestamp,
                f"{current_price:.2f}",
                f"{imbalance_data.get('imbalance', 0):.3f}",
                "YES" if imbalance_data.get("long_ok", False) else "NO",
                "YES" if imbalance_data.get("short_ok", False) else "NO",
                f"{wall_data.get('bid_wall_strength', 0):.2f}",
                f"{wall_data.get('ask_wall_strength', 0):.2f}",
                "YES" if wall_data.get("long_wall_ok", False) else "NO",
                "YES" if wall_data.get("short_wall_ok", False) else "NO",
                f"{delta_data.get('delta', 0):.2f}",
                f"{delta_data.get('z_score', 0):.2f}",
                "YES" if delta_data.get("long_ok", False) else "NO",
                "YES" if delta_data.get("short_ok", False) else "NO",
                f"{touch_data.get('nearest_bid', 0):.2f}",
                f"{touch_data.get('nearest_ask', 0):.2f}",
                f"{touch_data.get('bid_distance_ticks', 0):.2f}",
                f"{touch_data.get('ask_distance_ticks', 0):.2f}",
                "YES" if touch_data.get("long_touch_ok", False) else "NO",
                "YES" if touch_data.get("short_touch_ok", False) else "NO",
                htf_trend or "UNKNOWN",
                decision,
                reason,
            ]

            ws.append(row)
            self._style_data_row(ws, ws.max_row, len(row))

        except Exception as e:
            logger.error(f"Error in _log_parameters_sync: {e}", exc_info=True)

    def _update_daily_summary_sync(self) -> None:
        """Internal sync summary update"""
        try:
            ws_trades = self.wb["Trades"]
            ws_sum = self.wb["Daily Summary"]

            if ws_trades.max_row < 2:
                return

            pnls: List[float] = []
            durations: List[float] = []

            for row in range(2, ws_trades.max_row + 1):
                try:
                    pnls.append(float(ws_trades.cell(row=row, column=19).value or 0))
                    durations.append(float(ws_trades.cell(row=row, column=4).value or 0))
                except (TypeError, ValueError):
                    pass

            if not pnls:
                return

            total_trades = len(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = winning_trades / total_trades * 100.0 if total_trades else 0.0

            total_pnl = sum(pnls)
            max_win = max(wins) if wins else 0.0
            max_loss = min(losses) if losses else 0.0

            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            avg_hold = sum(durations) / len(durations) if durations else 0.0

            date_str = datetime.now().strftime("%Y-%m-%d")

            # Update or append
            existing_row = None
            for r in range(2, ws_sum.max_row + 1):
                if (ws_sum.cell(row=r, column=1).value or "") == date_str:
                    existing_row = r
                    break

            row_values = [
                date_str,
                total_trades,
                winning_trades,
                losing_trades,
                f"{win_rate:.1f}%",
                f"{total_pnl:.2f}",
                f"{max_win:.2f}",
                f"{max_loss:.2f}",
                f"{avg_win:.2f}",
                f"{avg_loss:.2f}",
                f"{profit_factor:.2f}",
                f"{avg_hold:.1f}",
            ]

            if existing_row:
                for c, val in enumerate(row_values, 1):
                    ws_sum.cell(row=existing_row, column=c, value=val)
                self._style_data_row(ws_sum, existing_row, len(row_values))
            else:
                ws_sum.append(row_values)
                self._style_data_row(ws_sum, ws_sum.max_row, len(row_values))

        except Exception as e:
            logger.error(f"Error updating daily summary: {e}", exc_info=True)

    def _get_cumulative_pnl(self) -> float:
        """Get cumulative P&L (must be called with lock held)"""
        try:
            ws = self.wb["Trades"]
            total = 0.0
            for row in range(2, ws.max_row + 1):
                val = ws.cell(row=row, column=19).value
                try:
                    total += float(val)
                except (TypeError, ValueError):
                    pass
            return total
        except Exception:
            return 0.0

    # ======================================================================
    # Sheet initialization (unchanged from original)
    # ======================================================================

    def _init_trades_sheet(self) -> None:
        """Initialize trades sheet"""
        ws = self.wb.create_sheet("Trades")
        headers = [
            "Trade ID", "Entry Time", "Exit Time", "Duration (min)", "Side",
            "Entry Price", "Exit Price", "Quantity (BTC)", "Margin Used (USDT)",
            "Leverage", "Notional (USDT)", "TP Price", "SL Price",
            "Entry Imbalance", "Entry Z-Score", "Entry Wall Volume",
            "Entry HTF Trend", "Exit Reason", "P&L (USDT)", "P&L %",
            "ROI % (on margin)", "Cumulative P&L (USDT)",
        ]
        ws.append(headers)
        self._style_header_row(ws, len(headers))

        widths = [16, 19, 19, 14, 8, 14, 14, 16, 16, 10, 16, 14, 14, 16, 16, 18, 16, 20, 14, 10, 16, 20]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def _init_parameters_sheet(self) -> None:
        """Initialize parameters sheet"""
        ws = self.wb.create_sheet("Parameters")
        headers = [
            "Timestamp", "Current Price", "Imbalance", "Long Imb OK", "Short Imb OK",
            "Bid Wall Str", "Ask Wall Str", "Long Wall OK", "Short Wall OK",
            "Delta", "Z-Score", "Long Z OK", "Short Z OK",
            "Nearest Bid", "Nearest Ask", "Bid Dist Ticks", "Ask Dist Ticks",
            "Long Touch OK", "Short Touch OK", "HTF Trend", "Decision", "Reason",
        ]
        ws.append(headers)
        self._style_header_row(ws, len(headers))

        for i in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(i)].width = 18

    def _init_summary_sheet(self) -> None:
        """Initialize summary sheet"""
        ws = self.wb.create_sheet("Daily Summary")
        headers = [
            "Date", "Total Trades", "Winning Trades", "Losing Trades", "Win Rate %",
            "Total P&L (USDT)", "Max Win (USDT)", "Max Loss (USDT)",
            "Avg Win (USDT)", "Avg Loss (USDT)", "Profit Factor", "Avg Hold Time (min)",
        ]
        ws.append(headers)
        self._style_header_row(ws, len(headers))

        for i in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(i)].width = 18

    @staticmethod
    def _style_header_row(ws, num_cols: int) -> None:
        """Style header row"""
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        for col in range(1, num_cols + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
            cell.border = border

    @staticmethod
    def _style_data_row(ws, row: int, num_cols: int, is_winning: Optional[bool] = None) -> None:
        """Style data row"""
        border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        if is_winning is True:
            fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
            font = Font(color="006100")
        elif is_winning is False:
            fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
            font = Font(color="9C0006")
        else:
            fill = None
            font = Font()

        for col in range(1, num_cols + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = border
            if fill:
                cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(horizontal="center", vertical="center")


# Add missing import
import time