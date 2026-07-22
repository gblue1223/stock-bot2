import os
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class ProgramTradeRecorder:
    """프로그램 매매 수급 틱 데이터 및 주문 집행 기록을 저장하는 레코더"""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(project_root, "data", "trading_logs")
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger("ProgramTradeRecorder")

        today_str = datetime.today().strftime("%Y%m%d")
        self.ticks_file = os.path.join(self.log_dir, f"program_ticks_{today_str}.csv")
        self.orders_file = os.path.join(self.log_dir, f"order_history_{today_str}.csv")

        self._init_files()

    def _init_files(self):
        """CSV 헤더 생성"""
        if not os.path.exists(self.ticks_file):
            with open(self.ticks_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "stock_code", "current_price", "net_buy_amt", 
                    "net_buy_irds", "net_buy_qty", "sell_amt", "buy_amt", 
                    "signal", "holding_qty", "executed_steps"
                ])

        if not os.path.exists(self.orders_file):
            with open(self.orders_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "stock_code", "action", "reason", "quantity", 
                    "price", "total_amount", "order_no", "return_code", "return_msg", 
                    "holding_qty_after", "avg_price_after", "pnl_amount", "pnl_pct"
                ])

    def log_tick(
        self, 
        stock_code: str, 
        current_price: float, 
        prog_data: Dict[str, Any], 
        signal_str: str, 
        holding_qty: int, 
        executed_steps: int
    ):
        """수급 틱 데이터 기록"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            now_str,
            stock_code,
            current_price,
            prog_data.get("net_buy_amt", 0.0),
            prog_data.get("net_buy_irds", 0.0),
            prog_data.get("net_buy_qty", 0.0),
            prog_data.get("sell_amt", 0.0),
            prog_data.get("buy_amt", 0.0),
            signal_str,
            holding_qty,
            executed_steps
        ]
        try:
            with open(self.ticks_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            self.logger.error(f"Failed to write tick log: {e}")

    def log_order(
        self,
        stock_code: str,
        action: str,
        reason: str,
        quantity: int,
        price: float,
        order_no: str = "",
        return_code: Any = "",
        return_msg: str = "",
        holding_qty_after: int = 0,
        avg_price_after: float = 0.0,
        pnl_amount: float = 0.0,
        pnl_pct: float = 0.0
    ):
        """주문 집행 기록"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_amount = quantity * price
        row = [
            now_str,
            stock_code,
            action,
            reason,
            quantity,
            price,
            total_amount,
            order_no,
            return_code,
            return_msg,
            holding_qty_after,
            avg_price_after,
            pnl_amount,
            pnl_pct
        ]
        try:
            with open(self.orders_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.logger.info(f"💾 [{stock_code}] 주문 내역 저장 완료: {self.orders_file}")
        except Exception as e:
            self.logger.error(f"Failed to write order log: {e}")
