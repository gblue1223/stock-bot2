"""
실시간 거래 로깅 모듈

이 모듈은 실시간 거래 중 모든 거래를 기록합니다.
타임스탬프, 진입/청산 가격, 보유 기간, 손익 등을 로깅합니다.
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    실시간 거래 로거
    
    모든 거래를 기록하고 다양한 형식으로 저장합니다:
    - JSON 파일: 구조화된 거래 데이터
    - CSV 파일: 표 형식의 거래 데이터
    - 로그 파일: 텍스트 형식의 거래 로그
    
    Args:
        log_dir: 로그 파일을 저장할 디렉토리
        session_name: 거래 세션 이름 (기본값: 현재 타임스탬프)
        enable_json: JSON 로깅 활성화 (기본값: True)
        enable_csv: CSV 로깅 활성화 (기본값: True)
        enable_text: 텍스트 로깅 활성화 (기본값: True)
    """
    
    def __init__(
        self,
        log_dir: str,
        session_name: Optional[str] = None,
        enable_json: bool = True,
        enable_csv: bool = True,
        enable_text: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 세션 이름 설정
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        
        # 로깅 옵션
        self.enable_json = enable_json
        self.enable_csv = enable_csv
        self.enable_text = enable_text
        
        # 거래 기록 저장
        self.trades: List[Dict[str, Any]] = []
        
        # 파일 경로 설정
        self.json_path = self.log_dir / f"trades_{session_name}.json"
        self.csv_path = self.log_dir / f"trades_{session_name}.csv"
        self.text_path = self.log_dir / f"trades_{session_name}.log"
        
        # CSV 파일 초기화 (헤더 작성)
        if self.enable_csv:
            self._initialize_csv()
        
        # 텍스트 로거 설정
        if self.enable_text:
            self._setup_text_logger()
        
        logger.info(f"TradeLogger initialized: session={session_name}, log_dir={log_dir}")
    
    def _initialize_csv(self):
        """CSV 파일 초기화 및 헤더 작성"""
        headers = [
            'timestamp',
            'stock_code',
            'entry_timestamp',
            'entry_price',
            'exit_timestamp',
            'exit_price',
            'holding_time_seconds',
            'profit_rate',
            'profit_amount',
            'transaction_cost',
            'net_profit',
            'action',
            'position_size',
            'metadata'
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        
        logger.info(f"CSV file initialized: {self.csv_path}")
    
    def _setup_text_logger(self):
        """텍스트 로거 설정"""
        self.text_logger = logging.getLogger(f"trade_logger_{self.session_name}")
        self.text_logger.setLevel(logging.INFO)
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(self.text_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.text_logger.addHandler(file_handler)
        
        logger.info(f"Text logger initialized: {self.text_path}")
    
    def log_trade(
        self,
        stock_code: str,
        entry_timestamp: str,
        entry_price: float,
        exit_timestamp: str,
        exit_price: float,
        holding_time: float,
        profit_rate: float,
        transaction_cost: float = 0.0043,
        position_size: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        거래 기록
        
        Args:
            stock_code: 종목 코드
            entry_timestamp: 진입 타임스탬프 (ISO 형식 또는 문자열)
            entry_price: 진입 가격
            exit_timestamp: 청산 타임스탬프 (ISO 형식 또는 문자열)
            exit_price: 청산 가격
            holding_time: 보유 시간 (초)
            profit_rate: 수익률 (소수)
            transaction_cost: 거래 비용 비율 (기본값: 0.0043 = 0.43%)
            position_size: 포지션 크기 (기본값: 1.0)
            metadata: 추가 메타데이터 (선택)
        """
        # 현재 타임스탬프
        current_timestamp = datetime.now().isoformat()
        
        # 수익 계산
        profit_amount = (exit_price - entry_price) * position_size
        net_profit = profit_amount - (entry_price * position_size * transaction_cost)
        
        # 거래 데이터 구성
        trade_data = {
            'timestamp': current_timestamp,
            'stock_code': stock_code,
            'entry_timestamp': entry_timestamp,
            'entry_price': float(entry_price),
            'exit_timestamp': exit_timestamp,
            'exit_price': float(exit_price),
            'holding_time_seconds': float(holding_time),
            'profit_rate': float(profit_rate),
            'profit_amount': float(profit_amount),
            'transaction_cost': float(transaction_cost),
            'net_profit': float(net_profit),
            'action': 'SELL' if exit_price > entry_price else 'SELL_LOSS',
            'position_size': float(position_size),
            'metadata': metadata or {}
        }
        
        # 메모리에 저장
        self.trades.append(trade_data)
        
        # JSON 로깅
        if self.enable_json:
            self._log_to_json(trade_data)
        
        # CSV 로깅
        if self.enable_csv:
            self._log_to_csv(trade_data)
        
        # 텍스트 로깅
        if self.enable_text:
            self._log_to_text(trade_data)
        
        logger.debug(f"Trade logged: {stock_code}, profit_rate={profit_rate:.4f}, "
                    f"holding_time={holding_time:.2f}s")
    
    def _log_to_json(self, trade_data: Dict[str, Any]):
        """JSON 파일에 거래 기록 (Optimized: Append-only JSONL)"""
        # 성능 최적화: 전체 파일을 읽고 다시 쓰는 대신, JSON Lines(ndjson) 형식으로 추가합니다.
        # 유효한 JSON 배열은 save_summary()에서 최종적으로 생성됩니다.
        
        # 파일 확장자가 .json이어도 내용은 줄 단위 JSON 객체로 저장 (Crash recovery용)
        try:
            with open(self.json_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trade_data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to append to JSON log: {e}")
    
    def _log_to_csv(self, trade_data: Dict[str, Any]):
        """CSV 파일에 거래 기록"""
        # 메타데이터를 JSON 문자열로 변환
        csv_data = trade_data.copy()
        csv_data['metadata'] = json.dumps(csv_data['metadata'], ensure_ascii=False)
        
        # CSV 파일에 추가
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data.keys())
            writer.writerow(csv_data)
    
    def _log_to_text(self, trade_data: Dict[str, Any]):
        """텍스트 파일에 거래 기록"""
        log_message = (
            f"Trade: {trade_data['stock_code']} | "
            f"Entry: {trade_data['entry_price']:.2f} @ {trade_data['entry_timestamp']} | "
            f"Exit: {trade_data['exit_price']:.2f} @ {trade_data['exit_timestamp']} | "
            f"Holding: {trade_data['holding_time_seconds']:.2f}s | "
            f"Profit Rate: {trade_data['profit_rate']:.4f} | "
            f"Net Profit: {trade_data['net_profit']:.2f}"
        )
        
        self.text_logger.info(log_message)
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        모든 거래 기록 반환
        
        Returns:
            거래 기록 리스트
        """
        return self.trades.copy()
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        거래 기록을 DataFrame으로 반환
        
        Returns:
            거래 기록 DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        거래 요약 통계 반환
        
        Returns:
            요약 통계 딕셔너리
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit_rate': 0.0,
                'avg_holding_time': 0.0,
                'total_profit': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0
            }
        
        df = self.get_trades_df()
        
        return {
            'total_trades': len(df),
            'win_rate': float((df['profit_rate'] > 0).mean()),
            'avg_profit_rate': float(df['profit_rate'].mean()),
            'avg_holding_time': float(df['holding_time_seconds'].mean()),
            'total_profit': float(df['net_profit'].sum()),
            'max_profit': float(df['net_profit'].max()),
            'max_loss': float(df['net_profit'].min()),
            'avg_net_profit': float(df['net_profit'].mean()),
            'std_profit_rate': float(df['profit_rate'].std())
        }
    
    def save_summary(self, output_path: Optional[str] = None):
        """
        거래 요약을 파일로 저장
        
        Args:
            output_path: 출력 파일 경로 (기본값: log_dir/summary_{session_name}.json)
        """
        if output_path is None:
            output_path = self.log_dir / f"summary_{self.session_name}.json"
        else:
            output_path = Path(output_path)
        
        summary = {
            'session_name': self.session_name,
            'start_time': self.trades[0]['timestamp'] if self.trades else None,
            'end_time': self.trades[-1]['timestamp'] if self.trades else None,
            'statistics': self.get_summary_stats(),
            'trades': self.trades
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved: {output_path}")
    
    def clear(self):
        """거래 기록 초기화"""
        self.trades.clear()
        logger.info("Trade records cleared")
    
    def close(self):
        """로거 종료 및 최종 요약 저장"""
        # 요약 저장
        if self.trades:
            self.save_summary()
        
        # 텍스트 로거 핸들러 닫기
        if self.enable_text and hasattr(self, 'text_logger'):
            for handler in self.text_logger.handlers[:]:
                handler.close()
                self.text_logger.removeHandler(handler)
        
        logger.info(f"TradeLogger closed: {len(self.trades)} trades logged")
