"""
TradeLogger 테스트

실시간 거래 로깅 기능을 테스트합니다.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime

from ai_trader.inference import TradeLogger


class TestTradeLogger:
    """TradeLogger 테스트 클래스"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """임시 로그 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def trade_logger(self, temp_log_dir):
        """TradeLogger 인스턴스 생성"""
        logger = TradeLogger(
            log_dir=temp_log_dir,
            session_name="test_session",
            enable_json=True,
            enable_csv=True,
            enable_text=True
        )
        yield logger
        logger.close()
    
    def test_initialization(self, temp_log_dir):
        """초기화 테스트"""
        logger = TradeLogger(
            log_dir=temp_log_dir,
            session_name="test_init"
        )
        
        # 로그 디렉토리 생성 확인
        assert Path(temp_log_dir).exists()
        
        # 파일 경로 확인
        assert logger.json_path.exists() == False  # 아직 거래 없음
        assert logger.csv_path.exists()  # CSV는 헤더만 작성됨
        
        logger.close()
    
    def test_log_trade(self, trade_logger):
        """거래 로깅 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071,
            transaction_cost=0.0043,
            position_size=1.0,
            metadata={'test': True}
        )
        
        # 거래 기록 확인
        trades = trade_logger.get_trades()
        assert len(trades) == 1
        
        trade = trades[0]
        assert trade['stock_code'] == "005930"
        assert trade['entry_price'] == 70000.0
        assert trade['exit_price'] == 70500.0
        assert trade['holding_time_seconds'] == 30.0
        assert trade['profit_rate'] == 0.0071
        assert trade['metadata']['test'] == True
    
    def test_multiple_trades(self, trade_logger):
        """여러 거래 로깅 테스트"""
        # 여러 거래 로깅
        for i in range(5):
            trade_logger.log_trade(
                stock_code=f"00593{i}",
                entry_timestamp=f"2025-10-05T09:00:{i:02d}",
                entry_price=70000.0 + i * 100,
                exit_timestamp=f"2025-10-05T09:00:{i+30:02d}",
                exit_price=70500.0 + i * 100,
                holding_time=30.0,
                profit_rate=0.0071,
                transaction_cost=0.0043,
                position_size=1.0
            )
        
        # 거래 기록 확인
        trades = trade_logger.get_trades()
        assert len(trades) == 5
    
    def test_json_logging(self, trade_logger):
        """JSON 로깅 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        # JSON 파일 확인
        assert trade_logger.json_path.exists()
        
        with open(trade_logger.json_path, 'r', encoding='utf-8') as f:
            trades = json.load(f)
        
        assert len(trades) == 1
        assert trades[0]['stock_code'] == "005930"
    
    def test_csv_logging(self, trade_logger):
        """CSV 로깅 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        # CSV 파일 확인
        assert trade_logger.csv_path.exists()
        
        with open(trade_logger.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0]['stock_code'] == "005930"
        assert float(rows[0]['entry_price']) == 70000.0
    
    def test_text_logging(self, trade_logger):
        """텍스트 로깅 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        # 텍스트 파일 확인
        assert trade_logger.text_path.exists()
        
        with open(trade_logger.text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "005930" in content
        assert "70000.00" in content
    
    def test_get_trades_df(self, trade_logger):
        """DataFrame 변환 테스트"""
        # 거래 로깅
        for i in range(3):
            trade_logger.log_trade(
                stock_code=f"00593{i}",
                entry_timestamp=f"2025-10-05T09:00:{i:02d}",
                entry_price=70000.0 + i * 100,
                exit_timestamp=f"2025-10-05T09:00:{i+30:02d}",
                exit_price=70500.0 + i * 100,
                holding_time=30.0,
                profit_rate=0.0071
            )
        
        # DataFrame 변환
        df = trade_logger.get_trades_df()
        
        assert len(df) == 3
        assert 'stock_code' in df.columns
        assert 'entry_price' in df.columns
        assert 'profit_rate' in df.columns
    
    def test_summary_stats(self, trade_logger):
        """요약 통계 테스트"""
        # 승리 거래
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        # 손실 거래
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:01:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:01:30",
            exit_price=69500.0,
            holding_time=30.0,
            profit_rate=-0.0071
        )
        
        # 요약 통계 확인
        stats = trade_logger.get_summary_stats()
        
        assert stats['total_trades'] == 2
        assert stats['win_rate'] == 0.5  # 50% 승률
        assert stats['avg_holding_time'] == 30.0
        assert 'total_profit' in stats
        assert 'max_profit' in stats
        assert 'max_loss' in stats
    
    def test_save_summary(self, trade_logger, temp_log_dir):
        """요약 저장 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        # 요약 저장
        summary_path = Path(temp_log_dir) / "test_summary.json"
        trade_logger.save_summary(str(summary_path))
        
        # 파일 확인
        assert summary_path.exists()
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        assert 'session_name' in summary
        assert 'statistics' in summary
        assert 'trades' in summary
        assert summary['statistics']['total_trades'] == 1
    
    def test_clear(self, trade_logger):
        """거래 기록 초기화 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071
        )
        
        assert len(trade_logger.get_trades()) == 1
        
        # 초기화
        trade_logger.clear()
        
        assert len(trade_logger.get_trades()) == 0
    
    def test_empty_stats(self, trade_logger):
        """빈 거래 기록 통계 테스트"""
        stats = trade_logger.get_summary_stats()
        
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0.0
        assert stats['avg_profit_rate'] == 0.0
    
    def test_profit_calculation(self, trade_logger):
        """수익 계산 테스트"""
        # 거래 로깅
        trade_logger.log_trade(
            stock_code="005930",
            entry_timestamp="2025-10-05T09:00:00",
            entry_price=70000.0,
            exit_timestamp="2025-10-05T09:00:30",
            exit_price=70500.0,
            holding_time=30.0,
            profit_rate=0.0071,
            transaction_cost=0.0043,
            position_size=1.0
        )
        
        trades = trade_logger.get_trades()
        trade = trades[0]
        
        # 수익 계산 확인
        expected_profit_amount = 70500.0 - 70000.0  # 500.0
        expected_net_profit = expected_profit_amount - (70000.0 * 0.0043)  # 500.0 - 301.0 = 199.0
        
        assert trade['profit_amount'] == expected_profit_amount
        assert abs(trade['net_profit'] - expected_net_profit) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
