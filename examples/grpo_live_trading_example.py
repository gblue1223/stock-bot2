"""
GRPO 실시간 거래 예제

이 스크립트는 GRPO 추론 엔진과 TradeLogger를 사용하여
실시간 거래를 시뮬레이션하고 모든 거래를 로깅하는 예제입니다.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import duckdb

from ai_trader.inference import GRPOInference, TradeLogger

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_live_trading(
    inference_engine: GRPOInference,
    trade_logger: TradeLogger,
    db_path: str,
    table_name: str = 'datasets',
    seq_len: int = 60,
    max_trades: int = 100
):
    """
    실시간 거래 시뮬레이션
    
    Args:
        inference_engine: GRPO 추론 엔진
        trade_logger: 거래 로거
        db_path: 데이터베이스 경로
        table_name: 테이블명
        seq_len: 시퀀스 길이
        max_trades: 최대 거래 횟수
    """
    logger.info("Starting live trading simulation...")
    
    # 데이터베이스 연결
    conn = duckdb.connect(db_path, read_only=True)
    
    # 특징 컬럼 가져오기
    columns_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    all_columns = columns_df['column_name'].tolist()
    exclude_columns = {'날짜', '종목코드', '번호'}
    feature_columns = [col for col in all_columns if col not in exclude_columns]
    
    # 랜덤 종목 및 날짜 선택
    query = f"""
        SELECT DISTINCT 종목코드, 날짜
        FROM {table_name}
        ORDER BY RANDOM()
        LIMIT 1
    """
    result = conn.execute(query).fetchdf()
    stock_code = result['종목코드'].iloc[0]
    date = result['날짜'].iloc[0]
    
    logger.info(f"Selected stock: {stock_code}, date: {date}")
    
    # 해당 종목/날짜의 데이터 로드
    query = f"""
        SELECT 종목코드, 날짜, 번호, {', '.join(feature_columns)}
        FROM {table_name}
        WHERE 종목코드 = ? AND 날짜 = ?
        ORDER BY 번호
    """
    df = conn.execute(query, [stock_code, date]).fetchdf()
    
    if len(df) < seq_len + 100:
        logger.error("Insufficient data for simulation")
        return
    
    # 메타데이터와 특징 분리
    metadata = df[['종목코드', '날짜', '번호']].values
    features = df[feature_columns].values.astype(np.float32)
    
    # 거래 시뮬레이션
    position = 0  # 0: 포지션 없음, 1: 매수 포지션
    entry_price = 0.0
    entry_time = 0
    entry_timestamp = ""
    trades_count = 0
    
    for step in range(seq_len, len(features)):
        # 현재 시퀀스 추출
        sequence = features[step - seq_len:step]
        
        # 행동 예측
        action, confidence = inference_engine.predict(sequence, deterministic=True)
        
        # 현재 가격 및 시간
        current_price = float(features[step, 0])  # 등락률을 가격으로 사용
        current_time = int(metadata[step, 2])
        current_timestamp = f"{date}T{current_time:06d}"
        
        # 행동 실행
        if action == 1 and position == 0:  # 매수
            position = 1
            entry_price = current_price
            entry_time = current_time
            entry_timestamp = current_timestamp
            logger.info(f"BUY: price={entry_price:.4f}, time={entry_timestamp}, confidence={confidence:.4f}")
        
        elif action == 2 and position == 1:  # 매도
            exit_price = current_price
            exit_timestamp = current_timestamp
            holding_time = float(current_time - entry_time)
            
            # 수익률 계산
            profit_rate = (exit_price - entry_price) / entry_price
            
            # 거래 로깅
            trade_logger.log_trade(
                stock_code=stock_code,
                entry_timestamp=entry_timestamp,
                entry_price=entry_price,
                exit_timestamp=exit_timestamp,
                exit_price=exit_price,
                holding_time=holding_time,
                profit_rate=profit_rate,
                transaction_cost=0.0043,  # 0.43%
                position_size=1.0,
                metadata={
                    'confidence': float(confidence),
                    'date': str(date)
                }
            )
            
            logger.info(f"SELL: price={exit_price:.4f}, time={exit_timestamp}, "
                       f"profit_rate={profit_rate:.4f}, holding_time={holding_time:.2f}s")
            
            # 포지션 청산
            position = 0
            trades_count += 1
            
            # 최대 거래 횟수 도달 시 종료
            if trades_count >= max_trades:
                logger.info(f"Reached max trades: {max_trades}")
                break
    
    # 포지션이 남아있으면 강제 청산
    if position == 1:
        exit_price = float(features[-1, 0])
        exit_timestamp = f"{date}T{int(metadata[-1, 2]):06d}"
        holding_time = float(int(metadata[-1, 2]) - entry_time)
        profit_rate = (exit_price - entry_price) / entry_price
        
        trade_logger.log_trade(
            stock_code=stock_code,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            holding_time=holding_time,
            profit_rate=profit_rate,
            transaction_cost=0.0043,
            position_size=1.0,
            metadata={
                'forced_liquidation': True,
                'date': str(date)
            }
        )
        
        logger.info(f"FORCED LIQUIDATION: price={exit_price:.4f}, profit_rate={profit_rate:.4f}")
    
    # 데이터베이스 연결 종료
    conn.close()
    
    # 거래 요약 출력
    summary = trade_logger.get_summary_stats()
    logger.info("=" * 60)
    logger.info("Trading Summary:")
    logger.info(f"  Total Trades: {summary['total_trades']}")
    logger.info(f"  Win Rate: {summary['win_rate']:.2%}")
    logger.info(f"  Avg Profit Rate: {summary['avg_profit_rate']:.4f}")
    logger.info(f"  Avg Holding Time: {summary['avg_holding_time']:.2f}s")
    logger.info(f"  Total Profit: {summary['total_profit']:.2f}")
    logger.info(f"  Max Profit: {summary['max_profit']:.2f}")
    logger.info(f"  Max Loss: {summary['max_loss']:.2f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='GRPO 실시간 거래 시뮬레이션')
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        required=True,
        help='임베딩 모델 체크포인트 경로'
    )
    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        help='정책 체크포인트 경로'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=r'C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb',
        help='DuckDB 데이터베이스 경로'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='datasets',
        help='테이블명'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='시퀀스 길이'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/live_trading',
        help='로그 디렉토리'
    )
    parser.add_argument(
        '--max-trades',
        type=int,
        default=100,
        help='최대 거래 횟수'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='디바이스'
    )
    
    args = parser.parse_args()
    
    # GRPO 추론 엔진 초기화
    logger.info("Initializing GRPO inference engine...")
    inference_engine = GRPOInference(
        embedding_model_path=args.embedding_model,
        policy_path=args.policy,
        device=args.device,
        use_torchscript=True,
        cache_size=1000
    )
    
    # 거래 로거 초기화
    logger.info("Initializing trade logger...")
    trade_logger = TradeLogger(
        log_dir=args.log_dir,
        session_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        enable_json=True,
        enable_csv=True,
        enable_text=True
    )
    
    try:
        # 실시간 거래 시뮬레이션
        simulate_live_trading(
            inference_engine=inference_engine,
            trade_logger=trade_logger,
            db_path=args.db,
            table_name=args.table,
            seq_len=args.seq_len,
            max_trades=args.max_trades
        )
    finally:
        # 로거 종료
        trade_logger.close()
        
        # 성능 통계 출력
        perf_stats = inference_engine.get_performance_stats()
        logger.info("=" * 60)
        logger.info("Inference Performance:")
        logger.info(f"  Mean Inference Time: {perf_stats['mean_inference_time_ms']:.2f}ms")
        logger.info(f"  Median Inference Time: {perf_stats['median_inference_time_ms']:.2f}ms")
        logger.info(f"  P95 Inference Time: {perf_stats['p95_inference_time_ms']:.2f}ms")
        logger.info(f"  P99 Inference Time: {perf_stats['p99_inference_time_ms']:.2f}ms")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
