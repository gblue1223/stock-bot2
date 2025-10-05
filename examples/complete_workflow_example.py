"""
완전한 워크플로우 예제

이 스크립트는 임베딩 모델 훈련부터 GRPO 훈련, 평가, 추론까지
전체 파이프라인을 단계별로 보여주는 종합 예제입니다.

사용법:
    python examples/complete_workflow_example.py --help
"""

import argparse
import logging
from pathlib import Path
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_train_embedding(args):
    """Step 1: 임베딩 모델 훈련"""
    logger.info("=" * 80)
    logger.info("STEP 1: 임베딩 모델 훈련")
    logger.info("=" * 80)
    
    from ai_trader.embedding.train_embedding import main as train_embedding
    
    # 임베딩 모델 훈련 인수 설정
    embedding_args = [
        '--db', args.db,
        '--table', args.table,
        '--out', str(Path(args.out) / 'embedding'),
        '--seq-len', str(args.seq_len),
        '--embedding-dim', str(args.embedding_dim),
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.embedding_epochs),
        '--lr', str(args.embedding_lr),
        '--device', args.device,
        '--val-every', '5',
        '--ckpt-every', '10'
    ]
    
    logger.info(f"Training embedding model with args: {' '.join(embedding_args)}")
    
    # 임베딩 모델 훈련 실행
    sys.argv = ['train_embedding.py'] + embedding_args
    train_embedding()
    
    embedding_checkpoint = Path(args.out) / 'embedding' / f'checkpoint_epoch{args.embedding_epochs}.pt'
    logger.info(f"✓ Embedding model trained: {embedding_checkpoint}")
    
    return embedding_checkpoint


def step2_evaluate_embedding(embedding_checkpoint, args):
    """Step 2: 임베딩 모델 평가"""
    logger.info("=" * 80)
    logger.info("STEP 2: 임베딩 모델 평가")
    logger.info("=" * 80)
    
    from ai_trader.embedding.evaluate_embedding import evaluate_embedding_model
    
    # 임베딩 모델 평가
    metrics = evaluate_embedding_model(
        checkpoint_path=str(embedding_checkpoint),
        db_path=args.db,
        table_name=args.table,
        split='test',
        device=args.device
    )
    
    logger.info("Embedding evaluation metrics:")
    logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    logger.info(f"  Temporal Coherence: {metrics['temporal_coherence']:.4f}")
    
    # 품질 확인
    if metrics['silhouette_score'] > 0.3 and metrics['temporal_coherence'] > 0.6:
        logger.info("✓ Embedding quality is good!")
    else:
        logger.warning("⚠ Embedding quality may need improvement")
        logger.warning("  Consider training for more epochs or adjusting hyperparameters")
    
    return metrics


def step3_train_grpo(embedding_checkpoint, args):
    """Step 3: GRPO 에이전트 훈련"""
    logger.info("=" * 80)
    logger.info("STEP 3: GRPO 에이전트 훈련")
    logger.info("=" * 80)
    
    from ai_trader.grpo.train_grpo import main as train_grpo
    
    # GRPO 훈련 인수 설정
    grpo_args = [
        '--embedding-model', str(embedding_checkpoint),
        '--db', args.db,
        '--table', args.table,
        '--out', str(Path(args.out) / 'grpo'),
        '--seq-len', str(args.seq_len),
        '--episodes-per-group', str(args.episodes_per_group),
        '--num-groups', str(args.num_groups),
        '--total-timesteps', str(args.total_timesteps),
        '--lr', str(args.grpo_lr),
        '--device', args.device,
        '--quick-exit-threshold', str(args.quick_exit_threshold),
        '--quick-exit-penalty', str(args.quick_exit_penalty),
        '--ckpt-every', '50000'
    ]
    
    logger.info(f"Training GRPO agent with args: {' '.join(grpo_args)}")
    
    # GRPO 훈련 실행
    sys.argv = ['train_grpo.py'] + grpo_args
    train_grpo()
    
    policy_checkpoint = Path(args.out) / 'grpo' / f'checkpoint_step{args.total_timesteps}.pt'
    logger.info(f"✓ GRPO agent trained: {policy_checkpoint}")
    
    return policy_checkpoint


def step4_evaluate_grpo(embedding_checkpoint, policy_checkpoint, args):
    """Step 4: GRPO 에이전트 평가"""
    logger.info("=" * 80)
    logger.info("STEP 4: GRPO 에이전트 평가")
    logger.info("=" * 80)
    
    from ai_trader.grpo.evaluate_grpo import evaluate_grpo_agent
    
    # GRPO 에이전트 평가
    metrics = evaluate_grpo_agent(
        embedding_model_path=str(embedding_checkpoint),
        policy_path=str(policy_checkpoint),
        db_path=args.db,
        table_name=args.table,
        device=args.device,
        num_episodes=100
    )
    
    logger.info("GRPO evaluation metrics:")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"  Avg Profit per Trade: {metrics['avg_profit_per_trade']:.4f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"  Avg Holding Time: {metrics['avg_holding_time']:.2f}s")
    
    # 성능 기준 확인
    criteria_met = (
        metrics['win_rate'] > 0.50 and
        metrics['sharpe_ratio'] > 1.0 and
        metrics['max_drawdown'] < 0.10 and
        metrics['avg_holding_time'] < 30
    )
    
    if criteria_met:
        logger.info("✓ All performance criteria met!")
    else:
        logger.warning("⚠ Some performance criteria not met")
        logger.warning("  Consider training for more timesteps or adjusting hyperparameters")
    
    return metrics


def step5_backtest(embedding_checkpoint, policy_checkpoint, args):
    """Step 5: 백테스팅"""
    logger.info("=" * 80)
    logger.info("STEP 5: 백테스팅")
    logger.info("=" * 80)
    
    from ai_trader.grpo.backtest import run_backtest
    
    # 백테스팅 실행
    results = run_backtest(
        embedding_model_path=str(embedding_checkpoint),
        policy_path=str(policy_checkpoint),
        db_path=args.db,
        table_name=args.table,
        seq_len=args.seq_len,
        transaction_cost_rate=0.00215,
        slippage_rate=0.0001,
        quick_exit_threshold=args.quick_exit_threshold,
        quick_exit_penalty=args.quick_exit_penalty,
        device=args.device,
        verbose=True
    )
    
    summary = results['summary']
    metrics = results['metrics']
    
    logger.info("Backtest results:")
    logger.info(f"  Episodes: {summary['num_episodes']}")
    logger.info(f"  Total Trades: {summary['total_trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"  Avg Profit per Trade: {metrics['avg_profit_per_trade']:.4f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    logger.info("✓ Backtest completed")
    
    return results


def step6_inference_demo(embedding_checkpoint, policy_checkpoint, args):
    """Step 6: 추론 데모"""
    logger.info("=" * 80)
    logger.info("STEP 6: 추론 데모")
    logger.info("=" * 80)
    
    import numpy as np
    from ai_trader.inference import GRPOInference
    
    # 추론 엔진 초기화
    logger.info("Initializing inference engine...")
    inference = GRPOInference(
        embedding_model_path=str(embedding_checkpoint),
        policy_path=str(policy_checkpoint),
        device=args.device,
        use_torchscript=True,
        cache_size=1000
    )
    
    # 테스트 시퀀스 생성
    logger.info("Running inference on test sequences...")
    action_names = {0: "보유", 1: "매수", 2: "매도"}
    
    for i in range(5):
        sequence = np.random.randn(args.seq_len, 60).astype(np.float32)
        action, confidence = inference.predict(sequence, deterministic=True)
        logger.info(f"  Test {i+1}: {action_names[action]} (confidence: {confidence:.4f})")
    
    # 성능 통계
    stats = inference.get_performance_stats()
    logger.info(f"Inference performance:")
    logger.info(f"  Mean inference time: {stats['mean_inference_time_ms']:.2f}ms")
    logger.info(f"  P95 inference time: {stats['p95_inference_time_ms']:.2f}ms")
    
    if stats['mean_inference_time_ms'] < 10:
        logger.info("✓ Target latency (<10ms) achieved!")
    else:
        logger.warning(f"⚠ Target latency not achieved ({stats['mean_inference_time_ms']:.2f}ms)")
    
    logger.info("✓ Inference demo completed")


def main():
    parser = argparse.ArgumentParser(
        description='완전한 워크플로우 예제: 임베딩 훈련 → GRPO 훈련 → 평가 → 추론',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 워크플로우 실행
  python examples/complete_workflow_example.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --out models/example_run

  # 특정 단계만 실행
  python examples/complete_workflow_example.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --out models/example_run \\
    --skip-embedding \\
    --skip-grpo

  # 빠른 테스트 (적은 에포크/타임스텝)
  python examples/complete_workflow_example.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --out models/quick_test \\
    --embedding-epochs 5 \\
    --total-timesteps 10000
        """
    )
    
    # 데이터베이스 설정
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='datasets',
        help='테이블명 (기본값: datasets)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='출력 디렉토리'
    )
    
    # 모델 설정
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='시퀀스 길이 (기본값: 60)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='임베딩 차원 (기본값: 128)'
    )
    
    # 임베딩 훈련 설정
    parser.add_argument(
        '--embedding-epochs',
        type=int,
        default=50,
        help='임베딩 훈련 에포크 수 (기본값: 50)'
    )
    parser.add_argument(
        '--embedding-lr',
        type=float,
        default=1e-4,
        help='임베딩 학습률 (기본값: 1e-4)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='배치 크기 (기본값: 128)'
    )
    
    # GRPO 훈련 설정
    parser.add_argument(
        '--episodes-per-group',
        type=int,
        default=12,
        help='그룹당 에피소드 수 (기본값: 12)'
    )
    parser.add_argument(
        '--num-groups',
        type=int,
        default=4,
        help='그룹 수 (기본값: 4)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=1000000,
        help='총 훈련 타임스텝 (기본값: 1000000)'
    )
    parser.add_argument(
        '--grpo-lr',
        type=float,
        default=3e-4,
        help='GRPO 학습률 (기본값: 3e-4)'
    )
    parser.add_argument(
        '--quick-exit-threshold',
        type=float,
        default=1.5,
        help='빠른 손절 시간 임계값 (초, 기본값: 1.5)'
    )
    parser.add_argument(
        '--quick-exit-penalty',
        type=float,
        default=0.01,
        help='빠른 손절 룰 위반 페널티 (기본값: 0.01)'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='디바이스 (기본값: cuda)'
    )
    
    # 워크플로우 제어
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='임베딩 훈련 건너뛰기 (기존 체크포인트 사용)'
    )
    parser.add_argument(
        '--skip-grpo',
        action='store_true',
        help='GRPO 훈련 건너뛰기 (기존 체크포인트 사용)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='평가 건너뛰기'
    )
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='백테스팅 건너뛰기'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("완전한 워크플로우 예제 시작")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db}")
    logger.info(f"Output directory: {args.out}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    try:
        # Step 1: 임베딩 모델 훈련
        if args.skip_embedding:
            embedding_checkpoint = Path(args.out) / 'embedding' / f'checkpoint_epoch{args.embedding_epochs}.pt'
            logger.info(f"Skipping embedding training, using: {embedding_checkpoint}")
        else:
            embedding_checkpoint = step1_train_embedding(args)
        
        # Step 2: 임베딩 모델 평가
        if not args.skip_evaluation:
            step2_evaluate_embedding(embedding_checkpoint, args)
        
        # Step 3: GRPO 에이전트 훈련
        if args.skip_grpo:
            policy_checkpoint = Path(args.out) / 'grpo' / f'checkpoint_step{args.total_timesteps}.pt'
            logger.info(f"Skipping GRPO training, using: {policy_checkpoint}")
        else:
            policy_checkpoint = step3_train_grpo(embedding_checkpoint, args)
        
        # Step 4: GRPO 에이전트 평가
        if not args.skip_evaluation:
            step4_evaluate_grpo(embedding_checkpoint, policy_checkpoint, args)
        
        # Step 5: 백테스팅
        if not args.skip_backtest:
            step5_backtest(embedding_checkpoint, policy_checkpoint, args)
        
        # Step 6: 추론 데모
        step6_inference_demo(embedding_checkpoint, policy_checkpoint, args)
        
        logger.info("=" * 80)
        logger.info("✓ 완전한 워크플로우 완료!")
        logger.info("=" * 80)
        logger.info(f"Embedding model: {embedding_checkpoint}")
        logger.info(f"GRPO policy: {policy_checkpoint}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"워크플로우 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
