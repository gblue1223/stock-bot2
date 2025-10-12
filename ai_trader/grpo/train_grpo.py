"""
GRPO 훈련 스크립트

이 스크립트는 GRPO 알고리즘을 사용하여 스캘핑 정책을 훈련합니다.

사용 예시:
    python -m ai_trader.grpo.train_grpo \
        --embedding-model models/embedding/checkpoint_epoch50.pt \
        --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \
        --table datasets \
        --out models/grpo_scalping \
        --seq-len 60 \
        --episodes-per-group 12 \
        --num-groups 4 \
        --total-timesteps 1000000 \
        --quick-exit-threshold 1.5 \
        --quick-exit-penalty 0.01 \
        --device cuda
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder
from ai_trader.grpo.env import GRPOScalpingEnv
from ai_trader.grpo.policy import GRPOPolicy
from ai_trader.grpo.grpo import GRPOTrainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='GRPO 스캘핑 정책 훈련',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 필수 인수
    parser.add_argument(
        '--embedding-model',
        type=str,
        required=True,
        help='훈련된 임베딩 모델 체크포인트 경로'
    )
    
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    
    # 선택적 인수
    parser.add_argument(
        '--table',
        type=str,
        default='datasets',
        help='데이터베이스 테이블명'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='models/grpo_scalping',
        help='출력 디렉토리 (체크포인트 및 로그 저장)'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='시퀀스 길이'
    )
    
    parser.add_argument(
        '--episodes-per-group',
        type=int,
        default=12,
        help='그룹당 에피소드 수'
    )
    
    parser.add_argument(
        '--num-groups',
        type=int,
        default=4,
        help='그룹 수'
    )
    
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=1000000,
        help='총 훈련 타임스텝 수'
    )
    
    parser.add_argument(
        '--quick-exit-threshold',
        type=float,
        default=1.5,
        help='빠른 손절 시간 임계값 (초)'
    )
    
    parser.add_argument(
        '--quick-exit-penalty',
        type=float,
        default=0.01,
        help='빠른 손절 룰 위반 페널티'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cpu', 'cuda'],
        help='디바이스 (cpu 또는 cuda)'
    )
    
    # GRPO 하이퍼파라미터
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='학습률'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='할인 계수'
    )
    
    parser.add_argument(
        '--clip-epsilon',
        type=float,
        default=0.2,
        help='PPO 클리핑 파라미터'
    )
    
    parser.add_argument(
        '--kl-target',
        type=float,
        default=0.01,
        help='KL 발산 목표값'
    )
    
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='엔트로피 계수'
    )
    
    parser.add_argument(
        '--value-coef',
        type=float,
        default=0.5,
        help='가치 손실 계수'
    )
    
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='그래디언트 클리핑 최대값'
    )
    
    # 환경 설정
    parser.add_argument(
        '--transaction-cost-rate',
        type=float,
        default=0.00215,
        help='거래 비용 비율 (0.215%)'
    )
    
    parser.add_argument(
        '--max-holding-time',
        type=float,
        default=60.0,
        help='최대 보유 시간 (초)'
    )
    
    parser.add_argument(
        '--holding-penalty-rate',
        type=float,
        default=0.001,
        help='장기 보유 페널티 비율'
    )
    
    # 정책 네트워크 설정
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='정책 네트워크 은닉 레이어 차원'
    )
    
    # 체크포인트 설정
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='체크포인트 저장 간격 (에피소드 단위)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='재개할 체크포인트 경로 (선택사항)'
    )
    
    return parser.parse_args()


def load_embedding_model(
    checkpoint_path: str,
    device: str
) -> MaskedAutoEncoder:
    """
    임베딩 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 경로
        device: 디바이스
        
    Returns:
        로드된 임베딩 모델
    """
    logger.info(f"Loading embedding model from: {checkpoint_path}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 설정 추출
        config = checkpoint.get('config', {})
        
        # 모델 생성 (MaskedAutoEncoder)
        model = MaskedAutoEncoder(
            input_dim=config.get('input_dim', 28),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            seq_len=config.get('seq_len', 60),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            mask_ratio=config.get('mask_ratio', 0.15)
        )
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Embedding model loaded successfully: "
                   f"embedding_dim={config.get('embedding_dim', 128)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(f"Cannot load embedding model from {checkpoint_path}: {e}")


def create_output_directory(output_dir: str) -> tuple:
    """
    출력 디렉토리 생성
    
    Args:
        output_dir: 출력 디렉토리 경로
        
    Returns:
        (checkpoint_dir, tensorboard_dir) 튜플
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    tensorboard_dir = output_path / 'tensorboard_logs'
    tensorboard_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory created: {output_dir}")
    logger.info(f"  - Checkpoints: {checkpoint_dir}")
    logger.info(f"  - TensorBoard logs: {tensorboard_dir}")
    
    return str(checkpoint_dir), str(tensorboard_dir)


def save_checkpoint(
    policy: GRPOPolicy,
    trainer: GRPOTrainer,
    checkpoint_path: str,
    iteration: int,
    args: argparse.Namespace
):
    """
    체크포인트 저장
    
    Args:
        policy: GRPO 정책
        trainer: GRPO 훈련기
        checkpoint_path: 체크포인트 저장 경로
        iteration: 현재 반복 횟수
        args: CLI 인수
    """
    checkpoint = {
        'state_dict': policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': {
            'embedding_dim': policy.embedding_dim,
            'hidden_dim': policy.hidden_dim,
            'action_dim': policy.action_dim
        },
        'training_metadata': {
            'iteration': iteration,
            'total_timesteps': trainer.total_timesteps,
            'num_updates': trainer.num_updates,
            'quick_exit_threshold': args.quick_exit_threshold,
            'quick_exit_penalty': args.quick_exit_penalty
        },
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'clip_epsilon': args.clip_epsilon,
            'kl_target': args.kl_target,
            'entropy_coef': args.entropy_coef,
            'value_coef': args.value_coef,
            'episodes_per_group': args.episodes_per_group,
            'num_groups': args.num_groups
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    """메인 함수"""
    # CLI 인수 파싱
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("GRPO 스캘핑 정책 훈련 시작")
    logger.info("=" * 80)
    logger.info(f"임베딩 모델: {args.embedding_model}")
    logger.info(f"데이터베이스: {args.db}")
    logger.info(f"테이블: {args.table}")
    logger.info(f"출력 디렉토리: {args.out}")
    logger.info(f"시퀀스 길이: {args.seq_len}")
    logger.info(f"그룹당 에피소드: {args.episodes_per_group}")
    logger.info(f"그룹 수: {args.num_groups}")
    logger.info(f"총 타임스텝: {args.total_timesteps}")
    logger.info(f"빠른 손절 임계값: {args.quick_exit_threshold}초")
    logger.info(f"빠른 손절 페널티: {args.quick_exit_penalty}")
    logger.info(f"디바이스: {args.device}")
    logger.info("=" * 80)
    
    # 출력 디렉토리 생성
    checkpoint_dir, tensorboard_dir = create_output_directory(args.out)
    
    # 임베딩 모델 로드
    embedding_model = load_embedding_model(args.embedding_model, args.device)
    
    # 임베딩 차원 추출
    embedding_dim = embedding_model.embedding_dim
    
    # 임베딩 모델의 입력 차원 추출
    input_dim = embedding_model.encoder.input_projection.in_features
    logger.info(f"Detected embedding model input dimension: {input_dim}")
    
    # GRPO 환경 생성
    logger.info("Creating GRPO environment...")
    env = GRPOScalpingEnv(
        embedding_model=embedding_model,
        db_path=args.db,
        table_name=args.table,
        seq_len=args.seq_len,
        embedding_dim=embedding_dim,
        expected_features=input_dim,  # 임베딩 모델의 입력 차원과 일치
        transaction_cost_rate=args.transaction_cost_rate,
        quick_exit_threshold=args.quick_exit_threshold,
        quick_exit_penalty=args.quick_exit_penalty,
        max_holding_time=args.max_holding_time,
        holding_penalty_rate=args.holding_penalty_rate,
        device=args.device
    )
    
    # GRPO 정책 생성
    logger.info("Creating GRPO policy...")
    policy = GRPOPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=3  # 보유/매수/매도
    )
    
    # GRPO 훈련기 생성
    logger.info("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        policy=policy,
        env=env,
        episodes_per_group=args.episodes_per_group,
        num_groups=args.num_groups,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        kl_target=args.kl_target,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        tensorboard_log_dir=tensorboard_dir
    )
    
    # 체크포인트 재개 (선택사항)
    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=args.device)
            policy.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['training_metadata']['iteration']
            trainer.total_timesteps = checkpoint['training_metadata']['total_timesteps']
            trainer.num_updates = checkpoint['training_metadata']['num_updates']
            logger.info(f"Resumed from iteration {start_iteration}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise
    
    # 총 에피소드 수 계산
    total_episodes = (args.total_timesteps // 100) * (args.episodes_per_group * args.num_groups)
    logger.info(f"Total episodes to train: {total_episodes}")
    
    # 훈련 시작
    logger.info("=" * 80)
    logger.info("Starting GRPO training...")
    logger.info("=" * 80)
    
    try:
        # 훈련 실행
        final_metrics = trainer.train(
            total_episodes=total_episodes,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_path=os.path.join(checkpoint_dir, 'checkpoint_iter{}.pt')
        )
        
        # 최종 체크포인트 저장
        final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
        save_checkpoint(
            policy=policy,
            trainer=trainer,
            checkpoint_path=final_checkpoint_path,
            iteration=final_metrics.get('num_updates', 0),
            args=args
        )
        
        logger.info("=" * 80)
        logger.info("GRPO training completed successfully!")
        logger.info(f"Total timesteps: {final_metrics['total_timesteps']}")
        logger.info(f"Total updates: {final_metrics['num_updates']}")
        logger.info(f"Final checkpoint: {final_checkpoint_path}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # 중단 시 체크포인트 저장
        interrupt_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_interrupted.pt')
        save_checkpoint(
            policy=policy,
            trainer=trainer,
            checkpoint_path=interrupt_checkpoint_path,
            iteration=trainer.num_updates,
            args=args
        )
        logger.info(f"Checkpoint saved: {interrupt_checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # 환경 종료
        env.close()
        logger.info("Environment closed")


if __name__ == '__main__':
    main()
