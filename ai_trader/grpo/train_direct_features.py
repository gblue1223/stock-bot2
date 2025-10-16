#!/usr/bin/env python3
"""
직접 특징 사용 GRPO 훈련 - 임베딩 우회

핵심 아이디어:
1. 임베딩 사용하지 않음
2. 원본 특징을 직접 정책에 입력
3. 단순하지만 효과적인 접근법
4. 거래 관련성 문제 해결
"""

import os
import sys
import logging
import torch
import time
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.environments import DirectFeatureEnv
from ai_trader.grpo.policies import DirectFeaturePolicy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """직접 특징 사용 GRPO 훈련"""
    
    logger.info("Starting DIRECT FEATURES GRPO Training...")
    logger.info("No embedding - direct raw features")
    logger.info("=" * 60)
    
    # 경로 설정
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    output_dir = "models/grpo_direct_features"
    
    # 경로 확인
    logger.info("Checking files...")
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return False
    
    logger.info("Database found")
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        # 1. 직접 특징 환경 생성
        logger.info("Creating DIRECT FEATURES environment...")
        env = DirectFeatureEnv(
            db_path=db_path,
            table_name='datasets_raw',
            seq_len=30,
            expected_features=24,
            transaction_cost_rate=0.00215,
            max_episode_steps=100,
            device=device
        )
        
        logger.info("DIRECT FEATURES environment created")
        logger.info("Direct features settings:")
        logger.info("  - No embedding (direct raw features)")
        logger.info("  - Sequence length: 30 (efficient)")
        logger.info("  - Transaction cost: 0.215% (realistic)")
        logger.info("  - Episode steps: 100 (short)")
        logger.info(f"  - Observation dim: {env.observation_space.shape[0]}")
        
        # 2. 직접 특징 정책 네트워크
        logger.info("Creating DIRECT FEATURES policy...")
        
        input_dim = env.observation_space.shape[0]
        policy = DirectFeaturePolicy(
            input_dim=input_dim,
            hidden_dim=64,
            action_dim=3
        )
        
        policy.to(device)
        logger.info("DIRECT FEATURES policy created")
        
        # 3. 보수적 훈련기 설정
        logger.info("Creating CONSERVATIVE trainer...")
        
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=4,
            num_groups=2,
            learning_rate=0.001,
            gamma=0.95,
            clip_epsilon=0.2,
            kl_target=0.01,
            entropy_coef=0.2,
            value_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("CONSERVATIVE trainer created")
        
        # 4. 훈련 설정
        logger.info("Starting DIRECT FEATURES training...")
        logger.info("=" * 60)
        
        total_timesteps = 2000
        episodes_per_iteration = 4 * 2
        total_episodes = (total_timesteps // 25) * episodes_per_iteration
        checkpoint_interval = 5
        
        logger.info(f"DIRECT FEATURES Configuration:")
        logger.info(f"  Total Timesteps: {total_timesteps}")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Episodes per Group: 4")
        logger.info(f"  Input Dimension: {input_dim}")
        logger.info(f"  Learning Rate: 0.001")
        logger.info(f"  Expected Time: 15-25 minutes")
        
        # 체크포인트 경로
        checkpoint_path = os.path.join(output_dir, 'checkpoints', 'checkpoint_iter{}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 훈련 시작
        start_time = time.time()
        
        final_metrics = trainer.train(
            total_episodes=total_episodes,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path
        )
        
        training_time = time.time() - start_time
        
        # 5. 결과 출력
        logger.info("=" * 60)
        logger.info("DIRECT FEATURES TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(output_dir, 'direct_features_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
        logger.info("DIRECT FEATURES completed!")
        logger.info("Expected improvements:")
        logger.info("  - No embedding quality issues")
        logger.info("  - Direct trading signal access")
        logger.info("  - Faster convergence")
        logger.info("  - Better trading relevance")
        logger.info(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False
    
    finally:
        # 정리
        if 'env' in locals():
            env.close()
        
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    success = main()
    if success:
        print("\nDirect feature training completed successfully!")
        print("Bypassed embedding issues!")
    else:
        print("\nTraining failed.")
    
    sys.exit(0 if success else 1)
