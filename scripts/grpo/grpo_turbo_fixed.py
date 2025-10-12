#!/usr/bin/env python3
"""
수정된 터보 GRPO 훈련 스크립트 - Win Rate 문제 해결

주요 수정사항:
1. 거래 비용 감소 (0.43% → 0.1%)
2. 보상 구조 개선
3. 스캘핑 특화 설정
"""

import os
import sys
import logging
import torch
import time
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
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


def main():
    """수정된 터보 GRPO 훈련"""
    
    logger.info("🚀 Starting FIXED Turbo GRPO Training...")
    logger.info("🔧 Win Rate 문제 해결 버전")
    logger.info("=" * 60)
    
    # 경로 설정
    embedding_model_path = r"C:\Users\user\Workspace\datasets@20251005\autoencoder\20251012_013148\model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    output_dir = "models/grpo_turbo_fixed"
    
    # 경로 확인
    logger.info("🔍 Checking files...")
    if not os.path.exists(embedding_model_path):
        logger.error(f"❌ Embedding model not found: {embedding_model_path}")
        return False
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    logger.info("✅ All files found")
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Using device: {device}")
    
    try:
        # 1. 임베딩 모델 로드
        logger.info("📥 Loading embedding model...")
        checkpoint = torch.load(embedding_model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        embedding_model = MaskedAutoEncoder(
            input_dim=config.get('input_dim', 28),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            seq_len=config.get('seq_len', 60),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            mask_ratio=config.get('mask_ratio', 0.15)
        )
        
        embedding_model.load_state_dict(checkpoint['model_state_dict'])
        embedding_model.to(device)
        logger.info("✅ Embedding model loaded")
        
        # 2. 수정된 환경 생성
        logger.info("🏗️ Creating FIXED environment...")
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            table_name='datasets',
            seq_len=60,
            embedding_dim=128,
            expected_features=28,
            # 🔧 수정된 설정
            transaction_cost_rate=0.0005,    # 0.00215 → 0.0005 (0.05%, 주석과 일치)
            quick_exit_threshold=2.0,        # 1.5초 → 2.0초 (더 관대)
            quick_exit_penalty=0.005,        # 0.01 → 0.005 (페널티 감소)
            max_holding_time=30.0,           # 60초 → 30초 (진짜 스캘핑)
            holding_penalty_rate=0.0005,     # 0.001 → 0.0005 (페널티 감소)
            max_episode_steps=1000,
            quick_exit_mode='penalty_only',  # 페널티만 부여 (강제 청산 안함)
            device=device
        )
        
        logger.info("✅ FIXED environment created")
        logger.info("🔧 Key changes:")
        logger.info("  - Transaction cost: 0.215% → 0.05%")
        logger.info("  - Max holding time: 60s → 30s")
        logger.info("  - Penalties reduced by 50%")
        logger.info("  - Quick exit mode: penalty_only (no forced close)")
        
        # 3. 정책 생성
        logger.info("🧠 Creating policy...")
        policy = GRPOPolicy(
            embedding_dim=128,
            hidden_dim=128,
            action_dim=3
        )
        
        policy.to(device)
        logger.info("✅ Policy created")
        
        # 4. 훈련기 생성
        logger.info("⚡ Creating trainer...")
        
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=6,        # 8 → 6 (더 작은 배치)
            num_groups=3,
            learning_rate=0.001,         # 0.0005 → 0.001 (더 빠른 학습)
            gamma=0.95,                  # 0.99 → 0.95 (단기 보상 중시)
            clip_epsilon=0.3,            # 0.2 → 0.3 (더 큰 업데이트)
            kl_target=0.02,              # 0.01 → 0.02 (더 관대한 KL)
            entropy_coef=0.02,           # 0.01 → 0.02 (더 많은 탐험)
            value_coef=0.5,
            max_grad_norm=1.0,           # 0.5 → 1.0 (더 큰 그래디언트)
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("✅ Trainer created")
        
        # 5. 수정된 훈련 실행
        logger.info("🔥 Starting FIXED training...")
        logger.info("=" * 60)
        
        # 더 작은 규모로 빠른 실험
        total_timesteps = 15000          # 30000 → 15000
        episodes_per_iteration = 6 * 3   # 18 episodes per iteration
        total_episodes = (total_timesteps // 100) * episodes_per_iteration
        checkpoint_interval = 50         # 100 → 50 (더 자주 저장)
        
        logger.info(f"📊 FIXED Training Configuration:")
        logger.info(f"  Total Timesteps: {total_timesteps}")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Episodes per Group: 6")
        logger.info(f"  Learning Rate: 0.001 (2x faster)")
        logger.info(f"  Gamma: 0.95 (short-term focus)")
        logger.info(f"  Entropy: 0.02 (more exploration)")
        
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
        
        # 6. 결과 출력
        logger.info("=" * 60)
        logger.info("🎉 FIXED TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        logger.info(f"  Episodes/Second: {total_episodes/training_time:.1f}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(output_dir, 'final_model_fixed.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
        logger.info("🎊 FIXED training completed!")
        logger.info("📈 Expected improvements:")
        logger.info("  - Win rate should be 40-60% (vs 3-9%)")
        logger.info("  - Fewer trades per episode")
        logger.info("  - Better scalping behavior")
        logger.info(f"📊 TensorBoard: tensorboard --logdir {tensorboard_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
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
        print("\n🎉 수정된 터보 훈련이 성공적으로 완료되었습니다!")
        print("📊 Win rate가 크게 개선될 것으로 예상됩니다!")
    else:
        print("\n❌ 훈련이 실패했습니다.")
    
    sys.exit(0 if success else 1)