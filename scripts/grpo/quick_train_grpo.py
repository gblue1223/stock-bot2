#!/usr/bin/env python3
"""
빠른 GRPO 훈련 - 효율적인 학습을 위한 최적화

문제점 해결:
1. 너무 긴 훈련 시간 (88시간 → 2-3시간)
2. 과도한 거래 빈도 제어
3. Win rate 빠른 개선
"""

import os
import sys
import logging
import torch
import time
from pathlib import Path

# 프로젝트 루트 추가
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


def main():
    """빠른 GRPO 훈련"""
    
    logger.info("🚀 Starting QUICK GRPO Training...")
    logger.info("⚡ 효율적인 학습을 위한 최적화")
    logger.info("=" * 60)
    
    # 경로 설정
    embedding_model_path = r"C:\Users\user\Workspace\datasets@20251005\autoencoder\20251012_013148\model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251005\datasets_norm_all.duckdb"
    output_dir = "models/grpo_scalping"
    
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
        embedding_model.eval()
        
        logger.info("✅ Embedding model loaded")
        
        # 2. 빠른 학습용 환경 생성
        logger.info("🏗️ Creating QUICK environment...")
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=db_path,
            table_name='datasets',
            seq_len=60,
            embedding_dim=128,
            expected_features=28,
            # 🔧 빠른 학습을 위한 설정
            transaction_cost_rate=0.00215,   # 현실적 유지
            quick_exit_threshold=5.0,        # 매우 관대 (학습 우선)
            quick_exit_penalty=0.001,        # 거의 없는 페널티
            max_holding_time=20.0,           # 20초 (진짜 스캘핑)
            holding_penalty_rate=0.0001,     # 거의 없는 페널티
            device=device
        )
        
        logger.info("✅ QUICK environment created")
        logger.info("🔧 Quick settings:")
        logger.info("  - Max holding time: 20s (ultra-short scalping)")
        logger.info("  - Minimal penalties (learning first)")
        logger.info("  - Quick exit threshold: 5s (very forgiving)")
        
        # 3. 작은 정책 네트워크 (빠른 학습)
        logger.info("🧠 Creating compact policy...")
        policy = GRPOPolicy(
            embedding_dim=128,
            hidden_dim=64,  # 128 → 64 (더 빠른 학습)
            action_dim=3
        )
        
        policy.to(device)
        logger.info("✅ Compact policy created")
        
        # 4. 공격적 훈련기 설정
        logger.info("⚡ Creating aggressive trainer...")
        
        os.makedirs(output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=3,        # 4 → 3 (더 빠른 반복)
            num_groups=2,                # 유지
            learning_rate=0.001,         # 0.0003 → 0.001 (3배 빠름)
            gamma=0.9,                   # 0.98 → 0.9 (단기 중시)
            clip_epsilon=0.4,            # 0.25 → 0.4 (큰 업데이트)
            kl_target=0.03,              # 0.015 → 0.03 (관대한 KL)
            entropy_coef=0.05,           # 0.03 → 0.05 (최대 탐험)
            value_coef=0.2,              # 0.3 → 0.2 (정책 중심)
            max_grad_norm=1.5,           # 0.8 → 1.5 (큰 그래디언트)
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("✅ Aggressive trainer created")
        
        # 5. 빠른 실험 설정
        logger.info("🔥 Starting QUICK training...")
        logger.info("=" * 60)
        
        # 매우 작은 규모로 빠른 검증
        total_timesteps = 3000           # 10000 → 3000 (매우 빠름)
        episodes_per_iteration = 3 * 2   # 6 episodes per iteration
        total_episodes = (total_timesteps // 100) * episodes_per_iteration
        checkpoint_interval = 10         # 25 → 10 (매우 자주)
        
        logger.info(f"📊 QUICK Configuration:")
        logger.info(f"  Total Timesteps: {total_timesteps}")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Episodes per Group: 3")
        logger.info(f"  Learning Rate: 0.001 (3x faster)")
        logger.info(f"  Gamma: 0.9 (ultra short-term)")
        logger.info(f"  Entropy: 0.05 (maximum exploration)")
        logger.info(f"  Expected Time: 30-60 minutes")
        
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
        logger.info("🎉 QUICK TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        logger.info(f"  Episodes/Second: {total_episodes/training_time:.1f}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(output_dir, 'quick_fix_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
        logger.info("🎊 QUICK completed!")
        logger.info("📈 Expected quick improvements:")
        logger.info("  - Win rate should reach 25-35% quickly")
        logger.info("  - Much shorter episodes")
        logger.info("  - Faster convergence")
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
        print("\n🎉 빠른 훈련이 완료되었습니다!")
        print("📊 Win rate가 빠르게 개선되었을 것입니다!")
    else:
        print("\n❌ 훈련이 실패했습니다.")
    
    sys.exit(0 if success else 1)