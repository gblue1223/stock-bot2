#!/usr/bin/env python3
"""
Modular GRPO Training Script

Allows selection of different environments and policies via command-line arguments.
"""

import os
import sys
import logging
import argparse
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
from ai_trader.grpo.environments import GRPOScalpingEnv, NormalizedFeatureEnv, DirectFeatureEnv
from ai_trader.grpo.policies import GRPOPolicy, NormalizedFeaturePolicy, DirectFeaturePolicy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_environment(args, device):
    """환경 생성"""
    if args.env == 'normalized':
        logger.info("Creating NormalizedFeatureEnv...")
        env = NormalizedFeatureEnv(
            db_path=args.db_path,
            table_name=args.table_name,
            seq_len=args.seq_len,
            expected_features=args.features,
            transaction_cost_rate=0.00215,
            max_episode_steps=args.episode_steps,
            device=device
        )
    elif args.env == 'direct':
        logger.info("Creating DirectFeatureEnv...")
        env = DirectFeatureEnv(
            db_path=args.db_path,
            table_name=args.table_name,
            seq_len=args.seq_len,
            expected_features=args.features,
            transaction_cost_rate=0.00215,
            max_episode_steps=args.episode_steps,
            device=device
        )
    elif args.env == 'scalping':
        if args.embedding_model is None:
            raise ValueError("--embedding_model is required for scalping environment")
        
        logger.info(f"Loading embedding model from {args.embedding_model}...")
        from ai_trader.embedding.masked_autoencoder import MaskedAutoEncoder
        
        embedding_model = MaskedAutoEncoder(
            input_dim=args.features,
            embedding_dim=args.embedding_dim
        )
        checkpoint = torch.load(args.embedding_model, map_location=device)
        embedding_model.load_state_dict(checkpoint['model_state_dict'])
        embedding_model.to(device)
        embedding_model.eval()
        
        logger.info("Creating GRPOScalpingEnv...")
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=args.db_path,
            table_name=args.table_name,
            seq_len=args.seq_len,
            embedding_dim=args.embedding_dim,
            expected_features=args.features,
            transaction_cost_rate=0.00215,
            max_episode_steps=args.episode_steps,
            quick_exit_mode=args.quick_exit_mode,
            device=device
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")
    
    return env


def create_policy(args, env, device):
    """정책 생성"""
    if args.policy == 'normalized':
        logger.info("Creating NormalizedFeaturePolicy...")
        policy = NormalizedFeaturePolicy(
            feature_dim=env.feature_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim
        )
    elif args.policy == 'direct':
        logger.info("Creating DirectFeaturePolicy...")
        input_dim = env.observation_space.shape[0]
        policy = DirectFeaturePolicy(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim
        )
    elif args.policy == 'grpo':
        logger.info("Creating GRPOPolicy...")
        policy = GRPOPolicy(
            embedding_dim=env.embedding_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim
        )
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    
    policy.to(device)
    return policy


def main():
    parser = argparse.ArgumentParser(description='GRPO Training (Modular)')
    parser.add_argument('--env', choices=['direct', 'normalized', 'scalping'], default='direct',
                        help='Environment type')
    parser.add_argument('--policy', choices=['direct', 'normalized', 'grpo'], default='direct',
                        help='Policy type')
    parser.add_argument('--db', dest='db_path', type=str, required=True,
                        help='Database path')
    parser.add_argument('--table', dest='table_name', type=str, default='datasets',
                        help='Table name')
    parser.add_argument('--seq_len', type=int, default=30,
                        help='Sequence length')
    parser.add_argument('--features', type=int, default=28,
                        help='Number of features')
    parser.add_argument('--episode_steps', type=int, default=150,
                        help='Max episode steps')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help='Path to embedding model (for scalping)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--quick_exit_mode', choices=['penalty_only', 'force_close'], 
                        default='penalty_only',
                        help='Quick exit mode for scalping env')

    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension')
    parser.add_argument('--episodes_per_group', type=int, default=6,
                        help='Episodes per group')
    parser.add_argument('--num_groups', type=int, default=3,
                        help='Number of groups')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='Clip epsilon')
    parser.add_argument('--kl_target', type=float, default=0.01,
                        help='KL target')
    parser.add_argument('--entropy_coef', type=float, default=0.05,
                        help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm')
    
    parser.add_argument('--total_timesteps', type=int, default=10000,
                        help='Total timesteps')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Checkpoint interval')
    parser.add_argument('--output_dir', type=str, default='models/grpo_modular',
                        help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MODULAR GRPO TRAINING")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Policy: {args.policy}")
    logger.info("=" * 60)
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 데이터베이스 확인
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        return False
    
    try:
        # 1. 환경 생성
        env = create_environment(args, device)
        logger.info(f"Environment created: {type(env).__name__}")
        
        # 2. 정책 생성
        policy = create_policy(args, env, device)
        logger.info(f"Policy created: {type(policy).__name__}")
        
        # 3. 훈련기 설정
        logger.info("Creating trainer...")
        os.makedirs(args.output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=args.episodes_per_group,
            num_groups=args.num_groups,
            learning_rate=args.lr,
            gamma=args.gamma,
            clip_epsilon=args.clip,
            kl_target=args.kl_target,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        logger.info("Trainer created")
        
        # 4. 훈련 설정
        episodes_per_iteration = args.episodes_per_group * args.num_groups
        total_episodes = (args.total_timesteps // 25) * episodes_per_iteration
        
        logger.info("=" * 60)
        logger.info("Training Configuration:")
        logger.info(f"  Total Timesteps: {args.total_timesteps}")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Episodes per Group: {args.episodes_per_group}")
        logger.info(f"  Number of Groups: {args.num_groups}")
        logger.info(f"  Learning Rate: {args.lr}")
        logger.info(f"  Gamma: {args.gamma}")
        logger.info(f"  Clip Epsilon: {args.clip}")
        logger.info(f"  Entropy Coef: {args.entropy_coef}")
        logger.info("=" * 60)
        
        # 체크포인트 경로
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'checkpoint_iter{}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 훈련 시작
        start_time = time.time()
        
        final_metrics = trainer.train(
            total_episodes=total_episodes,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_path=checkpoint_path
        )
        
        training_time = time.time() - start_time
        
        # 5. 결과 출력
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(args.output_dir, f'{args.env}_{args.policy}_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        logger.info("=" * 60)
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
    sys.exit(0 if success else 1)
