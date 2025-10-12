#!/usr/bin/env python3
"""
AutoEncoder + GRPO 통합 실행 스크립트

이 스크립트는 테스트된 AutoEncoder 모델(20251012_013148)을 GRPO 에이전트와 통합하여
스캘핑 거래 정책을 훈련합니다.

사용법:
    python integrate_autoencoder_grpo.py --db <database_path> --out <output_dir>
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
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


class AutoEncoderGRPOIntegration:
    """AutoEncoder와 GRPO 통합 클래스"""
    
    def __init__(self, autoencoder_path: str, device: str = 'auto'):
        """
        Args:
            autoencoder_path: AutoEncoder 모델 디렉토리 경로
            device: 사용할 디바이스
        """
        self.autoencoder_path = Path(autoencoder_path)
        
        # 디바이스 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # AutoEncoder 모델 로드
        self.autoencoder, self.model_info = self._load_autoencoder()
        
        logger.info("AutoEncoder model loaded successfully!")
        logger.info(f"Model parameters: {self.model_info['total_params']:,}")
        logger.info(f"Embedding dimension: {self.model_info['config']['embedding_dim']}")
    
    def _load_autoencoder(self):
        """AutoEncoder 모델 로드"""
        # 모델 정보 로드
        with open(self.autoencoder_path / 'model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # 모델 생성
        config = model_info['config']
        data_shape = model_info['data_shape']
        
        model = MaskedAutoEncoder(
            input_dim=data_shape['num_features'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            seq_len=data_shape['seq_len'],
            dropout=config['dropout'],
            mask_ratio=config['mask_ratio']
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(self.autoencoder_path / 'model.pt', map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, model_info
    
    def create_grpo_environment(self, db_path: str, table_name: str = 'datasets', **env_kwargs):
        """GRPO 환경 생성"""
        logger.info("Creating GRPO environment...")
        
        # 환경 설정
        env_config = {
            'embedding_model': self.autoencoder,
            'db_path': db_path,
            'table_name': table_name,
            'seq_len': self.model_info['data_shape']['seq_len'],
            'embedding_dim': self.model_info['config']['embedding_dim'],
            'device': self.device,
            **env_kwargs
        }
        
        env = GRPOScalpingEnv(**env_config)
        logger.info(f"Environment created with embedding dim: {env_config['embedding_dim']}")
        
        return env
    
    def create_grpo_policy(self, hidden_dim: int = 256, action_dim: int = 3):
        """GRPO 정책 네트워크 생성"""
        logger.info("Creating GRPO policy network...")
        
        policy = GRPOPolicy(
            embedding_dim=self.model_info['config']['embedding_dim'],
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )
        
        policy.to(self.device)
        logger.info(f"Policy network created: {self.model_info['config']['embedding_dim']} -> {hidden_dim} -> {action_dim}")
        
        return policy
    
    def train_grpo_agent(
        self,
        db_path: str,
        output_dir: str,
        table_name: str = 'datasets',
        episodes_per_group: int = 12,
        num_groups: int = 4,
        total_timesteps: int = 1000000,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """GRPO 에이전트 훈련"""
        logger.info("=" * 80)
        logger.info("GRPO AGENT TRAINING")
        logger.info("=" * 80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 환경 생성
        env = self.create_grpo_environment(db_path, table_name, **kwargs)
        
        # 2. 정책 네트워크 생성
        policy = self.create_grpo_policy()
        
        # 3. GRPO 트레이너 생성
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=episodes_per_group,
            num_groups=num_groups,
            learning_rate=learning_rate,
            device=self.device
        )
        
        # 4. 훈련 설정 저장
        training_config = {
            'autoencoder_model': str(self.autoencoder_path),
            'embedding_dim': self.model_info['config']['embedding_dim'],
            'seq_len': self.model_info['data_shape']['seq_len'],
            'episodes_per_group': episodes_per_group,
            'num_groups': num_groups,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate,
            'device': self.device,
            **kwargs
        }
        
        with open(output_path / 'grpo_training_config.json', 'w') as f:
            json.dump(training_config, f, indent=2)
        
        logger.info(f"Training configuration saved to: {output_path / 'grpo_training_config.json'}")
        
        # 5. 훈련 실행
        logger.info("Starting GRPO training...")
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info(f"Episodes per group: {episodes_per_group}")
        logger.info(f"Number of groups: {num_groups}")
        
        start_time = time.time()
        
        try:
            # Convert timesteps to episodes (rough estimate)
            estimated_episodes = max(10, total_timesteps // 100)  # Assume ~100 steps per episode
            
            history = trainer.train(
                total_episodes=estimated_episodes,
                checkpoint_interval=max(5, estimated_episodes // 4),
                checkpoint_path=str(output_path / 'checkpoints' / 'checkpoint_{}.pt')
            )
            
            training_time = time.time() - start_time
            
            # 6. 훈련 결과 저장
            training_results = {
                'training_time_minutes': training_time / 60,
                'total_timesteps': total_timesteps,
                'final_metrics': history,  # history is already a dict with final metrics
                'estimated_episodes': estimated_episodes
            }
            
            with open(output_path / 'grpo_training_results.json', 'w') as f:
                json.dump(training_results, f, indent=2)
            
            logger.info("=" * 80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Training time: {training_time/60:.1f} minutes")
            logger.info(f"Results saved to: {output_path}")
            
            if history:
                total_timesteps_trained = history.get('total_timesteps', 0)
                num_updates = history.get('num_updates', 0)
                logger.info(f"Total timesteps trained: {total_timesteps_trained}")
                logger.info(f"Number of policy updates: {num_updates}")
            
            return trainer, history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def test_integration(self, db_path: str, table_name: str = 'datasets', n_samples: int = 100):
        """통합 테스트 실행"""
        logger.info("=" * 80)
        logger.info("INTEGRATION TEST")
        logger.info("=" * 80)
        
        # 1. 환경 생성 테스트
        logger.info("1. Testing environment creation...")
        env = self.create_grpo_environment(db_path, table_name)
        
        # 2. 정책 네트워크 생성 테스트
        logger.info("2. Testing policy network creation...")
        policy = self.create_grpo_policy()
        
        # 3. 환경 리셋 테스트
        logger.info("3. Testing environment reset...")
        obs, info = env.reset()
        logger.info(f"   Initial observation shape: {obs.shape}")
        logger.info(f"   Expected embedding dim: {self.model_info['config']['embedding_dim']}")
        
        # 4. 정책 추론 테스트
        logger.info("4. Testing policy inference...")
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = policy(obs_tensor)
            # 로짓을 확률로 변환
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        
        logger.info(f"   Action logits: {action_logits.cpu().numpy()}")
        logger.info(f"   Action probabilities: {action_probs.cpu().numpy()}")
        logger.info(f"   Selected action: {action}")
        logger.info(f"   State value: {value.item():.6f}")
        
        # 5. 환경 스텝 테스트
        logger.info("5. Testing environment step...")
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"   Next observation shape: {next_obs.shape}")
        logger.info(f"   Reward: {reward:.6f}")
        logger.info(f"   Terminated: {terminated}")
        logger.info(f"   Truncated: {truncated}")
        
        # 6. 성능 벤치마크
        logger.info("6. Performance benchmark...")
        
        times = []
        for i in range(n_samples):
            obs, _ = env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                action_logits, value = policy(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000   # ms
        
        logger.info(f"   Average inference time: {avg_time:.2f}±{std_time:.2f}ms")
        
        if avg_time < 10:
            logger.info("   ✓ Meets real-time trading requirement (< 10ms)")
        else:
            logger.warning("   ⚠ May not meet real-time trading requirement (< 10ms)")
        
        logger.info("✓ Integration test completed successfully!")
        
        return {
            'observation_shape': obs.shape,
            'embedding_dim': self.model_info['config']['embedding_dim'],
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time
        }


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='AutoEncoder + GRPO 통합 실행',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 필수 인수
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='출력 디렉토리'
    )
    
    # 선택적 인수
    parser.add_argument(
        '--autoencoder-model',
        type=str,
        default='models/autoencoder/20251012_013148',
        help='AutoEncoder 모델 디렉토리 경로'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        default='datasets',
        help='데이터베이스 테이블명'
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
        help='총 훈련 스텝 수'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='학습률'
    )
    
    parser.add_argument(
        '--transaction-cost-rate',
        type=float,
        default=0.00215,
        help='거래 비용 비율 (0.215%)'
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
        '--max-holding-time',
        type=int,
        default=60,
        help='최대 보유 시간 (초)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='사용할 디바이스'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='통합 테스트만 실행 (훈련 생략)'
    )
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    logger.info("AutoEncoder + GRPO Integration")
    logger.info("=" * 80)
    logger.info(f"AutoEncoder model: {args.autoencoder_model}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Output directory: {args.out}")
    logger.info(f"Device: {args.device}")
    
    # 경로 확인
    autoencoder_path = Path(args.autoencoder_model)
    if not autoencoder_path.exists():
        logger.error(f"AutoEncoder model not found: {autoencoder_path}")
        return 1
    
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    try:
        # 통합 객체 생성
        integration = AutoEncoderGRPOIntegration(
            autoencoder_path=args.autoencoder_model,
            device=args.device
        )
        
        # 통합 테스트 실행
        test_results = integration.test_integration(
            db_path=args.db,
            table_name=args.table
        )
        
        if args.test_only:
            logger.info("Test-only mode completed successfully!")
            return 0
        
        # GRPO 에이전트 훈련
        trainer, history = integration.train_grpo_agent(
            db_path=args.db,
            output_dir=args.out,
            table_name=args.table,
            episodes_per_group=args.episodes_per_group,
            num_groups=args.num_groups,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            transaction_cost_rate=args.transaction_cost_rate,
            quick_exit_threshold=args.quick_exit_threshold,
            quick_exit_penalty=args.quick_exit_penalty,
            max_holding_time=args.max_holding_time
        )
        
        logger.info("Integration completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())