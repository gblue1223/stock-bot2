#!/usr/bin/env python3
"""
GRPO 훈련 스크립트 (Enhanced)

live_trading.py 아키텍처 기반으로 재구성:
- 포괄적인 로깅 시스템 (파일 + 콘솔)
- 설정 관리 클래스
- 강화된 에러 처리
- 진행 상황 모니터링
"""

import os
import sys
import json
import logging
import argparse
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# 환경 변수 로드 (모든 import 전에!)
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.environments import GRPOScalpingEnv
from ai_trader.grpo.policies import GRPOPolicy

# ========================================
# 로깅 설정 (live_trading.py 패턴)
# ========================================

# 로그 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 타임스탬프 포함 로그 파일명
log_filename = os.path.join(log_dir, f'train_scalping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# 로그 포맷
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 파일 핸들러 (상세 로그)
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)  # ✅ DEBUG -> INFO 변경 (속도 향상)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 콘솔 핸들러 (요약 로그)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 루트 로거 설정
logging.basicConfig(
    level=logging.INFO,  # ✅ DEBUG -> INFO 변경
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass



# ========================================
# 설정 관리 클래스 (live_trading.py 패턴)
# ========================================

class TrainingConfig:
    """훈련 설정 관리"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (JSON)
        """
        # 기본 설정 (scalping env + grpo policy만 지원)
        self.env = 'scalping'
        self.policy = 'grpo'
        self.db_path = None
        self.table_name = 'datasets'
        self.seq_len = 60
        self.features = 28
        self.episode_steps = 150
        
        # 임베딩 설정 (scalping env용)
        self.embedding_model = None
        self.embedding_dim = 128
        self.quick_exit_mode = 'penalty_only'
        self.quick_exit_penalty = 0.01  # 기본값
        self.stagnation_exit_seconds = 180  # 기본값
        
        # 정책 설정
        self.hidden_dim = 64
        self.action_dim = 3
        
        # 훈련 설정
        self.episodes_per_group = 6
        self.num_groups = 3
        self.lr = 5e-4
        self.gamma = 0.99
        self.clip = 0.2
        self.kl_target = 0.01
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # ✅ 정규화 설정 (live_trading.py와 일치)
        self.use_raw_data = True  # 원본 데이터 사용 (RollingNormalizer 적용)
        self.rolling_window_size = 1000  # live_trading.py와 동일
        self.rolling_min_samples = 100  # live_trading.py와 동일
        
        # 실행 설정
        self.total_timesteps = 10000
        self.checkpoint_interval = 10
        self.output_dir = 'models/grpo_modular'
        self.load_policy = None
        
        # 디바이스
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 설정 파일에서 로드
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """JSON 파일에서 설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 설정 업데이트
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"[OK] Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise
    
    def save_to_file(self, config_path: str):
        """설정을 JSON 파일로 저장"""
        try:
            config = {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_') and not callable(value)
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[OK] Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}", exc_info=True)
    
    def validate(self):
        """설정 검증"""
        errors = []
        
        if self.db_path is None:
            errors.append("db_path is required")
        elif not os.path.exists(self.db_path):
            errors.append(f"Database not found: {self.db_path}")
        
        if self.env != 'scalping':
            errors.append(f"Only 'scalping' env is supported (got: {self.env})")
        
        if self.policy != 'grpo':
            errors.append(f"Only 'grpo' policy is supported (got: {self.policy})")
        
        if self.embedding_model is None:
            errors.append("embedding_model is required for scalping environment")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("[OK] Configuration validated")


# ========================================
# 환경 및 정책 생성 함수
# ========================================

def create_environment(config: TrainingConfig, device: str):
    """환경 생성"""
    try:
        if config.embedding_model is None:
            raise ValueError("embedding_model is required for scalping environment")
        
        logger.info(f"[LOAD MODEL] Loading embedding model from {config.embedding_model}...")
        from ai_trader.embedding.autoencoder_model import MaskedAutoEncoder
        
        embedding_model = MaskedAutoEncoder(
            input_dim=config.features,
            embedding_dim=config.embedding_dim
        )
        checkpoint = torch.load(config.embedding_model, map_location=device)
        embedding_model.load_state_dict(checkpoint['model_state_dict'])
        embedding_model.to(device)
        embedding_model.eval()
        logger.info("[OK] Embedding model loaded")
        
        logger.info("[CREATE ENV] Creating GRPOScalpingEnv...")
        env = GRPOScalpingEnv(
            embedding_model=embedding_model,
            db_path=config.db_path,
            table_name=config.table_name,
            seq_len=config.seq_len,
            embedding_dim=config.embedding_dim,
            expected_features=config.features,
            transaction_cost_rate=0.00215,
            max_episode_steps=config.episode_steps,
            quick_exit_mode=config.quick_exit_mode,
            quick_exit_threshold=config.stagnation_exit_seconds,
            quick_exit_penalty=config.quick_exit_penalty,
            use_raw_data=config.use_raw_data,  # ✅ 추가
            rolling_window_size=config.rolling_window_size,  # ✅ 추가
            rolling_min_samples=config.rolling_min_samples,  # ✅ 추가
            device=device
        )
        logger.info("[OK] GRPOScalpingEnv created")
        logger.info(f"  Normalization: {'RollingNormalizer' if config.use_raw_data else 'Pre-normalized'}")
        if config.use_raw_data:
            logger.info(f"  Rolling window: {config.rolling_window_size}, min_samples: {config.rolling_min_samples}")
        
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        raise


def create_policy(config: TrainingConfig, env, device: str):
    """정책 생성"""
    try:
        logger.info("[CREATE POLICY] Creating GRPOPolicy...")
        policy = GRPOPolicy(
            embedding_dim=env.embedding_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )
        logger.info(f"[OK] GRPOPolicy created (embedding_dim={env.embedding_dim})")
        
        policy.to(device)
        return policy
    except Exception as e:
        logger.error(f"Failed to create policy: {e}", exc_info=True)
        raise



def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='GRPO Training (Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:

  # JSON 설정 파일 사용
  python ai_trader/grpo/train_scalping.py --config config/training_config.json

  # CLI 인자 사용 (scalping env + grpo policy)
  python ai_trader/grpo/train_scalping.py \
    --db "C:\\Users\\user\\Workspace\\datasets@raw\\datasets_all.duckdb" \
    --embedding_model "models/autoencoder@20251013/model.pt" \
    --seq_len 60 \
    --features 28 \
    --total_timesteps 100000 \
    --use_raw_data true \
    --output_dir "models/grpo_scalping" \
    --load_policy "models/grpo_scalping/checkpoints/checkpoint_iter1650.pt"
        """
    )
    
    # 설정 파일
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration file path (JSON)')
    
    # 데이터
    parser.add_argument('--db', dest='db_path', type=str, default=None,
                        help='Database path')
    parser.add_argument('--table', dest='table_name', type=str, default=None,
                        help='Table name')
    
    # 기타 인자들 (config 파일 오버라이드용)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--features', type=int, default=None)
    parser.add_argument('--episode_steps', type=int, default=None)
    parser.add_argument('--embedding_model', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--quick_exit_mode', choices=['penalty_only', 'force_close'], default=None)
    
    # ✅ 정규화 설정
    parser.add_argument('--use_raw_data', type=bool, default=None,
                        help='Use raw data with RollingNormalizer (default: True)')
    parser.add_argument('--rolling_window_size', type=int, default=None,
                        help='Rolling window size for normalization (default: 1000)')
    parser.add_argument('--rolling_min_samples', type=int, default=None,
                        help='Minimum samples for normalization (default: 100)')
    
    # 정책 설정
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--action_dim', type=int, default=None)
    parser.add_argument('--episodes_per_group', type=int, default=None)
    parser.add_argument('--num_groups', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--kl_target', type=float, default=None)
    parser.add_argument('--entropy_coef', type=float, default=None)
    parser.add_argument('--value_coef', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--total_timesteps', type=int, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--load_policy', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("[START] GRPO Training System (Enhanced)")
    logger.info("=" * 80)
    
    try:
        # 1. 설정 로드
        logger.info("[STEP 1/6] Loading configuration...")
        config = TrainingConfig(args.config)
        
        # CLI 인자로 오버라이드
        for key, value in vars(args).items():
            if value is not None and key != 'config' and hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"  Override: {key} = {value}")
        
        # 설정 검증
        config.validate()
        
        # GPU 확인
        device = config.device
        logger.info(f"[OK] Using device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 2. 환경 생성
        logger.info("[STEP 2/6] Creating environment...")
        env = create_environment(config, device)
        logger.info(f"[OK] Environment: {type(env).__name__}")
        logger.info(f"  Observation space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.n}")
        
        # 3. 정책 생성
        logger.info("[STEP 3/6] Creating policy...")
        policy = create_policy(config, env, device)
        logger.info(f"[OK] Policy: {type(policy).__name__}")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # 기존 모델 로드 (Fine-tuning)
        if config.load_policy:
            logger.info(f"[LOAD POLICY] Loading from {config.load_policy}...")
            try:
                checkpoint = torch.load(config.load_policy, map_location=device)
                
                if 'policy_state_dict' in checkpoint:
                    state_dict = checkpoint['policy_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                missing, unexpected = policy.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"  Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"  Unexpected keys: {len(unexpected)}")
                logger.info("[OK] Policy loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load policy: {e}", exc_info=True)
                raise
        
        # 4. 훈련기 설정
        logger.info("[STEP 4/6] Creating trainer...")
        os.makedirs(config.output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(config.output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=env,
            episodes_per_group=config.episodes_per_group,
            num_groups=config.num_groups,
            learning_rate=config.lr,
            gamma=config.gamma,
            clip_epsilon=config.clip,
            kl_target=config.kl_target,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        logger.info("[OK] Trainer created")
        
        # 5. 훈련 설정 출력
        episodes_per_iteration = config.episodes_per_group * config.num_groups
        total_episodes = (config.total_timesteps // 25) * episodes_per_iteration
        
        logger.info("=" * 80)
        logger.info("[STEP 5/6] Training Configuration:")
        logger.info(f"  Environment: {config.env}")
        logger.info(f"  Policy: {config.policy}")
        logger.info(f"  Total Timesteps: {config.total_timesteps:,}")
        logger.info(f"  Total Episodes: {total_episodes:,}")
        logger.info(f"  Episodes per Group: {config.episodes_per_group}")
        logger.info(f"  Number of Groups: {config.num_groups}")
        logger.info(f"  Learning Rate: {config.lr}")
        logger.info(f"  Gamma: {config.gamma}")
        logger.info(f"  Clip Epsilon: {config.clip}")
        logger.info(f"  Entropy Coef: {config.entropy_coef}")
        logger.info(f"  Checkpoint Interval: {config.checkpoint_interval}")
        logger.info("=" * 80)
        
        # 체크포인트 경로
        checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'checkpoint_iter{}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # 설정 저장
        config_save_path = os.path.join(config.output_dir, 'training_config.json')
        config.save_to_file(config_save_path)
        
        # 6. 훈련 시작
        logger.info("[STEP 6/6] Starting training...")
        start_time = time.time()
        
        final_metrics = trainer.train(
            total_episodes=total_episodes,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_path=checkpoint_path
        )
        
        training_time = time.time() - start_time
        
        # 7. 결과 출력
        logger.info("=" * 80)
        logger.info("[COMPLETE] Training Finished!")
        logger.info("=" * 80)
        logger.info(f"Results:")
        logger.info(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"  Total Episodes: {total_episodes:,}")
        logger.info(f"  Total Timesteps: {final_metrics['total_timesteps']:,}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(config.output_dir, f'{config.env}_{config.policy}_model.pt')
        trainer.save_checkpoint(final_model_path, final_metrics['num_updates'])
        logger.info(f"  Final Model: {final_model_path}")
        
        # 정규화 통계 저장 (실시간 거래용)
        if hasattr(env, 'mean') and hasattr(env, 'std'):
            normalization_stats = {
                'mean': env.mean.cpu().numpy().tolist() if torch.is_tensor(env.mean) else env.mean.tolist(),
                'std': env.std.cpu().numpy().tolist() if torch.is_tensor(env.std) else env.std.tolist()
            }
            stats_path = os.path.join(config.output_dir, 'normalization_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(normalization_stats, f, indent=2)
            logger.info(f"  Normalization Stats: {stats_path}")
        
        logger.info("=" * 80)
        logger.info(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")
        logger.info(f"Log File: {log_filename}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("[FAILED] Training failed with error:")
        logger.error(f"  {type(e).__name__}: {e}")
        logger.error("=" * 80)
        logger.error("Full traceback:", exc_info=True)
        return False
    
    finally:
        # 정리
        if 'env' in locals():
            try:
                env.close()
                logger.debug("Environment closed")
            except Exception as e:
                logger.debug(f"Error closing environment: {e}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
