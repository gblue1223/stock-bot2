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

# V2 환경 임포트
from ai_trader.grpo.environments.scalping_env_v2 import GRPOScalpingEnvV2
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.inference.trade_logger import TradeLogger
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
# Argument Parsing for Logging Level (Done early to configure logging)
parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
args_pre, _ = parser_pre.parse_known_args()

log_level = logging.DEBUG if args_pre.debug else logging.INFO

if args_pre.debug:
    print("DEBUG logging enabled")
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

# 루트 로거 설정
logging.basicConfig(
    level=log_level,
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
        self.seq_len = 120
        self.features = 28
        self.episode_steps = 150
        
        # 임베딩 설정 (scalping env용)
        self.embedding_model = None
        self.embedding_dim = 128
        self.parquet_path = 'datasets/parquet' # V2: Pre-computed embeddings path (default)
        self.quick_exit_mode = 'penalty_only'
        self.quick_exit_penalty = 0.01  # 기본값
        self.stagnation_exit_seconds = 180  # 기본값
        
        # 정책 설정
        self.hidden_dim = 128  # 64 -> 128 (네트워크 용량 증가)
        self.action_dim = 3
        
        # 훈련 설정
        self.episodes_per_group = 6
        self.num_groups = 3
        self.lr = 5e-4
        self.gamma = 0.99
        self.clip = 0.1  # 0.2 -> 0.1 (안정성 강화)
        self.kl_target = 0.01
        self.entropy_coef = 0.15  # Fix3: 0.05 → 0.15 (정책 탐색 강화, no-trade collapse 방지)
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
        self.num_workers = 4  # ✅ 병렬 작업자 수 추가 (기본값: 4)
        
        # ✅ 손절 및 분할 매수 설정
        self.base_price = 100000.0    # 기준 가격 (기본값: 10만원)
        self.stop_loss_pct = 2.0      # 손절 퍼센트 (2.0%)
        self.max_split_count = 1      # 최대 분할 매수 횟수 (1 = 단일 진입)
        self.min_holding_time = 2     # 최소 보유 시간 (초)
        self.max_holding_time = 100   # 최대 보유 시간 (초)
        self.transaction_cost_rate = 0.00215  # 거래 수수료율
        
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
            
            # 하위 호환성 (V1 config -> V2 config)
            if 'embedding_model' in config and 'parquet_path' not in config:
                config['parquet_path'] = config['embedding_model']
                
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
        # V2 환경에서는 db_path가 더 이상 필수가 아님 (parquet_path 사용)
        if self.db_path and not os.path.exists(self.db_path):
            errors.append(f"Database not found: {self.db_path}")
        
        if self.env != 'scalping':
            errors.append(f"Only 'scalping' env is supported (got: {self.env})")
        
        if self.policy != 'grpo':
            errors.append(f"Only 'grpo' policy is supported (got: {self.policy})")
        
        # V2: embedding_model OR parquet_path required
        # CLI 인자로 들어오는 경우가 있어서 여기서 엄격하게 체크하면 실패할 수 있음
        # 나중에 환경 생성 직전에 체크하도록 변경
        # if self.embedding_model is None and getattr(self, 'parquet_path', None) is None:
        #      errors.append("embedding_model or parquet_path is required")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("[OK] Configuration validated")


# ========================================
# 모델 및 환경 생성 함수
# ========================================

def load_embedding_model(config: TrainingConfig, device: str):
    """임베딩 모델 로드 (공유용) - GRPOScalpingEnvV2에서는 사용되지 않음"""
    # V2 환경에서는 임베딩 모델을 직접 로드하지 않고 Parquet 파일을 사용합니다.
    # 하지만 기존 코드가 이 함수를 호출할 수 있으므로 None을 반환하거나 더미를 반환.
    return None

def create_environment(config: TrainingConfig, device: str):
    """환경 생성 (단일 인스턴스)"""
    try:
        # V2 환경: parquet_path가 없으면 embedding_model을 parquet_path로 간주 (config.json 호환성)
        logger.debug(f"Config keys inside create_environment: {config.__dict__}")
        parquet_path = getattr(config, 'parquet_path', None)
        if not parquet_path:
            parquet_path = getattr(config, 'embedding_model', None)
            
        if not parquet_path:
             raise ValueError(f"parquet_path or embedding_model is required for V2 env. Config dictionary: {config.__dict__}")

        logging.info(f"Using Embeddings from: {parquet_path}")
        
        env = GRPOScalpingEnvV2(
            parquet_path=parquet_path,
            embedding_dim=config.embedding_dim,
            transaction_cost_rate=config.transaction_cost_rate,
            quick_exit_mode=config.quick_exit_mode,
            quick_exit_penalty=config.quick_exit_penalty,
            max_episode_steps=config.episode_steps,
            base_price=config.base_price,
            stop_loss_pct=config.stop_loss_pct,
            max_split_count=config.max_split_count,
            min_holding_time=config.min_holding_time,
            max_holding_time=config.max_holding_time,
            device=device
        )
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        raise



def create_policy(config: TrainingConfig, env, device: str):
    """정책 생성"""
    try:
        logger.info("[CREATE POLICY] Creating GRPOPolicy...")
        # env가 리스트일 수 있으므로 첫 번째 요소 사용
        ref_env = env[0] if isinstance(env, list) else env
        
        policy = GRPOPolicy(
            embedding_dim=ref_env.observation_space.shape[0],
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )
        logger.info(f"[OK] GRPOPolicy created (embedding_dim={ref_env.embedding_dim})")
        
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
    --embedding_model "models/autoencoder@20260109/model.pt" \
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
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
    
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
    parser.add_argument('--parquet_path', type=str, default=None,
                        help='Path to parquet embeddings directory (e.g. embeddings_v3)')
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--quick_exit_mode', choices=['penalty_only', 'force_close'], default=None)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel environment workers')
    
    # ✅ 손절 및 분할 매수 설정
    parser.add_argument('--stop_loss', dest='stop_loss_pct', type=float, default=None,
                        help='Stop loss percentage (default: 2.0)')
    parser.add_argument('--max_split', dest='max_split_count', type=int, default=None,
                        help='Max split buy count (default: 1)')
    parser.add_argument('--min_holding', dest='min_holding_time', type=float, default=None,
                        help='Min holding time in seconds (default: 2)')
    parser.add_argument('--max_holding', dest='max_holding_time', type=float, default=None,
                        help='Max holding time in seconds (default: 100)')
    parser.add_argument('--transaction_cost', dest='transaction_cost_rate', type=float, default=None,
                        help='Target transaction cost rate (default: 0.00215)')
    
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
    
    device = "cpu"
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
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU")
            device = 'cpu'
            config.device = device
            
        logger.info(f"[OK] Using device: {device}")
        if device == 'cuda':
            try:
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except Exception as e:
                logger.warning(f"Could not retrieve CUDA info: {e}")
        
        # 2. 환경 생성
        # 2. 환경 생성 (병렬 처리 지원)
        logger.info(f"[STEP 2/6] Creating environments (Workers: {config.num_workers})...")
        
        # 임베딩 모델 로드 (한 번만 수행하여 공유)
        embedding_model = load_embedding_model(config, device)
        # 데이터 병렬 처리를 위한 VecEnv 도입
        from ai_trader.grpo.environments import SubprocVecEnv, DummyVecEnv
        import functools
        
        # 워커 수만큼 환경 생성자 함수 리스트 생성
        env_fns = [functools.partial(create_environment, config, device) for _ in range(config.num_workers)]
            
        if config.num_workers > 1:
            logger.info(f"Initializing {config.num_workers} processes (SubprocVecEnv)...")
            vec_env = SubprocVecEnv(env_fns)
        else:
            logger.info(f"Initializing 1 process (DummyVecEnv) to prevent Memory Limits...")
            vec_env = DummyVecEnv(env_fns)
            
        logger.info(f"[OK] Environments created")
        
        # 로깅을 위해 첫 번째 환경(생성하여 참조만 확인)
        ref_env = create_environment(config, device)
        logger.info(f"  Observation space: {ref_env.observation_space.shape}")
        logger.info(f"  Action space: {ref_env.action_space.n}")
        logger.info(f"  Avg Transaction Cost: {ref_env.transaction_cost_rate}")
        if config.use_raw_data:
            logger.info(f"  Normalization: RollingNormalizer (Window: {config.rolling_window_size})")

        # 3. 정책 생성
        logger.info("[STEP 3/6] Creating policy...")
        policy = create_policy(config, ref_env, device) # 리스트 대신 ref_env 참조
        logger.info(f"[OK] Policy: {type(policy).__name__}")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # 기존 모델 로드 (Fine-tuning)
        start_iteration = 0  # 기본값 초기화
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
                
                # 체크포인트 파일명에서 시작 반복 횟수 추출 (load_policy가 설정된 경우)
                try:
                    import re
                    match = re.search(r'checkpoint_iter(\d+)\.pt', config.load_policy)
                    if match:
                        start_iteration = int(match.group(1))
                        logger.info(f"Resuming from iteration: {start_iteration}")
                except Exception as e:
                    logger.warning(f"Could not extract iteration number from checkpoint path: {e}")
                    start_iteration = 0
            except Exception as e:
                logger.error(f"Failed to load policy: {e}", exc_info=True)
                raise
        
        # 4. 훈련기 설정
        logger.info("[STEP 4/6] Creating trainer...")
        os.makedirs(config.output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(config.output_dir, 'tensorboard_logs')
        # 6. 트레이너 생성
        trainer = GRPOTrainer(
            policy=policy,
            env=vec_env,  # ✅ 환경 리스트가 아닌 VecEnv 인스턴스 전달
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
        
        # 7. 커리큘럼 러닝 콜백 정의
        
        # Set transaction cost for all environments
        current_cost_rate = 0.0
        target_cost_rate = 0.00215
        
        # Use first env to determine initial settings if not overridden
        initial_env_cost = ref_env.transaction_cost_rate
        
        if hasattr(args, 'transaction_cost_rate') and args.transaction_cost_rate is not None and args.transaction_cost_rate > 0:
             target_cost_rate = args.transaction_cost_rate
             # Fix4: 0.0 대신 목표값의 30%로 초기화 (no-trade trivial solution 차단)
             initial_curriculum_cost = target_cost_rate * 0.30
             logger.info(f"Curriculum Learning: Starting with {initial_curriculum_cost:.5f} transaction cost (30% of target {target_cost_rate}), targeting {target_cost_rate}")
             vec_env.env_method('set_transaction_cost_rate', initial_curriculum_cost)
             current_cost_rate = initial_curriculum_cost
        else:
             if initial_env_cost > 0:
                 target_cost_rate = initial_env_cost
                 # Fix4: 0.0 대신 목표값의 30%로 초기화 (no-trade trivial solution 차단)
                 initial_curriculum_cost = target_cost_rate * 0.30
                 logger.info(f"Curriculum Learning: Starting with {initial_curriculum_cost:.5f} transaction cost (30% of target {target_cost_rate}), targeting {target_cost_rate}")
                 vec_env.env_method('set_transaction_cost_rate', initial_curriculum_cost)
                 current_cost_rate = initial_curriculum_cost
             else:
                 current_cost_rate = initial_env_cost
                 target_cost_rate = initial_env_cost
                 logger.info(f"Curriculum Learning: Transaction cost already {current_cost_rate}. No curriculum applied.")

        def curriculum_callback(iteration: int, metrics: dict):
            nonlocal current_cost_rate
            nonlocal target_cost_rate
            
            if current_cost_rate >= target_cost_rate:
                return

            win_rate = metrics.get('mean_win_rate', 0.0)
            
            if win_rate > 0.65:
                # 30% 증가 설정
                step_size = target_cost_rate * 0.30
                
                # 새로운 수수료율 계산 (목표값 초과 방지)
                new_cost_rate = min(current_cost_rate + step_size, target_cost_rate)
                
                if new_cost_rate > current_cost_rate:
                    logger.info(f"Curriculum Update: Win rate {win_rate:.1f}% > 65%. "
                              f"Increasing transaction cost: {current_cost_rate:.5f} -> {new_cost_rate:.5f}")
                    current_cost_rate = new_cost_rate
                    vec_env.env_method('set_transaction_cost_rate', current_cost_rate)
                    
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
        logger.info(f"  Parallel Workers: {config.num_workers}")
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
            checkpoint_path=checkpoint_path,
            on_iteration_end=curriculum_callback,
            start_iteration=start_iteration
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
        
        # 정규화 통계 저장 (첫 번째 환경 기준)
        if hasattr(ref_env, 'normalizer') and ref_env.normalizer is not None:
            normalization_stats = {
                'mean': ref_env.normalizer.running_mean.tolist(),
                'std': ref_env.normalizer.running_std.tolist(),
                'count': int(ref_env.normalizer.count)
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
        if 'envs' in locals():
            for i, e in enumerate(envs):
                try:
                    e.close()
                except:
                    pass
            logger.debug(f"Closed {len(envs)} environments")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
