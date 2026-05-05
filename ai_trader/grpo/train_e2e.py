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

# E2E 환경 임포트
from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv as GRPOScalpingEnvE2E
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.inference.trade_logger import TradeLogger
from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E

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
        self.seq_len = 3000
        self.features = 28
        self.episode_steps = 600   # Fix C: 150 → 600 (더 긴 관찰 시간으로 타이밍 학습 유도)
        
        # 임베딩 사용 안함
        self.quick_exit_mode = 'penalty_only'
        self.quick_exit_penalty = 0.01  # 기본값
        self.stagnation_exit_seconds = 180  # 기본값
        
        # 정책 설정
        self.hidden_dim = 128       # FC hidden dim
        self.cnn_channels = 64      # CNN 필터 수 (VRAM 조절: 64/128/256/512)
        self.rnn_hidden_dim = 128   # GRU 은닉 크기 (VRAM 조절: 128/256/512/1024)
        self.action_dim = 3
        
        # 훈련 설정
        self.episodes_per_group = 16  # Fix B: 6 → 16 (GRPO 그래디언트 품질 향상)
        self.num_groups = 4            # Fix B: 3 → 4 (총 64개 에피소드/iteration)
        self.lr = 5e-4
        self.gamma = 0.99
        self.clip = 0.1  # 0.2 -> 0.1 (안정성 강화)
        self.kl_target = 0.01
        self.entropy_coef = 0.15  # Fix3: 0.05 → 0.15 (정책 탐색 강화, no-trade collapse 방지)
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 64  # VRAM 최적화를 위한 미니배치 크기
        
        
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
        self.early_exit_penalty = 0.2 # 최소 보유 시간 위반 페널티
        self.no_trade_penalty = 10.0  # 거래 안할 시 패널티 (기본값: 10.0)
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
            
            # 하위 호환성
            if 'db_path' not in config and 'parquet_path' in config:
                logger.warning("parquet_path found but E2E needs raw db_path.")
                
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
        
        # E2E: db_path required
        if not self.db_path:
             errors.append("db_path is required for E2E validation")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("[OK] Configuration validated")


# ========================================
# 모델 및 환경 생성 함수
# ========================================

def load_embedding_model(config: TrainingConfig, device: str):
    """임베딩 모델 로드 (공유용) - E2E 환경에서는 사용되지 않음"""
    return None

def create_environment(config: TrainingConfig, device: str):
    """환경 생성 (단일 인스턴스)"""
    try:
        if not config.db_path:
             raise ValueError(f"db_path is required for E2E env. Config dictionary: {config.__dict__}")

        logging.info(f"Using Raw DB from: {config.db_path}")
        
        env = GRPOScalpingEnvE2E(
            db_path=config.db_path,
            table_name=config.table_name,
            seq_len=config.seq_len,
            expected_features=config.features,
            transaction_cost_rate=config.transaction_cost_rate,
            quick_exit_mode=config.quick_exit_mode,
            quick_exit_penalty=config.quick_exit_penalty,
            max_episode_steps=config.episode_steps,
            base_price=config.base_price,
            stop_loss_pct=config.stop_loss_pct,
            max_split_count=config.max_split_count,
            min_holding_time=config.min_holding_time,
            max_holding_time=config.max_holding_time,
            early_exit_penalty=config.early_exit_penalty,
            no_trade_penalty=config.no_trade_penalty
        )
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        raise



def create_policy(config: TrainingConfig, env, device: str):
    """정책 생성"""
    try:
        logger.info("[CREATE POLICY] Creating GRPOPolicyE2E...")
        # env가 리스트일 수 있으므로 첫 번째 요소 사용
        ref_env = env[0] if isinstance(env, list) else env
        
        policy = GRPOPolicyE2E(
            obs_dim=ref_env.observation_space.shape[-1],
            cnn_channels=config.cnn_channels,
            rnn_hidden_dim=config.rnn_hidden_dim,
            fc_hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )
        logger.info(f"[OK] GRPOPolicyE2E created: obs_dim={ref_env.observation_space.shape[-1]}, "
                   f"cnn={config.cnn_channels}, rnn={config.rnn_hidden_dim}, fc={config.hidden_dim}")
        
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

  # CLI 인자 사용 (E2E)
  python ai_trader/grpo/train_e2e.py \
    --db "C:\\Users\\user\\Workspace\\datasets@raw\\datasets_all.duckdb" \
    --seq_len 3000 \
    --features 28 \
    --total_timesteps 100000 \
    --use_raw_data true \
    --output_dir "models/grpo_e2e"
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
    parser.add_argument('--early_exit_penalty', type=float, default=None,
                        help='Penalty for early exit before min_holding (default: 0.2)')
    parser.add_argument('--no_trade_penalty', type=float, default=None,
                        help='Penalty for making 0 trades in an episode (default: 10.0)')
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
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='FC hidden layer dim (default: 128)')
    parser.add_argument('--cnn_channels', type=int, default=None,
                        help='CNN filter count (default:64, large:128, xlarge:256)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=None,
                        help='GRU hidden size (default:128, large:256, xlarge:512)')
    parser.add_argument('--vram_preset', choices=['small', 'medium', 'large', 'xlarge'], default=None,
                        help=(
                            'VRAM 사용량 프리셋 (개별 옵션보다 우선 적용).\n'
                            '  small  : cnn=64,  rnn=128, fc=128, batch=64   (~4GB)\n'
                            '  medium : cnn=128, rnn=256, fc=256, batch=128  (~8GB)\n'
                            '  large  : cnn=256, rnn=512, fc=512, batch=256  (~16GB)\n'
                            '  xlarge : cnn=512, rnn=1024,fc=1024,batch=512  (~24GB+)'
                        ))
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
    parser.add_argument('--batch_size', type=int, default=None, help='Mini-batch size for PPO updates (default: 64)')
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
            if value is not None and key not in ('config', 'vram_preset') and hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"  Override: {key} = {value}")
        
        # vram_preset 적용 (개별 CLI보다 나중에 적용하여 최종 우선권 부여)
        VRAM_PRESETS = {
            'small':  dict(cnn_channels=64,  rnn_hidden_dim=128,  hidden_dim=128,  batch_size=64),
            'medium': dict(cnn_channels=128, rnn_hidden_dim=256,  hidden_dim=256,  batch_size=128),
            'large':  dict(cnn_channels=256, rnn_hidden_dim=512,  hidden_dim=512,  batch_size=256),
            'xlarge': dict(cnn_channels=512, rnn_hidden_dim=1024, hidden_dim=1024, batch_size=512),
        }
        if args.vram_preset:
            preset = VRAM_PRESETS[args.vram_preset]
            for k, v in preset.items():
                setattr(config, k, v)
            logger.info(f"[VRAM Preset '{args.vram_preset}'] cnn={config.cnn_channels}, "
                       f"rnn={config.rnn_hidden_dim}, fc={config.hidden_dim}, "
                       f"batch={config.batch_size}")
        
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
        
        # E2E 환경은 IPC 파이프 크기 한계로 인해 Windows에서 SubprocVecEnv가 자주 터짐 (BrokenPipeError)
        # Linux(Colab)에서는 정상 작동하므로 OS 체크 후 적용
        if config.num_workers > 1 and sys.platform == 'win32':
             logger.warning(f"Overriding num_workers={config.num_workers} to 1. SubprocVecEnv IPC crashes with seq_len=3000 in E2E on Windows.")
             config.num_workers = 1
             
        env_fns = [functools.partial(create_environment, config, device) for _ in range(config.num_workers)]
        
        if config.num_workers > 1:
            logger.info(f"Initializing {config.num_workers} processes (SubprocVecEnv) for parallel processing...")
            vec_env = SubprocVecEnv(env_fns)
        else:
            logger.info(f"Initializing 1 processes (DummyVecEnv)...")
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
                
                # ✅ 체크포인트 아키텍처 자동 감지 (shape 불일치 방지)
                # conv1.weight: (out_channels, in_channels, kernel) -> cnn_channels
                # gru.weight_hh_l0: (3*hidden, hidden)             -> rnn_hidden_dim
                # fc1.weight: (hidden_out, hidden_in)              -> hidden_dim
                ckpt_cnn = state_dict['conv1.weight'].shape[0]     if 'conv1.weight'     in state_dict else None
                ckpt_rnn = state_dict['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in state_dict else None
                ckpt_fc  = state_dict['fc1.weight'].shape[0]       if 'fc1.weight'       in state_dict else None
                
                needs_rebuild = (
                    (ckpt_cnn is not None and ckpt_cnn != config.cnn_channels) or
                    (ckpt_rnn is not None and ckpt_rnn != config.rnn_hidden_dim) or
                    (ckpt_fc  is not None and ckpt_fc  != config.hidden_dim)
                )
                
                if needs_rebuild:
                    logger.warning(
                        f"[ARCH MISMATCH] 체크포인트 아키텍처 "
                        f"(cnn={ckpt_cnn}, rnn={ckpt_rnn}, fc={ckpt_fc}) != "
                        f"현재 config (cnn={config.cnn_channels}, rnn={config.rnn_hidden_dim}, fc={config.hidden_dim}) "
                        f"-> 체크포인트에 맞게 정책을 재생성합니다."
                    )
                    if ckpt_cnn is not None: config.cnn_channels   = ckpt_cnn
                    if ckpt_rnn is not None: config.rnn_hidden_dim = ckpt_rnn
                    if ckpt_fc  is not None: config.hidden_dim      = ckpt_fc
                    policy = create_policy(config, ref_env, device)
                    total_params = sum(p.numel() for p in policy.parameters())
                    logger.info(f"  Rebuilt policy: cnn={config.cnn_channels}, rnn={config.rnn_hidden_dim}, fc={config.hidden_dim} ({total_params:,} params)")
                
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
            batch_size=config.batch_size,
            device=device,
            tensorboard_log_dir=tensorboard_dir
        )
        
        # 7. 커리큘럼 러닝 콜백 정의
        
        # Set transaction cost for all environments
        current_cost_rate = 0.0
        target_cost_rate = 0.00215
        
        # Use first env to determine initial settings if not overridden
        initial_env_cost = ref_env.transaction_cost_rate
        
        def _calc_curriculum_cost(iter_num: int, total_iter: int, t_rate: float) -> float:
             """주어진 iteration 번호에 맞는 curriculum 수수료 계산"""
             progress = iter_num / total_iter
             if progress < 0.05:
                 return 0.0
             elif progress < 0.20:
                 ratio = (progress - 0.05) / 0.15
                 return t_rate * ratio
             else:
                 return t_rate

        if hasattr(args, 'transaction_cost_rate') and args.transaction_cost_rate is not None and args.transaction_cost_rate > 0:
             target_cost_rate = args.transaction_cost_rate
        else:
             if initial_env_cost > 0:
                 target_cost_rate = initial_env_cost
             else:
                 current_cost_rate = initial_env_cost
                 target_cost_rate = initial_env_cost
                 logger.info(f"Curriculum Learning: Transaction cost already {current_cost_rate}. No curriculum applied.")

        if target_cost_rate > 0:
             # 재시작 시: 항상 cost=0에서 시작하고 warmup 동안 점진적으로 올림
             # (이전 학습이 cost=0으로 진행됐을 수 있으므로 갑작스러운 충격 방지)
             WARMUP_ITERS = 200  # 재시작 후 200 iter 동안 0 → resume_target 선형 증가
             resume_iter = start_iteration  # e.g. 1320 (체크포인트 파일명에서 파싱)
             total_iter_for_calc = getattr(config, 'total_iterations', 8000)
             resume_target_cost = _calc_curriculum_cost(resume_iter, total_iter_for_calc, target_cost_rate)

             # warmup 시작: cost=0
             initial_curriculum_cost = 0.0
             logger.info(
                 f"Curriculum Learning: Warmup start cost=0.0 → {resume_target_cost:.5f} "
                 f"over {WARMUP_ITERS} iters (then resume normal schedule from iter {resume_iter}). "
                 f"Target={target_cost_rate}"
             )
             vec_env.env_method('set_transaction_cost_rate', initial_curriculum_cost)
             current_cost_rate = initial_curriculum_cost

        def curriculum_callback(iteration: int, metrics: dict):
            """Curriculum Learning: resume warmup 후 정상 스케줄"""
            nonlocal current_cost_rate
            nonlocal target_cost_rate

            if current_cost_rate >= target_cost_rate:
                return

            total_iterations = metrics.get('total_iterations', 8000)

            # ── Phase 1: Resume Warmup ─────────────────────────────────────────
            # 재시작이 있는 경우(start_iteration > 0):
            # cost=0 → resume_target으로 WARMUP_ITERS 동안 선형 증가
            if start_iteration > 0:
                iters_since_resume = iteration - start_iteration
                if iters_since_resume <= WARMUP_ITERS:
                    warmup_ratio = iters_since_resume / WARMUP_ITERS
                    new_cost_rate = resume_target_cost * warmup_ratio
                    if abs(new_cost_rate - current_cost_rate) > 1e-7:
                        logger.info(
                            f"Curriculum Warmup ({iters_since_resume}/{WARMUP_ITERS}): "
                            f"Transaction cost {current_cost_rate:.6f} → {new_cost_rate:.6f}"
                        )
                        current_cost_rate = new_cost_rate
                        vec_env.env_method('set_transaction_cost_rate', current_cost_rate)
                    return  # warmup 중에는 일반 스케줄 skip

            # ── Phase 2: 정상 Curriculum 스케줄 ───────────────────────────────
            # 0~5%: cost=0  /  5~20%: 선형 증가  /  20%+: 목표 유지
            progress = iteration / total_iterations
            if progress < 0.05:
                new_cost_rate = 0.0
            elif progress < 0.20:
                ratio = (progress - 0.05) / 0.15
                new_cost_rate = target_cost_rate * ratio
            else:
                new_cost_rate = target_cost_rate

            if abs(new_cost_rate - current_cost_rate) > 1e-7:
                logger.info(
                    f"Curriculum Update (progress={progress:.1%}): "
                    f"Transaction cost {current_cost_rate:.6f} → {new_cost_rate:.6f}"
                )
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
        if config.use_raw_data and hasattr(ref_env, 'normalizer') and ref_env.normalizer is not None:
            # RollingNormalizer does not store global running mean/var
            # We just note that normalization is enabled
            norm_config = {
                'type': 'RollingNormalizer',
                'window_size': ref_env.normalizer.window_size,
                'min_samples': ref_env.normalizer.min_samples
            }
            
            stats_path = os.path.join(config.output_dir, 'normalizer.json')
            with open(stats_path, 'w') as f:
                json.dump(norm_config, f, indent=2)
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
