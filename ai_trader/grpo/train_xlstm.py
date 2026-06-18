#!/usr/bin/env python3
"""
xLSTM 기반 GRPO 훈련 스크립트 (train_xlstm.py)

재활용 가능 데이터 추출기(data_extractor.py)와 scalping_env_xlstm.py를 사용하여
최고 효율로 학습을 진행하고, GRU 체크포인트로부터 CNN 가중치를 부분 전이할 수 있습니다.
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

load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.inference.trade_logger import TradeLogger
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM



# 로깅 설정
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'train_xlstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument('--debug', action='store_true')
args_pre, _ = parser_pre.parse_known_args()
log_level = logging.DEBUG if args_pre.debug else logging.INFO

if args_pre.debug:
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)


class TrainingConfig:
    """훈련 설정 관리"""
    def __init__(self, config_path: Optional[str] = None):
        self.env = 'scalping'
        self.policy = 'xlstm'
        self.db_path = None
        self.extracted_dir = 'data/extracted_episodes'  # ✅ 기본값 추가
        self.table_name = 'datasets'
        self.seq_len = 3000
        self.features = 28
        self.episode_steps = 600
        
        self.quick_exit_mode = 'penalty_only'
        self.quick_exit_penalty = 0.01
        self.stagnation_exit_seconds = 180
        
        self.hidden_dim = 128
        self.cnn_channels = 64
        self.rnn_hidden_dim = 128
        self.action_dim = 3
        
        self.episodes_per_group = 16
        self.num_groups = 4
        self.lr = 5e-4
        self.gamma = 0.99
        self.clip = 0.1
        self.kl_target = 0.01
        self.entropy_coef = 0.15
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 64
        
        self.use_raw_data = True
        self.rolling_window_size = 1000
        self.rolling_min_samples = 100
        
        self.total_timesteps = 10000
        self.checkpoint_interval = 10
        self.output_dir = 'models/grpo_xlstm'
        self.load_policy = None
        self.num_workers = 4
        
        self.base_price = 100000.0
        self.stop_loss_pct = 2.0
        self.max_split_count = 1
        self.min_holding_time = 2
        self.max_holding_time = 100
        self.early_exit_penalty = 0.2
        self.no_trade_penalty = 10.0
        self.transaction_cost_rate = 0.00015  # 기본 거래 수수료율 (0.015%)
        self.buy_tax_rate = 0.0                # 매수 세금 (0%)
        self.sell_tax_rate = 0.0018            # 매도 세금 (0.18%)
        self.max_trades_per_episode = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
            
    def load_from_file(self, config_path: str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            logger.info(f"[OK] Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def save_to_file(self, config_path: str):
        try:
            config = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}", exc_info=True)

    def validate(self):
        errors = []
        # cache 디렉토리나 DB 파일 둘 중 하나는 존재해야 함
        if not self.extracted_dir and not self.db_path:
            errors.append("Either db_path or extracted_dir is required")
        elif self.extracted_dir and not os.path.exists(self.extracted_dir):
            logger.warning(f"extracted_dir ({self.extracted_dir}) not found. Falling back to DB checking.")
            if not self.db_path:
                errors.append("extracted_dir not found and no db_path specified")
            elif not os.path.exists(self.db_path):
                errors.append(f"Neither extracted_dir nor db_path exists: {self.db_path}")
                
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info("[OK] Configuration validated")


def create_environment(config: TrainingConfig, device: str):
    """환경 인스턴스 생성"""
    try:
        return GRPOScalpingEnvXLSTM(
            db_path=config.db_path,
            table_name=config.table_name,
            seq_len=config.seq_len,
            expected_features=config.features,
            transaction_cost_rate=config.transaction_cost_rate,
            buy_tax_rate=config.buy_tax_rate,
            sell_tax_rate=config.sell_tax_rate,
            quick_exit_mode=config.quick_exit_mode,
            quick_exit_penalty=config.quick_exit_penalty,
            max_episode_steps=config.episode_steps,
            base_price=config.base_price,
            stop_loss_pct=config.stop_loss_pct,
            max_split_count=config.max_split_count,
            min_holding_time=config.min_holding_time,
            max_holding_time=config.max_holding_time,
            early_exit_penalty=config.early_exit_penalty,
            no_trade_penalty=config.no_trade_penalty,
            max_trades_per_episode=config.max_trades_per_episode,
            extracted_dir=config.extracted_dir
        )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        raise


def create_policy(config: TrainingConfig, env, device: str):
    """정책 생성"""
    try:
        ref_env = env[0] if isinstance(env, list) else env
        policy = GRPOPolicyE2EXLSTM(
            obs_dim=ref_env.observation_space.shape[-1],
            cnn_channels=config.cnn_channels,
            rnn_hidden_dim=config.rnn_hidden_dim,
            fc_hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )
        policy.to(device)
        return policy
    except Exception as e:
        logger.error(f"Failed to create policy: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='xLSTM 기반 GRPO E2E 훈련')
    parser.add_argument('--config', type=str, default=None, help='JSON 설정 파일')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
    
    # 데이터
    parser.add_argument('--db', dest='db_path', type=str, default=None, help='Database path')
    parser.add_argument('--table', dest='table_name', type=str, default=None, help='Table name')
    parser.add_argument('--extracted_dir', type=str, default=None, help='추출 데이터 디렉토리 경로')
    
    # 기타 인자들
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--features', type=int, default=None)
    parser.add_argument('--episode_steps', type=int, default=None)
    parser.add_argument('--quick_exit_mode', choices=['penalty_only', 'force_close'], default=None)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel environment workers')
    
    # 손절 및 분할 매수 설정
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
                        help='Target transaction cost rate (default: 0.00015)')
    parser.add_argument('--buy_tax', dest='buy_tax_rate', type=float, default=None,
                        help='Buy tax rate (default: 0.0)')
    parser.add_argument('--sell_tax', dest='sell_tax_rate', type=float, default=None,
                        help='Sell tax rate (default: 0.0018)')
    parser.add_argument('--max_trades_per_episode', type=int, default=None,
                        help='Max trades allowed per episode (default: None/unlimited)')
    
    # 정규화 설정
    parser.add_argument('--use_raw_data',
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=None,
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
                        help='xLSTM hidden size (default:128, large:256, xlarge:512)')
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
    logger.info("[START] xLSTM-GRPO Training System")
    logger.info("=" * 80)
    
    device = "cpu"
    try:
        config = TrainingConfig(args.config)
        
        # VRAM Preset 적용
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
                
        # CLI 인자 오버라이드
        for key, value in vars(args).items():
            if value is not None and key not in ('config', 'vram_preset') and hasattr(config, key):
                setattr(config, key, value)
                
        config.validate()
        device = config.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            config.device = device
            
        logger.info(f"[OK] Using device: {device}")
        
        # 환경 생성 (extracted_dir 캐시 활용 시 Windows의 SubprocVecEnv도 안정 작동)
        logger.info(f"[STEP 2/6] Creating environments (Workers: {config.num_workers})...")
        from ai_trader.grpo.environments import SubprocVecEnv, DummyVecEnv
        import functools
        
        # 만약 db_path를 직접 쓰면서 Windows인 경우에는 multi-process 크래시 방지
        if config.extracted_dir is None and config.num_workers > 1 and sys.platform == 'win32':
             logger.warning("No extracted_dir provided, fallback to single process to prevent IPC crash on Windows.")
             config.num_workers = 1
             
        env_fns = [functools.partial(create_environment, config, device) for _ in range(config.num_workers)]
        
        if config.num_workers > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
            
        ref_env = vec_env.envs[0] if config.num_workers <= 1 else create_environment(config, device)
        
        # 정책 모델 생성
        logger.info("[STEP 3/6] Creating policy...")
        policy = create_policy(config, ref_env, device)
        
        # 모델 파라미터 수 로깅
        total_params = sum(p.numel() for p in policy.parameters())
        logger.info(f"  Total parameters: {total_params:,}")
        
        start_iteration = 0
        if config.load_policy:
            logger.info(f"[LOAD POLICY] Loading from {config.load_policy}...")
            try:
                checkpoint = torch.load(config.load_policy, map_location=device)
                state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
                
                # GRU 체크포인트 감지 시 CNN 가중치 전이(Transfer Learning) 적용
                has_gru = any('gru' in k for k in state_dict.keys())
                if has_gru:
                    logger.info("⚠️ GRU weights detected in checkpoint. Transferring CNN layers only. xLSTM/FC will be initialized randomly.")
                    cnn_state_dict = {k: v for k, v in state_dict.items() if k.startswith(('conv', 'bn'))}
                    missing, unexpected = policy.load_state_dict(cnn_state_dict, strict=False)
                    logger.info(f"  Transfer completed. CNN loaded successfully.")
                else:
                    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
                    logger.info("  Checkpoint loaded successfully.")
                    
                # Iteration 정보 복원
                try:
                    import re
                    match = re.search(r'checkpoint_iter(\d+)\.pt', config.load_policy)
                    if match:
                        start_iteration = int(match.group(1))
                        logger.info(f"Resuming from iteration: {start_iteration}")
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to load policy: {e}", exc_info=True)
                raise
                

            
        # 트레이너 생성
        logger.info("[STEP 4/6] Creating trainer...")
        os.makedirs(config.output_dir, exist_ok=True)
        tensorboard_dir = os.path.join(config.output_dir, 'tensorboard_logs')
        
        trainer = GRPOTrainer(
            policy=policy,
            env=vec_env,
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
        
        # 커리큘럼 콜백 설정 (train_e2e와 동일)
        current_cost_rate = 0.0
        target_cost_rate = config.transaction_cost_rate
        
        vec_env.env_method('set_transaction_cost_rate', current_cost_rate)
        
        def curriculum_callback(iteration: int, metrics: dict):
            nonlocal current_cost_rate
            nonlocal target_cost_rate
            
            trainer.extra_checkpoint_state = {'current_cost_rate': current_cost_rate}
            if current_cost_rate >= target_cost_rate:
                return
                
            total_iterations = metrics.get('total_iterations', 8000)
            progress = iteration / total_iterations
            
            if progress < 0.05:
                new_cost_rate = 0.0
            elif progress < 0.20:
                ratio = (progress - 0.05) / 0.15
                new_cost_rate = target_cost_rate * ratio
            else:
                new_cost_rate = target_cost_rate
                
            if abs(new_cost_rate - current_cost_rate) > 1e-7:
                logger.info(f"Curriculum Update: {current_cost_rate:.6f} -> {new_cost_rate:.6f}")
                current_cost_rate = new_cost_rate
                vec_env.env_method('set_transaction_cost_rate', current_cost_rate)
                
            trainer.extra_checkpoint_state = {'current_cost_rate': current_cost_rate}

        # 훈련 관련 메트릭스 출력
        episodes_per_iteration = config.episodes_per_group * config.num_groups
        TIMESTEP_TO_ITERATION_RATIO = 25
        total_episodes = (config.total_timesteps // TIMESTEP_TO_ITERATION_RATIO) * episodes_per_iteration
        
        logger.info("=" * 80)
        logger.info(f"Total Timesteps: {config.total_timesteps:,}")
        logger.info(f"Total Episodes: {total_episodes:,}")
        logger.info(f"Output Directory: {config.output_dir}")
        logger.info("=" * 80)
        
        checkpoint_path = os.path.join(config.output_dir, 'checkpoints', 'checkpoint_iter{}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        config.save_to_file(os.path.join(config.output_dir, 'training_config.json'))
        
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
        logger.info("=" * 80)
        logger.info(f"[COMPLETE] Training Finished in {training_time/60:.2f}m")
        
        final_model_path = os.path.join(config.output_dir, f'{config.env}_{config.policy}_model.pt')
        trainer.save_checkpoint(
            final_model_path, final_metrics['num_updates'],
            extra_state=trainer.extra_checkpoint_state
        )
        logger.info(f"Saved final model: {final_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    main()
