#!/usr/bin/env python3
"""
xLSTM 기반 GRPO 훈련 스크립트 (train_xlstm.py)

재활용 가능 데이터 추출기(data_extractor.py)와 scalping_env_xlstm.py를 사용하여
5단계 분할 매매 및 1~5초 급등 패턴/손절 훈련을 수행합니다.
"""

import os
import sys
import json
import logging
import math
import argparse
import random
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.evaluation import (
    chronological_date_split, compatible_resume_best, evaluate_policy,
    evaluation_signature, normalize_date, validate_checkpoint_dates,
)
from lib.observations import ObservationBuilder, validate_max_stages
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
        self.extracted_dir = 'data/extracted_episodes_v2'
        self.table_name = 'datasets'
        self.seq_len = 1024
        self.features = 27
        self.episode_steps = 300
        self.account_observations = True
        self.execution_observations = True
        self.execution_action_mask = True
        self.liquidation_max_steps = 300
        self.decision_interval_seconds = 0.0
        self.episode_duration_seconds = 0.0
        
        self.hidden_dim = 512
        self.cnn_channels = 256
        self.rnn_hidden_dim = 512
        self.action_dim = 3
        
        self.episodes_per_group = 4
        self.num_groups = 4
        self.lr = 3e-5
        self.gamma = 1.0  # Finite-episode net NAV: do not discount delayed profits.
        self.lambda_gae = 0.95
        self.group_advantage_coef = 1.0
        self.training_seed = 42
        self.clip = 0.1
        self.kl_target = 0.01
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 16
        
        self.use_raw_data = True
        self.rolling_window_size = 512
        self.rolling_min_samples = 64
        
        self.total_timesteps = 1_000_000
        self.checkpoint_interval = 10
        self.output_dir = 'models/grpo_xlstm'
        self.load_policy = None
        self.resume = False
        self.revert_patience = 0
        self.num_workers = 4
        self.checkpoint_segments = 16  # Gradient checkpointing: 시퀀스 분할 수 (메모리 절약)
        self.num_epochs = 2            # Number of epochs per policy update
        self.use_gae = True
        self.train_end_date = None
        self.validation_end_date = None
        self.validation_fraction = 0.2
        self.test_fraction = 0.2
        self.embargo_dates = 0
        self.evaluation_episodes = 64
        self.evaluation_interval = 5
        self.evaluation_workers = 8
        self.evaluation_seed = 42
        self.diagnostics_interval = 5
        self.diagnostics_max_samples = 256
        self.selection_require_liquidation = True
        self.no_trade_patience = 0
        self.profitable_min_round_trips = 20
        self.profitable_min_traded_dates = 3
        self.cache_max_bytes = 256 * 1024 * 1024
        self.initial_cash = 1_000_000.0
        self.max_stages = 1
        self.max_holding_seconds = 300.0
        self.stop_loss_pct = 2.0
        self.execution_config = {
            'order_latency_ms': 100, 'cancel_latency_ms': 50,
            'order_ttl_seconds': 2, 'spread_bps': 10, 'slippage_bps': 2,
            'fallback_depth': 100, 'require_order_book': True,
        }
        
        self.base_price = 100000.0
        self.price_scale = 1.0  # Direct DB price units; NPZ manifest declares its own units.
        self.no_trade_penalty = 0.0
        self.transaction_cost_rate = 0.00015  # 기본 거래 수수료율 (0.015%)
        self.buy_tax_rate = 0.0                # 매수 세금 (0%)
        self.sell_tax_rate = 0.0018            # 매도 세금 (0.18%)
        self.max_trades_per_episode = None
        self.step_reward_scale = 1.0  # Dense step reward 스케일 비율
        self.win_bonus = 0.0          # Legacy compatibility; NAV rewards have no bonuses.
        self.loss_penalty = 0.0
        self.buy_signal_bonus = 0.0
        
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
        validate_max_stages(self.max_stages)
        if not isinstance(self.selection_require_liquidation, bool):
            errors.append("selection_require_liquidation must be a boolean")
        if not isinstance(self.execution_action_mask, bool):
            errors.append('execution_action_mask must be a boolean')
        for name, minimum in (('no_trade_patience', 0), ('profitable_min_round_trips', 1),
                              ('profitable_min_traded_dates', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                errors.append(f'{name} must be an integer >= {minimum}')
        if not 0 < self.gamma <= 1:
            errors.append("gamma must be in (0, 1]")
        if not 0 <= self.lambda_gae <= 1:
            errors.append('lambda_gae must be in [0, 1]')
        if not isinstance(self.account_observations, bool):
            errors.append('account_observations must be a boolean')
        if not isinstance(self.execution_observations, bool):
            errors.append('execution_observations must be a boolean')
        if self.execution_observations and not self.account_observations:
            errors.append('execution_observations requires account_observations')
        for name in ('decision_interval_seconds', 'episode_duration_seconds', 'group_advantage_coef'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                errors.append(f'{name} must be finite and nonnegative')
        if not self.use_gae and self.group_advantage_coef == 0:
            errors.append('group_advantage_coef=0 requires use_gae')
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int) or not 0 <= self.training_seed < 2**32:
            errors.append('training_seed must be an integer in [0, 2**32)')
        if (isinstance(self.liquidation_max_steps, bool) or
                not isinstance(self.liquidation_max_steps, int) or self.liquidation_max_steps < 0):
            errors.append('liquidation_max_steps must be a nonnegative integer')
        if not isinstance(self.resume, bool):
            errors.append("resume must be a boolean")
        if self.resume and not self.load_policy:
            errors.append("--resume requires --load_policy")
        for name in ('seq_len', 'episode_steps', 'num_workers', 'episodes_per_group',
                     'num_groups', 'batch_size', 'num_epochs', 'total_timesteps',
                     'evaluation_episodes', 'evaluation_interval', 'evaluation_workers',
                     'diagnostics_interval', 'diagnostics_max_samples'):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        # cache 디렉토리나 DB 파일 둘 중 하나는 존재해야 함
        if not self.extracted_dir and not self.db_path:
            errors.append("Either db_path or extracted_dir is required")
        elif self.extracted_dir and not os.path.exists(self.extracted_dir):
            logger.warning(f"extracted_dir ({self.extracted_dir}) not found. Falling back to DB checking.")
            self.extracted_dir = None
            if not self.db_path:
                errors.append("extracted_dir not found and no db_path specified")
            elif not os.path.exists(self.db_path):
                errors.append(f"Neither extracted_dir nor db_path exists: {self.db_path}")
                
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info("[OK] Configuration validated")


def create_environment(config: TrainingConfig, device: str, allowed_dates=None, seed=None):
    """환경 인스턴스 생성"""
    try:
        environment = GRPOScalpingEnvXLSTM(
            db_path=config.db_path,
            table_name=config.table_name,
            seq_len=config.seq_len,
            expected_features=config.features,
            transaction_cost_rate=config.transaction_cost_rate,
            buy_tax_rate=config.buy_tax_rate,
            sell_tax_rate=config.sell_tax_rate,
            max_episode_steps=config.episode_steps,
            account_observations=config.account_observations,
            execution_observations=config.execution_observations,
            execution_action_mask=config.execution_action_mask,
            liquidation_max_steps=config.liquidation_max_steps,
            decision_interval_seconds=config.decision_interval_seconds,
            episode_duration_seconds=config.episode_duration_seconds,
            base_price=config.base_price,
            price_scale=config.price_scale,
            no_trade_penalty=config.no_trade_penalty,
            max_trades_per_episode=config.max_trades_per_episode,
            step_reward_scale=config.step_reward_scale,
            win_bonus=config.win_bonus,
            loss_penalty=config.loss_penalty,
            buy_signal_bonus=config.buy_signal_bonus,
            extracted_dir=config.extracted_dir,
            allowed_dates=allowed_dates,
            cache_max_bytes=config.cache_max_bytes,
            use_raw_data=config.use_raw_data,
            rolling_window_size=config.rolling_window_size,
            rolling_min_samples=config.rolling_min_samples,
            initial_cash=config.initial_cash,
            max_stages=config.max_stages,
            max_holding_seconds=config.max_holding_seconds,
            stop_loss_pct=config.stop_loss_pct,
            execution_config=config.execution_config,
        )
        if seed is not None:
            environment.np_random = np.random.default_rng(seed)
        return environment
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
            action_dim=config.action_dim,
            execution_action_mask=config.execution_action_mask,
            max_stages=config.max_stages,
            checkpoint_segments=config.checkpoint_segments
        )
        policy.to(device)
        return policy
    except Exception as e:
        logger.error(f"Failed to create policy: {e}", exc_info=True)
        raise


def prepare_date_splits(config):
    if config.extracted_dir:
        with open(Path(config.extracted_dir) / 'manifest.json', encoding='utf-8') as handle:
            dates = [episode['date'] for episode in json.load(handle)['episodes']]
    else:
        import duckdb
        table = '"' + config.table_name.replace('"', '""') + '"'
        with duckdb.connect(config.db_path, read_only=True) as connection:
            dates = [row[0] for row in connection.execute(f'SELECT DISTINCT "날짜" FROM {table}').fetchall()]
    return chronological_date_split(
        dates, config.train_end_date, config.validation_end_date,
        config.validation_fraction, config.test_fraction, config.embargo_dates)


def load_resume_best_candidates(source, source_path, output_dir, signature):
    """Only inspect named best checkpoints beside the source or in the output."""
    source_path = Path(source_path).resolve()
    directories = (source_path.parent, source_path.parent / 'checkpoints',
                   Path(output_dir).resolve() / 'checkpoints')
    paths = [directory / name for directory in directories
             for name in ('checkpoint_best.pt', 'checkpoint_best_profitable.pt')]
    candidates, seen = [], {source_path}
    for path in paths:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            candidate = torch.load(path, map_location='cpu', weights_only=True)
            if compatible_resume_best(candidate, source, signature):
                candidates.append(candidate)
                logger.info("Will re-evaluate compatible resume best: %s", path)
            else:
                logger.warning("Ignoring unrelated/incompatible resume best: %s", path)
        except Exception as exc:
            logger.warning("Cannot inherit resume best %s: %s", path, exc)
    return candidates


def restore_resume_progress(trainer, config, checkpoint, control_overrides, signature):
    """Restore counters, then explicitly override controls without mixing their history."""
    iteration = trainer.restore_training_progress(checkpoint)
    changed_controls = False
    for name in ('no_trade_patience', 'profitable_min_round_trips', 'profitable_min_traded_dates'):
        if name in control_overrides:
            value = control_overrides[name]
            changed_controls |= value != getattr(trainer, name)
            setattr(trainer, name, value)
        setattr(config, name, getattr(trainer, name))
    if changed_controls or checkpoint.get('extra_state', {}).get('evaluation_signature') != signature:
        trainer.training_control_state = type(trainer.training_control_state)()
        logger.info('Reset no-trade validation history because evaluation conditions or controls changed')
    config.lr = trainer.learning_rate
    config.group_advantage_coef = trainer.group_advantage_coef
    return iteration


def evaluate_final_test(policy, config, date_splits, device, final_metrics):
    """Keep the test partition sealed when a training experiment stops early."""
    reason = final_metrics.get('early_stop_reason')
    if reason:
        logger.info('Test evaluation skipped after early stop: %s', reason)
        return None, str(reason)
    environment = create_environment(config, device, date_splits['test'])
    try:
        return evaluate_policy(policy, environment, config.evaluation_episodes,
                               config.evaluation_seed, device), None
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description='xLSTM 기반 GRPO E2E 훈련')
    parser.add_argument('--config', type=str, default=None, help='JSON 설정 파일')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
    
    # 데이터
    parser.add_argument('--db', dest='db_path', type=str, default=None, help='Database path')
    parser.add_argument('--table', dest='table_name', type=str, default=None, help='Table name')
    parser.add_argument('--extracted_dir', type=str, default=None, help='추출 데이터 디렉토리 경로')
    parser.add_argument('--train_end_date', default=None)
    parser.add_argument('--validation_end_date', default=None)
    parser.add_argument('--validation_fraction', type=float, default=None)
    parser.add_argument('--test_fraction', type=float, default=None)
    parser.add_argument('--embargo_dates', type=int, default=None)
    parser.add_argument('--evaluation_episodes', type=int, default=None)
    parser.add_argument('--evaluation_interval', type=int, default=None)
    parser.add_argument('--evaluation_seed', type=int, default=None)
    parser.add_argument('--selection_require_liquidation', action=argparse.BooleanOptionalAction, default=None,
                        help='Only select validation checkpoints with no residual inventory (default: enabled)')
    parser.add_argument('--execution_action_mask', action=argparse.BooleanOptionalAction, default=None,
                        help='Use executable order-state masks in rollout, update and evaluation')
    parser.add_argument('--no_trade_patience', type=int, default=None,
                        help='Stop after this many stagnant no-trade validations; 0 disables')
    parser.add_argument('--profitable_min_round_trips', type=int, default=None)
    parser.add_argument('--profitable_min_traded_dates', type=int, default=None)
    parser.add_argument('--max_stages', '--max-stages', type=int, choices=range(1, 6), default=None,
                        help='Maximum position entries (1-5; default: 1, no split entries)')
    parser.add_argument('--max_holding_seconds', type=float, default=None)
    parser.add_argument('--stop_loss_pct', type=float, default=None)
    parser.add_argument('--initial_cash', type=float, default=None)
    
    # 기타 인자들
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--features', type=int, default=None)
    parser.add_argument('--episode_steps', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel environment workers')
    
    # 보상 및 비용 설정
    parser.add_argument('--no_trade_penalty', type=float, default=None,
                        help='Penalty for making 0 trades in an episode (default: 0.0)')
    parser.add_argument('--transaction_cost', dest='transaction_cost_rate', type=float, default=None,
                        help='Target transaction cost rate (default: 0.00015)')
    parser.add_argument('--buy_tax', dest='buy_tax_rate', type=float, default=None,
                        help='Buy tax rate (default: 0.0)')
    parser.add_argument('--sell_tax', dest='sell_tax_rate', type=float, default=None,
                        help='Sell tax rate (default: 0.0018)')
    parser.add_argument('--max_trades_per_episode', type=int, default=None,
                        help='Max trades allowed per episode (default: None/unlimited)')
    parser.add_argument('--step_reward_scale', type=float, default=None,
                        help='Scale for dense step reward (default: 1.0)')
    parser.add_argument('--win_bonus', type=float, default=None,
                        help='Bonus for winning trades (default: 5.0)')
    parser.add_argument('--loss_penalty', type=float, default=None,
                        help='Penalty for losing trades (default: 0.3)')
    parser.add_argument('--buy_signal_bonus', type=float, default=None,
                        help='Bonus for entering on positive buy signals (rising price + trade value increase, default: 0.5)')
    
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
    parser.add_argument('--checkpoint_segments', type=int, default=None,
                        help='Gradient checkpointing: number of segments to split the RNN sequence into. '
                             'Higher = less VRAM but slower (default: 16)')
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
    parser.add_argument('--lambda_gae', type=float, default=None)
    parser.add_argument('--group_advantage_coef', type=float, default=None)
    parser.add_argument('--training_seed', type=int, default=None)
    parser.add_argument('--account_observations', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--execution_observations', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--liquidation_max_steps', type=int, default=None)
    parser.add_argument('--decision_interval_seconds', type=float, default=None)
    parser.add_argument('--episode_duration_seconds', type=float, default=None)
    parser.add_argument('--evaluation_workers', type=int, default=None)
    parser.add_argument('--diagnostics_interval', type=int, default=None)
    parser.add_argument('--diagnostics_max_samples', type=int, default=None)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--kl_target', type=float, default=None)
    parser.add_argument('--entropy_coef', type=float, default=None)
    parser.add_argument('--value_coef', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--total_timesteps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None, help='Mini-batch size for PPO updates (default: 64)')
    parser.add_argument('--checkpoint_interval', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--load_policy', type=str, default=None,
                        help='Load compatible weights for a new fine-tuning run')
    parser.add_argument('--resume', action='store_true', default=None,
                        help='Also restore optimizer and counters; total_timesteps is the cumulative target')
    parser.add_argument('--revert_patience', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs per policy update')
    parser.add_argument('--use_gae', type=lambda x: x.lower() in ('true', '1', 'yes'), default=None,
                        help='Whether to use GAE (default: False for GRPO mode)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("[START] xLSTM-GRPO Training System")
    logger.info("=" * 80)
    
    device = "cpu"
    vec_env = ref_env = validation_env = test_env = trainer = None
    validation_envs = []
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
                
        # 체크포인트 파일로부터 모델 아키텍처 자동 감지 및 설정 조정
        if config.load_policy and os.path.exists(config.load_policy):
            logger.info(f"[CHECKPOINT DETECT] Loading checkpoint {config.load_policy} to verify/auto-adjust model dimensions...")
            try:
                # GPU OOM을 방지하기 위해 CPU에서 임시로 로드하여 검사
                checkpoint = torch.load(config.load_policy, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
                saved_schema = checkpoint.get('observation_schema', {})
                saved_config = {**checkpoint.get('config', {}),
                                **checkpoint.get('extra_state', {}).get('training_config', {})}
                if args.account_observations is None:
                    config.account_observations = saved_schema.get('version') in (3, 4)
                if args.execution_observations is None:
                    config.execution_observations = saved_schema.get('version') == 4
                if args.liquidation_max_steps is None:
                    config.liquidation_max_steps = saved_config.get('liquidation_max_steps', 0)
                if args.lambda_gae is None:
                    config.lambda_gae = saved_config.get('lambda_gae', 0.95)
                for name, default in (('decision_interval_seconds', 0.0),
                                      ('episode_duration_seconds', 0.0),
                                      ('execution_action_mask', False), ('no_trade_patience', 0),
                                      ('profitable_min_round_trips', 20), ('profitable_min_traded_dates', 3),
                                      ('group_advantage_coef', 1.0), ('training_seed', 42)):
                    if getattr(args, name) is None:
                        setattr(config, name, saved_config.get(name, default))
                
                has_gru = any('gru' in k for k in state_dict.keys())
                
                # CNN 채널 감지 및 자동 조정
                if 'conv1.weight' in state_dict:
                    chk_cnn = state_dict['conv1.weight'].shape[0]
                    if config.cnn_channels != chk_cnn:
                        logger.warning(f"[AUTO-CONFIG] Overriding cnn_channels from {config.cnn_channels} to {chk_cnn} to match checkpoint.")
                        config.cnn_channels = chk_cnn
                
                if not has_gru:
                    # xLSTM Hidden Dimension 감지 및 자동 조정
                    if 'xlstm.cells.0.w_q.weight' in state_dict:
                        chk_rnn = state_dict['xlstm.cells.0.w_q.weight'].shape[0]
                        if config.rnn_hidden_dim != chk_rnn:
                            logger.warning(f"[AUTO-CONFIG] Overriding rnn_hidden_dim from {config.rnn_hidden_dim} to {chk_rnn} to match checkpoint.")
                            config.rnn_hidden_dim = chk_rnn
                    # FC Hidden Dimension 감지 및 자동 조정
                    if 'fc1.weight' in state_dict:
                        chk_fc = state_dict['fc1.weight'].shape[0]
                        if config.hidden_dim != chk_fc:
                            logger.warning(f"[AUTO-CONFIG] Overriding hidden_dim (FC) from {config.hidden_dim} to {chk_fc} to match checkpoint.")
                            config.hidden_dim = chk_fc
                else:
                    logger.info("[CHECKPOINT DETECT] GRU checkpoint detected. Only CNN shapes will be auto-matched. RNN/FC will use config settings.")
            except Exception as e:
                logger.warning(f"Failed to auto-detect/adjust config from load_policy: {e}. Proceeding with manual config.")

        config.validate()
        date_splits = prepare_date_splits(config)
        os.makedirs(config.output_dir, exist_ok=True)
        with open(Path(config.output_dir) / 'date_splits.json', 'w', encoding='utf-8') as handle:
            json.dump(date_splits, handle, ensure_ascii=False, indent=2)
        device = config.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            config.device = device
            
        logger.info(f"[OK] Using device: {device}")
        random.seed(config.training_seed)
        np.random.seed(config.training_seed)
        torch.manual_seed(config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training_seed)
        logger.info('Training settings: execution_observations=%s, decision_interval_seconds=%s, '
                    'episode_duration_seconds=%s, group_advantage_coef=%s, lambda_gae=%s, training_seed=%s',
                    config.execution_observations, config.decision_interval_seconds,
                    config.episode_duration_seconds, config.group_advantage_coef,
                    config.lambda_gae, config.training_seed)
        logger.info('Execution action mask=%s; no-trade patience=%s; profitable candidate minimum=%s round trips / %s traded dates',
                    config.execution_action_mask, config.no_trade_patience,
                    config.profitable_min_round_trips, config.profitable_min_traded_dates)
        
        # 환경 생성 (extracted_dir 캐시 활용 시 Windows의 SubprocVecEnv도 안정 작동)
        logger.info(f"[STEP 2/6] Creating environments (Workers: {config.num_workers})...")
        from ai_trader.grpo.environments import SubprocVecEnv, DummyVecEnv
        import functools
        
        # 만약 db_path를 직접 쓰면서 Windows인 경우에는 multi-process 크래시 방지
        if config.extracted_dir is None and config.num_workers > 1 and sys.platform == 'win32':
             logger.warning("No extracted_dir provided, fallback to single process to prevent IPC crash on Windows.")
             config.num_workers = 1
             
        env_fns = [functools.partial(create_environment, config, device, date_splits['train'],
                                     config.training_seed + worker)
                   for worker in range(config.num_workers)]
        
        if config.num_workers > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
            
        ref_env = (vec_env.envs[0] if config.num_workers <= 1
                   else create_environment(config, device, date_splits['train']))
        for _ in range(min(config.evaluation_workers, config.evaluation_episodes)):
            validation_envs.append(create_environment(config, device, date_splits['validation']))
        
        # 정책 모델 생성
        logger.info("[STEP 3/6] Creating policy...")
        policy = create_policy(config, ref_env, device)
        
        # 모델 파라미터 수 로깅
        total_params = sum(p.numel() for p in policy.parameters())
        logger.info(f"  Total parameters: {total_params:,}")
        
        start_iteration = 0
        checkpoint = None
        if config.load_policy:
            logger.info(f"[LOAD POLICY] Loading from {config.load_policy}...")
            try:
                checkpoint = torch.load(config.load_policy, map_location=device, weights_only=False)
                ObservationBuilder.from_schema(ref_env.observation_schema).validate_schema(
                    checkpoint.get('observation_schema'))
                validate_checkpoint_dates(checkpoint, date_splits)
                state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
                
                # GRU 체크포인트 감지 시 CNN 가중치 전이(Transfer Learning) 적용
                has_gru = any('gru' in k for k in state_dict.keys())
                if has_gru:
                    if config.resume:
                        raise ValueError("Cannot resume a GRU optimizer into an xLSTM policy")
                    logger.info("⚠️ GRU weights detected in checkpoint. Transferring CNN layers only. xLSTM/FC will be initialized randomly.")
                    cnn_state_dict = {k: v for k, v in state_dict.items() if k.startswith(('conv', 'bn'))}
                    missing, unexpected = policy.load_state_dict(cnn_state_dict, strict=False)
                    logger.info(f"  Transfer completed. CNN loaded successfully.")
                else:
                    policy.load_state_dict(state_dict, strict=True)
                    logger.info("  Checkpoint loaded successfully.")
                    
                if not config.resume:
                    logger.info("Loaded weights for a new fine-tuning run (fresh optimizer and counters)")
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
            lambda_gae=config.lambda_gae,
            group_advantage_coef=config.group_advantage_coef,
            clip_epsilon=config.clip,
            kl_target=config.kl_target,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            batch_size=config.batch_size,
            device=device,
            tensorboard_log_dir=tensorboard_dir,
            use_gae=config.use_gae,
            num_epochs=config.num_epochs,
            observation_schema=ref_env.observation_schema,
            evaluation_callback=lambda current_policy: evaluate_policy(
                current_policy, validation_envs, config.evaluation_episodes,
                config.evaluation_seed, device),
            evaluation_interval=config.evaluation_interval,
            selection_require_liquidation=config.selection_require_liquidation,
            no_trade_patience=config.no_trade_patience,
            profitable_min_round_trips=config.profitable_min_round_trips,
            profitable_min_traded_dates=config.profitable_min_traded_dates,
            diagnostics_interval=config.diagnostics_interval,
            diagnostics_max_samples=config.diagnostics_max_samples,
        )
        trainer.extra_checkpoint_state = {'date_splits': date_splits,
                                          'training_config': dict(config.__dict__)}
        if checkpoint is not None:
            # Keep all ancestral exposure dates when writing a fine-tuned model.
            lineage = checkpoint['extra_state']['date_splits']
            trainer.extra_checkpoint_state['date_splits'] = {
                **date_splits,
                'train': sorted(set(date_splits['train']) | {normalize_date(day) for day in lineage.get('train', [])}),
                'validation': sorted(set(date_splits['validation']) | {normalize_date(day) for day in lineage.get('validation', [])}),
            }
        signature = evaluation_signature(dict(config.__dict__), date_splits, ref_env.observation_schema)
        if config.resume:
            control_overrides = {name: getattr(args, name) for name in
                                 ('no_trade_patience', 'profitable_min_round_trips', 'profitable_min_traded_dates')
                                 if getattr(args, name) is not None}
            start_iteration = restore_resume_progress(trainer, config, checkpoint, control_overrides, signature)
            if config.total_timesteps <= trainer.total_timesteps:
                raise ValueError("For --resume, total_timesteps must exceed the saved cumulative timesteps")
            trainer.extra_checkpoint_state['training_config'] = dict(config.__dict__)
            logger.info("Resuming iteration=%s, timesteps=%s, updates=%s with saved optimizer",
                        start_iteration, trainer.total_timesteps, trainer.num_updates)
        trainer.extra_checkpoint_state['evaluation_signature'] = signature
        resume_best_checkpoints = (load_resume_best_candidates(
            checkpoint, config.load_policy, config.output_dir, signature) if config.resume else [])
        
        # 7. 고정 거래 비용 적용 (커리큘럼 미사용)
        current_cost_rate = config.transaction_cost_rate
        vec_env.env_method('set_transaction_cost_rate', current_cost_rate)
        logger.info(f"[OK] Fixed transaction cost rate applied: {current_cost_rate:.6f}")

        # 훈련 관련 메트릭스 출력
        import math
        remaining_timesteps = config.total_timesteps - trainer.total_timesteps
        total_episodes = max(1, math.ceil(remaining_timesteps / config.episode_steps))
        
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
            on_iteration_end=None,
            start_iteration=start_iteration,
            max_timesteps=config.total_timesteps,
            revert_to_best_patience=config.revert_patience,
            resume=config.resume,
            resume_best_checkpoints=resume_best_checkpoints,
            preserve_initial_policy=(checkpoint is not None and not config.resume),
        )
        
        training_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"[COMPLETE] Training Finished in {training_time/60:.2f}m")
        
        # Test is opened only after validation has selected the final weights.
        selected_path = Path(config.output_dir) / 'checkpoints' / 'checkpoint_best.pt'
        if not selected_path.exists():
            raise RuntimeError("Training produced no validated checkpoint")
        selected = torch.load(selected_path, map_location=device, weights_only=False)
        policy.load_state_dict(selected['policy_state_dict'], strict=True)
        test_metrics, test_skip_reason = evaluate_final_test(
            policy, config, date_splits, device, final_metrics)
        report = {'date_splits': date_splits, 'training': final_metrics,
                  'validation': selected['extra_state']['validation_metrics'],
                  'test': test_metrics, 'test_skip_reason': test_skip_reason,
                  'test_used_for_model_selection': False}
        with open(Path(config.output_dir) / 'evaluation_report.json', 'w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        final_model_path = os.path.join(config.output_dir, f'{config.env}_{config.policy}_model.pt')
        import shutil
        shutil.copyfile(selected_path, final_model_path)
        logger.info(f"Saved final model: {final_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False
    finally:
        if trainer is not None and trainer.writer is not None:
            trainer.writer.close()
        if vec_env is not None:
            vec_env.close()
        for environment in (*validation_envs, test_env):
            if environment is not None:
                environment.close()
        if ref_env is not None and not (vec_env is not None and hasattr(vec_env, 'envs')
                                        and ref_env in vec_env.envs):
            ref_env.close()

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
