#!/usr/bin/env python3
"""
스캘핑 전략 백테스팅

GRPOScalpingEnv와 AutoEncoder 임베딩을 사용한 스캘핑 전략 검증

사용법:
    python scripts/live/backtest_scalping.py --db datasets.duckdb --episodes 100
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.environments.scalping_env import GRPOScalpingEnv
from ai_trader.grpo.policies.scalping_policy import GRPOPolicy

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalpingBacktest:
    """스캘핑 전략 백테스팅"""
    
    def __init__(
        self,
        policy_path: str,
        embedding_model_path: str,
        db_path: str,
        device: str = 'cuda',
        transaction_cost_rate: float = 0.00215,
        quick_exit_threshold: float = 1.5,
        quick_exit_penalty: float = 0.01,
        max_holding_time: float = 10.0,
        holding_penalty_rate: float = 0.001,
        max_episode_steps: int = None,
        quick_exit_mode: str = 'penalty_only'
    ):
        """
        Args:
            policy_path: 정책 모델 경로
            embedding_model_path: 임베딩 모델 경로 (AutoEncoder)
            db_path: 데이터베이스 경로
            device: 디바이스
            transaction_cost_rate: 거래 비용 비율 (기본값: 0.00215 = 0.215%)
            quick_exit_threshold: 빠른 손절 시간 임계값 (초)
            quick_exit_penalty: 빠른 손절 룰 위반 페널티
            max_holding_time: 최대 보유 시간 (초)
            holding_penalty_rate: 장기 보유 페널티 비율
            max_episode_steps: 에피소드당 최대 스텝 수
            quick_exit_mode: 빠른 손절 모드 ('penalty_only' 또는 'force_close')
        """
        self.device = device
        
        # AutoEncoder 임베딩 모델 로드
        logger.info(f"Loading embedding model: {embedding_model_path}")
        from ai_trader.embedding.autoencoder_model import AutoEncoderEmbedding
        
        self.embedding_model = AutoEncoderEmbedding(
            input_dim=28,  # 기본 특징 수
            embedding_dim=128,
            hidden_dim=256,
            seq_len=60
        )
        
        checkpoint = torch.load(embedding_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.embedding_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.embedding_model.load_state_dict(checkpoint)
        
        self.embedding_model.to(device)
        self.embedding_model.eval()
        
        # 환경 생성
        logger.info("Creating GRPOScalpingEnv...")
        self.env = GRPOScalpingEnv(
            embedding_model=self.embedding_model,
            db_path=db_path,
            table_name='datasets',
            seq_len=60,
            embedding_dim=128,
            expected_features=28,
            transaction_cost_rate=transaction_cost_rate,
            quick_exit_threshold=quick_exit_threshold,
            quick_exit_penalty=quick_exit_penalty,
            max_holding_time=max_holding_time,
            holding_penalty_rate=holding_penalty_rate,
            max_episode_steps=max_episode_steps,
            quick_exit_mode=quick_exit_mode,
            device=device
        )
        
        # 정책 모델 로드
        logger.info(f"Loading policy model: {policy_path}")
        
        # 정책 모델 생성 및 로드
        self.policy = GRPOPolicy(
            embedding_dim=128,  # 임베딩 차원
            hidden_dim=128,  # 체크포인트와 일치해야 함
            action_dim=3  # HOLD, BUY, SELL
        )
        
        checkpoint = torch.load(policy_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
        elif 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint)
        
        self.policy.to(device)
        self.policy.eval()
        
        logger.info("Backtest initialized")
        logger.info(f"  Transaction cost: {transaction_cost_rate*100:.3f}%")
        logger.info(f"  Quick exit threshold: {quick_exit_threshold}s")
        logger.info(f"  Quick exit mode: {quick_exit_mode}")
        logger.info(f"  Max holding time: {max_holding_time}s")
    
    def run_episode(self) -> Dict:
        """
        단일 에피소드 실행
        
        Returns:
            에피소드 결과
        """
        observation, info = self.env.reset()
        episode_reward = 0.0
        
        actions_taken = []
        
        while True:
            # 임베딩을 입력으로 예측 (GRPOScalpingEnv는 임베딩을 반환)
            # observation은 이미 임베딩 벡터 (embedding_dim,)
            
            # 예측 수행
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                
                # 정책 모델 실행
                action_logits, _ = self.policy(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                
                # 최대 확률 행동 선택 (deterministic)
                action = torch.argmax(action_probs, dim=-1).item()
                confidence = action_probs[0, action].item()
            
            actions_taken.append(action)
            
            # 환경 스텝
            next_observation, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            observation = next_observation
            
            if done:
                result = {
                    'reward': episode_reward,
                    'actions': actions_taken,
                    'episode_info': step_info.get('episode', {})
                }
                return result
    
    def run(self, num_episodes: int = 100) -> Dict:
        """
        백테스팅 실행
        
        Args:
            num_episodes: 에피소드 수
            
        Returns:
            백테스팅 결과
        """
        logger.info("=" * 80)
        logger.info(f"🔍 Running Scalping Backtest ({num_episodes} episodes)")
        logger.info("=" * 80)
        
        all_rewards = []
        all_returns = []
        all_trades = []
        all_win_rates = []
        all_sharpe_ratios = []
        all_quick_exit_violations = []
        
        for episode_idx in range(num_episodes):
            result = self.run_episode()
            
            all_rewards.append(result['reward'])
            
            ep_info = result['episode_info']
            if 'total_return' in ep_info:
                all_returns.append(ep_info['total_return'])
            if 'win_rate' in ep_info:
                all_win_rates.append(ep_info['win_rate'])
            if 'sharpe_ratio' in ep_info:
                all_sharpe_ratios.append(ep_info['sharpe_ratio'])
            if 'quick_exit_violations' in ep_info:
                all_quick_exit_violations.append(ep_info['quick_exit_violations'])
            if 'trades' in ep_info:
                all_trades.extend(ep_info['trades'])
            
            if (episode_idx + 1) % 10 == 0:
                logger.info(f"Progress: {episode_idx + 1}/{num_episodes} episodes")
        
        # 결과 분석
        results = self._analyze_results(
            all_rewards,
            all_returns,
            all_trades,
            all_win_rates,
            all_sharpe_ratios,
            all_quick_exit_violations
        )
        
        return results
    
    def _analyze_results(
        self,
        rewards: List[float],
        returns: List[float],
        trades: List[Dict],
        win_rates: List[float],
        sharpe_ratios: List[float],
        quick_exit_violations: List[int]
    ) -> Dict:
        """결과 분석"""
        
        # 거래 분석
        valid_trades = [t for t in trades if 'profit_rate' in t]
        
        winning_trades = [t for t in valid_trades if t.get('profit_rate', 0) > 0]
        losing_trades = [t for t in valid_trades if t.get('profit_rate', 0) <= 0]
        
        # 자동 청산 분석 (force_close 모드에서만 유효)
        forced_liquidations = [t for t in valid_trades if t.get('forced_liquidation', False)]
        
        results = {
            'episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_trades': len(valid_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(valid_trades) if valid_trades else 0,
            'forced_liquidations': len(forced_liquidations),
            'mean_return': np.mean(returns) if returns else 0,
            'total_return': np.sum(returns) if returns else 0,
            'mean_episode_win_rate': np.mean(win_rates) if win_rates else 0,
            'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'total_quick_exit_violations': sum(quick_exit_violations),
            'avg_quick_exit_violations_per_episode': np.mean(quick_exit_violations) if quick_exit_violations else 0
        }
        
        # 수익률 분석
        if valid_trades:
            profit_rates = [t['profit_rate'] * 100 for t in valid_trades]  # 퍼센트 변환
            results['mean_profit_rate'] = np.mean(profit_rates)
            results['median_profit_rate'] = np.median(profit_rates)
            results['std_profit_rate'] = np.std(profit_rates)
            results['max_profit_rate'] = np.max(profit_rates)
            results['min_profit_rate'] = np.min(profit_rates)
            
            if winning_trades:
                winning_profit_rates = [t['profit_rate'] * 100 for t in winning_trades]
                results['mean_winning_profit'] = np.mean(winning_profit_rates)
                results['max_winning_profit'] = np.max(winning_profit_rates)
            
            if losing_trades:
                losing_profit_rates = [t['profit_rate'] * 100 for t in losing_trades]
                results['mean_losing_profit'] = np.mean(losing_profit_rates)
                results['max_losing_profit'] = np.min(losing_profit_rates)
        
        # 보유 시간 분석
        if valid_trades:
            holding_times = [t.get('holding_time', 0) for t in valid_trades]
            results['mean_holding_time'] = np.mean(holding_times)
            results['median_holding_time'] = np.median(holding_times)
            results['max_holding_time'] = np.max(holding_times)
            results['min_holding_time'] = np.min(holding_times)
        
        # 에피소드당 거래 횟수
        if results['episodes'] > 0:
            results['trades_per_episode'] = results['total_trades'] / results['episodes']
        
        return results
    
    def print_results(self, results: Dict):
        """결과 출력"""
        logger.info("=" * 80)
        logger.info("📊 Scalping Backtest Results")
        logger.info("=" * 80)
        
        logger.info(f"\n📈 Performance:")
        logger.info(f"Episodes: {results['episodes']}")
        logger.info(f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        logger.info(f"Min/Max Reward: {results['min_reward']:.4f} / {results['max_reward']:.4f}")
        logger.info(f"Total Return: {results['total_return']:.4f}")
        logger.info(f"Mean Return per Episode: {results['mean_return']:.4f}")
        
        logger.info(f"\n💰 Trading:")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Trades per Episode: {results.get('trades_per_episode', 0):.2f}")
        logger.info(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.2%})")
        logger.info(f"Losing Trades: {results['losing_trades']}")
        logger.info(f"Forced Liquidations: {results['forced_liquidations']}")
        
        if 'mean_profit_rate' in results:
            logger.info(f"\n📊 Profit Analysis:")
            logger.info(f"Mean Profit Rate: {results['mean_profit_rate']:.2f}% ± {results.get('std_profit_rate', 0):.2f}%")
            logger.info(f"Median Profit Rate: {results['median_profit_rate']:.2f}%")
            logger.info(f"Min/Max Profit Rate: {results.get('min_profit_rate', 0):.2f}% / {results.get('max_profit_rate', 0):.2f}%")
            
            if 'mean_winning_profit' in results:
                logger.info(f"\n💚 Winning Trades:")
                logger.info(f"Mean Profit: {results['mean_winning_profit']:.2f}%")
                logger.info(f"Max Profit: {results.get('max_winning_profit', 0):.2f}%")
            
            if 'mean_losing_profit' in results:
                logger.info(f"\n💔 Losing Trades:")
                logger.info(f"Mean Loss: {results['mean_losing_profit']:.2f}%")
                logger.info(f"Max Loss: {results.get('max_losing_profit', 0):.2f}%")
        
        if 'mean_holding_time' in results:
            logger.info(f"\n⏱️ Holding Time Analysis:")
            logger.info(f"Mean: {results['mean_holding_time']:.2f}s")
            logger.info(f"Median: {results['median_holding_time']:.2f}s")
            logger.info(f"Min/Max: {results.get('min_holding_time', 0):.2f}s / {results.get('max_holding_time', 0):.2f}s")
        
        logger.info(f"\n📈 Risk Metrics:")
        logger.info(f"Mean Episode Win Rate: {results['mean_episode_win_rate']:.2%}")
        logger.info(f"Mean Sharpe Ratio: {results['mean_sharpe_ratio']:.4f}")
        
        logger.info(f"\n⚠️ Quick Exit Violations:")
        logger.info(f"Total: {results['total_quick_exit_violations']}")
        logger.info(f"Average per Episode: {results['avg_quick_exit_violations_per_episode']:.2f}")
        
        logger.info("=" * 80)
    
    def close(self):
        """리소스 정리"""
        self.env.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Scalping Strategy Backtest")
    parser.add_argument(
        '--policy',
        type=str,
        default='models/grpo_scalping@2025120/policy.pt',
        help='Policy model path'
    )
    parser.add_argument(
        '--embedding',
        type=str,
        default='models/autoencoder@20251013/model.pt',
        help='Embedding model path (AutoEncoder)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=r'C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb',
        help='Database path'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.00215,
        help='Transaction cost rate (default: 0.00215 = 0.215%%)'
    )
    parser.add_argument(
        '--quick-exit-threshold',
        type=float,
        default=1.5,
        help='Quick exit threshold in seconds (default: 1.5)'
    )
    parser.add_argument(
        '--quick-exit-penalty',
        type=float,
        default=0.01,
        help='Quick exit penalty (default: 0.01)'
    )
    parser.add_argument(
        '--max-holding-time',
        type=float,
        default=10.0,
        help='Maximum holding time in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--holding-penalty-rate',
        type=float,
        default=0.001,
        help='Holding penalty rate (default: 0.001)'
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=None,
        help='Maximum episode steps (default: None, no limit)'
    )
    parser.add_argument(
        '--quick-exit-mode',
        type=str,
        default='penalty_only',
        choices=['penalty_only', 'force_close'],
        help='Quick exit mode (default: penalty_only)'
    )
    
    args = parser.parse_args()
    
    # 경로 확인
    if not os.path.exists(args.policy):
        logger.error(f"❌ Policy model not found: {args.policy}")
        return False
    
    if not os.path.exists(args.embedding):
        logger.error(f"❌ Embedding model not found: {args.embedding}")
        return False
    
    if not os.path.exists(args.db):
        logger.error(f"❌ Database not found: {args.db}")
        return False
    
    try:
        # 백테스팅 실행
        backtest = ScalpingBacktest(
            policy_path=args.policy,
            embedding_model_path=args.embedding,
            db_path=args.db,
            device=args.device,
            transaction_cost_rate=args.transaction_cost,
            quick_exit_threshold=args.quick_exit_threshold,
            quick_exit_penalty=args.quick_exit_penalty,
            max_holding_time=args.max_holding_time,
            holding_penalty_rate=args.holding_penalty_rate,
            max_episode_steps=args.max_episode_steps,
            quick_exit_mode=args.quick_exit_mode
        )
        
        results = backtest.run(num_episodes=args.episodes)
        backtest.print_results(results)
        backtest.close()
        
        logger.info("\n✅ Backtest completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
