#!/usr/bin/env python3
"""
실전 거래 전략 백테스팅

EnhancedGRPOInference를 사용한 전략을 과거 데이터로 검증

중요: 
    - DirectFeatureEnv의 state[0]은 등락률(%)이므로,
    - 누적 수익률을 계산하여 Position의 가격으로 사용
    - cumulative_return *= (1 + return_rate)

사용법:
    python scripts/real/backtest_strategy.py --db datasets.duckdb --episodes 100
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.environments.direct_feature_env import DirectFeatureEnv, Action
from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Position

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyBacktest:
    """전략 백테스팅"""
    
    def __init__(
        self,
        model_path: str,
        db_path: str,
        device: str = 'cuda',
        min_buy_confidence: float = 0.0,
        stop_loss_rate: float = -2.0,
        take_profit_rate: float = 5.0,
        max_holding_period: int = 100
    ):
        """
        Args:
            model_path: 모델 경로
            db_path: 데이터베이스 경로
            device: 디바이스
            min_buy_confidence: Buy 최소 신뢰도
            stop_loss_rate: 손절 비율
            take_profit_rate: 익절 비율
            max_holding_period: 최대 보유 기간
        """
        self.device = device
        
        # 환경 생성
        logger.info("Creating environment...")
        self.env = DirectFeatureEnv(
            db_path=db_path,
            table_name='datasets_raw',
            seq_len=30,
            expected_features=24,
            transaction_cost_rate=0.00215,
            max_episode_steps=100,
            device=device
        )
        
        # 추론 엔진 초기화
        logger.info(f"Loading model: {model_path}")
        base_inference = GRPOInference(
            policy_path=model_path,
            embedding_model_path=None,
            device=device,
            use_torchscript=False
        )
        
        self.inference = EnhancedGRPOInference(
            base_inference=base_inference,
            min_buy_confidence=min_buy_confidence,
            min_sell_confidence=0.0,
            stop_loss_rate=stop_loss_rate,
            take_profit_rate=take_profit_rate,
            max_holding_period=max_holding_period,
            enable_auto_exit=True
        )
        
        logger.info("Backtest initialized")
    
    def run_episode(self) -> Dict:
        """
        단일 에피소드 실행
        
        Returns:
            에피소드 결과
        """
        state, info = self.env.reset()
        episode_reward = 0.0
        current_position = None
        
        trades = []
        actions_taken = []
        
        # 누적 가격 추적 (등락률로부터 계산)
        cumulative_return = 1.0  # 시작 = 100%
        
        while True:
            # 현재 등락률 (state[0])
            current_return_rate = state[0] if len(state) > 0 else 0.0
            
            # 누적 수익률 업데이트 (복리)
            cumulative_return *= (1 + current_return_rate)
            
            # 시퀀스 복원
            sequence = state.reshape(30, 24)
            
            # 예측
            action, confidence, pred_info = self.inference.predict(
                sequence=sequence,
                current_position=current_position,
                current_price=cumulative_return,  # 누적 수익률 사용
                deterministic=True
            )
            
            # 포지션 관리
            if action == Action.BUY.value and current_position is None:
                current_position = Position(
                    entry_price=cumulative_return,
                    entry_time=info.get('step', 0),
                    current_price=cumulative_return,
                    holding_period=0,
                    cumulative_return=0.0  # 누적 수익률 초기화
                )
                trades.append({
                    'action': 'BUY',
                    'price': cumulative_return,
                    'confidence': confidence,
                    'step': info.get('step', 0)
                })
            
            elif action == Action.SELL.value and current_position is not None:
                profit_rate = current_position.profit_rate
                trades.append({
                    'action': 'SELL',
                    'price': cumulative_return,
                    'confidence': confidence,
                    'profit_rate': profit_rate,
                    'holding_period': current_position.holding_period,
                    'reason': pred_info.get('exit_reason', 'Signal'),
                    'step': info.get('step', 0)
                })
                current_position = None
            
            actions_taken.append(action)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                result = {
                    'reward': episode_reward,
                    'trades': trades,
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
        logger.info(f"🔍 Running Backtest ({num_episodes} episodes)")
        logger.info("=" * 80)
        
        all_rewards = []
        all_returns = []
        all_trades = []
        all_win_rates = []
        
        for episode_idx in range(num_episodes):
            result = self.run_episode()
            
            all_rewards.append(result['reward'])
            all_trades.extend(result['trades'])
            
            ep_info = result['episode_info']
            if 'total_return' in ep_info:
                all_returns.append(ep_info['total_return'])
            if 'win_rate' in ep_info:
                all_win_rates.append(ep_info['win_rate'])
            
            if (episode_idx + 1) % 10 == 0:
                logger.info(f"Progress: {episode_idx + 1}/{num_episodes} episodes")
        
        # 결과 분석
        results = self._analyze_results(
            all_rewards,
            all_returns,
            all_trades,
            all_win_rates
        )
        
        return results
    
    def _analyze_results(
        self,
        rewards: List[float],
        returns: List[float],
        trades: List[Dict],
        win_rates: List[float]
    ) -> Dict:
        """결과 분석"""
        
        # 거래 분석
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.get('profit_rate', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('profit_rate', 0) <= 0]
        
        # 자동 청산 분석
        auto_exits = [t for t in sell_trades if t.get('reason') and t['reason'] != 'Signal']
        stop_losses = [t for t in auto_exits if 'Stop loss' in t.get('reason', '')]
        take_profits = [t for t in auto_exits if 'Take profit' in t.get('reason', '')]
        
        results = {
            'episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(sell_trades) if sell_trades else 0,
            'auto_exits': len(auto_exits),
            'stop_losses': len(stop_losses),
            'take_profits': len(take_profits),
            'mean_return': np.mean(returns) if returns else 0,
            'total_return': np.sum(returns) if returns else 0,
            'mean_win_rate': np.mean([wr for wr in win_rates if not np.isnan(wr)]) if win_rates else 0
        }
        
        # 수익률 분석
        if sell_trades:
            profit_rates = [t.get('profit_rate', 0) for t in sell_trades]
            results['mean_profit_rate'] = np.mean(profit_rates)
            results['median_profit_rate'] = np.median(profit_rates)
            results['std_profit_rate'] = np.std(profit_rates)
            results['max_profit_rate'] = np.max(profit_rates)
            results['min_profit_rate'] = np.min(profit_rates)
            
            if winning_trades:
                winning_profit_rates = [t['profit_rate'] for t in winning_trades]
                results['mean_winning_profit'] = np.mean(winning_profit_rates)
                results['max_winning_profit'] = np.max(winning_profit_rates)
            
            if losing_trades:
                losing_profit_rates = [t['profit_rate'] for t in losing_trades]
                results['mean_losing_profit'] = np.mean(losing_profit_rates)
                results['max_losing_profit'] = np.min(losing_profit_rates)
        
        # 샤프 비율 계산 (리스크 대비 수익)
        if returns and len(returns) > 1:
            returns_array = np.array(returns)
            results['sharpe_ratio'] = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        
        return results
    
    def print_results(self, results: Dict):
        """결과 출력"""
        logger.info("=" * 80)
        logger.info("📊 Backtest Results")
        logger.info("=" * 80)
        
        logger.info(f"\n📈 Performance:")
        logger.info(f"Episodes: {results['episodes']}")
        logger.info(f"Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        logger.info(f"Min/Max Reward: {results['min_reward']:.4f} / {results['max_reward']:.4f}")
        
        logger.info(f"\n💰 Trading:")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Winning Trades: {results['winning_trades']} ({results['win_rate']:.2%})")
        logger.info(f"Losing Trades: {results['losing_trades']}")
        
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
        
        if 'sharpe_ratio' in results:
            logger.info(f"\n📈 Risk Metrics:")
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        
        logger.info(f"\n✨ Auto Exits:")
        logger.info(f"Total Auto Exits: {results['auto_exits']}")
        logger.info(f"Stop Losses: {results['stop_losses']}")
        logger.info(f"Take Profits: {results['take_profits']}")
        
        logger.info(f"\n📈 Returns:")
        logger.info(f"Mean Return: {results['mean_return']:.4f}%")
        logger.info(f"Total Return: {results['total_return']:.4f}%")
        
        # 추론 엔진 통계
        inference_stats = self.inference.get_stats()
        logger.info(f"\n🤖 Inference Stats:")
        logger.info(f"Total Predictions: {inference_stats['total_predictions']}")
        logger.info(f"Buy Signal Rate: {inference_stats.get('buy_signal_rate', 0):.2%}")
        logger.info(f"Buy Filter Rate: {inference_stats.get('buy_filter_rate', 0):.2%}")
        logger.info(f"Auto Exit Rate: {inference_stats.get('auto_exit_rate', 0):.2%}")
        
        logger.info("=" * 80)
    
    def close(self):
        """리소스 정리"""
        self.env.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Strategy Backtest")
    parser.add_argument(
        '--model',
        type=str,
        default='models/grpo_direct_features@20251016/direct_features_model.pt',
        help='Model path'
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
        '--min-buy-confidence',
        type=float,
        default=0.0,
        help='Minimum buy confidence'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=-2.0,
        help='Stop loss rate (%)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=5.0,
        help='Take profit rate (%)'
    )
    
    args = parser.parse_args()
    
    # 경로 확인
    if not os.path.exists(args.model):
        logger.error(f"❌ Model not found: {args.model}")
        return False
    
    if not os.path.exists(args.db):
        logger.error(f"❌ Database not found: {args.db}")
        return False
    
    try:
        # 백테스팅 실행
        backtest = StrategyBacktest(
            model_path=args.model,
            db_path=args.db,
            device=args.device,
            min_buy_confidence=args.min_buy_confidence,
            stop_loss_rate=args.stop_loss,
            take_profit_rate=args.take_profit
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
