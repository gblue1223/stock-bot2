#!/usr/bin/env python3
"""
개선된 추론 엔진 검증 스크립트

주의점을 반영한 EnhancedGRPOInference 검증

중요:
    - DirectFeatureEnv의 state[0]은 등락률(%)
    - 누적 수익률을 계산하여 Position의 가격으로 사용
    - cumulative_return *= (1 + return_rate)
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.environments.direct_feature_env import DirectFeatureEnv, Action
from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Position

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_enhanced(
    model_path: str,
    db_path: str,
    num_episodes: int = 100,
    min_buy_confidence: float = 0.4,
    stop_loss_rate: float = -2.0,
    take_profit_rate: float = 5.0
):
    """
    개선된 모델 검증
    
    Args:
        model_path: 모델 경로
        db_path: 데이터베이스 경로
        num_episodes: 검증 에피소드 수
        min_buy_confidence: Buy 최소 신뢰도
        stop_loss_rate: 손절 비율
        take_profit_rate: 익절 비율
    """
    
    logger.info("=" * 80)
    logger.info("🔍 Enhanced Model Validation")
    logger.info("=" * 80)
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # 환경 생성
    logger.info("🎯 Creating environment...")
    env = DirectFeatureEnv(
        db_path=db_path,
        table_name='datasets_raw',
        seq_len=30,
        expected_features=24,
        transaction_cost_rate=0.00215,
        max_episode_steps=100,
        device=device
    )
    
    # 기본 추론 엔진 초기화
    logger.info(f"📦 Loading model: {model_path}")
    base_inference = GRPOInference(
        policy_path=model_path,
        embedding_model_path=None,
        device=device,
        use_torchscript=False
    )
    
    # 개선된 추론 엔진 초기화
    logger.info("✨ Creating enhanced inference engine...")
    enhanced_inference = EnhancedGRPOInference(
        base_inference=base_inference,
        min_buy_confidence=min_buy_confidence,
        min_sell_confidence=0.35,
        stop_loss_rate=stop_loss_rate,
        take_profit_rate=take_profit_rate,
        max_holding_period=100,
        enable_auto_exit=True
    )
    
    logger.info("=" * 80)
    logger.info(f"🎯 Running validation on {num_episodes} episodes...")
    logger.info("=" * 80)
    
    # 통계 수집
    episode_rewards = []
    episode_returns = []
    all_trades = []
    all_win_rates = []
    action_counts = defaultdict(int)
    
    # 개선된 통계
    total_buy_attempts = 0
    filtered_buys = 0
    auto_exits = 0
    stop_losses = 0
    take_profits = 0
    
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        current_position = None
        
        # 누적 수익률 추적 (등락률 -> 가격 변환)
        cumulative_return = 1.0  # 시작 = 100%
        
        while True:
            # 현재 등락률 (state[0])
            current_return_rate = state[0] if len(state) > 0 else 0.0
            
            # 누적 수익률 업데이트
            cumulative_return *= (1 + current_return_rate)
            
            # 시퀀스 복원 (평탄화된 상태를 시퀀스로)
            sequence = state.reshape(30, 24)
            
            # 개선된 예측
            action, confidence, pred_info = enhanced_inference.predict(
                sequence=sequence,
                current_position=current_position,
                current_price=cumulative_return,  # 누적 수익률 사용
                deterministic=True
            )
            
            # 통계 업데이트
            if pred_info['raw_action'] == Action.BUY.value:
                total_buy_attempts += 1
                if pred_info.get('filtered', False):
                    filtered_buys += 1
            
            if pred_info.get('auto_exit', False):
                auto_exits += 1
                exit_reason = pred_info.get('exit_reason', '')
                if 'Stop loss' in exit_reason:
                    stop_losses += 1
                elif 'Take profit' in exit_reason:
                    take_profits += 1
            
            # 포지션 관리
            if action == Action.BUY.value and current_position is None:
                current_position = Position(
                    entry_price=cumulative_return,
                    entry_time=info.get('step', 0),
                    current_price=cumulative_return,
                    holding_period=0
                )
            elif action == Action.SELL.value and current_position is not None:
                current_position = None
            
            # 행동 카운트
            action_counts[Action(action).name] += 1
            
            # 환경 스텝
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                # 에피소드 통계
                episode_rewards.append(episode_reward)
                
                if 'episode' in step_info:
                    ep_info = step_info['episode']
                    all_trades.append(ep_info.get('num_trades', 0))
                    all_win_rates.append(ep_info.get('win_rate', 0.0))
                    episode_returns.append(ep_info.get('total_return', 0.0))
                
                if (episode_idx + 1) % 10 == 0:
                    logger.info(f"Progress: {episode_idx + 1}/{num_episodes} episodes")
                
                break
    
    env.close()
    
    # 결과 분석
    logger.info("=" * 80)
    logger.info("📊 Validation Results")
    logger.info("=" * 80)
    
    # 기본 통계
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Mean Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    logger.info(f"Min Reward: {np.min(episode_rewards):.4f}")
    logger.info(f"Max Reward: {np.max(episode_rewards):.4f}")
    
    # 거래 통계
    if all_trades:
        logger.info(f"\n📈 Trading Statistics:")
        logger.info(f"Mean Trades per Episode: {np.mean(all_trades):.2f} ± {np.std(all_trades):.2f}")
        logger.info(f"Total Trades: {sum(all_trades)}")
    
    # 승률 통계
    if all_win_rates:
        valid_win_rates = [wr for wr in all_win_rates if not np.isnan(wr)]
        if valid_win_rates:
            logger.info(f"\n🎯 Win Rate Statistics:")
            logger.info(f"Mean Win Rate: {np.mean(valid_win_rates):.2%} ± {np.std(valid_win_rates):.2%}")
    
    # 수익률 통계
    if episode_returns:
        logger.info(f"\n💰 Return Statistics:")
        logger.info(f"Mean Return: {np.mean(episode_returns):.4f}%")
        logger.info(f"Total Return: {sum(episode_returns):.4f}%")
        logger.info(f"Positive Returns: {sum(1 for r in episode_returns if r > 0)} ({sum(1 for r in episode_returns if r > 0)/len(episode_returns)*100:.1f}%)")
    
    # 행동 분포
    total_actions = sum(action_counts.values())
    logger.info(f"\n🎮 Action Distribution:")
    for action_name, count in sorted(action_counts.items()):
        logger.info(f"{action_name}: {count} ({count/total_actions*100:.1f}%)")
    
    # 개선된 통계
    logger.info(f"\n✨ Enhanced Features:")
    logger.info(f"Total Buy Attempts: {total_buy_attempts}")
    filter_rate = (filtered_buys / total_buy_attempts * 100) if total_buy_attempts > 0 else 0
    logger.info(f"Filtered Buys: {filtered_buys} ({filter_rate:.1f}%)")
    logger.info(f"Auto Exits: {auto_exits}")
    logger.info(f"  - Stop Losses: {stop_losses}")
    logger.info(f"  - Take Profits: {take_profits}")
    
    # 추론 엔진 통계
    stats = enhanced_inference.get_stats()
    logger.info(f"\n📊 Inference Engine Stats:")
    logger.info(f"Total Predictions: {stats['total_predictions']}")
    logger.info(f"Buy Signal Rate: {stats.get('buy_signal_rate', 0):.2%}")
    logger.info(f"Buy Filter Rate: {stats.get('buy_filter_rate', 0):.2%}")
    logger.info(f"Auto Exit Rate: {stats.get('auto_exit_rate', 0):.2%}")
    
    logger.info("=" * 80)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_trades': np.mean(all_trades) if all_trades else 0,
        'mean_win_rate': np.mean([wr for wr in all_win_rates if not np.isnan(wr)]) if all_win_rates else 0,
        'buy_filter_rate': filtered_buys / total_buy_attempts if total_buy_attempts > 0 else 0,
        'auto_exits': auto_exits,
        'stop_losses': stop_losses,
        'take_profits': take_profits
    }


def compare_models(model_path: str, db_path: str):
    """
    기본 vs 개선 모델 비교
    
    신뢰도 분포를 고려하여 현실적인 임계값 사용:
    - 95.1%가 신뢰도 0.4 미만
    - 따라서 0.36, 0.38 등 더 낮은 임계값 필요
    """
    logger.info("=" * 80)
    logger.info("🔬 Comparing Base vs Enhanced Inference")
    logger.info("=" * 80)
    
    # 기본 모델 (신뢰도 필터링 없음, 자동 청산만)
    logger.info("\n1️⃣ Testing with no filtering (0.0)...")
    results_none = validate_enhanced(
        model_path=model_path,
        db_path=db_path,
        num_episodes=50,
        min_buy_confidence=0.0,  # 모든 Buy 허용
        stop_loss_rate=-2.0,
        take_profit_rate=5.0
    )
    
    # 약한 필터링 (하위 20% 제거)
    logger.info("\n2️⃣ Testing with light filtering (0.36)...")
    results_light = validate_enhanced(
        model_path=model_path,
        db_path=db_path,
        num_episodes=50,
        min_buy_confidence=0.36,  # 약한 필터링
        stop_loss_rate=-2.0,
        take_profit_rate=5.0
    )
    
    # 중간 필터링 (하위 50% 제거)
    logger.info("\n3️⃣ Testing with moderate filtering (0.358)...")
    results_moderate = validate_enhanced(
        model_path=model_path,
        db_path=db_path,
        num_episodes=50,
        min_buy_confidence=0.358,  # 중간 필터링 (많이 나오는 0.358159 기준)
        stop_loss_rate=-2.0,
        take_profit_rate=5.0
    )
    
    # 비교 결과
    logger.info("\n" + "=" * 80)
    logger.info("📊 Comparison Results")
    logger.info("=" * 80)
    
    logger.info(f"\n{'Metric':<30} {'None (0.0)':<15} {'Light (0.36)':<15} {'Moderate (0.358)':<15}")
    logger.info("-" * 75)
    logger.info(f"{'Mean Reward':<30} {results_none['mean_reward']:<15.4f} {results_light['mean_reward']:<15.4f} {results_moderate['mean_reward']:<15.4f}")
    logger.info(f"{'Mean Trades':<30} {results_none['mean_trades']:<15.2f} {results_light['mean_trades']:<15.2f} {results_moderate['mean_trades']:<15.2f}")
    logger.info(f"{'Buy Filter Rate':<30} {results_none['buy_filter_rate']:<15.2%} {results_light['buy_filter_rate']:<15.2%} {results_moderate['buy_filter_rate']:<15.2%}")
    logger.info(f"{'Auto Exits':<30} {results_none['auto_exits']:<15} {results_light['auto_exits']:<15} {results_moderate['auto_exits']:<15}")
    logger.info(f"{'Stop Losses':<30} {results_none['stop_losses']:<15} {results_light['stop_losses']:<15} {results_moderate['stop_losses']:<15}")
    logger.info(f"{'Take Profits':<30} {results_none['take_profits']:<15} {results_light['take_profits']:<15} {results_moderate['take_profits']:<15}")
    
    logger.info("\n💡 Recommendations:")
    logger.info("Based on the results above:")
    
    # 최고 보상 찾기
    best_reward = max(results_none['mean_reward'], results_light['mean_reward'], results_moderate['mean_reward'])
    if results_none['mean_reward'] == best_reward:
        logger.info("  🏆 Best: No filtering (0.0) - Auto exits alone work well")
    elif results_light['mean_reward'] == best_reward:
        logger.info("  🏆 Best: Light filtering (0.36) - Good balance")
    else:
        logger.info("  🏆 Best: Moderate filtering (0.358) - Quality over quantity")
    
    logger.info("=" * 80)


def main():
    """메인 함수"""
    
    model_path = "models/grpo_direct_features@20251016/direct_features_model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    
    # 경로 확인
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # 비교 검증 실행
        compare_models(model_path, db_path)
        
        logger.info("\n✅ Validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
