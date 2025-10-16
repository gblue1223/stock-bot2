#!/usr/bin/env python3
"""
직접 특징 모델 검증 스크립트

훈련된 모델을 테스트 데이터로 검증하고 성능 지표를 출력합니다.
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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# direct_features_train에서 환경과 정책 가져오기
from scripts.grpo.direct_features_train import DirectFeatureEnv, DirectFeaturePolicy, Action, Position

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model(
    model_path: str,
    db_path: str,
    num_episodes: int = 100,
    deterministic: bool = True
):
    """
    모델 검증
    
    Args:
        model_path: 모델 체크포인트 경로
        db_path: 데이터베이스 경로
        num_episodes: 검증 에피소드 수
        deterministic: 결정적 행동 선택 여부
    """
    
    logger.info("=" * 60)
    logger.info("🔍 Starting Model Validation")
    logger.info("=" * 60)
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Using device: {device}")
    
    # 환경 생성
    logger.info("🎯 Creating validation environment...")
    env = DirectFeatureEnv(
        db_path=db_path,
        table_name='datasets_raw',
        seq_len=30,
        expected_features=24,
        transaction_cost_rate=0.00215,
        max_episode_steps=100,
        device=device
    )
    
    # 모델 로드
    logger.info(f"📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = env.observation_space.shape[0]
    policy = DirectFeaturePolicy(
        input_dim=input_dim,
        hidden_dim=64,
        action_dim=3
    )
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.to(device)
    policy.eval()
    
    logger.info("✅ Model loaded successfully")
    logger.info(f"📊 Model info: iteration={checkpoint.get('iteration', 'N/A')}, "
                f"timesteps={checkpoint.get('total_timesteps', 'N/A')}")
    
    # 검증 실행
    logger.info(f"🎯 Running validation on {num_episodes} episodes...")
    logger.info("=" * 60)
    
    # 통계 수집
    episode_rewards = []
    episode_returns = []
    all_trades = []
    all_win_rates = []
    all_sharpe_ratios = []
    action_counts = defaultdict(int)
    
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while True:
            # 상태를 텐서로 변환
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # 정책에서 행동 선택
            with torch.no_grad():
                action_logits, _ = policy(state_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1).item()
                else:
                    action = torch.multinomial(action_probs, 1).item()
            
            # 행동 카운트
            action_counts[Action(action).name] += 1
            
            # 환경 스텝
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            state = next_state
            
            if done:
                # 에피소드 통계 수집
                episode_rewards.append(episode_reward)
                
                if 'episode' in step_info:
                    ep_info = step_info['episode']
                    all_trades.append(ep_info.get('num_trades', 0))
                    all_win_rates.append(ep_info.get('win_rate', 0.0))
                    all_sharpe_ratios.append(ep_info.get('sharpe_ratio', 0.0))
                    episode_returns.append(ep_info.get('total_return', 0.0))
                
                if (episode_idx + 1) % 10 == 0:
                    logger.info(f"Progress: {episode_idx + 1}/{num_episodes} episodes completed")
                
                break
    
    env.close()
    
    # 결과 분석
    logger.info("=" * 60)
    logger.info("📊 Validation Results")
    logger.info("=" * 60)
    
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
        logger.info(f"Episodes with Trades: {sum(1 for t in all_trades if t > 0)} ({sum(1 for t in all_trades if t > 0)/len(all_trades)*100:.1f}%)")
    
    # 승률 통계
    if all_win_rates:
        valid_win_rates = [wr for wr in all_win_rates if not np.isnan(wr)]
        if valid_win_rates:
            logger.info(f"\n🎯 Win Rate Statistics:")
            logger.info(f"Mean Win Rate: {np.mean(valid_win_rates):.2%} ± {np.std(valid_win_rates):.2%}")
            logger.info(f"Min Win Rate: {np.min(valid_win_rates):.2%}")
            logger.info(f"Max Win Rate: {np.max(valid_win_rates):.2%}")
            logger.info(f"Win Rate > 50%: {sum(1 for wr in valid_win_rates if wr > 0.5)} episodes")
    
    # 샤프 비율
    if all_sharpe_ratios:
        valid_sharpe = [sr for sr in all_sharpe_ratios if not np.isnan(sr) and not np.isinf(sr)]
        if valid_sharpe:
            logger.info(f"\n📉 Sharpe Ratio Statistics:")
            logger.info(f"Mean Sharpe: {np.mean(valid_sharpe):.4f} ± {np.std(valid_sharpe):.4f}")
            logger.info(f"Min Sharpe: {np.min(valid_sharpe):.4f}")
            logger.info(f"Max Sharpe: {np.max(valid_sharpe):.4f}")
            logger.info(f"Sharpe > 1.0: {sum(1 for sr in valid_sharpe if sr > 1.0)} episodes")
    
    # 수익률 통계
    if episode_returns:
        logger.info(f"\n💰 Return Statistics:")
        logger.info(f"Mean Return: {np.mean(episode_returns):.4f} ± {np.std(episode_returns):.4f}")
        logger.info(f"Total Return: {sum(episode_returns):.4f}")
        logger.info(f"Positive Returns: {sum(1 for r in episode_returns if r > 0)} ({sum(1 for r in episode_returns if r > 0)/len(episode_returns)*100:.1f}%)")
    
    # 행동 분포
    total_actions = sum(action_counts.values())
    logger.info(f"\n🎮 Action Distribution:")
    for action_name, count in sorted(action_counts.items()):
        logger.info(f"{action_name}: {count} ({count/total_actions*100:.1f}%)")
    
    logger.info("=" * 60)
    
    # 성능 평가
    logger.info("\n🎯 Performance Assessment:")
    
    mean_reward = np.mean(episode_rewards)
    mean_trades = np.mean(all_trades) if all_trades else 0
    mean_win_rate = np.mean([wr for wr in all_win_rates if not np.isnan(wr)]) if all_win_rates else 0
    mean_sharpe = np.mean([sr for sr in all_sharpe_ratios if not np.isnan(sr) and not np.isinf(sr)]) if all_sharpe_ratios else 0
    
    score = 0
    feedback = []
    
    # 보상 평가
    if mean_reward > 10:
        score += 3
        feedback.append("✅ Excellent reward performance")
    elif mean_reward > 5:
        score += 2
        feedback.append("✅ Good reward performance")
    elif mean_reward > 0:
        score += 1
        feedback.append("⚠️ Positive but low rewards")
    else:
        feedback.append("❌ Negative rewards - needs improvement")
    
    # 거래 활동 평가
    if mean_trades > 5:
        score += 2
        feedback.append("✅ Active trading strategy")
    elif mean_trades > 2:
        score += 1
        feedback.append("⚠️ Moderate trading activity")
    else:
        feedback.append("❌ Low trading activity")
    
    # 승률 평가
    if mean_win_rate > 0.55:
        score += 3
        feedback.append("✅ Excellent win rate")
    elif mean_win_rate > 0.50:
        score += 2
        feedback.append("✅ Good win rate")
    elif mean_win_rate > 0.45:
        score += 1
        feedback.append("⚠️ Acceptable win rate")
    else:
        feedback.append("❌ Low win rate")
    
    # 샤프 비율 평가
    if mean_sharpe > 1.0:
        score += 2
        feedback.append("✅ Excellent risk-adjusted returns")
    elif mean_sharpe > 0.5:
        score += 1
        feedback.append("⚠️ Moderate risk-adjusted returns")
    else:
        feedback.append("❌ Poor risk-adjusted returns")
    
    logger.info(f"Overall Score: {score}/10")
    for fb in feedback:
        logger.info(f"  {fb}")
    
    logger.info("=" * 60)
    
    return {
        'mean_reward': mean_reward,
        'mean_trades': mean_trades,
        'mean_win_rate': mean_win_rate,
        'mean_sharpe': mean_sharpe,
        'score': score
    }


def main():
    """메인 함수"""
    
    # 경로 설정
    model_path = "models/grpo_direct_features/direct_features_model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    
    # 경로 확인
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return False
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # 검증 실행
        results = validate_model(
            model_path=model_path,
            db_path=db_path,
            num_episodes=100,
            deterministic=True
        )
        
        logger.info("\n✅ Validation completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
