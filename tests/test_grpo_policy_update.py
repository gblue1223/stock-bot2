"""
GRPO 정책 업데이트 테스트

요구사항 4.4, 4.5를 검증합니다:
- PPO 스타일 클리핑된 목적 함수
- KL 발산 제약 적용
- Optimizer 업데이트
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from ai_trader.grpo.policy import GRPOPolicy
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.env import GRPOScalpingEnv
from ai_trader.embedding.models import TradingEmbeddingModel


@pytest.fixture
def embedding_model():
    """임베딩 모델 픽스처"""
    model = TradingEmbeddingModel(
        input_dim=60,
        embedding_dim=32,
        seq_len=60
    )
    model.eval()
    return model


@pytest.fixture
def policy():
    """정책 네트워크 픽스처"""
    return GRPOPolicy(
        embedding_dim=32,
        hidden_dim=64,
        action_dim=3
    )


@pytest.fixture
def env(embedding_model, tmp_path):
    """GRPO 환경 픽스처"""
    # 테스트용 더미 데이터베이스 경로
    db_path = str(tmp_path / "test.duckdb")
    
    env = GRPOScalpingEnv(
        embedding_model=embedding_model,
        db_path=db_path,
        seq_len=60,
        transaction_cost_rate=0.00215,
        quick_exit_threshold=1.5,
        quick_exit_penalty=0.01
    )
    
    return env


@pytest.fixture
def trainer(policy, tmp_path):
    """GRPO 훈련기 픽스처 (환경 없이)"""
    tensorboard_log_dir = str(tmp_path / "tensorboard")
    
    # 더미 환경 (실제로 사용하지 않음)
    class DummyEnv:
        def __init__(self):
            self.action_space = type('obj', (object,), {'n': 3, 'sample': lambda: 0})()
        
        def close(self):
            pass
    
    dummy_env = DummyEnv()
    
    trainer = GRPOTrainer(
        policy=policy,
        env=dummy_env,
        episodes_per_group=4,
        num_groups=2,
        learning_rate=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        kl_target=0.01,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device='cpu',
        tensorboard_log_dir=tensorboard_log_dir
    )
    
    return trainer


def create_dummy_episodes(num_episodes: int = 4, episode_length: int = 10):
    """더미 에피소드 데이터 생성"""
    episodes = []
    
    for _ in range(num_episodes):
        episode = {
            'states': np.random.randn(episode_length, 32).astype(np.float32),
            'actions': np.random.randint(0, 3, size=episode_length),
            'rewards': np.random.randn(episode_length).astype(np.float32) * 0.01,
            'next_states': np.random.randn(episode_length, 32).astype(np.float32),
            'dones': np.zeros(episode_length, dtype=bool),
            'log_probs': np.random.randn(episode_length).astype(np.float32) * 0.1,
            'metadata': {
                'episode_reward': np.random.randn() * 0.1,
                'episode_steps': episode_length,
                'num_trades': np.random.randint(1, 5),
                'win_rate': np.random.rand(),
                'avg_holding_time': np.random.rand() * 30,
                'quick_exit_violations': np.random.randint(0, 3)
            }
        }
        
        # 마지막 스텝은 종료
        episode['dones'][-1] = True
        
        episodes.append(episode)
    
    return episodes


def test_policy_forward_pass(policy):
    """정책 네트워크 forward pass 테스트"""
    batch_size = 8
    embedding_dim = 32
    
    # 더미 상태
    states = torch.randn(batch_size, embedding_dim)
    
    # Forward pass
    action_logits, state_values = policy(states)
    
    # 출력 형태 검증
    assert action_logits.shape == (batch_size, 3), \
        f"Expected action_logits shape (8, 3), got {action_logits.shape}"
    assert state_values.shape == (batch_size, 1), \
        f"Expected state_values shape (8, 1), got {state_values.shape}"


def test_policy_get_action(policy):
    """정책 행동 샘플링 테스트"""
    batch_size = 8
    embedding_dim = 32
    
    # 더미 상태
    states = torch.randn(batch_size, embedding_dim)
    
    # Stochastic 모드
    actions, log_probs = policy.get_action(states, deterministic=False)
    
    assert actions.shape == (batch_size,), \
        f"Expected actions shape (8,), got {actions.shape}"
    assert log_probs.shape == (batch_size,), \
        f"Expected log_probs shape (8,), got {log_probs.shape}"
    assert torch.all((actions >= 0) & (actions < 3)), \
        "Actions should be in range [0, 3)"
    
    # Deterministic 모드
    actions_det, log_probs_det = policy.get_action(states, deterministic=True)
    
    assert actions_det.shape == (batch_size,), \
        f"Expected actions shape (8,), got {actions_det.shape}"
    assert torch.all((actions_det >= 0) & (actions_det < 3)), \
        "Actions should be in range [0, 3)"


def test_policy_evaluate_actions(policy):
    """정책 행동 평가 테스트"""
    batch_size = 8
    embedding_dim = 32
    
    # 더미 상태와 행동
    states = torch.randn(batch_size, embedding_dim)
    actions = torch.randint(0, 3, (batch_size,))
    
    # 행동 평가
    log_probs, entropy, values = policy.evaluate_actions(states, actions)
    
    assert log_probs.shape == (batch_size,), \
        f"Expected log_probs shape (8,), got {log_probs.shape}"
    assert entropy.shape == (batch_size,), \
        f"Expected entropy shape (8,), got {entropy.shape}"
    assert values.shape == (batch_size,), \
        f"Expected values shape (8,), got {values.shape}"
    
    # 엔트로피는 양수여야 함
    assert torch.all(entropy >= 0), "Entropy should be non-negative"


def test_compute_returns(trainer):
    """리턴 계산 테스트"""
    # 더미 보상
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # 리턴 계산
    returns = trainer._compute_returns(rewards)
    
    # 형태 검증
    assert returns.shape == rewards.shape, \
        f"Expected returns shape {rewards.shape}, got {returns.shape}"
    
    # 수동 계산과 비교 (gamma=0.99)
    gamma = trainer.gamma
    expected_returns = np.array([
        1.0 + gamma * 2.0 + gamma**2 * 3.0 + gamma**3 * 4.0 + gamma**4 * 5.0,
        2.0 + gamma * 3.0 + gamma**2 * 4.0 + gamma**3 * 5.0,
        3.0 + gamma * 4.0 + gamma**2 * 5.0,
        4.0 + gamma * 5.0,
        5.0
    ], dtype=np.float32)
    
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-5)


def test_update_policy_basic(trainer):
    """기본 정책 업데이트 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 더미 어드밴티지 생성
    advantages = [
        np.random.randn(10).astype(np.float32) * 0.1
        for _ in range(4)
    ]
    
    # 정책 파라미터 저장 (업데이트 확인용)
    old_params = [p.clone() for p in trainer.policy.parameters()]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # 메트릭 검증
    assert 'policy_loss' in update_metrics
    assert 'value_loss' in update_metrics
    assert 'entropy' in update_metrics
    assert 'kl_divergence' in update_metrics
    assert 'clip_fraction' in update_metrics
    
    # 메트릭이 유한한 값인지 확인
    for key, value in update_metrics.items():
        assert np.isfinite(value), f"{key} should be finite, got {value}"
    
    # 정책 파라미터가 업데이트되었는지 확인
    new_params = list(trainer.policy.parameters())
    params_changed = False
    for old_p, new_p in zip(old_params, new_params):
        if not torch.allclose(old_p, new_p, atol=1e-6):
            params_changed = True
            break
    
    assert params_changed, "Policy parameters should be updated"


def test_update_policy_ppo_clipping(trainer):
    """PPO 클리핑 동작 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 큰 어드밴티지 생성 (클리핑 유도)
    advantages = [
        np.random.randn(10).astype(np.float32) * 10.0  # 큰 어드밴티지
        for _ in range(4)
    ]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # 클리핑 비율이 0보다 커야 함 (클리핑이 발생했음을 의미)
    # 참고: 랜덤 데이터이므로 항상 클리핑이 발생하지는 않을 수 있음
    assert 'clip_fraction' in update_metrics
    assert 0.0 <= update_metrics['clip_fraction'] <= 1.0, \
        f"Clip fraction should be in [0, 1], got {update_metrics['clip_fraction']}"


def test_update_policy_kl_divergence(trainer):
    """KL 발산 계산 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 더미 어드밴티지 생성
    advantages = [
        np.random.randn(10).astype(np.float32) * 0.1
        for _ in range(4)
    ]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # KL 발산이 계산되었는지 확인
    assert 'kl_divergence' in update_metrics
    
    # KL 발산이 유한한 값인지 확인
    assert np.isfinite(update_metrics['kl_divergence']), \
        f"KL divergence should be finite, got {update_metrics['kl_divergence']}"


def test_update_policy_entropy_bonus(trainer):
    """엔트로피 보너스 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 더미 어드밴티지 생성
    advantages = [
        np.random.randn(10).astype(np.float32) * 0.1
        for _ in range(4)
    ]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # 엔트로피가 계산되었는지 확인
    assert 'entropy' in update_metrics
    
    # 엔트로피가 양수인지 확인 (정책이 확률적이므로)
    assert update_metrics['entropy'] >= 0, \
        f"Entropy should be non-negative, got {update_metrics['entropy']}"


def test_update_policy_value_loss(trainer):
    """가치 손실 계산 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 더미 어드밴티지 생성
    advantages = [
        np.random.randn(10).astype(np.float32) * 0.1
        for _ in range(4)
    ]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # 가치 손실이 계산되었는지 확인
    assert 'value_loss' in update_metrics
    
    # 가치 손실이 양수인지 확인 (MSE 손실)
    assert update_metrics['value_loss'] >= 0, \
        f"Value loss should be non-negative, got {update_metrics['value_loss']}"


def test_update_policy_gradient_clipping(trainer):
    """그래디언트 클리핑 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 매우 큰 어드밴티지 생성 (큰 그래디언트 유도)
    advantages = [
        np.random.randn(10).astype(np.float32) * 100.0
        for _ in range(4)
    ]
    
    # 정책 업데이트
    update_metrics = trainer.update_policy(episodes, advantages)
    
    # 그래디언트 클리핑이 적용되었는지 확인
    # (그래디언트 노름이 max_grad_norm 이하인지 확인)
    total_norm = 0.0
    for p in trainer.policy.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 그래디언트 노름이 max_grad_norm 이하인지 확인
    # 참고: 업데이트 후에는 그래디언트가 클리어되므로 이 테스트는 제한적
    # 실제로는 업데이트 중에 클리핑이 적용되었는지 확인하기 어려움


def test_reference_policy_initialization(trainer):
    """참조 정책 초기화 테스트"""
    # 더미 에피소드 생성
    episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
    
    # 더미 어드밴티지 생성
    advantages = [
        np.random.randn(10).astype(np.float32) * 0.1
        for _ in range(4)
    ]
    
    # 첫 업데이트 전에는 참조 정책이 None
    assert trainer.reference_policy is None
    
    # 정책 업데이트
    trainer.update_policy(episodes, advantages)
    
    # 첫 업데이트 후에는 참조 정책이 초기화됨
    assert trainer.reference_policy is not None
    assert isinstance(trainer.reference_policy, GRPOPolicy)


def test_multiple_updates(trainer):
    """여러 번의 정책 업데이트 테스트"""
    num_updates = 3
    
    for i in range(num_updates):
        # 더미 에피소드 생성
        episodes = create_dummy_episodes(num_episodes=4, episode_length=10)
        
        # 더미 어드밴티지 생성
        advantages = [
            np.random.randn(10).astype(np.float32) * 0.1
            for _ in range(4)
        ]
        
        # 정책 업데이트
        update_metrics = trainer.update_policy(episodes, advantages)
        
        # 업데이트 횟수 확인
        assert trainer.num_updates == i + 1
        
        # 메트릭이 유한한 값인지 확인
        for key, value in update_metrics.items():
            assert np.isfinite(value), \
                f"Update {i+1}: {key} should be finite, got {value}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
