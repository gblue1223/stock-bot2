"""GAE and TensorBoard must never forward/transfer a whole long episode."""
import copy

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from ai_trader.grpo.grpo import GRPOTrainer


class RecordingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(19)
        self.actor = torch.nn.Linear(2, 3)
        self.critic = torch.nn.Linear(2, 1)
        self.forward_batches = []

    def forward(self, states):
        self.forward_batches.append((len(states), torch.is_grad_enabled()))
        features = states.mean(dim=1)
        return self.actor(features), self.critic(features)

    def evaluate_actions(self, states, actions):
        logits, values = self(states)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values.squeeze(-1)


def make_episode(policy, length, offset=0.0):
    states = np.arange(length * 8, dtype=np.float32).reshape(length, 4, 2) / 10 + offset
    actions = np.arange(length, dtype=np.int64) % 3
    with torch.no_grad():
        log_probs, _, values = policy.evaluate_actions(torch.from_numpy(states), torch.from_numpy(actions))
    rewards = np.linspace(-.1, .2, length, dtype=np.float32)
    return {
        'states': states, 'actions': actions, 'log_probs': log_probs.numpy(),
        'rewards': rewards, 'dones': np.arange(length) == length - 1,
        'metadata': {'episode_reward': float(rewards.sum()), 'episode_steps': length},
    }, values.numpy()


def record_observation_transfers(monkeypatch):
    transfers = []
    original = torch.Tensor.to

    def tracking_to(tensor, *args, **kwargs):
        if tensor.ndim == 3:
            transfers.append(len(tensor))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', tracking_to)
    return transfers


def test_gae_transfers_and_forwards_only_minibatches(monkeypatch):
    policy = RecordingPolicy()
    trainer = GRPOTrainer(policy, object(), batch_size=3, num_epochs=1, use_gae=True)
    episode, expected_values = make_episode(policy, 7)
    policy.forward_batches.clear()
    values_seen = []
    original_gae = trainer._compute_gae

    def checked_gae(rewards, dones, values):
        values_seen.append(values.copy())
        np.testing.assert_allclose(values, expected_values, rtol=1e-6, atol=1e-7)
        return original_gae(rewards, dones, values)

    monkeypatch.setattr(trainer, '_compute_gae', checked_gae)
    transfers = record_observation_transfers(monkeypatch)
    metrics = trainer.update_policy([episode], [np.linspace(-.2, .2, 7, dtype=np.float32)])
    assert len(values_seen) == 1
    assert [size for size, grad in policy.forward_batches if not grad] == [3, 3, 1]
    assert max(size for size, _ in policy.forward_batches) <= trainer.batch_size
    assert transfers and max(transfers) <= trainer.batch_size
    assert all(np.isfinite(value) for value in metrics.values())


def test_tensorboard_batches_preserve_episode_weighted_metrics(monkeypatch):
    policy = RecordingPolicy()
    trainer = GRPOTrainer(policy, object(), batch_size=3)
    trainer.reference_policy = copy.deepcopy(policy)
    with torch.no_grad():
        trainer.reference_policy.actor.bias.add_(torch.tensor([.2, -.1, .3]))
    episodes = [make_episode(policy, 7)[0], make_episode(policy, 5, offset=2.0)[0]]
    expected_entropies, expected_kls = [], []
    with torch.no_grad():
        for episode in episodes:
            states, actions = torch.from_numpy(episode['states']), torch.from_numpy(episode['actions'])
            logits, _ = policy(states)
            probabilities = torch.softmax(logits, dim=-1)
            # This is the pre-batching TensorBoard entropy definition.
            expected_entropies.append(-(probabilities * torch.log(probabilities + 1e-8)).sum(-1).mean().item())
            current, _, _ = policy.evaluate_actions(states, actions)
            reference, _, _ = trainer.reference_policy.evaluate_actions(states, actions)
            expected_kls.append((reference - current).mean().item())

    class Writer:
        def __init__(self):
            self.metrics = {}

        def add_scalar(self, key, value, iteration):
            self.metrics[key] = float(value)

    trainer.writer = Writer()
    policy.forward_batches.clear()
    trainer.reference_policy.forward_batches.clear()
    transfers = record_observation_transfers(monkeypatch)
    trainer._log_metrics(0, episodes, {0: episodes}, {'policy_loss': .123})

    assert [size for size, _ in policy.forward_batches] == [3, 3, 1, 3, 2]
    assert [size for size, _ in trainer.reference_policy.forward_batches] == [3, 3, 1, 3, 2]
    assert all(not grad for _, grad in policy.forward_batches)
    assert transfers and max(transfers) <= trainer.batch_size
    assert trainer.writer.metrics['group_0/policy_entropy'] == pytest.approx(np.mean(expected_entropies), abs=1e-6)
    assert trainer.writer.metrics['group_0/kl_divergence'] == pytest.approx(np.mean(expected_kls), abs=1e-6)
    assert trainer.writer.metrics['train/policy_loss'] == .123
