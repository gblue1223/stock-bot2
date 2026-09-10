"""Batching changes inference scheduling, never held-out paths or accounting."""
import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.evaluation import compatible_resume_best, evaluate_policy, evaluation_signature


class SeededEnv:
    def __init__(self, failure=None):
        self.paths = {}
        self.failure = failure
        self.closed = False

    def reset(self, seed):
        if self.failure == 'reset':
            raise RuntimeError('reset failed')
        assert seed not in self.paths
        self.seed, self.steps = seed, 0
        self.length = 1 + seed % 4
        self.paths[seed] = []
        return self.observation(), {}

    def observation(self):
        return np.array([[self.seed, self.steps]], dtype=np.float32)

    def step(self, action):
        if self.failure == 'step':
            raise RuntimeError('step failed')
        self.paths[self.seed].append(action)
        self.steps += 1
        sold = self.seed % 3
        episode = {
            'net_return': sum(self.paths[self.seed]) * 0.25 - sold * .21,
            'num_trades': sold, 'max_drawdown': sold * .21,
            'avg_holding_time': float(self.length), 'open_quantity': 0,
            'realized_net_pnl': float(sold), 'liquidation_complete': True,
            'execution_model': 'order_book',
            'episode_key': {'seed': self.seed, 'start_index': self.seed % 11},
            'submitted_orders': sold, 'submitted_quantity': sold * 10,
            'filled_quantity': sold * 10, 'partial_orders': 0,
            'cancelled_orders': 0, 'expired_orders': 0,
            'buy_action_outcomes': {'submitted': sold}, 'execution_blocked_checks': {},
            'round_trip_count': sold, 'round_trip_win_rate': float(sold == 1),
            'quantity_weighted_holding_time': float(self.length),
            'fill_count': sold * 2, 'fill_win_rate': .5 if sold else 0.,
            'fill_avg_holding_time': float(self.length),
            'round_trips': [{'quantity': 10, 'net_pnl': 1.0}] * sold,
            'liquidation_steps': self.length, 'liquidation_seconds': self.length * 1.25,
            'liquidation_stop_reason': 'flat', 'market_steps_taken': self.steps,
        }
        if self.failure == 'return':
            episode['net_return'] = float('nan')
        return self.observation(), 999., self.steps == self.length, False, {'episode': episode}

    def close(self):
        self.closed = True


class BatchedPolicy(torch.nn.Module):
    def __init__(self, invalid=None):
        super().__init__()
        self.batch_sizes = []
        self.invalid = invalid

    def get_action_with_probabilities(self, obs, deterministic=False):
        assert deterministic and not self.training and not torch.is_grad_enabled()
        self.batch_sizes.append(len(obs))
        actions = obs[:, 0, :].sum(-1).long() % 3
        probabilities = torch.nn.functional.one_hot(actions, 3).float() * .75 + .125
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        if self.invalid == 'probabilities':
            probabilities[0, 0] = float('inf')
        if self.invalid == 'actions':
            actions = actions.float()
            actions[0] = float('nan')
        if self.invalid == 'shape':
            actions = actions[:1]
        return actions, torch.zeros(len(obs)), probabilities

    def get_action(self, obs, deterministic=False):
        actions, log_probs, _ = self.get_action_with_probabilities(obs, deterministic)
        return actions, log_probs


@pytest.mark.parametrize('workers,episodes', [(2, 7), (3, 7), (8, 7), (8, 1)])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_batches_preserve_uneven_paths_actions_order_and_costs(workers, episodes, diagnostics):
    serial_env = SeededEnv()
    serial = evaluate_policy(BatchedPolicy(), serial_env, num_episodes=episodes,
                             seed=42, collect_diagnostics=diagnostics)
    environments = [SeededEnv() for _ in range(workers)]
    policy = BatchedPolicy()
    batched = evaluate_policy(policy, environments, num_episodes=episodes,
                              seed=42, collect_diagnostics=diagnostics)
    assert batched == serial
    assert {seed: actions for env in environments for seed, actions in env.paths.items()} == serial_env.paths
    assert sum(len(env.paths) for env in environments) == episodes
    assert set(serial_env.paths) == set(range(42, 42 + episodes))
    assert max(policy.batch_sizes) == min(workers, episodes)
    assert sum(policy.batch_sizes) == sum(len(path) for path in serial_env.paths.values())
    if episodes > 1:
        assert policy.batch_sizes[-1] < max(policy.batch_sizes)  # No padded final batch.
    assert policy.training and all(not env.closed for env in environments)
    assert batched['no_trade_episode_fraction'] == pytest.approx(
        sum(seed % 3 == 0 for seed in serial_env.paths) / episodes)
    assert batched['round_trip_count'] == sum(seed % 3 for seed in serial_env.paths)
    if diagnostics:
        rows = batched['diagnostics']['episodes']
        assert [row['seed'] for row in rows] == list(range(42, 42 + episodes))
        assert [row['steps'] for row in rows] == [1 + seed % 4 for seed in serial_env.paths]
        assert rows[0]['liquidation_stop_reason'] == 'flat'
        if episodes > 1:
            assert rows[1]['round_trips']


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('failure', ['reset', 'step', 'return', 'max_steps', 'probabilities', 'actions', 'shape'])
def test_failures_restore_original_mode_and_leave_caller_envs_open(training, failure):
    policy = BatchedPolicy(invalid=failure)
    policy.train(training)
    environments = [SeededEnv(failure=failure), SeededEnv()]
    with pytest.raises((RuntimeError, ValueError)):
        evaluate_policy(policy, environments, num_episodes=3,
                        max_steps=1 if failure == 'max_steps' else 10)
    assert policy.training is training
    assert all(not env.closed for env in environments)


def test_rejects_empty_or_shared_environments_before_reset():
    env = SeededEnv()
    for environments in ([], [env, env]):
        with pytest.raises(ValueError, match='independent'):
            evaluate_policy(BatchedPolicy(), environments)
    assert not env.paths


def test_absent_execution_metrics_do_not_claim_no_trades():
    class LegacyEnv(SeededEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info['episode'] = {'net_return': 0.0}
            return obs, reward, terminated, truncated, info

    metrics = evaluate_policy(BatchedPolicy(), [LegacyEnv(), LegacyEnv()], num_episodes=2)
    assert metrics['no_trade_episode_fraction'] is None
    assert metrics['round_trip_count'] is None
    assert metrics['round_trip_win_rate'] is None
    assert metrics['diagnostics']['episodes'][0]['round_trip_count'] is None


def test_liquidation_tail_changes_evaluation_signature_but_legacy_means_disabled():
    config, splits, schema = {}, {'validation': ['20260101']}, {'version': 2}
    legacy = evaluation_signature(config, splits, schema)
    assert legacy['settings']['liquidation_max_steps'] == 0
    assert legacy != evaluation_signature({'liquidation_max_steps': 300}, splits, schema)


def test_old_signed_resume_remains_compatible_only_without_a_liquidation_tail():
    config = {'output_dir': '/run', 'extracted_dir': '/data'}
    splits = {'train': ['20250101'], 'validation': ['20260101'], 'test': ['20270101']}
    schema = {'version': 2}
    signature = evaluation_signature(config, splits, schema)
    old_signature = copy.deepcopy(signature)
    old_signature['settings'].pop('liquidation_max_steps')
    checkpoint = {
        'policy_state_dict': {'weight': torch.ones(1)}, 'observation_schema': schema,
        'extra_state': {'date_splits': splits, 'training_config': config,
                        'evaluation_signature': old_signature},
    }
    assert compatible_resume_best(checkpoint, checkpoint, signature)
    with_tail = evaluation_signature({**config, 'liquidation_max_steps': 300}, splits, schema)
    assert not compatible_resume_best(checkpoint, checkpoint, with_tail)
    assert 'liquidation_max_steps' not in old_signature['settings']  # Read-only migration.


def test_real_xlstm_batched_outputs_preserve_fixed_policy_and_weights():
    from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

    class ObservationEnv(SeededEnv):
        def observation(self):
            obs = np.zeros((8, 17), dtype=np.float32)
            obs[:, 0] = self.seed / 100
            obs[:, 1] = self.steps / 10
            return obs

    torch.set_num_threads(1)
    policy = GRPOPolicyE2EXLSTM(obs_dim=17, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([2., 1., 3.]))
    before = copy.deepcopy(policy.state_dict())
    serial = evaluate_policy(policy, ObservationEnv(), num_episodes=7)
    batch = evaluate_policy(policy, [ObservationEnv() for _ in range(3)], num_episodes=7)
    assert batch == serial
    assert policy.training
    for name, weights in before.items():
        torch.testing.assert_close(policy.state_dict()[name], weights, rtol=0, atol=0)
