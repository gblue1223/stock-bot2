import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ai_trader.grpo.evaluation import evaluate_policy
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM


class ReportingEnv:
    def __init__(self, reports=None):
        self.reports = reports or [{}]
        self.actions = []
        self.seeds = []

    def reset(self, seed):
        self.seeds.append(seed)
        self.steps = 0
        return np.zeros((8, 17), np.float32), {}

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        episode = {'net_return': 0.0, 'episode_key': {'date': '20250502', 'start_index': 4},
                   **self.reports[(len(self.seeds) - 1) % len(self.reports)]}
        return np.zeros((8, 17), np.float32), 9999.0, self.steps == 3, False, {'episode': episode}


def hold_policy():
    torch.set_num_threads(1)
    policy = GRPOPolicyE2EXLSTM(obs_dim=17, cnn_channels=4, rnn_hidden_dim=4,
                               fc_hidden_dim=8, checkpoint_segments=2)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        # SELL has the highest raw logit but is masked while flat.
        policy.policy_head.bias.copy_(torch.tensor([2.0, 1.0, 3.0]))
    return policy


def test_probabilities_use_one_forward_and_preserve_actions_weights_and_metrics():
    policy = hold_policy()
    before = copy.deepcopy(policy.state_dict())
    plain_env, diagnostic_env = ReportingEnv(), ReportingEnv()
    plain = evaluate_policy(policy, plain_env, num_episodes=2, collect_diagnostics=False)
    with patch.object(policy, 'forward', wraps=policy.forward) as forward:
        detailed = evaluate_policy(policy, diagnostic_env, num_episodes=2)
        assert forward.call_count == 6
    diagnostics = detailed.pop('diagnostics')
    assert detailed == plain
    assert plain_env.actions == diagnostic_env.actions == [0] * 6
    assert plain_env.seeds == diagnostic_env.seeds == [42, 43]
    assert diagnostics['action_counts'] == {'hold': 6, 'buy': 0, 'sell': 0}
    assert diagnostics['probability_steps'] == 6
    assert diagnostics['mean_action_probabilities'] == pytest.approx(
        {'hold': 0.7310586, 'buy': 0.2689414, 'sell': 0.0})
    assert diagnostics['max_action_probabilities'] == diagnostics['mean_action_probabilities']
    assert [ep['seed'] for ep in diagnostics['episodes']] == [42, 43]
    assert all(ep['episode_key']['start_index'] == 4 for ep in diagnostics['episodes'])
    assert policy.training
    for key, value in before.items():
        torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
    json.dumps(diagnostics, allow_nan=False)


class BuyPolicy(torch.nn.Module):
    def get_action(self, obs, deterministic=False):
        assert deterministic
        return torch.tensor([1]), torch.tensor([0.0])


def test_execution_counters_are_propagated_and_fill_ratio_is_quantity_weighted():
    reports = []
    for submitted, filled, expired in ((10, 10, 0), (100, 0, 1)):
        reports.append({
            'submitted_orders': 1, 'submitted_quantity': submitted,
            'filled_quantity': filled, 'partial_orders': 0,
            'cancelled_orders': 0, 'expired_orders': expired,
            'buy_action_outcomes': {'submitted': 1, 'pending_buy': 2},
            'execution_blocked_checks': {'stale_quote': 2 * expired},
        })
    diagnostics = evaluate_policy(BuyPolicy(), ReportingEnv(reports), num_episodes=2)['diagnostics']
    assert diagnostics['action_counts']['buy'] == 6
    assert diagnostics['mean_action_probabilities'] is None
    assert diagnostics['probability_steps'] == 0
    assert diagnostics['submitted_orders'] == 2
    assert diagnostics['submitted_quantity'] == 110
    assert diagnostics['filled_quantity'] == 10
    assert diagnostics['fill_ratio'] == pytest.approx(10 / 110)
    assert diagnostics['expired_orders'] == 1
    assert diagnostics['buy_action_outcomes'] == {'submitted': 2, 'pending_buy': 4}
    assert diagnostics['execution_blocked_checks'] == {'stale_quote': 2}
    assert diagnostics['episodes'][1]['fill_ratio'] == 0.0


def test_missing_order_counters_are_unknown_not_zero():
    diagnostics = evaluate_policy(BuyPolicy(), ReportingEnv(), num_episodes=1)['diagnostics']
    assert diagnostics['action_counts']['buy'] == 3
    for field in ('submitted_orders', 'submitted_quantity', 'filled_quantity',
                  'fill_ratio', 'buy_action_outcomes', 'execution_blocked_checks'):
        assert diagnostics[field] is None
        assert diagnostics['episodes'][0][field] is None


def test_diagnostic_failure_restores_policy_mode():
    policy = hold_policy()
    with pytest.raises(RuntimeError, match='max_steps'):
        evaluate_policy(policy, ReportingEnv(), num_episodes=1, max_steps=1)
    assert policy.training
