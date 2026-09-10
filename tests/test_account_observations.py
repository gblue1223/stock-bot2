"""Account observations distinguish exposure without breaking legacy checkpoints."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.observations import ACCOUNT_FIELDS, ObservationBuilder
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import (
    EnhancedGRPOInferenceXLSTM, GRPOInferenceE2EXLSTM, Position,
)
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.pretrain_behavior_cloning import OfflineEpisodeDataset, distillation_loss


def account(quantity=0, price=100., initial_cash=10000., **overrides):
    state = dict.fromkeys(ACCOUNT_FIELDS, 0.)
    exposure = quantity * price / initial_cash
    state.update(cash_ratio=1. - exposure, position_value_ratio=exposure,
                 stage_1_value_ratio=exposure, remaining_steps_ratio=1.)
    state.update(overrides)
    return state


def builder(account_observations=True):
    return ObservationBuilder(['price'], seq_len=8, rolling_window_size=8,
                              rolling_min_samples=1, account_observations=account_observations)


def tiny_policy(spec):
    return GRPOPolicyE2EXLSTM(obs_dim=spec.obs_dim, cnn_channels=4,
                             rnn_hidden_dim=4, fc_hidden_dim=8, checkpoint_segments=2)


def test_schema_v2_is_exact_and_v3_roundtrips_with_explicit_fields():
    legacy = builder(False)
    assert legacy.schema == {'version': 2, 'feature_columns': ['price'], 'seq_len': 8,
                             'rolling_window_size': 8, 'rolling_min_samples': 1,
                             'max_holding_seconds': 300., 'max_stages': 1,
                             'stage_fields': ['is_active', 'profit_rate', 'holding_fraction'],
                             'normalization': 'causal_window_log_zscore_v1', 'feature_price_unit': 'krw'}
    modern = builder()
    assert modern.schema['version'] == 3
    assert modern.schema['account_fields'] == list(ACCOUNT_FIELDS)
    assert ObservationBuilder.from_schema(json.loads(json.dumps(modern.schema))).schema == modern.schema
    with pytest.raises(ValueError, match='Incompatible'):
        legacy.validate_schema(modern.schema)
    with pytest.raises(ValueError, match='Incompatible'):
        ObservationBuilder.from_schema(dict(modern.schema, account_fields=list(reversed(ACCOUNT_FIELDS))))


def test_quantity_changes_account_features_but_preserves_market_and_stage_layout():
    raw = np.full((8, 1), 100.)
    spec = builder()
    stage = [{'entry_price': 100., 'entry_time_seconds': 0., 'quantity': 1}]
    small = spec.build(raw, stage, 100., 1., account_state=account(1))
    stage[0]['quantity'] = 100
    large = spec.build(raw, stage, 100., 1., account_state=account(100))
    legacy = builder(False).build(raw, stage, 100., 1.)
    assert small.shape == (8, 1 + 11 + 15)
    np.testing.assert_array_equal(small[:, :1], large[:, :1])
    np.testing.assert_array_equal(small[:, -15:], large[:, -15:])
    np.testing.assert_array_equal(small[:, -15:], legacy[:, -15:])
    assert not np.array_equal(small, large)
    assert small[-1, 1 + ACCOUNT_FIELDS.index('position_value_ratio')] == pytest.approx(.01)
    assert large[-1, 1 + ACCOUNT_FIELDS.index('position_value_ratio')] == 1.


@pytest.mark.parametrize('field,value', [('cash_ratio', .5), ('pending_buy_value_ratio', .2),
                                        ('pending_sell_value_ratio', .2), ('exit_pending', 1.),
                                        ('remaining_steps_ratio', .1)])
def test_cash_pending_orders_exit_and_horizon_remain_observable(field, value):
    spec, raw = builder(), np.full((8, 1), 100.)
    first = spec.build(raw, account_state=account())
    second = spec.build(raw, account_state=account(**{field: value}))
    changed = np.flatnonzero(first[-1] != second[-1])
    np.testing.assert_array_equal(changed, [1 + ACCOUNT_FIELDS.index(field)])


@pytest.mark.parametrize('state,match', [
    (None, 'explicit account_state'), ({}, 'missing required'),
    (account(cash_ratio=np.nan), 'finite numeric'),
    (account(cash_ratio=-.1), 'negative'),
    (account(remaining_steps_ratio=1.1), 'remaining_steps_ratio'),
    (account(position_value_ratio=.1), 'sum of filled stage'),
    (account(stage_2_value_ratio=.1), 'filled FIFO'),
])
def test_account_schema_rejects_missing_nonfinite_or_inconsistent_state(state, match):
    with pytest.raises(ValueError, match=match):
        builder().build(np.ones((8, 1)), account_state=state)


def test_live_v3_requires_account_data_and_masks_keep_trailing_stage_layout(tmp_path):
    spec = builder()
    policy = tiny_policy(spec).eval()
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 10., 1.]))
    path = tmp_path / 'v3.pt'
    torch.save({'policy_state_dict': policy.state_dict(), 'observation_schema': spec.schema}, path)
    live = GRPOInferenceE2EXLSTM(path, device='cpu')
    raw = np.full((8, 1), 100.)
    with pytest.raises(ValueError, match='explicit account_state'):
        live.predict(raw)
    assert live.predict(raw, account_state=account())[0] == 1
    stage = [{'entry_price': 100., 'entry_time_seconds': 0.}]
    assert live.predict(raw, stages=stage, current_price=100., current_time_seconds=1.,
                        account_state=account(1))[0] == 2
    risk = EnhancedGRPOInferenceXLSTM(live)
    with pytest.raises(ValueError, match='explicit account_state'):
        risk.predict(raw, Position(100., 0., 100., 0.), 97., current_time_seconds=1.)


def test_single_rollout_forward_returns_same_likelihood_and_value_and_skips_checkpointing():
    torch.manual_seed(5)
    spec = builder()
    policy = tiny_policy(spec).train()
    obs = torch.from_numpy(spec.build(np.ones((8, 1)), account_state=account())).unsqueeze(0)
    with torch.no_grad(), patch.object(policy, 'forward', wraps=policy.forward) as forward, \
            patch('ai_trader.grpo.policies.scalping_policy_xlstm.cp.checkpoint') as checkpoint:
        actions, old_log_probs, old_values = policy.get_action_with_value(obs, deterministic=True)
        assert forward.call_count == 1
        checkpoint.assert_not_called()
    with patch('ai_trader.grpo.policies.scalping_policy_xlstm.cp.checkpoint',
               wraps=torch.utils.checkpoint.checkpoint) as checkpoint:
        log_probs, _, values = policy.evaluate_actions(obs, actions)
        assert checkpoint.call_count > 0
        torch.testing.assert_close(log_probs, old_log_probs)
        torch.testing.assert_close(values, old_values)
        (log_probs.sum() + values.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


@pytest.mark.parametrize('head,message', [('policy_head', 'action logits'), ('value_head', 'value predictions')])
def test_nonfinite_model_outputs_fail_with_named_error(head, message):
    spec, policy = builder(), tiny_policy(builder()).eval()
    obs = torch.from_numpy(spec.build(np.ones((8, 1)), account_state=account())).unsqueeze(0)
    with torch.no_grad():
        getattr(policy, head).bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match=message):
        policy.get_action_with_value(obs)


def test_nonfinite_observations_and_invalid_actions_are_not_silently_repaired():
    spec, policy = builder(), tiny_policy(builder()).eval()
    obs = torch.from_numpy(spec.build(np.ones((8, 1)), account_state=account())).unsqueeze(0)
    with pytest.raises(ValueError, match='integer indices'):
        policy.evaluate_actions(obs, torch.tensor([3]))
    obs[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='observation contains nonfinite'):
        policy(obs)


def test_raw_distillation_cannot_invent_account_states_for_v3(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'metadata': {'feature_columns': ['price'], 'price_unit': 'krw'}, 'episodes': []}), encoding='utf-8')
    with pytest.raises(ValueError, match='requires recorded account states'):
        OfflineEpisodeDataset(tmp_path, 8, observation_schema=builder().schema)


def test_distillation_mask_uses_trailing_stages_with_extra_account_features():
    states = torch.zeros(1, 8, 27)
    states[:, :, 1:12] = 10.  # Must not be interpreted as active stage flags.
    teacher = torch.tensor([[0., 2., 20.]])
    student = torch.tensor([[0., 2., -20.]], requires_grad=True)
    loss = distillation_loss(teacher, torch.zeros(1, 1), student, torch.zeros(1, 1),
                             states, feature_count=1, temperature=1.)
    torch.testing.assert_close(loss, torch.tensor(0.), atol=1e-6, rtol=0.)
