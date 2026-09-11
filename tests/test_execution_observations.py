"""Execution costs stay observable without changing legacy feature or mask layout."""
import json

import numpy as np
import pytest
import torch

from lib.observations import (
    ACCOUNT_FIELDS, EXECUTION_FIELDS, EXECUTION_NORMALIZATION, ObservationBuilder,
)
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import (
    EnhancedGRPOInferenceXLSTM, GRPOInferenceE2EXLSTM, Position,
)
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.pretrain_behavior_cloning import OfflineEpisodeDataset, distillation_loss


def builder(version=4):
    return ObservationBuilder(['price'], seq_len=8, rolling_window_size=8,
                              rolling_min_samples=1, account_observations=version >= 3,
                              execution_observations=version >= 4)


def account(quantity=0, price=100.):
    result = dict.fromkeys(ACCOUNT_FIELDS, 0.)
    exposure = quantity * price / 10000.
    result.update(cash_ratio=1. - exposure, position_value_ratio=exposure,
                  stage_1_value_ratio=exposure, remaining_steps_ratio=1.)
    return result


def execution(**overrides):
    result = dict.fromkeys(EXECUTION_FIELDS, 0.)
    result.update(spread_bps=2., quoted_round_trip_cost_bps=8., liquidation_return=-.0008,
                  breakeven_return=.0008, buy_depth_ratio=1., sell_depth_ratio=1.,
                  quote_timestamp_known=1., remaining_time_fraction=1.)
    result.update(overrides)
    return result


def test_execution_schema_roundtrips_and_v2_v3_dictionaries_remain_exact():
    old = builder(2)
    assert old.schema == {'version': 2, 'feature_columns': ['price'], 'seq_len': 8,
                          'rolling_window_size': 8, 'rolling_min_samples': 1,
                          'max_holding_seconds': 300., 'max_stages': 1,
                          'stage_fields': ['is_active', 'profit_rate', 'holding_fraction'],
                          'normalization': 'causal_window_log_zscore_v1', 'feature_price_unit': 'krw'}
    v3 = builder(3)
    assert v3.schema == dict(old.schema, version=3, account_fields=list(ACCOUNT_FIELDS),
                             account_normalization='initial_cash_marked_value_v1')
    spec = builder()
    assert spec.schema == dict(v3.schema, version=4, execution_fields=list(EXECUTION_FIELDS),
                               execution_normalization=EXECUTION_NORMALIZATION)
    loaded = ObservationBuilder.from_schema(json.loads(json.dumps(spec.schema)))
    assert loaded.schema == spec.schema
    assert loaded.account_observations and loaded.execution_observations
    assert loaded.obs_dim == 37
    assert ObservationBuilder([f'feature_{i}' for i in range(27)], account_observations=True,
                              execution_observations=True).obs_dim == 63
    for changed in (dict(spec.schema, execution_fields=list(reversed(EXECUTION_FIELDS))),
                    dict(spec.schema, execution_normalization='unknown'),
                    {k: v for k, v in spec.schema.items() if k != 'account_fields'}):
        with pytest.raises(ValueError, match='Incompatible'):
            ObservationBuilder.from_schema(changed)


@pytest.mark.parametrize('kwargs,match', [
    ({'execution_observations': True}, 'requires account_observations'),
    ({'account_observations': True, 'execution_observations': 1}, 'boolean'),
])
def test_execution_schema_requires_account_features_and_explicit_boolean(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ObservationBuilder(['price'], **kwargs)


def test_same_price_different_fees_remain_distinguishable_with_unchanged_legacy_slices():
    raw, spec = np.full((8, 1), 100.), builder()
    stages = [{'entry_price': 100., 'entry_time_seconds': 0.}]
    low_cost = spec.build(raw, stages, 100., 1., account(1), execution())
    high_cost = spec.build(raw, stages, 100., 1., account(1), execution(
        quoted_round_trip_cost_bps=28., liquidation_return=-.0028, breakeven_return=.0028))
    legacy = builder(3).build(raw, stages, 100., 1., account(1))
    np.testing.assert_array_equal(low_cost[:, :12], legacy[:, :12])
    np.testing.assert_array_equal(low_cost[:, -15:], legacy[:, -15:])
    np.testing.assert_array_equal(low_cost[:, :12], high_cost[:, :12])
    np.testing.assert_array_equal(low_cost[:, -15:], high_cost[:, -15:])
    changed = np.flatnonzero(low_cost[-1] != high_cost[-1])
    np.testing.assert_array_equal(changed, [12 + EXECUTION_FIELDS.index(name) for name in
        ('quoted_round_trip_cost_bps', 'liquidation_return', 'breakeven_return')])


def test_signed_liquidation_and_breakeven_returns_are_not_clipped():
    values = execution(liquidation_return=.04, breakeven_return=-.02,
                       quote_age_fraction=2., episode_elapsed_fraction=1.05)
    observation = builder().build(np.ones((8, 1)), account_state=account(), execution_state=values)
    for name, expected in values.items():
        np.testing.assert_allclose(observation[:, 12 + EXECUTION_FIELDS.index(name)], expected)


@pytest.mark.parametrize('state,match', [
    (None, 'explicit execution_state'), ({}, 'missing required'),
    (execution(spread_bps=np.nan), 'finite numeric'),
    (execution(liquidation_return=np.inf), 'finite numeric'),
    (execution(liquidation_return=-1e100), 'finite float32'),
    (execution(spread_bps=-.1), 'spread_bps cannot be negative'),
    (execution(quoted_round_trip_cost_bps=-1.), 'quoted_round_trip_cost_bps cannot be negative'),
    (execution(quote_age_fraction=-1.), 'quote_age_fraction cannot be negative'),
    (execution(episode_elapsed_fraction=-1.), 'episode_elapsed_fraction cannot be negative'),
    (execution(buy_depth_ratio=1.01), 'buy_depth_ratio must be'),
    (execution(sell_depth_ratio=-.01), 'sell_depth_ratio must be'),
    (execution(remaining_time_fraction=1.01), 'remaining_time_fraction must be'),
    (execution(quote_timestamp_known=.5), 'quote_timestamp_known must be'),
    (execution(liquidation_return=[0.]), 'finite numeric'),
])
def test_execution_schema_rejects_missing_nonfinite_and_invalid_ranges(state, match):
    with pytest.raises(ValueError, match=match):
        builder().build(np.ones((8, 1)), account_state=account(), execution_state=state)


def test_live_v4_enforces_execution_state_even_on_risk_exit_and_preserves_action_masks(tmp_path):
    spec = builder()
    policy = GRPOPolicyE2EXLSTM(obs_dim=spec.obs_dim, cnn_channels=4,
                               rnn_hidden_dim=4, fc_hidden_dim=8).eval()
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 10., 1.]))
    path = tmp_path / 'v4.pt'
    torch.save({'policy_state_dict': policy.state_dict(), 'observation_schema': spec.schema}, path)
    live = GRPOInferenceE2EXLSTM(path, device='cpu')
    raw = np.full((8, 1), 100.)
    with pytest.raises(ValueError, match='explicit account_state'):
        live.predict(raw, execution_state=execution())
    with pytest.raises(ValueError, match='explicit execution_state'):
        live.build_observation(raw, account_state=account())
    with pytest.raises(ValueError, match='explicit execution_state'):
        live.predict(raw, account_state=account())
    # Large execution features cannot be mistaken for active stage flags.
    assert live.predict(raw, account_state=account(),
                        execution_state=execution(spread_bps=500.))[0] == 1
    assert live.predict(raw, stages=[{'entry_price': 100., 'entry_time_seconds': 0.}],
                        current_price=100., current_time_seconds=1., account_state=account(1),
                        execution_state=execution())[0] == 2
    risk = EnhancedGRPOInferenceXLSTM(live)
    with pytest.raises(ValueError, match='explicit execution_state'):
        risk.predict(raw, Position(100., 0., 100., 0.), 97., current_time_seconds=1.,
                     account_state=account(1, 97.))
    assert risk.predict(raw, Position(100., 0., 100., 0.), 97., current_time_seconds=1.,
                        account_state=account(1, 97.), execution_state=execution())[2]['risk_exit_all']


def test_raw_distillation_rejects_fabricated_v4_execution_and_account_states(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'metadata': {'feature_columns': ['price'], 'price_unit': 'krw'}, 'episodes': []}), encoding='utf-8')
    with pytest.raises(ValueError, match='Schema v4.*recorded account states and execution states'):
        OfflineEpisodeDataset(tmp_path, 8, observation_schema=builder().schema)


def test_distillation_mask_ignores_extra_execution_features():
    states = torch.zeros(1, 8, builder().obs_dim)
    states[:, :, 1:-15] = 10.
    teacher = torch.tensor([[0., 2., 20.]])
    student = torch.tensor([[0., 2., -20.]], requires_grad=True)
    loss = distillation_loss(teacher, torch.zeros(1, 1), student, torch.zeros(1, 1),
                             states, feature_count=1, temperature=1.)
    torch.testing.assert_close(loss, torch.tensor(0.), atol=1e-6, rtol=0.)
