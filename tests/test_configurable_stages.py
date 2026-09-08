"""Position sizing, masks and checkpoint replay share the configured stage limit."""
import json

import numpy as np
import pytest
import torch

from lib.observations import ObservationBuilder, action_mask
from ai_trader.grpo.train_xlstm import TrainingConfig
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import GRPOInferenceE2EXLSTM
from test_scalping_accounting import ReplayFixture


@pytest.mark.parametrize('limit', [1, 2, 5])
def test_entry_budget_stage_limit_and_fifo_exit(limit):
    env = ReplayFixture([100] * 20, max_stages=limit)
    try:
        env.reset()
        for _ in range(limit + 2):
            env.step(1)
        assert env.position_steps == env.max_split_count == limit
        buys = [order for order in env.simulator.orders if order.side == 'buy']
        assert len(buys) == limit
        assert all(order.quantity == 100 // limit for order in buys)
        before = env.quantity
        env.step(2)
        assert env.quantity == before - 100 // limit
        if limit == 1:
            assert env.quantity == 0
            env.step(1)
            assert env.position_steps == 1  # May re-enter after a confirmed exit.
        assert env.observation_schema['max_stages'] == limit
        assert env.observation_space.shape == (1, 17)  # Five reserved slots.
    finally:
        env.close()


def test_partial_entry_reserves_single_stage_until_order_expires():
    depth = dict(ask_prices=np.full((8, 1), 100.), ask_sizes=np.full((8, 1), 2.),
                 bid_prices=np.full((8, 1), 99.), bid_sizes=np.full((8, 1), 100.))
    env = ReplayFixture([100] * 8, max_stages=1, execution=depth)
    try:
        env.reset()
        env.step(1)
        assert env.quantity == 2 and env.position_steps == 1
        for _ in range(3):
            env.step(1)
        assert len([o for o in env.simulator.orders if o.side == 'buy']) == 1
        env.step(2)
        assert env.quantity == 0
    finally:
        env.close()


@pytest.mark.parametrize('limit', [1, 2, 5])
def test_policy_inference_and_schema_restore_limit(tmp_path, limit):
    builder = ObservationBuilder(['price'], seq_len=8, max_stages=limit)
    policy = GRPOPolicyE2EXLSTM(obs_dim=builder.obs_dim, cnn_channels=4,
                               rnn_hidden_dim=4, fc_hidden_dim=8, max_stages=limit).eval()
    stages = [{'entry_price': 100., 'entry_time_seconds': 0.}] * limit
    raw = np.full((8, 1), 100.)
    full = builder.build(raw, stages, 100., 1.)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([0., 10., 1.]))
        logits, _ = policy(torch.from_numpy(full).unsqueeze(0))
    assert logits.softmax(-1)[0, 1] == 0
    assert not action_mask(limit, limit)[1]
    path = tmp_path / 'checkpoint.pt'
    # Existing v2 five-stage checkpoints have this schema and no max_stages in config.
    torch.save({'policy_state_dict': policy.state_dict(),
                'observation_schema': json.loads(json.dumps(builder.schema))}, path)
    live = GRPOInferenceE2EXLSTM(path, device='cpu')
    assert live.policy.max_stages == live.observation_builder.max_stages == limit
    assert live.predict(raw)[0] == 1
    assert live.predict(raw, stages=stages, current_price=100., current_time_seconds=1.)[0] == 2
    with pytest.raises(ValueError, match='max_stages'):
        live.build_observation(raw, stages=stages + stages[:1], current_price=100., current_time_seconds=1.)
    other = ObservationBuilder(['price'], seq_len=8, max_stages=2 if limit == 1 else 1)
    with pytest.raises(ValueError, match='Incompatible'):
        other.validate_schema(builder.schema)


@pytest.mark.parametrize('invalid', [0, 6, -1, 1.5, True, '2'])
def test_invalid_stage_limits_fail_before_data_loading(invalid):
    with pytest.raises(ValueError, match='max_stages'):
        ReplayFixture([100] * 8, max_stages=invalid)
    config = TrainingConfig()
    config.max_stages = invalid
    with pytest.raises(ValueError, match='max_stages'):
        config.validate()


def test_default_is_one():
    assert TrainingConfig().max_stages == ObservationBuilder(['price']).max_stages == 1
    np.testing.assert_array_equal(action_mask(1), [True, False, True])
