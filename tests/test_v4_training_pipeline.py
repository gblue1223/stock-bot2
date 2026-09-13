"""Exercise replay, cached rollout, update and evaluation with opening prices."""
import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.environments import DummyVecEnv
from ai_trader.grpo.evaluation import evaluate_policy
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.train_xlstm import TrainingConfig, create_environment, create_policy
from lib.market_data import canonical_feature_columns


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_v4_timed_rollout_update_and_batched_validation(tmp_path, device):
    columns = canonical_feature_columns(29)
    rows = 80
    price = 100. + np.arange(rows) * .03
    features = np.zeros((rows, len(columns)), np.float32)
    features[:, columns.index('현재가')] = price
    features[:, columns.index('시초가')] = price[0]
    features[:, columns.index('시초가대비등락률')] = (price / price[0] - 1) * 100
    features[:, columns.index('누적거래대금')] = np.arange(rows) * 1000
    metadata = np.array([['TEST', '20260908', str(90000000 + i * 250)] for i in range(rows)])
    np.savez(tmp_path / 'episode.npz', features=features, metadata=metadata,
             execution_last_price=price, execution_bid_prices=(price - .05)[:, None],
             execution_ask_prices=(price + .05)[:, None],
             execution_bid_sizes=np.full((rows, 1), 1000),
             execution_ask_sizes=np.full((rows, 1), 1000))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'metadata': {'feature_columns': columns, 'price_unit': 'krw'},
        'episodes': [{'file_path': 'episode.npz', 'stock_code': 'TEST',
                      'date': '20260908', 'length': rows}],
    }), encoding='utf-8')
    config = TrainingConfig()
    config.__dict__.update(extracted_dir=str(tmp_path), seq_len=8, episode_steps=4,
                           liquidation_max_steps=8, rolling_window_size=8, rolling_min_samples=1,
                           decision_interval_seconds=.5, episode_duration_seconds=2.,
                           cnn_channels=4, rnn_hidden_dim=4, hidden_dim=8, checkpoint_segments=2,
                           batch_size=4, initial_cash=1000., num_epochs=1, group_advantage_coef=0.)
    config.validate()
    factory = lambda seed: create_environment(config, device, ['20260908'], seed)
    vector = DummyVecEnv([lambda: factory(17), lambda: factory(18)])
    validators = [factory(19), factory(20)]
    try:
        torch.manual_seed(19)
        policy = create_policy(config, vector.envs[0], device)
        with torch.no_grad():
            # Retain input-dependent logits so batch/encoder differences remain
            # visible to the likelihood check, while preserving trading coverage.
            policy.policy_head.weight.mul_(0.1)
            policy.policy_head.bias.copy_(torch.tensor([0., 2., 1.], device=device))
        assert torch.count_nonzero(policy.policy_head.weight) > 0
        trainer = GRPOTrainer(policy, vector, episodes_per_group=2, num_groups=1,
                              batch_size=4, num_epochs=1, group_advantage_coef=0.,
                              device=device, observation_schema=vector.envs[0].observation_schema,
                              policy_update_checks=True, kl_probe_samples=4)
        episodes = trainer.collect_rollouts(4)
        assert len(episodes) == 4
        assert all(ep['states'].shape == (4, 8, 65) for ep in episodes)
        for ep in episodes:
            assert ep['metadata']['policy_duration_seconds'] == 2.
            assert ep['metadata']['market_steps_taken'] >= 8
            assert sum(ep['rewards']) == pytest.approx(ep['metadata']['net_return'], abs=1e-5)
            assert ep['metadata']['pnl_attribution_residual'] == pytest.approx(0., abs=1e-8)
            with torch.no_grad():
                logs, _, values = policy.evaluate_actions(torch.from_numpy(ep['states']).to(device),
                                                         torch.from_numpy(ep['actions']).to(device),
                                                         **({'action_masks': ep['action_masks']}
                                                            if policy.execution_action_mask else {}))
            np.testing.assert_allclose(logs.cpu().numpy(), ep['log_probs'], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(values.cpu().numpy(), ep['values'], rtol=1e-5, atol=1e-6)
        before = policy.policy_head.bias.detach().clone()
        metrics = trainer.update_policy(episodes, [np.zeros(4, np.float32)] * 4)
        assert metrics['optimizer_steps'] > 0
        assert all(np.isfinite(value) for value in metrics.values())
        assert not torch.equal(before, policy.policy_head.bias)
        serial = evaluate_policy(policy, validators[0], num_episodes=3, seed=31, device=device)
        batched = evaluate_policy(policy, validators, num_episodes=3, seed=31, device=device)
        assert serial['episode_net_returns'] == pytest.approx(batched['episode_net_returns'])
        assert serial['diagnostics']['action_counts'] == batched['diagnostics']['action_counts']
        assert batched['mean_policy_duration_seconds'] == 2.
        assert batched['pnl_attribution_residual'] == pytest.approx(0., abs=1e-8)
    finally:
        vector.close()
        for env in validators:
            env.close()
