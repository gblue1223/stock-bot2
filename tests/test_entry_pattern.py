"""Time/notional boundaries, causal gating and actual BUY policy gradients."""
import json

import duckdb
import numpy as np
import pandas as pd
import pytest
import torch

from ai_trader.grpo.entry_pattern import (DEFAULT_ENTRY_PATTERN, entry_signal,
    fill_target, validate_entry_pattern, cumulative_buy_from_signed_trades)
from ai_trader.grpo.data_extractor import extract_data
from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.train_xlstm import TrainingConfig, create_environment, create_policy
from test_grpo_memory_batches import RecordingPolicy


CFG = dict(DEFAULT_ENTRY_PATTERN)


def test_amount_rise_and_time_must_qualify_together():
    t = np.array([0., 4.999, 5., 60., 65.001])
    c = np.array([0., 3e9, 3e9, 3e9, 3e9])
    p = np.array([100., 101., 101., 101., 101.])
    np.testing.assert_array_equal(entry_signal(t, p, c, CFG), [False, False, True, True, False])
    assert not entry_signal(t, p, c - np.minimum(c, 1), CFG).any()
    assert not entry_signal(t, np.minimum(p, 100.999), c, CFG).any()
    # Separate windows may not independently supply the two conditions.
    assert not entry_signal([0, 10, 20, 65], [102, 100, 100, 101], [0, 3e9, 3e9, 3e9], CFG)[-1]


def test_irregular_window_filter_matches_bruteforce_and_is_causal():
    rng = np.random.default_rng(33)
    t = np.cumsum(rng.choice([0., .1, 1., 4.3], 300))
    c = np.cumsum(rng.uniform(0, 4e8, len(t)))
    p = rng.uniform(98, 103, len(t))
    expected = []
    for i in range(len(t)):
        left = max(0, np.searchsorted(t, t[i] - 60, side='right') - 1)
        right = np.searchsorted(t, t[i] - 5, side='right') - 1
        expected.append(any(c[i] - c[j] >= 3e9 and p[i] >= p[j] * 1.01 for j in range(left, right + 1)))
    actual = entry_signal(t, p, c, CFG)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:120], entry_signal(t[:120], p[:120], c[:120], CFG))


@pytest.mark.parametrize('values', [[0, -1, 4e9], [0, 4e9, 3e9], [0, np.nan, 4e9]])
def test_invalid_buy_source_is_rejected(values):
    with pytest.raises(ValueError, match='cumulative BUY'):
        entry_signal([0, 5, 10], [100, 101, 102], values, CFG)


def test_signed_trades_exclude_sells_and_repeated_quote_rows():
    result = cumulative_buy_from_signed_trades([100, 100, 100, 101, 101, 102],
        [20, 10, 10, -5, -5, 2], [20, 30, 30, 35, 35, 37])
    np.testing.assert_array_equal(result, [0, 1000, 1000, 1000, 1000, 1204])
    with pytest.raises(ValueError, match='increments must match'):
        cumulative_buy_from_signed_trades([100, 101], [1, 2], [1, 5])


def test_no_negative_signs_means_all_trades_are_buys():
    # Strings without '+' and zero are valid; no negative sample is required.
    result = cumulative_buy_from_signed_trades([100, 100, 101, 101, 102, 102],
        ['0', '10', '5', '5', '2', '0'], [0, 10, 15, 15, 17, 17])
    np.testing.assert_array_equal(result, [0, 1000, 1505, 1505, 1709, 1709])


@pytest.mark.parametrize('key,value', [('min_buy_notional_krw', 0), ('policy_coef', float('nan')),
    ('min_window_seconds', 4), ('max_window_seconds', 61), ('target_min_seconds', .5),
    ('target_max_seconds', 6), ('min_price_return', True)])
def test_config_rejects_invalid_settings(key, value):
    with pytest.raises(ValueError):
        validate_entry_pattern({key: value})


def test_fill_target_uses_exact_inclusive_window_and_censors_incomplete_data():
    t = np.array([0., .999, 1., 5., 5.001, 6.])
    assert fill_target(t, np.array([99, 200, 99, 99, 200, 200]), 0, 100, CFG) == -1
    assert fill_target(t, np.array([99, 99, 100, 99, 99, 99]), 0, 100, CFG) == 1
    assert fill_target(t, np.array([99, 99, 99, 100, 99, 99]), 0, 100, CFG) == 1
    assert fill_target(t[:3], np.array([99, 99, 200]), 0, 100, CFG) is None
    assert fill_target(np.array([0., 6.]), np.array([99, 200]), 0, 100, CFG) is None
    # Order-decision price is irrelevant: compare against actual execution cost.
    assert fill_target(t, np.full(len(t), 100), 0, 101, CFG) == -1


def pattern_database(path, with_source=True):
    seconds = np.arange(100)
    times = [90000000 + (int(s) // 60) * 100000 + (int(s) % 60) * 1000 for s in seconds]
    # Only the first stock has the rising 30억 event at 5 seconds.
    frames = []
    for stock, rises in [('000001', True), ('000002', False)]:
        prices = np.where(seconds < 5, 100., 101. if rises else 100.)
        prices[7:12] = 102. if rises else 100.
        frame = pd.DataFrame({'종목코드': stock, '날짜': '20260101', '종목명': 'TEST',
            '시간': times, '현재가': prices, '등락률': prices - 100, '누적거래대금': seconds * 1e9})
        if with_source:
            frame['cumulative_buy_notional_krw'] = np.minimum(seconds, 5) * 6e8
        for side, delta in [('매수', -.1), ('매도', .1)]:
            for level in range(1, 11):
                frame[f'{side}호가{level}'] = prices + delta
                frame[f'{side}호가수량{level}'] = 1000
        frames.append(frame)
    with duckdb.connect(str(path)) as conn:
        conn.register('source_frame', pd.concat(frames))
        conn.execute('CREATE TABLE datasets AS SELECT * FROM source_frame')


def extract_pattern(tmp_path):
    db, output = tmp_path / 'source.duckdb', tmp_path / 'episodes'
    pattern_database(db)
    extract_data(str(db), 'datasets', str(output), 2, 29, 10, entry_pattern_config=CFG)
    return output


def test_missing_buy_data_fails_before_writing_output(tmp_path):
    db, output = tmp_path / 'source.duckdb', tmp_path / 'episodes'
    pattern_database(db, with_source=False)
    with pytest.raises(ValueError, match='actual cumulative BUY'):
        extract_data(str(db), 'datasets', str(output), 2, 29, 10, entry_pattern_config=CFG)
    assert not output.exists()


def test_extraction_sampling_masks_fill_labels_and_nav_accounting(tmp_path):
    output = extract_pattern(tmp_path)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert [ep['stock_code'] for ep in manifest['episodes']] == ['000001']
    assert manifest['metadata']['rejected_episodes']
    config = TrainingConfig()
    config.__dict__.update(extracted_dir=str(output), seq_len=2, features=29, episode_steps=10,
        liquidation_max_steps=2, decision_interval_seconds=1, episode_duration_seconds=10,
        rolling_window_size=2, rolling_min_samples=1, entry_pattern_config=CFG,
        initial_cash=1000., transaction_cost_rate=0, sell_tax_rate=0,
        execution_config={'order_latency_ms': 0, 'slippage_bps': 0, 'spread_bps': 0})
    env = create_environment(config, 'cpu')
    try:
        for seed in range(8):
            env.reset(seed=seed)
            assert env._entry_allowed()
        observation, info = env.reset(options={'episode_index': 0, 'start_index': 4})
        assert env.current_time_seconds == 9 * 3600 + 5
        assert info['action_mask'][1]
        rewards = []
        for action in [1, 2] + [0] * 12:
            _, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break
        ep = info['episode']
        assert sum(rewards) == pytest.approx(ep['net_return'])
        assert ep['entry_pattern']['policy_credit'][0] == 1
        assert ep['entry_pattern']['labeled_buy_decisions'] == 1
        assert ep['entry_pattern']['success_quantity'] > 0
        # The signal has expired by this late part of the same file.
        env.current_step = len(env.episode_data) - 1
        env._episode_entry_signal = np.zeros(len(env.episode_data), dtype=bool)
        assert not env.action_masks()[1]
        with pytest.raises(ValueError, match='entry-pattern gate'):
            env.reset(options={'episode_index': 0, 'start_index': 0})
    finally:
        env.close()
    with pytest.raises(ValueError, match='selection settings differ'):
        GRPOScalpingEnvXLSTM(extracted_dir=str(output), seq_len=2, expected_features=29,
                            execution_action_mask=True, entry_pattern_config={**CFG, 'min_buy_notional_krw': 4e9})


@pytest.mark.parametrize('target', [1., -1.])
def test_auxiliary_objective_changes_buy_probability_without_nav_reward(target):
    torch.set_num_threads(1)
    policy = RecordingPolicy()
    states = np.ones((4, 2, 2), dtype=np.float32)
    actions = np.ones(4, dtype=np.int64)
    with torch.no_grad():
        old_logs, _, _ = policy.evaluate_actions(torch.from_numpy(states), torch.from_numpy(actions))
    episode = {'states': states, 'actions': actions, 'log_probs': old_logs.numpy(),
               'rewards': np.zeros(4, np.float32), 'dones': np.array([False, False, False, True]),
               'metadata': {'entry_pattern': {'policy_credit': [target] * 4}}}
    trainer = GRPOTrainer(policy, object(), use_gae=False, batch_size=4, num_epochs=1,
                          entropy_coef=0, learning_rate=.001)
    metrics = trainer.update_policy([episode], [np.zeros(4, np.float32)])
    with torch.no_grad():
        new_logs, _, _ = policy.evaluate_actions(torch.from_numpy(states), torch.from_numpy(actions))
    assert (new_logs.mean() - old_logs.mean()).item() * target > 0
    assert metrics['entry_pattern/credited_buy_decisions'] == 4
    np.testing.assert_array_equal(episode['rewards'], 0)


def test_new_preset_and_sequence_length():
    assert TrainingConfig().seq_len == 2048
    from pathlib import Path
    config = TrainingConfig(str(Path(__file__).parents[1] / 'config/scalping_v3.example.json'))
    assert config.seq_len == 2048
    assert config.entry_pattern_config == CFG


def test_xlstm_rollout_update_and_evaluation_with_entry_patterns(tmp_path):
    from ai_trader.grpo.environments import DummyVecEnv
    from ai_trader.grpo.evaluation import evaluate_policy
    output = extract_pattern(tmp_path)
    config = TrainingConfig()
    config.__dict__.update(extracted_dir=str(output), seq_len=8, features=29, episode_steps=8,
        liquidation_max_steps=2, decision_interval_seconds=1, episode_duration_seconds=8,
        rolling_window_size=8, rolling_min_samples=1, entry_pattern_config=CFG,
        cnn_channels=4, rnn_hidden_dim=4, hidden_dim=8, checkpoint_segments=2,
        initial_cash=1000., transaction_cost_rate=0, sell_tax_rate=0,
        execution_config={'order_latency_ms': 0, 'slippage_bps': 0, 'spread_bps': 0})
    factory = lambda: create_environment(config, 'cpu', seed=42)
    vector, validation = DummyVecEnv([factory]), factory()
    try:
        policy = create_policy(config, vector.envs[0], 'cpu')
        with torch.no_grad():
            policy.policy_head.bias.copy_(torch.tensor([0., 3., 0.]))
        trainer = GRPOTrainer(policy, vector, batch_size=4, num_epochs=1,
                              policy_update_checks=True, group_advantage_coef=0.,
                              observation_schema=vector.envs[0].observation_schema)
        episodes = trainer.collect_rollouts(2)
        assert sum(ep['metadata']['entry_pattern']['labeled_buy_decisions'] for ep in episodes) > 0
        for ep in episodes:
            assert sum(ep['rewards']) == pytest.approx(ep['metadata']['net_return'])
        metrics = trainer.update_policy(episodes, [np.zeros(len(ep['rewards']), np.float32) for ep in episodes])
        assert metrics['optimizer_steps'] > 0
        assert metrics['entry_pattern/credited_buy_decisions'] > 0
        result = evaluate_policy(policy, validation, num_episodes=2, seed=42, device='cpu')
        assert result['entry_pattern']['labeled_buy_decisions'] > 0
        assert result['entry_pattern']['success_rate'] is not None
    finally:
        vector.close()
        validation.close()


def test_old_signatures_remain_legacy_and_new_objective_invalidates_best():
    from test_profit_experiment_config import legacy_best_checkpoint
    from ai_trader.grpo.evaluation import compatible_resume_best, evaluation_signature
    old, signature = legacy_best_checkpoint()
    old['extra_state']['evaluation_signature']['settings'].pop('entry_pattern_config')
    assert compatible_resume_best(old, old, signature)
    changed = evaluation_signature({**old['extra_state']['training_config'], 'entry_pattern_config': CFG},
                                   signature['date_splits'], signature['observation_schema'])
    assert not compatible_resume_best(old, old, changed)
