import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.evaluation import chronological_date_split, evaluate_policy, validate_checkpoint_dates
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM


def tiny_policy():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return GRPOPolicyE2EXLSTM(obs_dim=17, cnn_channels=4, rnn_hidden_dim=4,
                             fc_hidden_dim=8, checkpoint_segments=2)


def test_rollout_update_and_batch_sizes_have_the_same_likelihood():
    policy = tiny_policy()
    obs = torch.randn(4, 16, 17)
    policy.eval()
    with torch.no_grad():
        actions, old = policy.get_action(obs)
    before = policy.bn1.running_mean.clone()
    policy.train()
    new, _, values = policy.evaluate_actions(obs, actions)
    alone, _, _ = policy.evaluate_actions(obs[:1], actions[:1])
    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alone, old[:1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(policy.bn1.running_mean, before)
    (new.sum() + values.sum()).backward()
    assert policy.conv1.weight.grad is not None
    assert torch.isfinite(policy.conv1.weight.grad).all()


def test_grouping_cannot_remove_returns_by_clustering_outcomes():
    trainer = GRPOTrainer(tiny_policy(), env=object(), num_groups=4)
    episodes = [dict(rewards=np.full(10, reward, dtype=np.float32),
                     metadata={'episode_reward': reward * 10,
                               'initial_market_indicators': [0.1, 0.0, 0.5]})
                for reward in (-10, -1, 1, 10) for _ in range(4)]
    groups = trainer.group_episodes(episodes)
    advantages = trainer.compute_group_relative_advantages(groups)
    assert len(groups) == 1
    values = next(iter(advantages.values()))
    assert any(np.any(value < 0) for value in values)
    assert any(np.any(value > 0) for value in values)


def test_date_splits_hold_out_whole_dates_and_apply_embargo():
    days = ['202601%02d' % day for day in range(1, 11)]
    split = chronological_date_split(days + days, '20260105', '20260108', embargo_dates=1)
    assert split == {'train': days[:5], 'validation': days[6:8], 'test': days[9:]}
    assert set(split['train']).isdisjoint(split['validation'])
    assert set(split['train']).isdisjoint(split['test'])
    with pytest.raises(ValueError, match='three distinct'):
        chronological_date_split(days[:2])


def test_pretrained_date_lineage_cannot_contaminate_holdouts():
    splits = {'train': ['20260101'], 'validation': ['20260102'], 'test': ['20260103']}
    validate_checkpoint_dates({'extra_state': {'date_splits': splits}}, splits)
    with pytest.raises(ValueError, match='lineage'):
        validate_checkpoint_dates({}, splits)
    with pytest.raises(ValueError, match='reserved'):
        validate_checkpoint_dates({'extra_state': {'date_splits': {'train': ['20260102']}}}, splits)
    with pytest.raises(ValueError, match='model selection'):
        validate_checkpoint_dates({'extra_state': {'date_splits': {
            'train': ['20260101'], 'validation': ['20260103']}}}, splits)
    with pytest.raises(ValueError, match='model selection'):
        validate_checkpoint_dates({'extra_state': {'date_splits': {
            'train': ['20260101'], 'validation': ['20260104']}}}, splits)


def test_explicit_resume_restores_optimizer_and_all_counters():
    source = GRPOTrainer(torch.nn.Linear(1, 1), object(), learning_rate=0.0123)
    source.policy(torch.ones(1, 1)).sum().backward()
    source.optimizer.step()
    checkpoint = {'optimizer_state_dict': copy.deepcopy(source.optimizer.state_dict()),
                  'total_timesteps': 1234, 'num_updates': 27, 'iteration': 29}
    target = GRPOTrainer(torch.nn.Linear(1, 1), object(), learning_rate=0.5)
    assert not target.optimizer.state
    assert target.total_timesteps == target.num_updates == 0
    iteration = target.restore_training_progress(checkpoint)
    assert (target.total_timesteps, target.num_updates, iteration) == (1234, 27, 29)
    assert target.learning_rate == 0.0123
    assert target.optimizer.param_groups[0]['lr'] == 0.0123
    for key, expected_state in checkpoint['optimizer_state_dict']['state'].items():
        for field, value in expected_state.items():
            torch.testing.assert_close(target.optimizer.state_dict()['state'][key][field], value)
    with pytest.raises(ValueError, match='full training checkpoint'):
        target.restore_training_progress({'iteration': 123})


class EvaluationEnv:
    def __init__(self):
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return np.zeros((8, 17), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((8, 17)), 9999, True, False, {
            'episode': {'net_return': -0.2, 'num_trades': 1,
                        'liquidation_complete': False, 'open_quantity': 3}}


def test_evaluation_uses_net_nav_and_repeatable_seeds():
    policy = tiny_policy()
    env = EvaluationEnv()
    before = copy.deepcopy(policy.state_dict())
    first = evaluate_policy(policy, env, num_episodes=2, seed=91)
    second = evaluate_policy(policy, env, num_episodes=2, seed=91)
    assert first == second
    assert first['mean_net_return'] == -0.2
    assert first['incomplete_liquidation_episodes'] == 2
    assert first['max_open_quantity'] == 3
    assert env.seeds == [91, 92, 91, 92]
    assert policy.training
    for key, value in before.items():
        torch.testing.assert_close(value, policy.state_dict()[key])


def test_best_checkpoint_scores_the_exact_updated_weights(tmp_path):
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.zero_()
    evaluations = []

    def evaluate(current):
        weight = float(current.weight.item())
        evaluations.append(weight)
        return {'mean_net_return': weight}

    class SyntheticTrainer(GRPOTrainer):
        def collect_rollouts(self, num_episodes):
            self.total_timesteps += 1
            reward = 100 if self.num_updates == 0 else -100
            return [{'rewards': np.array([reward], dtype=np.float32),
                     'metadata': {'episode_reward': reward, 'episode_steps': 1,
                                  'initial_market_indicators': [0, 0, 0]}}]

        def update_policy(self, episodes, advantages):
            with torch.no_grad():
                self.policy.weight.add_(1)
            self.num_updates += 1
            return {'policy_loss': 0.0}

    trainer = SyntheticTrainer(policy, object(), episodes_per_group=1, num_groups=1,
                               evaluation_callback=evaluate)
    trainer.train(2, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    saved = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=False)
    assert evaluations == [1.0, 2.0]
    assert saved['policy_state_dict']['weight'].item() == 2
    assert saved['extra_state']['best_validation_return'] == 2
    assert saved['extra_state']['selection_metric'] == 'validation.mean_net_return'


@pytest.mark.parametrize('use_gae', [False, True])
def test_cpu_policy_update_has_finite_gradients(use_gae):
    policy = tiny_policy()
    trainer = GRPOTrainer(policy, object(), batch_size=2, num_epochs=1, use_gae=use_gae)
    states = torch.randn(4, 8, 17)
    policy.eval()
    with torch.no_grad():
        actions, log_probs = policy.get_action(states)
    episode = {'states': states.numpy(), 'actions': actions.numpy(),
               'log_probs': log_probs.numpy(), 'rewards': np.array([0, .1, -.2, .3], dtype=np.float32),
               'dones': np.array([False, False, False, True])}
    before = policy.policy_head.weight.detach().clone()
    metrics = trainer.update_policy([episode], [np.array([-.3, -.1, .1, .3], dtype=np.float32)])
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, policy.policy_head.weight)


def test_action_masks_are_shared_by_rollout_and_update():
    policy = tiny_policy()
    states = torch.zeros(2, 8, 17)
    states[1, :, -15::3] = 1
    logits, _ = policy(states)
    probabilities = torch.softmax(logits, dim=-1)
    assert probabilities[0, 2] == 0  # Cannot sell flat inventory.
    assert probabilities[1, 1] == 0  # Cannot buy a sixth stage.


@pytest.mark.parametrize('workers', [1, 2])
def test_training_cli_end_to_end_with_three_date_holdout(tmp_path, workers):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    data = tmp_path / 'episodes'
    data.mkdir()
    episodes = []
    for index, date in enumerate(('20260101', '20260102', '20260103')):
        rows = 24
        price = 10000 + np.arange(rows) * ((-1) ** index) * 2
        features = np.column_stack((price, (price / 10000 - 1) * 100,
                                    1000000 + np.arange(rows) * 10000)).astype(np.float32)
        metadata = np.column_stack((np.full(rows, 5930), np.full(rows, int(date)),
                                    90000000 + np.arange(rows) * 1000)).astype(np.int64)
        name = 'episode_' + date + '.npz'
        np.savez(data / name, features=features, metadata=metadata,
                 execution_last_price=price.astype(np.float64))
        episodes.append({'file_path': name, 'stock_code': '005930', 'date': date, 'length': rows})
    manifest = {'metadata': {'schema_version': 2, 'feature_columns': ['현재가', '등락률', '누적거래대금'],
                              'price_unit': 'krw', 'feature_transform': 'raw', 'return_rate_index': 1},
                'episodes': episodes}
    (data / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    output = tmp_path / 'output'
    config = {'extracted_dir': str(data), 'output_dir': str(output), 'device': 'cpu',
              'seq_len': 8, 'features': 3, 'episode_steps': 4, 'total_timesteps': 8,
              'episodes_per_group': 2, 'num_groups': 1, 'num_workers': workers, 'batch_size': 2,
              'num_epochs': 1, 'cnn_channels': 4, 'rnn_hidden_dim': 4, 'hidden_dim': 8,
              'checkpoint_segments': 2, 'evaluation_episodes': 1, 'evaluation_interval': 1,
              'rolling_window_size': 8, 'rolling_min_samples': 1, 'max_holding_seconds': 2,
              'use_gae': True}
    expected_stages = 1 if workers == 1 else 5
    if workers == 2:
        config['max_stages'] = 2  # CLI must override this JSON value.
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    stage_args = [] if workers == 1 else ['--max_stages', '5']
    process = subprocess.run([sys.executable, '-m', 'ai_trader.grpo.train_xlstm', '--config', str(config_path), *stage_args],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True,
                             env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=30)
    assert process.returncode == 0, process.stderr.decode('utf-8', errors='replace')[-4000:]
    report = json.loads((output / 'evaluation_report.json').read_text(encoding='utf-8'))
    assert report['date_splits'] == {'train': ['20260101'], 'validation': ['20260102'], 'test': ['20260103']}
    assert report['training'] == {'total_timesteps': 8, 'num_updates': 1}
    assert report['test_used_for_model_selection'] is False
    assert report['test']['synthetic_execution_episodes'] == 1
    selected = torch.load(output / 'scalping_xlstm_model.pt', weights_only=False)
    assert selected['observation_schema']['feature_columns'] == manifest['metadata']['feature_columns']
    assert selected['extra_state']['validation_metrics'] == report['validation']
    assert selected['observation_schema']['max_stages'] == expected_stages
    assert selected['extra_state']['training_config']['max_stages'] == expected_stages
    config['max_stages'] = expected_stages

    # A historical iteration larger than this run's length must never suppress
    # fine-tuning; only --resume restores progress and Adam state.
    selected['iteration'] = 100
    previous = tmp_path / 'historical_checkpoint.pt'
    torch.save(selected, previous)
    source_optimizer_step = next(iter(selected['optimizer_state_dict']['state'].values()))['step'].item()
    for resume in (False, True):
        destination = tmp_path / ('resumed' if resume else 'fine_tuned')
        next_config = {**config, 'output_dir': str(destination), 'load_policy': str(previous),
                       'total_timesteps': 16 if resume else 8}
        next_path = tmp_path / ('resume.json' if resume else 'fine_tune.json')
        next_path.write_text(json.dumps(next_config), encoding='utf-8')
        command = [sys.executable, '-m', 'ai_trader.grpo.train_xlstm', '--config', str(next_path)]
        if resume:
            command.append('--resume')
        run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                             env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=30)
        assert run.returncode == 0, run.stderr.decode('utf-8', errors='replace')[-4000:]
        progress = torch.load(destination / 'checkpoints' / (
            'checkpoint_iter101.pt' if resume else 'checkpoint_iter1.pt'), weights_only=False)
        assert progress['iteration'] == (101 if resume else 1)
        assert progress['total_timesteps'] == (16 if resume else 8)
        assert progress['num_updates'] == (2 if resume else 1)
        step = next(iter(progress['optimizer_state_dict']['state'].values()))['step'].item()
        assert step == source_optimizer_step * (2 if resume else 1)
        run_report = json.loads((destination / 'evaluation_report.json').read_text(encoding='utf-8'))
        assert run_report['training'] == {'total_timesteps': 16 if resume else 8,
                                         'num_updates': 2 if resume else 1}
