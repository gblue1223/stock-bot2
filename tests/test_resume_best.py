import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_trader.grpo.evaluation import compatible_resume_best, evaluation_signature
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.train_xlstm import load_resume_best_candidates


SPLITS = {'train': ['20260101'], 'validation': ['20260102'], 'test': ['20260103']}
SCHEMA = {'version': 'test-schema'}


def checkpoint(weight, iteration, config, *, signed=True):
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(weight)
    trainer = GRPOTrainer(policy, object(), observation_schema=SCHEMA)
    extra = {'training_config': dict(config), 'date_splits': copy.deepcopy(SPLITS),
             'best_validation_return': 999.0}  # Must be re-evaluated, not trusted.
    if signed:
        extra['evaluation_signature'] = evaluation_signature(config, SPLITS, SCHEMA)
    return {'policy_state_dict': copy.deepcopy(policy.state_dict()),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'observation_schema': SCHEMA, 'iteration': iteration,
            'total_timesteps': iteration, 'num_updates': iteration, 'extra_state': extra}


class WorseningTrainer(GRPOTrainer):
    def collect_rollouts(self, num_episodes):
        self.total_timesteps += 1
        return [{'rewards': np.array([0.0], dtype=np.float32),
                 'metadata': {'episode_reward': 0.0, 'episode_steps': 1,
                              'initial_market_indicators': [0, 0, 0]}}]

    def update_policy(self, episodes, advantages):
        with torch.no_grad():
            self.policy.weight.sub_(1)
        self.num_updates += 1
        return {'policy_loss': 0.0}


def test_resume_loader_preserves_compatible_profitable_candidate(tmp_path):
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    config = {'output_dir': str(tmp_path), 'extracted_dir': '/data/episodes'}
    source = checkpoint(2, 10, config)
    source_path = folder / 'checkpoint_iter10.pt'
    torch.save(source, source_path)
    for name, weight in [('checkpoint_best.pt', 4), ('checkpoint_best_profitable.pt', 3)]:
        torch.save(checkpoint(weight, 8, config), folder / name)
    candidates = load_resume_best_candidates(source, source_path, tmp_path,
                                             evaluation_signature(config, SPLITS, SCHEMA))
    assert sorted(float(item['policy_state_dict']['weight'].item()) for item in candidates) == [3., 4.]


@pytest.mark.parametrize('new_output', [False, True])
def test_resume_preserves_revalidated_prior_best_after_worse_update(tmp_path, new_output):
    source_dir = tmp_path / 'source'
    (source_dir / 'checkpoints').mkdir(parents=True)
    config = {'output_dir': str(source_dir), 'extracted_dir': '/data/episodes',
              'db_path': None, 'table_name': 'datasets'}
    source = checkpoint(2, 10, config)
    prior_best = checkpoint(5, 8, config, signed=False)  # Compatible legacy v2 metadata.
    source_path = source_dir / 'checkpoints' / 'checkpoint_iter10.pt'
    torch.save(source, source_path)
    torch.save(prior_best, source_dir / 'checkpoints' / 'checkpoint_best.pt')
    output = tmp_path / 'new_output' if new_output else source_dir
    (output / 'checkpoints').mkdir(parents=True, exist_ok=True)
    signature = evaluation_signature(config, SPLITS, SCHEMA)
    candidates = load_resume_best_candidates(source, source_path, output, signature)
    assert len(candidates) == 1

    policy = torch.nn.Linear(1, 1, bias=False)
    policy.load_state_dict(source['policy_state_dict'])
    evaluated = []

    def evaluate(current):
        value = float(current.weight.item())
        evaluated.append(value)
        return {'mean_net_return': value}

    trainer = WorseningTrainer(policy, object(), episodes_per_group=1, num_groups=1,
                              observation_schema=SCHEMA, evaluation_callback=evaluate)
    trainer.restore_training_progress(source)
    trainer.extra_checkpoint_state = {'training_config': {**config, 'output_dir': str(output)},
                                      'date_splits': SPLITS, 'evaluation_signature': signature}
    metrics = trainer.train(1, start_iteration=10, max_timesteps=11, resume=True,
                            checkpoint_path=str(output / 'checkpoints' / 'checkpoint_iter{}.pt'),
                            resume_best_checkpoints=candidates)
    selected = torch.load(output / 'checkpoints' / 'checkpoint_best.pt', weights_only=True)
    assert evaluated == [2.0, 5.0, 1.0]
    assert selected['policy_state_dict']['weight'].item() == 5
    assert selected['extra_state']['best_validation_return'] == 5  # Not stored stale 999.
    assert selected['iteration'] == 8
    assert metrics == {'total_timesteps': 11, 'num_updates': 11}
    assert policy.weight.item() == 1  # Training continued from the requested source.


def test_fine_tuning_starts_an_independent_best(tmp_path):
    policy = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(2)
    trainer = WorseningTrainer(policy, object(), episodes_per_group=1, num_groups=1,
                              observation_schema=SCHEMA,
                              evaluation_callback=lambda current: {'mean_net_return': float(current.weight.item())})
    trainer.train(1, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'),
                  resume=False, resume_best_checkpoints=[checkpoint(99, 8, {})])
    selected = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert selected['policy_state_dict']['weight'].item() == 1
    assert selected['extra_state']['best_validation_return'] == 1


def test_best_revert_keeps_current_learning_rate_and_saves_it(tmp_path):
    source = checkpoint(2, 10, {})
    prior_best = checkpoint(5, 8, {})
    policy = torch.nn.Linear(1, 1, bias=False)
    policy.load_state_dict(source['policy_state_dict'])
    trainer = WorseningTrainer(
        policy, object(), episodes_per_group=1, num_groups=1, observation_schema=SCHEMA,
        evaluation_callback=lambda current: {'mean_net_return': float(current.weight.item())})
    trainer.restore_training_progress(source)
    trainer.extra_checkpoint_state = {'training_config': {'lr': 3e-4}}
    trainer.set_learning_rate(1e-5)
    trainer.train(1, start_iteration=10, max_timesteps=11, resume=True,
                  checkpoint_interval=1, revert_to_best_patience=1,
                  checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'),
                  resume_best_checkpoints=[prior_best])
    assert policy.weight.item() == 5  # The imported best really was loaded.
    assert (trainer.total_timesteps, trainer.num_updates) == (11, 11)
    assert trainer.learning_rate == trainer.optimizer.param_groups[0]['lr'] == 1e-5
    saved = torch.load(tmp_path / 'checkpoint_iter11.pt', weights_only=True)
    assert saved['config']['learning_rate'] == 1e-5
    assert saved['optimizer_state_dict']['param_groups'][0]['lr'] == 1e-5
    assert saved['extra_state']['training_config']['lr'] == 1e-5
    imported = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert imported['optimizer_state_dict']['param_groups'][0]['lr'] == 3e-4
    assert imported['config']['learning_rate'] == 3e-4
    assert imported['extra_state']['training_config']['lr'] == 3e-4
    assert trainer.extra_checkpoint_state['training_config']['lr'] == 1e-5


@pytest.mark.parametrize('difference', ['schema', 'architecture', 'dates', 'lineage_dates', 'fees', 'source_output', 'dataset'])
def test_unrelated_or_differently_evaluated_best_is_rejected(difference):
    config = {'output_dir': '/run/source', 'extracted_dir': '/data/source',
              'db_path': None, 'table_name': 'datasets'}
    source, candidate = checkpoint(2, 10, config), checkpoint(5, 8, config)
    if difference == 'schema':
        candidate['observation_schema'] = {'version': 'other'}
    elif difference == 'architecture':
        candidate['policy_state_dict']['weight'] = torch.zeros(2, 1)
    elif difference == 'dates':
        candidate['extra_state']['evaluation_signature']['date_splits']['validation'] = ['20260104']
    elif difference == 'lineage_dates':
        candidate['extra_state']['date_splits']['validation'] = ['20260104']
    elif difference == 'fees':
        candidate['extra_state']['evaluation_signature']['settings']['sell_tax_rate'] = .01
    elif difference == 'source_output':
        candidate['extra_state']['training_config']['output_dir'] = '/run/other'
    else:
        candidate['extra_state']['training_config']['extracted_dir'] = '/data/other'
    assert not compatible_resume_best(candidate, source, evaluation_signature(config, SPLITS, SCHEMA))
