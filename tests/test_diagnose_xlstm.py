import json
from types import SimpleNamespace

import pytest
import torch

from ai_trader.grpo import diagnose_xlstm as diagnostic
from lib.observations import ObservationBuilder


@pytest.fixture
def replay(tmp_path, monkeypatch):
    data = tmp_path / 'episodes'
    data.mkdir()
    schema = ObservationBuilder(['현재가', '등락률', '누적거래대금'], seq_len=4,
                                rolling_window_size=4, rolling_min_samples=2).schema
    saved_config = dict(extracted_dir=str(data), db_path=None, evaluation_episodes=8,
                        evaluation_seed=42, device='cuda', episode_steps=300,
                        transaction_cost_rate=.00015, buy_tax_rate=0., sell_tax_rate=.0018,
                        execution_config={'order_latency_ms': 100, 'slippage_bps': 2})
    splits = {'train': ['20240902'], 'validation': ['20250103', '20250104'],
              'test': ['20250401']}
    checkpoint = dict(iteration=5, total_timesteps=24000,
                      policy_state_dict={'selected_weight': torch.tensor([5.])},
                      observation_schema=schema,
                      extra_state={'training_config': saved_config, 'date_splits': splits})
    path = tmp_path / 'checkpoint_best.pt'
    torch.save(checkpoint, path)
    seen = {}

    class Environment:
        observation_schema = schema

        def close(self):
            seen['closed'] = True

    class Policy:
        def load_state_dict(self, weights, strict):
            seen['loaded_weight'] = weights['selected_weight'].item()
            seen['strict'] = strict

    def environment(config, device, allowed_dates):
        seen.update(config=config, device=device, allowed_dates=allowed_dates)
        env = Environment()
        env.valid_keys = [(f'{day}.npz', '000001', day, 100) for day in allowed_dates] if config.extracted_dir else [
            ('000001', day, 100) for day in allowed_dates]
        seen['environment'] = env
        return env

    def policy(config, env, device):
        seen['policy_created'] = True
        return Policy()

    def evaluate(policy, env, **kwargs):
        seen['evaluation'] = kwargs
        return {'mean_num_trades': 0, 'diagnostics': {'action_counts': {'hold': 2400}}}

    monkeypatch.setattr(diagnostic, '_training_factories', lambda: (SimpleNamespace, environment, policy))
    monkeypatch.setattr(diagnostic, 'evaluate_policy', evaluate)
    return SimpleNamespace(path=path, checkpoint=checkpoint, seen=seen, directory=data,
                           tmp_path=tmp_path, schema=schema, environment_factory=environment)


def test_cli_uses_exact_best_weights_saved_config_and_validation_dates(replay):
    # A neighboring newer model must never replace the explicitly selected best.
    torch.save({'policy_state_dict': {'selected_weight': torch.tensor([21.])}},
               replay.tmp_path / 'checkpoint_iter21.pt')
    original = replay.path.read_bytes()
    output = replay.tmp_path / 'validation_diagnostics.json'
    assert diagnostic.main(['--checkpoint', str(replay.path), '--output', str(output),
                            '--device', 'cpu']) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert replay.seen['loaded_weight'] == 5
    assert replay.seen['strict'] is True
    assert replay.seen['allowed_dates'] == ['20250103', '20250104']
    assert replay.seen['evaluation'] == dict(num_episodes=8, seed=42, device='cpu', collect_diagnostics=True)
    assert replay.seen['config'].episode_steps == 300
    assert replay.seen['config'].execution_config == replay.checkpoint['extra_state']['training_config']['execution_config']
    assert report['checkpoint_iteration'] == 5
    assert report['checkpoint_total_timesteps'] == 24000
    assert report['split'] == 'validation'
    assert report['metrics']['diagnostics']['action_counts']['hold'] == 2400
    assert report['costs']['sell_tax_rate'] == .0018
    assert replay.seen['closed']
    assert replay.path.read_bytes() == original
    assert not (replay.tmp_path / 'evaluation_report.json').exists()


@pytest.mark.parametrize('missing', ['training_config', 'date_splits', 'observation_schema'])
def test_missing_provenance_fails_before_creating_environment(replay, missing):
    target = replay.checkpoint if missing == 'observation_schema' else replay.checkpoint['extra_state']
    target.pop(missing)
    torch.save(replay.checkpoint, replay.path)
    with pytest.raises(ValueError, match=missing):
        diagnostic.run_diagnostics(replay.path, device='cpu')
    assert 'environment' not in replay.seen


@pytest.mark.parametrize('kind', ['extracted_dir', 'db_path'])
def test_moved_data_override_preserves_costs_split_and_records_changes(replay, kind):
    config = replay.checkpoint['extra_state']['training_config']
    config['extracted_dir'] = '/content/drive/MyDrive/old_episodes'
    torch.save(replay.checkpoint, replay.path)
    moved = replay.tmp_path / ('moved_episodes' if kind == 'extracted_dir' else 'moved.duckdb')
    moved.mkdir() if kind == 'extracted_dir' else moved.touch()
    report = diagnostic.run_diagnostics(replay.path, split='test', episodes=2, seed=7,
                                        device='cpu', **{kind: str(moved)})
    assert report['dates'] == ['20250401']
    assert report['overrides'][kind] == str(moved.resolve())
    other = 'db_path' if kind == 'extracted_dir' else 'extracted_dir'
    assert getattr(replay.seen['config'], other) is None
    assert report['overrides'][other] is None
    assert report['execution_config'] == config['execution_config']
    assert report['costs']['transaction_cost_rate'] == config['transaction_cost_rate']
    assert report['saved_training_config']['extracted_dir'] == '/content/drive/MyDrive/old_episodes'
    assert replay.seen['evaluation']['num_episodes'] == 2
    assert replay.seen['evaluation']['seed'] == 7


def test_existing_output_refused_before_checkpoint_load(replay, monkeypatch):
    output = replay.tmp_path / 'diagnostics.json'
    output.write_text('keep this report', encoding='utf-8')
    monkeypatch.setattr(diagnostic.torch, 'load', lambda *args, **kwargs: pytest.fail('must not load'))
    with pytest.raises(FileExistsError, match='already exists'):
        diagnostic.main(['--checkpoint', str(replay.path), '--output', str(output)])
    assert output.read_text(encoding='utf-8') == 'keep this report'


def test_original_evaluation_report_name_is_never_written(replay):
    with pytest.raises(ValueError, match='separate diagnostic filename'):
        diagnostic.main(['--checkpoint', str(replay.path),
                         '--output', str(replay.tmp_path / 'evaluation_report.json')])
    assert not replay.seen


def test_output_created_by_other_process_during_replay_is_not_overwritten(replay, monkeypatch):
    output = replay.tmp_path / 'diagnostics.json'

    def concurrent_writer(*args, **kwargs):
        output.write_text('concurrent report', encoding='utf-8')
        return {'metrics': {}}

    monkeypatch.setattr(diagnostic, 'run_diagnostics', concurrent_writer)
    with pytest.raises(FileExistsError):
        diagnostic.main(['--checkpoint', str(replay.path), '--output', str(output)])
    assert output.read_text(encoding='utf-8') == 'concurrent report'


def test_schema_mismatch_fails_and_closes_environment(replay, monkeypatch):
    def incompatible(*args, **kwargs):
        env = replay.environment_factory(*args, **kwargs)
        env.observation_schema = dict(replay.schema, seq_len=999)
        return env

    monkeypatch.setattr(diagnostic, '_training_factories', lambda: (SimpleNamespace, incompatible, None))
    with pytest.raises(ValueError, match='Incompatible observation_schema'):
        diagnostic.run_diagnostics(replay.path, device='cpu')
    assert replay.seen['closed']
    assert 'policy_created' not in replay.seen


def test_missing_dates_in_moved_data_cannot_silently_shrink_split(replay, monkeypatch):
    def partial_data(*args, **kwargs):
        env = replay.environment_factory(*args, **kwargs)
        env.valid_keys = env.valid_keys[:1]
        return env

    monkeypatch.setattr(diagnostic, '_training_factories', lambda: (SimpleNamespace, partial_data, None))
    with pytest.raises(ValueError, match='Environment dates differ'):
        diagnostic.run_diagnostics(replay.path, device='cpu')
    assert replay.seen['closed']
