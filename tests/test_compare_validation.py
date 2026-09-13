"""Paired validation contract tests use tiny CPU checkpoints and mocked replay."""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from ai_trader.grpo import compare_validation as comparison
from ai_trader.grpo.runtime_precision import precision_metadata, restore_precision, set_tf32
from lib.observations import ObservationBuilder


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # No test in this module creates a CUDA context or executes a GPU forward.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    original_precision = precision_metadata()
    data = tmp_path / 'data'
    data.mkdir()
    schema = ObservationBuilder(['price'], seq_len=4, rolling_window_size=4,
                                rolling_min_samples=2).schema
    config = {'device': 'cuda', 'lr': 3e-5, 'episode_steps': 300,
              'evaluation_episodes': 64, 'evaluation_seed': 42,
              'transaction_cost_rate': .00015, 'buy_tax_rate': 0., 'sell_tax_rate': .0018,
              'execution_config': {'order_latency_ms': 100, 'slippage_bps': 2},
              'initial_cash': 1_000_000., 'execution_action_mask': True}
    splits = {'train': ['20240102'], 'validation': ['20250102', '20250103'], 'test': ['20260102']}
    source = {'iteration': 19, 'total_timesteps': 52484, 'policy_state_dict': {'weight': torch.tensor([0.])},
              'observation_schema': schema, 'extra_state': {'training_config': config, 'date_splits': splits}}
    source_path = tmp_path / 'source.pt'
    torch.save(source, source_path)
    source_hash = comparison._hash(source_path)
    report = {'format': 'grpo_learning_rate_comparison', 'format_version': 1,
              'source_checkpoint': str(source_path), 'source_checkpoint_sha256': source_hash,
              'model_exported': True, 'variants': []}
    for index, rate in enumerate((3e-5, 1e-5, 3e-6)):
        candidate = deepcopy(source)
        candidate['policy_state_dict']['weight'] = torch.tensor([float(index + 1)])
        candidate['extra_state']['training_config']['lr'] = rate
        candidate['extra_state']['export_provenance'] = {
            'source_checkpoint_sha256': source_hash, 'learning_rate': rate}
        candidate_path = tmp_path / f'candidate_{index}.pt'
        torch.save(candidate, candidate_path)
        report['variants'].append({'learning_rate': rate, 'status': 'ok',
                                   'candidate_checkpoint': str(candidate_path),
                                   'candidate_sha256': comparison._hash(candidate_path)})
    report_path = tmp_path / 'lr_comparison.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    state = SimpleNamespace(source=source_path, report_path=report_path, report=report,
                            output=tmp_path / 'validation_comparison.json', data=data,
                            calls=[], all_no_trade=False, mutate=None)

    def replay(path, *, split, episodes, seed, device, extracted_dir):
        state.calls.append((str(path), split, episodes, seed, device, str(extracted_dir)))
        assert precision_metadata()['matmul_fp32_precision'] == 'ieee'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        number = int(saved['policy_state_dict']['weight'].item())
        value = 0. if state.all_no_trade else [0., .1, -.2, .3][number]
        traded = value != 0
        returns = [value] * episodes
        probabilities = {'hold': .2 if traded else .7, 'buy': .7 if traded else .2, 'sell': .1}
        details = [{'episode_index': index, 'seed': seed + index,
                    'episode_key': {'stock_code': f'{index:06d}', 'date': '20250102', 'start_index': index * 10},
                    'net_return': value} for index in range(episodes)]
        result = {'checkpoint': str(path), 'split': split, 'deterministic': True, 'device': device,
                  'dates': saved['extra_state']['date_splits']['validation'],
                  'date_splits': saved['extra_state']['date_splits'], 'observation_schema': saved['observation_schema'],
                  'saved_training_config': saved['extra_state']['training_config'],
                  'execution_config': config['execution_config'],
                  'costs': {key: config[key] for key in ('transaction_cost_rate', 'buy_tax_rate', 'sell_tax_rate')},
                  'metrics': {'num_episodes': episodes, 'seed': seed, 'mean_net_return': value,
                              'episode_net_returns': returns, 'total_fees': 2. * episodes if traded else 0.,
                              'mean_realized_net_pnl': value * 10000, 'round_trip_count': episodes if traded else 0,
                              'mean_num_trades': 1 if traded else 0, 'max_drawdown': .5 if value < 0 else 0,
                              'no_trade_episode_fraction': 0. if traded else 1.,
                              'incomplete_liquidation_episodes': 0, 'max_open_quantity': 0,
                              'profitable_with_trades': value > 0, 'evaluation_outcome': 'profitable' if value > 0 else 'no_trade',
                              'diagnostics': {'episodes': details, 'action_counts': {'hold': 8, 'buy': int(traded)},
                                              'mean_action_probabilities': probabilities,
                                              'submitted_orders': 2 * episodes if traded else 0,
                                              'filled_quantity': 2 * episodes if traded else 0}}}
        # Constructors/replay may use RNG. The caller must restore its starting state.
        random.random()
        torch.rand(1)
        if state.mutate:
            state.mutate(result, number)
        return result

    monkeypatch.setattr(comparison, 'run_diagnostics', replay)
    yield state
    restore_precision(original_precision)


def _run(experiment, **kwargs):
    return comparison.run_comparison(experiment.source, experiment.report_path,
                                     output=experiment.output, extracted_dir=experiment.data,
                                     episodes=8, seed=42, device='cpu', **kwargs)


def _rewrite_report(experiment):
    experiment.report_path.write_text(json.dumps(experiment.report), encoding='utf-8')


def test_source_and_three_candidates_use_paired_paths_and_report_return_deltas(experiment):
    original_bytes = experiment.source.read_bytes()
    result = _run(experiment)
    assert len(experiment.calls) == 4
    assert all(call[1:5] == ('validation', 8, 42, 'cpu') for call in experiment.calls)
    assert result['status'] == 'completed' and not result['holdout_test_evaluated']
    assert result['precision_restored'] is result['rng_restored'] is True
    assert result['summary']['candidate_count'] == 3
    assert result['summary']['improved_mean_return_candidate_count'] == 2
    assert result['summary']['small_sample'] is True
    assert [item['paired']['mean_net_return_delta_pp'] for item in result['candidates']] == pytest.approx([.1, -.2, .3])
    first = result['candidates'][0]
    assert first['metrics']['total_fees'] == 16
    assert first['metrics']['round_trip_count'] == 8
    assert first['metrics']['mean_action_probabilities']['buy'] == .7
    assert first['paired']['improved_episodes'] == 8
    for item in [result['baseline'], *result['candidates']]:
        assert Path(item['evaluation_report']).is_file()
    assert experiment.source.read_bytes() == original_bytes
    assert json.loads(experiment.output.read_text(encoding='utf-8'))['status'] == 'completed'


def test_all_no_trade_is_a_completed_comparison_without_improvement_claim(experiment):
    experiment.all_no_trade = True
    result = _run(experiment)
    assert result['status'] == 'completed'
    assert result['summary']['all_no_trade'] is True
    assert result['summary']['improved_mean_return_candidate_count'] == 0
    assert result['summary']['profitable_with_trades_candidate_count'] == 0
    assert all(item['paired']['both_no_trade'] for item in result['candidates'])


@pytest.mark.parametrize('change', ['hash', 'source_hash', 'provenance', 'conditions', 'no_candidates', 'failed_candidate'])
def test_invalid_candidate_inputs_fail_before_any_evaluation(experiment, change):
    variant = experiment.report['variants'][0]
    if change in ('hash', 'source_hash'):
        target, key = ((variant, 'candidate_sha256') if change == 'hash' else
                       (experiment.report, 'source_checkpoint_sha256'))
        target[key] = '0' * 64
    elif change in ('provenance', 'conditions'):
        path = Path(variant['candidate_checkpoint'])
        candidate = torch.load(path, weights_only=True)
        if change == 'provenance':
            candidate['extra_state']['export_provenance']['source_checkpoint_sha256'] = '0' * 64
        else:
            candidate['extra_state']['training_config']['sell_tax_rate'] = .1
        torch.save(candidate, path)
        variant['candidate_sha256'] = comparison._hash(path)
    elif change == 'no_candidates':
        experiment.report['variants'] = []
    else:
        variant['status'] = 'error'
    _rewrite_report(experiment)
    with pytest.raises(ValueError):
        _run(experiment)
    assert experiment.calls == []
    assert not experiment.output.exists()


@pytest.mark.parametrize('change', ['path', 'seed', 'cost', 'schema', 'split', 'return', 'missing_identity'])
def test_replay_mismatch_is_rejected_and_partial_evaluations_preserved(experiment, change):
    def mutate(result, number):
        if number != 1:
            return
        if change == 'path':
            result['metrics']['diagnostics']['episodes'][0]['episode_key']['start_index'] += 1
        elif change == 'seed':
            result['metrics']['diagnostics']['episodes'][0]['seed'] += 1
        elif change == 'cost':
            result['costs']['sell_tax_rate'] = .2
        elif change == 'schema':
            result['observation_schema']['seq_len'] = 100
        elif change == 'split':
            result['split'] = 'test'
        elif change == 'return':
            result['metrics']['mean_net_return'] = 900
        else:
            result['metrics']['diagnostics']['episodes'][0]['episode_key'] = {}
    experiment.mutate = mutate
    with pytest.raises(ValueError):
        _run(experiment)
    assert len(experiment.calls) == 2
    failed = json.loads(experiment.output.read_text(encoding='utf-8'))
    assert failed['status'] == 'failed' and not failed['holdout_test_evaluated']
    assert failed['candidates'] == []
    assert Path(failed['baseline']['evaluation_report']).exists()


@pytest.mark.parametrize('failure', [False, True])
def test_precision_and_rng_are_restored_after_success_or_failure(experiment, failure):
    set_tf32(True)
    before_precision = precision_metadata()
    before_python = random.getstate()
    before_torch = torch.get_rng_state().clone()
    if failure:
        def fail(result, number):
            if number == 1:
                raise RuntimeError('Replay failed')
        experiment.mutate = fail
        with pytest.raises(RuntimeError, match='Replay failed'):
            _run(experiment)
    else:
        _run(experiment)
    assert precision_metadata() == before_precision
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)


@pytest.mark.parametrize('existing', ['summary', 'directory'])
def test_existing_outputs_are_preserved_before_replay(experiment, existing):
    if existing == 'summary':
        experiment.output.write_text('keep', encoding='utf-8')
    else:
        experiment.output.with_name(experiment.output.stem + '_evaluations').mkdir()
    with pytest.raises(FileExistsError):
        _run(experiment)
    assert experiment.calls == []
    if existing == 'summary':
        assert experiment.output.read_text(encoding='utf-8') == 'keep'


def test_cli_argument_contract_uses_validation_only(experiment):
    assert comparison.main(['--checkpoint', str(experiment.source), '--comparison-report', str(experiment.report_path),
                            '--output', str(experiment.output), '--extracted-dir', str(experiment.data),
                            '--episodes', '8', '--seed', '42', '--device', 'cpu']) == 0
    assert len(experiment.calls) == 4 and all(call[1] == 'validation' for call in experiment.calls)
