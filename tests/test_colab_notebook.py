"""Exercise notebook configuration logic without a Colab account or CUDA device."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ai_trader.grpo.train_xlstm import TrainingConfig
from lib.observations import ACCOUNT_FIELDS, EXECUTION_FIELDS, ObservationBuilder


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'ai_trader/grpo/colab_train_xlstm.ipynb'
GIB = 1024 ** 3
NOTEBOOK_NAMES = ('colab_train_xlstm.ipynb', 'colab_train_xlstm_astral.ipynb',
                  'colab_train_xlstm_5splits_astral.ipynb', 'colab_train_xlstm_return_priority.ipynb')


def sources():
    return {cell['id']: ''.join(cell['source'])
            for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
            if cell['cell_type'] == 'code'}


def helpers():
    namespace = {'GiB': GIB, 'ObservationBuilder': ObservationBuilder}
    exec(compile(sources()['profile-functions'], str(NOTEBOOK), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_notebook_and_embedded_runner_are_valid_python(name, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and not cell['outputs']
            ast.parse(''.join(cell['source']), filename=cell['id'])
    assert len({cell['id'] for cell in notebook['cells']}) == len(notebook['cells'])
    # Compile the code passed to the training subprocess, not just its string literal.
    assignments = ast.parse(sources()['gpu-probe']).body
    setup = next(ast.literal_eval(node.value) for node in assignments
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'GPU_SETUP' for target in node.targets))
    train_assignment = next(node for node in ast.parse(sources()['train']).body
                            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                               and target.id == 'runner' for target in node.targets))
    scope = {'GPU_SETUP': setup}
    exec(compile(ast.Module(body=[train_assignment], type_ignores=[]), '<runner>', 'exec'), scope)
    result = subprocess.run([sys.executable, '-B', '-c', scope['runner'], '--help'],
                            cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                            capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--resume' in result.stdout and '--config' in result.stdout
    for option in ('--execution_observations', '--no-execution_observations',
                   '--decision_interval_seconds', '--episode_duration_seconds',
                   '--group_advantage_coef', '--training_seed', '--lambda_gae'):
        assert option in result.stdout


@pytest.mark.parametrize('vram,free,ram,cpus,batch,max_workers,episodes', [
    (40, 38, 50, 12, 16, 4, 16),
    (80, 75, 70, 12, 32, 8, 16),
    (80, 75, 12, 12, 32, 2, 8),
    (40, 10, 26, 2, 8, 1, 16),
    (40, 38, 6, 12, 16, 1, 8),
])
def test_profile_respects_host_ram_and_cpu(vram, free, ram, cpus, batch, max_workers, episodes):
    namespace = helpers()
    profile = namespace['a100_profile'](vram, free, ram, cpus)
    assert profile['batch_size'] == batch
    assert 1 <= profile['num_workers'] <= max_workers
    assert profile['episodes_per_group'] * profile['num_groups'] == episodes
    assert namespace['estimated_host_gib'](profile) <= .70 * ram
    assert profile['seq_len'] == 1024 and profile['rnn_hidden_dim'] == 512


@pytest.mark.parametrize('hardware', [(10, 9, 50, 4), (40, 2, 50, 4), (40, 38, 3, 4)])
def test_profile_rejects_insufficient_memory(hardware):
    with pytest.raises(ValueError):
        helpers()['a100_profile'](*hardware)


def default_config_namespace(**settings):
    scope = helpers()
    scope.update(Path=Path, json=json, TrainingConfig=TrainingConfig,
                 torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (38 * GIB, 40 * GIB))),
                 VRAM_GIB=40, CPU_COUNT=12, available_ram_gib=lambda: 50,
                 LOCAL_DATA_DIR=Path('/unused/data'), MANIFEST_SHA256='test',
                 manifest={'episodes': [{'length': 10000}]})
    # This cell checks paths and constructs settings; it writes no files or GPU tensors.
    tree = ast.parse(sources()['config'])
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in settings:
                node.value = ast.Constant(settings[name])
    exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), 'exec'), scope)
    return scope


def test_default_settings_are_supported_and_use_current_reward_and_execution():
    config = default_config_namespace()['CONFIG']
    assert not set(config) - set(TrainingConfig().__dict__)
    assert config['device'] == 'cuda' and config['load_policy'] is None
    assert config['use_raw_data'] and config['use_gae']
    assert config['batch_size'] == 16 and config['num_workers'] <= 4
    assert config['max_trades_per_episode'] is None
    assert config['max_stages'] == 1
    assert config['total_timesteps'] == 1_000_000 and config['gamma'] == 1.0
    assert config['evaluation_episodes'] == 64
    assert config['selection_require_liquidation'] is True
    assert config['execution_config']['require_order_book'] is True
    assert (config['cnn_channels'], config['rnn_hidden_dim'], config['hidden_dim']) == (256, 512, 512)
    for field in ('win_bonus', 'loss_penalty', 'buy_signal_bonus', 'no_trade_penalty'):
        assert config[field] == 0
    assert config['max_holding_seconds'] == 300
    assert config['execution_config']['order_latency_ms'] > 0
    assert config['validation_fraction'] == config['test_fraction'] == .2
    assert config['account_observations'] is False and config['liquidation_max_steps'] == 0
    assert config['execution_observations'] is False
    assert config['decision_interval_seconds'] == config['episode_duration_seconds'] == 0.
    assert config['execution_action_mask'] is False
    assert config['no_trade_patience'] == 0
    assert config['profitable_min_round_trips'] == 20
    assert config['profitable_min_traded_dates'] == 3
    assert config['group_advantage_coef'] == 1. and config['training_seed'] == 42
    assert config['evaluation_workers'] == 1


def checkpoint_and_base():
    base = default_config_namespace()['CONFIG']
    saved = copy.deepcopy(base)
    saved.update(seq_len=2048, rnn_hidden_dim=256, lr=.000123, num_workers=8,
                 batch_size=256, total_timesteps=6000, sell_tax_rate=.003,
                 max_holding_seconds=120, rolling_window_size=256)
    saved['execution_config']['slippage_bps'] = 7
    checkpoint = {
        'observation_schema': ObservationBuilder(['현재가', '등락률'], seq_len=2048,
            rolling_window_size=256, rolling_min_samples=64, max_holding_seconds=120).schema,
        'total_timesteps': 6000,
        'optimizer_state_dict': {}, 'num_updates': 3, 'iteration': 3,
        'extra_state': {'training_config': saved,
                        'date_splits': {'train': ['20260101'], 'validation': ['20260102'], 'test': ['20260103']}},
    }
    base['load_policy'] = '/unused/checkpoint.pt'
    return checkpoint, base


@pytest.mark.parametrize('mode', ['resume', 'finetune'])
def test_restore_preserves_training_schema_and_costs_but_adapts_resource_settings(mode):
    checkpoint, base = checkpoint_and_base()
    result = helpers()['restore_run_config'](base, checkpoint, mode)
    assert result['seq_len'] == 2048 and result['rnn_hidden_dim'] == 256
    assert result['max_holding_seconds'] == 120 and result['rolling_window_size'] == 256
    assert result['sell_tax_rate'] == .003 and result['execution_config']['slippage_bps'] == 7
    assert result['batch_size'] == 16 and result['num_workers'] == 4
    assert result['total_timesteps'] == base['total_timesteps']
    assert result['resume'] == (mode == 'resume')
    assert result['lr'] == (.000123 if mode == 'resume' else base['lr'])
    assert checkpoint['extra_state']['training_config']['batch_size'] == 256


def test_restore_rejects_old_checkpoint_and_exhausted_resume_budget():
    checkpoint, base = checkpoint_and_base()
    with pytest.raises(ValueError, match='observation_schema'):
        helpers()['restore_run_config'](base, {}, 'resume')
    base['total_timesteps'] = checkpoint['total_timesteps']
    with pytest.raises(ValueError, match='TOTAL_TIMESTEPS'):
        helpers()['restore_run_config'](base, checkpoint, 'resume')
    # Fine tuning starts a new budget; it need not exceed the old cumulative steps.
    assert not helpers()['restore_run_config'](base, checkpoint, 'finetune')['resume']


def test_legacy_five_stage_notebook_resume_restores_recorded_limit():
    checkpoint, base = checkpoint_and_base()
    checkpoint['observation_schema']['max_stages'] = 5
    checkpoint['extra_state']['training_config'].pop('max_stages')
    restored = helpers()['restore_run_config'](base, checkpoint, 'resume')
    assert base['max_stages'] == 1 and restored['max_stages'] == 5


@pytest.mark.parametrize('name', ['colab_train_xlstm_astral.ipynb', 'colab_train_xlstm_5splits_astral.ipynb'])
def test_astral_notebooks_share_return_priority_defaults(name, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    test_default_settings_are_supported_and_use_current_reward_and_execution()
    for source in sources().values():
        ast.parse(source)


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
@pytest.mark.parametrize('version', [2, 3, 4])
@pytest.mark.parametrize('mode', ['resume', 'finetune'])
def test_notebook_resume_uses_checkpoint_schema_and_tail_not_current_defaults(
        name, version, mode, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    checkpoint, base = checkpoint_and_base()
    saved = checkpoint['extra_state']['training_config']
    schema = ObservationBuilder(['현재가', '등락률'], seq_len=2048,
        rolling_window_size=256, rolling_min_samples=64, max_holding_seconds=120,
        account_observations=version >= 3, execution_observations=version == 4).schema
    checkpoint['observation_schema'] = schema
    saved['features'] = len(schema['feature_columns'])
    # Old configuration never recorded these fields; v3 restores its saved tail.
    saved.pop('account_observations', None)
    saved.pop('execution_observations', None)
    saved.pop('liquidation_max_steps', None)
    for key in ('decision_interval_seconds', 'episode_duration_seconds', 'group_advantage_coef', 'training_seed'):
        saved.pop(key, None)
    if version >= 3:
        saved['liquidation_max_steps'] = 37
    if version == 4:
        saved.update(decision_interval_seconds=1.25, episode_duration_seconds=120.,
                     group_advantage_coef=0., training_seed=1234)
    scope = helpers()
    restored = scope['restore_run_config'](base, checkpoint, mode)
    assert restored['account_observations'] is (version >= 3)
    assert restored['execution_observations'] is (version == 4)
    assert restored['liquidation_max_steps'] == (37 if version >= 3 else 0)
    assert restored['decision_interval_seconds'] == (1.25 if version == 4 else 0.)
    assert restored['episode_duration_seconds'] == (120. if version == 4 else 0.)
    assert restored['group_advantage_coef'] == (0. if version == 4 else 1.)
    assert restored['training_seed'] == (1234 if version == 4 else 42)
    assert restored['evaluation_workers'] == base['evaluation_workers']
    expected_dim = (len(schema['feature_columns']) + 15 + (len(ACCOUNT_FIELDS) if version >= 3 else 0)
                    + (len(EXECUTION_FIELDS) if version == 4 else 0))
    assert scope['model_obs_dim'](restored) == expected_dim
    constructor = next(node for node in ast.walk(ast.parse(sources()['gpu-probe']))
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == 'GRPOPolicyE2EXLSTM')
    dimension = next(keyword.value for keyword in constructor.keywords if keyword.arg == 'obs_dim')
    probe_scope = {**scope, 'config': restored, 'observation': SimpleNamespace(shape=(2048, expected_dim))}
    assert eval(compile(ast.Expression(dimension), '<gpu-probe-dimension>', 'eval'), probe_scope) == expected_dim
    assert any("(CONFIG['seq_len'], model_obs_dim(CONFIG))" in source for source in sources().values())


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_notebook_memory_estimate_counts_account_fields(name, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    scope = helpers()
    profile = scope['a100_profile'](40, 38, 50, 12)
    old = {**profile, 'account_observations': False, 'execution_observations': False}
    new = {**profile, 'account_observations': True, 'execution_observations': False}
    expected_extra = (3 * profile['episodes_per_group'] * profile['num_groups']
                      * profile['episode_steps'] * profile['seq_len'] * len(ACCOUNT_FIELDS) * 4 / GIB)
    assert scope['estimated_host_gib'](new) - scope['estimated_host_gib'](old) >= expected_extra - 1e-9


def test_return_priority_notebook_enables_new_observations_and_liquidation_tail(monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    config = default_config_namespace()['CONFIG']
    assert config['account_observations'] is True
    assert config['execution_observations'] is True
    assert helpers()['model_obs_dim'](config) == 63
    assert config['liquidation_max_steps'] == 300
    assert 1 < config['evaluation_workers'] <= config['num_workers']
    assert config['lambda_gae'] == .95
    assert config['decision_interval_seconds'] == 1.
    assert config['episode_duration_seconds'] == 300.
    assert config['execution_action_mask'] is True
    assert config['no_trade_patience'] == 3
    assert config['profitable_min_round_trips'] == 20
    assert config['profitable_min_traded_dates'] == 3
    assert config['group_advantage_coef'] == 1. and config['training_seed'] == 42


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_notebook_memory_estimate_counts_all_ten_execution_channels(name, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    scope = helpers()
    profile = scope['a100_profile'](40, 38, 50, 12)
    old = {**profile, 'account_observations': True, 'execution_observations': False, 'evaluation_workers': 8}
    new = {**old, 'execution_observations': True}
    expected_extra = (3 * profile['episodes_per_group'] * profile['num_groups']
                      * profile['episode_steps'] * profile['seq_len'] * len(EXECUTION_FIELDS) * 4 / GIB)
    assert scope['model_obs_dim'](new) - scope['model_obs_dim'](old) == 10
    assert scope['estimated_host_gib'](new) - scope['estimated_host_gib'](old) >= expected_extra


@pytest.mark.parametrize('experiment,expected', [
    ('cost_observations', {}),
    ('timed_decisions', {'decision_interval_seconds': 1., 'episode_duration_seconds': 300.}),
    ('gae_only', {'group_advantage_coef': 0.}),
    ('long_credit', {'lambda_gae': .99}),
    ('monte_carlo_credit', {'lambda_gae': 1.}),
    ('timed_gae_only', {'decision_interval_seconds': 1., 'episode_duration_seconds': 300.,
                        'group_advantage_coef': 0.}),
    ('timed_long_credit', {'decision_interval_seconds': 1., 'episode_duration_seconds': 300.,
                           'group_advantage_coef': 0., 'lambda_gae': .99}),
    ('timed_monte_carlo_credit', {'decision_interval_seconds': 1., 'episode_duration_seconds': 300.,
                                  'group_advantage_coef': 0., 'lambda_gae': 1.}),
])
def test_return_priority_experiments_change_only_the_declared_comparison(experiment, expected, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    baseline = default_config_namespace(EXPERIMENT='cost_observations')['CONFIG']
    config = default_config_namespace(EXPERIMENT=experiment)['CONFIG']
    assert config == {**baseline, **expected}


def test_return_priority_rejects_unknown_experiment(monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    with pytest.raises(ValueError, match='EXPERIMENT'):
        default_config_namespace(EXPERIMENT='unrecorded_variant')


def test_timed_notebook_memory_estimate_accounts_for_replaying_full_market_rows(monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    scope = helpers()
    config = default_config_namespace(EXPERIMENT='timed_decisions')['CONFIG']
    assert scope['estimated_host_gib'](config, episode_rows=100000) > scope['estimated_host_gib'](config)
