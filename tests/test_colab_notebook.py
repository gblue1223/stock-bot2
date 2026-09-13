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
NOTEBOOK = ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb'
GIB = 1024 ** 3
NOTEBOOK_NAMES = ('colab_train_xlstm_return_priority.ipynb',)


def sources():
    return {cell['id']: ''.join(cell['source'])
            for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
            if cell['cell_type'] == 'code'}


def helpers():
    namespace = {'GiB': GIB, 'ObservationBuilder': ObservationBuilder}
    exec(compile(sources()['profile-functions'], '<notebook-profile-functions>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('name', (*NOTEBOOK_NAMES, 'colab_train_xlstm_entry_pattern_2048.ipynb'))
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
])
def test_profile_respects_host_ram_and_cpu(vram, free, ram, cpus, batch, max_workers, episodes):
    namespace = helpers()
    profile = namespace['a100_profile'](vram, free, ram, cpus)
    assert profile['batch_size'] == batch
    assert 1 <= profile['num_workers'] <= max_workers
    assert profile['episodes_per_group'] * profile['num_groups'] == episodes
    assert namespace['estimated_host_gib'](profile) <= .70 * ram
    assert profile['seq_len'] == 2048 and profile['rnn_hidden_dim'] == 512


@pytest.mark.parametrize('hardware', [(10, 9, 50, 4), (40, 2, 50, 4), (40, 38, 3, 4),
                                      (40, 38, 6, 12)])
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
    scope = default_config_namespace()
    config = scope['CONFIG']
    assert scope['ENABLE_TF32'] is True
    assert scope['RESUME_LR'] == 0.0
    assert config['resume_lr'] is None and config['capture_update_bundle'] is None
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
    assert config['account_observations'] is True and config['liquidation_max_steps'] == 300
    assert config['execution_observations'] is True
    assert config['decision_interval_seconds'] == 1.
    assert config['episode_duration_seconds'] == 300.
    assert config['execution_action_mask'] is True
    assert config['no_trade_patience'] == 3
    assert config['no_trade_max_validations'] == 4
    assert config['policy_update_checks'] is True
    assert config['profitable_min_round_trips'] == 20
    assert config['profitable_min_traded_dates'] == 3
    assert config['group_advantage_coef'] == 0. and config['training_seed'] == 42
    assert config['evaluation_workers'] == config['num_workers']


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


@pytest.mark.parametrize('requested_lr', [None, 1e-5])
def test_notebook_resume_lr_uses_current_request_not_saved_request(requested_lr):
    checkpoint, base = checkpoint_and_base()
    checkpoint['extra_state']['training_config'].update(resume_lr=3e-6, capture_update_bundle='previous.pt')
    base['resume_lr'] = requested_lr
    restored = helpers()['restore_run_config'](base, checkpoint, 'resume')
    assert restored['resume_lr'] == requested_lr
    assert restored['lr'] == (requested_lr if requested_lr is not None else .000123)
    assert restored['capture_update_bundle'] is None


@pytest.mark.parametrize('value', [-1., True, '1e-5', float('nan'), float('inf'), 1e-5])
def test_notebook_rejects_invalid_or_nonresume_lr(value):
    with pytest.raises(ValueError, match='RESUME_LR'):
        default_config_namespace(RESUME_LR=value)


def test_restore_rejects_old_checkpoint_and_exhausted_resume_budget():
    checkpoint, base = checkpoint_and_base()
    with pytest.raises(ValueError, match='observation_schema'):
        helpers()['restore_run_config'](base, {}, 'resume')
    base['total_timesteps'] = checkpoint['total_timesteps']
    with pytest.raises(ValueError, match='TOTAL_TIMESTEPS'):
        helpers()['restore_run_config'](base, checkpoint, 'resume')
    # Fine tuning starts a new budget; it need not exceed the old cumulative steps.
    assert not helpers()['restore_run_config'](base, checkpoint, 'finetune')['resume']


def test_checkpoint_resume_restores_recorded_stage_limit():
    checkpoint, base = checkpoint_and_base()
    checkpoint['observation_schema']['max_stages'] = 5
    checkpoint['extra_state']['training_config'].pop('max_stages')
    restored = helpers()['restore_run_config'](base, checkpoint, 'resume')
    assert base['max_stages'] == 1 and restored['max_stages'] == 5


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
    assert helpers()['model_obs_dim'](config) == 65
    assert config['liquidation_max_steps'] == 300
    assert 1 < config['evaluation_workers'] <= config['num_workers']
    assert config['lambda_gae'] == .95
    assert config['decision_interval_seconds'] == 1.
    assert config['episode_duration_seconds'] == 300.
    assert config['execution_action_mask'] is True
    assert config['no_trade_patience'] == 3
    assert config['no_trade_max_validations'] == 4
    assert config['policy_update_checks'] is True
    assert config['rollout_logprob_tolerance'] == 1e-3
    assert config['kl_probe_samples'] == 32
    assert config['profitable_min_round_trips'] == 20
    assert config['profitable_min_traded_dates'] == 3
    assert config['group_advantage_coef'] == 0. and config['training_seed'] == 42


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
    with pytest.raises(ValueError, match='EXPERIMENT') as error:
        default_config_namespace(EXPERIMENT='unrecorded_variant')
    assert "'unrecorded_variant'" in str(error.value)
    assert 'timed_gae_only' in str(error.value)


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
@pytest.mark.parametrize('mode', ['resume', 'finetune'])
def test_missing_checkpoint_update_checks_remain_disabled(name, mode, monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo' / name)
    checkpoint, base = checkpoint_and_base()
    for key in ('policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples',
                'no_trade_max_validations'):
        checkpoint['extra_state']['training_config'].pop(key, None)
    restored = helpers()['restore_run_config'](base, checkpoint, mode)
    assert restored['policy_update_checks'] is False
    assert restored['rollout_logprob_tolerance'] == 1e-3
    assert restored['kl_probe_samples'] == 32
    assert restored['no_trade_max_validations'] == 0


def likelihood_notebook_helpers(monkeypatch, **extra):
    import gc
    import numpy as np
    import torch
    from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    tree = ast.parse(sources()['gpu-probe'])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    scope = dict(np=np, torch=torch, json=json, Path=Path, gc=gc,
                 TrainingConfig=TrainingConfig, ObservationBuilder=ObservationBuilder,
                 GRPOPolicyE2EXLSTM=GRPOPolicyE2EXLSTM, **extra)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook-likelihood>', 'exec'), scope)
    return scope


def test_likelihood_probe_collects_distinct_environment_states_and_valid_masks(monkeypatch):
    import numpy as np

    class Environment:
        closed = False
        cache_cleared = False
        calls = 0

        def reset(self, seed):
            self.index = seed
            return np.full((4, 3), seed, dtype=np.float32), {}

        def action_masks(self):
            return np.array([True, self.index % 2 == 0, self.index % 2 != 0])

        def step(self, action):
            assert self.action_masks()[action]
            self.calls += 1
            self.index += 1
            return np.full((4, 3), self.index, dtype=np.float32), 0., False, False, {}

        def close(self):
            self.closed = True

        @classmethod
        def clear_episode_cache(cls):
            cls.cache_cleared = True

    environment = Environment()
    scope = likelihood_notebook_helpers(monkeypatch, create_environment=lambda *args: environment)
    states, masks = scope['collect_likelihood_observations']({'num_workers': 3}, ['20260101'], 8, 42)
    assert states.shape == (8, 4, 3) and masks.shape == (8, 3)
    assert len(np.unique(states[:, 0, 0])) > 1
    assert masks[:, 0].all() and masks.dtype == bool
    assert environment.calls == 8 and environment.closed and environment.cache_cleared


@pytest.mark.parametrize('initial_training', [False, True])
@pytest.mark.parametrize('mismatch', [False, True])
def test_likelihood_probe_uses_rollout_and_update_batches_with_grad_path(
        monkeypatch, initial_training, mismatch):
    import numpy as np
    import torch

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 3)
            self.calls = []

        def distribution(self, states, masks):
            logits = self.head(states[:, -1])
            if mismatch and self.training:
                logits = logits + torch.tensor([2., -2., 0.])
            return torch.distributions.Categorical(logits=logits.masked_fill(~masks, -torch.inf))

        def get_action_with_value(self, states, deterministic, action_masks):
            self.calls.append(('rollout', len(states), self.training, torch.is_grad_enabled()))
            dist = self.distribution(states, action_masks)
            action = dist.sample()
            return action, dist.log_prob(action), states[:, 0, 0]

        def evaluate_actions_with_distribution(self, states, actions, action_masks):
            self.calls.append(('update', len(states), self.training, torch.is_grad_enabled()))
            dist = self.distribution(states, action_masks)
            return dist.log_prob(actions), dist.entropy(), states[:, 0, 0], dist.logits

    torch.manual_seed(42)
    policy = Policy().train(initial_training)
    states = np.random.default_rng(42).normal(size=(8, 3, 4)).astype(np.float32)
    masks = np.tile([True, True, False], (8, 1))
    config = dict(num_workers=3, batch_size=5, execution_action_mask=True, rollout_logprob_tolerance=1e-3)
    scope = likelihood_notebook_helpers(monkeypatch)
    if mismatch:
        with pytest.raises(ValueError, match='likelihood mismatch'):
            scope['check_policy_likelihood_batches'](policy, states, masks, config, device='cpu')
    else:
        result = scope['check_policy_likelihood_batches'](policy, states, masks, config, device='cpu')
        assert result['num_samples'] == 8 and result['max_abs_error'] < 1e-3
        assert 'reference_log_probs' not in result
        assert [call[1] for call in policy.calls if call[0] == 'update'] == [5, 3]
    assert [call[1] for call in policy.calls if call[0] == 'rollout'] == [3, 3, 2]
    assert all(not training and not grad for kind, _, training, grad in policy.calls if kind == 'rollout')
    assert all(training and grad for kind, _, training, grad in policy.calls if kind == 'update')
    assert policy.training is initial_training


def test_diagnostic_checkpoint_is_separate_from_training_weights(monkeypatch, tmp_path):
    import numpy as np
    import torch
    from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

    schema = ObservationBuilder(['현재가', '등락률'], seq_len=8).schema
    config = dict(cnn_channels=4, rnn_hidden_dim=4, hidden_dim=8, max_stages=1,
                  checkpoint_segments=2, execution_action_mask=True, num_workers=2, batch_size=4,
                  rollout_logprob_tolerance=1e-3, load_policy='unused-training-policy.pt')
    policy = GRPOPolicyE2EXLSTM(obs_dim=17, cnn_channels=4, rnn_hidden_dim=4,
        fc_hidden_dim=8, max_stages=1, checkpoint_segments=2, execution_action_mask=True)
    checkpoint_path = tmp_path / 'diagnostic.pt'
    torch.save(dict(policy_state_dict=policy.state_dict(), observation_schema=schema,
                    extra_state={'training_config': config}), checkpoint_path)
    scope = likelihood_notebook_helpers(monkeypatch, OBSERVATION_SCHEMA=schema)
    states = np.random.default_rng(7).normal(size=(4, 8, 17)).astype(np.float32)
    masks = np.tile([True, True, False], (4, 1))
    result = scope['run_likelihood_preflight'](config, states, masks, str(checkpoint_path), device='cpu')
    assert result['source'] == str(checkpoint_path) and result['num_samples'] == 4
    assert config['load_policy'] == 'unused-training-policy.pt'
    fresh = scope['run_likelihood_preflight'](config, states, masks, device='cpu')
    assert fresh['source'] == 'fresh_nonzero_policy_head' and fresh['policy_head_nonzero']


def test_likelihood_fingerprint_changes_with_checks_checkpoint_or_tf32(monkeypatch):
    scope = likelihood_notebook_helpers(monkeypatch)
    fingerprint = scope['likelihood_probe_fingerprint']
    original = fingerprint({'policy_update_checks': True}, '', True)
    assert original != fingerprint({'policy_update_checks': False}, '', True)
    assert original != fingerprint({'policy_update_checks': True}, 'checkpoint_iter18.pt', True)
    assert original != fingerprint({'policy_update_checks': True}, '', False)


def test_likelihood_fingerprint_detects_runtime_precision_changes(monkeypatch):
    from ai_trader.grpo import runtime_precision

    actual = {'matmul_fp32_precision': 'tf32'}
    monkeypatch.setattr(runtime_precision, 'precision_metadata', lambda: dict(actual))
    fingerprint = likelihood_notebook_helpers(monkeypatch)['likelihood_probe_fingerprint']
    original = fingerprint({'policy_update_checks': True}, '', True)
    actual['matmul_fp32_precision'] = 'ieee'
    assert original != fingerprint({'policy_update_checks': True}, '', True)


@pytest.mark.parametrize('selected_tf32', [False, True])
def test_resume_keeps_current_tf32_checkbox_instead_of_saved_value(tmp_path, selected_tf32, capsys):
    checkpoint_path = tmp_path / 'prior_run/checkpoints/checkpoint_iter18.pt'
    checkpoint_path.parent.mkdir(parents=True)
    (checkpoint_path.parent.parent / 'colab_run.json').write_text(json.dumps({
        'manifest_sha256': 'same-data', 'enable_tf32': not selected_tf32, 'seed': 321}), encoding='utf-8')
    resume_block = next(node for node in ast.walk(ast.parse(sources()['config']))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == 'MODE'
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == 'resume')
    scope = dict(MODE='resume', Path=Path, json=json, LOAD_POLICY=str(checkpoint_path),
                 ENABLE_TF32=selected_tf32, MANIFEST_SHA256='same-data', SEED=42)
    exec(compile(ast.Module(body=[resume_block], type_ignores=[]), '<notebook-resume-precision>', 'exec'), scope)
    assert scope['ENABLE_TF32'] is selected_tf32
    assert scope['SEED'] == 321
    output = capsys.readouterr().out
    assert repr(selected_tf32) in output and repr(not selected_tf32) in output


@pytest.mark.parametrize('specified_output', [False, True])
def test_failure_replay_cell_runs_without_training_or_gpu_probe_state(tmp_path, specified_output):
    bundle_path = tmp_path / 'failure bundle.pt'
    bundle_path.write_bytes(b'fixture')
    expected_output = tmp_path / ('manual.json' if specified_output else 'failure bundle_tf32_replay_123.json')
    calls = []

    def replay(command, **kwargs):
        calls.append(command)
        assert command[command.index('--bundle') + 1] == str(bundle_path)
        assert command[command.index('--output') + 1] == str(expected_output)
        expected_output.write_text(json.dumps({'diagnostic_only': True}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='replay complete')

    tree = ast.parse(sources()['likelihood-replay'])
    replacements = {'LIKELIHOOD_FAILURE_BUNDLE': str(bundle_path),
                    'DIAGNOSTIC_OUTPUT': str(expected_output) if specified_output else ''}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in replacements:
                node.value = ast.Constant(replacements[node.targets[0].id])
    scope = dict(Path=Path, json=json, sys=sys, os=os, REPO_PATH=ROOT,
                 time=SimpleNamespace(time_ns=lambda: 123), subprocess=SimpleNamespace(
                     run=replay, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))
    exec(compile(ast.fix_missing_locations(tree), '<independent-failure-replay>', 'exec'), scope)
    assert len(calls) == 1
    assert 'ai_trader.grpo.diagnose_likelihood' in calls[0]
    assert '--rollout_logprob_tolerance' not in calls[0]


@pytest.mark.parametrize('enabled', [False, True])
def test_training_subprocess_explicitly_passes_checked_settings_for_resume(monkeypatch, enabled):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    config = default_config_namespace()['CONFIG']
    config['policy_update_checks'] = enabled
    tree = ast.parse(sources()['train'])
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'command' for target in node.targets))
    scope = dict(sys=sys, runner='diagnostic-runner', CONFIG_PATH=Path('/unused/colab_config.json'), CONFIG=config)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<training-command>', 'exec'), scope)
    command = scope['command']
    assert ('--policy_update_checks' if enabled else '--no-policy_update_checks') in command
    for name in ('rollout_logprob_tolerance', 'kl_probe_samples', 'no_trade_max_validations'):
        assert command[command.index('--' + name) + 1] == str(config[name])


@pytest.mark.parametrize('resume_lr', [None, 1e-5])
def test_training_command_only_passes_an_explicit_resume_lr(resume_lr):
    config = default_config_namespace()['CONFIG']
    config['resume_lr'] = resume_lr
    tree = ast.parse(sources()['train'])
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'command' for target in node.targets):
            nodes.append(node)
        if isinstance(node, ast.If) and 'resume_lr' in ast.unparse(node.test):
            nodes.append(node)
    scope = dict(sys=sys, runner='runner', CONFIG_PATH=Path('/unused/config.json'), CONFIG=config)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<resume-lr-command>', 'exec'), scope)
    command = scope['command']
    assert ('--resume_lr' in command) is (resume_lr is not None)
    if resume_lr is not None:
        assert command[command.index('--resume_lr') + 1] == str(resume_lr)


def test_update_diagnostic_cell_streams_commands_without_training_state(tmp_path, capsys):
    import io

    calls = []

    class Process:
        stdout = io.StringIO('collecting rollout\ncomparison complete\n')

        def wait(self):
            calls.append('wait')
            return 0

    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs['start_new_session'] and kwargs['bufsize'] == 1
        return Process()

    tree = ast.parse(sources()['update-lr'])
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    scope = dict(REPO_PATH=ROOT, os=os, subprocess=SimpleNamespace(Popen=launch,
        PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, CalledProcessError=subprocess.CalledProcessError))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<stream-update-diagnostic>', 'exec'), scope)
    scope['run_update_diagnostic'](['python', '-m', 'capture'], {})
    assert calls == [['python', '-m', 'capture'], 'wait']
    assert 'collecting rollout' in capsys.readouterr().out
    assert Process.stdout.closed


def test_timed_notebook_memory_estimate_accounts_for_replaying_full_market_rows(monkeypatch):
    monkeypatch.setitem(globals(), 'NOTEBOOK', ROOT / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb')
    scope = helpers()
    config = default_config_namespace(EXPERIMENT='timed_decisions')['CONFIG']
    assert scope['estimated_host_gib'](config, episode_rows=100000) > scope['estimated_host_gib'](config)
