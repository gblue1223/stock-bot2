"""Exercise local orchestration with small real child processes, never a GPU job."""
import os
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from ai_trader.grpo import local_checks
from ai_trader.grpo.local_checks import make_commands, run_stage


def _python(code):
    return [sys.executable, '-X', 'utf8', '-u', '-c', code]


def _argument(command, name):
    """Accept both argparse spellings without fixing command formatting."""
    if name in command:
        return command[command.index(name) + 1]
    for value in command:
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    raise AssertionError(f'Missing {name} in {command}')


def _module(command):
    return command[command.index('-m') + 1]


def _read_log(result, output_dir):
    log = Path(result['log'])
    assert log.resolve().is_relative_to(output_dir.resolve())
    return log.read_text(encoding='utf-8')


def test_run_stage_streams_and_saves_utf8_stdout_stderr_and_environment(tmp_path, capsys):
    command = _python(
        "import os, sys; print('검증 성공 ✓', flush=True); "
        "print('stderr 진단', file=sys.stderr, flush=True); "
        "print(os.environ['LOCAL_CHECK_TEST_VALUE'], flush=True)")
    result = run_stage('unicode', command, output_dir=tmp_path, timeout_seconds=10,
                       env={**os.environ, 'LOCAL_CHECK_TEST_VALUE': '전달된 설정'})
    assert result['name'] == 'unicode'
    assert result['status'] == 'passed'
    assert result['returncode'] == 0
    assert result['command'] == command
    assert result['elapsed_seconds'] >= 0
    log = _read_log(result, tmp_path)
    output = capsys.readouterr().out
    for expected in ('검증 성공 ✓', 'stderr 진단', '전달된 설정'):
        assert expected in log
        assert expected in output


def test_run_stage_nonzero_is_failed_and_retains_failure_log(tmp_path):
    result = run_stage('failure', _python("import sys; print('expected failure'); sys.exit(7)"),
                       output_dir=tmp_path, timeout_seconds=10)
    assert result['status'] == 'failed'
    assert result['returncode'] == 7
    assert 'expected failure' in _read_log(result, tmp_path)


def test_run_stage_displays_progress_before_child_finishes(tmp_path, monkeypatch):
    gate = tmp_path / 'output_was_streamed'
    token = 'actual child output is streaming'
    original_stdout = sys.stdout

    class OutputGate:
        def __init__(self):
            self.written = ''

        def write(self, text):
            self.written += text
            if token in self.written:
                gate.touch()
            return original_stdout.write(text)

        def __getattr__(self, name):
            return getattr(original_stdout, name)

    monkeypatch.setattr(sys, 'stdout', OutputGate())
    # Construct the token inside the child: printing a command cannot open the gate.
    code = (
        'from pathlib import Path; import sys, time; '
        f'print("".join(map(chr, {list(map(ord, token))!r})), flush=True); '
        f'gate = Path({str(gate)!r}); deadline = time.monotonic() + 2'
        '\nwhile not gate.exists() and time.monotonic() < deadline: time.sleep(.01)'
        '\nsys.exit(0 if gate.exists() else 9)')
    result = run_stage('streaming', _python(code), output_dir=tmp_path, timeout_seconds=5)
    assert result['status'] == 'passed', 'Progress was buffered until the child exited'
    assert result['returncode'] == 0


def test_run_stage_times_out_without_waiting_for_child_output(tmp_path):
    started = time.monotonic()
    result = run_stage('silent', _python('import time; time.sleep(60)'),
                       output_dir=tmp_path, timeout_seconds=.5)
    assert result['status'] == 'timeout'
    assert result['returncode'] != 0
    assert time.monotonic() - started < 15
    assert result['elapsed_seconds'] >= .4
    _read_log(result, tmp_path)


def test_run_stage_timeout_stops_its_descendant_process(tmp_path):
    ready = tmp_path / 'descendant_ready'
    released = tmp_path / 'timeout_returned'
    late_output = tmp_path / 'descendant_survived'
    descendant = (
        'from pathlib import Path; import time; '
        f'Path({str(ready)!r}).write_text("ready"); '
        f'released = Path({str(released)!r}); deadline = time.monotonic() + 20'
        '\nwhile not released.exists() and time.monotonic() < deadline: time.sleep(.01)'
        f'\nif released.exists(): Path({str(late_output)!r}).write_text("survived")')
    parent = (
        'import subprocess, sys, time; from pathlib import Path; '
        f'subprocess.Popen([sys.executable, "-c", {descendant!r}], '
        'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); '
        f'ready = Path({str(ready)!r}); '
        '\nwhile not ready.exists(): time.sleep(.01)'
        '\nprint("descendant ready", flush=True)'
        '\ntime.sleep(60)')
    result = run_stage('descendants', _python(parent), output_dir=tmp_path,
                       timeout_seconds=1.0)
    assert result['status'] == 'timeout'
    assert 'descendant ready' in _read_log(result, tmp_path)
    # Permit platform cleanup latency; the descendant must be gone on return.
    released.touch()
    deadline = time.monotonic() + 1
    while not late_output.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert not late_output.exists(), 'A timed-out stage left its own descendant running'


@pytest.fixture
def commands(tmp_path):
    return make_commands(
        sys.executable, output_dir=tmp_path, checkpoint=tmp_path / 'checkpoint19.pt',
        extracted_dir=tmp_path / '추출 데이터', bundle=tmp_path / 'frozen_rollouts.pt',
        device='cuda', validation_episodes=8, num_workers=2,
        learning_rates=[3e-5, 1e-5, 3e-6])


def test_make_commands_uses_requested_python_and_writes_regression_results(commands, tmp_path):
    assert set(commands) == {'regression', 'capture', 'entry_credit', 'compare', 'validation'}
    assert all(command[0] == sys.executable for command in commands.values())
    regression = commands['regression']
    assert _module(regression) == 'pytest'
    assert Path(_argument(regression, '--junitxml')) == tmp_path / 'pytest.xml'
    assert Path(_argument(regression, '--basetemp')) == tmp_path / 'pytest_tmp'
    assert f'cache_dir={tmp_path / "pytest_cache"}' in regression


def test_capture_command_only_collects_requested_checkpoint_and_bundle(commands, tmp_path):
    capture = commands['capture']
    assert _module(capture) == 'ai_trader.grpo.train_xlstm'
    assert Path(_argument(capture, '--config')) == tmp_path / 'capture_config.json'
    assert '--resume' in capture
    assert '--policy_update_checks' in capture
    assert Path(_argument(capture, '--load_policy')) == tmp_path / 'checkpoint19.pt'
    assert Path(_argument(capture, '--capture_update_bundle')) == tmp_path / 'frozen_rollouts.pt'
    assert '--resume_lr' not in capture


def test_entry_credit_runs_without_gpu_or_update_options(commands, tmp_path):
    diagnostic = commands['entry_credit']
    assert _module(diagnostic) == 'ai_trader.grpo.diagnose_entry_credit'
    assert Path(_argument(diagnostic, '--bundle')) == tmp_path / 'frozen_rollouts.pt'
    assert Path(_argument(diagnostic, '--output')) == tmp_path / 'entry_credit.json'
    assert '--device' not in diagnostic and '--learning-rates' not in diagnostic


def test_capture_command_passes_real_training_parser_and_keeps_config_device(commands, tmp_path, monkeypatch):
    from test_profit_experiment_config import capture_cli_config

    config_path = tmp_path / 'capture_config.json'
    config_path.write_text(json.dumps({'device': 'cuda', 'num_workers': 8, 'batch_size': 32}),
                           encoding='utf-8')
    capture = commands['capture']
    # Exercise the real parser, then stop at config validation before any model or GPU work.
    parsed = capture_cli_config(monkeypatch, capture[6:])
    assert parsed['device'] == 'cuda'
    assert parsed['num_workers'] == 2
    assert parsed['batch_size'] == 32
    assert parsed['resume'] is True
    assert parsed['policy_update_checks'] is True
    assert Path(parsed['load_policy']) == tmp_path / 'checkpoint19.pt'
    assert Path(parsed['extracted_dir']) == tmp_path / '추출 데이터'
    assert Path(parsed['capture_update_bundle']) == tmp_path / 'frozen_rollouts.pt'


def test_compare_command_keeps_saved_batching_and_requested_learning_rates(commands, tmp_path):
    compare = commands['compare']
    assert _module(compare) == 'ai_trader.grpo.diagnose_update_lr'
    assert Path(_argument(compare, '--bundle')) == tmp_path / 'frozen_rollouts.pt'
    assert Path(_argument(compare, '--output')) == tmp_path / 'lr_comparison.json'
    assert _argument(compare, '--device') == 'cuda'
    assert Path(_argument(compare, '--source-checkpoint')) == tmp_path / 'checkpoint19.pt'
    assert Path(_argument(compare, '--export-dir')) == tmp_path / 'candidates'
    rates_index = compare.index('--learning-rates') + 1
    rates = []
    for value in compare[rates_index:]:
        if value.startswith('--'):
            break
        rates.append(float(value))
    assert rates == [3e-5, 1e-5, 3e-6]
    for command in (commands['capture'], compare):
        assert not any(argument.startswith(('--batch_size', '--batch-size', '--update-batch-size'))
                       for argument in command)


def test_validation_command_uses_only_saved_validation_with_local_data(commands, tmp_path):
    validation = commands['validation']
    assert _module(validation) == 'ai_trader.grpo.compare_validation'
    assert '--split' not in validation
    assert _argument(validation, '--seed') == '42'
    assert Path(_argument(validation, '--comparison-report')) == tmp_path / 'lr_comparison.json'
    assert _argument(validation, '--episodes') == '8'
    assert _argument(validation, '--device') == 'cuda'
    assert Path(_argument(validation, '--checkpoint')) == tmp_path / 'checkpoint19.pt'
    assert Path(_argument(validation, '--extracted-dir')) == tmp_path / '추출 데이터'
    assert Path(_argument(validation, '--output')) == tmp_path / 'validation_comparison.json'
    assert all('--partition' not in command and 'test' not in command for command in commands.values())
    training_commands = [command for command in commands.values()
                         if _module(command) == 'ai_trader.grpo.train_xlstm']
    assert training_commands and all('--capture_update_bundle' in command for command in training_commands)


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    checkpoint = tmp_path / 'source.pt'
    data = tmp_path / 'data'
    hardware = {'ram_available_gib': 64, 'cuda_available': False}
    monkeypatch.setattr(local_checks, 'collect_hardware', lambda: dict(hardware))
    inputs = ModuleType('ai_trader.grpo.local_check_inputs')

    def prepare(checkpoint_path, extracted_dir, output_dir, *, device, num_workers):
        assert Path(checkpoint_path) == checkpoint
        assert Path(extracted_dir) == data
        assert device == 'cpu'
        return {'checkpoint': str(checkpoint),
                'config': {'batch_size': 32, 'policy_update_checks': False, 'evaluation_seed': 73}}

    inputs.prepare_local_config = prepare

    def provenance(bundle_path, checkpoint_path):
        assert Path(bundle_path).is_file()
        assert Path(checkpoint_path) == checkpoint
        return {'source_verified': True}

    inputs.validate_bundle_source = provenance
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    arguments = ['--checkpoint', str(checkpoint), '--extracted-dir', str(data),
                 '--output-dir', str(output), '--device', 'cpu']
    return output, arguments


def _stage_result(name, command, output_dir, status='passed'):
    return {'name': name, 'command': command, 'status': status,
            'returncode': 0 if status == 'passed' else 7, 'elapsed_seconds': .01,
            'log': str(output_dir / (name + '.log'))}


@pytest.mark.parametrize('failed_stage,status', [('regression', 'failed'),
                                                ('capture', 'timeout'), ('entry_credit', 'failed'), ('compare', 'failed'),
                                                ('validation', 'failed')])
def test_orchestration_stops_after_first_unsuccessful_stage(orchestration, monkeypatch,
                                                           failed_stage, status):
    output, arguments = orchestration
    visited = []

    def stage(name, command, *, output_dir, timeout_seconds, env):
        visited.append(name)
        return _stage_result(name, command, output_dir,
                             status if name == failed_stage else 'passed')

    monkeypatch.setattr(local_checks, 'run_stage', stage)
    assert local_checks.main(arguments) == 1
    sequence = ['regression', 'capture', 'entry_credit', 'compare', 'validation']
    assert visited == sequence[:sequence.index(failed_stage) + 1]
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == status
    assert summary['failed_stage'] == failed_stage
    assert summary['holdout_test_evaluated'] is False


def test_orchestration_exports_candidates_and_compares_same_validation_conditions(orchestration, monkeypatch):
    output, arguments = orchestration
    seen = []
    candidate = output / 'candidates' / 'lr_00_1e-05.pt'

    def stage(name, command, *, output_dir, timeout_seconds, env):
        seen.append(name)
        assert env['SCALPING_TF32'] == '0'
        assert env['PYTHONUTF8'] == '1'
        if name == 'compare':
            assert _argument(command, '--source-checkpoint') == _argument(arguments, '--checkpoint')
            assert Path(_argument(command, '--export-dir')) == candidate.parent
            (output_dir / 'lr_comparison.json').write_text(json.dumps({
                'variants': [{'learning_rate': 1e-5, 'status': 'ok',
                              'candidate_checkpoint': str(candidate), 'candidate_sha256': 'abc'}]}), encoding='utf-8')
        elif name == 'validation':
            assert _module(command) == 'ai_trader.grpo.compare_validation'
            assert _argument(command, '--checkpoint') == _argument(arguments, '--checkpoint')
            assert _argument(command, '--extracted-dir') == _argument(arguments, '--extracted-dir')
            assert _argument(command, '--seed') == '73'
            assert _argument(command, '--episodes') == '8'
            assert Path(_argument(command, '--comparison-report')) == output_dir / 'lr_comparison.json'
            assert Path(_argument(command, '--output')) == output_dir / 'validation_comparison.json'
            (output_dir / 'validation_comparison.json').write_text(json.dumps({
                'summary': {'candidate_count': 1, 'all_no_trade': True,
                            'improved_mean_return_candidate_count': 0}}), encoding='utf-8')
        return _stage_result(name, command, output_dir)

    monkeypatch.setattr(local_checks, 'run_stage', stage)
    assert local_checks.main(arguments) == 0
    assert seen == ['regression', 'capture', 'entry_credit', 'compare', 'validation']
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'passed'
    assert summary['holdout_test_evaluated'] is False
    assert summary['validation']['all_no_trade'] is True
    assert summary['validation']['candidate_count'] == 1
    assert summary['comparison'][0]['candidate_checkpoint'] == str(candidate)
    assert 'every exported LR candidate' in summary['validation_scope']
    assert [item['name'] for item in summary['stages']] == seen



def test_plan_only_does_not_launch_any_test_or_diagnostic(orchestration, monkeypatch):
    output, arguments = orchestration
    monkeypatch.setattr(local_checks, 'run_stage', lambda *a, **kw: pytest.fail('Plan must not run a stage'))
    assert local_checks.main([*arguments, '--plan-only']) == 0
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'planned'
    assert summary['stages'] == []
    assert summary['holdout_test_evaluated'] is False


def test_supplied_bundle_skips_capture_and_requires_explicit_regression_skip(orchestration, tmp_path, monkeypatch):
    output, arguments = orchestration
    bundle = tmp_path / 'supplied.pt'
    bundle.write_bytes(b'fixture is not loaded by this orchestration test')
    visited = []

    def stage(name, command, *, output_dir, timeout_seconds, env):
        visited.append(name)
        assert name == 'entry_credit'
        assert Path(_argument(command, '--bundle')) == bundle
        return _stage_result(name, command, output_dir, 'failed')

    monkeypatch.setattr(local_checks, 'run_stage', stage)
    assert local_checks.main([*arguments, '--bundle', str(bundle), '--skip-regression']) == 1
    assert visited == ['entry_credit']
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert 'capture' not in summary['commands']
    assert 'regression' not in summary['commands']
    assert summary['bundle_provenance']['source_verified'] is True


def test_mismatched_supplied_bundle_blocks_every_stage(orchestration, tmp_path, monkeypatch):
    output, arguments = orchestration
    bundle = tmp_path / 'different_checkpoint.pt'
    bundle.write_bytes(b'different policy and Adam fixture')

    def mismatched(bundle_path, checkpoint_path):
        raise ValueError('Bundle policy/Adam does not match the requested source checkpoint')

    monkeypatch.setattr(sys.modules['ai_trader.grpo.local_check_inputs'], 'validate_bundle_source', mismatched)
    monkeypatch.setattr(local_checks, 'run_stage', lambda *a, **kw: pytest.fail('Mismatched inputs must stop before stages'))
    assert local_checks.main([*arguments, '--bundle', str(bundle)]) == 1
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'failed'
    assert summary['stages'] == []
    assert 'does not match' in summary['error']['message']
    assert summary['holdout_test_evaluated'] is False


def test_missing_result_artifacts_cannot_return_success(orchestration, monkeypatch):
    output, arguments = orchestration

    def stage(name, command, *, output_dir, timeout_seconds, env):
        return _stage_result(name, command, output_dir)

    monkeypatch.setattr(local_checks, 'run_stage', stage)
    assert local_checks.main(arguments) == 1
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'failed'
    assert summary['error']['type'] == 'FileNotFoundError'


@pytest.fixture
def resource_clock(monkeypatch):
    clock = {'now': 0., 'sleeps': []}

    def advance(seconds):
        assert 0 < seconds <= 5
        clock['sleeps'].append(seconds)
        clock['now'] += seconds

    monkeypatch.setattr(local_checks, 'time', SimpleNamespace(
        monotonic=lambda: clock['now'], sleep=advance))
    return clock


def test_resource_wait_retries_until_memory_is_released(monkeypatch, resource_clock):
    unavailable = {'ram_available_gib': 5.7, 'cuda_available': True, 'gpu_free_gib': 3.}
    recovered = {'ram_available_gib': 42., 'cuda_available': True, 'gpu_free_gib': 7.7}
    snapshots = iter([unavailable, recovered])
    monkeypatch.setattr(local_checks, 'collect_hardware', lambda: next(snapshots))
    result = local_checks.wait_for_resources(
        device='cuda', min_free_ram_gib=8, min_free_vram_gib=4,
        timeout_seconds=60, poll_seconds=5)
    assert result == recovered
    assert resource_clock['sleeps'] == [5]
    # Waiting must not reinterpret insufficient memory as an acceptable threshold.
    with pytest.raises(RuntimeError, match='RAM'):
        local_checks._check_resources(unavailable, 'cuda', 8, 4)


def test_resource_wait_returns_immediately_when_thresholds_are_met(monkeypatch, resource_clock):
    available = {'ram_available_gib': 8., 'cuda_available': True, 'gpu_free_gib': 4.}
    calls = []

    def collect():
        calls.append(True)
        return dict(available)

    monkeypatch.setattr(local_checks, 'collect_hardware', collect)
    assert local_checks.wait_for_resources(
        device='cuda', min_free_ram_gib=8, min_free_vram_gib=4,
        timeout_seconds=0, poll_seconds=5) == available
    assert calls == [True]
    assert resource_clock['sleeps'] == []


@pytest.mark.parametrize('wait_seconds', [0, 10])
def test_resource_wait_timeout_preserves_threshold_and_blocks_next_stage(
        orchestration, monkeypatch, resource_clock, wait_seconds):
    output, arguments = orchestration
    visited = []

    def collect():
        return {'ram_available_gib': 5.7 if visited else 42., 'cuda_available': False}

    def stage(name, command, *, output_dir, timeout_seconds, env):
        visited.append(name)
        return _stage_result(name, command, output_dir)

    monkeypatch.setattr(local_checks, 'collect_hardware', collect)
    monkeypatch.setattr(local_checks, 'run_stage', stage)
    assert local_checks.main([*arguments, '--resource-wait-seconds', str(wait_seconds)]) == 1
    assert visited == ['regression']
    assert sum(resource_clock['sleeps']) == wait_seconds
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'failed'
    assert summary['error']['type'] == 'RuntimeError'
    assert 'RAM' in summary['error']['message']
    assert [item['name'] for item in summary['stages']] == ['regression']
    assert summary['holdout_test_evaluated'] is False
