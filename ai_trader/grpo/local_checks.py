"""Run local regressions, fixed-rollout LR updates and paired validation in sequence."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]


def collect_hardware(python_executable=sys.executable):
    # A short child keeps CUDA contexts out of the long-lived orchestration process.
    code = """import json,platform,psutil,torch
m=psutil.virtual_memory()
result=dict(cpu=platform.processor(),physical_cpus=psutil.cpu_count(logical=False),logical_cpus=psutil.cpu_count(),
 ram_total_gib=m.total/2**30,ram_available_gib=m.available/2**30,torch_version=str(torch.__version__),
 cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available())
if result['cuda_available']:
 free,total=torch.cuda.mem_get_info()
 result.update(gpu_name=torch.cuda.get_device_name(),gpu_capability=list(torch.cuda.get_device_capability()),
               gpu_total_gib=total/2**30,gpu_free_gib=free/2**30)
print(json.dumps(result))
"""
    result = subprocess.run([str(python_executable), '-X', 'utf8', '-c', code],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                            check=True, timeout=60)
    return json.loads(result.stdout)


def _stop_process_tree(process):
    """Stop only the child we started and its workers, including Windows spawn workers."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        import psutil
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        # psutil tracks process creation times, so a recycled PID is not killed.
        # Kill workers before their parent can disappear from the process tree.
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        _, alive = psutil.wait_procs(descendants, timeout=5)
        if alive:
            raise RuntimeError('A diagnostic worker could not be stopped')
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_stage(name, command, *, output_dir, timeout_seconds, env=None):
    """Stream stdout/stderr while enforcing a timeout even when a worker is silent."""
    if timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f'{name}.log'
    started = time.monotonic()
    last_progress = started
    process = None
    reader = None
    messages = queue.Queue()
    settings = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                if os.name == 'nt' else {'start_new_session': True})
    result = {'name': name, 'command': [str(arg) for arg in command], 'log': str(log_path.resolve()),
              'status': 'failed', 'returncode': None}
    print(f'[{name}] starting', flush=True)
    with log_path.open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(result['command'], cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1, **settings)

            def read_output():
                try:
                    for line in process.stdout:
                        messages.put(line)
                finally:
                    messages.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            finished_reading = False
            while not finished_reading or process.poll() is None:
                if time.monotonic() - last_progress >= 60:
                    message = f'[{name}] running: {time.monotonic() - started:.0f}s\n'
                    log.write(message)
                    log.flush()
                    print(message, end='', flush=True)
                    last_progress = time.monotonic()
                if time.monotonic() - started >= timeout_seconds:
                    result['status'] = 'timeout'
                    _stop_process_tree(process)
                    break
                try:
                    line = messages.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    finished_reading = True
                else:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
            result['returncode'] = process.wait(timeout=5)
            if result['status'] != 'timeout':
                result['status'] = 'passed' if result['returncode'] == 0 else 'failed'
        finally:
            if process is not None:
                _stop_process_tree(process)
            if reader is not None:
                reader.join(timeout=5)
            while not messages.empty():
                line = messages.get_nowait()
                if line is not None:
                    log.write(line)
                    print(line, end='', flush=True)
            if process is not None and process.stdout is not None:
                process.stdout.close()
    result['elapsed_seconds'] = time.monotonic() - started
    print(f'[{name}] {result["status"]}: {result["elapsed_seconds"]:.1f}s', flush=True)
    return result


def make_commands(python_executable, *, output_dir, checkpoint, extracted_dir, bundle,
                  device, validation_episodes, num_workers, learning_rates, validation_seed=42):
    directory = Path(output_dir)
    python = [str(python_executable), '-X', 'utf8', '-u']
    return {
        'regression': [*python, '-m', 'pytest', '--basetemp', str(directory / 'pytest_tmp'),
                       '--junitxml', str(directory / 'pytest.xml'), '-o', f'cache_dir={directory / "pytest_cache"}'],
        'capture': [*python, '-m', 'ai_trader.grpo.train_xlstm',
                    '--config', str(directory / 'capture_config.json'),
                    '--resume', '--load_policy', str(checkpoint), '--extracted_dir', str(extracted_dir),
                    '--capture_update_bundle', str(bundle), '--policy_update_checks',
                    '--num_workers', str(num_workers)],
        'entry_credit': [*python, '-m', 'ai_trader.grpo.diagnose_entry_credit',
                         '--bundle', str(bundle), '--output', str(directory / 'entry_credit.json')],
        'compare': [*python, '-m', 'ai_trader.grpo.diagnose_update_lr',
                    '--bundle', str(bundle), '--output', str(directory / 'lr_comparison.json'),
                    '--source-checkpoint', str(checkpoint), '--export-dir', str(directory / 'candidates'),
                    '--learning-rates', *[str(rate) for rate in learning_rates], '--device', device],
        'validation': [*python, '-m', 'ai_trader.grpo.compare_validation',
                       '--checkpoint', str(checkpoint), '--output', str(directory / 'validation_comparison.json'),
                       '--comparison-report', str(directory / 'lr_comparison.json'),
                       '--extracted-dir', str(extracted_dir), '--seed', str(validation_seed),
                       '--episodes', str(validation_episodes), '--device', device],
    }


def _write_summary(directory, report):
    path = directory / 'summary.json'
    temporary = directory / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _check_resources(hardware, device, min_free_ram_gib, min_free_vram_gib):
    if hardware['ram_available_gib'] < min_free_ram_gib:
        raise RuntimeError(f'Insufficient available RAM: {hardware["ram_available_gib"]:.1f} GiB; '
                           f'need {min_free_ram_gib:.1f} GiB. Close other workloads and rerun.')
    if device == 'cuda':
        if not hardware['cuda_available']:
            raise RuntimeError('CUDA requested but unavailable')
        if hardware['gpu_free_gib'] < min_free_vram_gib:
            raise RuntimeError(f'Insufficient available VRAM: {hardware["gpu_free_gib"]:.1f} GiB; '
                               f'need {min_free_vram_gib:.1f} GiB. Saved model/batch settings were not changed.')


def wait_for_resources(*, device, min_free_ram_gib, min_free_vram_gib,
                       timeout_seconds=60, poll_seconds=5):
    """Allow bounded time for the OS to release a finished stage's memory."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        hardware = collect_hardware()
        try:
            _check_resources(hardware, device, min_free_ram_gib, min_free_vram_gib)
            return hardware
        except RuntimeError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (device == 'cuda' and not hardware['cuda_available']):
                raise
            print(f'Waiting for resources ({remaining:.0f}s remaining): {exc}', flush=True)
            time.sleep(min(poll_seconds, remaining))


def run_checks(args):
    from .local_check_inputs import prepare_local_config

    directory = Path(args.output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {'format_version': 2, 'status': 'running', 'started_at': datetime.now().astimezone().isoformat(),
              'output_dir': str(directory), 'stages': [], 'holdout_test_evaluated': False,
              'validation_scope': 'Source checkpoint and every exported LR candidate on identical validation episodes, seed and costs.',
              'comparison_scope': 'Within this local FP32 runtime; not an A100/TF32 reproduction.',
              'checkpoint': str(Path(args.checkpoint).expanduser().resolve())}
    exit_code = 1
    try:
        hardware = collect_hardware()
        device = ('cuda' if hardware['cuda_available'] else 'cpu') if args.device == 'auto' else args.device
        report['hardware'] = hardware
        report['device'] = device
        hardware = wait_for_resources(device=device, min_free_ram_gib=args.min_free_ram_gib,
                                      min_free_vram_gib=args.min_free_vram_gib,
                                      timeout_seconds=args.resource_wait_seconds)
        report['hardware'] = hardware
        prepared = prepare_local_config(args.checkpoint, args.extracted_dir, directory / 'capture_unused',
                                        device=device, num_workers=args.num_workers)
        config = prepared.pop('config')
        report['inputs'] = prepared
        report['explicit_capture_overrides'] = {'policy_update_checks': True, 'tf32': False}
        bundle = Path(args.bundle).expanduser().resolve() if args.bundle else directory / 'fixed_rollout.pt'
        if args.bundle and not bundle.is_file():
            raise FileNotFoundError(f'Existing bundle not found: {bundle}')
        if args.bundle:
            from .local_check_inputs import validate_bundle_source
            report['bundle_provenance'] = validate_bundle_source(bundle, prepared['checkpoint'])
        config['policy_update_checks'] = True
        (directory / 'capture_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        commands = make_commands(sys.executable, output_dir=directory, checkpoint=prepared['checkpoint'],
                                 extracted_dir=Path(args.extracted_dir).expanduser().resolve(), bundle=bundle,
                                 device=device, validation_episodes=args.validation_episodes,
                                 num_workers=args.num_workers, learning_rates=args.learning_rates,
                                 validation_seed=config.get('evaluation_seed', 42))
        names = [name for name in commands if not (name == 'regression' and args.skip_regression)
                 and not (name == 'capture' and args.bundle)]
        report['commands'] = {name: commands[name] for name in names}
        report['bundle'] = str(bundle)
        report['bundle_source'] = 'supplied bundle' if args.bundle else 'new local rollout; may differ from Colab collection'
        _write_summary(directory, report)
        if args.plan_only:
            report['status'] = 'planned'
            exit_code = 0
        else:
            child_env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', SCALPING_TF32='0')
            # compare_validation sets and restores FP32 itself for all four policies.
            for name in names:
                current_hardware = wait_for_resources(device=device, min_free_ram_gib=args.min_free_ram_gib,
                    min_free_vram_gib=args.min_free_vram_gib, timeout_seconds=args.resource_wait_seconds)
                report.setdefault('stage_start_resources', {})[name] = current_hardware
                stage = run_stage(name, commands[name], output_dir=directory,
                                  timeout_seconds=args.stage_timeout, env=child_env)
                report['stages'].append(stage)
                _write_summary(directory, report)
                if stage['status'] != 'passed':
                    report['status'] = stage['status']
                    report['failed_stage'] = name
                    break
            else:
                report['status'] = 'passed'
                exit_code = 0
                comparison = json.loads((directory / 'lr_comparison.json').read_text(encoding='utf-8'))
                validation = json.loads((directory / 'validation_comparison.json').read_text(encoding='utf-8'))
                report['comparison'] = [{key: variant.get(key) for key in
                    ('learning_rate', 'status', 'guard_reason', 'full_rollout_kl',
                     'candidate_checkpoint', 'candidate_sha256')} for variant in comparison['variants']]
                report['validation'] = validation['summary']
                report['validation_report'] = str(directory / 'validation_comparison.json')
                report['entry_credit_report'] = str(directory / 'entry_credit.json')
                print('Validation comparison: ' + json.dumps(report['validation'], ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        exit_code = 130
    except Exception as exc:
        exit_code = 1
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        print(f'Local checks failed: {exc}', file=sys.stderr, flush=True)
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        _write_summary(directory, report)
    print(f'Local checks {report["status"]}: {directory / "summary.json"}', flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--extracted-dir', required=True)
    parser.add_argument('--bundle', help='Reuse an existing fixed-rollout bundle instead of capturing another')
    parser.add_argument('--output-dir', default=str(ROOT / '.test_artifacts' / 'local_checks' /
                                                   (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--validation-episodes', type=int, default=8)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[3e-5, 1e-5, 3e-6])
    parser.add_argument('--stage-timeout', type=float, default=7200, help='Seconds per stage, including silent worker hangs')
    parser.add_argument('--min-free-ram-gib', type=float, default=8)
    parser.add_argument('--min-free-vram-gib', type=float, default=4)
    parser.add_argument('--resource-wait-seconds', type=float, default=60,
                        help='Bounded wait for RAM/VRAM to recover between stages; zero fails immediately')
    parser.add_argument('--skip-regression', action='store_true', help='Explicitly omit pytest on a repeated diagnostic run')
    parser.add_argument('--plan-only', action='store_true', help='Validate inputs and write commands without running tests')
    args = parser.parse_args(argv)
    if args.num_workers < 1 or args.validation_episodes < 1:
        parser.error('workers and validation episodes must be positive')
    import math
    if not math.isfinite(args.resource_wait_seconds) or args.resource_wait_seconds < 0:
        parser.error('resource wait must be finite and nonnegative')
    if any(not math.isfinite(value) or value <= 0 for value in
           [*args.learning_rates, args.stage_timeout, args.min_free_ram_gib, args.min_free_vram_gib]):
        parser.error('rates, timeout and resource limits must be finite and positive')
    return run_checks(args)


if __name__ == '__main__':
    raise SystemExit(main())
