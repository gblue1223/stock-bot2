"""Analyze BUY learning signals and linked realized trades without inference or an update."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .entry_credit import analyze_entry_credit
from .grpo import GRPOTrainer
from .update_diagnostic import BUNDLE_FORMAT, BUNDLE_VERSION


def _array(value, name, size=None):
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    result = np.asarray(value)
    if result.ndim != 1 or not len(result) or (size is not None and len(result) != size):
        raise ValueError(f'{name} must be a nonempty aligned one-dimensional array')
    if not np.issubdtype(result.dtype, np.number) and result.dtype != np.bool_:
        raise ValueError(f'{name} must be numeric')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def analyze_bundle(bundle):
    """Replay the trainer's target arithmetic using only small cached one-dimensional arrays."""
    if (not isinstance(bundle, dict) or bundle.get('format') != BUNDLE_FORMAT
            or bundle.get('format_version') != BUNDLE_VERSION):
        raise ValueError('Unsupported fixed update bundle format/version')
    params = bundle['trainer_hyperparameters']
    use_gae = params['use_gae']
    if not isinstance(use_gae, bool):
        raise ValueError('use_gae must be boolean')
    for name in ('gamma', 'lambda_gae', 'group_advantage_coef'):
        value = params[name]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or value < 0
                or (name != 'group_advantage_coef' and value > 1)):
            raise ValueError(f'Invalid {name}')
    if not use_gae and params['group_advantage_coef'] == 0:
        raise ValueError('Zero group coefficient requires GAE')
    source_episodes, advantages = bundle['episodes'], bundle['advantages']
    if not source_episodes or len(source_episodes) != len(advantages):
        raise ValueError('Expected episodes and one group advantage array per episode')
    settings = SimpleNamespace(gamma=params['gamma'], lambda_gae=params['lambda_gae'])
    episodes, gaes, groups, targets, values, hybrid = [], [], [], [], [], []
    for index, (source, advantage) in enumerate(zip(source_episodes, advantages)):
        rewards = _array(source['rewards'], f'episode {index} rewards')
        size = len(rewards)
        episode = {name: _array(source[name], f'episode {index} {name}', size)
                   for name in ('actions', 'dones')}
        if not np.isin(episode['dones'], (0, 1)).all():
            raise ValueError('dones must be boolean or zero/one')
        episode.update(rewards=rewards, metadata=source.get('metadata', {}))
        group = params['group_advantage_coef'] * _array(advantage, 'group advantage', size).astype(np.float32)
        if use_gae:
            # Preserve the trainer's operation order and float32 population std.
            group = params['group_advantage_coef'] * (
                _array(advantage, 'group advantage', size).astype(np.float32) / size)
            cached = _array(source['values'], 'cached values', size).astype(np.float32)
            gae, target = GRPOTrainer._compute_gae(settings, rewards, episode['dones'], cached)
            gaes.append(gae)
            values.append(cached)
            hybrid.append(gae + group)
        else:
            target = GRPOTrainer._compute_returns(settings, rewards)
            hybrid.append(group)
        episodes.append(episode)
        targets.append(target)
        groups.append(group)
    pre_normalized = np.concatenate(hybrid)
    normalized = (pre_normalized - pre_normalized.mean()) / (pre_normalized.std() + 1e-8)
    report = analyze_entry_credit(
        episodes, raw_gae=np.concatenate(gaes) if use_gae else None,
        group_component=np.concatenate(groups), pre_normalized=pre_normalized,
        normalized=normalized, returns=np.concatenate(targets),
        cached_values=np.concatenate(values) if use_gae else None,
        gamma=params['gamma'], lambda_gae=params['lambda_gae'])
    return {
        'format': 'grpo_entry_credit_diagnostic', 'format_version': 1,
        'model_forward_performed': False, 'optimizer_update_performed': False,
        'holdout_test_evaluated': False,
        'settings': {name: params[name] for name in
                     ('use_gae', 'gamma', 'lambda_gae', 'group_advantage_coef')},
        'progress': bundle.get('progress', {}),
        'sample_count': len(normalized), 'episode_count': len(episodes), 'report': report,
        'limitations': [
            'Observed training paths only; this is not a profitability or counterfactual evaluation.',
            'Realized outcomes and advantage signs are compared only for completed entry orders.',
            'Legacy bundles without entry metadata cannot establish trade-level attribution.',
            'Advantage is a pre-update actor signal, not the full shared-network optimizer gradient.',
        ],
    }


def run_diagnostics(bundle_path):
    path = Path(bundle_path).expanduser().resolve()
    # Do not scan, concatenate or copy the large observation/policy/Adam storages.
    bundle = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    report = analyze_bundle(bundle)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    report['bundle'] = str(path)
    report['bundle_sha256'] = digest.hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from .diagnose_update_lr import write_report_exclusive
    if Path(args.output).expanduser().exists():
        raise FileExistsError(f'Diagnostic report already exists: {args.output}')
    report = run_diagnostics(args.bundle)
    destination = write_report_exclusive(args.output, report)
    metrics = report['report']['metrics']
    summary = {key: metrics[key] for key in (
        'entry_attribution_complete', 'missing_episode_count', 'sample_count',
        'entry_count', 'complete_entry_count', 'action/buy/sample_count',
        'action/buy/normalized_advantage_mean', 'outcome/profitable/sample_count',
        'outcome/profitable/normalized_advantage_negative_count', 'outcome/loss/sample_count')}
    print('Entry credit: ' + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'Entry credit report saved: {destination}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
