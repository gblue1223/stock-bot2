"""Entry diagnostics preserve optimizer inputs and survive portable rollout capture."""
import copy
import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.diagnose_entry_credit import analyze_bundle, main, run_diagnostics
from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.update_diagnostic import BUNDLE_FORMAT, BUNDLE_VERSION, prepare_rollouts
from test_grpo_memory_batches import RecordingPolicy, make_episode
from test_update_lr_diagnostic import assert_tree_equal, snapshot_rng


def _metadata(episode):
    return {'entry_diagnostics': {
        'version': 1, 'decision_count': len(episode['actions']),
        'buy_decisions': [{'decision_index': int(index), 'entry_order_id': None,
                           'outcome': 'position_limit'}
                          for index in np.flatnonzero(episode['actions'] == 1)],
        'entries': []}, 'episode_start': {
            'episode_key': {'stock_code': 'TEST', 'date': '20260101', 'start_index': 7},
            'market_context': np.zeros(100), 'unrelated': object()}, 'unrelated': object()}


def _bundle(episodes, advantages, trainer):
    return {'format': BUNDLE_FORMAT, 'format_version': BUNDLE_VERSION,
            'trainer_hyperparameters': {name: getattr(trainer, name) for name in
                ('gamma', 'lambda_gae', 'use_gae', 'group_advantage_coef')},
            'episodes': episodes, 'advantages': advantages}


@pytest.mark.parametrize('use_gae', [True, False])
def test_diagnostics_preserve_policy_adam_and_rng_and_offline_matches_update(tmp_path, use_gae):
    torch.set_num_threads(1)
    policy = RecordingPolicy()
    episode, values = make_episode(policy, 7)
    episode['values'] = values
    traced = copy.deepcopy(episode)
    traced['metadata'] = _metadata(traced)
    advantages = [np.linspace(-.25, .4, 7, dtype=np.float32)]
    trainers = [GRPOTrainer(copy.deepcopy(policy), object(), batch_size=3, num_epochs=2,
                            gamma=1., lambda_gae=.95, use_gae=use_gae,
                            group_advantage_coef=.25) for _ in range(2)]
    rngs = []
    for trainer, inputs in zip(trainers, ([episode], [traced])):
        np.random.seed(41)
        torch.manual_seed(43)
        trainer.policy.forward_batches.clear()
        metrics = trainer.update_policy(inputs, advantages)
        assert all(np.isfinite(value) for value in metrics.values())
        rngs.append(snapshot_rng())
    assert_tree_equal(trainers[0].policy.state_dict(), trainers[1].policy.state_dict())
    assert_tree_equal(trainers[0].optimizer.state_dict(), trainers[1].optimizer.state_dict())
    assert_tree_equal(rngs[0], rngs[1])
    assert trainers[0].policy.forward_batches == trainers[1].policy.forward_batches
    offline = analyze_bundle(_bundle([traced], advantages, trainers[1]))
    assert offline['report'] == trainers[1].last_entry_credit_report
    assert not offline['model_forward_performed'] and not offline['optimizer_update_performed']
    trainers[1]._record_entry_credit(2, str(tmp_path / 'checkpoint_iter{}.pt'))
    saved = json.loads((tmp_path / 'diagnostics/entry_credit/iteration_000002.json').read_text('utf-8'))
    assert saved['report'] == offline['report']


def test_prepare_rollouts_keeps_only_portable_entry_metadata_and_replays_it(tmp_path):
    trainer = GRPOTrainer(RecordingPolicy(), object())
    episode, values = make_episode(trainer.policy, 5)
    episode['values'] = values
    episode['metadata'] = _metadata(episode)
    episodes, advantages = prepare_rollouts([episode], [np.zeros(5, np.float32)], action_dim=3, masked=False)
    assert set(episodes[0]['metadata']) == {'entry_diagnostics', 'episode_start'}
    assert episodes[0]['metadata']['episode_start'] == {
        'episode_key': {'stock_code': 'TEST', 'date': '20260101', 'start_index': 7}}
    before = episodes[0]['metadata']['entry_diagnostics']['decision_count']
    episode['metadata']['entry_diagnostics']['decision_count'] = 100
    assert episodes[0]['metadata']['entry_diagnostics']['decision_count'] == before
    path = tmp_path / 'inputs.pt'
    torch.save(_bundle(episodes, advantages, trainer), path)
    report = run_diagnostics(path)
    assert report['sample_count'] == 5
    assert len(report['bundle_sha256']) == 64
    output = tmp_path / 'report.json'
    assert main(['--bundle', str(path), '--output', str(output)]) == 0
    assert json.loads(output.read_text('utf-8')) == report
    with pytest.raises(FileExistsError):
        main(['--bundle', str(path), '--output', str(output)])


def test_offline_does_not_access_observation_storage_and_legacy_stays_unavailable():
    class UnreadableStates:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Offline attribution must not copy observations')

    trainer = GRPOTrainer(RecordingPolicy(), object())
    episode, values = make_episode(trainer.policy, 5)
    episode.update(values=values, states=UnreadableStates())
    report = analyze_bundle(_bundle([episode], [np.zeros(5, np.float32)], trainer))
    assert report['report']['metrics']['entry_attribution_available'] == 0
    assert report['report']['entries'] == []
