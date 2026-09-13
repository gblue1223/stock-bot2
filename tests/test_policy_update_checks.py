"""Reject the step that breaks the KL constraint, including its Adam moments."""
import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.grpo import GRPOTrainer
from ai_trader.grpo.policy_update_checks import check_rollout_likelihood, exact_policy_kl, snapshot_optimizer_step


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


class CheckedPolicy(torch.nn.Module):
    execution_action_mask = True

    def __init__(self):
        super().__init__()
        torch.manual_seed(713)
        self.actor = torch.nn.Linear(2, 3)
        self.critic = torch.nn.Linear(2, 1)
        self.register_buffer('sentinel', torch.tensor(7.))
        self.calls = []
        self.training_shift = 0.

    def evaluate_actions_with_distribution(self, states, actions, action_masks=None):
        self.calls.append((len(states), torch.is_grad_enabled(), self.training))
        features = states.mean(1)
        logits = self.actor(features)
        if self.training:
            logits = logits + logits.new_tensor([self.training_shift, 0., 0.])
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), self.critic(features).squeeze(-1), distribution.logits

    def evaluate_actions(self, states, actions, action_masks=None):
        return self.evaluate_actions_with_distribution(states, actions, action_masks)[:3]


def episode(policy, n=6, masks=None):
    states = np.linspace(-1, 1, n * 4, dtype=np.float32).reshape(n, 2, 2)
    if masks is None:
        masks = np.array([[True, False, False], [True, True, False], [True, False, True]] * ((n + 2) // 3))[:n]
    actions = np.where(masks[:, 2], 2, np.where(masks[:, 1], 1, 0))
    device = next(policy.parameters()).device
    policy.eval()
    with torch.no_grad():
        logs, _, values = policy.evaluate_actions(torch.from_numpy(states).to(device),
                                                 torch.from_numpy(actions).to(device),
                                                 torch.from_numpy(masks).to(device))
    return {'states': states, 'actions': actions, 'log_probs': logs.detach().cpu().numpy().copy(), 'values': values.detach().cpu().numpy().copy(),
            'action_masks': masks, 'rewards': np.linspace(-.2, .4, n, dtype=np.float32),
            'dones': np.arange(n) == n - 1}


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_state_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('name,value', [('policy_update_checks', 1), ('policy_update_checks', 'true'),
                                      ('rollout_logprob_tolerance', True), ('rollout_logprob_tolerance', np.nan),
                                      ('rollout_logprob_tolerance', -.1), ('kl_probe_samples', 0),
                                      ('kl_probe_samples', True), ('kl_probe_samples', 3.2)])
def test_invalid_update_check_settings_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        GRPOTrainer(CheckedPolicy(), object(), **{name: value})


@pytest.mark.parametrize('kl_target', [True, np.nan, 0., -1.])
def test_enabled_guard_cannot_have_an_invalid_kl_limit(kl_target):
    with pytest.raises(ValueError, match='kl_target'):
        GRPOTrainer(CheckedPolicy(), object(), policy_update_checks=True, kl_target=kl_target)


def test_likelihood_check_uses_train_gradient_path_and_preserves_cached_masks_and_mode():
    policy = CheckedPolicy()
    data = episode(policy, 7)
    assert torch.count_nonzero(policy.actor.weight) == policy.actor.weight.numel()
    before = copy.deepcopy(policy.state_dict())
    policy.calls.clear()
    result = check_rollout_likelihood(policy, data['states'], data['actions'], data['log_probs'],
                                      data['action_masks'], batch_size=3)
    assert policy.calls == [(3, True, True), (3, True, True), (1, True, True)]
    assert not policy.training
    assert result['num_samples'] == 7 and result['max_abs_error'] < 1e-6
    assert result['reference_log_probs'].device.type == 'cpu'
    assert torch.all(result['reference_log_probs'].exp()[~torch.from_numpy(data['action_masks'])] == 0)
    assert_state_equal(before, policy.state_dict())


@pytest.mark.parametrize('corruption', ['old_log_prob', 'training_path', 'cached_mask', 'nonfinite'])
def test_preflight_mismatch_fails_before_any_optimizer_step(corruption):
    policy = CheckedPolicy()
    data = episode(policy)
    if corruption == 'old_log_prob':
        data['log_probs'][1] += .05
    elif corruption == 'training_path':
        policy.training_shift = .5
    elif corruption == 'cached_mask':
        data['action_masks'][1, 1] = False
    else:
        data['log_probs'][1] = np.nan
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=3, num_epochs=1)
    before = snapshot_optimizer_step(policy, trainer.optimizer)
    with pytest.raises((ValueError, FloatingPointError)):
        trainer.update_policy([data], [np.zeros(len(data['states']))])
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))
    assert not trainer.optimizer.state


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_failed_first_step_restores_nonzero_head_buffers_and_initialized_adam_exactly(device):
    policy = CheckedPolicy()
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=6,
                          num_epochs=1, kl_target=.001, learning_rate=.01, device=device)
    # Initialize nonzero Adam moments before the guarded step.
    sum(parameter.square().sum() for parameter in policy.parameters()).backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    assert any(state['exp_avg'].abs().sum() > 0 for state in trainer.optimizer.state.values())
    trainer.optimizer.param_groups[0]['lr'] = 50.
    data = episode(policy)
    before = snapshot_optimizer_step(policy, trainer.optimizer)
    original_step = trainer.optimizer.step

    def modifying_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        policy.sentinel.add_(1)
        return result

    trainer.optimizer.step = modifying_step
    metrics = trainer.update_policy([data], [np.zeros(6)])
    assert metrics['optimizer_attempted_steps'] == metrics['optimizer_rejected_steps'] == 1
    assert metrics['optimizer_accepted_steps'] == metrics['optimizer_steps'] == 0
    assert metrics['post_update_kl_early_stopped'] == 1
    assert metrics['last_post_update_probe_kl'] > trainer.kl_target * 1.5
    assert metrics['policy_loss'] == metrics['value_loss'] == 0  # Rejected proposal is excluded.
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))


def test_current_minibatch_guard_catches_change_outside_a_hold_only_fixed_probe(monkeypatch):
    monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))
    policy = CheckedPolicy()
    masks = np.array([[True, False, False], [True, True, False]])
    data = episode(policy, 2, masks)
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, kl_probe_samples=1,
                          batch_size=1, num_epochs=1, kl_target=.001, learning_rate=50., value_coef=0.)
    snapshots = []
    original_step = trainer.optimizer.step

    def remember_step(*args, **kwargs):
        snapshots.append(snapshot_optimizer_step(policy, trainer.optimizer))
        return original_step(*args, **kwargs)

    trainer.optimizer.step = remember_step
    metrics = trainer.update_policy([data], [np.zeros(2)])
    assert metrics['optimizer_accepted_steps'] == 1 and metrics['optimizer_attempted_steps'] == 2
    assert metrics['optimizer_rejected_steps'] == 1
    assert metrics['last_post_update_probe_kl'] == 0
    assert metrics['last_post_update_minibatch_kl'] > trainer.kl_target * 1.5
    assert_state_equal(snapshots[1], snapshot_optimizer_step(policy, trainer.optimizer))


def test_fixed_probe_catches_shared_value_update_outside_current_hold_only_minibatch(monkeypatch):
    monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))

    class SharedPolicy(CheckedPolicy):
        def evaluate_actions_with_distribution(self, states, actions, action_masks=None):
            value = self.actor.bias[0]
            logits = torch.stack([value.expand(len(states)), -value.expand(len(states)),
                                  torch.zeros(len(states))], dim=-1)
            logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
            distribution = torch.distributions.Categorical(logits=logits)
            return distribution.log_prob(actions), distribution.entropy(), value.expand(len(states)), distribution.logits

    policy = SharedPolicy()
    data = episode(policy, 2, np.array([[True, False, False], [True, True, False]]))
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, kl_probe_samples=2,
                          batch_size=1, num_epochs=1, kl_target=.001, learning_rate=50.)
    before = snapshot_optimizer_step(policy, trainer.optimizer)
    metrics = trainer.update_policy([data], [np.zeros(2)])
    assert metrics['optimizer_rejected_steps'] == 1 and metrics['optimizer_accepted_steps'] == 0
    assert metrics['last_post_update_minibatch_kl'] == 0
    assert metrics['last_post_update_probe_kl'] > trainer.kl_target * 1.5
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_nonfinite_post_step_exception_restores_model_and_adam_before_raising(device):
    policy = CheckedPolicy()
    data = episode(policy)
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=6, num_epochs=1,
                          device=device)
    before = snapshot_optimizer_step(policy, trainer.optimizer)
    original_step = trainer.optimizer.step

    def broken_step(*args, **kwargs):
        original_step(*args, **kwargs)
        with torch.no_grad():
            policy.actor.weight[0, 0] = np.nan

    trainer.optimizer.step = broken_step
    with pytest.raises((ValueError, FloatingPointError)):
        trainer.update_policy([data], [np.zeros(6)])
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))


def test_accepted_updates_keep_batch_bound_and_exact_kl_is_zero_for_hold_only_states():
    policy = CheckedPolicy()
    data = episode(policy, 5, np.tile([True, False, False], (5, 1)))
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=2, num_epochs=1,
                          learning_rate=.001, kl_probe_samples=4)
    policy.calls.clear()
    metrics = trainer.update_policy([data], [np.zeros(5)])
    assert metrics['optimizer_attempted_steps'] == metrics['optimizer_accepted_steps'] == 3
    assert metrics['optimizer_rejected_steps'] == 0
    assert metrics['last_post_update_probe_kl'] == metrics['last_post_update_minibatch_kl'] == 0.
    assert metrics['rollout_likelihood_samples'] == 5 and metrics['kl_probe_samples'] == 4
    assert max(n for n, _, _ in policy.calls) <= 2
    assert all(np.isfinite(value) for value in metrics.values())


def test_exact_kl_uses_reference_direction_and_zero_probability_mask_support():
    policy = CheckedPolicy()
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.copy_(torch.tensor([np.log(.4), np.log(.6), 0.]))
    result = exact_policy_kl(policy, np.zeros((1, 1, 2), np.float32), np.array([0]),
                             np.array([[np.log(.8), np.log(.2), -np.inf]]),
                             np.array([[True, True, False]]))
    assert result['mean_kl'] == pytest.approx(.8 * np.log(2) + .2 * np.log(1 / 3), abs=1e-7)


@pytest.mark.parametrize('checks', [True, False])
def test_rare_buy_sampled_excess_uses_exact_guard_when_available_and_legacy_fallback_otherwise(monkeypatch, checks):
    monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))

    class LegacyPolicy(CheckedPolicy):
        evaluate_actions_with_distribution = None

        def evaluate_actions(self, states, actions, action_masks=None):
            return CheckedPolicy.evaluate_actions_with_distribution(self, states, actions, action_masks)[:3]

    policy = CheckedPolicy() if checks else LegacyPolicy()
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.copy_(torch.tensor([np.log(.99), np.log(.01), 0.]))
    data = episode(policy, 2, np.tile([True, True, False], (2, 1)))
    trainer = GRPOTrainer(policy, object(), policy_update_checks=checks, batch_size=1,
                          num_epochs=1, kl_probe_samples=2, kl_target=.01)
    original_step = trainer.optimizer.step

    def drift_rare_action(*args, **kwargs):
        original_step(*args, **kwargs)
        with torch.no_grad():
            policy.actor.weight.zero_()
            policy.actor.bias.copy_(torch.tensor([np.log(.988), np.log(.012), 0.]))

    trainer.optimizer.step = drift_rare_action
    policy.calls.clear()
    metrics = trainer.update_policy([data], [np.zeros(2)])
    expected = .99 * np.log(.99 / .988) + .01 * np.log(.01 / .012)
    assert metrics['last_sampled_kl'] == pytest.approx(.2 - np.log(1.2), abs=1e-7)
    assert metrics['last_sampled_kl'] > .015 > expected
    assert metrics['optimizer_steps'] == metrics['optimizer_attempted_steps'] == (2 if checks else 1)
    assert metrics['optimizer_rejected_steps'] == 0
    assert metrics['pre_update_kl_early_stopped'] == int(not checks)
    assert metrics['pre_update_kl_uses_exact'] == metrics['pre_update_exact_kl_checked'] == int(checks)
    assert metrics['post_update_kl_early_stopped'] == 0
    assert metrics['pre_update_exact_only_exceedances'] == 0
    assert metrics['pre_update_sampled_only_exceedances'] == int(checks)
    if checks:
        assert metrics['last_checked_kl'] == pytest.approx(expected, abs=1e-7)
        assert metrics['pre_update_minibatch_exact_kl'] == pytest.approx(expected, abs=1e-7)
        assert metrics['pre_update_minibatch_max_state_kl'] == pytest.approx(expected, abs=1e-7)
        assert metrics['pre_update_minibatch_samples'] == 1
        # Two preflight, two main, four post-step probe forwards; no extra pre-guard forward.
        assert len(policy.calls) == 8
    else:
        assert metrics['last_checked_kl'] == metrics['last_sampled_kl']
        assert metrics['pre_update_minibatch_exact_kl'] == metrics['pre_update_minibatch_samples'] == 0
        assert len(policy.calls) == 2  # A legacy policy needs neither distribution API nor pre/post probes.
    assert all(size == 1 for size, _, _ in policy.calls)


def test_exact_pre_guard_stops_when_sampled_kl_underestimates_a_new_minibatch(monkeypatch):
    monkeypatch.setattr(np.random, 'permutation', lambda n: np.arange(n))
    policy = CheckedPolicy()
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.copy_(torch.tensor([np.log(.99), np.log(.01), 0.]))
    masks = np.array([[True, False, False], [True, True, False]])
    data = episode(policy, 2, masks)
    # A common HOLD draw does not reveal the tenfold increase in rare BUY probability.
    data['actions'][:] = 0
    with torch.no_grad():
        logs, _, _ = policy.evaluate_actions(torch.from_numpy(data['states']),
                                             torch.from_numpy(data['actions']), torch.from_numpy(masks))
    data['log_probs'] = logs.numpy().copy()
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=1,
                          num_epochs=1, kl_probe_samples=1, kl_target=.01)
    original_step = trainer.optimizer.step
    accepted = []

    def drift_outside_first_hold_only_batch(*args, **kwargs):
        original_step(*args, **kwargs)
        with torch.no_grad():
            policy.actor.weight.zero_()
            policy.actor.bias.copy_(torch.tensor([np.log(.9), np.log(.1), 0.]))
        accepted.append(snapshot_optimizer_step(policy, trainer.optimizer))

    trainer.optimizer.step = drift_outside_first_hold_only_batch
    policy.calls.clear()
    metrics = trainer.update_policy([data], [np.zeros(2)])
    expected = .99 * np.log(.99 / .9) + .01 * np.log(.01 / .1)
    sampled = .9 / .99 - 1 - np.log(.9 / .99)
    assert metrics['last_sampled_kl'] == pytest.approx(sampled, abs=1e-7)
    assert metrics['last_checked_kl'] == pytest.approx(expected, abs=1e-7)
    assert metrics['last_sampled_kl'] < .015 < metrics['last_checked_kl']
    assert metrics['pre_update_kl_early_stopped'] == metrics['pre_update_kl_uses_exact'] == 1
    assert metrics['pre_update_exact_only_exceedances'] == 1
    assert metrics['pre_update_sampled_only_exceedances'] == 0
    assert metrics['post_update_kl_early_stopped'] == metrics['optimizer_rejected_steps'] == 0
    assert metrics['optimizer_accepted_steps'] == metrics['optimizer_attempted_steps'] == len(accepted) == 1
    assert metrics['last_post_update_probe_kl'] == metrics['last_post_update_minibatch_kl'] == 0
    assert_state_equal(accepted[0], snapshot_optimizer_step(policy, trainer.optimizer))
    # Two preflight, two main, one post-step forward. No second optimizer step or extra pre-guard forward.
    assert len(policy.calls) == 5 and all(size == 1 for size, _, _ in policy.calls)


def test_failure_bundle_io_error_preserves_original_mismatch_and_optimizer(monkeypatch, tmp_path, caplog):
    from ai_trader.grpo import grpo
    policy = CheckedPolicy()
    data = episode(policy)
    data['log_probs'][1] += .05
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=3)
    trainer.extra_checkpoint_state = {'training_config': {'output_dir': str(tmp_path), 'training_seed': 42}}
    before = snapshot_optimizer_step(policy, trainer.optimizer)

    def cannot_write(*args, **kwargs):
        raise OSError('diagnostic disk full')

    monkeypatch.setattr(grpo, 'save_likelihood_failure', cannot_write)
    with pytest.raises(ValueError, match='Rollout likelihood mismatch'):
        trainer.update_policy([data], [np.zeros(len(data['states']))])
    assert 'diagnostic disk full' in caplog.text and 'preserving original mismatch' in caplog.text
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))
    assert not list(tmp_path.rglob('*.pt'))


def test_trainer_saves_failed_batch_and_current_weights_before_raising(tmp_path):
    from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
    policy = GRPOPolicyE2EXLSTM(obs_dim=2, cnn_channels=4, rnn_hidden_dim=4,
                                fc_hidden_dim=8, execution_action_mask=True)
    data = episode(policy, 7)
    # The first batch passes; capture the second batch, including the bad value.
    data['log_probs'][4] += .05
    trainer = GRPOTrainer(policy, object(), policy_update_checks=True, batch_size=3)
    trainer.num_updates, trainer.total_timesteps = 19, 52484
    trainer.extra_checkpoint_state = {'training_config': {
        'output_dir': str(tmp_path), 'training_seed': 42, 'num_workers': 8}}
    before = snapshot_optimizer_step(policy, trainer.optimizer)
    with pytest.raises(ValueError, match='Rollout likelihood mismatch'):
        trainer.update_policy([data], [np.zeros(7)])
    paths = list((tmp_path / 'diagnostics').glob('likelihood_update_000020_*.pt'))
    assert len(paths) == 1
    bundle = torch.load(paths[0], map_location='cpu', weights_only=True)
    assert bundle['failure']['offset'] == 3
    assert bundle['failure']['saved_batch_size'] == 3
    assert bundle['failure']['total_rollout_samples'] == 7
    assert bundle['context']['iteration'] == 20 and bundle['context']['num_updates'] == 19
    assert bundle['context']['rollout_batch_size'] == 8
    torch.testing.assert_close(bundle['batch']['states'], torch.from_numpy(data['states'][3:6]))
    torch.testing.assert_close(bundle['batch']['actions'], torch.from_numpy(data['actions'][3:6]).long())
    torch.testing.assert_close(bundle['batch']['action_masks'], torch.from_numpy(data['action_masks'][3:6]))
    assert_state_equal(before[0], bundle['policy_state_dict'])
    assert_state_equal(before, snapshot_optimizer_step(policy, trainer.optimizer))
    assert trainer.num_updates == 19


def test_update_check_configuration_restores_and_legacy_remains_disabled(tmp_path):
    trainer = GRPOTrainer(CheckedPolicy(), object(), policy_update_checks=True,
                          rollout_logprob_tolerance=.002, kl_probe_samples=7)
    filename = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(str(filename), 0)
    checkpoint = torch.load(filename, weights_only=True)
    restored = GRPOTrainer(CheckedPolicy(), object())
    restored.restore_training_progress(checkpoint)
    assert (restored.policy_update_checks, restored.rollout_logprob_tolerance, restored.kl_probe_samples) == (True, .002, 7)
    for name in ('policy_update_checks', 'rollout_logprob_tolerance', 'kl_probe_samples'):
        del checkpoint['config'][name]
    restored.restore_training_progress(checkpoint)
    assert (restored.policy_update_checks, restored.rollout_logprob_tolerance, restored.kl_probe_samples) == (False, .001, 32)
