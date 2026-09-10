"""Cached rollout predictions, update guards and persistent validation evidence."""
import copy
import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.grpo import GRPOTrainer
from test_grpo_memory_batches import RecordingPolicy, make_episode
from test_training_return_priority import make_trainer


def test_cached_rollout_values_eliminate_gae_forward_and_keep_update_equivalent():
    torch.set_num_threads(1)
    left = RecordingPolicy()
    right = copy.deepcopy(left)
    episode, values = make_episode(left, 7)
    cached = {**episode, 'values': values}
    trainers = [GRPOTrainer(policy, object(), batch_size=3, num_epochs=1)
                for policy in (left, right)]
    left.forward_batches.clear()
    right.forward_batches.clear()
    advantage = np.linspace(-.1, .1, 7, dtype=np.float32)
    np.random.seed(31)
    first = trainers[0].update_policy([cached], [advantage])
    np.random.seed(31)
    second = trainers[1].update_policy([episode], [advantage])
    assert not any(not grad for _, grad in left.forward_batches)
    assert any(not grad for _, grad in right.forward_batches)
    for name in ('policy_loss', 'value_loss', 'rollout_value_mean', 'explained_variance_rollout'):
        assert first[name] == pytest.approx(second[name], rel=1e-6, abs=1e-7)
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[name])
    assert trainers[0].reference_policy is None


def test_large_current_kl_stops_before_any_optimizer_step():
    policy = RecordingPolicy()
    episode, values = make_episode(policy, 6)
    episode.update(values=values, log_probs=episode['log_probs'] - 2)
    before = copy.deepcopy(policy.state_dict())
    trainer = GRPOTrainer(policy, object(), batch_size=3, num_epochs=2)
    metrics = trainer.update_policy([episode], [np.ones(6, np.float32)])
    assert metrics['optimizer_steps'] == 0 and metrics['kl_early_stopped'] == 1
    assert metrics['last_checked_kl'] > trainer.kl_target * 1.5
    for name, value in before.items():
        torch.testing.assert_close(value, policy.state_dict()[name])
    assert not trainer.optimizer.state


def test_invalid_cached_values_fail_instead_of_training_on_sanitized_targets():
    policy = RecordingPolicy()
    episode, values = make_episode(policy, 4)
    values[0] = np.nan
    episode['values'] = values
    trainer = GRPOTrainer(policy, object())
    with pytest.raises(ValueError, match='cached rollout values'):
        trainer.update_policy([episode], [np.ones(4, np.float32)])


def test_tied_validation_keeps_every_current_diagnostic_and_original_best(tmp_path):
    count = 0

    def evaluate(policy):
        nonlocal count
        count += 1
        return {'mean_net_return': 0.0, 'diagnostics': {
            'action_counts': {'buy': count}, 'mean_action_probabilities': {'buy': count / 10}}}

    trainer = make_trainer(evaluation_callback=evaluate)
    trainer.train(total_episodes=4, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    paths = sorted((tmp_path / 'validation').glob('iteration_*.json'))
    assert len(paths) == 2
    assert [json.loads(p.read_text())['metrics']['diagnostics']['action_counts']['buy']
            for p in paths] == [1, 2]
    best = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=False)
    assert best['iteration'] == 1
    assert best['extra_state']['validation_metrics']['diagnostics']['action_counts']['buy'] == 1


class Writer:
    def __init__(self):
        self.metrics = {}

    def add_scalar(self, name, value, iteration):
        self.metrics[name] = value


def test_policy_diagnostics_have_a_sample_budget_and_separate_interval():
    policy = RecordingPolicy()
    episodes = [make_episode(policy, 40)[0] for _ in range(4)]
    trainer = GRPOTrainer(policy, object(), batch_size=3,
                          diagnostics_max_samples=16, diagnostics_interval=5)
    trainer.writer = Writer()
    policy.forward_batches.clear()
    trainer._log_metrics(0, episodes, {0: episodes}, {})
    assert sum(n for n, _ in policy.forward_batches) == 16
    assert trainer.writer.metrics['group_0/diagnostic_samples'] == 16
    policy.forward_batches.clear()
    trainer._log_metrics(1, episodes, {0: episodes}, {})
    assert policy.forward_batches == []


def test_cached_values_are_stored_by_vector_rollout():
    import gymnasium as gym
    from ai_trader.grpo.environments import DummyVecEnv

    class Env(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, shape=(4, 2), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

        def reset(self, seed=None, options=None):
            self.n = 0
            return np.zeros((4, 2), np.float32), {'market_context': [0, 0, 0]}

        def step(self, action):
            self.n += 1
            return np.full((4, 2), self.n, np.float32), 0., self.n == 3, False, {'episode': {}}

    class Policy(RecordingPolicy):
        def get_action_with_value(self, states, deterministic=False):
            logits, values = self(states)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            return actions, dist.log_prob(actions), values.squeeze(-1)

    policy = Policy()
    env = DummyVecEnv([Env, Env])
    try:
        trainer = GRPOTrainer(policy, env)
        episodes = trainer.collect_rollouts(2)
        assert len(policy.forward_batches) == 3
        for episode in episodes:
            with torch.no_grad():
                _, expected = policy(torch.from_numpy(episode['states']))
            np.testing.assert_allclose(episode['values'], expected.numpy().reshape(-1), atol=1e-7)
    finally:
        env.close()


def test_return_priority_notebook_uses_new_schema_width_and_preserves_old_resume():
    from pathlib import Path
    from lib.observations import ObservationBuilder
    from ai_trader.grpo.train_xlstm import TrainingConfig

    notebook = Path(__file__).resolve().parents[1] / 'ai_trader/grpo/colab_train_xlstm_return_priority.ipynb'
    cells = {cell['id']: ''.join(cell.get('source', []))
             for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']}
    scope = {'GiB': 1024 ** 3, 'ObservationBuilder': ObservationBuilder}
    exec(compile(cells['profile-functions'], str(notebook), 'exec'), scope)
    defaults = vars(TrainingConfig()).copy()
    assert defaults['account_observations'] and defaults['liquidation_max_steps'] == 300
    assert scope['model_obs_dim'](defaults) == 53
    assert scope['model_obs_dim']({**defaults, 'account_observations': False}) == 42
    profile = scope['a100_profile'](40, 38, 50, 12)
    assert profile['account_observations']
    assert scope['estimated_host_gib'](profile) < 35
    legacy = {k: v for k, v in defaults.items()
              if k not in ('account_observations', 'liquidation_max_steps')}
    schema = ObservationBuilder(['현재가', '등락률']).schema
    checkpoint = {'observation_schema': schema, 'extra_state': {
        'training_config': legacy, 'date_splits': {'train': ['20260101']}},
        'optimizer_state_dict': {}, 'total_timesteps': 0, 'num_updates': 0, 'iteration': 0}
    restored = scope['restore_run_config'](defaults, checkpoint, 'resume')
    assert not restored['account_observations'] and restored['liquidation_max_steps'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_cached_update_preserves_mask_likelihood_and_finite_gradients():
    from test_account_observations import builder, account, tiny_policy

    torch.manual_seed(17)
    spec = builder()
    policy = tiny_policy(spec).cuda().eval()
    raw = np.arange(100., 108.).reshape(-1, 1)
    flat = spec.build(raw, account_state=account())
    held = spec.build(raw, [{'entry_price': 100., 'entry_time_seconds': 0.}],
                      107., 1., account_state=account(5, 107.))
    states = torch.from_numpy(np.stack([flat, held, flat, held])).cuda()
    with torch.no_grad():
        actions, log_probs, values = policy.get_action_with_value(states)
    policy.train()
    new_logs, _, _ = policy.evaluate_actions(states, actions)
    torch.testing.assert_close(new_logs, log_probs, rtol=1e-5, atol=1e-6)
    assert (actions[[0, 2]] != 2).all() and (actions[[1, 3]] != 1).all()
    episode = {'states': states.cpu().numpy(), 'actions': actions.cpu().numpy(),
               'log_probs': log_probs.cpu().numpy(), 'values': values.cpu().numpy(),
               'rewards': np.array([-.1, .02, -.05, .08], np.float32),
               'dones': np.array([False, False, False, True])}
    trainer = GRPOTrainer(policy, object(), batch_size=2, num_epochs=1, device='cuda')
    before = policy.policy_head.weight.detach().clone()
    metrics = trainer.update_policy([episode], [np.zeros(4, np.float32)])
    assert metrics['optimizer_steps'] == 2
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, policy.policy_head.weight)
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
