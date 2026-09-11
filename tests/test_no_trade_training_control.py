"""No-trade collapse may stop learning; only audited trading creates profit candidates."""
import numpy as np
import pytest
import torch

from ai_trader.grpo.training_control import TrainingControlState, profitable_candidate_evidence
from test_training_return_priority import make_trainer


def validation(*, buy=.4, net_return=0., round_trips=0, dates=('20260901', '20260902', '20260903')):
    episodes = []
    for date in dates:
        episodes.append({'episode_key': {'date': date}, 'filled_quantity': round_trips * 2,
                         'round_trip_count': round_trips, 'liquidation_complete': True,
                         'open_quantity': 0})
    return {'mean_net_return': net_return, 'no_trade_episode_fraction': 0. if round_trips else 1.,
            'num_episodes': len(episodes), 'round_trip_count': round_trips * len(episodes),
            'incomplete_liquidation_episodes': 0, 'max_open_quantity': 0,
            'diagnostics': {'filled_quantity': sum(ep['filled_quantity'] for ep in episodes),
                            'mean_action_probabilities': {'buy': buy}, 'episodes': episodes}}


@pytest.mark.parametrize('name,value', [('no_trade_patience', True), ('no_trade_patience', -1),
                                      ('no_trade_patience', 3.), ('no_trade_patience', np.nan),
                                      ('profitable_min_round_trips', 0), ('profitable_min_round_trips', False),
                                      ('profitable_min_traded_dates', np.inf), ('profitable_min_traded_dates', '3')])
def test_invalid_control_settings_fail(name, value):
    with pytest.raises(ValueError, match=name):
        make_trainer(evaluation_callback=lambda _: validation(), **{name: value})


def test_no_trade_stops_after_nonimproving_probability_trend_and_keeps_cash_best(tmp_path):
    calls = []

    def evaluate(_):
        calls.append(len(calls))
        return validation(buy=.5 - len(calls) * .05)

    trainer = make_trainer(no_trade_patience=3, evaluation_callback=evaluate)
    result = trainer.train(total_episodes=20, checkpoint_interval=100,
                           checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert len(calls) == 4  # First zero-return validation improves the unset baseline.
    assert result['early_stop_reason'] == 'no_trade_collapse'
    assert result['last_completed_iteration'] == result['iterations_completed'] == 4
    assert result['total_timesteps'] == 16 and result['num_updates'] == 4
    saved = torch.load(tmp_path / 'checkpoint_iter4.pt', weights_only=True)
    assert saved['training_control_state']['no_trade_streak'] == 3
    assert saved['training_control_state']['stop_reason'] == 'no_trade_collapse'
    best = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    assert best['iteration'] == 1 and best['extra_state']['best_validation_return'] == 0.
    assert not (tmp_path / 'checkpoint_best_profitable.pt').exists()


def test_default_zero_patience_never_stops_a_verified_no_trade_run():
    trainer = make_trainer(evaluation_callback=lambda _: validation())
    result = trainer.train(total_episodes=12)
    assert result == {'total_timesteps': 24, 'num_updates': 6}
    assert trainer.training_control_state.no_trade_streak == 5


def test_unknown_or_increasing_probability_breaks_consecutive_collapse_evidence():
    state = TrainingControlState()
    observe = lambda metrics, improved=False: state.observe(metrics, iteration=state.validation_count + 1,
                                                           score_improved=improved, patience=2)
    assert not observe(validation(buy=.4), True)
    assert not observe(validation(buy=.3))
    unknown = validation(buy=.2)
    del unknown['diagnostics']['filled_quantity']
    assert not observe(unknown)
    assert state.no_trade_streak == 0 and state.previous_buy_probability is None
    assert not observe(validation(buy=.2))
    assert not observe(validation(buy=.3))  # Increased entry interest breaks the trend.
    assert not observe(validation(buy=.2))
    assert observe(validation(buy=.1))
    assert state.validation_count == 7 and state.no_trade_evidence_count == 6


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(no_trade_episode_fraction=True),
    lambda m: m.update(no_trade_episode_fraction=np.nan),
    lambda m: m.update(num_episodes=False),
    lambda m: m['diagnostics'].update(filled_quantity=False),
    lambda m: m['diagnostics']['mean_action_probabilities'].update(buy=np.nan),
    lambda m: m['diagnostics']['mean_action_probabilities'].update(buy=True),
    lambda m: m['diagnostics']['episodes'][0].update(filled_quantity=1),
])
def test_malformed_no_trade_evidence_cannot_stop(mutation):
    state = TrainingControlState()
    state.observe(validation(), iteration=1, score_improved=True, patience=1)
    metrics = validation(buy=.2)
    mutation(metrics)
    assert not state.observe(metrics, iteration=2, score_improved=False, patience=1)
    assert state.validation_count == 2 and state.no_trade_streak == 0


def test_resume_keeps_streak_without_counting_source_revalidation_twice(tmp_path):
    first_values = iter([validation(buy=.4), validation(buy=.3)])
    first = make_trainer(no_trade_patience=3, evaluation_callback=lambda _: next(first_values))
    path = str(tmp_path / 'checkpoint_iter{}.pt')
    first.train(total_episodes=4, checkpoint_interval=1, checkpoint_path=path)
    checkpoint = torch.load(tmp_path / 'checkpoint_iter2.pt', weights_only=True)
    assert checkpoint['training_control_state']['no_trade_streak'] == 1
    resumed_values = iter([validation(buy=.29), validation(buy=.28), validation(buy=.27)])
    resumed = make_trainer(no_trade_patience=99, evaluation_callback=lambda _: next(resumed_values))
    start = resumed.load_checkpoint(str(tmp_path / 'checkpoint_iter2.pt'))
    assert resumed.no_trade_patience == 3
    result = resumed.train(total_episodes=20, start_iteration=2, resume=True, checkpoint_path=path)
    assert result['iterations_completed'] == 2 and result['last_completed_iteration'] == 4
    assert result['training_control']['validation_count'] == 4
    assert result['total_timesteps'] == 16 and result['num_updates'] == 4


def test_revert_preserves_observed_counters_and_budget(tmp_path):
    trainer = make_trainer(no_trade_patience=2, evaluation_callback=lambda _: validation())
    result = trainer.train(total_episodes=20, revert_to_best_patience=1,
                           checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert result['early_stop_reason'] == 'no_trade_collapse'
    assert result['total_timesteps'] == 12 and result['num_updates'] == 3
    assert result['training_control']['validation_count'] == 3
    assert result['training_control']['no_trade_streak'] == 2


def test_revert_to_imported_best_does_not_undo_explicit_control_settings(tmp_path):
    trainer = make_trainer(no_trade_patience=2, profitable_min_round_trips=4,
                           profitable_min_traded_dates=2, evaluation_callback=lambda _: validation())
    filename = tmp_path / 'imported.pt'
    trainer.save_checkpoint(str(filename), 0)
    imported = torch.load(filename, weights_only=True)
    imported['config'].update(no_trade_patience=0, profitable_min_round_trips=99,
                              profitable_min_traded_dates=99)
    result = trainer.train(total_episodes=20, resume=True, resume_best_checkpoints=[imported],
                           revert_to_best_patience=1, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert result['early_stop_reason'] == 'no_trade_collapse'
    assert (trainer.no_trade_patience, trainer.profitable_min_round_trips, trainer.profitable_min_traded_dates) == (2, 4, 2)


def test_validation_failure_retains_prevalidation_recovery_checkpoint(tmp_path):
    def fail(_):
        raise RuntimeError('validation failed')

    trainer = make_trainer(evaluation_callback=fail)
    with pytest.raises(RuntimeError, match='validation failed'):
        trainer.train(total_episodes=2, checkpoint_interval=1,
                      checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    checkpoint = torch.load(tmp_path / 'checkpoint_iter1.pt', weights_only=True)
    assert checkpoint['num_updates'] == 1 and checkpoint['total_timesteps'] == 4
    assert checkpoint['training_control_state']['validation_count'] == 0


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(mean_net_return=0.),
    lambda m: m.update(mean_net_return=True),
    lambda m: m.update(mean_net_return=np.nan),
    lambda m: m.update(incomplete_liquidation_episodes=False),
    lambda m: m.update(max_open_quantity=1),
    lambda m: m.update(round_trip_count=99),
    lambda m: m['diagnostics'].update(filled_quantity=False),
    lambda m: m['diagnostics']['episodes'][0].update(liquidation_complete=1),
    lambda m: m['diagnostics']['episodes'][0]['episode_key'].update(date='20269999'),
    lambda m: m['diagnostics']['episodes'][0].pop('liquidation_complete'),
    lambda m: m['diagnostics']['episodes'].pop(),
])
def test_invalid_or_contradictory_profit_evidence_fails_closed(mutation):
    metrics = validation(net_return=1., round_trips=7)
    mutation(metrics)
    assert not profitable_candidate_evidence(metrics, 20, 3)['eligible']


def test_profit_candidate_requires_completed_trades_on_distinct_actual_dates():
    metrics = validation(net_return=1., round_trips=7)
    assert profitable_candidate_evidence(metrics, 20, 3)['eligible']
    assert not profitable_candidate_evidence(metrics, 22, 3)['eligible']
    for episode in metrics['diagnostics']['episodes']:
        episode['episode_key']['date'] = '20260901'
    evidence = profitable_candidate_evidence(metrics, 20, 3)
    assert not evidence['eligible'] and evidence['reason'] == 'insufficient_traded_dates'


def test_actual_replay_evaluation_exposes_all_profit_candidate_evidence():
    from ai_trader.grpo.evaluation import evaluate_policy
    from test_scalping_accounting import ReplayFixture

    class DatedReplay(ReplayFixture):
        def reset(self, seed=None, options=None):
            self.replay_date = f'202609{1 + seed % 3:02d}'
            return super().reset(seed=seed, options=options)

        def _sample_episode_start(self, max_attempts=100):
            features, metadata = super()._sample_episode_start(max_attempts)
            metadata[:, 1] = self.replay_date
            self.episode_key = {'date': self.replay_date, 'stock_code': 'TEST'}
            return features, metadata

    class EnterAndHold(torch.nn.Module):
        def get_action(self, observations, deterministic=False):
            actions, logs, _ = self.get_action_with_probabilities(observations, deterministic)
            return actions, logs

        def get_action_with_probabilities(self, observations, deterministic=False):
            held = observations[:, -1, -15] > .5
            actions = torch.where(held, 0, 1)
            probabilities = torch.nn.functional.one_hot(actions, 3).float()
            return actions, torch.zeros(len(observations)), probabilities

    env = DatedReplay([100, 100, 110, 110], max_stages=1, initial_cash=1000,
                      transaction_cost_rate=.001, sell_tax_rate=.002)
    try:
        metrics = evaluate_policy(EnterAndHold(), env, num_episodes=3, seed=0)
        assert metrics['mean_net_return'] > 0
        evidence = profitable_candidate_evidence(metrics, 3, 3)
        assert evidence['eligible'], evidence
        assert evidence['round_trip_count'] == 3 and evidence['traded_date_count'] == 3
        assert metrics['diagnostics']['episodes'][0]['liquidation_complete'] is True
        without_diagnostics = evaluate_policy(EnterAndHold(), env, num_episodes=3, seed=0,
                                               collect_diagnostics=False)
        assert not profitable_candidate_evidence(without_diagnostics, 3, 3)['eligible']
    finally:
        env.close()


def test_cash_and_sparse_positive_results_do_not_overwrite_qualified_profit_checkpoint(tmp_path):
    values = iter([validation(), validation(net_return=1., round_trips=7),
                   validation(net_return=2., round_trips=1), validation()])
    trainer = make_trainer(evaluation_callback=lambda _: next(values))
    trainer.train(total_episodes=8, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    best = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    profit = torch.load(tmp_path / 'checkpoint_best_profitable.pt', weights_only=True)
    assert best['iteration'] == 3 and best['extra_state']['best_validation_return'] == 2.
    assert profit['iteration'] == 2
    assert profit['extra_state']['best_profitable_validation_return'] == 1.
    assert profit['extra_state']['profitable_candidate_evidence']['round_trip_count'] == 21


def test_resume_revalidates_profit_candidate_separately_from_best_return(tmp_path):
    def evaluate(policy):
        weight = float(policy.weight.item())
        if weight == 9:
            return validation(net_return=1., round_trips=7)
        if weight == 10:
            return validation(net_return=2., round_trips=1)
        return validation()

    trainer = make_trainer(evaluation_callback=evaluate)
    candidates = []
    for iteration, weight in ((7, 9.), (8, 10.)):
        trainer.policy.weight.data.fill_(weight)
        filename = tmp_path / f'candidate{iteration}.pt'
        trainer.save_checkpoint(str(filename), iteration)
        candidates.append(torch.load(filename, weights_only=True))
    trainer.policy.weight.data.fill_(5.)
    trainer.train(total_episodes=2, resume=True, start_iteration=10,
                  resume_best_checkpoints=candidates,
                  checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    best = torch.load(tmp_path / 'checkpoint_best.pt', weights_only=True)
    profit = torch.load(tmp_path / 'checkpoint_best_profitable.pt', weights_only=True)
    assert best['policy_state_dict']['weight'].item() == 10.
    assert profit['policy_state_dict']['weight'].item() == 9.
    assert profit['iteration'] == 7
    assert trainer.policy.weight.item() == 4.


@pytest.mark.parametrize('mutation', [lambda state: state.update(no_trade_streak=True),
                                    lambda state: state.update(previous_buy_probability=np.nan),
                                    lambda state: state.update(no_trade_streak=9),
                                    lambda state: state.update(stop_reason='unknown')])
def test_malformed_control_checkpoint_is_rejected(mutation):
    state = TrainingControlState().to_dict()
    mutation(state)
    with pytest.raises(ValueError):
        TrainingControlState.from_dict(state)


def test_legacy_checkpoint_has_disabled_stop_and_empty_control_state(tmp_path):
    trainer = make_trainer(evaluation_callback=lambda _: validation(), no_trade_patience=3)
    filename = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(str(filename), 0)
    checkpoint = torch.load(filename, weights_only=True)
    del checkpoint['training_control_state']
    del checkpoint['config']['no_trade_patience']
    trainer.restore_training_progress(checkpoint)
    assert trainer.no_trade_patience == 0
    assert trainer.training_control_state == TrainingControlState()


@pytest.mark.parametrize('resume', [False, True])
def test_unqualified_prior_profit_checkpoint_is_archived_without_overwriting_artifacts(tmp_path, resume):
    trainer = make_trainer(evaluation_callback=lambda _: validation())
    prior = tmp_path / 'checkpoint_best_profitable.pt'
    trainer.save_checkpoint(str(prior), 0, extra_state={'best_profitable_validation_return': 99.})
    prior_bytes = prior.read_bytes()
    candidate = torch.load(prior, weights_only=True)
    collision = tmp_path / 'checkpoint_unqualified_profitable_0.pt'
    collision.write_bytes(b'pre-existing archived artifact')
    trainer.train(total_episodes=2, resume=resume, resume_best_checkpoints=[candidate],
                  checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert not prior.exists()
    assert collision.read_bytes() == b'pre-existing archived artifact'
    assert (tmp_path / 'checkpoint_unqualified_profitable_0_1.pt').read_bytes() == prior_bytes
    assert (tmp_path / 'checkpoint_best.pt').exists()  # The verified cash baseline remains selectable.


@pytest.mark.parametrize('saved_mask', [True, 0, 'false'])
def test_progress_restore_validates_execution_mask_before_restoring_optimizer(tmp_path, saved_mask):
    trainer = make_trainer()
    filename = tmp_path / 'checkpoint.pt'
    trainer.save_checkpoint(str(filename), 0)
    checkpoint = torch.load(filename, weights_only=True)
    checkpoint['config']['execution_action_mask'] = saved_mask
    with pytest.raises(ValueError, match='execution_action_mask'):
        trainer.restore_training_progress(checkpoint)
