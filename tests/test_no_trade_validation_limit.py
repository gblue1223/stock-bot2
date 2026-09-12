"""A verified no-trade budget is independent of changing action probabilities."""
import copy

import numpy as np
import pytest
import torch

from ai_trader.grpo.training_control import TrainingControlState
from test_no_trade_training_control import validation
from test_training_return_priority import make_trainer


def observe(state, metrics, *, limit=4, patience=0, improved=False):
    return state.observe(metrics, iteration=state.validation_count + 1,
                         score_improved=improved, patience=patience, max_validations=limit)


def test_limit_counts_first_validation_and_ignores_increasing_buy_probability():
    state = TrainingControlState()
    for index, buy in enumerate([.1, .2, .3, .4], 1):
        assert observe(state, validation(buy=buy), improved=index == 1) is (index == 4)
        assert state.consecutive_no_trade_validations == index
        assert state.no_trade_streak == 0
    assert state.stop_reason == 'no_trade_limit'


def test_limit_does_not_require_action_probabilities_but_collapse_still_does():
    state = TrainingControlState()
    metrics = validation()
    del metrics['diagnostics']['mean_action_probabilities']
    assert not observe(state, metrics, limit=2, patience=1)
    assert observe(state, metrics, limit=2, patience=1)
    assert state.stop_reason == 'no_trade_limit'
    assert state.no_trade_evidence_count == state.no_trade_streak == 0
    assert state.previous_buy_probability is None
    assert TrainingControlState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(no_trade_episode_fraction=.5),
    lambda m: m.update(mean_net_return=np.nan),
    lambda m: m.update(mean_net_return=.01),
    lambda m: m.update(incomplete_liquidation_episodes=1),
    lambda m: m.pop('incomplete_liquidation_episodes'),
    lambda m: m.update(max_open_quantity=1),
    lambda m: m.update(num_episodes=0),
    lambda m: m['diagnostics'].pop('filled_quantity'),
    lambda m: m['diagnostics'].update(filled_quantity=1),
    lambda m: m['diagnostics']['episodes'][0].update(filled_quantity=1),
    lambda m: m['diagnostics']['episodes'][0].update(open_quantity=1),
    lambda m: m['diagnostics']['episodes'][0].update(liquidation_complete=False),
    lambda m: m['diagnostics']['episodes'].pop(),
])
def test_trade_or_incomplete_evidence_breaks_the_limit_streak(mutation):
    state = TrainingControlState()
    assert not observe(state, validation(), limit=2)
    metrics = validation()
    mutation(metrics)
    assert not observe(state, metrics, limit=2)
    assert state.consecutive_no_trade_validations == 0
    assert not observe(state, validation(), limit=2)
    assert state.consecutive_no_trade_validations == 1


def test_collapse_reason_has_priority_when_both_stop_thresholds_are_met():
    state = TrainingControlState()
    assert not observe(state, validation(buy=.4), limit=2, patience=1, improved=True)
    assert observe(state, validation(buy=.3), limit=2, patience=1)
    assert state.stop_reason == 'no_trade_collapse'
    assert state.consecutive_no_trade_validations == 2 and state.no_trade_streak == 1


@pytest.mark.parametrize('location', ['aggregate', 'episode'])
@pytest.mark.parametrize('value', [1, -1, np.nan, np.inf, True, None, '0', .5])
def test_present_round_trip_evidence_must_be_valid_zero(location, value):
    state = TrainingControlState()
    assert not observe(state, validation(), limit=2)
    metrics = validation()
    target = metrics if location == 'aggregate' else metrics['diagnostics']['episodes'][0]
    target['round_trip_count'] = value
    assert not observe(state, metrics, limit=2, patience=1)
    assert state.consecutive_no_trade_validations == state.no_trade_streak == 0
    assert state.previous_buy_probability is None


def test_legacy_missing_round_trip_counts_do_not_invalidate_verified_no_trade_evidence():
    metrics = validation()
    del metrics['round_trip_count']
    for episode in metrics['diagnostics']['episodes']:
        del episode['round_trip_count']
    state = TrainingControlState()
    assert observe(state, metrics, limit=1)
    assert state.stop_reason == 'no_trade_limit'


def test_zero_limit_is_disabled_without_changing_existing_collapse_counters():
    state = TrainingControlState()
    for _ in range(8):
        assert not observe(state, validation(), limit=0)
    assert state.consecutive_no_trade_validations == 8 and state.no_trade_streak == 7


def test_legacy_exact_six_field_checkpoint_restores_only_new_counter_to_zero():
    state = TrainingControlState(validation_count=3, no_trade_evidence_count=3,
                                 no_trade_streak=2, previous_buy_probability=.3,
                                 last_validation_iteration=15, stop_reason='no_trade_collapse')
    legacy = state.to_dict()
    del legacy['consecutive_no_trade_validations']
    before = copy.deepcopy(legacy)
    restored = TrainingControlState.from_dict(legacy)
    assert restored == state
    assert legacy == before


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(unrecognized=0),
    lambda s: s.pop('no_trade_streak'),
    lambda s: s.update(consecutive_no_trade_validations=True),
    lambda s: s.update(consecutive_no_trade_validations=-1),
    lambda s: s.update(consecutive_no_trade_validations=1.5),
    lambda s: s.update(consecutive_no_trade_validations=1),
])
def test_invalid_or_unknown_checkpoint_fields_are_rejected(mutation):
    saved = TrainingControlState().to_dict()
    mutation(saved)
    with pytest.raises(ValueError):
        TrainingControlState.from_dict(saved)


def test_training_stops_on_fourth_no_trade_validation_with_increasing_buy_probabilities(tmp_path):
    values = iter([validation(buy=value) for value in [.1, .2, .3, .4]])
    trainer = make_trainer(evaluation_callback=lambda _: next(values), no_trade_max_validations=4)
    result = trainer.train(total_episodes=20, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert result['early_stop_reason'] == 'no_trade_limit'
    assert result['last_completed_iteration'] == 4
    checkpoint = torch.load(tmp_path / 'checkpoint_iter4.pt', weights_only=True)
    assert checkpoint['config']['no_trade_max_validations'] == 4
    assert checkpoint['training_control_state']['consecutive_no_trade_validations'] == 4
    assert checkpoint['training_control_state']['stop_reason'] == 'no_trade_limit'


def test_limit_resume_ignores_source_revalidation_and_preserves_revert_progress(tmp_path):
    source = make_trainer(evaluation_callback=lambda _: validation(), no_trade_max_validations=4)
    pattern = str(tmp_path / 'checkpoint_iter{}.pt')
    source.train(total_episodes=4, checkpoint_interval=1, checkpoint_path=pattern)
    restored = make_trainer(evaluation_callback=lambda _: validation(), no_trade_max_validations=99)
    restored.load_checkpoint(str(tmp_path / 'checkpoint_iter2.pt'))
    assert restored.no_trade_max_validations == 4
    assert restored.training_control_state.consecutive_no_trade_validations == 2
    result = restored.train(total_episodes=20, start_iteration=2, resume=True,
                            revert_to_best_patience=1, checkpoint_path=pattern)
    assert result['early_stop_reason'] == 'no_trade_limit'
    assert result['last_completed_iteration'] == 4 and result['iterations_completed'] == 2
    assert result['training_control']['consecutive_no_trade_validations'] == 4
    assert result['training_control']['validation_count'] == 4


def test_limit_revert_does_not_restore_disabled_setting_from_imported_best(tmp_path):
    trainer = make_trainer(evaluation_callback=lambda _: validation(), no_trade_max_validations=3)
    filename = tmp_path / 'source.pt'
    trainer.save_checkpoint(str(filename), 0)
    imported = torch.load(filename, weights_only=True)
    imported['config']['no_trade_max_validations'] = 0
    result = trainer.train(total_episodes=20, resume=True, resume_best_checkpoints=[imported],
                           revert_to_best_patience=1, checkpoint_path=str(tmp_path / 'checkpoint_iter{}.pt'))
    assert result['early_stop_reason'] == 'no_trade_limit'
    assert result['training_control']['validation_count'] == 3
    assert trainer.no_trade_max_validations == 3
