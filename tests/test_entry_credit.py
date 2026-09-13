"""Realized entry attribution must not change or invent frozen learning targets."""
from copy import deepcopy
import json

import numpy as np
import pytest

from ai_trader.grpo.entry_credit import analyze_entry_credit


def entry(order_id=1, index=0, *, net=2., filled=5, sold=5, submitted=5, complete=True):
    # Entry fees may include unmatched quantity; only matched_fees explains realized net PnL.
    return dict(entry_order_id=order_id, decision_index=index, submitted_quantity=submitted,
                filled_quantity=filled, sold_quantity=sold, open_quantity=filled - sold,
                order_status='filled', complete=complete, net_pnl=net, mid_price_pnl=net + 1.,
                spread_cost=.2, depth_cost=.1, slippage_cost=.3, matched_fees=.4,
                pnl_attribution_residual=0., quantity_weighted_holding_time=20. if sold else 0.,
                exit_reason='policy' if sold else '', entry_fees=100., exit_fees=.2)


def episode(rewards, values, *, actions=None, entries=None, blocked=None):
    n = len(rewards)
    actions = np.asarray(actions if actions is not None else [1] + [0] * (n - 2) + [2], dtype=np.int64)
    result = dict(actions=actions, rewards=np.asarray(rewards, dtype=np.float64),
                  values=np.asarray(values, dtype=np.float32), dones=np.array([False] * (n - 1) + [True]))
    if entries is not None:
        order_ids = {row['decision_index']: row['entry_order_id'] for row in entries}
        decisions = [dict(decision_index=int(index), entry_order_id=order_ids.get(index),
                          outcome='submitted' if index in order_ids else blocked or 'position_limit')
                     for index in np.flatnonzero(actions == 1)]
        result['metadata'] = {'entry_diagnostics': {'version': 1, 'decision_count': n,
                                                  'buy_decisions': decisions, 'entries': entries}}
    return result


def analyze(episodes, **overrides):
    # Supply exactly the trainer-computed recurrence and population normalization.
    raw, targets = [], []
    for ep in episodes:
        gae = np.zeros(len(ep['actions']), dtype=np.float32)
        previous = 0.
        for index in reversed(range(len(gae))):
            terminal = bool(ep['dones'][index])
            next_value = ep['values'][index + 1] if index + 1 < len(gae) and not terminal else 0.
            delta = ep['rewards'][index] + next_value - ep['values'][index]
            previous = delta + .95 * (not terminal) * previous
            gae[index] = previous
        raw.append(gae)
        targets.append(gae + ep['values'])
    raw = np.concatenate(raw)
    kwargs = dict(raw_gae=raw, group_component=np.zeros_like(raw), pre_normalized=raw,
                  normalized=(raw - raw.mean()) / (raw.std() + 1e-8), returns=np.concatenate(targets),
                  cached_values=np.concatenate([ep['values'] for ep in episodes]), gamma=1., lambda_gae=.95)
    kwargs.update(overrides)
    return analyze_entry_credit(episodes, **kwargs)


def test_profitable_negative_and_losing_positive_credit_are_not_conflated():
    episodes = [episode([-.3, .1, .5], [1., 0., 0.], entries=[entry(net=2.)]),
                episode([-.3, 0., -.5], [-3., 0., 0.], entries=[entry(net=-2.)])]
    report = analyze(episodes)
    profit, loss = report['entries']
    assert profit['outcome'] == 'profitable' and profit['raw_gae'] < 0
    assert profit['normalized_advantage'] < 0
    assert loss['outcome'] == 'loss' and loss['raw_gae'] > 0
    assert loss['normalized_advantage'] > 0
    metrics = report['metrics']
    assert metrics['outcome/profitable/raw_gae_negative_count'] == 1
    assert metrics['outcome/profitable/normalized_advantage_negative_fraction'] == 1.
    assert metrics['outcome/loss/raw_gae_negative_fraction'] == 0.
    assert report['outcome_summary']['profitable']['exit_reason_counts'] == {'policy': 1}
    assert metrics['outcome/profitable/holding_time_mean'] == 20.
    assert profit['computed_pnl_attribution_residual'] == pytest.approx(0.)
    assert profit['entry_fees'] == 100.  # These are not incorrectly deducted again.
    assert all(np.isfinite(value) for value in metrics.values())
    json.dumps(report, allow_nan=False)


def test_delayed_credit_decomposes_supplied_gae_and_respects_terminals_and_boundaries():
    first = episode([-.3, .1, .5], [1., 0., 0.], entries=[entry()])
    second = episode([6., 5.], [5., 3.], entries=[])
    second['dones'][:] = True  # Never take a next-value or next-GAE across a terminal.
    report = analyze([first, second])
    buy = report['entries'][0]
    assert buy['immediate_reward'] == -.3
    assert buy['bootstrap_value_delta'] == -1.
    assert buy['future_td_trace'] == pytest.approx(.95 * (.1 + .95 * .5), abs=1e-7)
    assert buy['raw_gae'] == pytest.approx(buy['immediate_td_error'] + buy['future_td_trace'])
    assert abs(buy['gae_trace_consistency_residual']) < 1e-7
    blocked = report['buy_decisions'][1]
    assert blocked['future_td_trace'] == 0. and blocked['bootstrap_value_delta'] == -5.


def test_unfilled_incomplete_blocked_and_partial_completed_entries_are_separate():
    rows = [entry(1, 0, net=0., filled=0, sold=0, complete=False),
            entry(2, 1, net=4., filled=5, sold=2, complete=False),
            entry(3, 2, net=1., filled=2, sold=2, submitted=5)]
    ep = episode([-.2, -.1, -.1, 0., .5], np.zeros(5), actions=[1, 1, 1, 1, 2], entries=rows)
    report = analyze([ep])
    assert [row['outcome'] for row in report['entries']] == ['unfilled', 'incomplete', 'profitable']
    assert not report['entries'][0]['realized_pnl_available']
    assert report['entries'][1]['realized_pnl_available']
    assert not report['entries'][1]['complete_pnl_available']
    assert report['entries'][2]['partial_entry_fill']
    assert report['metrics']['complete_entry_count'] == 1
    assert report['metrics']['blocked_buy_decision_count'] == 1
    assert report['metrics']['outcome/profitable/sample_count'] == 1
    assert report['metrics']['outcome/profitable/complete_net_pnl_mean'] == 1.
    assert report['metrics']['outcome/incomplete/complete_net_pnl_available'] == 0
    assert report['metrics']['outcome/incomplete/realized_components_available'] == 1
    assert report['metrics']['outcome/incomplete/net_pnl_sum'] == 4.
    assert report['metrics']['outcome/unfilled/realized_components_available'] == 0
    assert report['metrics']['outcome/unfilled/mid_price_pnl_sum'] == 0.


def test_legacy_trace_absence_and_partial_coverage_do_not_invent_zero_profit():
    legacy = episode([-.1, .2], [0., 0.])
    report = analyze([legacy])
    assert report['availability']['reason'] == 'entry_trace_unavailable'
    assert report['entries'] == []
    assert report['metrics']['outcome/profitable/complete_net_pnl_available'] == 0
    assert report['metrics']['action/buy/sample_count'] == 1
    current = episode([-.1, .2], [0., 0.], entries=[])
    mixed = analyze([legacy, current])
    assert mixed['availability']['reason'] == 'partial_legacy_metadata'
    assert mixed['metrics']['entry_attribution_complete'] == 0
    assert mixed['metrics']['available_episode_count'] == mixed['metrics']['missing_episode_count'] == 1


def test_non_gae_training_and_all_blocked_buys_remain_supported():
    ep = episode([-.1, .2], [0., 0.], entries=[])
    raw = np.array([-1., 1.], dtype=np.float32)
    del ep['dones']
    report = analyze_entry_credit([ep], raw_gae=None, group_component=raw, pre_normalized=raw,
                                  normalized=raw, returns=raw, cached_values=None, gamma=1., lambda_gae=.95)
    assert report['availability']['entry_attribution_complete']
    assert report['metrics']['entry_count'] == 0
    assert report['metrics']['blocked_buy_decision_count'] == 1
    assert report['metrics']['action/buy/raw_gae_available'] == 0
    assert report['buy_decisions'][0]['future_td_trace'] is None
    assert report['metrics']['outcome/profitable/normalized_advantage_available'] == 0


@pytest.mark.parametrize('mutate', [
    lambda t: t.update(version=2),
    lambda t: t.update(decision_count=3),
    lambda t: t.update(buy_decisions=[]),
    lambda t: t['buy_decisions'].append(deepcopy(t['buy_decisions'][0])),
    lambda t: t['entries'][0].update(entry_order_id=99),
    lambda t: t['entries'][0].update(complete=1),
    lambda t: t['entries'][0].update(open_quantity=1),
    lambda t: t['entries'][0].update(net_pnl=float('nan')),
    lambda t: t['entries'][0].update(entry_order_id=True),
    lambda t: t['entries'][0].update(filled_quantity=0, sold_quantity=0, complete=True),
])
def test_present_malformed_trace_fails_loudly(mutate):
    ep = episode([-.1, .2], [0., 0.], entries=[entry()])
    mutate(ep['metadata']['entry_diagnostics'])
    with pytest.raises(ValueError, match='entry credit'):
        analyze([ep])


def test_present_null_trace_is_not_silently_legacy_and_array_shapes_are_checked():
    ep = episode([-.1, .2], [0., 0.])
    ep['metadata'] = {'entry_diagnostics': None}
    with pytest.raises(ValueError, match='version 1'):
        analyze([ep])
    del ep['metadata']
    with pytest.raises(ValueError, match='raw_gae'):
        analyze([ep], raw_gae=np.zeros(3))


def test_metadata_and_training_arrays_are_unchanged_and_observations_are_never_read():
    class UnreadableObservations:
        def __array__(self, *args, **kwargs):
            raise AssertionError('entry diagnostics must not read observations')

    ep = episode([-.1, .2], [0., 0.], entries=[entry()])
    original = deepcopy(ep)
    ep['states'] = UnreadableObservations()
    for value in ep.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    analyze([ep])
    assert ep['metadata'] == original['metadata']
    for name in ('actions', 'rewards', 'values', 'dones'):
        np.testing.assert_array_equal(ep[name], original[name])


def test_accounting_residual_is_reported_without_rewriting_source_attribution():
    row = entry()
    row['mid_price_pnl'] += .5
    ep = episode([-.1, .2], [0., 0.], entries=[row])
    report = analyze([ep])
    assert report['entries'][0]['computed_pnl_attribution_residual'] == pytest.approx(-.5)
    assert report['entries'][0]['recorded_residual_difference'] == pytest.approx(-.5)
    assert report['metrics']['outcome/profitable/pnl_attribution_residual_max_abs'] == pytest.approx(.5)
    summary = report['outcome_summary']['profitable']
    assert summary['realized_components_available'] == 1
    assert summary['computed_pnl_attribution_residual_sum'] == pytest.approx(-.5)
    assert summary['net_pnl_sum'] == pytest.approx(
        summary['mid_price_pnl_sum'] - summary['spread_cost_sum'] - summary['depth_cost_sum']
        - summary['slippage_cost_sum'] - summary['matched_fees_sum']
        + summary['computed_pnl_attribution_residual_sum'])


def test_normalization_can_reverse_profitable_entry_sign_without_changing_raw_credit():
    # Positive total NAV reward and positive raw entry GAE; below-batch-mean GAE
    # nevertheless gives the actor a negative normalized entry advantage.
    ep = episode([-.45, .5], [0., 0.], entries=[entry(net=.1)])
    report = analyze([ep])
    buy = report['entries'][0]
    assert buy['raw_gae'] == pytest.approx(.025)
    assert buy['normalized_advantage'] < 0.
    metrics = report['metrics']
    assert metrics['outcome/profitable/raw_gae_negative_count'] == 0
    assert metrics['outcome/profitable/normalized_advantage_negative_count'] == 1


def test_supplied_mixture_is_preserved_instead_of_recomputed_from_entry_profit():
    ep = episode([-.1, .2], [0., 0.], entries=[entry(net=-10.)])
    group = np.array([1.5, -1.5], dtype=np.float32)
    before = np.array([1.59, -1.3], dtype=np.float32)
    after = np.array([1., -1.], dtype=np.float32)
    report = analyze([ep], group_component=group, pre_normalized=before, normalized=after)
    buy = report['entries'][0]
    assert buy['group_component'] == 1.5
    assert buy['pre_normalized_advantage'] == float(before[0])
    assert buy['normalized_advantage'] == 1.
    assert buy['outcome'] == 'loss'


def test_episode_identity_uses_optional_primitive_copy_only():
    ep = episode([-.1, .2], [0., 0.], entries=[entry()])
    assert analyze([ep])['entries'][0]['episode_key'] == {}
    ep['metadata']['episode_start'] = {
        'episode_key': {'stock_code': '005930', 'date': '2026-09-10',
                        'start_index': np.int64(17), 'unrelated': object()}, 'large': object()}
    report = analyze([ep])
    expected = {'stock_code': '005930', 'date': '2026-09-10', 'start_index': 17}
    assert report['entries'][0]['episode_key'] == expected
    assert report['buy_decisions'][0]['episode_key'] == expected
    ep['metadata']['episode_start']['episode_key']['start_index'] = 999
    assert report['entries'][0]['episode_key']['start_index'] == 17
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('dtype', [np.int32, np.float32])
def test_legacy_numeric_zero_one_dones_match_boolean_decomposition(dtype):
    ep = episode([-.1, .2], [0., 0.], entries=[entry()])
    expected = analyze([ep])
    ep['dones'] = ep['dones'].astype(dtype)
    assert analyze([ep]) == expected
    ep['dones'][0] = 2
    with pytest.raises(ValueError, match='episode dones'):
        analyze([ep])
