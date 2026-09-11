"""Validation distinguishes no-trade baselines and aggregates actual cost exposure."""
import numpy as np
import pytest
import torch

from ai_trader.grpo.evaluation import _additional_metrics, evaluate_policy


def test_exit_holding_uses_quantity_not_episode_or_round_trip_count():
    episodes = [
        dict(exit_signal_round_trip_count=1, exit_signal_quantity=10,
             exit_signal_net_pnl=-2, exit_signal_holding_seconds=1,
             mid_price_pnl=4, spread_cost=2, depth_cost=1, slippage_cost=1,
             matched_fees=2, pnl_attribution_residual=0),
        dict(exit_signal_round_trip_count=2, exit_signal_quantity=90,
             exit_signal_net_pnl=5, exit_signal_holding_seconds=9,
             mid_price_pnl=10, spread_cost=2, depth_cost=1, slippage_cost=1,
             matched_fees=1, pnl_attribution_residual=0),
    ]
    result = _additional_metrics(episodes)
    assert result['exit_reasons']['signal'] == {
        'round_trip_count': 3, 'quantity': 100., 'net_pnl': 3.,
        'quantity_weighted_holding_seconds': 8.2,
    }
    assert result['mid_price_pnl'] == 14
    assert result['mean_matched_fees'] == 1.5
    assert (result['mid_price_pnl'] - result['spread_cost'] - result['depth_cost']
            - result['slippage_cost'] - result['matched_fees']) == 3
    # Legacy checkpoints must not turn missing diagnostics into apparent zero cost.
    assert result['exit_reasons']['stop_loss']['net_pnl'] is None
    result = _additional_metrics([episodes[0], {}])
    assert result['mid_price_pnl'] is None
    assert result['exit_reasons']['signal']['quantity_weighted_holding_seconds'] is None


class Policy(torch.nn.Module):
    def get_action(self, obs, deterministic=False):
        assert deterministic
        return torch.zeros(len(obs), dtype=torch.long), torch.zeros(len(obs))


class Episode:
    def __init__(self, net_return, filled, incomplete):
        self.metrics = dict(net_return=net_return, num_trades=int(bool(filled)),
                            max_drawdown=0., avg_holding_time=0.,
                            open_quantity=int(incomplete), liquidation_complete=not incomplete,
                            realized_net_pnl=net_return)
        if filled is not None:
            self.metrics['filled_quantity'] = filled

    def reset(self, seed):
        return np.zeros((1, 1), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1, 1), dtype=np.float32), 0., True, False, {'episode': self.metrics}


@pytest.mark.parametrize('net_return,filled,incomplete,outcome,profitable', [
    (0., 0, False, 'no_trade', False),
    (1., 2, False, 'profitable', True),
    (-1., 2, False, 'nonprofitable', False),
    (1., 2, True, 'incomplete_liquidation', False),
    (0., None, False, 'unknown_execution', None),
])
def test_validation_reports_economic_outcome_separately_from_selection(
        net_return, filled, incomplete, outcome, profitable):
    metrics = evaluate_policy(Policy(), Episode(net_return, filled, incomplete), num_episodes=1)
    assert metrics['no_trade_reference_net_return'] == 0
    assert metrics['evaluation_outcome'] == outcome
    assert metrics['profitable_with_trades'] is profitable
