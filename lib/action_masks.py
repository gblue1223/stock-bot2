"""Causal order eligibility shared by replay and live inference.

This permits order submission, not guaranteed execution: future liquidity and
prices remain unknown. Risk exits are handled by the execution/risk controller
independently of the policy mask.
"""
from collections.abc import Mapping
from numbers import Real

import numpy as np

from lib.observations import validate_max_stages


def executable_action_mask(state, max_stages):
    """Return HOLD/BUY/SELL eligibility from explicit current execution state.

    occupied_stages counts distinct entry order IDs across fills and pending
    buys, so a partially filled order reserves exactly one stage. buy_budget is
    the available cash capped by the configured per-stage budget; buy_unit_cost
    includes the current quote, execution slippage/rounding, and entry fees.
    exit_active also covers a risk exit already due at the current timestamp.
    Live callers must supply every field, including flat-account affordability.
    """
    max_stages = validate_max_stages(max_stages)
    fields = ('filled_stages', 'occupied_stages', 'pending_buy', 'pending_sell',
              'exit_active', 'within_trade_limit', 'buy_budget', 'buy_unit_cost')
    if not isinstance(state, Mapping) or any(field not in state for field in fields):
        raise ValueError('execution action masking requires explicit action_mask_state fields: '
                         + ', '.join(fields))
    for name in ('filled_stages', 'occupied_stages'):
        value = state[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not 0 <= value <= max_stages):
            raise ValueError(f'action_mask_state {name} must be an integer within stage limits')
    if state['filled_stages'] > state['occupied_stages']:
        raise ValueError('occupied_stages must include every filled stage')
    for name in ('pending_buy', 'pending_sell', 'exit_active', 'within_trade_limit'):
        if not isinstance(state[name], (bool, np.bool_)):
            raise ValueError(f'action_mask_state {name} must be a boolean')
    for name in ('buy_budget', 'buy_unit_cost'):
        value = state[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'action_mask_state {name} must be finite and nonnegative')
    if state['buy_unit_cost'] <= 0:
        raise ValueError('action_mask_state buy_unit_cost must be positive')
    buy = (not state['exit_active'] and not state['pending_buy'] and not state['pending_sell']
           and state['within_trade_limit'] and state['occupied_stages'] < max_stages
           and state['buy_budget'] >= state['buy_unit_cost'])
    sell = (state['filled_stages'] > 0 and not state['pending_sell'] and not state['exit_active'])
    return np.array([True, buy, sell], dtype=np.bool_)
