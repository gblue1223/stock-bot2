"""Causal HOLD/SELL teaching targets for a bounded scalping window.

These are explicit policy preferences, not counterfactual realized returns.
Quotes value the current inventory; actual orders still replay with latency.
"""


def exit_preference(age_seconds, quoted_net_pnl, config):
    """Prefer HOLD before 1s, take net profit by 5s, then attempt an exit.

    The deadline is measured from the stage's first actual fill. Unknown quote
    proceeds censor profit-taking labels, but elapsed time alone can establish
    a deadline exit. No future market values enter this rule.
    """
    if age_seconds >= config['target_max_seconds']:
        return 2, 'horizon_elapsed'
    if age_seconds < config['target_min_seconds']:
        return 0, 'before_target_window'
    if quoted_net_pnl is None:
        return None, 'quote_unavailable'
    return (2, 'net_profit_available') if quoted_net_pnl > 0 else (0, 'await_net_profit')
