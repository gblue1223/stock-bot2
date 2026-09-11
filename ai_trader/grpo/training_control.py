"""Evidence-based training stops and profitable validation candidate checks."""
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Integral, Real

import numpy as np


def finite_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def nonnegative_count(value):
    result = finite_number(value)
    return int(result) if result is not None and result >= 0 and result.is_integer() else None


def integer_setting(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass
class TrainingControlState:
    validation_count: int = 0
    no_trade_evidence_count: int = 0
    no_trade_streak: int = 0
    previous_buy_probability: float | None = None
    last_validation_iteration: int = 0
    stop_reason: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, state):
        if state is None:
            return cls()  # Checkpoints preceding this feature.
        if not isinstance(state, dict) or set(state) != set(cls.__dataclass_fields__):
            raise ValueError('Invalid checkpoint training_control_state')
        for key in ('validation_count', 'no_trade_evidence_count', 'no_trade_streak',
                    'last_validation_iteration'):
            integer_setting(f'training_control_state.{key}', state[key])
        probability = state['previous_buy_probability']
        if probability is not None:
            probability = finite_number(probability)
            if probability is None or not 0 <= probability <= 1:
                raise ValueError('Invalid checkpoint previous_buy_probability')
        if state['stop_reason'] not in (None, 'no_trade_collapse'):
            raise ValueError('Invalid checkpoint training control stop_reason')
        if not state['no_trade_streak'] <= state['no_trade_evidence_count'] <= state['validation_count']:
            raise ValueError('Invalid checkpoint training control counters')
        if state['no_trade_streak'] and probability is None:
            raise ValueError('A no-trade streak requires a prior buy probability')
        return cls(**{**state, 'previous_buy_probability': probability})

    def observe(self, metrics, *, iteration, score_improved, patience):
        """Unknown evidence resets continuity; revalidating a resume is not a new step."""
        self.validation_count += 1
        self.last_validation_iteration = iteration
        self.stop_reason = None
        diagnostics = metrics.get('diagnostics')
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        probabilities = diagnostics.get('mean_action_probabilities')
        probabilities = probabilities if isinstance(probabilities, dict) else {}
        buy = finite_number(probabilities.get('buy'))
        no_trade = finite_number(metrics.get('no_trade_episode_fraction'))
        filled = nonnegative_count(diagnostics.get('filled_quantity'))
        episodes = nonnegative_count(metrics.get('num_episodes'))
        mean_return = finite_number(metrics.get('mean_net_return'))
        verified = (no_trade == 1 and filled == 0 and episodes is not None and episodes > 0
                    and buy is not None and 0 <= buy <= 1 and mean_return == 0
                    and nonnegative_count(metrics.get('incomplete_liquidation_episodes')) == 0
                    and finite_number(metrics.get('max_open_quantity')) == 0)
        detailed = diagnostics.get('episodes')
        if verified and detailed is not None:
            verified = (isinstance(detailed, list) and len(detailed) == episodes
                        and all(isinstance(ep, dict) and nonnegative_count(ep.get('filled_quantity')) == 0
                                and finite_number(ep.get('open_quantity')) == 0
                                and isinstance(ep.get('liquidation_complete'), (bool, np.bool_))
                                and ep['liquidation_complete']
                                for ep in detailed))
        if not verified:
            self.no_trade_streak = 0
            self.previous_buy_probability = None
            return False
        self.no_trade_evidence_count += 1
        nonincreasing = self.previous_buy_probability is not None and buy <= self.previous_buy_probability + 1e-12
        self.no_trade_streak = self.no_trade_streak + 1 if nonincreasing and not score_improved else 0
        self.previous_buy_probability = buy
        if patience and self.no_trade_streak >= patience:
            self.stop_reason = 'no_trade_collapse'
        return self.stop_reason is not None


def profitable_candidate_evidence(metrics, min_round_trips, min_traded_dates):
    """Require realized trading over observed dates; never infer missing execution data."""
    evidence = {'eligible': False, 'reason': 'unknown_or_invalid_evidence',
                'min_round_trips': min_round_trips, 'min_traded_dates': min_traded_dates}
    mean_return = finite_number(metrics.get('mean_net_return'))
    if mean_return is None or mean_return <= 0:
        return {**evidence, 'reason': 'nonpositive_return'}
    incomplete = nonnegative_count(metrics.get('incomplete_liquidation_episodes'))
    open_quantity = finite_number(metrics.get('max_open_quantity'))
    if incomplete != 0 or open_quantity != 0:
        return {**evidence, 'reason': 'incomplete_or_unknown_liquidation'}
    no_trade = finite_number(metrics.get('no_trade_episode_fraction'))
    count = nonnegative_count(metrics.get('round_trip_count'))
    episode_count = nonnegative_count(metrics.get('num_episodes'))
    diagnostics = metrics.get('diagnostics')
    if not isinstance(diagnostics, dict):
        return evidence
    filled = nonnegative_count(diagnostics.get('filled_quantity'))
    episodes = diagnostics.get('episodes')
    if (no_trade is None or not 0 <= no_trade < 1 or count is None or filled is None or filled <= 0
            or episode_count is None or episode_count <= 0 or not isinstance(episodes, list)
            or len(episodes) != episode_count):
        return evidence
    dates = set()
    sum_filled = sum_round_trips = no_trade_episodes = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            return evidence
        episode_fills = nonnegative_count(episode.get('filled_quantity'))
        episode_round_trips = nonnegative_count(episode.get('round_trip_count'))
        episode_open = finite_number(episode.get('open_quantity'))
        complete = episode.get('liquidation_complete')
        if (episode_fills is None or episode_round_trips is None or episode_open != 0
                or not isinstance(complete, (bool, np.bool_)) or not complete):
            return evidence
        sum_filled += episode_fills
        sum_round_trips += episode_round_trips
        no_trade_episodes += int(episode_fills == 0)
        if episode_fills > 0:
            key = episode.get('episode_key')
            date = key.get('date') if isinstance(key, dict) else None
            if isinstance(date, (bool, np.bool_)) or not isinstance(date, (str, Integral)):
                return evidence
            text = str(date).replace('-', '')
            try:
                if len(text) != 8:
                    return evidence
                dates.add(datetime.strptime(text, '%Y%m%d').strftime('%Y%m%d'))
            except ValueError:
                return evidence
        elif episode_round_trips:
            return evidence
    if sum_filled != filled or sum_round_trips != count or not np.isclose(no_trade, no_trade_episodes / episode_count):
        return evidence
    evidence.update(round_trip_count=count, traded_date_count=len(dates), traded_dates=sorted(dates),
                    mean_net_return=mean_return, filled_quantity=filled)
    if count < min_round_trips:
        return {**evidence, 'reason': 'insufficient_round_trips'}
    if len(dates) < min_traded_dates:
        return {**evidence, 'reason': 'insufficient_traded_dates'}
    return {**evidence, 'eligible': True, 'reason': 'qualified_validation_candidate'}
