"""Causal scalping environment with cash/quantity accounting and order replay.

Observation features never determine fills after normalization. Rewards are
changes in liquidation-marked equity, expressed as percent of initial capital.
"""
import logging
import re
from typing import Optional
import numpy as np
import duckdb
import gymnasium as gym
from gymnasium import spaces

from lib.market_data import parse_time_seconds, times_to_seconds
from lib.observations import ObservationBuilder, validate_max_stages
from lib.action_masks import executable_action_mask
from .execution import ExecutionSimulator
from ai_trader.grpo.entry_pattern import validate_entry_pattern, fill_target

logger = logging.getLogger(__name__)


class GRPOScalpingEnv(gym.Env):
    metadata = {'render_modes': []}
    MAX_STAGES = 5

    def __init__(self, db_path, table_name='datasets', seq_len=3000,
                 expected_features=27, transaction_cost_rate=0.00015,
                 buy_tax_rate=0.0, sell_tax_rate=0.0018, no_trade_penalty=0.0,
                 max_episode_steps=None, use_raw_data=True,
                 rolling_window_size=1000, rolling_min_samples=100,
                 base_price=100000.0, max_trades_per_episode=None,
                 step_reward_scale=1.0, win_bonus=0.0, loss_penalty=0.0,
                 buy_signal_bonus=0.0, initial_cash=1000000.0,
                 max_holding_seconds=300.0, stop_loss_pct=2.0,
                 execution_config=None, allowed_dates=None, price_scale=1.0, max_stages=1,
                 account_observations=False, liquidation_max_steps=0,
                 execution_observations=False, decision_interval_seconds=0.0,
                 episode_duration_seconds=0.0, execution_action_mask=False, entry_pattern_config=None):
        super().__init__()
        self.entry_pattern_config = validate_entry_pattern(entry_pattern_config)
        if self.entry_pattern_config is not None and not execution_action_mask:
            raise ValueError('entry_pattern_config requires execution_action_mask=True')
        self.max_stages = validate_max_stages(max_stages)
        if not isinstance(execution_action_mask, (bool, np.bool_)):
            raise ValueError('execution_action_mask must be a boolean')
        self.execution_action_mask = bool(execution_action_mask)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table_name):
            raise ValueError('table_name must be a simple SQL identifier')
        if seq_len < 1 or expected_features < 1 or initial_cash <= 0 or not np.isfinite(initial_cash):
            raise ValueError('invalid sequence/features/initial capital')
        if max_holding_seconds <= 0 or not np.isfinite(max_holding_seconds):
            raise ValueError('max_holding_seconds must be finite and positive')
        if stop_loss_pct <= 0 or not np.isfinite(stop_loss_pct):
            raise ValueError('stop_loss_pct must be finite and positive')
        if not use_raw_data:
            raise ValueError('raw features/prices required; normalized databases cannot supply execution prices')
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError('max_episode_steps must be positive')
        if (isinstance(liquidation_max_steps, (bool, np.bool_))
                or not isinstance(liquidation_max_steps, (int, np.integer))
                or liquidation_max_steps < 0):
            raise ValueError('liquidation_max_steps must be a nonnegative integer')
        self.liquidation_max_steps = int(liquidation_max_steps)
        self.account_observations = account_observations
        self.execution_observations = execution_observations
        for name, value in (('decision_interval_seconds', decision_interval_seconds),
                            ('episode_duration_seconds', episode_duration_seconds)):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, float(value))
        if max_trades_per_episode is not None and max_trades_per_episode < 1:
            raise ValueError('max_trades_per_episode must be positive')
        if price_scale <= 0 or not np.isfinite(price_scale) or base_price <= 0:
            raise ValueError('invalid price unit/base price')
        if any(x != 0 for x in (no_trade_penalty, win_bonus, loss_penalty, buy_signal_bonus)) or step_reward_scale != 1:
            logger.warning('Legacy reward-shaping arguments are ignored: rewards equal net NAV changes.')
        self.db_path, self.table_name = db_path, table_name
        self.seq_len, self.expected_features = int(seq_len), int(expected_features)
        self.use_raw_data = True
        self.rolling_window_size, self.rolling_min_samples = rolling_window_size, rolling_min_samples
        self.initial_cash, self.base_price = float(initial_cash), float(base_price)
        self.price_scale = float(price_scale)
        if self.price_scale not in (1.0, 1000000.0):
            raise ValueError('price_scale must declare KRW (1) or million KRW (1000000)')
        self.price_unit = 'million_krw' if self.price_scale == 1000000 else 'krw'
        self.max_episode_steps, self.max_trades_per_episode = max_episode_steps, max_trades_per_episode
        self.max_holding_seconds, self.stop_loss_pct = float(max_holding_seconds), float(stop_loss_pct)
        self.allowed_dates = None if allowed_dates is None else {str(int(d)) for d in allowed_dates}
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.buy_tax_rate, self.sell_tax_rate = float(buy_tax_rate), float(sell_tax_rate)
        for rate in (self.transaction_cost_rate, self.buy_tax_rate, self.sell_tax_rate):
            if not np.isfinite(rate) or not 0 <= rate < 1:
                raise ValueError('fee/tax rates must lie in [0, 1)')
        self.round_trip_cost = self.buy_fee_rate + self.sell_fee_rate
        self.no_trade_penalty = self.win_bonus = self.loss_penalty = self.buy_signal_bonus = 0.0
        self.step_reward_scale = 1.0
        self.simulator = ExecutionSimulator(execution_config)
        self.conn, self.feature_columns, self.valid_keys = None, None, None
        self.normalizer = None  # Compatibility; normalization belongs to ObservationBuilder.
        self.raw_accum_trade_value = None
        self.episode_execution = {}
        self._reset_options = {}
        self._init_metadata_from_db()
        if not self.feature_columns or len(self.feature_columns) != self.expected_features:
            raise ValueError('manifest feature names/count must exactly match expected_features')
        self.observation_builder = ObservationBuilder(
            self.feature_columns, seq_len=self.seq_len,
            rolling_window_size=self.rolling_window_size,
            rolling_min_samples=self.rolling_min_samples,
            max_holding_seconds=self.max_holding_seconds,
            feature_price_unit=self.price_unit, max_stages=self.max_stages,
            account_observations=self.account_observations,
            execution_observations=self.execution_observations)
        self.obs_dim = self.observation_builder.obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.seq_len, self.obs_dim), np.float32)
        self.action_space = spaces.Discrete(3)

    @property
    def buy_fee_rate(self):
        return self.transaction_cost_rate + self.buy_tax_rate

    @property
    def sell_fee_rate(self):
        return self.transaction_cost_rate + self.sell_tax_rate

    @property
    def observation_schema(self):
        return self.observation_builder.schema

    def get_observation_schema(self):
        return self.observation_schema

    @property
    def position(self):
        return int(bool(getattr(self, 'stages', [])))

    @property
    def position_steps(self):
        return len(getattr(self, 'stages', []))

    @property
    def max_split_count(self):
        return self.max_stages

    @property
    def quantity(self):
        return sum(st['quantity'] for st in self.stages)

    def _connect_db(self):
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def _ensure_db_connection(self):
        if self.conn is None:
            self._connect_db()

    def _init_metadata_from_db(self):
        if self.entry_pattern_config is not None:
            raise ValueError('Entry-pattern training requires newly extracted NPZ data with entry signals')
        self._connect_db()
        self.feature_columns = self._get_feature_columns()
        self.return_rate_index = self.feature_columns.index('등락률') if '등락률' in self.feature_columns else None
        self.accum_trade_value_index = self.feature_columns.index('누적거래대금') if '누적거래대금' in self.feature_columns else None
        self._preload_valid_keys()
        self.conn.close()
        self.conn = None

    def _get_feature_columns(self):
        from lib.market_data import canonical_feature_columns
        columns = canonical_feature_columns(self.expected_features)
        schema = self.conn.execute(f'DESCRIBE "{self.table_name}"').fetchdf()
        self._db_columns = schema['column_name'].tolist()
        required = set(columns) - {'시초가', '시초가대비등락률'}
        if required <= set(self._db_columns):
            return columns
        raise ValueError('DB must contain the explicit canonical feature schema; extract/version data first')

    def _preload_valid_keys(self):
        rows = self.conn.execute(
            f'SELECT 종목코드, 날짜, COUNT(*) FROM "{self.table_name}" '
            'GROUP BY 종목코드, 날짜 HAVING COUNT(*) >= ?', [self.seq_len + 1]).fetchall()
        self.valid_keys = [row for row in rows if self.allowed_dates is None or str(int(row[1])) in self.allowed_dates]
        if not self.valid_keys:
            raise ValueError('no episodes satisfy history/date requirements')

    def _sample_episode_start(self, max_attempts=100):
        index = self._reset_options.get('episode_index')
        index = int(self.np_random.integers(len(self.valid_keys))) if index is None else int(index)
        stock, date, _ = self.valid_keys[index]
        tie = ', "번호"' if '번호' in self._db_columns else ''
        df = self.conn.execute(
            f'SELECT * FROM "{self.table_name}" WHERE 종목코드=? AND 날짜=? '
            f'ORDER BY TRY_CAST("시간" AS DOUBLE){tie}', [str(stock), date]).fetchdf()
        # Forward-only filling; never copy a later observation to an earlier event.
        if '시초가' in self.feature_columns:
            from lib.market_data import add_opening_price_features
            df = add_opening_price_features(df)
            if not (df['시초가'] > 0).any():
                raise ValueError('missing_opening_price: supply 시가/시초가 or a 09:00:00 tick')
        features = df[self.feature_columns].ffill().fillna(0).to_numpy(dtype=np.float32)
        metadata = df[['종목코드', '날짜', '시간']].to_numpy()
        from lib.market_data import execution_arrays
        execution = execution_arrays(df)
        for key in ('last_price', 'bid_prices', 'ask_prices'):
            if key in execution:
                execution[key] = execution[key] * self.price_scale
        start, end = self._episode_slice_bounds(metadata)
        self.episode_execution = {key: value[start:end] for key, value in execution.items()}
        self.episode_key = {'stock_code': str(stock), 'date': str(date), 'start_index': start}
        return features[start:end], metadata[start:end]

    def _episode_slice_bounds(self, metadata):
        """Reserve policy time and a raw-event liquidation suffix, without leaking it."""
        length = len(metadata)
        if length < self.seq_len + 1:
            raise ValueError('Episode is shorter than the observation window plus one step')
        options = self._reset_options
        if not (self.decision_interval_seconds or self.episode_duration_seconds):
            needed = (self.seq_len + self.max_episode_steps + self.liquidation_max_steps
                      if self.max_episode_steps is not None else length)
            max_start = max(0, length - needed)
            start = self._choose_episode_start(max_start)
            if not 0 <= start <= max_start:
                raise ValueError('Requested start_index is outside the episode')
            return start, min(length, start + needed)
        timestamps = times_to_seconds(metadata[:, 2])
        if np.any(np.diff(timestamps) < 0) or not np.isfinite(timestamps).all():
            raise ValueError('episode timestamps must be finite and monotonic')
        def decision_end(start_index):
            index = start_index + self.seq_len - 1
            deadline = timestamps[index] + self.episode_duration_seconds if self.episode_duration_seconds else np.inf
            for _ in range(self.max_episode_steps or length):
                target = min(timestamps[index] + self.decision_interval_seconds, deadline)
                index = max(index + 1, int(np.searchsorted(timestamps, target, side='left')))
                if index >= length - 1:
                    return length - 1
                if timestamps[index] >= deadline:
                    break
            return index

        # Each policy observation rounds up to a real event. Repeated gaps can
        # accumulate, so nominal interval * decision count cannot reserve the tail.
        # The actual final index is monotonic in the start, allowing a bounded search.
        last_policy = length - 1 - self.liquidation_max_steps
        lo, hi = 0, length - self.seq_len - 1
        while lo < hi:
            candidate = (lo + hi + 1) // 2
            if decision_end(candidate) <= last_policy:
                lo = candidate
            else:
                hi = candidate - 1
        max_start = lo
        start = self._choose_episode_start(max_start)
        if not 0 <= start <= max_start:
            raise ValueError('Requested start_index leaves insufficient policy time/history and liquidation tail')
        index = decision_end(start)
        end = min(length, index + self.liquidation_max_steps + 1)
        return start, end

    def _choose_episode_start(self, max_start):
        options = self._reset_options
        if self.entry_pattern_config is None:
            return int(options['start_index']) if 'start_index' in options else int(self.np_random.integers(max_start + 1))
        candidates = np.flatnonzero(self._full_entry_signal[self.seq_len - 1:self.seq_len + max_start])
        if not len(candidates):
            raise ValueError('No qualifying entry leaves enough observation/policy history in this episode')
        start = int(options['start_index']) if 'start_index' in options else int(self.np_random.choice(candidates))
        if start not in candidates:
            raise ValueError('Requested start_index does not satisfy the entry-pattern gate')
        return start

    def _entry_allowed(self):
        return self.entry_pattern_config is None or bool(self._episode_entry_signal[self.current_step])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_options = dict(options or {})
        self._ensure_db_connection()
        self.episode_execution = {}
        self.raw_accum_trade_value = None
        self.episode_data, self.episode_metadata = self._sample_episode_start()
        self.episode_data = np.asarray(self.episode_data, dtype=np.float32)
        self.episode_length = len(self.episode_data)
        if self.episode_data.shape != (self.episode_length, self.expected_features) or self.episode_length <= self.seq_len:
            raise ValueError('invalid episode shape/length')
        if not np.isfinite(self.episode_data).all():
            raise ValueError('nonfinite raw features')
        self.timestamps = times_to_seconds(self.episode_metadata[:, 2])
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) < 0):
            raise ValueError('episode timestamps must be finite and monotonic')
        if len(set(map(str, self.episode_metadata[:, 1]))) != 1:
            raise ValueError('an episode must stay within one trading date')
        for key, values in self.episode_execution.items():
            if len(values) != self.episode_length:
                raise ValueError(f'execution array is not aligned: {key}')
        self._compute_prices()
        self.current_step = self.seq_len - 1
        self.decision_steps = min(self.episode_length - self.seq_len,
                                  self.max_episode_steps or self.episode_length)
        self.decision_end_index = self.current_step + self.decision_steps
        self.episode_start_time_seconds = float(self.timestamps[self.current_step])
        self.__dict__.pop('policy_end_time_seconds', None)
        self.episode_deadline = (self.episode_start_time_seconds + self.episode_duration_seconds
                                 if self.episode_duration_seconds else np.inf)
        self.liquidation_steps = 0
        self.liquidation_seconds = 0.0
        self.liquidation_stop_reason = 'disabled' if not self.liquidation_max_steps else 'not_started'
        self.stages, self.episode_trades, self.episode_rewards = [], [], []
        self.cash, self.realized_net_pnl = self.initial_cash, 0.0
        self.total_entry_fees = self.total_exit_fees = 0.0
        self.loss_holding_violations = 0
        self.buy_action_outcomes = dict.fromkeys((
            'submitted', 'risk_exit_active', 'signal_exit_active', 'max_stages', 'max_trades',
            'pending_buy', 'insufficient_budget_for_one_share'), 0)
        self._buy_decisions = []
        self._pattern_fills = []
        if self.entry_pattern_config is not None:
            self.buy_action_outcomes['entry_signal_not_met'] = 0
        self._entry_order_decisions = {}
        self._exit_requested, self._done = False, False
        self._exit_reason = None
        self._entry_exit_reasons = {}
        self._signal_exit_order_id = None
        self.simulator.reset()
        self._update_market_fields()
        self.simulator.process(self._snapshot(self.current_step))
        self.equity = self.initial_cash
        self.equity_history = [self.equity]
        if self.raw_accum_trade_value is None and self.accum_trade_value_index is not None:
            self.raw_accum_trade_value = self.episode_data[:, self.accum_trade_value_index].copy()
        # Market grouping uses only pre-action history, never rewards/future prices.
        history = self.prices[:self.seq_len]
        changes = np.diff(history) / history[:-1]
        self.market_context = [float(np.std(changes)) if len(changes) else 0.0,
                               float(history[-1] / history[0] - 1),
                               float(np.ptp(history) / history[0])]
        return self._get_current_observation(), {
            'episode_key': getattr(self, 'episode_key', {}),
            'market_context': self.market_context, 'observation_schema': self.observation_schema,
            'execution_model': self.simulator.summary()['execution_model'],
            **({'action_mask': self.action_masks()} if self.execution_action_mask else {})}

    def _compute_prices(self):
        if 'last_price' in self.episode_execution:
            self.prices = np.asarray(self.episode_execution['last_price'], dtype=np.float64)
            self.price_source = 'raw_last_price'
        elif '현재가' in self.feature_columns:
            self.prices = self.episode_data[:, self.feature_columns.index('현재가')].astype(np.float64) * self.price_scale
            self.price_source = 'raw_feature_price'
        elif self.return_rate_index is not None:
            self.prices = self.base_price * (1 + self.episode_data[:, self.return_rate_index].astype(np.float64) / 100)
            self.price_source = 'relative_base_price'
        else:
            raise ValueError('raw price or raw percentage return is required')
        if not np.isfinite(self.prices).all() or np.any(self.prices <= 0):
            raise ValueError('invalid raw execution prices; do not substitute normalized/zero prices')

    def _get_current_price(self):
        return float(self.prices[self.current_step])

    def _update_market_fields(self):
        self.current_price = self._get_current_price()
        self.current_time = self.episode_metadata[self.current_step, 2]
        self.current_time_seconds = float(self.timestamps[self.current_step])

    def _snapshot(self, index):
        kwargs = {}
        keys = ('bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes')
        if all(k in self.episode_execution for k in keys):
            for side in ('bid', 'ask'):
                prices, sizes = self.episode_execution[side + '_prices'][index], self.episode_execution[side + '_sizes'][index]
                # Zero/zero levels represent empty book slots, not a fillable quote.
                kwargs['bids' if side == 'bid' else 'asks'] = [
                    (float(p), float(q)) for p, q in zip(prices, sizes) if p != 0 or q != 0]
        if 'quote_timestamp' in self.episode_execution:
            kwargs['quote_timestamp'] = float(self.episode_execution['quote_timestamp'][index])
        return self.simulator.make_snapshot(float(self.timestamps[index]), float(self.prices[index]), **kwargs)

    def _get_current_observation(self):
        start = max(0, self.current_step - self.seq_len + 1)
        return self.observation_builder.build(
            self.episode_data[start:self.current_step + 1], self.stages,
            current_price=self.current_price, current_time_seconds=self.current_time_seconds,
            account_state=self._account_state() if self.account_observations else None,
            execution_state=self._execution_state() if self.execution_observations else None)

    def _account_state(self):
        """Account values known now; remaining time describes decisions, not replay tail."""
        mark = self.simulator.liquidation_mark()
        ask = self.simulator.snapshot.asks[0][0] if self.simulator.snapshot.asks else self.current_price
        buy_price = self.simulator.execution_price(ask, 'buy')
        cash_ratio = self.cash / self.initial_cash
        # Multiplying fees before/after quantity can differ at machine precision.
        # Preserve meaningful overdraws for schema validation instead of hiding them.
        if -1e-12 < cash_ratio < 0:
            cash_ratio = 0.0
        state = {
            'cash_ratio': cash_ratio,
            'position_value_ratio': self.quantity * mark / self.initial_cash,
            'pending_buy_value_ratio': sum(o.remaining for o in self._pending('buy')) * buy_price / self.initial_cash,
            'pending_sell_value_ratio': sum(o.remaining for o in self._pending('sell')) * mark / self.initial_cash,
            'exit_pending': float(self._exit_requested or self._signal_exit_order_id is not None),
            'remaining_steps_ratio': max(0, self.decision_steps - len(self.episode_rewards)) / self.decision_steps,
        }
        for index in range(self.MAX_STAGES):
            quantity = self.stages[index]['quantity'] if index < len(self.stages) else 0
            state[f'stage_{index + 1}_value_ratio'] = quantity * mark / self.initial_cash
        return state

    def _execution_state(self):
        """Current quotes and cost basis only; estimates do not promise a future fill."""
        snapshot = self.simulator.snapshot
        half = self.simulator.config.spread_bps / 20000
        bid = snapshot.bids[0][0] if snapshot.bids else snapshot.last_price * (1 - half)
        ask = snapshot.asks[0][0] if snapshot.asks else snapshot.last_price * (1 + half)
        if not snapshot.bids:
            bid = min(bid, ask)
        if not snapshot.asks:
            ask = max(ask, bid)
        mid = (bid + ask) / 2
        buy = self.simulator.execution_price(ask, 'buy') * (1 + self.buy_fee_rate)
        sell = self.simulator.execution_price(bid, 'sell') * (1 - self.sell_fee_rate)
        basis = sum(st['quantity'] * st['entry_price'] + st['entry_fees'] for st in self.stages)
        proceeds = self.quantity * sell
        budget = min(self.cash, self.initial_cash / self.max_stages)
        known = snapshot.quote_timestamp is not None
        age = snapshot.timestamp - snapshot.quote_timestamp if known else 0.0
        max_age = self.simulator.config.max_quote_age_seconds
        stale = known and age > max_age
        available = self.simulator._remaining
        buy_depth = sum(q * self.simulator.execution_price(p, 'buy') * (1 + self.buy_fee_rate)
                        for p, q in available['buy'].items()) if not stale else 0.0
        sell_depth = sum(available['sell'].values()) if not stale else 0.0
        elapsed = max(0.0, self.current_time_seconds - self.episode_start_time_seconds)
        scale = self.episode_duration_seconds or self.max_holding_seconds
        return {
            'spread_bps': (ask - bid) / mid * 10000,
            'quoted_round_trip_cost_bps': (1 - sell / buy) * 10000,
            'liquidation_return': proceeds / basis - 1 if basis > 0 else 0.0,
            'breakeven_return': basis / proceeds - 1 if proceeds > 0 else 0.0,
            'buy_depth_ratio': float(np.clip(buy_depth / budget, 0, 1)) if budget > 0 else 0.0,
            'sell_depth_ratio': float(np.clip(sell_depth / self.quantity, 0, 1)) if self.quantity else 0.0,
            'quote_age_fraction': age / max_age if max_age else float(age > 0),
            'quote_timestamp_known': float(known),
            'episode_elapsed_fraction': elapsed / scale,
            'remaining_time_fraction': max(0.0, 1 - elapsed / scale) if self.episode_duration_seconds else 0.0,
        }

    def _calculate_seconds_diff(self, start_time_val, end_time_val):
        difference = parse_time_seconds(end_time_val) - parse_time_seconds(start_time_val)
        if not np.isfinite(difference) or difference < 0:
            raise ValueError('invalid/backwards elapsed time')
        return float(difference)

    def set_transaction_cost_rate(self, rate):
        if not np.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError('invalid transaction fee')
        self.transaction_cost_rate = float(rate)
        self.round_trip_cost = self.buy_fee_rate + self.sell_fee_rate

    def _apply_fill(self, fill):
        value = fill.quantity * fill.price
        mid = fill.mid_price if fill.mid_price is not None else fill.price
        book = fill.book_price if fill.book_price is not None else fill.price
        top = fill.top_quote if fill.top_quote is not None else book
        if fill.side == 'buy':
            if self.entry_pattern_config is not None:
                self._pattern_fills.append((fill.order_id, fill.timestamp, fill.price, fill.quantity))
            fee = value * self.buy_fee_rate
            self.cash -= value + fee
            self.total_entry_fees += fee
            stage = next((s for s in self.stages if s['order_id'] == fill.order_id), None)
            if stage is None:
                self.stages.append({'order_id': fill.order_id, 'quantity': fill.quantity,
                    'entry_price': fill.price, 'entry_time_seconds': fill.timestamp,
                    'entry_time': self.current_time, 'entry_fees': fee,
                    'entry_time_sum': fill.quantity * fill.timestamp,
                    'entry_mid_value': fill.quantity * mid,
                    'entry_spread_cost': fill.quantity * (top - mid),
                    'entry_depth_cost': fill.quantity * (book - top),
                    'entry_slippage_cost': fill.quantity * (fill.price - book),
                    'entry_lots': [[fill.quantity, fill.timestamp]]})
            else:
                qty = stage['quantity'] + fill.quantity
                stage['entry_price'] = (stage['entry_price'] * stage['quantity'] + value) / qty
                stage['quantity'] = qty
                stage['entry_fees'] += fee
                stage['entry_time_sum'] += fill.quantity * fill.timestamp
                stage['entry_mid_value'] += fill.quantity * mid
                stage['entry_spread_cost'] += fill.quantity * (top - mid)
                stage['entry_depth_cost'] += fill.quantity * (book - top)
                stage['entry_slippage_cost'] += fill.quantity * (fill.price - book)
                stage['entry_lots'].append([fill.quantity, fill.timestamp])
            if self._exit_reason or self._signal_exit_order_id == fill.order_id:
                self._entry_exit_reasons.setdefault(fill.order_id, self._exit_reason or 'signal')
        else:
            self.cash += value * (1 - self.sell_fee_rate)
            self.total_exit_fees += value * self.sell_fee_rate
            remaining = fill.quantity
            while remaining:
                stage = self.stages[0]
                qty = min(remaining, stage['quantity'])
                entry_fee = stage['entry_fees'] * qty / stage['quantity']
                exit_fee = qty * fill.price * self.sell_fee_rate
                mean_entry_time = stage['entry_time_sum'] / stage['quantity']
                ratio = qty / stage['quantity']
                entry_mid = stage['entry_mid_value'] * ratio
                spread_cost = stage['entry_spread_cost'] * ratio + qty * (mid - top)
                depth_cost = stage['entry_depth_cost'] * ratio + qty * (top - book)
                slippage_cost = stage['entry_slippage_cost'] * ratio + qty * (book - fill.price)
                # Average-cost inventory allocates each acquisition lot pro rata.
                # This reports actual <=1s exposure without classifying by a mean.
                fast_quantity = sum(lot_qty * ratio for lot_qty, timestamp in stage['entry_lots']
                                    if fill.timestamp - timestamp <= 1.0 + 1e-9)
                reason = self._entry_exit_reasons.setdefault(stage['order_id'], fill.reason)
                pnl = qty * (fill.price - stage['entry_price']) - entry_fee - exit_fee
                self.realized_net_pnl += pnl
                self.episode_trades.append({
                    'entry_price': stage['entry_price'], 'exit_price': fill.price,
                    'quantity': qty, 'entry_fee': entry_fee, 'exit_fee': exit_fee,
                    'holding_time': fill.timestamp - stage['entry_time_seconds'],
                    'holding_share_seconds': qty * (fill.timestamp - mean_entry_time),
                    'profit_rate': fill.price / stage['entry_price'] - 1,
                    'net_pnl': pnl, 'net_return': pnl / self.initial_cash * 100,
                    'reward': pnl / self.initial_cash * 100,
                    'weight': qty * stage['entry_price'] / self.initial_cash,
                    'exit_reason': reason, 'order_id': fill.order_id,
                    'mid_price_pnl': qty * mid - entry_mid,
                    'spread_cost': spread_cost, 'depth_cost': depth_cost,
                    'slippage_cost': slippage_cost,
                    'subsecond_exit_quantity': fast_quantity,
                    'entry_order_id': stage['order_id']})
                stage['quantity'] -= qty
                stage['entry_fees'] -= entry_fee
                stage['entry_time_sum'] -= qty * mean_entry_time
                for name in ('entry_mid_value', 'entry_spread_cost', 'entry_depth_cost', 'entry_slippage_cost'):
                    stage[name] *= 1 - ratio
                stage['entry_lots'] = [[lot_qty * (1 - ratio), timestamp]
                                       for lot_qty, timestamp in stage['entry_lots']]
                remaining -= qty
                if stage['quantity'] == 0:
                    self.stages.pop(0)

    def _pending(self, side):
        return [o for o in self.simulator.orders if o.active and o.side == side]

    def _buy_budget_and_unit_cost(self):
        ask = self.simulator.snapshot.asks[0][0] if self.simulator.snapshot.asks else self.current_price
        cash = self.cash
        # Match account observation tolerance without hiding material overdrafts.
        if -1e-12 < cash / self.initial_cash < 0:
            cash = 0.0
        return (min(cash, self.initial_cash / self.max_stages),
                self.simulator.execution_price(ask, 'buy') * (1 + self.buy_fee_rate))

    def action_mask_state(self):
        """Read current eligibility without submitting/cancelling any orders."""
        pending_buy, pending_sell = self._pending('buy'), self._pending('sell')
        occupied = {stage['order_id'] for stage in self.stages} | {order.order_id for order in pending_buy}
        budget, unit_cost = self._buy_budget_and_unit_cost()
        stop, overdue = self._risk_exit_due(self.current_time_seconds)
        legacy_horizon = (not self.liquidation_max_steps and not self.decision_interval_seconds
                          and len(self.episode_rewards) + 1 >= self.decision_steps)
        return {
            'filled_stages': len(self.stages), 'occupied_stages': len(occupied),
            'pending_buy': bool(pending_buy), 'pending_sell': bool(pending_sell),
            'exit_active': bool(self._done or self._exit_requested or self._signal_exit_order_id is not None
                                or stop or overdue or legacy_horizon),
            'within_trade_limit': (self.max_trades_per_episode is None
                                   or len(self.episode_trades) < self.max_trades_per_episode),
            'buy_budget': budget, 'buy_unit_cost': unit_cost,
        }

    def action_masks(self):
        """Return authoritative policy eligibility at the current decision."""
        mask = executable_action_mask(self.action_mask_state(), self.max_stages)
        if not self._entry_allowed():
            mask[1] = False
        return mask

    def _request_exit(self, reason, timestamp):
        if self._exit_reason is None:
            self._exit_reason = reason
        self._exit_requested = True
        for stage in self.stages:
            self._entry_exit_reasons.setdefault(stage['order_id'], self._exit_reason)
        self.simulator.cancel_all(timestamp, side='buy')
        pending_qty = sum(o.remaining for o in self._pending('sell'))
        qty = self.quantity - pending_qty
        if qty > 0:
            self.simulator.submit('sell', qty, timestamp, reason=self._exit_reason)

    def _request_stage_exit(self, timestamp):
        """Finish the selected FIFO stage, including fills racing its cancellation."""
        self.simulator.cancel_all(timestamp, side='buy')
        target = next((stage for stage in self.stages
                       if stage['order_id'] == self._signal_exit_order_id), None)
        if self._signal_exit_order_id is not None:
            self._entry_exit_reasons.setdefault(self._signal_exit_order_id, 'signal')
        pending_qty = sum(order.remaining for order in self._pending('sell'))
        qty = (target['quantity'] if target is not None else 0) - pending_qty
        if qty > 0:
            self.simulator.submit('sell', qty, timestamp, reason='signal')

    def _equity(self):
        return self.cash + self.quantity * self.simulator.liquidation_mark() * (1 - self.sell_fee_rate)

    def _advance_market(self, index):
        self.current_step = index
        self._update_market_fields()
        fills = self.simulator.process(
            self._snapshot(index), cash_available=self.cash, sell_available=self.quantity,
            buy_fee_rate=self.buy_fee_rate, sell_fee_rate=self.sell_fee_rate)
        for fill in fills:
            self._apply_fill(fill)
        self.equity = self._equity()
        self.equity_history.append(self.equity)
        return fills

    def _liquidate_after_decisions(self):
        """Replay a bounded real market suffix without any more policy decisions."""
        started = self.current_time_seconds
        fills = []
        self._request_exit('episode_end', started)
        while (self.quantity or self._pending('buy')):
            if self.current_step >= self.episode_length - 1:
                self.liquidation_stop_reason = 'end_of_data'
                break
            if self.liquidation_steps >= self.liquidation_max_steps:
                self.liquidation_stop_reason = 'step_limit'
                break
            # Include buy fills racing cancellation and retry expired/partial sells.
            self._request_exit('episode_end_retry', self.current_time_seconds)
            fills.extend(self._advance_market(self.current_step + 1))
            self.liquidation_steps += 1
        else:
            self.liquidation_stop_reason = 'flat'
        self.liquidation_seconds = self.current_time_seconds - started
        return fills

    def _risk_exit_due(self, now):
        stop = any((self.simulator.liquidation_mark() / st['entry_price'] - 1) * 100 <= -self.stop_loss_pct
                   for st in self.stages)
        overdue = any(now - st['entry_time_seconds'] >= self.max_holding_seconds for st in self.stages)
        return stop, overdue

    def _check_risk(self, now):
        stop, overdue = self._risk_exit_due(now)
        if stop or overdue:
            self.loss_holding_violations += int(stop)
            self._request_exit('stop_loss' if stop else 'max_holding', now)

    def _settle_exit_state(self):
        if self._exit_requested and not self.stages and not self._pending('buy'):
            self._exit_requested = False
            self._exit_reason = None
        if (self._signal_exit_order_id is not None and not self._pending('buy')
                and not any(stage['order_id'] == self._signal_exit_order_id for stage in self.stages)):
            self._signal_exit_order_id = None

    def step(self, action):
        if self._done:
            raise RuntimeError('reset() required after episode end')
        if not self.action_space.contains(action):
            raise ValueError('invalid action')
        decision_index = len(self.episode_rewards)
        entry_order_id = None
        buy_outcome = None
        now = self.current_time_seconds
        next_index = self.current_step + 1
        horizon = len(self.episode_rewards) + 1 >= self.decision_steps
        equity_before = self.equity
        # Risk decisions use only the latest available quote and already known deadlines.
        self._check_risk(now)
        if horizon and not self.liquidation_max_steps and not self.decision_interval_seconds:
            self._request_exit('episode_end', now)
        if self._exit_requested:
            self._request_exit('risk_exit_retry', now)
            if action == 1:
                self.buy_action_outcomes['risk_exit_active'] += 1
                buy_outcome = 'risk_exit_active'
        elif self._signal_exit_order_id is not None:
            self._request_stage_exit(now)
            if action == 1:
                self.buy_action_outcomes['signal_exit_active'] += 1
                buy_outcome = 'signal_exit_active'
        elif action == 1:
            open_order_ids = {st['order_id'] for st in self.stages} | {o.order_id for o in self._pending('buy')}
            within_limit = self.max_trades_per_episode is None or len(self.episode_trades) < self.max_trades_per_episode
            if not self._entry_allowed():
                outcome = 'entry_signal_not_met'
            elif len(open_order_ids) >= self.max_stages:
                outcome = 'max_stages'
            elif not within_limit:
                outcome = 'max_trades'
            elif self._pending('buy'):
                outcome = 'pending_buy'
            else:
                budget, expected = self._buy_budget_and_unit_cost()
                qty = int(budget / expected)
                if qty > 0:
                    order = self.simulator.submit('buy', qty, now)
                    entry_order_id = int(order.order_id)
                    self._entry_order_decisions[entry_order_id] = decision_index
                    outcome = 'submitted'
                else:
                    outcome = 'insufficient_budget_for_one_share'
            # Exactly one outcome per BUY decision; guards retain their original priority.
            self.buy_action_outcomes[outcome] += 1
            buy_outcome = outcome
        elif action == 2 and self.stages and not self._pending('sell'):
            self._signal_exit_order_id = self.stages[0]['order_id']
            self._request_stage_exit(now)
        if action == 1:
            self._buy_decisions.append({'decision_index': decision_index,
                                        'entry_order_id': entry_order_id,
                                        'outcome': buy_outcome})
        target_time = now + self.decision_interval_seconds
        fills = []
        while True:
            current = self.current_time_seconds
            next_index = self.current_step + 1
            # Deadlines are timer events; market data is never inspected in advance.
            timers = []
            if self.stages:
                deadline = min(s['entry_time_seconds'] + self.max_holding_seconds for s in self.stages)
                if current < deadline <= self.timestamps[next_index]:
                    timers.append((deadline, 'max_holding'))
            if current < self.episode_deadline <= self.timestamps[next_index]:
                timers.append((self.episode_deadline, 'episode_end'))
            if (horizon and not self.liquidation_max_steps and self.decision_interval_seconds
                    and current < target_time <= self.timestamps[next_index]):
                timers.append((target_time, 'episode_end'))
            for timestamp, reason in sorted(timers):
                self._request_exit(reason, timestamp)
            fills.extend(self._advance_market(next_index))
            terminated = self.current_step >= self.episode_length - 1
            if self.max_trades_per_episode is not None and len(self.episode_trades) >= self.max_trades_per_episode and not self.stages:
                terminated = True
            duration_end = self.current_time_seconds >= self.episode_deadline
            if duration_end or terminated or not self.decision_interval_seconds or self.current_time_seconds >= target_time:
                horizon = horizon or duration_end
                break
            self._settle_exit_state()
            self._check_risk(self.current_time_seconds)
            if self._exit_requested:
                self._request_exit(self._exit_reason, self.current_time_seconds)
            elif self._signal_exit_order_id is not None:
                self._request_stage_exit(self.current_time_seconds)
        if horizon or terminated:
            self.policy_end_time_seconds = self.current_time_seconds
        if self.liquidation_max_steps and (horizon or terminated):
            fills.extend(self._liquidate_after_decisions())
            # This is a completed finite trading task, including its closeout outcome.
            terminated = True
        reward = (self.equity - equity_before) / self.initial_cash * 100
        self.episode_rewards.append(float(reward))
        truncated = bool(horizon and not terminated)
        self._done = bool(terminated or truncated)
        if self._done:
            # Do not invent fills at the last price when latency/depth prevents liquidation.
            self.simulator.cancel_all(self.current_time_seconds, end_of_replay=True)
        else:
            self._settle_exit_state()
        info = {'position': self.position, 'position_steps': self.position_steps,
                'current_price': self.current_price, 'current_time': self.current_time,
                'cash': self.cash, 'equity': self.equity, 'fills': [fill.__dict__ for fill in fills],
                'loss_holding_violations': self.loss_holding_violations}
        if self._done:
            info['episode'] = self._calculate_episode_metadata()
        if self.execution_action_mask:
            info['action_mask'] = self.action_masks()
        observation = np.zeros(self.observation_space.shape, np.float32) if self._done else self._get_current_observation()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _calculate_entry_diagnostics(self, fragments_by_entry):
        """Attribute existing fill accounting to the policy decision that submitted a BUY.

        Amounts are in the account's price/cash unit, not return percentages.
        Realized costs cover sold quantities only; entry_fees also include open lots.
        """
        stages_by_entry = {}
        for stage in self.stages:
            stages_by_entry.setdefault(stage['order_id'], []).append(stage)
        entries = []
        for order in self.simulator.orders:
            if order.side != 'buy' or order.order_id not in self._entry_order_decisions:
                continue
            fragments = fragments_by_entry.get(order.order_id, [])
            stages = stages_by_entry.get(order.order_id, [])
            sold = int(sum(trade['quantity'] for trade in fragments))
            matched_entry_fees = float(sum(trade['entry_fee'] for trade in fragments))
            exit_fees = float(sum(trade['exit_fee'] for trade in fragments))
            matched_fees = matched_entry_fees + exit_fees
            net_pnl = float(sum(trade['net_pnl'] for trade in fragments))
            attribution = {name: float(sum(trade[name] for trade in fragments))
                           for name in ('mid_price_pnl', 'spread_cost', 'depth_cost', 'slippage_cost')}
            residual = net_pnl - (attribution['mid_price_pnl'] - attribution['spread_cost']
                                  - attribution['depth_cost'] - attribution['slippage_cost'] - matched_fees)
            fallback_reason = fragments[0]['exit_reason'] if fragments else (
                'unfilled' if order.filled_quantity == 0 else 'unrealized')
            entries.append({
                'entry_order_id': int(order.order_id),
                'decision_index': self._entry_order_decisions[order.order_id],
                'submitted_quantity': int(order.quantity), 'filled_quantity': int(order.filled_quantity),
                'sold_quantity': sold, 'open_quantity': int(sum(stage['quantity'] for stage in stages)),
                'order_status': order.status,
                'complete': bool(order.filled_quantity > 0 and sold == order.filled_quantity and not order.active),
                'net_pnl': net_pnl, **attribution, 'matched_fees': matched_fees,
                'pnl_attribution_residual': float(residual),
                'entry_fees': matched_entry_fees + float(sum(stage['entry_fees'] for stage in stages)),
                'exit_fees': exit_fees,
                'quantity_weighted_holding_time': (
                    float(sum(trade['holding_share_seconds'] for trade in fragments)) / sold if sold else 0.0),
                'exit_reason': self._entry_exit_reasons.get(order.order_id, fallback_reason),
            })
        return {'version': 1, 'decision_count': len(self.episode_rewards),
                'buy_decisions': [decision.copy() for decision in self._buy_decisions], 'entries': entries}

    def _entry_pattern_report(self):
        credits = np.zeros(len(self.episode_rewards), dtype=np.float64)
        quantities = np.zeros(len(credits), dtype=np.float64)
        success = failure = censored = 0
        for order_id, timestamp, price, quantity in self._pattern_fills:
            target = fill_target(self._entry_target_times, self._entry_target_prices,
                                 timestamp, price, self.entry_pattern_config)
            if target is None:
                censored += quantity
                continue
            decision = self._entry_order_decisions[order_id]
            credits[decision] += target * quantity
            quantities[decision] += quantity
            if target > 0:
                success += quantity
            else:
                failure += quantity
        credits = np.divide(credits, quantities, out=np.zeros_like(credits), where=quantities > 0)
        credits *= self.entry_pattern_config['policy_coef']
        return {'config': self.entry_pattern_config.copy(), 'policy_credit': credits.tolist(),
                'success_quantity': success, 'failure_quantity': failure, 'censored_quantity': censored,
                'labeled_buy_decisions': int(np.count_nonzero(quantities)),
                'success_rate': success / (success + failure) if success + failure else None}

    def _calculate_episode_metadata(self):
        net_return = (self.equity - self.initial_cash) / self.initial_cash * 100
        trades = self.episode_trades
        net_trades = [t['net_return'] for t in trades]
        fragments_by_entry = {}
        for trade in trades:
            fragments_by_entry.setdefault(trade['entry_order_id'], []).append(trade)
        round_trips = []
        for order in self.simulator.orders:
            fragments = fragments_by_entry.get(order.order_id, [])
            sold = sum(t['quantity'] for t in fragments)
            if order.side != 'buy' or order.active or sold == 0 or sold != order.filled_quantity:
                continue
            pnl = sum(t['net_pnl'] for t in fragments)
            round_trips.append({
                'entry_order_id': order.order_id, 'quantity': sold,
                'exit_reason': self._entry_exit_reasons.get(order.order_id, fragments[0]['exit_reason']),
                'net_pnl': pnl, 'net_return': pnl / self.initial_cash * 100,
                'fill_count': len(fragments),
                'quantity_weighted_holding_time': sum(t['holding_share_seconds'] for t in fragments) / sold,
            })
        sold_quantity = sum(t['quantity'] for t in trades)
        fill_holding = float(np.mean([t['holding_time'] for t in trades])) if trades else 0.0
        fill_win_rate = float(np.mean([t['net_pnl'] > 0 for t in trades])) if trades else 0.0
        eq = np.asarray(self.equity_history)
        drawdowns = 1 - eq / np.maximum.accumulate(eq)
        event_returns = np.diff(eq) / self.initial_cash * 100
        sharpe = (float(event_returns.mean() / event_returns.std())
                  if len(event_returns) > 1 and event_returns.std() > 1e-12 else 0.0)
        attribution = {name: float(sum(t[name] for t in trades))
                       for name in ('mid_price_pnl', 'spread_cost', 'depth_cost', 'slippage_cost')}
        matched_fees = float(sum(t['entry_fee'] + t['exit_fee'] for t in trades))
        attribution['matched_fees'] = matched_fees
        attribution['pnl_attribution_residual'] = float(
            self.realized_net_pnl - (attribution['mid_price_pnl'] - attribution['spread_cost']
                                    - attribution['depth_cost'] - attribution['slippage_cost'] - matched_fees))
        exit_metrics = {}
        for reason in ('signal', 'stop_loss', 'max_holding', 'episode_end'):
            matching = [trade for trade in round_trips if trade['exit_reason'] == reason]
            reason_quantity = sum(trade['quantity'] for trade in matching)
            exit_metrics.update({
                f'exit_{reason}_round_trip_count': len(matching),
                f'exit_{reason}_quantity': reason_quantity,
                f'exit_{reason}_net_pnl': float(sum(trade['net_pnl'] for trade in matching)),
                f'exit_{reason}_holding_seconds': (sum(trade['quantity'] * trade['quantity_weighted_holding_time']
                                                     for trade in matching) / reason_quantity if reason_quantity else 0.0),
            })
        return {
            **({'entry_pattern': self._entry_pattern_report()} if self.entry_pattern_config is not None else {}),
            'total_return': float(net_return), 'net_return': float(net_return),
            'realized_net_pnl': float(self.realized_net_pnl),
            'gross_realized_pnl': float(sum(t['quantity'] * (t['exit_price'] - t['entry_price']) for t in trades)),
            'unrealized_net_pnl': float(self.equity - self.initial_cash - self.realized_net_pnl),
            'num_trades': len(trades), 'trades': trades,
            # Legacy fields/caps count FIFO sell fragments; do not silently reinterpret them.
            'avg_holding_time': fill_holding, 'win_rate': fill_win_rate,
            'fill_count': len(trades), 'fill_avg_holding_time': fill_holding,
            'fill_win_rate': fill_win_rate,
            'round_trip_count': len(round_trips), 'round_trips': round_trips,
            'subsecond_exit_quantity_ratio': sum(t['subsecond_exit_quantity'] for t in trades) / sold_quantity if sold_quantity else 0.0,
            'subsecond_round_trip_ratio': float(np.mean([t['quantity_weighted_holding_time'] <= 1.0 + 1e-9 for t in round_trips])) if round_trips else 0.0,
            'round_trip_win_rate': float(np.mean([t['net_pnl'] > 0 for t in round_trips])) if round_trips else 0.0,
            'quantity_weighted_holding_time': sum(t['holding_share_seconds'] for t in trades) / sold_quantity if sold_quantity else 0.0,
            'total_entry_fees': self.total_entry_fees, 'total_exit_fees': self.total_exit_fees,
            'total_fees': self.total_entry_fees + self.total_exit_fees,
            'avg_profit_per_trade': float(np.mean(net_trades)) if net_trades else 0.0,
            'max_drawdown': float(drawdowns.max() * 100),
            'sharpe_ratio': sharpe,
            'sharpe_ratio_kind': 'unannualized_event_nav_changes',
            'loss_holding_violations': self.loss_holding_violations,
            'buy_action_outcomes': self.buy_action_outcomes.copy(),
            'entry_diagnostics': self._calculate_entry_diagnostics(fragments_by_entry),
            'open_quantity': self.quantity,
            'max_open_holding_seconds': max((self.current_time_seconds - s['entry_time_seconds'] for s in self.stages), default=0.0),
            'liquidation_complete': self.quantity == 0,
            'liquidation_steps': self.liquidation_steps,
            'liquidation_seconds': self.liquidation_seconds,
            'liquidation_stop_reason': self.liquidation_stop_reason,
            'market_steps_taken': self.current_step - (self.seq_len - 1),
            'episode_start_time_seconds': self.episode_start_time_seconds,
            'episode_end_time_seconds': self.current_time_seconds,
            'episode_duration_seconds': self.current_time_seconds - self.episode_start_time_seconds,
            'policy_duration_seconds': getattr(self, 'policy_end_time_seconds', self.current_time_seconds) - self.episode_start_time_seconds,
            'decision_interval_seconds': self.decision_interval_seconds,
            'configured_episode_duration_seconds': self.episode_duration_seconds,
            'attribution_reference_kind': 'quote_mid_or_last_price_if_one_side_missing',
            **attribution, **exit_metrics,
            'episode_length': self.episode_length, 'steps_taken': len(self.episode_rewards),
            'market_context': self.market_context, 'episode_key': getattr(self, 'episode_key', {}),
            'price_source': self.price_source, **self.simulator.summary()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['conn'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = None

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
