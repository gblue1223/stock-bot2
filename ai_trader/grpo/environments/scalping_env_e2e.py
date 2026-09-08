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
from .execution import ExecutionSimulator

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
                 execution_config=None, allowed_dates=None, price_scale=1.0, max_stages=1):
        super().__init__()
        self.max_stages = validate_max_stages(max_stages)
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
            feature_price_unit=self.price_unit, max_stages=self.max_stages)
        self.obs_dim = self.expected_features + self.MAX_STAGES * 3
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
        if all(c in self._db_columns for c in columns):
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
        features = df[self.feature_columns].ffill().fillna(0).to_numpy(dtype=np.float32)
        metadata = df[['종목코드', '날짜', '시간']].to_numpy()
        from lib.market_data import execution_arrays
        execution = execution_arrays(df)
        for key in ('last_price', 'bid_prices', 'ask_prices'):
            if key in execution:
                execution[key] = execution[key] * self.price_scale
        needed = len(features) if self.max_episode_steps is None else min(len(features), self.seq_len + self.max_episode_steps)
        start = self._reset_options.get('start_index')
        start = int(self.np_random.integers(len(features) - needed + 1)) if start is None else int(start)
        if start < 0 or start + self.seq_len >= len(features):
            raise ValueError('start_index does not leave history and a future event')
        end = min(start + needed, len(features))
        self.episode_execution = {key: value[start:end] for key, value in execution.items()}
        self.episode_key = {'stock_code': str(stock), 'date': str(date), 'start_index': start}
        return features[start:end], metadata[start:end]

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
        self.stages, self.episode_trades, self.episode_rewards = [], [], []
        self.cash, self.realized_net_pnl = self.initial_cash, 0.0
        self.loss_holding_violations = 0
        self._exit_requested, self._done = False, False
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
            'execution_model': self.simulator.summary()['execution_model']}

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
            current_price=self.current_price, current_time_seconds=self.current_time_seconds)

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
        if fill.side == 'buy':
            fee = value * self.buy_fee_rate
            self.cash -= value + fee
            stage = next((s for s in self.stages if s['order_id'] == fill.order_id), None)
            if stage is None:
                self.stages.append({'order_id': fill.order_id, 'quantity': fill.quantity,
                    'entry_price': fill.price, 'entry_time_seconds': fill.timestamp,
                    'entry_time': self.current_time, 'entry_fees': fee})
            else:
                qty = stage['quantity'] + fill.quantity
                stage['entry_price'] = (stage['entry_price'] * stage['quantity'] + value) / qty
                stage['quantity'] = qty
                stage['entry_fees'] += fee
        else:
            self.cash += value * (1 - self.sell_fee_rate)
            remaining = fill.quantity
            while remaining:
                stage = self.stages[0]
                qty = min(remaining, stage['quantity'])
                entry_fee = stage['entry_fees'] * qty / stage['quantity']
                exit_fee = qty * fill.price * self.sell_fee_rate
                pnl = qty * (fill.price - stage['entry_price']) - entry_fee - exit_fee
                self.realized_net_pnl += pnl
                self.episode_trades.append({
                    'entry_price': stage['entry_price'], 'exit_price': fill.price,
                    'quantity': qty, 'entry_fee': entry_fee, 'exit_fee': exit_fee,
                    'holding_time': fill.timestamp - stage['entry_time_seconds'],
                    'profit_rate': fill.price / stage['entry_price'] - 1,
                    'net_pnl': pnl, 'net_return': pnl / self.initial_cash * 100,
                    'reward': pnl / self.initial_cash * 100,
                    'weight': qty * stage['entry_price'] / self.initial_cash,
                    'exit_reason': fill.reason, 'order_id': fill.order_id})
                stage['quantity'] -= qty
                stage['entry_fees'] -= entry_fee
                remaining -= qty
                if stage['quantity'] == 0:
                    self.stages.pop(0)

    def _pending(self, side):
        return [o for o in self.simulator.orders if o.active and o.side == side]

    def _request_exit(self, reason, timestamp):
        self._exit_requested = True
        self.simulator.cancel_all(timestamp, side='buy')
        pending_qty = sum(o.remaining for o in self._pending('sell'))
        qty = self.quantity - pending_qty
        if qty > 0:
            self.simulator.submit('sell', qty, timestamp, reason=reason)

    def _equity(self):
        return self.cash + self.quantity * self.simulator.liquidation_mark() * (1 - self.sell_fee_rate)

    def step(self, action):
        if self._done:
            raise RuntimeError('reset() required after episode end')
        if not self.action_space.contains(action):
            raise ValueError('invalid action')
        now = self.current_time_seconds
        next_index = self.current_step + 1
        horizon = next_index >= self.episode_length - 1
        if self.max_episode_steps is not None:
            horizon |= next_index - (self.seq_len - 1) >= self.max_episode_steps
        # Risk decisions use only the latest available quote and already known deadlines.
        stop = any((self.simulator.liquidation_mark() / st['entry_price'] - 1) * 100 <= -self.stop_loss_pct for st in self.stages)
        overdue = any(now - st['entry_time_seconds'] >= self.max_holding_seconds for st in self.stages)
        if stop or overdue:
            self.loss_holding_violations += int(stop)
            self._request_exit('stop_loss' if stop else 'max_holding', now)
        if horizon:
            self._request_exit('episode_end', now)
        if self._exit_requested:
            self._request_exit('risk_exit_retry', now)
        elif action == 1:
            open_order_ids = {st['order_id'] for st in self.stages} | {o.order_id for o in self._pending('buy')}
            within_limit = self.max_trades_per_episode is None or len(self.episode_trades) < self.max_trades_per_episode
            if len(open_order_ids) < self.max_stages and within_limit and not self._pending('buy'):
                ask = self.simulator.snapshot.asks[0][0] if self.simulator.snapshot.asks else self.current_price
                expected = self.simulator.execution_price(ask, 'buy') * (1 + self.buy_fee_rate)
                qty = int(min(self.cash, self.initial_cash / self.max_stages) / expected)
                if qty > 0:
                    self.simulator.submit('buy', qty, now)
        elif action == 2 and self.stages and not self._pending('sell'):
            self.simulator.submit('sell', self.stages[0]['quantity'], now)
        # A holding deadline is a timer event, even when no market tick arrives then.
        if self.stages:
            deadline = min(s['entry_time_seconds'] + self.max_holding_seconds for s in self.stages)
            if now < deadline <= self.timestamps[next_index]:
                self._request_exit('max_holding', deadline)
        self.current_step = next_index
        self._update_market_fields()
        fills = self.simulator.process(
            self._snapshot(next_index), cash_available=self.cash, sell_available=self.quantity,
            buy_fee_rate=self.buy_fee_rate, sell_fee_rate=self.sell_fee_rate)
        for fill in fills:
            self._apply_fill(fill)
        self.equity = self._equity()
        reward = (self.equity - self.equity_history[-1]) / self.initial_cash * 100
        self.episode_rewards.append(float(reward))
        self.equity_history.append(self.equity)
        terminated = self.current_step >= self.episode_length - 1
        if self.max_trades_per_episode is not None and len(self.episode_trades) >= self.max_trades_per_episode and not self.stages:
            terminated = True
        truncated = bool(horizon and not terminated)
        self._done = bool(terminated or truncated)
        if self._done:
            # Do not invent fills at the last price when latency/depth prevents liquidation.
            self.simulator.cancel_all(self.current_time_seconds, end_of_replay=True)
        elif self._exit_requested and not self.stages and not self._pending('buy'):
            self._exit_requested = False
        info = {'position': self.position, 'position_steps': self.position_steps,
                'current_price': self.current_price, 'current_time': self.current_time,
                'cash': self.cash, 'equity': self.equity, 'fills': [fill.__dict__ for fill in fills],
                'loss_holding_violations': self.loss_holding_violations}
        if self._done:
            info['episode'] = self._calculate_episode_metadata()
        observation = np.zeros(self.observation_space.shape, np.float32) if self._done else self._get_current_observation()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _calculate_episode_metadata(self):
        net_return = (self.equity - self.initial_cash) / self.initial_cash * 100
        trades = self.episode_trades
        net_trades = [t['net_return'] for t in trades]
        eq = np.asarray(self.equity_history)
        drawdowns = 1 - eq / np.maximum.accumulate(eq)
        rewards = np.asarray(self.episode_rewards)
        sharpe = float(rewards.mean() / rewards.std()) if len(rewards) > 1 and rewards.std() > 1e-12 else 0.0
        return {
            'total_return': float(net_return), 'net_return': float(net_return),
            'realized_net_pnl': float(self.realized_net_pnl),
            'unrealized_net_pnl': float(self.equity - self.initial_cash - self.realized_net_pnl),
            'num_trades': len(trades), 'trades': trades,
            'avg_holding_time': float(np.mean([t['holding_time'] for t in trades])) if trades else 0.0,
            'win_rate': float(np.mean([t['net_pnl'] > 0 for t in trades])) if trades else 0.0,
            'avg_profit_per_trade': float(np.mean(net_trades)) if net_trades else 0.0,
            'max_drawdown': float(drawdowns.max() * 100),
            'sharpe_ratio': sharpe, 'sharpe_ratio_kind': 'unannualized_event_nav_changes',
            'loss_holding_violations': self.loss_holding_violations,
            'open_quantity': self.quantity,
            'max_open_holding_seconds': max((self.current_time_seconds - s['entry_time_seconds'] for s in self.stages), default=0.0),
            'liquidation_complete': self.quantity == 0,
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
