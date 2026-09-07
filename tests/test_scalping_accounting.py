import numpy as np
import pytest

from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv


class ReplayFixture(GRPOScalpingEnv):
    def __init__(self, prices, seconds=None, execution=None, **kwargs):
        self.input_prices = np.asarray(prices, dtype=np.float64)
        self.input_seconds = np.arange(len(prices), dtype=float) if seconds is None else np.asarray(seconds)
        self.input_execution = execution or {}
        defaults = dict(seq_len=1, expected_features=2, initial_cash=10000,
                        transaction_cost_rate=0, sell_tax_rate=0,
                        rolling_min_samples=1, stop_loss_pct=50,
                        execution_config=dict(order_latency_ms=0, slippage_bps=0, spread_bps=0))
        defaults.update(kwargs)
        super().__init__('unused', **defaults)

    def _init_metadata_from_db(self):
        self.feature_columns = ['현재가', '등락률']
        self.return_rate_index = 1
        self.accum_trade_value_index = None

    def _ensure_db_connection(self):
        pass

    def _sample_episode_start(self, max_attempts=100):
        features = np.column_stack((self.input_prices, (self.input_prices / self.input_prices[0] - 1) * 100))
        packed = []
        for seconds in self.input_seconds:
            total_ms = int(round((9 * 3600 + seconds) * 1000))
            h, rem = divmod(total_ms, 3600000)
            m, rem = divmod(rem, 60000)
            s, ms = divmod(rem, 1000)
            packed.append(h * 10000000 + m * 100000 + s * 1000 + ms)
        metadata = np.array([['TEST', '20260908', t] for t in packed], dtype=object)
        self.episode_execution = {'last_price': self.input_prices.copy(), **self.input_execution}
        return features.astype(np.float32), metadata


def finish(env, actions):
    env.reset()
    rewards = []
    for action in actions:
        _, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            return sum(rewards), info['episode']
    raise AssertionError('episode did not end')


def test_last_tick_loss_is_used_and_reward_reconciles_to_cash():
    env = ReplayFixture([100, 100, 101, 90])
    reward, info = finish(env, [1, 0, 0])
    assert info['trades'][-1]['exit_price'] == 90
    assert info['total_return'] == pytest.approx(-2)
    assert reward == pytest.approx(info['net_return'])
    assert env.cash == pytest.approx(9800)
    assert info['liquidation_complete']


def test_small_price_gain_after_fees_remains_a_negative_reward():
    env = ReplayFixture([100, 100, 100.01, 100.01], transaction_cost_rate=.00015, sell_tax_rate=.0018,
                        win_bonus=5, buy_signal_bonus=.5, no_trade_penalty=50)
    reward, info = finish(env, [1, 0, 0])
    assert reward < 0
    assert reward == pytest.approx(info['realized_net_pnl'] / env.initial_cash * 100)
    assert info['win_rate'] == 0


def test_no_trade_is_zero_even_with_legacy_no_trade_penalty():
    env = ReplayFixture([100, 110, 90], no_trade_penalty=50)
    reward, info = finish(env, [0, 0])
    assert reward == info['total_return'] == 0


def test_max_holding_timer_fires_across_market_gap_and_does_not_clip_time():
    env = ReplayFixture([100, 100, 110, 110, 110], seconds=[0, 1, 601, 602, 603], max_holding_seconds=300)
    env.reset()
    env.step(1)
    _, _, _, _, info = env.step(0)
    # No quote arrived before the timer order expired: do not invent a fill.
    assert info['position'] == 1
    assert env.simulator.summary()['expired_orders'] == 1
    _, _, _, _, info = env.step(0)
    assert info['position'] == 0
    assert env.episode_trades[0]['holding_time'] == 601
    assert env.episode_trades[0]['exit_reason'] == 'max_holding'
    assert env._calculate_seconds_diff(90000000, 93000000) == 1800


def test_unfilled_terminal_liquidation_is_marked_and_reported_not_invented():
    n = 4
    execution = {'bid_prices': np.full((n, 1), 99.), 'ask_prices': np.full((n, 1), 101.),
                 'bid_sizes': np.array([[100], [100], [0], [0]]), 'ask_sizes': np.full((n, 1), 100.)}
    env = ReplayFixture([100] * n, execution=execution)
    reward, info = finish(env, [1, 0, 0])
    assert not info['liquidation_complete']
    assert info['open_quantity'] > 0 and info['num_trades'] == 0
    assert reward == pytest.approx(info['total_return'])


def test_raw_price_unchanged_by_normalization_and_future_tail():
    a = ReplayFixture([100, 100.1, 100.2, 100.3], seq_len=2)
    b = ReplayFixture([100, 100.1, 500, 600], seq_len=2)
    obs_a, _ = a.reset()
    obs_b, _ = b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    assert a.current_price == b.current_price == 100.1


def test_backwards_time_fails_instead_of_hiding_bad_data():
    env = ReplayFixture([100, 100, 100], seconds=[0, 2, 1])
    with pytest.raises(ValueError, match='monotonic'):
        env.reset()
