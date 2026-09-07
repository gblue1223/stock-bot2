import json

import numpy as np
import pytest
import torch

from ai_trader.grpo.backtest import main, run_backtest
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from lib.observations import ObservationBuilder


def checkpoint_fixture(tmp_path):
    directory = tmp_path / 'episodes'
    directory.mkdir()
    dates = ['20260101', '20260102', '20260103']
    columns = ['현재가', '등락률', '누적거래대금']
    episodes = []
    for date in dates:
        prices = np.arange(8) + 10000.
        features = np.column_stack((prices, np.arange(8) / 100, np.arange(8) + 1000)).astype(np.float32)
        metadata = np.array([['000001', date, str(90000000 + i * 1000)] for i in range(8)])
        filename = date + '.npz'
        np.savez(directory / filename, features=features, metadata=metadata,
                 execution_last_price=prices, execution_bid_prices=prices[:, None] - 10,
                 execution_ask_prices=prices[:, None] + 10,
                 execution_bid_sizes=np.full((8, 1), 100), execution_ask_sizes=np.full((8, 1), 100))
        episodes.append(dict(file_path=filename, stock_code='000001', date=date, length=8))
    (directory / 'manifest.json').write_text(json.dumps({
        'metadata': dict(schema_version=2, feature_columns=columns, price_unit='krw', return_rate_index=1),
        'episodes': episodes}), encoding='utf-8')
    builder = ObservationBuilder(columns, seq_len=4, rolling_window_size=4, rolling_min_samples=2)
    policy = GRPOPolicyE2EXLSTM(obs_dim=18, cnn_channels=4, rnn_hidden_dim=4, fc_hidden_dim=8)
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.copy_(torch.tensor([5., 0., 0.]))
    config = dict(extracted_dir=str(directory), episode_steps=3, initial_cash=1000000,
                  transaction_cost_rate=.00015, buy_tax_rate=0, sell_tax_rate=.0018,
                  stop_loss_pct=2, execution_config={'require_order_book': True}, evaluation_episodes=2)
    path = tmp_path / 'policy.pt'
    torch.save(dict(policy_state_dict=policy.state_dict(), observation_schema=builder.schema,
                    extra_state=dict(training_config=config,
                        date_splits={'train': dates[:1], 'validation': dates[1:2], 'test': dates[2:]})), path)
    return path


def test_backtest_cli_reuses_recorded_holdout_settings_and_writes_json(tmp_path):
    path = checkpoint_fixture(tmp_path)
    output = tmp_path / 'report.json'
    assert main(['--policy', str(path), '--output', str(output), '--order-latency-ms', '250']) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['dates'] == ['20260103']
    assert report['execution_config']['order_latency_ms'] == 250
    assert report['metrics']['mean_net_return'] == 0
    assert report['metrics']['synthetic_execution_episodes'] == 0
    assert report['costs']['sell_tax_rate'] == .0018


def test_backtest_rejects_unrecorded_holdout_or_zero_episodes(tmp_path):
    path = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match='positive'):
        run_backtest(path, episodes=0)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['extra_state'].pop('date_splits')
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='date_splits'):
        run_backtest(path)
