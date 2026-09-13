"""Opening references remain causal across extraction, slicing and live inputs."""
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from ai_trader.grpo.data_extractor import extract_data
from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.inference.enhanced_grpo_infer_xlstm import GRPOInferenceE2EXLSTM
from lib.market_data import add_opening_price_features, canonical_feature_columns
from lib.observations import ObservationBuilder


def test_opening_price_is_causal_and_ignores_premarket_and_later_changes():
    frame = pd.DataFrame({'시간': [85959999, 90000000, 90000100, 100000000],
                          '현재가': [90., np.nan, 100., 110.]})
    result = add_opening_price_features(frame)
    assert result['시초가'].tolist() == [0., 0., 100., 100.]
    np.testing.assert_allclose(result['시초가대비등락률'], [0., 0., 0., 10.])
    pd.testing.assert_frame_equal(add_opening_price_features(frame.iloc[:3]), result.iloc[:3])
    frame['시가'] = [80., 95., 99., 99.]
    assert add_opening_price_features(frame)['시초가'].tolist() == [0., 95., 95., 95.]
    # A 10:00 sample cannot stand in for the missing market open.
    assert add_opening_price_features(frame.drop(columns='시가').iloc[-1:])['시초가'].tolist() == [0.]


@pytest.mark.parametrize('unit,scale', [('krw', 1.), ('million_krw', 1e6)])
def test_extraction_retains_open_before_time_filter_and_random_episode_slice(tmp_path, unit, scale):
    columns = canonical_feature_columns(27)
    frame = pd.DataFrame({c: np.ones(6) for c in columns})
    frame['현재가'] = np.array([9990., 10000., 10100., 11000., 11100., 11200.]) / scale
    frame['시간'] = [85959999, 90000000, 90000100, 100000000, 100001000, 100002000]
    frame['번호'] = np.arange(6)
    frame['종목코드'], frame['날짜'] = '000001', '20260913'
    source, output = tmp_path / 'source.duckdb', tmp_path / 'episodes'
    with duckdb.connect(str(source)) as conn:
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        conn.register('input_frame', shuffled)
        conn.execute('CREATE TABLE datasets AS SELECT * FROM input_frame')
    direct = GRPOScalpingEnvXLSTM(db_path=str(source), seq_len=1, expected_features=29,
                                price_scale=scale, max_episode_steps=1, rolling_min_samples=1)
    try:
        direct.reset(options={'episode_index': 0, 'start_index': 3})
        assert direct.episode_data[0, -2] == pytest.approx(10000. / scale)
    finally:
        direct.close()
    assert extract_data(str(source), 'datasets', str(output), 1, 29, 1, price_unit=unit,
                        time_start=100000000, time_end=100002000)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['metadata']['feature_columns'] == canonical_feature_columns(29)
    with np.load(output / manifest['episodes'][0]['file_path']) as data:
        np.testing.assert_allclose(data['features'][:, -2], 10000. / scale)
        np.testing.assert_allclose(data['features'][:, -1], [10., 11., 12.])
        np.testing.assert_array_equal(data['execution_last_price'], [11000., 11100., 11200.])
    env = GRPOScalpingEnvXLSTM(extracted_dir=str(output), seq_len=1, expected_features=29,
                             max_episode_steps=1, rolling_min_samples=1)
    try:
        observation, _ = env.reset(options={'episode_index': 0, 'start_index': 1})
        assert env.episode_data[0, -2] == pytest.approx(10000. / scale)
        assert observation[-1, 28] == pytest.approx(.11)
    finally:
        env.close()


def test_missing_opening_price_is_reported_instead_of_invented(tmp_path):
    columns = canonical_feature_columns(27)
    frame = pd.DataFrame({c: [1., 2.] for c in columns})
    frame['종목코드'], frame['날짜'] = '000001', '20260913'
    frame['시간'] = [100000000, 100001000]
    source = tmp_path / 'source.duckdb'
    with duckdb.connect(str(source)) as conn:
        conn.register('input_frame', frame)
        conn.execute('CREATE TABLE datasets AS SELECT * FROM input_frame')
    with pytest.raises(ValueError, match='missing_opening_price'):
        extract_data(str(source), 'datasets', str(tmp_path / 'episodes'), 1, 29, 1)


def test_fixed_opening_price_survives_normalization_and_schema_roundtrip():
    columns = canonical_feature_columns(29)
    builder = ObservationBuilder(columns, seq_len=3, rolling_window_size=3, rolling_min_samples=1)
    raw = np.ones((3, 29))
    raw[:, columns.index('현재가')] = [100., 105., 110.]
    raw[:, -2] = 100.
    raw[:, -1] = [0., 5., 10.]
    result = builder.normalize(raw)
    np.testing.assert_allclose(result[:, -2], result[0, columns.index('현재가')])
    assert result[-1, -2] != 0
    np.testing.assert_allclose(result[:, -1], [0., .05, .1])
    restored = ObservationBuilder.from_schema(builder.schema)
    np.testing.assert_array_equal(builder.build(raw), restored.build(raw))
    live = GRPOInferenceE2EXLSTM.__new__(GRPOInferenceE2EXLSTM)
    live.observation_builder = restored
    np.testing.assert_array_equal(live.build_observation(raw, feature_columns=columns), builder.build(raw))
    legacy = ObservationBuilder(canonical_feature_columns(27))
    assert legacy.schema['normalization'] == 'causal_window_log_zscore_v1'
    with pytest.raises(ValueError, match='Incompatible'):
        restored.validate_schema(legacy.schema)
