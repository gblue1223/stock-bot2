"""Regression coverage for causal data production and legacy replay migration."""
import json
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

from ai_trader.grpo.data_extractor import extract_data, get_feature_columns
from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from lib.market_data import canonical_feature_columns, parse_time_seconds, times_to_seconds, execution_arrays
from lib.normalization import compute_stock_name_scalar, compute_stock_name_scalar_batch, compute_time_features_batch, compute_time_features
from scripts.data.generate_datasets import merge_csv_files
from scripts.data.normalize_datasets import apply_feature_normalization, ensure_table_duckdb, fill_missing_values


def market_frame(times=(100000000, 90000000, 90000000, 90001000)):
    columns = canonical_feature_columns()
    frame = pd.DataFrame({c: np.arange(len(times), dtype=float) + 1 for c in columns})
    frame["현재가"] = 10000 + np.arange(len(times)) * 10
    frame["번호"] = np.arange(len(times))
    frame["날짜"], frame["종목코드"], frame["종목명"] = "20260908", "000001", "TEST"
    frame["시간"] = [str(float(t)) for t in times]
    frame["매도호가1"], frame["매수호가1"] = frame["현재가"] + 10, frame["현재가"] - 10
    frame["매도호가수량1"], frame["매수호가수량1"] = 50, 70
    return frame


def test_time_conversion_preserves_milliseconds_and_rejects_invalid():
    result = times_to_seconds(["90023133.0", "105957916.0"])
    np.testing.assert_allclose(result, [32423.133, 39597.916])
    assert parse_time_seconds("09:00:23.133") == pytest.approx(result[0])
    assert np.isnan(parse_time_seconds("invalid"))
    assert np.isnan(parse_time_seconds("246000000"))


def test_source_merge_retains_event_time_and_raw_depth_without_future_fill():
    frames = {
        "execution": pd.DataFrame({"번호": [1, 3], "시간": [90000000, 90002000], "현재가": [10000, 10010], "등락률": [1, 1.1]}),
        "orderbook": pd.DataFrame({"번호": [2], "시간": [90001000], "매도호가1": [10010], "매도호가수량1": [100]}),
        "trader": pd.DataFrame({"번호": [4], "시간": [90003000], "거래회전율": [0.2]}),
    }
    with patch("scripts.data.generate_datasets.load_and_clean_csv", side_effect=lambda k: frames[k].copy()):
        out = merge_csv_files({k: k for k in frames}, "000001", "TEST")
    assert out["시간"].tolist() == [90000000, 90001000, 90002000, 90003000]
    assert out["매도호가1"].tolist() == [0, 10010, 10010, 10010]
    assert out["매도대기금액1"].tolist() == [0, 1.001, 1.001, 1.001]
    assert out["거래회전율"].tolist() == [0, 0, 0, 0.2]
    assert out["현재가"].tolist() == [10000, 10000, 10010, 10010]
    assert out["호가시간"].tolist() == [0, 90001000, 90001000, 90001000]
    np.testing.assert_array_equal(execution_arrays(out)["quote_timestamp"], [0, 32401, 32401, 32401])


def test_fill_and_derived_features_are_prefix_invariant():
    assert fill_missing_values(pd.DataFrame({"등락률": [None, 5]}))["등락률"].tolist() == [0, 5]
    frame = market_frame((90000000, 90001000, 90002000))
    short = apply_feature_normalization(frame.iloc[:2].copy())
    long = apply_feature_normalization(frame.copy()).iloc[:2]
    pd.testing.assert_frame_equal(short, long)
    assert not long.columns.duplicated().any()
    assert len(compute_time_features_batch(frame["시간"])) == 3
    batch = compute_time_features_batch(pd.Series(["90023133.0"]))
    np.testing.assert_allclose([x.iloc[0] for x in batch], compute_time_features("90023133.0"))
    assert batch[2].iloc[0] == pytest.approx((23.133 - 10800) / 10800)
    names = pd.Series(["삼성전자", "다른종목"])
    assert compute_stock_name_scalar_batch(names).iloc[0] == compute_stock_name_scalar("삼성전자")


def test_database_time_remains_numeric_after_schema_updates():
    with duckdb.connect(":memory:") as conn:
        frame = market_frame((90000000, 100000000))
        ensure_table_duckdb(conn, frame)
        conn.register("source", frame)
        conn.execute("INSERT INTO datasets SELECT * FROM source")
        ensure_table_duckdb(conn, frame)
        assert conn.execute('SELECT "시간" FROM datasets ORDER BY "시간"').fetchall() == [(90000000,), (100000000,)]


def test_extraction_orders_by_event_time_and_separates_execution_prices(tmp_path):
    frame = market_frame()
    source = tmp_path / "source.duckdb"
    output = tmp_path / "episodes"
    with duckdb.connect(str(source)) as conn:
        conn.register("source", frame)
        conn.execute("CREATE TABLE datasets AS SELECT * FROM source")
        # Database physical order and extra execution columns cannot select features.
        assert get_feature_columns(conn, "datasets", 27) == canonical_feature_columns()
        conn.execute('ALTER TABLE datasets DROP COLUMN "등락률"')
        with pytest.raises(ValueError, match="Missing"):
            get_feature_columns(conn, "datasets", 27)
        conn.execute('ALTER TABLE datasets ADD COLUMN "등락률" DOUBLE DEFAULT 1')
    assert extract_data(str(source), "datasets", str(output), 1, 27, 100, price_unit="krw")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["schema_version"] == 2
    with np.load(output / manifest["episodes"][0]["file_path"]) as data:
        np.testing.assert_array_equal(data["execution_last_price"], [10010, 10020, 10030, 10000])
        np.testing.assert_array_equal(data["execution_ask_prices"][:, 0], [10020, 10030, 10040, 10010])
        assert np.all(np.diff(times_to_seconds(data["metadata"][:, 2])) >= 0)
    with pytest.raises(FileExistsError):
        extract_data(str(source), "datasets", str(output), 1, 27, 100)


def legacy_episode(directory, name="first.npz", offset=0):
    columns = ["현재가", "등락률", "누적거래대금"] + [
        f"{side}대기금액{i}" for side in ("매도", "매수") for i in range(1, 11)
    ] + ["종목명_scalar", "시간_sin", "시간_cos", "시간_scalar"]
    values = np.ones((4, 27), dtype=np.float32)
    values[:, 0] = np.asarray([0.010, 0.011, 0.012, 0.013]) + offset
    values[:, 1] = [2, 3, 4, 5]
    metadata = np.asarray([["000001", "20260908", t] for t in ["100000000.0", "90000000.0", "90000000.0", "90001000.0"]])
    np.savez_compressed(directory / name, features=values, metadata=metadata)
    return columns


def test_legacy_cache_is_raw_stable_sorted_price_scaled_and_bounded(tmp_path):
    columns = legacy_episode(tmp_path)
    legacy_episode(tmp_path, "second.npz", 0.02)
    manifest = {"metadata": {"feature_columns": columns, "return_rate_index": 1},
                "episodes": [{"file_path": name, "stock_code": "000001", "date": 20260908, "length": 4}
                             for name in ("first.npz", "second.npz")]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    GRPOScalpingEnvXLSTM.clear_episode_cache()
    env = GRPOScalpingEnvXLSTM(extracted_dir=str(tmp_path), seq_len=1, cache_max_bytes=1500)
    _, info = env.reset(seed=42, options={"episode_index": 0, "start_index": 0})
    assert env.episode_data[:, 1].tolist() == [3, 4, 5, 2]  # raw percentages, stable equal-time order
    np.testing.assert_allclose(env.prices, [11000, 12000, 13000, 10000], rtol=1e-6)
    assert "bid_prices" not in env.episode_execution  # never invent real order-book depth
    original = env.episode_data.copy()
    env.episode_data[:] = 999
    env.reset(seed=42, options={"episode_index": 0, "start_index": 0})
    np.testing.assert_array_equal(env.episode_data, original)
    env.reset(seed=42, options={"episode_index": 1, "start_index": 0})
    assert env._episode_cache_bytes <= 1500
    assert len(env._episode_cache) <= 1
    assert info["episode_key"]["start_index"] == 0
    env.close()
    GRPOScalpingEnvXLSTM.clear_episode_cache()
