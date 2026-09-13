import json
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

import ai_trader.grpo.data_extractor as extractor
from lib.market_data import canonical_feature_columns


def raw_database(path, unit='krw', two_stocks=False):
    frames = []
    for stock in ('000001', '000002') if two_stocks else ('000001',):
        frame = pd.DataFrame({'종목코드': stock, '날짜': '20260101', '종목명': 'TEST',
            '시간': ['110000001.0', '90000000.0', '90000000.0', '85959999.0', '100000000.0'],
            '번호': [5, 2, 1, 0, 4], '현재가': [10500., 10100., 10000., 9990., 10400.],
            '등락률': [5., 1., 0., -.1, 4.], '누적거래대금': [5., 2., 1., 0., 4.]})
        scale = 1 if unit == 'krw' else 1e6
        for side, difference in [('매수', -10), ('매도', 10)]:
            for level in range(1, 11):
                frame[f'{side}호가{level}'] = (frame['현재가'] + level * difference) / scale
                frame[f'{side}호가수량{level}'] = 10 * level
        frame['현재가'] /= scale
        frames.append(frame)
    with duckdb.connect(str(path)) as conn:
        conn.register('raw', pd.concat(frames))
        conn.execute('CREATE TABLE datasets AS SELECT * FROM raw')


@pytest.mark.parametrize('unit,executor_kind', [('krw', 'thread'), ('million_krw', 'process')])
@pytest.mark.parametrize('feature_count', [27, 29])
def test_raw_quotes_create_notional_features_and_keep_execution_units(tmp_path, unit, executor_kind, feature_count):
    source, output = tmp_path / 'raw.duckdb', tmp_path / 'episodes'
    raw_database(source, unit)
    assert extractor.extract_data(str(source), 'datasets', str(output), 1, feature_count, 2,
        price_unit=unit, time_start=90000000, time_end=110000000, workers=2, compression_level=1,
        executor_kind=executor_kind)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['metadata']['feature_transform'] == 'raw'
    assert not manifest['metadata']['source_has_quote_timestamp']
    assert len(manifest['episodes']) == 1
    with np.load(output / manifest['episodes'][0]['file_path'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['execution_last_price'], [10000, 10100, 10400])
        np.testing.assert_array_equal(data['execution_ask_prices'][:, 0], [10010, 10110, 10410])
        assert data['execution_bid_sizes'].shape == (3, 10)
        idx = canonical_feature_columns().index('매도대기금액1')
        np.testing.assert_allclose(data['features'][:, idx], [.1001, .1011, .1041], rtol=1e-6)
        assert 'execution_quote_timestamp' not in data.files
        if feature_count == 29:
            np.testing.assert_allclose(data['features'][:, -2], 10000 if unit == 'krw' else .01)
            np.testing.assert_allclose(data['features'][:, -1], [0., 1., 4.], atol=1e-5)
    with pytest.raises(FileExistsError):
        extractor.extract_data(str(source), 'datasets', str(output), 1, 27, 2, price_unit=unit)


def test_interrupted_extraction_resumes_without_rewriting_completed_files(tmp_path):
    source, output = tmp_path / 'raw.duckdb', tmp_path / 'episodes'
    raw_database(source, two_stocks=True)
    original_writer = extractor._write_episode

    def interrupted_writer(target, arrays, compression_level):
        if '000002' in target.name:
            raise OSError('simulated interruption')
        original_writer(target, arrays, compression_level)

    with patch.object(extractor, '_write_episode', side_effect=interrupted_writer):
        with pytest.raises(OSError, match='simulated interruption'):
            extractor.extract_data(str(source), 'datasets', str(output), 1, 27, 2, workers=1)
    assert not (output / 'manifest.json').exists()
    first = output / 'episode_000001_20260101.npz'
    before = (first.read_bytes(), first.stat().st_mtime_ns)
    journal = output / 'completed.jsonl'
    assert len(journal.read_text(encoding='utf-8').splitlines()) == 1
    with journal.open('a', encoding='utf-8') as handle:
        handle.write('{"unfinished":')
    assert extractor.extract_data(str(source), 'datasets', str(output), 1, 27, 2, workers=2, resume=True)
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert [entry['stock_code'] for entry in manifest['episodes']] == ['000001', '000002']
    assert len(journal.read_text(encoding='utf-8').splitlines()) == 2


def test_resume_rejects_source_changes_and_different_settings(tmp_path):
    source, output = tmp_path / 'raw.duckdb', tmp_path / 'episodes'
    raw_database(source)
    with patch.object(extractor, '_write_episode', side_effect=OSError('interrupt')):
        with pytest.raises(OSError):
            extractor.extract_data(str(source), 'datasets', str(output), 1, 27, 2)
    with pytest.raises(ValueError, match='fingerprint/settings'):
        extractor.extract_data(str(source), 'datasets', str(output), 2, 27, 2, resume=True)
    with duckdb.connect(str(source)) as conn:
        conn.execute('UPDATE datasets SET "현재가" = 12345')
    with pytest.raises(ValueError, match='fingerprint/settings'):
        extractor.extract_data(str(source), 'datasets', str(output), 1, 27, 2, resume=True)
