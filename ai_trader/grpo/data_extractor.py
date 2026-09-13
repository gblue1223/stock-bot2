#!/usr/bin/env python3
"""Extract causal raw episodes with a versioned feature/execution contract."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import Counter
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.market_data import (canonical_feature_columns, validate_feature_columns,
                             chronological_order, execution_arrays, PRICE_SCALES,
                             add_opening_price_features)
from lib.normalization import compute_stock_name_scalar_batch, compute_time_features_batch
from ai_trader.grpo.entry_pattern import (validate_entry_pattern, entry_signal, signal_ranges,
                                          cumulative_buy_from_signed_trades)

logger = logging.getLogger("DataExtractor")


def _identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def _depth_sources(name):
    for side in ('매도', '매수'):
        prefix = side + '대기금액'
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            level = name[len(prefix):]
            return [f'{side}호가{level}', f'{side}호가수량{level}']
    return []


def get_feature_columns(conn, table_name: str, expected_features: int) -> list:
    """Select a registered schema by name; database order cannot change semantics."""
    schema = conn.execute(f"DESCRIBE {_identifier(table_name)}").fetchdf()
    available = dict(zip(schema.column_name, schema.column_type))
    requested = canonical_feature_columns(expected_features)
    derived = {"종목명_scalar", "시간_sin", "시간_cos", "시간_scalar"}
    if '현재가' in available:
        derived.update({'시초가', '시초가대비등락률'})
    derivable = {name for name in requested if _depth_sources(name)
                 and set(_depth_sources(name)) <= set(available)}
    missing = set(requested) - set(available) - derived - derivable
    if missing:
        raise ValueError(f"Missing required model features: {sorted(missing)}")
    for name in (set(requested) - derived) & set(available):
        if not any(t in available[name].upper() for t in ("DOUBLE", "FLOAT", "REAL", "INT", "DECIMAL")):
            raise ValueError(f"Feature {name} must be numeric, found {available[name]}")
    return validate_feature_columns(requested, expected_features)


def _episode_arrays(frame, feature_cols, price_unit, entry_pattern_config=None,
                    buy_notional_column='cumulative_buy_notional_krw', buy_notional_scale=1.0,
                    buy_notional_source='cumulative_buy_notional'):
    """Build raw model features and genuine depth without looking beyond a row."""
    frame = frame.iloc[chronological_order(frame['시간'])].reset_index(drop=True)
    pattern_arrays = {}
    if entry_pattern_config is not None:
        from lib.market_data import times_to_seconds
        # Preserve missing/invalid source values: never manufacture BUY trades
        # from total turnover, order-book depth, or execution-strength ratios.
        prices = pd.to_numeric(frame['현재가'], errors='coerce').to_numpy(dtype=np.float64) * PRICE_SCALES[price_unit]
        if buy_notional_source == 'signed_volume_positive_buy':
            cumulative_buy = cumulative_buy_from_signed_trades(prices,
                pd.to_numeric(frame['거래량'], errors='coerce'), pd.to_numeric(frame['누적거래량'], errors='coerce'))
        else:
            cumulative_buy = pd.to_numeric(frame[buy_notional_column], errors='coerce').to_numpy(dtype=np.float64) * buy_notional_scale
        signal = entry_signal(times_to_seconds(frame['시간']), prices, cumulative_buy, entry_pattern_config)
        pattern_arrays = {'entry_signal': signal, 'cumulative_buy_notional_krw': cumulative_buy}
    if '시초가' in feature_cols:
        frame = add_opening_price_features(frame)
        if not (frame['시초가'] > 0).any():
            raise ValueError('missing_opening_price: supply 시가/시초가 or a 09:00:00 tick')
    for column in frame:
        if column not in {'날짜', '종목코드', '종목명', '시간', '번호'}:
            frame[column] = pd.to_numeric(frame[column], errors='coerce').ffill().fillna(0.0)
    for name in feature_cols:
        if name not in frame and _depth_sources(name):
            price, size = _depth_sources(name)
            # Waiting notional features use million KRW; execution prices use KRW.
            frame[name] = frame[price] * PRICE_SCALES[price_unit] * frame[size] / 1_000_000.0
    names = frame['종목명'].fillna('').astype(str) if '종목명' in frame else pd.Series('', index=frame.index)
    unique = pd.Series(names.unique())
    encoding = dict(zip(unique, compute_stock_name_scalar_batch(unique)))
    frame['종목명_scalar'] = names.map(encoding)
    sine, cosine, scalar = compute_time_features_batch(frame['시간'], use_zscore_for_scalar=False)
    frame['시간_sin'], frame['시간_cos'], frame['시간_scalar'] = sine, cosine, scalar
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite_features')
    metadata = frame[['종목코드', '날짜', '시간']].to_numpy().astype(str)
    execution = execution_arrays(frame, price_unit)
    if any(not np.isfinite(value).all() for value in execution.values()):
        raise ValueError('nonfinite_execution')
    if any(np.any(value < 0) for value in execution.values()):
        raise ValueError('negative_execution')
    return {'features': values, 'metadata': metadata, **pattern_arrays,
            **{'execution_' + key: value for key, value in execution.items()}}


def _write_episode(target, arrays, compression_level):
    partial = target.with_suffix('.npz.part')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compression_level, allowZip64=True) as archive:
        for name, values in arrays.items():
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, values, allow_pickle=False)
    os.replace(partial, target)


def _atomic_json(path, payload):
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(partial, path)


def _extract_one_episode(row, settings, conn):
    stock, date, expected_rows = row
    output = Path(settings['output'])
    entry = {'stock_code': str(stock).zfill(6), 'date': int(date)}
    if shutil.disk_usage(output).free < 5 * 1024 ** 3:
        raise OSError('Less than 5 GiB output disk space remains; resume on sufficient storage')
    cursor = conn.cursor()
    try:
        frame = cursor.execute(settings['query'], [stock, date, *settings['query_time_parameters']]).fetchdf()
    finally:
        cursor.close()
    selected = np.ones(len(frame), dtype=bool)
    if settings['time_parameters']:
        lower, upper = settings['time_parameters']
        selected = pd.to_numeric(frame['시간']).between(lower, upper).to_numpy()
    if int(selected.sum()) != expected_rows:
        raise RuntimeError('Source row count changed during extraction')
    try:
        arrays = _episode_arrays(frame, settings['feature_cols'], settings['price_unit'],
            settings.get('entry_pattern_config'), settings.get('buy_notional_column', 'cumulative_buy_notional_krw'),
            settings.get('buy_notional_scale', 1.0), settings.get('buy_notional_source', 'cumulative_buy_notional'))
        arrays = {key: values[selected] for key, values in arrays.items()}
        if 'entry_signal' in arrays and not arrays['entry_signal'][settings['seq_len'] - 1:-1].any():
            raise ValueError('no_qualifying_entry: BUY notional and price-rise conditions not met')
    except ValueError as exc:
        return {**entry, 'status': 'rejected', 'reason': str(exc), 'source_rows': len(frame)}
    del frame
    filename = f"episode_{entry['stock_code']}_{date}.npz"
    target = (output / filename).resolve()
    if target.parent != output:
        raise ValueError('Episode filename escapes the output directory')
    if target.exists() and not settings['resume']:
        raise FileExistsError(f'Refusing to overwrite {target}')
    _write_episode(target, arrays, settings['compression_level'])
    return {**entry, 'status': 'ok', 'file_path': filename,
            **({'entry_signal_ranges': signal_ranges(arrays['entry_signal'])} if 'entry_signal' in arrays else {}),
            'length': len(arrays['features']), 'bytes': target.stat().st_size,
            'execution_fields': [name for name in arrays if name.startswith('execution_')]}


def _init_process_worker(settings):
    global _worker_settings, _worker_connection
    _worker_settings = settings
    _worker_connection = duckdb.connect(settings['source'], read_only=True,
        config={'threads': settings['threads'], 'memory_limit': settings['memory_limit']})
    _worker_connection.execute('SET enable_progress_bar=false')


def _extract_process_episode(row):
    return _extract_one_episode(row, _worker_settings, _worker_connection)


def extract_data(db_path: str, table_name: str, output_dir: str, seq_len: int,
                 features: int, max_steps: int, limit: int = None, price_unit: str = "krw",
                 *, time_start=None, time_end=None, workers=1, threads=2,
                  memory_limit="4GB", compression_level=6, resume=False, executor_kind="thread",
                  entry_pattern_config=None, buy_notional_column='cumulative_buy_notional_krw',
                  buy_notional_scale=1.0, buy_notional_source='cumulative_buy_notional'):
    """Extract directly from raw depth or prepared features, with bounded workers.

    Existing output is protected. Explicit resume accepts only the same source
    fingerprint and extraction settings, then skips durably journaled episodes.
    The complete manifest is published only when every key has been processed.
    """
    entry_pattern_config = validate_entry_pattern(entry_pattern_config)
    if buy_notional_source not in ('cumulative_buy_notional', 'signed_volume_positive_buy'):
        raise ValueError('Unknown buy_notional_source')
    if not np.isfinite(buy_notional_scale) or buy_notional_scale <= 0:
        raise ValueError('buy_notional_scale must be finite and positive')
    if price_unit not in PRICE_SCALES:
        raise ValueError(f"Unknown price unit: {price_unit}")
    if seq_len < 1 or max_steps < 1 or workers < 1 or threads < 1:
        raise ValueError("Sequence/episode lengths and worker/thread counts must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if executor_kind not in ('thread', 'process'):
        raise ValueError('executor_kind must be thread or process')
    if (time_start is None) != (time_end is None):
        raise ValueError("Specify both time_start and time_end")
    if time_start is not None:
        from lib.market_data import parse_time_seconds
        if (not np.isfinite(parse_time_seconds(time_start)) or
                not np.isfinite(parse_time_seconds(time_end)) or time_start >= time_end):
            raise ValueError("Invalid time window")
    source = Path(db_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if entry_pattern_config is not None:
        with duckdb.connect(str(source), read_only=True) as check:
            columns = check.execute(f'DESCRIBE {_identifier(table_name)}').fetchdf().column_name.tolist()
        required_buy = ({'거래량', '누적거래량'} if buy_notional_source == 'signed_volume_positive_buy'
                        else {buy_notional_column})
        if not (required_buy | {'현재가'}) <= set(columns):
            raise ValueError(f'Entry-pattern extraction requires actual cumulative BUY execution notional column '
                             f'{buy_notional_column!r} and 현재가. Total 거래대금 and 매수대기금액 are not substitutes.')
    output = Path(output_dir).resolve()
    state_path, journal_path = output / "extraction_state.json", output / "completed.jsonl"
    source_stat = source.stat()
    identity = {"extractor_version": 4 if entry_pattern_config is not None else 3, "source_db": str(source),
                "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
                "table": table_name, "seq_len": seq_len, "features": features,
                "max_steps": max_steps, "limit": limit, "price_unit": price_unit,
                "time_start": time_start, "time_end": time_end,
                 "compression_level": compression_level}
    if entry_pattern_config is not None:
        identity.update(entry_pattern_config=entry_pattern_config,
                        buy_notional_column=buy_notional_column, buy_notional_scale=buy_notional_scale,
                        buy_notional_source=buy_notional_source)
    if (output / "manifest.json").exists():
        raise FileExistsError("Output manifest already exists; use a new directory for regeneration")
    if resume:
        if not state_path.is_file() or json.loads(state_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume source fingerprint/settings do not match the original extraction")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output is not empty; use a new directory or explicit resume")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(state_path, identity)

    completed = {}
    if resume and journal_path.exists():
        # An interrupted final append may be incomplete; rewrite only complete
        # records so a new append cannot join onto a damaged JSON fragment.
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError("Corrupt extraction journal")
                break
            key = (entry["stock_code"], entry["date"])
            if key in completed:
                raise ValueError("Duplicate extraction journal entry")
            if entry["status"] == "ok":
                target = (output / entry["file_path"]).resolve()
                if not target.is_relative_to(output) or not target.is_file() or target.stat().st_size != entry["bytes"]:
                    raise ValueError("Completed episode is missing or changed; inspect the extraction output")
            completed[key] = entry
        journal_path.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n"
                                         for entry in completed.values()), encoding="utf-8")

    started = time.monotonic()
    conn = duckdb.connect(str(source), read_only=True,
                         config={"threads": threads, "memory_limit": memory_limit})
    try:
        conn.execute('SET enable_progress_bar=false')
        feature_cols = get_feature_columns(conn, table_name, features)
        schema = conn.execute(f"DESCRIBE {_identifier(table_name)}").fetchdf().column_name.tolist()
        wanted = set(feature_cols) | {"종목코드", "날짜", "시간", "종목명", "번호", "현재가", "호가시간", "quote_timestamp"}
        if entry_pattern_config is not None:
            wanted.update({'거래량', '누적거래량'} if buy_notional_source == 'signed_volume_positive_buy' else {buy_notional_column})
        if '시초가' in feature_cols:
            wanted.add('시가')
        for name in feature_cols:
            wanted.update(_depth_sources(name))
        for side in ("매수", "매도"):
            for level in range(1, 11):
                wanted.update((f"{side}호가{level}", f"{side}호가수량{level}"))
        projection = ", ".join(_identifier(name) for name in schema if name in wanted)
        condition = "" if time_start is None else ' AND TRY_CAST("시간" AS DOUBLE) BETWEEN ? AND ?'
        time_parameters = [] if time_start is None else [time_start, time_end]
        logger.info("Scanning eligible stock/date keys in %s ...", source)
        keys = conn.execute(f"""
            SELECT "종목코드", "날짜", COUNT(*) AS cnt FROM {_identifier(table_name)}
            WHERE TRUE {condition}
            GROUP BY "종목코드", "날짜" HAVING COUNT(*) >= ?
            ORDER BY "날짜", "종목코드"
        """, [*time_parameters, seq_len + 1]).fetchdf()
        if limit is not None:
            keys = keys.sample(n=min(limit, len(keys)), random_state=42)
        records = list(keys.itertuples(index=False, name=None))
        expected_keys = {(str(stock).zfill(6), int(date)) for stock, date, _ in records}
        if not set(completed) <= expected_keys:
            raise ValueError("Journal contains keys outside this extraction")
        pending = [row for row in records if (str(row[0]).zfill(6), int(row[1])) not in completed]
        tie = '"번호", rowid' if "번호" in schema else "rowid"
        query_condition, query_time_parameters = condition, time_parameters
        if ('시초가' in feature_cols or entry_pattern_config is not None) and time_start is not None:
            # Compute the day's open before trimming a later training window.
            query_condition = ' AND TRY_CAST("시간" AS DOUBLE) <= ?'
            query_time_parameters = [time_end]
        query = f"""SELECT {projection} FROM {_identifier(table_name)}
                    WHERE "종목코드" = ? AND "날짜" = ? {query_condition}
                    ORDER BY TRY_CAST("시간" AS DOUBLE), {tie}"""
        logger.info("Eligible episodes=%s, rows=%s, completed=%s, workers=%s",
                    len(records), sum(row[2] for row in records), len(completed), workers)

        settings = dict(output=str(output), source=str(source), query=query,
                        entry_pattern_config=entry_pattern_config, buy_notional_column=buy_notional_column,
                        buy_notional_scale=buy_notional_scale, buy_notional_source=buy_notional_source, seq_len=seq_len,
                        time_parameters=time_parameters, feature_cols=feature_cols,
                        query_time_parameters=query_time_parameters,
                        price_unit=price_unit, resume=resume, compression_level=compression_level,
                        threads=threads, memory_limit=memory_limit)
        if executor_kind == 'process':
            # Release the parent's DB buffer before starting isolated workers.
            conn.close()
            conn = None
            executor = ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_process_worker, initargs=(settings,))
            extract_one = _extract_process_episode
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            def extract_one(row):
                return _extract_one_episode(row, settings, conn)
        futures = []
        try:
            futures = [executor.submit(extract_one, row) for row in pending]
            with journal_path.open("a", encoding="utf-8") as journal:
                for future in as_completed(futures):
                    entry = future.result()
                    journal.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    completed[(entry["stock_code"], entry["date"])] = entry
                    if entry["status"] != "ok":
                        logger.warning("Rejected %s/%s: %s", entry["stock_code"], entry["date"], entry["reason"])
                    if len(completed) % 25 == 0 or len(completed) == len(records):
                        rows_done = sum(item.get("length", 0) for item in completed.values())
                        bytes_done = sum(item.get("bytes", 0) for item in completed.values())
                        progress = {"completed": len(completed), "total": len(records),
                                    "rows": rows_done, "bytes": bytes_done,
                                    "elapsed_seconds": round(time.monotonic() - started, 1)}
                        _atomic_json(output / "progress.json", progress)
                        logger.info("Progress %s/%s | rows=%s | output=%.2f GiB | elapsed=%.1fs",
                                    len(completed), len(records), rows_done, bytes_done / 1024**3,
                                    progress["elapsed_seconds"])
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        rejected = dict(Counter(item["reason"] for item in completed.values() if item["status"] != "ok"))
        manifest = [{key: value for key, value in item.items() if key != "status"}
                    for item in sorted(completed.values(), key=lambda item: (item["date"], item["stock_code"]))
                    if item["status"] == "ok"]
        if not manifest:
            raise ValueError(f"No valid episodes extracted; rejected={rejected}")
        final_stat = source.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError("Source database changed during extraction")
        payload = {"metadata": {"schema_version": 2, "feature_columns": feature_cols,
                    "entry_pattern_config": entry_pattern_config,
                    "return_rate_index": feature_cols.index("등락률"), "expected_features": features,
                    "feature_transform": "raw", "price_unit": price_unit,
                    "execution_price_unit": "krw", "timestamp_format": "HHMMSSmmm",
                    "source_db": str(source), "rejected_episodes": rejected,
                    "extraction": identity, "source_has_quote_timestamp":
                    bool({"호가시간", "quote_timestamp"} & set(schema))},
                   "episodes": manifest}
        _atomic_json(output / "manifest.json", payload)
        logger.info("Extraction complete: %s episodes, %s rows, rejected=%s, output=%s",
                    len(manifest), sum(item["length"] for item in manifest), rejected, output)
        return True
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="datasets")
    parser.add_argument("--output_dir", default="data/extracted_episodes_opening")
    parser.add_argument("--seq_len", type=int, default=3000)
    parser.add_argument("--features", type=int, default=29)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--time-start", type=int, help="Inclusive HHMMSSmmm start (requires --time-end)")
    parser.add_argument("--time-end", type=int, help="Inclusive HHMMSSmmm end")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=('thread', 'process'), default='thread',
                        help="Process workers bypass the Python GIL; DB memory limit is per process")
    parser.add_argument("--threads", type=int, default=2, help="DuckDB threads per process, or shared in thread mode")
    parser.add_argument("--memory-limit", default="4GB", help="DB buffer limit per process; NumPy/Pandas use extra RAM")
    parser.add_argument("--compression-level", type=int, default=6, choices=range(10))
    parser.add_argument("--resume", action="store_true", help="Resume the same interrupted extraction")
    parser.add_argument('--entry-pattern-config', help='Training JSON containing entry_pattern_config (or the config object itself)')
    parser.add_argument('--buy-notional-column', default='cumulative_buy_notional_krw',
                        help='Actual cumulative BUY execution notional for one stock/date; never total turnover')
    parser.add_argument('--buy-notional-scale', type=float, default=1.0, help='Multiply source notional by this to obtain KRW')
    parser.add_argument('--buy-notional-source', default='cumulative_buy_notional',
                        choices=('cumulative_buy_notional', 'signed_volume_positive_buy'),
                        help='Signed mode explicitly declares 거래량 positive=BUY, negative=SELL; deduplicates via 누적거래량')
    parser.add_argument("--price-unit", required=True, choices=tuple(PRICE_SCALES),
                        help="Unit of source current/quote prices; never inferred from their magnitude")
    args = parser.parse_args()
    pattern_config = None
    if args.entry_pattern_config:
        pattern_config = json.loads(Path(args.entry_pattern_config).read_text(encoding='utf-8'))
        pattern_config = pattern_config.get('entry_pattern_config', pattern_config)
    extract_data(args.db, args.table, args.output_dir, args.seq_len, args.features,
                 args.max_steps, args.limit, args.price_unit,
                 time_start=args.time_start, time_end=args.time_end, workers=args.workers,
                 threads=args.threads, memory_limit=args.memory_limit,
                 compression_level=args.compression_level, resume=args.resume,
                 executor_kind=args.executor, entry_pattern_config=pattern_config,
                 buy_notional_column=args.buy_notional_column, buy_notional_scale=args.buy_notional_scale,
                 buy_notional_source=args.buy_notional_source)
