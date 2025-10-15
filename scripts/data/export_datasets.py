import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import duckdb
import numpy as np
import pandas as pd

# Reference constants aligned with scripts/data/normalize_datasets.py
INPUT_TABLE = "datasets"
TEXT_COLUMNS = {"종목코드", "종목명", *{f"매도거래원{i}" for i in range(1, 6)}, *{f"매수거래원{i}" for i in range(1, 6)}}
DROP_COLUMNS = {"종류", "씨리얼"}

# Default minimum required per-minute cumulative traded value (누적거래대금)
DEFAULT_TRADE_VALUE_PER_MINUTE: float = 3000.0
# Default minimum number of minutes that must satisfy the trade value threshold
DEFAULT_MIN_QUALIFYING_MINUTES: int = 1

# Final column order required (unnormalized)
FINAL_COLUMNS: List[str] = [
    "종목코드", "종목명", "시간", "등락률",
    "누적거래대금", "거래회전율", "체결강도",
    *[f"매도대기금액{i}" for i in range(1, 11)],
    *[f"매수대기금액{i}" for i in range(1, 11)],
]


def _to_time_ms(val: object) -> int:
    try:
        s = str(val)
        if not s or s.lower() == 'nan':
            return 0
        if '.' in s:
            s = s.split('.', 1)[0]
        digits = ''.join(ch for ch in s if ch.isdigit())
        if not digits:
            return 0
        digits = digits[-9:].rjust(9, '0')
        hh = int(digits[0:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])
        ms = int(digits[6:9])
        hh = max(0, min(23, hh))
        mm = max(0, min(59, mm))
        ss = max(0, min(59, ss))
        ms = max(0, min(999, ms))
        if hh > 20:
            return 0
        return int(f"{hh:02d}{mm:02d}{ss:02d}{ms:03d}")
    except Exception:
        return 0


def _time_ms_to_seconds(t: int) -> int:
    try:
        t9 = _to_time_ms(t)
        s = str(int(t9)).rjust(9, '0')
        hh = int(s[0:2])
        mm = int(s[2:4])
        ss = int(s[4:6])
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return 0


def _clean_column_name(col: str) -> str:
    c = re.sub(r"\s+", "", col)
    if c == "스탬프":
        return "시간"
    return c


def _count_trade_value_minutes(df: pd.DataFrame, trade_threshold_per_minute: float) -> Optional[int]:
    if '누적거래대금' not in df.columns or '시간' not in df.columns:
        return None
    trade_values = pd.to_numeric(df['누적거래대금'], errors='coerce')
    time_secs = df['시간'].apply(_time_ms_to_seconds)
    time_secs = pd.to_numeric(time_secs, errors='coerce')
    mask = trade_values.notna() & time_secs.notna()
    if mask.sum() == 0:
        return None
    trade_values = trade_values[mask].astype(float)
    time_secs = time_secs[mask].astype(int)
    if trade_values.empty:
        return None
    order = np.argsort(time_secs.values, kind="mergesort")
    trade_values = trade_values.iloc[order]
    time_secs = time_secs.iloc[order]
    minutes = (time_secs // 60).astype(int)
    df_group = pd.DataFrame({'minute': minutes.values, 'trade': trade_values.values})
    deltas = df_group.groupby('minute')['trade'].agg(lambda s: float(s.max() - s.min()))
    qualifying = (deltas >= trade_threshold_per_minute).sum()
    return int(qualifying)


def _load_ignoring_stocks(csv_path: Optional[str]) -> Set[str]:
    """Load a set of stock names to ignore from a CSV with a column named '종목명'."""
    ignore: Set[str] = set()
    if not csv_path:
        return ignore
    try:
        if not os.path.exists(csv_path):
            return ignore
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns:
            if str(c).strip() == '종목명':
                col = c
                break
        if col is None:
            return ignore
        vals = df[col].astype(str).str.strip()
        ignore = set(v for v in vals if v)
    except Exception:
        return set()
    return ignore


def find_duckdb_groups(input_db: str) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    if not os.path.exists(input_db):
        return {}
    groups = {}
    try:
        conn = duckdb.connect(input_db, read_only=True)
        try:
            try:
                conn.execute(f"DESCRIBE {INPUT_TABLE}")
            except Exception:
                print(f"경고: 입력 DB에 '{INPUT_TABLE}' 테이블이 없습니다.")
                return {}
            query = f"""
                SELECT DISTINCT "종목코드", "종목명", "날짜"
                FROM {INPUT_TABLE}
                WHERE "종목코드" IS NOT NULL 
                  AND "종목명" IS NOT NULL 
                  AND "날짜" IS NOT NULL
                ORDER BY "날짜", "종목코드"
            """
            results = conn.execute(query).fetchall()
            for code, name, date in results:
                group_key = f"{code}_{name}_{date}"
                groups[group_key] = {
                    "merged": (input_db, str(code), str(name), str(date))
                }
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 그룹 스캔 실패: {type(e).__name__}: {e}")
        return {}
    return groups


def load_and_clean_from_duckdb(
    db_path: str,
    code: str,
    name: str,
    date: str,
    *,
    time_start: int = 90000000,
    time_end: int = 110000000,
    trade_threshold_per_minute: float = DEFAULT_TRADE_VALUE_PER_MINUTE,
    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    ignoring_stocks_set: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        conn = duckdb.connect(db_path, read_only=True)
        try:
            query = f"""
                SELECT *
                FROM {INPUT_TABLE}
                WHERE "종목코드" = ? AND "날짜" = ?
                ORDER BY "시간"
            """
            df = conn.execute(query, [code, date]).df()
            if df.empty:
                return pd.DataFrame(), "no_rows"
            df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})
            if '종목명' in df.columns:
                ig = ignoring_stocks_set or set()
                if ig:
                    mask_ignore = df['종목명'].astype(str).str.strip().isin(ig)
                    if mask_ignore.any():
                        df = df[~mask_ignore]
                        if df.empty:
                            return pd.DataFrame(), "ignored_stock"
            if '시간' in df.columns:
                conv = df['시간'].apply(_to_time_ms)
                df['시간_hhmmssmmm'] = conv
                df = df[(df['시간_hhmmssmmm'] >= time_start) & (df['시간_hhmmssmmm'] < time_end)]
                if len(df['시간_hhmmssmmm']) == 0:
                    return pd.DataFrame(), "no_time_in_window"
                # Set '시간' to numeric HHMMSSmmm and drop helper
                df.loc[:, '시간'] = df['시간_hhmmssmmm']
                df = df.drop(columns=['시간_hhmmssmmm'])
            if trade_threshold_per_minute > 0:
                qualifying_minutes = _count_trade_value_minutes(df, trade_threshold_per_minute)
                if qualifying_minutes is None:
                    return pd.DataFrame(), "insufficient_data_for_liquidity"
                if qualifying_minutes < min_qualifying_minutes:
                    return pd.DataFrame(), "insufficient_liquidity"
            df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
            return df, None
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 로드 실패 ({code}, {date}): {type(e).__name__}: {e}")
        return pd.DataFrame(), "load_error"


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    text_cols = [col for col in df.columns if col in TEXT_COLUMNS]
    numeric_cols = [col for col in df.columns if col not in TEXT_COLUMNS and col != '번호']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna('').infer_objects(copy=False)
    for col in numeric_cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors='coerce')
            series = series.ffill().bfill().fillna(0)
            df[col] = series
    return df


def _compute_order_book_amounts(df: pd.DataFrame) -> pd.DataFrame:
    add_cols: Dict[str, pd.Series] = {}
    for i in range(1, 11):
        ask_price_col = f"매도호가{i}"
        ask_qty_col = f"매도호가수량{i}"
        ask_amt_col = f"매도대기금액{i}"
        if ask_price_col in df.columns and ask_qty_col in df.columns:
            price = pd.to_numeric(df[ask_price_col], errors="coerce").fillna(0.0)
            qty = pd.to_numeric(df[ask_qty_col], errors="coerce").fillna(0.0)
            add_cols[ask_amt_col] = (price * qty).astype(float) / 1_000_000
        bid_price_col = f"매수호가{i}"
        bid_qty_col = f"매수호가수량{i}"
        bid_amt_col = f"매수대기금액{i}"
        if bid_price_col in df.columns and bid_qty_col in df.columns:
            price = pd.to_numeric(df[bid_price_col], errors="coerce").fillna(0.0)
            qty = pd.to_numeric(df[bid_qty_col], errors="coerce").fillna(0.0)
            add_cols[bid_amt_col] = (price * qty).astype(float) / 1_000_000
    if add_cols:
        df = df.assign(**add_cols)
    return df


def merge_from_duckdb(
    group_info: Dict[str, Tuple[str, str, str, str]],
    code: str,
    name: str,
    date: str,
    *,
    time_start: int = 90000000,
    time_end: int = 110000000,
    trade_threshold_per_minute: float = DEFAULT_TRADE_VALUE_PER_MINUTE,
    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    ignoring_stocks_set: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if "merged" in group_info:
        db_path, code, name, date = group_info["merged"]
        df, reason = load_and_clean_from_duckdb(
            db_path,
            code,
            name,
            date,
            time_start=time_start,
            time_end=time_end,
            trade_threshold_per_minute=trade_threshold_per_minute,
            min_qualifying_minutes=min_qualifying_minutes,
            ignoring_stocks_set=ignoring_stocks_set,
        )
        if df.empty:
            return pd.DataFrame(), reason
        df = fill_missing_values(df)
        df['종목코드'] = df.get('종목코드', pd.Series(index=df.index, dtype=object)).fillna(code).replace({"": code})
        df['종목명'] = df.get('종목명', pd.Series(index=df.index, dtype=object)).fillna(name).replace({"": name})
        df = _compute_order_book_amounts(df)
        missing_final_columns = [col for col in FINAL_COLUMNS if col not in df.columns]
        if missing_final_columns:
            add_cols: Dict[str, pd.Series] = {}
            for col in missing_final_columns:
                if col in TEXT_COLUMNS:
                    add_cols[col] = pd.Series([""] * len(df), index=df.index, dtype=object)
                else:
                    add_cols[col] = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
            df = pd.concat([df, pd.DataFrame(add_cols, index=df.index)], axis=1)
        df = fill_missing_values(df)
        df = df[["번호", *FINAL_COLUMNS]] if "번호" in df.columns else df[FINAL_COLUMNS]
        return df
    return pd.DataFrame()


def ensure_table_duckdb(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, table: str):
    try:
        conn.execute(f"DESCRIBE {table}")
        exists = True
    except Exception:
        exists = False
    if not exists:
        col_defs: list[str] = []
        for col in df.columns:
            series = df[col]
            if col == '번호' or col == '시간' or pd.api.types.is_integer_dtype(series):
                duck_type = 'BIGINT'
            elif col in TEXT_COLUMNS or col in {'날짜', '종목명'} or series.dtype == object:
                duck_type = 'VARCHAR'
            else:
                duck_type = 'DOUBLE'
            col_defs.append(f'"{col}" {duck_type}')
        create_sql = f"CREATE TABLE {table} ({', '.join(col_defs)})"
        conn.execute(create_sql)
    else:
        existing_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        existing_cols = [row[1] for row in existing_info]
        existing_types = {row[1]: (row[2] or "").upper() for row in existing_info}
        for col in df.columns:
            if col not in existing_cols:
                series = df[col]
                if col == '시간' or col == '번호' or pd.api.types.is_integer_dtype(series):
                    col_type = 'BIGINT'
                elif col in TEXT_COLUMNS or col == '날짜' or series.dtype == object:
                    col_type = 'VARCHAR'
                else:
                    col_type = 'DOUBLE'
                conn.execute(f"ALTER TABLE {table} ADD COLUMN \"{col}\" {col_type}")
        text_like = set(TEXT_COLUMNS) | {"날짜", "종목코드"}
        for col in (c for c in df.columns if c in text_like and c in existing_types):
            ctype = existing_types.get(col, "")
            if "CHAR" not in ctype and "STRING" not in ctype and "VARCHAR" not in ctype:
                conn.execute(f"ALTER TABLE {table} ALTER COLUMN \"{col}\" TYPE VARCHAR")


def _month_key_from_yyyymmdd(date_str: str) -> str:
    """Extract YYYYMM from YYYYMMDD string."""
    return date_str[:6] if len(date_str) >= 6 else date_str


def _monthly_db_path(base_db_path: str, yyyymm: str) -> str:
    """Return a per-month DuckDB path based on base path and yyyymm.
    Example: base 'datasets.duckdb' -> 'datasets_YYYYMM.duckdb' in same directory.
    """
    p = Path(base_db_path)
    stem = p.stem
    suffix = p.suffix or ".duckdb"
    return str(p.with_name(f"{stem}_{yyyymm}{suffix}"))


def _checkpoint_db_once(db_path: str) -> Tuple[str, bool, str]:
    """Run DuckDB CHECKPOINT once for the given DB file. Returns (path, ok, msg)."""
    try:
        conn = duckdb.connect(db_path)
        try:
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
        return db_path, True, ""
    except Exception as e:
        return db_path, False, f"{type(e).__name__}: {e}"


def _parallel_checkpoint_months(base_db_path: str, months: List[str], workers: int) -> None:
    """Run CHECKPOINT across the given months' DB shards in parallel using up to `workers` processes."""
    import concurrent.futures as _fut
    if not months:
        return
    # Build existing paths only
    month_paths = []
    for m in months:
        p = _monthly_db_path(base_db_path, m)
        if os.path.exists(p):
            month_paths.append(p)
    if not month_paths:
        return
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(month_paths))
    if used_workers > 1:
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {ex.submit(_checkpoint_db_once, path): path for path in month_paths}
            for fut in _fut.as_completed(futs):
                try:
                    fut.result()
                except Exception:
                    pass
    else:
        for path in month_paths:
            try:
                _checkpoint_db_once(path)
            except Exception:
                pass


def save_groups_raw(
    input_db: str,
    output_db: str,
    *,
    table: str = "datasets_raw",
    time_start: int = 90000000,
    time_end: int = 110000000,
    trade_threshold_per_minute: float = DEFAULT_TRADE_VALUE_PER_MINUTE,
    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    ignoring_stocks_csv: Optional[str] = None,
):
    groups = find_duckdb_groups(input_db)
    if not groups:
        print("입력 DB에서 처리할 그룹을 찾지 못했습니다.")
        return

    # process sequentially (safe) and write into a single output DB
    total = len(groups)
    print(f"총 {total}개 그룹 처리 시작 -> {output_db}:{table}")

    # Ensure the output DB directory exists before connecting
    try:
        out_path = Path(output_db)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    conn_out = duckdb.connect(str(output_db))
    try:
        done = 0
        ignoring_set = _load_ignoring_stocks(ignoring_stocks_csv)
        reason_counts: Dict[str, int] = {}
        for group_key, group_info in groups.items():
            try:
                parts = group_key.split('_')
                code = parts[0]
                date = parts[-1]
                name = '_'.join(parts[1:-1])
                df, reason = merge_from_duckdb(
                    group_info,
                    code,
                    name,
                    date,
                    time_start=time_start,
                    time_end=time_end,
                    trade_threshold_per_minute=trade_threshold_per_minute,
                    min_qualifying_minutes=min_qualifying_minutes,
                    ignoring_stocks_set=ignoring_set,
                )
                if df.empty:
                    if reason:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    continue
                # add 날짜 and reorder for convenient querying
                df = df.copy()
                df['날짜'] = date
                df = pd.concat([df['날짜'], df.drop(columns=['날짜'])], axis=1)
                ensure_table_duckdb(conn_out, df, table)
                conn_out.register("_batch_df", df)
                try:
                    conn_out.execute(f"DELETE FROM {table} WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
                except Exception:
                    pass
                conn_out.execute(f"INSERT INTO {table} SELECT * FROM _batch_df")
                conn_out.unregister("_batch_df")
                done += 1
                if done % 20 == 0:
                    try:
                        conn_out.execute("CHECKPOINT")
                    except Exception:
                        pass
                if done % 50 == 0 or done == total:
                    print(f"진행률: {done}/{total}")
            except Exception as e:
                print(f"경고: 그룹 처리 실패({group_key}): {type(e).__name__}: {e}")
                reason_counts['exception'] = reason_counts.get('exception', 0) + 1
        try:
            conn_out.execute("CHECKPOINT")
        except Exception:
            pass
    finally:
        conn_out.close()
    skipped = total - done
    if skipped > 0:
        print("스킵 요약(단일 출력):")
        for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {v}")
    print("완료: 원본 FINAL_COLUMNS 저장 완료")


def _process_month_groups(
    month_pairs: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]],
    month_db_path: str,
    yyyymm: str,
    table: str,
    time_start: int,
    time_end: int,
    trade_threshold_per_minute: float,
    min_qualifying_minutes: int,
    checkpoint_interval: int,
    ignoring_stocks_csv: Optional[str],
) -> int:
    """Process all groups for a specific month and write to month_db_path/table sequentially."""
    # Ensure parent dir
    try:
        p = Path(month_db_path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    processed_count = 0
    ignoring_set = _load_ignoring_stocks(ignoring_stocks_csv)
    reason_counts: Dict[str, int] = {}
    try:
        conn_out = duckdb.connect(month_db_path)
    except Exception as e:
        print(f"  {yyyymm}: DB 연결 실패 -> {type(e).__name__}: {e}")
        return 0
    try:
        total = len(month_pairs)
        print(f"  {yyyymm}: {total} 그룹 처리 시작 -> {month_db_path}:{table}")
        for idx, (group_key, group_info) in enumerate(month_pairs, 1):
            try:
                parts = group_key.split('_')
                code = parts[0]
                date = parts[-1]
                name = '_'.join(parts[1:-1])
                df = merge_from_duckdb(
                    group_info,
                    code,
                    name,
                    date,
                    time_start=time_start,
                    time_end=time_end,
                    trade_threshold_per_minute=trade_threshold_per_minute,
                    min_qualifying_minutes=min_qualifying_minutes,
                )
                if df.empty:
                    continue
                df = df.copy()
                df['날짜'] = date
                df = pd.concat([df['날짜'], df.drop(columns=['날짜'])], axis=1)
                ensure_table_duckdb(conn_out, df, table)
                conn_out.register("_batch_df", df)
                try:
                    conn_out.execute(f"DELETE FROM {table} WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
                except Exception:
                    pass
                conn_out.execute(f"INSERT INTO {table} SELECT * FROM _batch_df")
                conn_out.unregister("_batch_df")
                processed_count += 1
                if checkpoint_interval > 0 and processed_count % checkpoint_interval == 0:
                    try:
                        conn_out.execute("CHECKPOINT")
                    except Exception:
                        pass
                if processed_count % 50 == 0 or processed_count == total:
                    print(f"  {yyyymm}: 진행률 {processed_count}/{total}")
            except Exception as e:
                print(f"  {yyyymm}: 그룹 처리 실패({group_key}): {type(e).__name__}: {e}")
                reason_counts['exception'] = reason_counts.get('exception', 0) + 1
        try:
            conn_out.execute("CHECKPOINT")
        except Exception:
            pass
    finally:
        try:
            conn_out.close()
        except Exception:
            pass
    skipped = total - processed_count
    if skipped > 0:
        print(f"  {yyyymm}: 스킵 요약:")
        for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"    - {k}: {v}")
    return processed_count


def export_datasets(
    input_db: str,
    output_db: str,
    *,
    table: str = "datasets_raw",
    workers: int = 1,
    checkpoint_interval: int = 100,
    time_start: int = 90000000,
    time_end: int = 110000000,
    trade_threshold_per_minute: float = DEFAULT_TRADE_VALUE_PER_MINUTE,
    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    single_output: bool = False,
    tmp_dir: Optional[str] = None,
    ignoring_stocks_csv: Optional[str] = None,
):
    """Top-level export orchestrator with optional monthly parallelism.
    - If single_output is True: write all into one DB/table (calls save_groups_raw).
    - Else: shard by month and process months in parallel using up to `workers`.
    """
    if single_output:
        return save_groups_raw(
            input_db,
            output_db,
            table=table,
            time_start=time_start,
            time_end=time_end,
            trade_threshold_per_minute=trade_threshold_per_minute,
            min_qualifying_minutes=min_qualifying_minutes,
            ignoring_stocks_csv=ignoring_stocks_csv,
        )

    if not os.path.exists(input_db):
        print(f"입력 DuckDB 파일이 존재하지 않습니다: {input_db}")
        return

    # Scan groups from input
    print("DuckDB 데이터 스캔 및 그룹화 중...")
    complete_groups = find_duckdb_groups(input_db)
    if not complete_groups:
        print("처리할 유효 데이터 그룹을 찾지 못했습니다.")
        return

    # Group by month
    monthly_groups: Dict[str, List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]] = {}
    for group_key, group_info in complete_groups.items():
        parts = group_key.split("_")
        date = parts[-1]
        yyyymm = _month_key_from_yyyymmdd(date)
        monthly_groups.setdefault(yyyymm, []).append((group_key, group_info))

    months = sorted(monthly_groups.keys())
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(months))
    print(f"월별 처리 시작: 대상 {len(months)}개월, 병렬 workers={used_workers}")

    # Run months in parallel; each month writes to its own DB shard to avoid conflicts
    if used_workers > 1:
        import concurrent.futures as _fut
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {}
            for yyyymm in months:
                month_db = _monthly_db_path(output_db, yyyymm)
                pairs = monthly_groups[yyyymm]
                fut = ex.submit(
                    _process_month_groups,
                    pairs,
                    month_db,
                    yyyymm,
                    table,
                    time_start,
                    time_end,
                    trade_threshold_per_minute,
                    min_qualifying_minutes,
                    checkpoint_interval,
                    ignoring_stocks_csv,
                )
                futs[fut] = (yyyymm, month_db, len(pairs))
            for fut in _fut.as_completed(futs):
                yyyymm, month_db, n = futs[fut]
                try:
                    processed = fut.result()
                    print(f"월 처리 완료: {yyyymm} ({processed}/{n}) -> {month_db}")
                except Exception as e:
                    print(f"경고: 월 처리 실패 {yyyymm}: {type(e).__name__}: {e}")
    else:
        for yyyymm in months:
            month_db = _monthly_db_path(output_db, yyyymm)
            pairs = monthly_groups[yyyymm]
            processed = _process_month_groups(
                pairs,
                month_db,
                yyyymm,
                table,
                time_start,
                time_end,
                trade_threshold_per_minute,
                min_qualifying_minutes,
                checkpoint_interval,
                ignoring_stocks_csv,
            )
            print(f"월 처리 완료: {yyyymm} ({processed}/{len(pairs)}) -> {month_db}")

    # Final checkpoints on produced shards
    try:
        _parallel_checkpoint_months(output_db, months, used_workers)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="DuckDB에서 FINAL_COLUMNS 원본(비정규화) 추출/저장")
    ap.add_argument("--input-db", required=True, help="입력 DuckDB 파일 경로 (테이블: datasets)")
    ap.add_argument("--output-db", required=True, help="출력 DuckDB 파일 경로")
    ap.add_argument("--table-name", default="datasets_raw", help="출력 테이블명 (기본: datasets_raw)")
    ap.add_argument("--time-start", type=int, default=90000000, help="시작 시간 HHMMSSmmm (기본: 090000000)")
    ap.add_argument("--time-end", type=int, default=110000000, help="종료 시간 HHMMSSmmm 미포함 (기본: 110000000)")
    ap.add_argument("--trade-threshold-per-minute", type=float, default=DEFAULT_TRADE_VALUE_PER_MINUTE, help="분당 누적거래대금 증가 임계값(백만원)")
    ap.add_argument("--qualifying-minutes", type=int, default=DEFAULT_MIN_QUALIFYING_MINUTES, help="임계 충족 분 최소 개수")
    ap.add_argument("--workers", type=int, default=1, help="월별 병렬 처리 프로세스 수 (기본: 1)")
    ap.add_argument("--checkpoint-interval", type=int, default=100, help="몇 개 그룹 처리마다 CHECKPOINT 수행할지 (기본: 100)")
    ap.add_argument("--single-output", action="store_true", help="월별 샤드 대신 단일 출력 DB에 순차 반영")
    ap.add_argument("--tmp-dir", default=None, help="임시 파일/로그 등을 저장할 디렉터리 (선택)")
    ap.add_argument("--ignoring-stocks-csv", default=os.path.join("scripts", "data", "ignoring_stocks.csv"),
                    help="무시할 종목명 리스트 CSV 경로 (기본: scripts/data/ignoring_stocks.csv, '종목명' 컬럼 필요)")
    args = ap.parse_args()

    export_datasets(
        input_db=args.input_db,
        output_db=args.output_db,
        table=args.table_name,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        time_start=args.time_start,
        time_end=args.time_end,
        trade_threshold_per_minute=args.trade_threshold_per_minute,
        min_qualifying_minutes=args.qualifying_minutes,
        single_output=args.single_output,
        tmp_dir=args.tmp_dir,
        ignoring_stocks_csv=args.ignoring_stocks_csv,
    )


if __name__ == "__main__":
    main()
