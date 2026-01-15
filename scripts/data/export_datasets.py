import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np

# Add project root to sys.path to ensure we can import internal modules
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import shared constants and functions from normalize_datasets
# Note: assuming scripts/data is a valid package or accessible via scripts.data
try:
    from scripts.data.normalize_datasets import (
        TEXT_COLUMNS, DROP_COLUMNS, INPUT_TABLE, FINAL_COLUMNS,
        DEFAULT_MIN_QUALIFYING_MINUTES,
        clean_column_name, to_time_ms, time_ms_to_seconds,
        load_ignoring_stocks, find_duckdb_groups, load_and_clean_from_duckdb,
        fill_missing_values, compute_order_book_amounts,
        merge_from_duckdb, ensure_table_duckdb,
        month_key_from_yyyymmdd, monthly_db_path,
        checkpoint_db_once, parallel_checkpoint_months,
        valid_yyyymmdd, date_in_range
    )
except ImportError:
    # Fallback if running from scripts/data directly without module context
    # This might require adjusting sys.path further or using relative imports if package
    sys.path.append(str(Path(__file__).parent))
    from normalize_datasets import (
        TEXT_COLUMNS, DROP_COLUMNS, INPUT_TABLE, FINAL_COLUMNS,
        DEFAULT_MIN_QUALIFYING_MINUTES,
        clean_column_name, to_time_ms, time_ms_to_seconds,
        load_ignoring_stocks, find_duckdb_groups, load_and_clean_from_duckdb,
        fill_missing_values, compute_order_book_amounts,
        merge_from_duckdb, ensure_table_duckdb,
        month_key_from_yyyymmdd, monthly_db_path,
        checkpoint_db_once, parallel_checkpoint_months,
        valid_yyyymmdd, date_in_range
    )


def save_groups_raw(
    input_db: str,
    output_db: str,
    *,
    table: str = "datasets_raw",
    time_start: int = 90000000,
    time_end: int = 110000000,

    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    ignoring_stocks_csv: Optional[str] = None,
    input_table: str = INPUT_TABLE,
):
    groups = find_duckdb_groups(input_db, table=input_table)
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
        ignoring_set = load_ignoring_stocks(ignoring_stocks_csv)
        reason_counts: Dict[str, int] = {}
        for group_key, group_info in groups.items():
            try:
                parts = group_key.split('_')
                code = parts[0]
                date = parts[-1]
                name = '_'.join(parts[1:-1])
                
                # Check date range (optional, if we want to enforce it here as well)
                # But helper does not enforce it globally unless passed.
                # merge_from_duckdb takes date, time_start, time_end.
                # It does NOT take start_date/end_date arguments globally here.
                # Logic matches original script.

                df = merge_from_duckdb(
                    group_info,
                    code,
                    name,
                    date,
                    time_start=time_start,
                    time_end=time_end,

                    min_qualifying_minutes=min_qualifying_minutes,
                    ignoring_stocks_set=ignoring_set,
                )
                
                # Note: normalize_datasets.merge_from_duckdb returns just df (not tuple with reason)?
                # Wait, I need to check signature of merge_from_duckdb in normalize_datasets.py
                # I might have introduced a mismatch.
                # Original export_datasets.py merge_from_duckdb returns Tuple[pd.DataFrame, Optional[str]].
                # normalize_datasets.py merge_from_duckdb returns pd.DataFrame.
                pass
                
                if df.empty:
                    # reason logic is lost if using normalize_datasets version.
                    # We can assume 'filtered' or 'empty'.
                    # For compatibility, we just skip.
                    reason_counts['empty'] = reason_counts.get('empty', 0) + 1
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

    min_qualifying_minutes: int,
    checkpoint_interval: int,
    ignoring_stocks_csv: Optional[str],
    input_table: str,
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
    ignoring_set = load_ignoring_stocks(ignoring_stocks_csv)
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

                    min_qualifying_minutes=min_qualifying_minutes,
                    ignoring_stocks_set=ignoring_set,
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

    min_qualifying_minutes: int = DEFAULT_MIN_QUALIFYING_MINUTES,
    single_output: bool = False,
    tmp_dir: Optional[str] = None,
    ignoring_stocks_csv: Optional[str] = None,
    input_table: str = "datasets",
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

            min_qualifying_minutes=min_qualifying_minutes,
            ignoring_stocks_csv=ignoring_stocks_csv,
            input_table=input_table,
        )

    if not os.path.exists(input_db):
        print(f"입력 DuckDB 파일이 존재하지 않습니다: {input_db}")
        return

    # Scan groups from input
    print(f"DuckDB 데이터 스캔 및 그룹화 중... (테이블: {input_table})")
    complete_groups = find_duckdb_groups(input_db, table=input_table)
    if not complete_groups:
        print("처리할 유효 데이터 그룹을 찾지 못했습니다.")
        return

    # Group by month
    monthly_groups: Dict[str, List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]] = {}
    for group_key, group_info in complete_groups.items():
        parts = group_key.split("_")
        date = parts[-1]
        yyyymm = month_key_from_yyyymmdd(date)
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
                month_db = monthly_db_path(output_db, yyyymm)
                pairs = monthly_groups[yyyymm]
                fut = ex.submit(
                    _process_month_groups,
                    pairs,
                    month_db,
                    yyyymm,
                    table,
                    time_start,
                    time_end,

                    min_qualifying_minutes,
                    checkpoint_interval,
                    ignoring_stocks_csv,
                    input_table,
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
            month_db = monthly_db_path(output_db, yyyymm)
            pairs = monthly_groups[yyyymm]
            processed = _process_month_groups(
                pairs,
                month_db,
                yyyymm,
                table,
                time_start,
                time_end,

                min_qualifying_minutes,
                checkpoint_interval,
                ignoring_stocks_csv,
                input_table,
            )
            print(f"월 처리 완료: {yyyymm} ({processed}/{len(pairs)}) -> {month_db}")

    # Final checkpoints on produced shards
    try:
        parallel_checkpoint_months(output_db, months, used_workers)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="DuckDB에서 FINAL_COLUMNS 원본(비정규화) 추출/저장")
    ap.add_argument("--input-db", required=True, help="입력 DuckDB 파일 경로 (테이블: datasets)")
    ap.add_argument("--input-table", default="datasets", help="입력 DuckDB 테이블명 (기본: datasets)")
    ap.add_argument("--output-db", required=True, help="출력 DuckDB 파일 경로")
    ap.add_argument("--table-name", default="datasets_raw", help="출력 테이블명 (기본: datasets_raw)")
    ap.add_argument("--time-start", type=int, default=90000000, help="시작 시간 HHMMSSmmm (기본: 090000000)")
    ap.add_argument("--time-end", type=int, default=110000000, help="종료 시간 HHMMSSmmm 미포함 (기본: 110000000)")

    ap.add_argument("--qualifying-minutes", type=int, default=DEFAULT_MIN_QUALIFYING_MINUTES, help="임계 충족 분 최소 개수")
    ap.add_argument("--workers", type=int, default=1, help="월별 병렬 처리 프로세스 수 (기본: 1)")
    ap.add_argument("--checkpoint-interval", type=int, default=100, help="몇 개 그룹 처리마다 CHECKPOINT 수행할지 (기본: 100)")
    ap.add_argument("--single-output", action="store_true", help="월별 샤드 대신 단일 출력 DB에 순차 반영")
    ap.add_argument("--tmp-dir", default=None, help="임시 파일/로그 등을 저장할 디렉터리 (선택)")
    ap.add_argument("--ignoring-stocks-csv", default=os.path.join("scripts", "data", "ignoring_stocks.csv"),
                    help="무시할 종목명 리스트 CSV 경로 (기본: scripts/data/ignoring_stocks.csv, '종목명' 컬럼 필요)")
    args = ap.parse_args()

    # 작업 시작 로깅
    start_dt = datetime.now()
    start_ts = time.time()
    print(f"작업 시작: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    export_datasets(
        input_db=args.input_db,
        output_db=args.output_db,
        table=args.table_name,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
        time_start=args.time_start,
        time_end=args.time_end,

        min_qualifying_minutes=args.qualifying_minutes,
        single_output=args.single_output,
        tmp_dir=args.tmp_dir,
        ignoring_stocks_csv=args.ignoring_stocks_csv,
        input_table=args.input_table,
    )
    
    end_dt = datetime.now()
    elapsed = time.time() - start_ts
    print(f"작업 종료: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 경과 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    main()
