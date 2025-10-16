import argparse
import os
import re
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures as _fut
import tempfile as _tempfile
import shutil as _shutil
import pickle as _pickle

import numpy as np
import pandas as pd

# Global sequential counter for '번호'
NO_COUNTER: int = 1

FILENAME_PATTERN = re.compile(r"^(?P<code>\d{6})_(?P<name>.+?)_(?P<type>[^_]+)_(?P<date>\d{8})\.csv$")
TEXT_COLUMNS = {"종목코드", "종목명", "시간", *{f"매도거래원{i}" for i in range(1, 6)}, *{f"매수거래원{i}" for i in range(1, 6)}}
DROP_COLUMNS = {"종류", "씨리얼"}
REQUIRED_TYPES = {"execution", "orderbook", "trader"}

# Final column order required
FINAL_COLUMNS: List[str] = [
    "종목코드", "종목명", "시간", "현재가", "등락률", "거래량", "누적거래량", "누적거래대금", "시가", "고가", "저가",
    "전일거래량대비", "전일거래량대비비율", "거래회전율", "체결강도",
    # 매도호가/수량/직전대비 1~10
    *[f"매도호가{i}" for i in range(1, 11)],
    *[f"매도호가수량{i}" for i in range(1, 11)],
    *[f"매도호가직전대비{i}" for i in range(1, 11)],
    # 매수호가/수량/직전대비 1~10
    *[f"매수호가{i}" for i in range(1, 11)],
    *[f"매수호가수량{i}" for i in range(1, 11)],
    *[f"매수호가직전대비{i}" for i in range(1, 11)],
    "매도호가총잔량", "매도호가총잔량직전대비", "매수호가총잔량", "매수호가총잔량직전대비",
    # 거래원 관련 (문자열 컬럼들)
    *[f"매도거래원{i}" for i in range(1, 6)],
    *[f"매도거래원수량{i}" for i in range(1, 6)],
    *[f"매도거래원별증감{i}" for i in range(1, 6)],
    *[f"매수거래원{i}" for i in range(1, 6)],
    *[f"매수거래원수량{i}" for i in range(1, 6)],
    *[f"매수거래원별증감{i}" for i in range(1, 6)],
]


def find_csv_files(folder_path: str) -> Dict[str, List[str]]:
    """
    폴더를 recursive하게 탐색하여 CSV 파일들을 찾고 그룹화
    """
    csv_files = {}
    folder = Path(folder_path)
    
    for csv_file in folder.rglob("*.csv"):
        match = FILENAME_PATTERN.match(csv_file.name)
        if not match:
            continue
            
        file_info = match.groupdict()
        
        # after_hours 타입은 무시
        if file_info["type"] == "after_hours":
            continue
            
        # 그룹 키: code_name_date
        group_key = f"{file_info['code']}_{file_info['name']}_{file_info['date']}"
        
        if group_key not in csv_files:
            csv_files[group_key] = {}
            
        csv_files[group_key][file_info["type"]] = str(csv_file)
    
    # 필요한 3개 타입이 모두 있는 그룹만 반환
    complete_groups = {}
    for group_key, files in csv_files.items():
        if all(file_type in files for file_type in REQUIRED_TYPES):
            complete_groups[group_key] = files
            
    return complete_groups


def _clean_column_name(col: str) -> str:
    # 내부 공백 제거 및 알려진 별칭 통일
    c = re.sub(r"\s+", "", col)
    if c == "스탬프":
        return "시간"
    return c


def load_and_clean_csv(file_path: str) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 기본 정리: 컬럼 이름 정규화 및 불필요 컬럼 제거
    """
    # 인코딩 이슈 대비 기본 옵션 적용
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(file_path, encoding=enc, engine="python", on_bad_lines="skip")
            break
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(file_path)

    # 컬럼명 정규화
    df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})
    
    # 중복 컬럼명 처리: 동일 이름 컬럼이 여러 개면, 행 단위로 첫 번째 유효값을 선택해 단일 컬럼으로 축약
    if df.columns.duplicated().any():
        new_cols = {}
        for col in dict.fromkeys(df.columns):  # preserve order, unique keys
            same = [c for c in df.columns if c == col]
            if len(same) == 1:
                continue
            # coalesce across duplicates
            block = df[same]
            new_col = block.bfill(axis=1).iloc[:, 0]
            new_cols[col] = new_col
        # assign coalesced
        for col, series in new_cols.items():
            df[col] = series
        # drop duplicates keeping first
        df = df.loc[:, ~df.columns.duplicated()]
 
    # 불필요 컬럼 제거
    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
    
    # '번호' 숫자화 보정
    if '번호' in df.columns:
        df['번호'] = pd.to_numeric(df['번호'], errors='coerce')
    
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    비어있는 데이터를 직전/직후 데이터로 채우기
    """
    # 최후 방어: 중복 컬럼 제거(이미 로드 시 처리했지만 합병 과정에서 생길 수 있음)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 번호 컬럼 기준으로 정렬
    df = df.sort_values('번호').reset_index(drop=True)
    
    # 텍스트 컬럼과 숫자 컬럼 분리
    text_cols = [col for col in df.columns if col in TEXT_COLUMNS]
    numeric_cols = [col for col in df.columns if col not in TEXT_COLUMNS and col != '번호']
    
    # 텍스트 컬럼: forward fill 후 backward fill, 그래도 없으면 빈 문자열
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna('').infer_objects(copy=False)
    
    # 숫자 컬럼: forward fill 후 backward fill, 그래도 없으면 0
    for col in numeric_cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                # 안전장치: 만약 여전히 DataFrame이면 첫 열 사용
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors='coerce')
            series = series.ffill().bfill().fillna(0)
            df[col] = series
    
    return df


def _coalesce_into_base(base: pd.DataFrame, temp: pd.DataFrame, overlap_cols: List[str]) -> pd.DataFrame:
    """
    base와 temp(번호로 병합된 상태)에서 동일 컬럼이 있을 때 base의 NaN을 temp의 값으로 보완
    overlap_cols에 대해 base[col] = base[col].combine_first(temp[f"{col}_new"]) 수행
    """
    for col in overlap_cols:
        new_col = f"{col}_new"
        if new_col in temp.columns:
            # 숫자/문자 모두 지원되는 combine_first 사용
            base[col] = base[col].combine_first(temp[new_col])
            temp = temp.drop(columns=[new_col])
    return base, temp


def merge_csv_files(files: Dict[str, str], code: str, name: str) -> pd.DataFrame:
    """
    3개의 CSV 파일을 번호 컬럼 기준으로 merge
    """
    dfs = {}
    
    # 각 파일 로드
    for file_type, file_path in files.items():
        df = load_and_clean_csv(file_path)
        df = fill_missing_values(df)
        dfs[file_type] = df
    
    # 번호의 합집합 구성 (세 소스 모두 포함)
    all_nums = sorted(set().union(*(df['번호'].dropna().astype(int).tolist() for df in dfs.values())))
    merged_df = pd.DataFrame({"번호": all_nums})

    # 세 소스 순차 병합: 겹치는 컬럼은 값 보완(coalesce), 새로운 컬럼은 추가
    for key in ("execution", "orderbook", "trader"):
        src = dfs.get(key)
        if src is None:
            continue
        # 번호만 남기거나 전체를 준비
        to_merge = src.copy()
        # full outer를 흉내내기 위해 base 기준 left merge
        temp = merged_df.merge(to_merge, on="번호", how="left", suffixes=("", "_new"))
        # 겹치는 컬럼 목록 산출 (번호 제외, _new 붙은 대상만)
        overlap = [c for c in to_merge.columns if c != "번호" and c in merged_df.columns]
        merged_df, temp = _coalesce_into_base(merged_df, temp, overlap)
        # non-overlap 신규 컬럼들을 merged_df에 반영
        new_cols = [c for c in temp.columns if c not in merged_df.columns]
        if new_cols:
            merged_df = temp[[*merged_df.columns, *new_cols]]
        else:
            merged_df = temp[merged_df.columns]
    
    # 종목코드, 종목명 추가
    merged_df['종목코드'] = merged_df.get('종목코드', pd.Series(index=merged_df.index, dtype=object)).fillna(code).replace({"": code})
    merged_df['종목명'] = merged_df.get('종목명', pd.Series(index=merged_df.index, dtype=object)).fillna(name).replace({"": name})

    # 누락 컬럼 생성 (최종 스키마 강제)
    for col in FINAL_COLUMNS:
        if col not in merged_df.columns:
            merged_df[col] = "" if col in TEXT_COLUMNS else 0

    # 채우기(직전/직후)로 결측 제거
    merged_df = fill_missing_values(merged_df)

    # 최종 컬럼 순서 맞추기 (정확히 스키마 강제)
    merged_df = merged_df[["번호", *FINAL_COLUMNS]]
    
    return merged_df


def ensure_datasets_table_duckdb(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    table = "datasets"
    # Create table if not exists
    try:
        conn.execute(f"DESCRIBE {table}")
        exists = True
    except Exception:
        exists = False
    if not exists:
        # Build explicit schema to preserve types (avoid df.head(0) inference)
        col_defs: list[str] = []
        for col in df.columns:
            series = df[col]
            if col == '번호' or pd.api.types.is_integer_dtype(series):
                duck_type = 'BIGINT'
            elif col in TEXT_COLUMNS or col in {'날짜', '종목명'} or series.dtype == object:
                duck_type = 'VARCHAR'
            else:
                duck_type = 'DOUBLE'
            col_defs.append(f'"{col}" {duck_type}')
        create_sql = f"CREATE TABLE {table} ({', '.join(col_defs)})"
        conn.execute(create_sql)
    else:
        # Use correct indices from PRAGMA table_info: (cid, name, type, null, default, pk)
        existing_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        existing_cols = [row[1] for row in existing_info]
        existing_types = {row[1]: (row[2] or "").upper() for row in existing_info}
        for col in df.columns:
            if col not in existing_cols:
                series = df[col]
                if col in TEXT_COLUMNS or col == '날짜' or series.dtype == object:
                    col_type = 'VARCHAR'
                elif pd.api.types.is_integer_dtype(series) or col == '번호':
                    col_type = 'BIGINT'
                else:
                    col_type = 'DOUBLE'
                conn.execute(f"ALTER TABLE {table} ADD COLUMN \"{col}\" {col_type}")
        # Enforce VARCHAR for known text columns if mismatched (e.g., '종목명' mistakenly INT)
        text_like = set(TEXT_COLUMNS) | {"날짜", "종목명"}
        # Drop dependent index before altering types to avoid catalog error
        try:
            conn.execute("DROP INDEX IF EXISTS idx_datasets_code_date")
        except Exception:
            pass
        for col in (c for c in df.columns if c in text_like and c in existing_types):
            ctype = existing_types.get(col, "")
            if "CHAR" not in ctype and "STRING" not in ctype and "VARCHAR" not in ctype:
                conn.execute(f"ALTER TABLE {table} ALTER COLUMN \"{col}\" TYPE VARCHAR")
    # Helpful index (recreate if dropped)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_code_date ON datasets(\"종목코드\", \"날짜\")")


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


def _valid_yyyymmdd(s: Optional[str]) -> Optional[str]:
    """Return s if it matches YYYYMMDD (8 digits), else None."""
    if s is None:
        return None
    s = s.strip()
    return s if re.fullmatch(r"\d{8}", s) else None


def _date_in_range(date: str, start: Optional[str], end: Optional[str]) -> bool:
    """Check if YYYYMMDD `date` is within [start, end] (inclusive). None means open bound."""
    if start is not None and date < start:
        return False
    if end is not None and date > end:
        return False
    return True


def _ingest_pickle_into_db_path(db_path: str, pkl_path: str, code: str, date: str, group_key: str):
    """Open a DuckDB connection to db_path and ingest the pickle contents, then remove the pickle."""
    try:
        with open(pkl_path, 'rb') as f:
            merged_df = _pickle.load(f)
        # Overwrite/assign global sequential '번호'
        global NO_COUNTER
        n_rows = len(merged_df)
        if n_rows > 0:
            merged_df['번호'] = np.arange(NO_COUNTER, NO_COUNTER + n_rows, dtype=np.int64)
            NO_COUNTER += n_rows
        conn = duckdb.connect(db_path)
        try:
            ensure_datasets_table_duckdb(conn, merged_df)
            conn.register("_batch_df", merged_df)
            try:
                conn.execute("DELETE FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
            except Exception:
                pass
            conn.execute("INSERT INTO datasets SELECT * FROM _batch_df")
            conn.unregister("_batch_df")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        print(f"완료(즉시 반영): {group_key} - {len(merged_df)} 행 -> {db_path}")
    finally:
        try:
            os.remove(pkl_path)
        except Exception:
            pass


def _prep_pkl_metadata(pkl_path: str) -> Optional[Tuple[str, str, str, str]]:
    """Top-level helper: from a pickle file path, extract (group_key, code, date, pkl_path).
    Returns None if filename doesn't match expected pattern. Using only built-in types for pickling safety.
    """
    try:
        name = os.path.basename(pkl_path)
        group_key = os.path.splitext(name)[0]
        parts = group_key.split('_')
        if len(parts) < 2:
            return None
        code = parts[0]
        date = parts[-1]
        return group_key, code, date, pkl_path
    except Exception:
        return None


def _process_single_group_to_pickle(group_key: str, files: Dict[str, str], tmp_root: str) -> Optional[str]:
    """Process a single group and save to pickle file. Returns pickle path or None on error."""
    try:
        parts = group_key.split('_')
        code = parts[0]
        date = parts[-1]
        name = '_'.join(parts[1:-1])
        
        merged_df = merge_csv_files(files, code, name)
        merged_df = merged_df.copy()
        merged_df['날짜'] = date
        merged_df = pd.concat([merged_df['날짜'], merged_df.drop(columns=['날짜'])], axis=1)
        
        # Save to pickle
        out_path = Path(tmp_root) / f"{group_key}.pkl"
        out_tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(out_tmp, 'wb') as f:
            _pickle.dump(merged_df, f, protocol=_pickle.HIGHEST_PROTOCOL)
        try:
            os.replace(out_tmp, out_path)
        except Exception:
            try:
                os.remove(out_tmp)
            except Exception:
                pass
            raise
        return str(out_path)
    except Exception as e:
        print(f"  경고: 그룹 처리 실패({group_key}): {type(e).__name__}: {e}")
        return None


def _ingest_pickles_to_db(pickle_paths: List[str], db_path: str, yyyymm: str, checkpoint_interval: int) -> int:
    """Ingest pickle files to DB sequentially with periodic checkpoints."""
    import duckdb
    processed_count = 0
    
    for pkl_path in pickle_paths:
        if not os.path.exists(pkl_path):
            continue
            
        try:
            # Load pickle and extract metadata
            with open(pkl_path, 'rb') as f:
                merged_df = _pickle.load(f)
            # Overwrite/assign global sequential '번호'
            global NO_COUNTER
            n_rows = len(merged_df)
            if n_rows > 0:
                merged_df['번호'] = np.arange(NO_COUNTER, NO_COUNTER + n_rows, dtype=np.int64)
                NO_COUNTER += n_rows
            
            group_key = Path(pkl_path).stem
            parts = group_key.split('_')
            code = parts[0]
            date = parts[-1]
            
            # Write to DB
            conn = duckdb.connect(db_path)
            try:
                ensure_datasets_table_duckdb(conn, merged_df)
                conn.register("_batch_df", merged_df)
                try:
                    conn.execute("DELETE FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
                except Exception:
                    pass
                conn.execute("INSERT INTO datasets SELECT * FROM _batch_df")
                conn.unregister("_batch_df")
            finally:
                conn.close()
            
            processed_count += 1
            print(f"  {yyyymm}: [{processed_count}] {group_key} - {len(merged_df)} 행 -> DB")
            
            # Remove pickle file after successful ingestion
            try:
                os.remove(pkl_path)
            except Exception:
                pass
            
            # Periodic checkpoint
            if checkpoint_interval > 0 and processed_count % checkpoint_interval == 0:
                try:
                    conn_ck = duckdb.connect(db_path)
                    try:
                        conn_ck.execute("CHECKPOINT")
                        print(f"  {yyyymm}: 체크포인트 실행 ({processed_count} 그룹 완료)")
                    finally:
                        conn_ck.close()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"  경고: {yyyymm} pickle 처리 실패({pkl_path}): {type(e).__name__}: {e}")
    
    # Final checkpoint
    if processed_count > 0:
        try:
            conn_ck = duckdb.connect(db_path)
            try:
                conn_ck.execute("CHECKPOINT")
                print(f"  {yyyymm}: 최종 체크포인트 실행 ({processed_count} 그룹 완료)")
            finally:
                conn_ck.close()
        except Exception:
            pass
    
    return processed_count


def _process_monthly_groups(month_groups: List[Tuple[str, Dict[str, str]]], 
                           db_path: str, tmp_root: str, yyyymm: str, 
                           checkpoint_interval: int, group_workers: int = 1) -> int:
    """Process all groups for a specific month with parallel group processing.
    Returns the number of processed groups.
    """
    if not month_groups:
        return 0
    
    total_groups = len(month_groups)
    print(f"  {yyyymm}: {total_groups} 그룹 처리 시작 (group_workers={group_workers})", flush=True)
    
    # Step 1: Process groups to pickle files in parallel
    pickle_paths: List[str] = []
    max_group_workers = max(1, int(group_workers))
    used_group_workers = min(max_group_workers, total_groups)
    
    if used_group_workers > 1:
        print(f"  {yyyymm}: 병렬 그룹 처리 (workers={used_group_workers})", flush=True)
        with _fut.ProcessPoolExecutor(max_workers=used_group_workers) as ex:
            futs = []
            for group_key, files in month_groups:
                fut = ex.submit(_process_single_group_to_pickle, group_key, files, tmp_root)
                futs.append(fut)
            done = 0
            for fut in _fut.as_completed(futs):
                try:
                    pkl_path = fut.result()
                    if pkl_path:
                        pickle_paths.append(pkl_path)
                        # Incremental ingest when enough pickles are ready
                        if checkpoint_interval > 0 and len(pickle_paths) >= checkpoint_interval:
                            batch = pickle_paths[:checkpoint_interval]
                            print(f"  {yyyymm}: 증분 반영 시작 (batch={len(batch)}) -> {db_path}", flush=True)
                            try:
                                _ingest_pickles_to_db(batch, db_path, yyyymm, checkpoint_interval)
                                # remove ingested ones from buffer
                                pickle_paths = pickle_paths[checkpoint_interval:]
                                print(f"  {yyyymm}: 증분 반영 완료 (누적 대기 {len(pickle_paths)} pickle)", flush=True)
                            except Exception as e:
                                print(f"  {yyyymm}: 증분 반영 실패: {type(e).__name__}: {e}", flush=True)
                except Exception as e:
                    print(f"  {yyyymm}: 그룹 처리 중 오류: {type(e).__name__}: {e}", flush=True)
                finally:
                    done += 1
                    if done % 5 == 0 or done == total_groups:
                        print(f"  {yyyymm}: pickle 생성 진행률 [{done}/{total_groups}]", flush=True)
    else:
        print(f"  {yyyymm}: 순차 그룹 처리", flush=True)
        done = 0
        for group_key, files in month_groups:
            pkl_path = _process_single_group_to_pickle(group_key, files, tmp_root)
            if pkl_path:
                pickle_paths.append(pkl_path)
                # Incremental ingest when buffer reaches checkpoint size
                if checkpoint_interval > 0 and len(pickle_paths) >= checkpoint_interval:
                    batch = pickle_paths[:checkpoint_interval]
                    print(f"  {yyyymm}: 증분 반영 시작 (batch={len(batch)}) -> {db_path}", flush=True)
                    try:
                        _ingest_pickles_to_db(batch, db_path, yyyymm, checkpoint_interval)
                        pickle_paths = pickle_paths[checkpoint_interval:]
                        print(f"  {yyyymm}: 증분 반영 완료 (누적 대기 {len(pickle_paths)} pickle)", flush=True)
                    except Exception as e:
                        print(f"  {yyyymm}: 증분 반영 실패: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == total_groups:
                print(f"  {yyyymm}: pickle 생성 진행률 [{done}/{total_groups}]", flush=True)
     
    # Step 2: Ingest pickle files to DB sequentially (to avoid DB write conflicts)
    if pickle_paths:
        print(f"  {yyyymm}: {len(pickle_paths)} pickle 파일을 DB에 순차 반영", flush=True)
        processed_count = _ingest_pickles_to_db(pickle_paths, db_path, yyyymm, checkpoint_interval)
    else:
        processed_count = 0
     
    return processed_count


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
    
    print(f"최종 체크포인트 실행: {len(month_paths)}개 DB 파일")
    
    if used_workers > 1:
        print(f"  병렬 실행 (workers={used_workers})")
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {ex.submit(_checkpoint_db_once, path): path for path in month_paths}
            completed = 0
            for fut in _fut.as_completed(futs):
                path = futs[fut]
                try:
                    _, ok, msg = fut.result()
                    completed += 1
                    if ok:
                        print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 완료: {os.path.basename(path)}")
                    else:
                        print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {msg}")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {type(e).__name__}: {e}")
    else:
        print("  순차 실행")
        for i, path in enumerate(month_paths, 1):
            _, ok, msg = _checkpoint_db_once(path)
            if ok:
                print(f"  [{i}/{len(month_paths)}] CHECKPOINT 완료: {os.path.basename(path)}")
            else:
                print(f"  [{i}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {msg}")


def _sweep_and_ingest_tmp(base_db_path: str, tmp_root: Path, workers: int = 1, checkpoint_interval: int = 20, *, single_output: bool = False):
    """Scan tmp_root for any leftover .pkl files and ingest them in the current process.
    This supports resume-on-start and graceful Ctrl+C handling.
    """
    if not tmp_root.exists():
        return
    pkls = sorted(p for p in tmp_root.glob("*.pkl"))
    if not pkls:
        return
    print(f"임시 체크포인트 {len(pkls)}개를 DB에 반영합니다 (resume/cleanup)...")
    prepared: list[tuple[str, str, str, str]] = []
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(pkls))
    if used_workers > 1:
        print(f"- 메타데이터 병렬 준비: workers={used_workers}")
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = [ex.submit(_prep_pkl_metadata, str(p)) for p in pkls]
            done = 0
            for fut in _fut.as_completed(futs):
                try:
                    res = fut.result()
                    if res is not None:
                        prepared.append(res)
                    else:
                        # remove unknown naming
                        pass
                finally:
                    done += 1
    else:
        print("- 메타데이터 직렬 준비 (workers=1)")
        for p in pkls:
            res = _prep_pkl_metadata(str(p))
            if res is not None:
                prepared.append(res)
            else:
                try:
                    os.remove(p)
                except Exception:
                    pass

    print("- DuckDB 반영: 순차 처리 (쓰기 경합 방지)")
    month_counts: Dict[str, int] = {}
    for i, (group_key, code, date, pkl_path) in enumerate(prepared, 1):
        try:
            yyyymm = _month_key_from_yyyymmdd(date)
            db_path = base_db_path if single_output else _monthly_db_path(base_db_path, yyyymm)
            _ingest_pickle_into_db_path(db_path, pkl_path, code, date, group_key)
            # per-month checkpoint interval
            if checkpoint_interval > 0:
                month_counts[yyyymm] = month_counts.get(yyyymm, 0) + 1
                if month_counts[yyyymm] % checkpoint_interval == 0:
                    try:
                        conn_ck = duckdb.connect(db_path)
                        try:
                            conn_ck.execute("CHECKPOINT")
                        finally:
                            conn_ck.close()
                    except Exception:
                        pass
        finally:
            # Always try to remove the pickle to free disk space
            try:
                os.remove(pkl_path)
            except Exception:
                pass


def generate_datasets(input_folder: str, output_db: str, *,
                      skip_existing: bool = True,
                      force_recreate: bool = False,
                      workers: int = 1,
                      group_workers: int = 1,
                      tmp_dir: Optional[str] = None,
                      checkpoint_interval: int = 20,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      single_output: bool = False):
    """
    메인 데이터 생성 함수 (정규화 없음, DuckDB 전용)
    - 입력 폴더를 스캔하여 유효 CSV 그룹을 찾음
    - 그룹을 날짜(YYYYMMDD)에서 월(YYYYMM)로 묶어 월별 DuckDB 샤드에 기록
    - 최대 `workers`개의 월을 병렬로 처리
    - 각 월 내부에서는 최대 `group_workers`개의 그룹을 병렬 처리
    - `checkpoint-interval`마다 CHECKPOINT 실행
    - 작업 중단 복구를 위해 temp 디렉토리에 단계별 체크포인트(.pkl)를 사용하고 시작 시 반영
    - 선택적으로 `start_date` ~ `end_date` (YYYYMMDD) 범위의 날짜만 처리
    """

    # 입력 폴더 검증
    folder = Path(input_folder)
    if not folder.exists():
        print(f"입력 폴더가 존재하지 않습니다: {input_folder}")
        return

    # temp 디렉토리 준비 및 resume 처리
    p = Path(output_db)
    _tmp_base = Path(tmp_dir) if tmp_dir else p.with_suffix(p.suffix + ".tmp")
    _tmp_base.mkdir(parents=True, exist_ok=True)
    # 남은 부분 파일 정리
    for part in _tmp_base.glob("*.pkl.part"):
        try:
            os.remove(part)
        except Exception:
            pass
    # 이전 실행의 완료된 체크포인트 반영
    # 이전 실행의 완료된 체크포인트 반영 (single-output 모드 고려)
    _sweep_and_ingest_tmp(output_db, _tmp_base, workers=group_workers, checkpoint_interval=checkpoint_interval, single_output=single_output)

    # CSV 그룹 스캔
    print("CSV 파일 스캔 및 그룹화 중...")
    complete_groups = find_csv_files(str(folder))
    if not complete_groups:
        print("처리할 유효 CSV 그룹을 찾지 못했습니다.")
        return

    # 날짜 범위 필터링 (옵션)
    if start_date or end_date:
        s = _valid_yyyymmdd(start_date)
        e = _valid_yyyymmdd(end_date)
        if start_date and s is None:
            print(f"경고: start-date 형식이 잘못되었습니다(YYYYMMDD 기대): {start_date} -> 무시합니다")
        if end_date and e is None:
            print(f"경고: end-date 형식이 잘못되었습니다(YYYYMMDD 기대): {end_date} -> 무시합니다")
        # 범위가 뒤바뀐 경우 자동 수정
        if s is not None and e is not None and s > e:
            print(f"경고: start-date({s}) > end-date({e}) 이므로 서로 교체합니다.")
            s, e = e, s
        filt: Dict[str, Dict[str, str]] = {}
        for group_key, files in complete_groups.items():
            parts = group_key.split("_")
            date = parts[-1]
            if _date_in_range(date, s, e):
                filt[group_key] = files
        complete_groups = filt
        if not complete_groups:
            print("지정한 날짜 범위에 해당하는 그룹이 없습니다.")
            return

    # 그룹을 월별로 묶기
    monthly_groups: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for group_key, files in complete_groups.items():
        parts = group_key.split("_")
        date = parts[-1]
        yyyymm = _month_key_from_yyyymmdd(date)
        monthly_groups.setdefault(yyyymm, []).append((group_key, files))

    # force-recreate 처리
    if force_recreate:
        if single_output:
            if os.path.exists(output_db):
                try:
                    os.remove(output_db)
                    print(f"삭제 후 재생성 예정(단일): {output_db}")
                except Exception as e:
                    print(f"경고: 단일 DB 삭제 실패 {output_db}: {type(e).__name__}: {e}")
        else:
            for yyyymm in monthly_groups.keys():
                db_path = _monthly_db_path(output_db, yyyymm)
                if os.path.exists(db_path):
                    try:
                        os.remove(db_path)
                        print(f"삭제 후 재생성 예정: {db_path}")
                    except Exception as e:
                        print(f"경고: DB 삭제 실패 {db_path}: {type(e).__name__}: {e}")

    # skip-existing: 각 월 DB에서 이미 존재하는 (종목코드, 날짜) 그룹 제거
    def _filter_skip_existing_for_month(yyyymm: str, groups: List[Tuple[str, Dict[str, str]]]) -> List[Tuple[str, Dict[str, str]]]:
        if not skip_existing:
            return groups
        db_path = _monthly_db_path(output_db, yyyymm)
        # DB가 없으면 전부 유지
        if not os.path.exists(db_path):
            return groups
        try:
            conn = duckdb.connect(db_path)
            try:
                try:
                    conn.execute("DESCRIBE datasets")
                    table_exists = True
                except Exception:
                    table_exists = False
                if not table_exists:
                    return groups
                keep: List[Tuple[str, Dict[str, str]]] = []
                # Batch existence check by date per code would be ideal; simple loop for clarity
                for group_key, files in groups:
                    parts = group_key.split('_')
                    code = parts[0]
                    date = parts[-1]
                    try:
                        q = conn.execute("SELECT 1 FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=? LIMIT 1", [code, date]).fetchone()
                    except Exception:
                        q = None
                    if q is None:
                        keep.append((group_key, files))
                return keep
            finally:
                conn.close()
        except Exception:
            # 보수적으로 모두 처리
            return groups

    if single_output:
        # 단일 DB 파일에서 스킵 여부 확인
        if skip_existing and os.path.exists(output_db):
            try:
                conn = duckdb.connect(output_db)
                try:
                    try:
                        conn.execute("DESCRIBE datasets")
                        table_exists = True
                    except Exception:
                        table_exists = False
                    if table_exists:
                        for yyyymm in list(monthly_groups.keys()):
                            orig_n = len(monthly_groups[yyyymm])
                            keep: List[Tuple[str, Dict[str, str]]] = []
                            for group_key, files in monthly_groups[yyyymm]:
                                parts = group_key.split('_')
                                code = parts[0]
                                date = parts[-1]
                                try:
                                    q = conn.execute("SELECT 1 FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=? LIMIT 1", [code, date]).fetchone()
                                except Exception:
                                    q = None
                                if q is None:
                                    keep.append((group_key, files))
                            monthly_groups[yyyymm] = keep
                            if len(keep) == 0:
                                print(f"{yyyymm}: 스킵할 항목만 존재하여 건너뜁니다 (원래 {orig_n} 그룹)")
                                del monthly_groups[yyyymm]
                finally:
                    conn.close()
            except Exception:
                # 에러 시 보수적으로 전부 처리
                pass
        else:
            # 출력 DB가 없거나 skip 비활성화면 그대로 진행
            pass
    else:
        for yyyymm in list(monthly_groups.keys()):
            orig_n = len(monthly_groups[yyyymm])
            monthly_groups[yyyymm] = _filter_skip_existing_for_month(yyyymm, monthly_groups[yyyymm])
            if len(monthly_groups[yyyymm]) == 0:
                print(f"{yyyymm}: 스킵할 항목만 존재하여 건너뜁니다 (원래 {orig_n} 그룹)")
                del monthly_groups[yyyymm]

    if not monthly_groups:
        print("처리할 신규 그룹이 없습니다.")
        # 임시 디렉토리 정리 후 종료
        try:
            _shutil.rmtree(_tmp_base)
        except Exception:
            pass
        return

    # 월 목록 및 병렬 처리 설정
    months = sorted(monthly_groups.keys())
    max_workers = max(1, int(workers))
    used_workers = 1 if single_output else min(max_workers, len(months))

    print(f"월별 처리 시작: 대상 {len(months)}개월, 병렬 workers={used_workers}")

    # 병렬로 월별 처리 실행
    if used_workers > 1:
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {}
            for yyyymm in months:
                db_path = output_db if single_output else _monthly_db_path(output_db, yyyymm)
                groups = monthly_groups[yyyymm]
                fut = ex.submit(_process_monthly_groups, groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers))
                futs[fut] = (yyyymm, len(groups), db_path)
            done = 0
            total = len(futs)
            for fut in _fut.as_completed(futs):
                yyyymm, n_groups, db_path = futs[fut]
                try:
                    processed = fut.result()
                    print(f"월 처리 완료: {yyyymm} ({processed}/{n_groups}) -> {db_path}")
                except Exception as e:
                    print(f"경고: 월 처리 실패 {yyyymm}: {type(e).__name__}: {e}")
                finally:
                    done += 1
    else:
        # 직렬 처리
        for yyyymm in months:
            db_path = output_db if single_output else _monthly_db_path(output_db, yyyymm)
            groups = monthly_groups[yyyymm]
            processed = _process_monthly_groups(groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers))
            print(f"월 처리 완료: {yyyymm} ({processed}/{len(groups)}) -> {db_path}")

    # 처리된 월들에 대해 병렬 최종 CHECKPOINT 수행 (선택적)
    try:
        if single_output:
            # 단일 파일만 체크포인트
            conn = duckdb.connect(output_db)
            try:
                conn.execute("CHECKPOINT")
            finally:
                conn.close()
        else:
            processed_months = months
            if processed_months:
                _parallel_checkpoint_months(output_db, processed_months, max_workers)
    except Exception:
        pass

    # 임시 디렉토리 정리
    try:
        _shutil.rmtree(_tmp_base)
    except Exception:
        pass

    print(f"데이터 생성 완료: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="CSV 데이터셋 생성 스크립트 (정규화 없음, DuckDB)")
    parser.add_argument("input_folder", help="입력 CSV 폴더 경로")
    parser.add_argument("-o", "--output", default="datasets.duckdb", help="출력 DuckDB 파일명")
    # Skip-existing 옵션 (기본 활성화). 비활성화하려면 --no-skip-existing 사용
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="이미 DB에 해당 (종목코드, 날짜) 그룹이 존재하면 스킵합니다 (기본: 활성화)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="이미 존재하는 그룹도 다시 처리합니다")
    # Force recreate DB if exists (useful when file is corrupted or version-mismatched)
    parser.add_argument("--force-recreate", action="store_true",
                        help="출력 DuckDB 파일이 존재하면 삭제 후 새로 생성합니다 (손상/버전 문제 해결용)")
    # 병렬 처리 관련
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="월별 병렬 처리에 사용할 프로세스 수 (기본: CPU 코어 수)")
    parser.add_argument("--group-workers", type=int, default=1,
                        help="각 월 내에서 그룹 병렬 처리에 사용할 프로세스 수 (기본: 1)")
    parser.add_argument("--tmp-dir", default=None,
                        help="임시 결과 저장 디렉토리 (기본: <output>.tmp)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="몇 개 그룹 처리마다 DuckDB CHECKPOINT를 실행할지 지정 (0이면 비활성화, 기본: 100)")
    # 날짜 범위 옵션
    parser.add_argument("--start-date", dest="start_date", default=None,
                        help="처리 시작 날짜 (YYYYMMDD)")
    parser.add_argument("--end-date", dest="end_date", default=None,
                        help="처리 종료 날짜 (YYYYMMDD)")
    parser.add_argument("--single-output", dest="single_output", action="store_true",
                        help="모든 월 데이터를 단일 출력 DuckDB 파일에 순차적으로 append 합니다 (병렬 월 처리 비활성화)")

    args = parser.parse_args()

    generate_datasets(
        args.input_folder,
        args.output,
        skip_existing=args.skip_existing,
        force_recreate=args.force_recreate,
        workers=args.workers,
        group_workers=args.group_workers,
        tmp_dir=args.tmp_dir,
        checkpoint_interval=args.checkpoint_interval,
        start_date=args.start_date,
        end_date=args.end_date,
        single_output=args.single_output,
    )


if __name__ == "__main__":
    main()
