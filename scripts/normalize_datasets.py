import argparse
import os
import re
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures as _fut
import shutil as _shutil
import pickle as _pickle

import numpy as np
import pandas as pd

# Global sequential counter for '번호'
NO_COUNTER: int = 1

INPUT_TABLE = "datasets"  # 입력 DuckDB 테이블명
TEXT_COLUMNS = {"종목코드", "종목명", "시간", *{f"매도거래원{i}" for i in range(1, 6)}, *{f"매수거래원{i}" for i in range(1, 6)}}
DROP_COLUMNS = {"종류", "씨리얼"}
IGNORING_STOCKS_SET: set[str] = set()

def _load_ignoring_stocks(csv_path: Optional[str]) -> set[str]:
    """Load ignoring stock names from a CSV file that contains a '종목명' column.
    Returns a set of stripped names. Returns empty set on error or if path is None.
    """
    if not csv_path:
        return set()
    try:
        if not os.path.exists(csv_path):
            print(f"경고: ignoring_stocks CSV를 찾을 수 없습니다: {csv_path}")
            return set()
        df_csv = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", on_bad_lines="skip")
        if '종목명' not in df_csv.columns:
            # Try case-insensitive match
            cols_lower = {c.lower(): c for c in df_csv.columns}
            key = cols_lower.get('종목명'.lower())
            if key:
                names = df_csv[key]
            else:
                print(f"경고: ignoring_stocks CSV에 '종목명' 컬럼이 없습니다: {csv_path}")
                return set()
        else:
            names = df_csv['종목명']
        s = (
            names.astype(str)
                 .str.replace('\r', '', regex=False)
                 .str.replace('\n', '', regex=False)
                 .str.strip()
        )
        return {x for x in s.tolist() if x and x.lower() != 'nan'}
    except Exception as e:
        print(f"경고: ignoring_stocks CSV 로드 실패: {type(e).__name__}: {e}")
        return set()

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


def find_duckdb_groups(input_db: str) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    """
    입력 DuckDB에서 데이터 그룹을 찾고 그룹화
    Returns: {group_key: {"merged": (db_path, code, name, date)}}
    """
    if not os.path.exists(input_db):
        return {}
    
    groups = {}
    try:
        conn = duckdb.connect(input_db, read_only=True)
        try:
            # 테이블 존재 확인
            try:
                conn.execute(f"DESCRIBE {INPUT_TABLE}")
            except Exception:
                print(f"경고: 입력 DB에 '{INPUT_TABLE}' 테이블이 없습니다.")
                return {}
            
            # 종목코드, 종목명, 날짜별로 그룹화
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
                # 입력 DB에서는 이미 병합된 데이터이므로 단일 타입으로 처리
                groups[group_key] = {
                    "merged": (input_db, str(code), str(name), str(date))
                }
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 그룹 스캔 실패: {type(e).__name__}: {e}")
        return {}
    
    return groups


def _clean_column_name(col: str) -> str:
    # 내부 공백 제거 및 알려진 별칭 통일
    c = re.sub(r"\s+", "", col)
    if c == "스탬프":
        return "시간"
    return c


def load_and_clean_from_duckdb(db_path: str, code: str, name: str, date: str, 
                                time_start: int = 90000000, time_end: int = 110000000) -> pd.DataFrame:
    """
    DuckDB에서 특정 종목코드/날짜의 데이터를 로드하고 기본 정리
    time_start: 시작 시간 (기본: 90000000 = 오전 9시)
    time_end: 종료 시간 (기본: 110000000 = 오전 11시, 미포함)
    """
    try:
        conn = duckdb.connect(db_path, read_only=True)
        try:
            query = f"""
                SELECT *
                FROM {INPUT_TABLE}
                WHERE "종목코드" = ? AND "날짜" = ?
                ORDER BY "번호"
            """
            df = conn.execute(query, [code, date]).df()
            
            if df.empty:
                return pd.DataFrame()
            
            # 컬럼명 정규화 (이미 정규화되어 있을 수 있지만 안전장치)
            df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})

            # 종목명 필터링: IGNORING_STOCKS에 포함된 종목은 제외
            if '종목명' in df.columns:
                mask_ignore = df['종목명'].astype(str).str.strip().isin(IGNORING_STOCKS_SET)
                if mask_ignore.any():
                    df = df[~mask_ignore]
            
            # 시간 필터링 적용
            if '시간' in df.columns:
                # 시간 컬럼을 숫자로 변환
                df['시간_numeric'] = pd.to_numeric(df['시간'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0).astype(int)
                # 시간 범위 필터링: time_start <= 시간 < time_end
                df = df[(df['시간_numeric'] >= time_start) & (df['시간_numeric'] < time_end)]
                # 임시 컬럼 제거
                df = df.drop(columns=['시간_numeric'])
            
            # 불필요 컬럼 제거
            df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
            
            # '번호' 숫자화 보정
            if '번호' in df.columns:
                df['번호'] = pd.to_numeric(df['번호'], errors='coerce')
            
            return df
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 로드 실패 ({code}, {date}): {type(e).__name__}: {e}")
        return pd.DataFrame()


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


 


def merge_from_duckdb(group_info: Dict[str, Tuple[str, str, str, str]], code: str, name: str, date: str,
                       time_start: int = 90000000, time_end: int = 110000000) -> pd.DataFrame:
    """
    DuckDB에서 데이터를 로드하여 병합 (이미 병합된 데이터인 경우 그대로 반환)
    """
    # 입력 DB에서는 이미 병합된 상태이므로 단순히 로드만 수행
    if "merged" in group_info:
        db_path, code, name, date = group_info["merged"]
        df = load_and_clean_from_duckdb(db_path, code, name, date, time_start, time_end)
        
        if df.empty:
            return pd.DataFrame()
        
        # 결측값 채우기
        df = fill_missing_values(df)
        
        # 종목코드, 종목명 보정
        df['종목코드'] = df.get('종목코드', pd.Series(index=df.index, dtype=object)).fillna(code).replace({"": code})
        df['종목명'] = df.get('종목명', pd.Series(index=df.index, dtype=object)).fillna(name).replace({"": name})
        
        # 누락 컬럼 생성 (최종 스키마 강제)
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col in TEXT_COLUMNS else 0
        
        # 채우기(직전/직후)로 결측 제거
        df = fill_missing_values(df)
        
        # 최종 컬럼 순서 맞추기
        df = df[["번호", *FINAL_COLUMNS]]
        
        return df
    
    # 레거시: 여러 타입이 분리되어 있는 경우 (향후 확장용)
    return pd.DataFrame()


def _signed_log1p(arr: pd.Series) -> pd.Series:
    """Signed log1p that supports negative values: sign(x) * log1p(|x|)."""
    x = pd.to_numeric(arr, errors="coerce").fillna(0)
    return np.sign(x) * np.log1p(np.abs(x))


def _standard_scale(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    mean = float(x.mean())
    std = float(x.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - mean) / std


def _minmax_scale(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_v = float(x.min())
    max_v = float(x.max())
    rng = max_v - min_v
    if rng == 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - min_v) / rng


def apply_feature_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns per rules:
    - Log + Standard: specified quantity and flow columns
    - Standard only: *_총잔량직전대비
    - Character-level scalar encoding for broker categorical columns (append scalar, keep originals)
    - Character-level scalar encoding for '종목명' (append scalar, keep original)
    - '시간' normalized to '시간_scalar' by dividing numeric value by 90000000.0 (keep original '시간')
    - Min-Max: remaining numeric columns (excluding '번호' and text columns and already-normalized columns)
    """
    out = df.copy()

    # Define column groups
    logstd_cols = set([
        "거래량", "누적거래량", "누적거래대금", "거래회전율",
        *[f"매도호가수량{i}" for i in range(1, 11)],
        *[f"매수호가수량{i}" for i in range(1, 11)],
        "매도호가총잔량", "매수호가총잔량",
        *[f"매도거래원수량{i}" for i in range(1, 6)],
        *[f"매수거래원수량{i}" for i in range(1, 6)],
        *[f"매도거래원별증감{i}" for i in range(1, 6)],
        *[f"매수거래원별증감{i}" for i in range(1, 6)],
        # 외국계 추정 관련 (없으면 생성 후 0)
        "외국계매도추정합", "외국계매수추정합", "외국계매도추정합변동", "외국계매수추정합변동",
    ])

    stdonly_cols = set(["매도호가총잔량직전대비", "매수호가총잔량직전대비", "등락률"])

    broker_cat_cols = [*[f"매도거래원{i}" for i in range(1, 6)], *[f"매수거래원{i}" for i in range(1, 6)]]

    # Ensure optional columns exist
    for col in list(logstd_cols | stdonly_cols):
        if col not in out.columns:
            out[col] = 0

    # Log + Standard scaling (signed log1p then z-score)
    for col in sorted(logstd_cols):
        if col in out.columns:
            out[col] = _standard_scale(_signed_log1p(out[col]))

    # Standard only
    for col in sorted(stdonly_cols):
        if col in out.columns:
            out[col] = _standard_scale(out[col])

    # Character-level scalar encoding for broker categorical columns (append scalar, keep originals)
    broker_scalar_cols: List[str] = []
    for col in broker_cat_cols:
        if col in out.columns:
            cats = out[col].astype(str).replace({"nan": ""})
            # Build per-column char dictionary
            unique_chars = sorted(set("".join(cats.tolist())))
            if len(unique_chars) == 0:
                # empty column -> scalar zeros
                out[f"{col}_scalar"] = 0.0
                broker_scalar_cols.append(f"{col}_scalar")
                continue
            char_to_id = {ch: i + 1 for i, ch in enumerate(unique_chars)}  # 1..N
            max_id = float(len(unique_chars))

            def _encode_scalar(s: str) -> float:
                if not s:
                    return 0.0
                ids = [char_to_id.get(ch, 0) for ch in s]
                if not ids:
                    return 0.0
                # mean of ids normalized by max id -> [0,1]
                return float(np.mean(ids)) / max_id

            out[f"{col}_scalar"] = cats.apply(_encode_scalar).astype(float)
            broker_scalar_cols.append(f"{col}_scalar")

    # Character-level scalar encoding for '종목명'
    if '종목명' in out.columns:
        cats = out['종목명'].astype(str).replace({"nan": ""})
        unique_chars = sorted(set("".join(cats.tolist())))
        if len(unique_chars) == 0:
            out["종목명_scalar"] = 0.0
        else:
            char_to_id = {ch: i + 1 for i, ch in enumerate(unique_chars)}
            max_id = float(len(unique_chars))

            def _encode_scalar_name(s: str) -> float:
                if not s:
                    return 0.0
                ids = [char_to_id.get(ch, 0) for ch in s]
                if not ids:
                    return 0.0
                return float(np.mean(ids)) / max_id

            out["종목명_scalar"] = cats.apply(_encode_scalar_name).astype(float)
        broker_scalar_cols.append("종목명_scalar")

    # '시간' -> '시간_scalar' (numeric normalization by 90000000.0)
    if '시간' in out.columns:
        def _to_num_time(v) -> float:
            try:
                # fast path for numeric
                val = float(v)
                return val
            except Exception:
                s = str(v)
                digits = re.sub(r"\D", "", s)
                if not digits:
                    return 0.0
                try:
                    return float(digits)
                except Exception:
                    return 0.0

        num_time = out['시간'].apply(_to_num_time).astype(float)
        denom = 90000000.0
        out['시간_scalar'] = (num_time / denom).astype(float)
        broker_scalar_cols.append('시간_scalar')

    # Min-Max for remaining numeric columns not already processed
    processed = set(["번호"]) | TEXT_COLUMNS | logstd_cols | stdonly_cols | set(broker_scalar_cols)
    numeric_rest = [c for c in out.columns if c not in processed and pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_rest:
        out[col] = _minmax_scale(out[col])

    # Final safety: fill any remaining NaNs
    for col in out.columns:
        if col in TEXT_COLUMNS:
            out[col] = out[col].astype(str).replace({"nan": ""}).fillna("")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


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


 


def _process_single_group_to_pickle(group_key: str, group_info: Dict[str, Tuple[str, str, str, str]], tmp_root: str,
                                     time_start: int = 90000000, time_end: int = 110000000,
                                     ignoring_stocks_csv: Optional[str] = None) -> Optional[str]:
    """Process a single group and save to pickle file. Returns pickle path or None on error."""
    try:
        # Ensure ignore set is loaded inside worker/subprocess
        global IGNORING_STOCKS_SET
        if not IGNORING_STOCKS_SET:
            IGNORING_STOCKS_SET = _load_ignoring_stocks(ignoring_stocks_csv)
        parts = group_key.split('_')
        code = parts[0]
        date = parts[-1]
        name = '_'.join(parts[1:-1])
        
        merged_df = merge_from_duckdb(group_info, code, name, date, time_start, time_end)
        
        if merged_df.empty:
            return None
        
        merged_df = apply_feature_normalization(merged_df)
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


def _process_monthly_groups(month_groups: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]], 
                           db_path: str, tmp_root: str, yyyymm: str, 
                           checkpoint_interval: int, group_workers: int = 1,
                           time_start: int = 90000000, time_end: int = 110000000,
                           ignoring_stocks_csv: Optional[str] = None) -> int:
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
            for group_key, group_info in month_groups:
                fut = ex.submit(_process_single_group_to_pickle, group_key, group_info, tmp_root, time_start, time_end, ignoring_stocks_csv)
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
        for group_key, group_info in month_groups:
            pkl_path = _process_single_group_to_pickle(group_key, group_info, tmp_root, time_start, time_end, ignoring_stocks_csv)
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


def _sweep_and_ingest_tmp(base_db_path: str, tmp_root: Path, workers: int = 1, checkpoint_interval: int = 20):
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
            db_path = _monthly_db_path(base_db_path, yyyymm)
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


def normalize_datasets(input_db: str, output_db: str, *,
                       skip_existing: bool = True,
                       compact_only: bool = False,
                       force_recreate: bool = False,
                       workers: int = 1,
                       group_workers: int = 1,
                       tmp_dir: Optional[str] = None,
                       checkpoint_interval: int = 20,
                       time_start: int = 90000000,
                       time_end: int = 110000000,
                       ignoring_stocks_csv: Optional[str] = None):
    """
    메인 정규화 함수 (DuckDB 전용)
    - 입력 DuckDB를 스캔하여 유효 데이터 그룹을 찾음
    - 그룹을 날짜(YYYYMMDD)에서 월(YYYYMM)로 묶어 월별 DuckDB 샤드에 기록
    - 최대 `workers`개의 월을 병렬로 처리
    - 각 월 내부에서는 최대 `group_workers`개의 그룹을 병렬 처리
    - `checkpoint-interval`마다 CHECKPOINT 실행
    - 작업 중단 복구를 위해 temp 디렉토리에 단계별 체크포인트(.pkl)를 사용하고 시작 시 반영
    """
    # compact-only 모드: 입력 DB를 읽지 않고 지정한 DB에 대해 최적화만 수행
    if compact_only:
        print("compact-only 모드: 입력 DB 처리 없이 출력 DB 최적화만 수행합니다.")
        p = Path(output_db)
        stem = p.stem
        suffix = p.suffix or ".duckdb"
        parent = p.parent
        month_files = sorted(parent.glob(f"{stem}_*{suffix}"))
        months: List[str] = []
        for f in month_files:
            m = f.stem.replace(f"{stem}_", "")
            if re.fullmatch(r"\d{6}", m):
                months.append(m)
        if not months and os.path.exists(output_db):
            _parallel_checkpoint_months(output_db, [], 1)
            try:
                conn = duckdb.connect(output_db)
                try:
                    conn.execute("CHECKPOINT")
                finally:
                    conn.close()
                print(f"DB 유지보수 완료: {output_db}")
            except Exception as e:
                print(f"경고: 단일 DB 체크포인트 실패: {type(e).__name__}: {e}")
            return
        _parallel_checkpoint_months(output_db, months, workers)
        print("월별 DB 유지보수 완료")
        return

    # 입력 DB 검증
    if not os.path.exists(input_db):
        print(f"입력 DuckDB 파일이 존재하지 않습니다: {input_db}")
        return

    # temp 디렉토리 준비 및 resume 처리
    p = Path(output_db)
    _tmp_base = Path(tmp_dir) if tmp_dir else p.with_suffix(p.suffix + ".tmp")
    _tmp_base.mkdir(parents=True, exist_ok=True)
    for part in _tmp_base.glob("*.pkl.part"):
        try:
            os.remove(part)
        except Exception:
            pass
    _sweep_and_ingest_tmp(output_db, _tmp_base, workers=group_workers, checkpoint_interval=checkpoint_interval)

    # Load ignoring stocks once in the main process
    global IGNORING_STOCKS_SET
    IGNORING_STOCKS_SET = _load_ignoring_stocks(ignoring_stocks_csv)

    # DuckDB 그룹 스캔
    print("DuckDB 데이터 스캔 및 그룹화 중...")
    complete_groups = find_duckdb_groups(input_db)
    if not complete_groups:
        print("처리할 유효 데이터 그룹을 찾지 못했습니다.")
        return

    # 그룹을 월별로 묶기
    monthly_groups: Dict[str, List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]] = {}
    for group_key, group_info in complete_groups.items():
        parts = group_key.split("_")
        date = parts[-1]
        yyyymm = _month_key_from_yyyymmdd(date)
        monthly_groups.setdefault(yyyymm, []).append((group_key, group_info))

    # force-recreate: 대상 월 DB 삭제
    if force_recreate:
        for yyyymm in monthly_groups.keys():
            db_path = _monthly_db_path(output_db, yyyymm)
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    print(f"삭제 후 재생성 예정: {db_path}")
                except Exception as e:
                    print(f"경고: DB 삭제 실패 {db_path}: {type(e).__name__}: {e}")

    # skip-existing: 각 월 DB에서 이미 존재하는 (종목코드, 날짜) 그룹 제거
    def _filter_skip_existing_for_month(yyyymm: str, groups: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]) -> List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]:
        if not skip_existing:
            return groups
        db_path = _monthly_db_path(output_db, yyyymm)
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
                keep: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]] = []
                for group_key, group_info in groups:
                    parts = group_key.split('_')
                    code = parts[0]
                    date = parts[-1]
                    try:
                        q = conn.execute("SELECT 1 FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=? LIMIT 1", [code, date]).fetchone()
                    except Exception:
                        q = None
                    if q is None:
                        keep.append((group_key, group_info))
                return keep
            finally:
                conn.close()
        except Exception:
            return groups

    for yyyymm in list(monthly_groups.keys()):
        orig_n = len(monthly_groups[yyyymm])
        monthly_groups[yyyymm] = _filter_skip_existing_for_month(yyyymm, monthly_groups[yyyymm])
        if len(monthly_groups[yyyymm]) == 0:
            print(f"{yyyymm}: 스킵할 항목만 존재하여 건너뜁니다 (원래 {orig_n} 그룹)")
            del monthly_groups[yyyymm]

    if not monthly_groups:
        print("처리할 신규 그룹이 없습니다.")
        try:
            _shutil.rmtree(_tmp_base)
        except Exception:
            pass
        return

    # 월 목록 및 병렬 처리 설정
    months = sorted(monthly_groups.keys())
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(months))

    print(f"월별 처리 시작: 대상 {len(months)}개월, 병렬 workers={used_workers}")

    # 병렬로 월별 처리 실행
    if used_workers > 1:
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {}
            for yyyymm in months:
                db_path = _monthly_db_path(output_db, yyyymm)
                groups = monthly_groups[yyyymm]
                fut = ex.submit(_process_monthly_groups, groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers), time_start, time_end, ignoring_stocks_csv)
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
        for yyyymm in months:
            db_path = _monthly_db_path(output_db, yyyymm)
            groups = monthly_groups[yyyymm]
            processed = _process_monthly_groups(groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers), time_start, time_end, ignoring_stocks_csv)
            print(f"월 처리 완료: {yyyymm} ({processed}/{len(groups)}) -> {db_path}")

    # 처리된 월들에 대해 병렬 최종 CHECKPOINT 수행
    try:
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

    print(f"정규화 완료: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="DuckDB 데이터셋 정규화 스크립트")
    parser.add_argument("input_db", help="입력 DuckDB 파일 경로")
    parser.add_argument("-o", "--output", default="normalized.duckdb", help="출력 DuckDB 파일명")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="이미 DB에 해당 (종목코드, 날짜) 그룹이 존재하면 스킵합니다 (기본: 활성화)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="이미 존재하는 그룹도 다시 처리합니다")
    parser.add_argument("--compact-only", action="store_true",
                        help="입력 DB 처리 없이 지정한 DuckDB에 대해 PRAGMA optimize/checkpoint만 수행합니다")
    parser.add_argument("--force-recreate", action="store_true",
                        help="출력 DuckDB 파일이 존재하면 삭제 후 새로 생성합니다 (손상/버전 문제 해결용)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="월별 병렬 처리에 사용할 프로세스 수 (기본: CPU 코어 수)")
    parser.add_argument("--group-workers", type=int, default=1,
                        help="각 월 내에서 그룹 병렬 처리에 사용할 프로세스 수 (기본: 1)")
    parser.add_argument("--tmp-dir", default=None,
                        help="임시 결과 저장 디렉토리 (기본: <output>.tmp)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="몇 개 그룹 처리마다 DuckDB CHECKPOINT를 실행할지 지정 (0이면 비활성화, 기본: 100)")
    parser.add_argument("--time-start", type=int, default=90000000,
                        help="시작 시간 (기본: 90000000 = 오전 9시)")
    parser.add_argument("--time-end", type=int, default=110000000,
                        help="종료 시간, 미포함 (기본: 110000000 = 오전 11시)")
    parser.add_argument("--ignoring-stocks-csv", default=os.path.join("scripts", "ignoring_stocks.csv"),
                        help="무시할 종목명 리스트 CSV 경로 (기본: scripts/ignoring_stocks.csv, '종목명' 컬럼 필요)")
     
    args = parser.parse_args()
     
    if not args.compact_only:
        if not os.path.exists(args.input_db):
            print(f"입력 DuckDB 파일이 존재하지 않습니다: {args.input_db}")
            return
     
    normalize_datasets(
        args.input_db,
        args.output,
        skip_existing=args.skip_existing,
        compact_only=args.compact_only,
        force_recreate=args.force_recreate,
        workers=args.workers,
        group_workers=args.group_workers,
        tmp_dir=args.tmp_dir,
        checkpoint_interval=args.checkpoint_interval,
        time_start=args.time_start,
        time_end=args.time_end,
        ignoring_stocks_csv=args.ignoring_stocks_csv,
    )


if __name__ == "__main__":
    main()
