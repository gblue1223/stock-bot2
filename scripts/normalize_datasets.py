import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

    stdonly_cols = set(["매도호가총잔량직전대비", "매수호가총잔량직전대비"])

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


def create_sqlite_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame):
    """
    SQLite 테이블 생성
    - 단일 테이블 사용을 가정하고, 존재하지 않으면 생성합니다.
    - 복합 기본키: (날짜, 종목코드, 번호)
    """
    # 컬럼 정의 생성
    col_definitions = []

    # 스키마 기준 타입 결정
    for col in df.columns:
        if col == '번호':
            col_definitions.append(f'"{col}" INTEGER')
        elif col in TEXT_COLUMNS or col == '날짜':
            col_definitions.append(f'"{col}" TEXT')
        else:
            col_definitions.append(f'"{col}" REAL')

    # 복합 PK 추가
    col_definitions.append('PRIMARY KEY("날짜", "종목코드", "번호")')

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_definitions)}) WITHOUT ROWID"
    conn.execute(create_sql)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    row = cur.fetchone()
    return row is not None


def _get_pk_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cols = []
    for cid, name, ctype, notnull, dflt, pk in conn.execute(f"PRAGMA table_info('{table_name}')"):
        if pk:
            cols.append((pk, name))
    # pk value indicates order (1..N)
    cols_sorted = [name for _, name in sorted(cols, key=lambda x: x[0])]
    return cols_sorted


def _recreate_table_with_pk(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame):
    temp_name = f"{table_name}__new"
    # 1) Create temp with desired schema
    create_sqlite_table(conn, temp_name, df)

    # 2) If old exists, migrate data (column intersection)
    if _table_exists(conn, table_name):
        # existing columns
        existing_cols = [row[1] for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        new_cols = list(df.columns)
        common = [c for c in new_cols if c in existing_cols]
        if common:
            cols_list = ", ".join([f'"{c}"' for c in common])
            conn.execute(f"INSERT OR IGNORE INTO {temp_name} ({cols_list}) SELECT {cols_list} FROM {table_name}")

        # drop old and rename
        conn.execute(f"DROP TABLE {table_name}")
    # 3) rename new -> final
    conn.execute(f"ALTER TABLE {temp_name} RENAME TO {table_name}")


def ensure_datasets_table(conn: sqlite3.Connection, df: pd.DataFrame):
    table_name = "datasets"
    if not _table_exists(conn, table_name):
        create_sqlite_table(conn, table_name, df)
        return
    # Check PK
    pk_cols = _get_pk_columns(conn, table_name)
    desired = ["날짜", "종목코드", "번호"]
    if pk_cols != desired:
        print("기존 'datasets' 테이블이 원하는 기본키와 다릅니다. 테이블을 마이그레이션합니다...")
        _recreate_table_with_pk(conn, table_name, df)
    # 마이그레이션 후에도 컬럼 보강 절차 진행

    # Ensure all needed columns exist (추가 생성)
    existing_cols = [row[1] for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
    for col in df.columns:
        if col in existing_cols:
            continue
        # 타입 추론: 번호 -> INTEGER, 텍스트/날짜 -> TEXT, 그 외 REAL
        if col == '번호':
            col_type = 'INTEGER'
        elif col in TEXT_COLUMNS or col == '날짜':
            col_type = 'TEXT'
        else:
            col_type = 'REAL'
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN \"{col}\" {col_type}")

    # Optional: helpful index for queries by code/date
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_code_date ON datasets(\"종목코드\", \"날짜\")")


def insert_data_to_sqlite(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame):
    """
    데이터를 SQLite에 삽입 (중복 시 갱신: INSERT OR REPLACE)
    """
    # 필수 컬럼 체크
    required_cols = {'날짜', '종목코드', '번호'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 작은 청크로 나눠 upsert (큰 청크가 더 빠름)
    chunk_size = 1000
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join([f'"{c}"' for c in cols])
    sql = f"INSERT OR REPLACE INTO {table_name} ({col_list}) VALUES ({placeholders})"

    # 단일 트랜잭션으로 일괄 처리 (큰 성능 향상)
    try:
        conn.execute("BEGIN")
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            conn.executemany(sql, [tuple(row) for row in chunk.itertuples(index=False, name=None)])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _apply_sqlite_pragmas(conn: sqlite3.Connection, *, page_size: int | None, journal_mode: str, synchronous: str, auto_vacuum: str):
    """
    Apply space/IO related PRAGMAs. These are built-in and do not require external modules.
    Note: page_size and auto_vacuum must be set before creating tables to fully take effect.
    """
    cur = conn.cursor()
    if page_size:
        cur.execute(f"PRAGMA page_size={int(page_size)}")
    if auto_vacuum.lower() in {"none", "full", "incremental"}:
        # none(0), full(1), incremental(2)
        mapping = {"none": 0, "full": 1, "incremental": 2}
        cur.execute(f"PRAGMA auto_vacuum={mapping[auto_vacuum.lower()]}")
    # journal_mode: DELETE|TRUNCATE|PERSIST|MEMORY|WAL|OFF
    cur.execute(f"PRAGMA journal_mode={journal_mode}")
    # synchronous: OFF(0)|NORMAL(1)|FULL(2)|EXTRA(3)
    cur.execute(f"PRAGMA synchronous={synchronous}")
    # 추가 성능 튜닝 (안전 범위 내)
    cur.execute("PRAGMA temp_store=MEMORY")
    # 음수면 KB 단위의 페이지 수를 의미 (예: -200000 ~= 200MB)
    cur.execute("PRAGMA cache_size=-200000")
    # 메모리 매핑 (일부 빌드에서만 영향)
    cur.execute("PRAGMA mmap_size=134217728")  # 128MB
    cur.close()


def normalize_datasets(input_folder: str, output_db: str, *,
                       compact: bool = False,
                       page_size: int | None = None,
                       journal_mode: str = "WAL",
                       synchronous: str = "NORMAL",
                       auto_vacuum: str = "full",
                       vacuum_into: str | None = None):
    """
    메인 정규화 함수
    """
    print(f"CSV 파일 검색 중: {input_folder}")
    csv_groups = find_csv_files(input_folder)
    
    if not csv_groups:
        print("처리할 CSV 파일 그룹을 찾을 수 없습니다.")
        return
    
    print(f"발견된 그룹 수: {len(csv_groups)}")
    
    # SQLite 연결
    conn = sqlite3.connect(output_db)
    
    # Apply PRAGMAs before creating any table for best effect
    _apply_sqlite_pragmas(conn,
                          page_size=page_size,
                          journal_mode=journal_mode,
                          synchronous=synchronous,
                          auto_vacuum=auto_vacuum)
    
    try:
        for group_key, files in csv_groups.items():
            print(f"처리 중: {group_key}")
            
            # 그룹 키에서 정보 추출
            parts = group_key.split('_')
            code = parts[0]
            date = parts[-1]
            name = '_'.join(parts[1:-1])
            
            # CSV 파일들 merge
            merged_df = merge_csv_files(files, code, name)
            # 피처 정규화 적용
            merged_df = apply_feature_normalization(merged_df)
            
            # 날짜 컬럼 추가 (그룹 키의 날짜 사용) - 사본을 만들어 단편화 방지
            merged_df = merged_df.copy()
            merged_df['날짜'] = date
            merged_df = pd.concat([merged_df['날짜'], merged_df.drop(columns=['날짜'])], axis=1)

            # SQLite 테이블 생성/보장 (+ 필요 시 PK 스키마로 마이그레이션)
            ensure_datasets_table(conn, merged_df)

            # 데이터 삽입 (upsert)
            insert_data_to_sqlite(conn, "datasets", merged_df)
            
            print(f"완료: {len(merged_df)} 행")
    
    finally:
        try:
            # Analyze and optimize sqlite internal stats
            conn.execute("ANALYZE")
            conn.execute("PRAGMA optimize")
            # VACUUM compacts the database; use when --compact is requested
            if vacuum_into:
                # If supported by this SQLite build, write compacted copy
                conn.execute(f"VACUUM INTO '{vacuum_into}'")
            elif compact:
                conn.execute("VACUUM")
        finally:
            conn.close()
    
    print(f"정규화 완료: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="CSV 데이터셋 정규화 스크립트")
    parser.add_argument("input_folder", help="입력 CSV 폴더 경로")
    parser.add_argument("-o", "--output", default="normalized.db", help="출력 SQLite DB 파일명")
    # Built-in SQLite space/IO tuning (no external compression modules)
    parser.add_argument("--compact", action="store_true", help="마지막에 VACUUM 실행으로 DB를 컴팩트하게 만듭니다")
    parser.add_argument("--page-size", type=int, default=4096, help="페이지 크기 (바이트). 테이블 생성 전 설정 권장. 예: 4096, 8192, 16384")
    parser.add_argument("--journal-mode", choices=["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"], default="WAL",
                        help="저널 모드 설정")
    parser.add_argument("--synchronous", choices=["OFF", "NORMAL", "FULL", "EXTRA"], default="NORMAL", help="동기화 수준 설정")
    parser.add_argument("--auto-vacuum", choices=["none", "full", "incremental"], default="full",
                        help="자동 VACUUM 모드 설정")
    parser.add_argument("--vacuum-into", default=None,
                        help="지원 시 VACUUM INTO 경로로 압축/컴팩트된 복사본을 생성합니다 (예: output_compact.db)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"입력 폴더가 존재하지 않습니다: {args.input_folder}")
        return
    
    normalize_datasets(
        args.input_folder,
        args.output,
        compact=args.compact,
        page_size=args.page_size,
        journal_mode=args.journal_mode,
        synchronous=args.synchronous,
        auto_vacuum=args.auto_vacuum,
        vacuum_into=args.vacuum_into,
    )


# python scripts/normalize_datasets.py models/test_datasets -o models/test_datasets.db
# python scripts/normalize_datasets.py "D:\Workspace\Project\stock-bot\hoga-crawler\data" -o models/datasets.db
if __name__ == "__main__":
    main()
