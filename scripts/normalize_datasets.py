import argparse
import os
import re
import duckdb
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


def ensure_datasets_table_duckdb(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    table = "datasets"
    # Create table if not exists
    try:
        conn.execute(f"DESCRIBE {table}")
        exists = True
    except Exception:
        exists = False
    if not exists:
        conn.register("_schema_df", df.head(0))
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _schema_df")
        conn.unregister("_schema_df")
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
                elif pd.api.types.is_integer_dtype(series):
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


def normalize_datasets(input_folder: str, output_db: str, *,
                       skip_existing: bool = True,
                       compact_only: bool = False):
    """
    메인 정규화 함수 (DuckDB 전용)
    """
    # compact-only 모드: CSV 처리 없이 DB 유지보수만 수행
    if compact_only:
        print("compact-only 모드: CSV 처리 없이 DB 최적화만 수행합니다.")
        conn = duckdb.connect(output_db)
        try:
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
        print(f"DB 유지보수 완료: {output_db}")
        return

    print(f"CSV 파일 검색 중: {input_folder}")
    csv_groups = find_csv_files(input_folder)
    
    if not csv_groups:
        print("처리할 CSV 파일 그룹을 찾을 수 없습니다.")
        return
    
    print(f"발견된 그룹 수: {len(csv_groups)}")
    
    # DuckDB 연결
    conn = duckdb.connect(output_db)
    
    try:
        for group_key, files in csv_groups.items():
            print(f"처리 중: {group_key}")
            
            # 그룹 키에서 정보 추출
            parts = group_key.split('_')
            code = parts[0]
            date = parts[-1]
            name = '_'.join(parts[1:-1])
            
            # 이미 존재하는 그룹은 스킵 (옵션)
            if skip_existing:
                try:
                    conn.execute("DESCRIBE datasets")
                    exists = conn.execute(
                        "SELECT 1 FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=? LIMIT 1",
                        [code, date]
                    ).fetchone()
                    if exists:
                        print(f"스킵(이미 존재): {group_key}")
                        continue
                except Exception:
                    # 테이블이 없으면 계속 진행하여 생성
                    pass
            
            # CSV 파일들 merge
            merged_df = merge_csv_files(files, code, name)
            # 피처 정규화 적용
            merged_df = apply_feature_normalization(merged_df)
            
            # 날짜 컬럼 추가 (그룹 키의 날짜 사용) - 사본을 만들어 단편화 방지
            merged_df = merged_df.copy()
            merged_df['날짜'] = date
            merged_df = pd.concat([merged_df['날짜'], merged_df.drop(columns=['날짜'])], axis=1)

            # DuckDB: 테이블 보장 후, 그룹 단위 delete-then-insert
            ensure_datasets_table_duckdb(conn, merged_df)
            conn.register("_batch_df", merged_df)
            try:
                conn.execute("DELETE FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
            except Exception:
                pass
            conn.execute("INSERT INTO datasets SELECT * FROM _batch_df")
            conn.unregister("_batch_df")
            
            print(f"완료: {len(merged_df)} 행")
    
    finally:
        try:
            # DuckDB maintenance
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
    
    print(f"정규화 완료: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="CSV 데이터셋 정규화 스크립트 (DuckDB)")
    parser.add_argument("input_folder", help="입력 CSV 폴더 경로")
    parser.add_argument("-o", "--output", default="normalized.duckdb", help="출력 DuckDB 파일명")
    # Skip-existing 옵션 (기본 활성화). 비활성화하려면 --no-skip-existing 사용
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="이미 DB에 해당 (종목코드, 날짜) 그룹이 존재하면 스킵합니다 (기본: 활성화)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="이미 존재하는 그룹도 다시 처리합니다")
    # Compact-only 모드: CSV를 읽지 않고 지정한 DB에 대해 최적화만 수행
    parser.add_argument("--compact-only", action="store_true",
                        help="CSV 처리 없이 지정한 DuckDB에 대해 PRAGMA optimize/checkpoint만 수행합니다")
    
    args = parser.parse_args()
    
    # compact-only인 경우 입력 폴더 존재 여부는 체크하지 않음
    if not args.compact_only:
        if not os.path.exists(args.input_folder):
            print(f"입력 폴더가 존재하지 않습니다: {args.input_folder}")
            return
    
    normalize_datasets(
        args.input_folder,
        args.output,
        skip_existing=args.skip_existing,
        compact_only=args.compact_only,
    )


if __name__ == "__main__":
    main()
