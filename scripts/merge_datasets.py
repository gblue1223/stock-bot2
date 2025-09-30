#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set

import duckdb

# Known text-like columns from normalize_datasets.py (subset sufficient for type decisions)
TEXT_COLUMNS: Set[str] = {
    "종목코드", "종목명", "시간",
    *{f"매도거래원{i}" for i in range(1, 6)},
    *{f"매수거래원{i}" for i in range(1, 6)},
}

DEFAULT_TABLE = "datasets"


def _list_duckdb_files(paths: List[str], pattern: str, recursive: bool) -> List[str]:
    files: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            if recursive:
                files.extend([str(x) for x in pp.rglob(pattern)])
            else:
                files.extend([str(x) for x in pp.glob(pattern)])
        else:
            # allow direct file match and glob
            if any(ch in p for ch in "*?[]"):
                files.extend(glob.glob(p))
            elif pp.suffix.lower() in {".duckdb", ".db"} and pp.exists():
                files.append(str(pp))
    # unique, preserve order
    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            out.append(f)
    return out


def _extract_date_key(path: str) -> int:
    """Extract YYYYMMDD from filename; returns integer for sorting, fallback to 0 if not found."""
    name = os.path.basename(path)
    m = re.search(r"(20\d{6}|19\d{6})", name)  # greedy first YYYYMMDD
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _sort_files(files: List[str], sort_by: str, reverse: bool) -> List[str]:
    if sort_by == "name":
        return sorted(files, key=lambda p: os.path.basename(p), reverse=reverse)
    if sort_by == "mtime":
        return sorted(files, key=lambda p: os.path.getmtime(p), reverse=reverse)
    if sort_by == "date":
        return sorted(files, key=_extract_date_key, reverse=reverse)
    return files


def _table_exists(conn: duckdb.DuckDBPyConnection, db: str, table: str) -> bool:
    try:
        conn.execute(f"DESCRIBE {db}.{table}")
        return True
    except Exception:
        return False


def _read_schema(conn: duckdb.DuckDBPyConnection, db: str, table: str) -> List[Tuple[str, str]]:
    # Use qualified table name inside the pragma argument instead of qualifying the pragma itself
    rows = conn.execute(f"PRAGMA table_info('{db}.{table}')").fetchall()
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return [(r[1], (r[2] or '').upper()) for r in rows]


def _choose_union_type(col: str, types: Set[str]) -> str:
    # Force VARCHAR for known text-like columns
    if col in TEXT_COLUMNS or col in {"날짜"}:
        return "VARCHAR"
    # Prefer wider/more general types
    normalized = {t.upper() for t in types if t}
    if col == "번호":
        return "BIGINT"
    if any("CHAR" in t or "STRING" in t or "VARCHAR" in t for t in normalized):
        return "VARCHAR"
    if any(t in {"DOUBLE", "FLOAT", "REAL"} for t in normalized):
        return "DOUBLE"
    if any("DECIMAL" in t for t in normalized):
        return "DOUBLE"  # simplify
    # default integer
    return "BIGINT"


def _build_union_schema(inputs: List[str], table: str, strict: bool = False) -> List[Tuple[str, str]]:
    # Attach each input transiently to read schema
    conn = duckdb.connect(":memory:")
    try:
        cols_types: Dict[str, Set[str]] = {}
        first_schema: List[Tuple[str, str]] | None = None
        first_db: str | None = None
        for idx, path in enumerate(inputs, 1):
            alias = f"db{idx}"
            try:
                conn.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
                if not _table_exists(conn, alias, table):
                    continue
                schema = _read_schema(conn, alias, table)
                if strict:
                    if first_schema is None:
                        first_schema = schema
                        first_db = path
                    else:
                        if schema != first_schema:
                            # compute deltas
                            def to_map(s):
                                return {c: t for c, t in s}
                            a = to_map(first_schema)
                            b = to_map(schema)
                            missing = [c for c in a.keys() if c not in b]
                            extra = [c for c in b.keys() if c not in a]
                            type_diff = [c for c in a.keys() & b.keys() if (a[c] or '').upper() != (b[c] or '').upper()]
                            raise SystemExit(
                                "스키마 불일치로 종료:\n"
                                f"  기준: {os.path.basename(first_db or '')}\n"
                                f"  문제 파일: {os.path.basename(path)}\n"
                                f"  누락 컬럼: {missing}\n"
                                f"  추가 컬럼: {extra}\n"
                                f"  타입 상이: {[(c, a[c], b[c]) for c in type_diff]}"
                            )
                for name, ctype in schema:
                    cols_types.setdefault(name, set()).add(ctype or "")
            finally:
                try:
                    conn.execute(f"DETACH {alias}")
                except Exception:
                    pass
        if not cols_types:
            return []
        # Provide a stable column order: prioritize 날짜, 번호, then the rest alphabetically
        ordered_cols = []
        for special in ["날짜", "번호"]:
            if special in cols_types:
                ordered_cols.append(special)
        for c in sorted(k for k in cols_types.keys() if k not in {"날짜", "번호"}):
            ordered_cols.append(c)
        if strict and first_schema is not None:
            # keep the exact order/types from the first schema
            return first_schema
        return [(c, _choose_union_type(c, cols_types[c])) for c in ordered_cols]
    finally:
        conn.close()


def _ensure_output_table(conn: duckdb.DuckDBPyConnection, table: str, schema: List[Tuple[str, str]]):
    if not schema:
        raise RuntimeError("No union schema computed; ensure inputs contain the table to merge.")
    # Create if not exists
    try:
        conn.execute(f"DESCRIBE {table}")
        exists = True
    except Exception:
        exists = False
    if not exists:
        coldefs = ", ".join([f'"{c}" {t}' for c, t in schema])
        conn.execute(f"CREATE TABLE {table} ({coldefs})")
    else:
        # Add missing columns and adjust types for known text-like columns
        info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        existing_types = {r[1]: (r[2] or '').upper() for r in info}
        existing_cols = set(existing_types.keys())
        # Add missing
        for c, t in schema:
            if c not in existing_cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN \"{c}\" {t}")
        # Relax type to VARCHAR if needed for text-like columns
        for c, t in schema:
            if c in TEXT_COLUMNS | {"날짜"}:
                et = existing_types.get(c, t)
                if "CHAR" not in et and "STRING" not in et and "VARCHAR" not in et:
                    conn.execute(f"ALTER TABLE {table} ALTER COLUMN \"{c}\" TYPE VARCHAR")


def _delete_existing_by_keys(conn: duckdb.DuckDBPyConnection, table: str, src_alias: str, keys: List[str]):
    # Build a DELETE using a semi-join on the distinct key set from source
    key_expr = ", ".join([f'\"{k}\"' for k in keys])
    conn.execute(
        f"""
        DELETE FROM {table} t
        WHERE ({key_expr}) IN (SELECT DISTINCT {key_expr} FROM {src_alias}.{table})
        """
    )


def merge_duckdb_files(inputs: List[str], output: str, table: str = DEFAULT_TABLE,
                       dedup_keys: List[str] | None = None, memory_limit: str = "4GB",
                       threads: int = max(1, os.cpu_count() or 1), strict_schema: bool = True,
                       sort_by: str = "name", reverse_sort: bool = False, 
                       temp_directory: str | None = None) -> None:
    if temp_directory is None:
        temp_directory = tempfile.gettempdir()
    # Compute union or strict schema first
    union_schema = _build_union_schema(inputs, table, strict=strict_schema)
    if not union_schema:
        raise SystemExit(f"No '{table}' table found in provided inputs.")

    # Prepare output directory
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Sort files for deterministic processing
    inputs_sorted = _sort_files(inputs, sort_by=sort_by, reverse=reverse_sort)

    conn = duckdb.connect(output)
    try:
        # Pragmas for performance
        conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
        conn.execute(f"PRAGMA threads={int(threads)}")

        # Pragmas for performance
        conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
        conn.execute(f"PRAGMA threads={int(threads)}")
        conn.execute(f"SET temp_directory='{tempfile.gettempdir()}'")

        _ensure_output_table(conn, table, union_schema)

        # Pre-compute total rows for progress
        total_rows = 0
        row_counts: List[int] = []
        for idx, path in enumerate(inputs_sorted, 1):
            alias = f"cnt{idx}"
            try:
                conn.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
                if _table_exists(conn, alias, table):
                    cnt = conn.execute(f"SELECT COUNT(*) FROM {alias}.{table}").fetchone()[0]
                else:
                    cnt = 0
                row_counts.append(int(cnt))
                total_rows += int(cnt)
            finally:
                try:
                    conn.execute(f"DETACH {alias}")
                except Exception:
                    pass

        inserted_so_far = 0

        # Insert from each input with detailed logs
        n = len(inputs_sorted)
        for idx, path in enumerate(inputs_sorted, 1):
            alias = f"src{idx}"
            file_rows = row_counts[idx - 1] if idx - 1 < len(row_counts) else 0
            pct = (idx - 1) / n * 100.0
            print(f"[{idx}/{n}] ({pct:5.1f}%) 병합 준비: {os.path.basename(path)} | 행수≈{file_rows}")
            try:
                conn.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
                # Skip if table missing
                if not _table_exists(conn, alias, table):
                    print(f"  경고: 테이블 '{table}' 미존재 -> 건너뜀")
                    continue
                # Optional: delete existing rows that would duplicate by dedup keys (keep latest file wins)
                if dedup_keys:
                    _delete_existing_by_keys(conn, table, alias, dedup_keys)
                # Insert by name; requires all columns exist in target
                conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM {alias}.{table}")
                inserted_so_far += file_rows
                pct_after = idx / n * 100.0
                global_pct_rows = (inserted_so_far / total_rows * 100.0) if total_rows > 0 else 0.0
                print(
                    f"  완료: [{idx}/{n}] ({pct_after:5.1f}%) {os.path.basename(path)} -> 누적 {inserted_so_far}/{total_rows}행 ({global_pct_rows:5.1f}%)"
                )
            finally:
                try:
                    conn.execute(f"DETACH {alias}")
                except Exception:
                    pass
        # Create index after merge
        try:
            print("인덱스 생성중: idx_datasets_code_date (종목코드, 날짜)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_code_date ON datasets(\"종목코드\", \"날짜\")")
            print("인덱스 생성 완료: idx_datasets_code_date (종목코드, 날짜)")
        except Exception as e:
            print(f"경고: 인덱스 생성 실패: {type(e).__name__}: {e}")
        # Final checkpoint
        conn.execute("CHECKPOINT")
        print(f"완료: {len(inputs_sorted)}개 DuckDB 파일 병합 -> {output}")
    finally:
        conn.close()


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple DuckDB files into one, unifying schema for a target table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Input files or directories (globs allowed for files)")
    p.add_argument("--pattern", default="*.duckdb", help="File pattern when scanning directories")
    p.add_argument("--no-recursive", action="store_true", help="Do not search directories recursively")
    p.add_argument("--table", default=DEFAULT_TABLE, help="Table name to merge")
    p.add_argument("--out", required=True, help="Output DuckDB file path")
    p.add_argument("--dedup-keys", default="", help="Comma-separated column names to deduplicate on (keep latest file)")
    p.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit (e.g., 4GB, 1TB)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1)), help="DuckDB threads")
    p.add_argument("--sort-by", choices=["name", "mtime", "date"], default="name", help="Sort input files by")
    p.add_argument("--reverse", action="store_true", help="Reverse sort order")
    p.add_argument("--temp-dir", default=None, help="Temporary directory for DuckDB spill-to-disk (default: system temp)")
    
    strict_group = p.add_mutually_exclusive_group()
    strict_group.add_argument("--strict-schema", dest="strict_schema", action="store_true", help="Fail if any input schema differs (default)")
    strict_group.add_argument("--no-strict-schema", dest="strict_schema", action="store_false", help="Allow union of differing schemas")
    p.set_defaults(strict_schema=True)
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    files = _list_duckdb_files(args.inputs, args.pattern, recursive=not args.no_recursive)
    if not files:
        print("입력 DuckDB 파일을 찾지 못했습니다.")
        return 2

    # Apply sorting per user options
    files = _sort_files(files, sort_by=args.sort_by, reverse=args.reverse)

    dedup_keys = [k.strip() for k in args.dedup_keys.split(',') if k.strip()] if args.dedup_keys else None

    try:
        merge_duckdb_files(
            files,
            args.out,
            table=args.table,
            dedup_keys=dedup_keys,
            memory_limit=args.memory_limit,
            threads=args.threads,
            strict_schema=args.strict_schema,
            sort_by=args.sort_by,
            reverse_sort=args.reverse,
            temp_directory=args.temp_dir,
        )
        return 0
    except SystemExit as se:
        print(str(se))
        return 2
    except Exception as e:
        print(f"오류: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
