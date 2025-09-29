#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import glob
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


def _table_exists(conn: duckdb.DuckDBPyConnection, db: str, table: str) -> bool:
    try:
        conn.execute(f"DESCRIBE {db}.{table}")
        return True
    except Exception:
        return False


def _read_schema(conn: duckdb.DuckDBPyConnection, db: str, table: str) -> List[Tuple[str, str]]:
    # Returns list of (name, type)
    rows = conn.execute(f"PRAGMA {db}.table_info('{table}')").fetchall()
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


def _build_union_schema(inputs: List[str], table: str) -> List[Tuple[str, str]]:
    # Attach each input transiently to read schema
    conn = duckdb.connect(":memory:")
    try:
        cols_types: Dict[str, Set[str]] = {}
        for idx, path in enumerate(inputs, 1):
            alias = f"db{idx}"
            try:
                conn.execute(f"ATTACH '{path}' AS {alias} (READ_ONLY)")
                if not _table_exists(conn, alias, table):
                    continue
                for name, ctype in _read_schema(conn, alias, table):
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
                       threads: int = max(1, os.cpu_count() or 1)) -> None:
    # Compute union schema first
    union_schema = _build_union_schema(inputs, table)
    if not union_schema:
        raise SystemExit(f"No '{table}' table found in provided inputs.")

    # Prepare output directory
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(output)
    try:
        # Pragmas for performance
        conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
        conn.execute(f"PRAGMA threads={int(threads)}")
        conn.execute("PRAGMA temp_directory='' ")  # use OS temp

        _ensure_output_table(conn, table, union_schema)

        # Insert from each input
        for idx, path in enumerate(inputs, 1):
            alias = f"src{idx}"
            print(f"[{idx}/{len(inputs)}] 병합 중: {os.path.basename(path)}")
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
            finally:
                try:
                    conn.execute(f"DETACH {alias}")
                except Exception:
                    pass
        # Final checkpoint
        conn.execute("CHECKPOINT")
        print(f"완료: {len(inputs)}개 DuckDB 파일 병합 -> {output}")
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
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    files = _list_duckdb_files(args.inputs, args.pattern, recursive=not args.no_recursive)
    if not files:
        print("입력 DuckDB 파일을 찾지 못했습니다.")
        return 2

    dedup_keys = [k.strip() for k in args.dedup_keys.split(',') if k.strip()] if args.dedup_keys else None

    try:
        merge_duckdb_files(files, args.out, table=args.table, dedup_keys=dedup_keys,
                           memory_limit=args.memory_limit, threads=args.threads)
        return 0
    except SystemExit as se:
        print(str(se))
        return 2
    except Exception as e:
        print(f"오류: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
