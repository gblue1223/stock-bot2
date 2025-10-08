import duckdb
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


KEY_COLS = ["날짜", "종목코드", "번호"]


def get_table_schema(conn, table: str) -> List[Tuple[int, str, str, int, Optional[str], int]]:
    """DuckDB table schema via PRAGMA table_info.
    Returns tuples approx: (column_id, column_name, data_type, null, default, primary_key)
    """
    return list(conn.execute(f"PRAGMA table_info('{table}')").fetchall())


def list_tables(conn) -> List[str]:
    # SHOW TABLES returns rows with first column = table name
    return [r[0] for r in conn.execute("SHOW TABLES").fetchall()]


def table_exists(conn, table: str) -> bool:
    try:
        conn.execute(f"DESCRIBE {table}")
        return True
    except Exception:
        return False


def get_real_columns(conn, table: str = "datasets") -> List[str]:
    cols: List[str] = []
    for cid, name, ctype, is_null, dflt, pk in get_table_schema(conn, table):
        t = (ctype or "").upper()
        # Consider common floating-point types in DuckDB
        if any(tok in t for tok in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
            cols.append(name)
    return cols


def get_integer_columns(conn, table: str = "datasets") -> List[str]:
    cols: List[str] = []
    for cid, name, ctype, is_null, dflt, pk in get_table_schema(conn, table):
        t = (ctype or "").upper()
        if "INT" in t:
            cols.append(name)
    return cols


def load_real_dataframe(
    db_path: str,
    table: str = "datasets",
    code: Optional[str] = None,
    date: Optional[str] = None,
    limit: Optional[int] = None,
    order_asc: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load only REAL-typed (or numeric-like) columns from DuckDB. Always includes keys if available for ordering and filtering.
    Returns (df, real_cols_in_df).
    """
    conn = duckdb.connect(db_path)
    try:
        if not table_exists(conn, table):
            available = list_tables(conn)
            raise ValueError(
                f"Table '{table}' not found in DB '{db_path}'. Available tables: {available}"
            )

        real_cols = get_real_columns(conn, table)
        # Ensure we can order by 번호 if integer
        integer_cols = get_integer_columns(conn, table)
        order_col = "번호" if "번호" in integer_cols else None
        select_cols = [*(c for c in KEY_COLS if c in integer_cols or c in real_cols), *real_cols]
        # de-dup while preserving order
        seen = set()
        select_cols = [c for c in select_cols if not (c in seen or seen.add(c))]

        where: List[str] = []
        params: List[str] = []
        if code:
            where.append('"종목코드" = ?')
            params.append(code)
        if date:
            where.append('"날짜" = ?')
            params.append(date)

        if select_cols:
            col_sql = ", ".join([f'"{c}"' for c in select_cols])
            sql = f"SELECT {col_sql} FROM {table}"
        else:
            # Fallback: select all, we'll infer numeric columns later
            sql = f"SELECT * FROM {table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if order_col:
            sql += f" ORDER BY \"{order_col}\" {'ASC' if order_asc else 'DESC'}"
        if limit:
            sql += f" LIMIT {int(limit)}"

        df = conn.execute(sql, params).df()

        # Determine usable REAL/numeric columns present in the DataFrame
        if select_cols:
            real_in_df = [c for c in real_cols if c in df.columns]
        else:
            # Infer numerics from dtypes if schema/types were unavailable
            real_in_df = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in KEY_COLS]

        # Only include existing key columns to avoid KeyError
        existing_keys = [k for k in KEY_COLS if k in df.columns]
        if existing_keys:
            return df[[*existing_keys, *real_in_df]], real_in_df
        else:
            return df[real_in_df], real_in_df
    finally:
        conn.close()


@dataclass
class SequenceData:
    X: np.ndarray  # [N, T, F]
    y: np.ndarray  # [N] or [N, D]
    feature_names: List[str]


def make_sequences(
    df: pd.DataFrame,
    real_cols: List[str],
    seq_len: int = 60,
    horizon: int = 1,
    target_col: Optional[str] = None,
    dropna: bool = True,
) -> SequenceData:
    """
    Create sliding window sequences from REAL columns ordered by '번호' if present.
    - target_col: if None, use the first column (e.g., close price-like). Target is value at t+horizon for that column.
    - Returns numpy arrays.
    """
    if target_col is None:
        if not real_cols:
            raise ValueError("No REAL columns available for target selection")
        target_col = real_cols[0]
    if target_col not in real_cols:
        raise ValueError(f"target_col '{target_col}' not in REAL columns")

    feats = df[real_cols].copy()
    if dropna:
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_list = []
    y_list = []
    T = len(feats)
    target_vals = feats[target_col].to_numpy().astype(np.float32)
    F = feats.shape[1]
    feat_vals = feats.to_numpy(dtype=np.float32)

    for i in range(T - seq_len - horizon + 1):
        X_list.append(feat_vals[i : i + seq_len])
        y_list.append(target_vals[i + seq_len + horizon - 1])

    if not X_list:
        raise ValueError("Not enough rows to create sequences. Reduce seq_len/horizon or load more data.")

    X = np.stack(X_list, axis=0)  # [N, T, F]
    y = np.array(y_list, dtype=np.float32)
    return SequenceData(X=X, y=y, feature_names=real_cols)


def train_val_split(seq: SequenceData, val_ratio: float = 0.2) -> Tuple[SequenceData, SequenceData]:
    n = len(seq.X)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    return (
        SequenceData(seq.X[:n_train], seq.y[:n_train], seq.feature_names),
        SequenceData(seq.X[n_train:], seq.y[n_train:], seq.feature_names),
    )
