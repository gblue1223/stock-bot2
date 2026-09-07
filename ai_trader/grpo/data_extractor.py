#!/usr/bin/env python3
"""Extract causal raw episodes with a versioned feature/execution contract."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.market_data import (canonical_feature_columns, validate_feature_columns,
                             chronological_order, execution_arrays, PRICE_SCALES)
from lib.normalization import compute_stock_name_scalar_batch, compute_time_features_batch

logger = logging.getLogger("DataExtractor")


def _identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def get_feature_columns(conn, table_name: str, expected_features: int) -> list:
    """Select a registered schema by name; database order cannot change semantics."""
    schema = conn.execute(f"DESCRIBE {_identifier(table_name)}").fetchdf()
    available = dict(zip(schema.column_name, schema.column_type))
    requested = canonical_feature_columns(expected_features)
    derived = {"종목명_scalar", "시간_sin", "시간_cos", "시간_scalar"}
    missing = set(requested) - set(available) - derived
    if missing:
        raise ValueError(f"Missing required model features: {sorted(missing)}")
    for name in set(requested) - derived:
        if not any(t in available[name].upper() for t in ("DOUBLE", "FLOAT", "REAL", "INT", "DECIMAL")):
            raise ValueError(f"Feature {name} must be numeric, found {available[name]}")
    return validate_feature_columns(requested, expected_features)


def extract_data(db_path: str, table_name: str, output_dir: str, seq_len: int,
                 features: int, max_steps: int, limit: int = None, price_unit: str = "krw"):
    """Write a new extraction. Existing manifests are protected from overwrites."""
    if price_unit not in PRICE_SCALES:
        raise ValueError(f"Unknown price unit: {price_unit}")
    db_path = str(Path(db_path).resolve())
    output = Path(output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError("Output manifest already exists; use a new directory for regeneration")
    if not Path(db_path).is_file():
        raise FileNotFoundError(db_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    rejected = {}
    conn = duckdb.connect(db_path, read_only=True)
    try:
        feature_cols = get_feature_columns(conn, table_name, features)
        schema = conn.execute(f"DESCRIBE {_identifier(table_name)}").fetchdf().column_name.tolist()
        # Eligibility uses only enough history plus a decision point. Do not
        # remove an entire day based on its later return or maximum volatility.
        keys = conn.execute(f"""
            SELECT "종목코드", "날짜", COUNT(*) AS cnt FROM {_identifier(table_name)}
            GROUP BY "종목코드", "날짜" HAVING COUNT(*) >= ?
            ORDER BY "날짜", "종목코드"
        """, [seq_len + 1]).fetchdf()
        if limit is not None:
            keys = keys.sample(n=min(limit, len(keys)), random_state=42)
        for row in keys.itertuples(index=False):
            stock, date, _ = row
            tie = '"번호", rowid' if "번호" in schema else "rowid"
            frame = conn.execute(f"""
                SELECT * FROM {_identifier(table_name)}
                WHERE "종목코드" = ? AND "날짜" = ?
                ORDER BY TRY_CAST("시간" AS DOUBLE), {tie}
            """, [stock, date]).fetchdf()
            try:
                frame = frame.iloc[chronological_order(frame["시간"])].reset_index(drop=True)
                # No backward fill: an event can use only previously seen values.
                for column in frame:
                    if column not in {"날짜", "종목코드", "종목명", "시간", "번호"}:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce").ffill().fillna(0.0)
                names = frame["종목명"] if "종목명" in frame else pd.Series("", index=frame.index)
                frame["종목명_scalar"] = compute_stock_name_scalar_batch(names)
                sine, cosine, scalar = compute_time_features_batch(frame["시간"], use_zscore_for_scalar=False)
                frame["시간_sin"], frame["시간_cos"], frame["시간_scalar"] = sine, cosine, scalar
                values = frame[feature_cols].to_numpy(dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite_features")
                metadata = frame[["종목코드", "날짜", "시간"]].to_numpy().astype(str)
                execution = execution_arrays(frame, price_unit)
                if any(not np.isfinite(v).all() for v in execution.values()):
                    raise ValueError("nonfinite_execution")
                filename = f"episode_{str(stock).zfill(6)}_{date}.npz"
                target = output / filename
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite {target}")
                partial = target.with_suffix(".npz.part")
                with partial.open("wb") as stream:
                    np.savez_compressed(stream, features=values, metadata=metadata,
                                        **{"execution_" + k: v for k, v in execution.items()})
                os.replace(partial, target)
                manifest.append({"file_path": filename, "stock_code": str(stock).zfill(6),
                                 "date": int(date), "length": len(values)})
            except FileExistsError:
                raise
            except ValueError as exc:
                reason = str(exc)
                rejected[reason] = rejected.get(reason, 0) + 1
                logger.warning("Rejected %s/%s: %s", stock, date, reason)
        if not manifest:
            raise ValueError(f"No valid episodes extracted; rejected={rejected}")
        payload = {"metadata": {"schema_version": 2, "feature_columns": feature_cols,
                    "return_rate_index": feature_cols.index("등락률"), "expected_features": features,
                    "feature_transform": "raw", "price_unit": price_unit,
                    "execution_price_unit": "krw", "timestamp_format": "HHMMSSmmm",
                    "source_db": db_path, "rejected_episodes": rejected},
                   "episodes": manifest}
        partial = output / "manifest.json.part"
        partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(partial, output / "manifest.json")
        return True
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="datasets")
    parser.add_argument("--output_dir", default="data/extracted_episodes_v2")
    parser.add_argument("--seq_len", type=int, default=3000)
    parser.add_argument("--features", type=int, default=27)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--price-unit", required=True, choices=tuple(PRICE_SCALES),
                        help="Unit of source current/quote prices; never inferred from their magnitude")
    args = parser.parse_args()
    extract_data(args.db, args.table, args.output_dir, args.seq_len, args.features,
                 args.max_steps, args.limit, args.price_unit)
