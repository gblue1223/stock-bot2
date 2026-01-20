# Optimization Walkthrough

## Goal
Increase training speed by optimizing resource usage in `scalping_env.py` and `trade_logger.py`.

## Changes Implemented

### 1. `scalping_env.py`: Optimized Database Queries

**Problem**: 
- The environment was executing a `DESCRIBE` query (`_get_feature_columns`) on every episode reset.
- It was fetching the *entire* day's data for a stock (potentially tens of thousands of rows) just to sample a small window (e.g., 210 rows).

**Solution**:
- **Cached Feature Columns**: Pre-fetched feature column names in `__init__` and reused them, eliminating one DB query per reset.
- **Efficient Data Sampling**: Implemented `LIMIT` and `OFFSET` in the SQL query. Now, the environment queries the row count first (cached), calculates a random offset, and fetches *only* the required rows.

```python
# Before
df = self.conn.execute("SELECT ... FROM ... WHERE ...").fetchdf() # Fetches 20,000+ rows

# After
query = "... LIMIT ? OFFSET ?"
df = self.conn.execute(query, [stock_code, date, needed_len, offset]).fetchdf() # Fetches ~200 rows
```

### 2. `trade_logger.py`: Optimized Logging IO

**Problem**:
- The `_log_to_json` method was reading the entire JSON log file, appending a new record, and rewriting the *entire* file to disk for *every single trade*. This is O(N^2) complexity and causes massive slowdowns as the log grows.

**Solution**:
- **Append-Only Logging**: Switched to an append-only approach (JSON Lines) for real-time logging. This reduces the operation to O(1).
- The full standard JSON array is still saved at the end of the session via `save_summary`.

```python
# Before
trades = json.load(f) # Read all
trades.append(new_trade)
json.dump(trades, f) # Write all

# After
f.write(json.dumps(trade_data) + '\n') # Append one line
```

## Additional Recommendations for Speed

To further increase training speed, consider the following:

1.  **Vectorized Environments**: 
    - Currently, `train_scalping.py` runs a single environment instance. 
    - Use `SubprocVecEnv` to run multiple environments in parallel processes. This will parallelize the database queries and physics steps, significantly increasing throughput (FPS).

2.  **In-Memory Dataset**:
    - If your RAM permits, load the entire dataset into memory (or use a shared memory arrow table) instead of querying DuckDB per episode. This would eliminate DB latency entirely.
