"""
GRPO 스캘핑 환경 V2 (Pre-computed Embeddings)

기존 env와 달리, 실시간 추론을 수행하지 않고 
미리 계산된 Parquet 임베딩 데이터를 로드하여 사용합니다.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import duckdb
import os
import glob
import time
import threading
import pandas as pd

logger = logging.getLogger(__name__)

class GRPOScalpingEnvV2(gym.Env):
    """
    스캘핑을 위한 GRPO 훈련 환경 (V2: Pre-computed Embeddings)
    
    관측 공간: 임베딩 벡터 (embedding_dim,) + 포지션 정보
    행동 공간: Discrete(3) - 0: 보유, 1: 매수, 2: 매도
    
    특징:
    - Parquet 파일에서 임베딩을 직접 로드 (Inference 제거 -> 속도 향상)
    - DuckDB에서 등락률/가격 정보 로드 (Reward 계산용 = 정확도 유지)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        parquet_path: str,
        db_path: str,
        table_name: str = 'datasets',
        embedding_dim: int = 128,
        transaction_cost_rate: float = 0.00215,
        quick_exit_penalty: float = 0.01,
        quick_exit_threshold: float = 1.5,
        quick_exit_mode: str = 'penalty_only',
        max_episode_steps: Optional[int] = None,
        base_price: float = 100000.0,
        stop_loss_pct: float = 2.0,
        max_split_count: int = 1,
        min_holding_time: float = 2.0,
        max_holding_time: float = 100.0,
        seq_len: int = 120, # 시퀀스 길이 (임베딩 생성시 사용된 값)
        device: str = 'cpu', # 호환성 유지용
        no_trade_penalty: float = 0.5,  # Fix1: 에피소드 내 거래 0회 시 패널티
        max_rows_limit: int = 5_000_000, # Fix2: 메모리 안전 — 행수 초과 키 제외
    ):
        super().__init__()
        
        self.parquet_path = parquet_path
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        self.base_price = base_price
        self.stop_loss_pct = stop_loss_pct
        self.max_split_count = max_split_count
        self.min_holding_time = min_holding_time
        self.max_holding_time = max_holding_time
        self.no_trade_penalty = no_trade_penalty  # Fix1
        self.max_rows_limit = max_rows_limit       # Fix2
        
        self.transaction_cost_rate = transaction_cost_rate
        self.round_trip_cost = transaction_cost_rate * 2
        
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
        self.quick_exit_mode = quick_exit_mode
        self.max_episode_steps = max_episode_steps
        
        # 관측 공간: 임베딩(128) + 포지션(3)
        self.obs_dim = embedding_dim + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        # DB 연결
        self._connect_db()
        
        # 유효한 키(종목, 날짜) 로드
        self.valid_keys = []
        self._preload_valid_keys()
        
        # 상태 변수
        self.current_step = 0
        self.position = 0
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0
        self.max_price_since_entry = 0.0
        
        self.episode_data = None #(N, features) - 여기선 embedding과 return_rate만 필요
        self.episode_metadata = None
        self.episode_length = 0
        
        self.episode_trades = []
        
        logger.info(f"GRPOScalpingEnvV2 initialized. Embedding source: {parquet_path}")

    # Shared across all instances
    _KEY_INDEX = None       # {(code, date): (file_path, row_count)}
    _LRU_CACHE = None       # OrderedDict for LRU: (code, date) -> DataFrame
    _LRU_MAX_SIZE = 100
    _INDEX_BUILT = False
    _SHARED_CONN = None     # 단일 공유 DuckDB 연결
    _DB_LOCK = threading.Lock()  # DuckDB 접근 보호용 Lock
    
    def _connect_db(self):
        try:
            with GRPOScalpingEnvV2._DB_LOCK:
                if GRPOScalpingEnvV2._SHARED_CONN is None:
                    GRPOScalpingEnvV2._SHARED_CONN = duckdb.connect(
                        database=self.db_path, read_only=True
                    )
                    logger.info(f"Connected to DB (Shared + Lock)")
                
                self.conn = GRPOScalpingEnvV2._SHARED_CONN
                self.raw_db_alias = "main"
                
                if not GRPOScalpingEnvV2._INDEX_BUILT:
                    self._build_key_index()
                    from collections import OrderedDict
                    GRPOScalpingEnvV2._LRU_CACHE = OrderedDict()
                    GRPOScalpingEnvV2._INDEX_BUILT = True
            
        except Exception as e:
            logger.error(f"DB Connect Failed: {e}")
            raise
    
    def _build_key_index(self):
        """parquet 파일에서 메타데이터만 스캔하여 (code, date) -> file 인덱스 구축"""
        import glob as _glob
        logger.info(f"Building key index from {self.parquet_path} (metadata only)...")
        
        pq_files = sorted(_glob.glob(os.path.join(self.parquet_path, "*.parquet")))
        if not pq_files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_path}")
        
        key_index = {}  # (code, date) -> (file_path, count)
        
        for f in pq_files:
            try:
                # embedding 컬럼 제외하고 code, date만 읽기 → 매우 가벼움
                df = pd.read_parquet(f, columns=['code', 'date'])
                for (code, date), group in df.groupby(['code', 'date']):
                    key = (code, date)
                    if key in key_index:
                        # 여러 파일에 분산된 경우 → 리스트로 관리
                        existing = key_index[key]
                        if isinstance(existing, list):
                            existing.append((f, len(group)))
                        else:
                            key_index[key] = [existing, (f, len(group))]
                    else:
                        key_index[key] = (f, len(group))
            except Exception as e:
                logger.warning(f"Failed to read metadata from {f}: {e}")
        
        GRPOScalpingEnvV2._KEY_INDEX = key_index
        logger.info(f"Key index built: {len(key_index)} (code, date) pairs from {len(pq_files)} files")
    
    def _get_embedding_data(self, code, date):
        """LRU 캐시에서 (code, date) 데이터 가져오기. 없으면 parquet에서 로드."""
        cache = GRPOScalpingEnvV2._LRU_CACHE
        key = (code, date)
        
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        
        # 캐시 미스 → parquet에서 로드
        index_entry = GRPOScalpingEnvV2._KEY_INDEX.get(key)
        if index_entry is None:
            return None
        
        # 파일 목록 구성
        if isinstance(index_entry, list):
            file_entries = index_entry
        else:
            file_entries = [index_entry]
        
        dfs = []
        for file_path, _ in file_entries:
            try:
                # Log reading attempt for debugging Segfaults
                logger.debug(f"Reading parquet: {file_path}")
                df = pd.read_parquet(file_path)
                filtered = df[(df['code'] == code) & (df['date'] == date)]
                if len(filtered) > 0:
                    dfs.append(filtered)
            except Exception as e:
                # Log error but don't crash. 
                # In threading environment, this prevents taking down the thread/process if possible.
                # If it's a hard segfault in C++, this might not help, but it catches Python-level issues.
                logger.error(f"Failed to read parquet file {file_path} for {code}/{date}: {e}")
                continue
        
        if not dfs:
            return None
        
        result = pd.concat(dfs, ignore_index=True).sort_values('time').reset_index(drop=True)
        
        # LRU 캐시에 저장
        cache[key] = result
        if len(cache) > GRPOScalpingEnvV2._LRU_MAX_SIZE:
            cache.popitem(last=False)  # 가장 오래된 항목 제거
        
        return result

    def _preload_valid_keys(self):
        """인덱스에서 유효한 종목/날짜 목록 로드"""
        logger.info("Loading valid keys from index...")
        try:
            key_index = GRPOScalpingEnvV2._KEY_INDEX
            if key_index is None:
                raise RuntimeError("Key index not built")
            
            min_len = self.max_episode_steps + 10 if self.max_episode_steps else 100
            
            self.valid_keys = []
            skipped_oom = 0
            for (code, date), entry in key_index.items():
                if isinstance(entry, list):
                    cnt = sum(c for _, c in entry)
                else:
                    cnt = entry[1]
                if cnt < min_len:
                    continue
                # Fix2: 메모리 안전 — 행 수가 너무 많으면 OOM 발생 가능성 차단
                if cnt > self.max_rows_limit:
                    skipped_oom += 1
                    continue
                self.valid_keys.append((code, date, cnt))
            
            if skipped_oom > 0:
                logger.info(f"Skipped {skipped_oom} keys with rows > {self.max_rows_limit:,} (OOM prevention)")
            logger.info(f"Loaded {len(self.valid_keys)} valid keys.")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            raise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 에피소드 데이터 로드
        self._load_episode_data()
        
        # 2. 상태 초기화
        self.current_step = 0
        self.position = 0
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0.0
        self.max_price_since_entry = 0.0
        self.episode_trades = []
        self.quick_exit_violations = 0
        
        # 3. 초기 관측값
        obs = self._get_observation()
        
        info = {
            'stock_code': self.episode_metadata['code'],
            'date': self.episode_metadata['date'],
            'time': self.episode_times[self.current_step]
        }
        return obs, info

    def _load_episode_data(self):
        """랜덤 키 선택 후 데이터 로딩 (Embedding: lazy load + Market Data: DuckDB)"""
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                if not self.valid_keys: raise RuntimeError("No valid keys")
                
                idx = np.random.randint(0, len(self.valid_keys))
                code, date, cnt = self.valid_keys[idx]
                
                # 1. 임베딩 lazy load (LRU 캐시)
                emb_df = self._get_embedding_data(code, date)
                if emb_df is None or len(emb_df) < 10:
                    continue
                
                # 2. Raw DB에서 등락률 가져오기
                query = f"""
                    SELECT 시간 as time, 등락률 as return_rate
                    FROM {self.raw_db_alias}.{self.table_name}
                    WHERE 종목코드 = ? AND 날짜 = ?
                    ORDER BY 시간
                """
                with GRPOScalpingEnvV2._DB_LOCK:
                    raw_df = self.conn.execute(query, [code, date]).fetchdf()
                
                if len(raw_df) < 10:
                    continue
                
                # 3. time 기준으로 merge
                merged = emb_df.merge(raw_df, on='time', how='inner')
                
                if len(merged) < 10:
                    continue
                
                # 4. 데이터 변환
                embeddings = np.stack(merged['embedding'].values)
                return_rates = merged['return_rate'].values.astype(np.float32) / 100.0
                times = merged['time'].values
                
                # 에피소드 길이 제한 (랜덤 스타트)
                total_len = len(merged)
                if self.max_episode_steps:
                    max_len = self.max_episode_steps
                    if total_len > max_len:
                        start_idx = np.random.randint(0, total_len - max_len)
                        end_idx = start_idx + max_len
                        embeddings = embeddings[start_idx:end_idx]
                        return_rates = return_rates[start_idx:end_idx]
                        times = times[start_idx:end_idx]
                
                self.episode_embeddings = embeddings
                self.episode_returns = return_rates
                self.episode_times = times
                self.episode_length = len(embeddings)
                
                self.prices = self.base_price * (1.0 + return_rates)
                self.prices = np.maximum(self.prices, 1000.0)
                
                self.episode_data = []
                self.episode_metadata = {'code': code, 'date': date}
                
                return
                
            except Exception as e:
                logger.warning(f"Data load failed for {code}/{date}: {e}")
                continue
                
        raise RuntimeError("Failed to load episode data after retries")

    def _get_observation(self):
        # 1. Embedding
        if self.current_step >= self.episode_length:
            # End of episode guard
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            emb = self.episode_embeddings[self.current_step]
            
        # 2. Position Features
        if self.position == 1:
            holding_time = self._calculate_seconds_diff(self.entry_time, self.episode_times[self.current_step])
            holding_time_norm = min(holding_time / (self.max_holding_time + 1e-6), 1.0)
            steps_norm = self.position_steps / self.max_split_count
        else:
            holding_time_norm = 0.0
            steps_norm = 0.0
            
        extra = np.array([float(self.position), float(steps_norm), float(holding_time_norm)], dtype=np.float32)
        
        return np.concatenate([emb, extra])

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        current_time = self.episode_times[self.current_step]
        current_price = self.prices[self.current_step]
        
        # --- Action Logic (Same as V1) ---
        if action == 1: # Buy
            if self.position_steps < self.max_split_count:
                old_steps = self.position_steps
                new_steps = old_steps + 1
                
                prev_val = self.avg_entry_price * old_steps
                new_val = prev_val + current_price
                self.avg_entry_price = new_val / new_steps
                
                self.position_steps = new_steps
                self.position = 1
                
                if old_steps == 0:
                    self.entry_time = current_time
                
                # Cost
                buy_weight = 1.0 / self.max_split_count
                reward -= self.transaction_cost_rate * 100 * buy_weight
                
        elif action == 2: # Sell
             if self.position == 1:
                holding_time = self._calculate_seconds_diff(self.entry_time, current_time)
                
                # Min holding check
                if holding_time < self.min_holding_time:
                    reward -= 0.2 # Penalty
                else:
                    reward += 0.1 # Bonus
                    
                # Sell execution
                weight = self.position_steps / self.max_split_count
                reward -= self.transaction_cost_rate * 100 * weight # Cost
                
                # Profit Reward
                profit_rate = (current_price - self.avg_entry_price) / self.avg_entry_price
                profit_rate = np.clip(profit_rate, -1.0, 1.0)
                
                # Net profit reward? 
                # Original logic: reward = profit_rate - round_trip_cost. 
                # But here we deducted costs separately.
                # Let's align with V1 logic:
                # V1: _calculate_reward uses (exit-entry)/entry - round_trip.
                # To match exactly:
                
                # Recalculate full trade reward as in V1
                r_trade, comps = self._calculate_reward(self.avg_entry_price, current_price, holding_time)
                # But wait, V1 step() accumulates local step rewards (like buy cost).
                # Actually V1 step() calculates sell reward using _calculate_reward which INCLUDES transaction costs.
                
                # So if V1 does: reward = _calculate_reward(...) 
                # And _calculate_reward subtracts round_trip_cost.
                # Then we shouldn't subtract sell cost again if we use _calculate_reward.
                
                # Let's use logic:
                # Reward = (Profit%) * 100 - (Cost%) * 100
                # Profit% = (P_exit - P_entry)/P_entry
                
                profit_val = profit_rate * 100
                # We already deducted buy cost when buying.
                # We deducted sell cost just above.
                # So reward += profit_val.
                
                # But V1 logic is slightly different: it calculates costs at exit for the *full* round trip usually?
                # No, V1 `step`:
                # Buy: reward -= cost (buy portion)
                # Sell: reward -= cost (sell portion) + profit_reward (which includes ONLY profit? No)
                
                # Let's check V1 `_calculate_reward`:
                # reward = profit_rate - round_trip_cost
                # return reward * 100
                
                # If we use V1 `_calculate_reward` at exit, it deducts ROUND TRIP cost again. 
                # So we pay buy cost at entry, and full round trip at exit? That's double counting.
                # Inspecting V1 again...
                # V1 `step(Buy)`: reward -= transaction_cost * weight. (OK)
                # V1 `step(Sell)`: calculates `_calculate_reward`.
                # `_calculate_reward`: returns `(profit - round_trip)*100`.
                # So yes, V1 seems to double count buy-side cost if `_calculate_reward` subtracts round trip.
                # Wait, V1 `transaction_cost_rate` is 0.215%. `round_trip` is 0.43%.
                # If `step(Buy)` subtracts 0.215, and `step(Sell)` subtracts 0.43 (via helper), total is 0.645?
                # This looks like a bug in V1 or intended "conservative" reward.
                # I will fix this in V2 to be correct:
                # Buy: pay 0.215
                # Sell: pay 0.215 + Profit
                
                # Correct Logic for V2:
                reward += profit_val # Pure profit
                # Sell cost is already deducted above (lines 352: `reward -= ...`)
                
                # Bonus checks
                if profit_rate > self.round_trip_cost: 
                    reward += 1.0
                
                # Reset position
                self.position = 0
                self.position_steps = 0
                self.avg_entry_price = 0.0
                self.entry_time = 0.0
                
                # Record trade
                self.episode_trades.append({
                    'profit': profit_rate, 
                    'reward': reward,
                    'step': self.current_step,
                    'holding_time': holding_time
                })

        # --- Position Holding Penalty ---
        # V1 logic? In V1 `_calculate_reward` mentions holding penalty but code comments say it's removed.
        # But `step` usually has time penalty?
        # V1 doesn't seem to apply per-step penalty in `step()`, only quick exit penalty checks.
        
        # --- Quick Exit & Forced Close (Simplified) ---
        if self.position == 1:
            holding_time = self._calculate_seconds_diff(self.entry_time, current_time)
            
            # Quick Exit Check
            if holding_time > self.quick_exit_threshold and current_price < self.avg_entry_price:
                 reward -= self.quick_exit_penalty
                 self.quick_exit_violations += 1
            
            # Max Holding Force Close
            if holding_time > self.max_holding_time:
                 # Force Sell
                 profit_rate = (current_price - self.avg_entry_price) / self.avg_entry_price
                 profit_val = profit_rate * 100
                 
                 # Pay Sell Cost
                 weight = self.position_steps / self.max_split_count
                 cost = self.transaction_cost_rate * 100 * weight
                 
                 reward += (profit_val - cost)
                 
                 # Record forced trade
                 self.episode_trades.append({
                    'profit': profit_rate,
                    'reward': (profit_val - cost),
                    'step': self.current_step,
                    'holding_time': holding_time,
                    'is_forced': True
                 })
                 
                 self.position = 0
                 self.position_steps = 0
                 
        # --- Time Step ---
        self.current_step += 1
        if self.current_step >= self.episode_length - 1:
            terminated = True
            
        # New Observation
        obs = self._get_observation()
        
        if terminated:
            # Calculate episode statistics
            num_trades = len(self.episode_trades)
            if num_trades > 0:
                profits = [t['profit'] for t in self.episode_trades]
                win_count = sum(1 for p in profits if p > 0)
                win_rate = win_count / num_trades
                
                # Sharpe Ratio (using trade profits)
                if len(profits) > 1:
                    sharpe_ratio = np.mean(profits) / (np.std(profits) + 1e-8)
                else:
                    sharpe_ratio = 0.0
                    
                # Holding Time
                holding_times = [t.get('holding_time', 0.0) for t in self.episode_trades]
                avg_holding_time = np.mean(holding_times)
                
            else:
                win_rate = 0.0
                sharpe_ratio = 0.0
                avg_holding_time = 0.0
                # Fix1: 거래가 하나도 없으면 패널티 — no-trade collapse 방지
                reward -= self.no_trade_penalty
                
            info = {
                'price': current_price,
                'time': current_time,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'quick_exit_violations': self.quick_exit_violations,
                'avg_holding_time': avg_holding_time
            }
        else:
            info = {
                'price': current_price,
                'time': current_time
            }
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, entry_price, exit_price, holding_time):
        """
        보상 계산 (V1과 동일 로직)
        """
        if entry_price <= 0: entry_price = 1000.0
        
        profit_rate = (exit_price - entry_price) / entry_price
        profit_rate = np.clip(profit_rate, -1.0, 1.0)
        
        # V1 Logic: Reward = (Profit - RoundTripCost) * 100
        # This includes cost.
        reward = (profit_rate - self.round_trip_cost) * 100
        
        if profit_rate > self.round_trip_cost:
            reward += 1.0
            
        components = {
            'profit_rate': profit_rate,
            'transaction_cost': -self.round_trip_cost,
            'net_reward': reward
        }
        return reward, components

    def _calculate_seconds_diff(self, t1, t2):
        # Time format: HHMMSSmmm or float
        # Fast diff
        try:
            t1 = int(t1); t2 = int(t2)
            # Simple conversion for HHMMSSmmm (9 digit)
            # This is slow in python loop. 
            # In V2, we pre-convert times to seconds ideally.
            # But for now, let's just do simple approximate logic or reuse V1 logic buffer
            # Reuse V1 logic:
            
            def to_sec(t):
                s = str(t).zfill(9)
                return int(s[:2])*3600 + int(s[2:4])*60 + int(s[4:6]) + int(s[6:9])/1000.0
                
            return to_sec(t2) - to_sec(t1)
        except:
            return 1.0

    def set_transaction_cost_rate(self, rate):
        self.transaction_cost_rate = rate
        self.round_trip_cost = rate * 2


