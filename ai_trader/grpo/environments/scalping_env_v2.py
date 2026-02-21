"""
GRPO 스캘핑 환경 V2 (Pre-computed Embeddings + Market Data)

Parquet 파일에 embedding + 등락률 등 모든 시장 데이터가 포함되어 있어
DuckDB 없이 Parquet 파일만으로 동작합니다 (속도 향상).
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import glob
import time
import threading
import pandas as pd

logger = logging.getLogger(__name__)

class GRPOScalpingEnvV2(gym.Env):
    """
    스캘핑을 위한 GRPO 훈련 환경 (V2: Pre-computed Embeddings + Market Data)
    
    관측 공간: 임베딩 벡터 (embedding_dim,) + 포지션 정보
    행동 공간: Discrete(3) - 0: 보유, 1: 매수, 2: 매도
    
    특징:
    - Parquet 파일(embeddings_v3)에서 embedding + 등락률 등 모든 데이터 로드
    - DuckDB 의존성 완전 제거 → 에피소드마다 DB 쿼리 없음 → 속도 향상
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        parquet_path: str,
        db_path: str = None,       # 하위호환 유지 (사용 안 함)
        table_name: str = 'datasets',  # 하위호환 유지 (사용 안 함)
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
        seq_len: int = 120,
        device: str = 'cpu',
        no_trade_penalty: float = 0.5,
        max_rows_limit: int = 5_000_000,
    ):
        super().__init__()
        
        self.parquet_path = parquet_path
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        self.base_price = base_price
        self.stop_loss_pct = stop_loss_pct
        self.max_split_count = max_split_count
        self.min_holding_time = min_holding_time
        self.max_holding_time = max_holding_time
        self.no_trade_penalty = no_trade_penalty
        self.max_rows_limit = max_rows_limit
        
        self.transaction_cost_rate = transaction_cost_rate
        self.round_trip_cost = transaction_cost_rate * 2
        
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
        self.quick_exit_mode = quick_exit_mode
        self.max_episode_steps = max_episode_steps
        
        # 관측 공간: 임베딩(128) + 호가창(20) + 종목명/거래대금(2) + 포지션(3) = 153
        self.obs_dim = embedding_dim + 20 + 2 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        # 유효한 키(종목, 날짜) 인덱스 구축 및 로드
        self.valid_keys = []
        self._build_key_index_if_needed()
        self._preload_valid_keys()
        
        # 상태 변수
        self.current_step = 0
        self.position = 0
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0
        self.max_price_since_entry = 0.0
        
        self.episode_data = None
        self.episode_metadata = None
        self.episode_length = 0
        
        self.episode_trades = []
        
        logger.info(f"GRPOScalpingEnvV2 initialized (DuckDB-free). Source: {parquet_path}")

    # Shared across all instances
    _KEY_INDEX = None       # {(code, date): (file_path, row_count)}
    _LRU_CACHE = None       # OrderedDict for LRU: (code, date) -> DataFrame
    _LRU_MAX_SIZE = 50
    _INDEX_BUILT = False
    _INDEX_LOCK = threading.Lock()  # 인덱스 구축 보호용 Lock
    _CACHE_LOCK = threading.Lock()  # LRU 캐시 + parquet I/O 보호용 Lock

    # 학습에 실제로 필요한 컬럼만 로드 (메모리 절약 + 원본 호가창 및 시계열 피처 추가)
    _REQUIRED_COLUMNS = ['date', 'time', 'code', 'embedding', '등락률', '종목명_scalar', '누적거래대금'] + \
                        [f'매도대기금액{i}' for i in range(1, 11)] + \
                        [f'매수대기금액{i}' for i in range(1, 11)]

    def _build_key_index_if_needed(self):
        """인덱스가 아직 구축되지 않은 경우에만 구축 (스레드 안전)"""
        with GRPOScalpingEnvV2._INDEX_LOCK:
            if not GRPOScalpingEnvV2._INDEX_BUILT:
                self._build_key_index()
                from collections import OrderedDict
                GRPOScalpingEnvV2._LRU_CACHE = OrderedDict()
                GRPOScalpingEnvV2._INDEX_BUILT = True
    
    def _build_key_index(self):
        """parquet 파일에서 메타데이터만 스캔하여 (code, date) -> file 인덱스 구축"""
        import glob as _glob
        import pyarrow.parquet as pq
        import json
        
        index_cache_path = os.path.join(self.parquet_path, "key_index_v3.json")
        
        if os.path.exists(index_cache_path):
            logger.info(f"Loading key index from cache: {index_cache_path}")
            try:
                with open(index_cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON keys are strings e.g. "('005930', '20240102')". Convert back to tuple.
                    import ast
                    key_index = {ast.literal_eval(k): v for k, v in data.items()}
                    GRPOScalpingEnvV2._KEY_INDEX = key_index
                    logger.info(f"Key index loaded: {len(key_index)} pairs from cache")
                    return
            except Exception as e:
                logger.warning(f"Failed to load index cache: {e}. Rebuilding...")

        logger.info(f"Building key index from {self.parquet_path} (metadata only)...")
        
        pq_files = sorted(_glob.glob(os.path.join(self.parquet_path, "*.parquet")))
        if not pq_files:
            raise FileNotFoundError(f"No parquet files in {self.parquet_path}")
        
        key_index = {}  # (code, date) -> (file_path, count) or [(file_path, count)]
        
        for f in pq_files:
            try:
                # pyarrow를 사용해 스키마만 로드 (메모리 절약)
                try:
                    parquet_file = pq.ParquetFile(f)
                    if '등락률' not in parquet_file.schema.names:
                        continue
                except Exception as e:
                    logger.debug(f"Skipping {f} (Schema error: {e})")
                    continue
                    
                # 2. 인덱스 구축 진행 (컬럼이 있는 파일만, code와 date만 로드)
                df = pd.read_parquet(f, columns=['code', 'date'])
                for (code, date), group in df.groupby(['code', 'date']):
                    key = (code, date)
                    if key in key_index:
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
        
        # 캐시에 저장
        try:
            with open(index_cache_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in key_index.items()}, f)
            logger.info(f"Saved key index cache to {index_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")
    
    def _get_embedding_data(self, code, date):
        """LRU 캐시에서 (code, date) 데이터 가져오기. 없으면 parquet에서 로드.
        
        Thread-safe: _CACHE_LOCK으로 OrderedDict 접근과 pyarrow I/O를 직렬화.
        pyarrow의 read_parquet은 동일 파일에 대한 동시 접근 시 segfault 유발 가능.
        """
        key = (code, date)
        
        with GRPOScalpingEnvV2._CACHE_LOCK:
            cache = GRPOScalpingEnvV2._LRU_CACHE
            
            # 캐시 히트
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            
            # 캐시 미스 → parquet에서 로드 (Lock 내부에서 I/O 수행)
            index_entry = GRPOScalpingEnvV2._KEY_INDEX.get(key)
            if index_entry is None:
                return None
            
            # 파일 목록 구성 (JSON 캐시에서 로드 시 형태 정규화)
            if isinstance(index_entry, list):
                # flat [path, count] vs nested [[path1, count1], [path2, count2]]
                if len(index_entry) == 2 and isinstance(index_entry[0], str):
                    file_entries = [index_entry]  # single entry
                else:
                    file_entries = index_entry  # multiple entries
            else:
                file_entries = [index_entry]
            
            dfs = []
            for file_path, _ in file_entries:
                try:
                    # PyArrow pushdown filters를 이용해 필요한 에피소드만 C++ 레벨에서 메모리에 로드
                    df = pd.read_parquet(
                        file_path, 
                        columns=self._REQUIRED_COLUMNS,
                        filters=[('code', '=', code), ('date', '=', str(date))]
                    )
                    
                    # date가 숫자/문자열 형태로 섞여 있을 수 있으니 fallback 추가
                    if len(df) == 0 and not isinstance(date, int):
                        try:
                            df = pd.read_parquet(
                                file_path, 
                                columns=self._REQUIRED_COLUMNS,
                                filters=[('code', '=', code), ('date', '=', int(date))]
                            )
                        except Exception:
                            pass

                    # 혹시 필터가 완벽히 동작하지 않았을 경우를 대비한 2차 필터링
                    filtered = df[(df['code'] == code) & (df['date'].astype(str) == str(date))]
                    if len(filtered) > 0:
                        dfs.append(filtered)
                    del df  # 즉시 해제
                except Exception as e:
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
                    # flat [path, count] vs nested [[path1, count1], ...]
                    if len(entry) == 2 and isinstance(entry[0], str):
                        cnt = entry[1]
                    else:
                        cnt = sum(item[1] for item in entry)
                else:
                    cnt = entry[1]
                if cnt < min_len:
                    continue
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
        
        # 3. 초기 관측값 (float32 보장)
        obs = self._get_observation()
        
        # info의 값이 string인 경우 numpy array 변환 시 string array가 되는 것을 방지하기 위해 타입 명시 필요
        info = {
            'stock_code': str(self.episode_metadata['code']),
            'date': str(self.episode_metadata['date']),
            'time': float(self.episode_times[self.current_step])
        }
        return obs, info

    def _load_episode_data(self):
        """랜덤 키 선택 후 데이터 로딩 (Parquet only — DuckDB 없음)"""
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                if not self.valid_keys: raise RuntimeError("No valid keys")
                
                idx = np.random.randint(0, len(self.valid_keys))
                code, date, cnt = self.valid_keys[idx]
                
                # Parquet LRU 캐시에서 로드 (embedding + 등락률 모두 포함)
                df = self._get_embedding_data(code, date)
                if df is None or len(df) < 10:
                    continue
                
                # 등락률 컬럼 확인
                if '등락률' not in df.columns:
                    logger.warning(f"'등락률' column missing in parquet for {code}/{date}. 재생성 필요.")
                    continue
                
                # 데이터 변환 (명시적으로 float32 타입 보장)
                embeddings = np.stack(df['embedding'].values).astype(np.float32)
                return_rates = df['등락률'].values.astype(np.float32) / 100.0
                times = df['time'].values
                
                # 호가창 데이터 20개 추출 및 정규화
                orderbook_cols = [f'매도대기금액{i}' for i in range(1, 11)] + [f'매수대기금액{i}' for i in range(1, 11)]
                orderbooks = df[orderbook_cols].values.astype(np.float32)
                
                # 호가창 원본 데이터는 단위가 큼 (억원/천만원 단위). 모델 입력을 위해 약간 스케일링
                orderbooks = np.clip(orderbooks / 1e8, 0, 100.0) 
                
                # 추가 피처: 종목명_scalar 및 누적거래대금 추출
                name_scalars = df['종목명_scalar'].values.astype(np.float32)
                volumes = np.clip(df['누적거래대금'].values.astype(np.float32) / 1e10, 0, 100.0)
                extra_features = np.column_stack([name_scalars, volumes]).astype(np.float32)
                
                # 에피소드 길이 제한 (랜덤 스타트)
                total_len = len(df)
                if self.max_episode_steps:
                    max_len = self.max_episode_steps
                    if total_len > max_len:
                        start_idx = np.random.randint(0, total_len - max_len)
                        end_idx = start_idx + max_len
                        embeddings = embeddings[start_idx:end_idx]
                        return_rates = return_rates[start_idx:end_idx]
                        times = times[start_idx:end_idx]
                        orderbooks = orderbooks[start_idx:end_idx]
                        extra_features = extra_features[start_idx:end_idx]
                
                self.episode_embeddings = embeddings
                self.episode_returns = return_rates
                self.episode_times = times
                self.episode_orderbooks = orderbooks
                self.episode_extra_features = extra_features
                self.episode_length = len(embeddings)
                
                self.prices = self.base_price * (1.0 + return_rates)
                self.prices = np.maximum(self.prices, 1000.0)
                
                self.episode_data = []
                self.episode_metadata = {'code': code, 'date': date}
                
                return
                
            except Exception as e:
                c = code if 'code' in locals() else "Unknown"
                d = date if 'date' in locals() else "Unknown"
                logger.warning(f"Data load failed for {c}/{d}: {e}")
                continue
                
        raise RuntimeError("Failed to load episode data after retries")

    def _get_observation(self):
        # 1. Embedding & Orderbooks & Extra Features
        if self.current_step >= self.episode_length:
            # End of episode guard
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
            ob = np.zeros(20, dtype=np.float32)
            xf = np.zeros(2, dtype=np.float32)
        else:
            emb = self.episode_embeddings[self.current_step]
            ob = self.episode_orderbooks[self.current_step]
            xf = self.episode_extra_features[self.current_step]
            
        # 2. Position Features
        if self.position == 1:
            holding_time = self._calculate_seconds_diff(self.entry_time, self.episode_times[self.current_step])
            holding_time_norm = min(holding_time / (self.max_holding_time + 1e-6), 1.0)
            steps_norm = self.position_steps / self.max_split_count
        else:
            holding_time_norm = 0.0
            steps_norm = 0.0
            
        extra = np.array([float(self.position), float(steps_norm), float(holding_time_norm)], dtype=np.float32)
        
        return np.concatenate([emb, ob, xf, extra]).astype(np.float32)

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
                profits = [float(t['profit']) for t in self.episode_trades]
                win_count = sum(1 for p in profits if p > 0)
                win_rate = float(win_count / num_trades)
                
                # Sharpe Ratio (using trade profits)
                if len(profits) > 1:
                    sharpe_ratio = float(np.mean(profits) / (np.std(profits) + 1e-8))
                else:
                    sharpe_ratio = 0.0
                    
                # Holding Time
                holding_times = [float(t.get('holding_time', 0.0)) for t in self.episode_trades]
                avg_holding_time = float(np.mean(holding_times))
                
            else:
                win_rate = 0.0
                sharpe_ratio = 0.0
                avg_holding_time = 0.0
                # Fix1: 거래가 하나도 없으면 패널티 — no-trade collapse 방지
                reward -= self.no_trade_penalty
                
            info = {
                'price': float(current_price),
                'time': float(current_time),
                'num_trades': int(num_trades),
                'win_rate': float(win_rate),
                'sharpe_ratio': float(sharpe_ratio),
                'quick_exit_violations': int(self.quick_exit_violations),
                'avg_holding_time': float(avg_holding_time)
            }
        else:
            info = {
                'price': float(current_price),
                'time': float(current_time)
            }
        
        return obs, float(reward), bool(terminated), bool(truncated), info

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


