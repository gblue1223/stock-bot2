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
        device: str = 'cpu' # 호환성 유지용
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

    # Shared connection for same-process instances
    _SHARED_CONN = None
    
    def _connect_db(self):
        try:
            # Singleton Pattern for DB Connection
            if GRPOScalpingEnvV2._SHARED_CONN is not None:
                self.conn = GRPOScalpingEnvV2._SHARED_CONN
                self.raw_db_alias = "raw_db" # Assumed fixed alias for shared conn
                return

            import uuid
            # DuckDB In-Memory 연결 후, 원본 DB와 Parquet를 attach/read
            self.conn = duckdb.connect(database=':memory:') # 메인은 메모리 DB
            
            # 원본 DB Attach (Read Only)
            self.raw_db_alias = "raw_db"
            self.conn.execute(f"ATTACH '{self.db_path}' AS {self.raw_db_alias} (READ_ONLY)")
            
            # Parquet 파일 경로 패턴
            # 윈도우 경로인 경우 역슬래시 처리 주의
            pq_pattern = os.path.join(self.parquet_path, "*.parquet").replace("\\", "/")
            
            # 뷰 생성 (Parquet 파일들 전체를 하나의 테이블처럼)
            # datasets 테이블과 조인하기 위해 뷰 생성
            self.conn.execute(f"""
                CREATE VIEW embeddings_view AS 
                SELECT * FROM read_parquet('{pq_pattern}')
            """)
            
            logger.info(f"Connected to DB and created embeddings_view (Shared Connection)")
            
            # Save to class variable
            GRPOScalpingEnvV2._SHARED_CONN = self.conn
            
        except Exception as e:
            logger.error(f"DB Connect Failed: {e}")
            raise

    def _preload_valid_keys(self):
        """임베딩이 존재하는 종목/날짜 목록 로드"""
        logger.info("Loading valid keys from EMBEDDINGS (this might take a moment)...")
        try:
            # 임베딩 뷰에서 종목/날짜 추출
            # DISTINCT가 꽤 느릴 수 있음. processed_stocks.txt가 있으면 그걸 쓰는게 나을수도?
            # 일단 정확성을 위해 뷰에서 조회.
            query = """
                SELECT code, date, COUNT(*) as cnt
                FROM embeddings_view
                GROUP BY code, date
            """
            df = self.conn.execute(query).fetchdf()
            
            # 최소 길이 필터링
            min_len = self.max_episode_steps + 10 if self.max_episode_steps else 100
            valid_df = df[df['cnt'] >= min_len]
            
            self.valid_keys = list(zip(valid_df['code'], valid_df['date'], valid_df['cnt']))
            logger.info(f"Loaded {len(self.valid_keys)} valid keys from parquet files.")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            raise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 에피소드 데이터 로드 (JOIN Query)
        self._load_episode_data()
        
        # 2. 상태 초기화
        self.current_step = 0
        self.position = 0
        self.position_steps = 0
        self.avg_entry_price = 0.0
        self.entry_time = 0.0
        self.max_price_since_entry = 0.0
        self.episode_trades = []
        
        # 3. 초기 관측값
        obs = self._get_observation()
        
        info = {
            'stock_code': self.episode_metadata['code'],
            'date': self.episode_metadata['date'],
            'time': self.episode_times[self.current_step]
        }
        return obs, info

    def _load_episode_data(self):
        """랜덤 키 선택 후 데이터 로딩 (Embedding + Market Data)"""
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                if not self.valid_keys: raise RuntimeError("No valid keys")
                
                idx = np.random.randint(0, len(self.valid_keys))
                code, date, cnt = self.valid_keys[idx]
                
                # Query: Join Embeddings with Raw Data to get Return Rate & Time
                # 임베딩 Parquet에는 (date, time, code, embedding)이 있음.
                # Raw DB에는 (날짜, 시간, 종목코드, 등락률)이 있음.
                # JOIN 조건: code=종목코드, date=날짜, time=시간
                
                # 주의: Parquet의 time과 DB의 시간 포맷이 같은지 확인 필요.
                # generate_embeddings 스크립트는 원본 그대로 저장했으므로 같을 것임.
                
                query = f"""
                    SELECT 
                        t1.embedding, 
                        t2.등락률 as return_rate, 
                        t1.time 
                    FROM embeddings_view t1
                    JOIN {self.raw_db_alias}.{self.table_name} t2 
                        ON t1.code = t2.종목코드 
                        AND t1.date = t2.날짜 
                        AND t1.time = t2.시간
                    WHERE t1.code = ? AND t1.date = ?
                    ORDER BY t1.time
                """
                
                df = self.conn.execute(query, [code, date]).fetchdf()
                
                if len(df) < 10:
                    continue

                # 데이터 변환
                # embedding 컬럼은 list/array 형태일 것임. numpy로 변환.
                # DuckDB fetchdf는 list of floats로 가져옴.
                
                # DataFrame -> Numpy Struct array or dict of arrays
                embeddings = np.stack(df['embedding'].values) # (N, 128)
                return_rates = df['return_rate'].values.astype(np.float32) / 100.0 # (%) -> ratio
                times = df['time'].values
                
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
                
                # 구조화된 데이터로 저장
                # (빠른 접근을 위해 numpy array 유지)
                self.episode_embeddings = embeddings
                self.episode_returns = return_rates
                self.episode_times = times
                self.episode_length = len(embeddings)
                
                # 가격 재계산 (Base Price 기준) -> reward 계산용
                # (1 + r) * base
                # 벡터 연산
                self.prices = self.base_price * (1.0 + return_rates)
                self.prices = np.maximum(self.prices, 1000.0) # 하한가 방어
                
                # Simple access wrapper
                self.episode_data = [] # Not really used in this struct, keeping for compatibility if needed
                # Just use indices
                
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
                    'step': self.current_step
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
            
            # Max Holding Force Close
            if holding_time > self.max_holding_time:
                 # Force Sell
                 profit_rate = (current_price - self.avg_entry_price) / self.avg_entry_price
                 profit_val = profit_rate * 100
                 
                 # Pay Sell Cost
                 weight = self.position_steps / self.max_split_count
                 cost = self.transaction_cost_rate * 100 * weight
                 
                 reward += (profit_val - cost)
                 
                 self.position = 0
                 self.position_steps = 0
                 
        # --- Time Step ---
        self.current_step += 1
        if self.current_step >= self.episode_length - 1:
            terminated = True
            
        # New Observation
        obs = self._get_observation()
        
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


