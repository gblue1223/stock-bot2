"""
xLSTM 및 E2E 학습을 위한 최적화된 Gymnasium 환경

이 모듈은 기존 GRPOScalpingEnv를 상속받아, 사전 추출된 데이터(.npz)를 로드하여
학습 속도를 높이고 멀티프로세싱 안정성을 보장하는 환경을 구현합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv

logger = logging.getLogger(__name__)


class GRPOScalpingEnvXLSTM(GRPOScalpingEnv):
    """
    xLSTM 전용 최적화 환경
    
    데이터 디렉토리(extracted_dir)가 지정되면 사전 추출된 npz 파일을 고속 로드하고,
    그렇지 않으면 기존 DuckDB 환경으로 폴백하여 호환성을 유지합니다.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: str = 'datasets',
        seq_len: int = 3000,
        expected_features: int = 28,
        transaction_cost_rate: float = 0.00215,
        buy_tax_rate: float = 0.0,
        sell_tax_rate: float = 0.0018,
        no_trade_penalty: float = 10.0,
        quick_exit_penalty: float = 0.01,
        quick_exit_threshold: float = 1.5,
        quick_exit_mode: str = 'penalty_only',
        max_episode_steps: Optional[int] = None,
        use_raw_data: bool = True,
        rolling_window_size: int = 1000,
        rolling_min_samples: int = 100,
        base_price: float = 100000.0,
        stop_loss_pct: float = 2.0,
        max_split_count: int = 1,
        min_holding_time: float = 2.0,
        max_holding_time: float = 100.0,
        early_exit_penalty: float = 0.2,
        max_trades_per_episode: Optional[int] = None,
        extracted_dir: Optional[str] = None  # ✅ 추가: 사전 추출 데이터 디렉토리
    ):
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.manifest_data = None
        
        # extracted_dir가 있는 경우 db_path를 더미로 전달하여 초기 통과
        effective_db_path = db_path if db_path else (str(self.extracted_dir) if self.extracted_dir else "dummy.db")
        
        super().__init__(
            db_path=effective_db_path,
            table_name=table_name,
            seq_len=seq_len,
            expected_features=expected_features,
            transaction_cost_rate=transaction_cost_rate,
            buy_tax_rate=buy_tax_rate,
            sell_tax_rate=sell_tax_rate,
            no_trade_penalty=no_trade_penalty,
            quick_exit_penalty=quick_exit_penalty,
            quick_exit_threshold=quick_exit_threshold,
            quick_exit_mode=quick_exit_mode,
            max_episode_steps=max_episode_steps,
            use_raw_data=use_raw_data,
            rolling_window_size=rolling_window_size,
            rolling_min_samples=rolling_min_samples,
            base_price=base_price,
            stop_loss_pct=stop_loss_pct,
            max_split_count=max_split_count,
            min_holding_time=min_holding_time,
            max_holding_time=max_holding_time,
            early_exit_penalty=early_exit_penalty,
            max_trades_per_episode=max_trades_per_episode
        )
        
        if self.extracted_dir:
            logger.info(f"Loaded environment using pre-extracted data from: {self.extracted_dir}")
        else:
            logger.info("Loaded environment using live DuckDB queries")

    def _init_metadata_from_db(self):
        """메타데이터 초기화 오버라이드 (추출 디렉토리 지원)"""
        if self.extracted_dir is not None:
            manifest_path = self.extracted_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
                
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest_data = json.load(f)
                
            metadata = self.manifest_data.get("metadata", {})
            self.feature_columns = metadata.get("feature_columns")
            self.return_rate_index = metadata.get("return_rate_index", 0)
            
            logger.info(f"[Cache Mode] Feature mapping loaded: return_rate_index={self.return_rate_index}")
            
            # valid_keys 세팅
            self.valid_keys = []
            for ep in self.manifest_data.get("episodes", []):
                self.valid_keys.append((ep["file_path"], ep["stock_code"], ep["date"], ep["length"]))
            
            logger.info(f"[Cache Mode] Loaded {len(self.valid_keys)} valid keys from manifest.")
        else:
            super()._init_metadata_from_db()

    def _ensure_db_connection(self):
        """데이터베이스 연결 및 에피소드 초기화 오버라이드"""
        if self.extracted_dir is not None:
            # DB 연결 없이 내부 상태변수들만 초기화
            self.current_step = 0
            self.position = 0
            self.position_steps = 0
            self.avg_entry_price = 0.0
            self.entry_time = 0
            self.current_price = 0.0
            self.current_time = 0
            self.max_price_since_entry = 0.0
            self.episode_trades = []
            self.episode_rewards = []
            self.quick_exit_violations = 0
            self.episode_data = None
            self.episode_metadata = None
            self.episode_length = 0
        else:
            super()._ensure_db_connection()

    def _sample_episode_start(self, max_attempts=100) -> Tuple[np.ndarray, np.ndarray]:
        """에피소드 시작 샘플링 오버라이드"""
        if self.extracted_dir is not None:
            if not self.valid_keys:
                raise RuntimeError("No valid extracted episodes in keys list")
                
            attempt = 0
            while attempt < max_attempts:
                try:
                    # 무작위 에피소드 선택
                    idx = np.random.randint(0, len(self.valid_keys))
                    file_path_rel, stock_code, date, length = self.valid_keys[idx]
                    
                    full_path = self.extracted_dir / file_path_rel
                    data = np.load(full_path, allow_pickle=True)
                    
                    features = data['features']
                    metadata = data['metadata']
                    
                    # 에피소드 스텝만큼 슬라이싱
                    if self.max_episode_steps is not None:
                        needed_len = self.seq_len + self.max_episode_steps
                        if len(features) > needed_len:
                            max_start_idx = len(features) - needed_len
                            start_idx = np.random.randint(0, max_start_idx)
                            end_idx = start_idx + needed_len
                            features = features[start_idx:end_idx]
                            metadata = metadata[start_idx:end_idx]
                            
                    return features, metadata
                except Exception as e:
                    logger.warning(f"Failed to load extracted episode (attempt {attempt+1}): {e}")
                    attempt += 1
            raise RuntimeError(f"Failed to sample episode from extracted dir after {max_attempts} attempts")
        else:
            return super()._sample_episode_start(max_attempts)
            
    def __getstate__(self):
        state = super().__getstate__()
        state['manifest_data'] = None  # 직렬화 시 캐시된 메타데이터 제외하여 크기 축소
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # 역직렬화 시 manifest 파일 다시 로드 필요하면 복원
        if self.extracted_dir:
            manifest_path = Path(self.extracted_dir) / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        self.manifest_data = json.load(f)
                except Exception:
                    pass
