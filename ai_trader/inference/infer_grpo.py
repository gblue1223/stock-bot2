"""
GRPO 추론 엔진

이 모듈은 실시간 매매를 위한 GRPO 추론 엔진을 정의합니다.
임베딩 모델과 정책을 로드하고, TorchScript 컴파일로 최적화하며,
임베딩 캐시를 구현하여 빠른 추론을 제공합니다.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from ai_trader.embedding.models import TradingEmbeddingModel
from ai_trader.grpo.policy import GRPOPolicy

logger = logging.getLogger(__name__)


class GRPOInference:
    """
    실시간 매매를 위한 GRPO 추론 엔진
    
    최적화:
    - TorchScript 컴파일로 추론 속도 향상
    - 임베딩 캐시로 중복 계산 방지
    - 목표 지연 시간: < 10ms per sample
    
    Args:
        embedding_model_path: 임베딩 모델 체크포인트 경로
        policy_path: 정책 체크포인트 경로
        device: 디바이스 ('cuda' 또는 'cpu')
        use_torchscript: TorchScript 컴파일 사용 여부 (기본값: True)
        cache_size: 임베딩 캐시 크기 (기본값: 1000)
        normalization_stats: 정규화 통계 (mean, std)
    """
    
    def __init__(
        self,
        embedding_model_path: Union[str, Path],
        policy_path: Union[str, Path],
        device: str = 'cuda',
        use_torchscript: bool = True,
        cache_size: int = 1000,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_torchscript = use_torchscript
        self.cache_size = cache_size
        
        logger.info(f"Initializing GRPOInference on device: {self.device}")
        
        # 정규화 통계 설정 (임베딩 모델 로드 전에 초기화)
        self.normalization_stats = normalization_stats
        
        # 임베딩 모델 로드
        self.embedding_model = self._load_embedding_model(embedding_model_path)
        logger.info("Embedding model loaded successfully")
        
        # 정책 로드
        self.policy = self._load_policy(policy_path)
        logger.info("Policy loaded successfully")
        
        # 정규화 통계 설정 (임베딩 모델 로드 후 업데이트될 수 있음)
        if normalization_stats is not None:
            self.mean = torch.tensor(
                normalization_stats['mean'], 
                dtype=torch.float32, 
                device=self.device
            )
            self.std = torch.tensor(
                normalization_stats['std'], 
                dtype=torch.float32, 
                device=self.device
            )
            logger.info("Normalization statistics loaded from parameter")
        elif self.normalization_stats is not None:
            # 임베딩 모델 로드 중에 설정된 경우
            self.mean = torch.tensor(
                self.normalization_stats['mean'], 
                dtype=torch.float32, 
                device=self.device
            )
            self.std = torch.tensor(
                self.normalization_stats['std'], 
                dtype=torch.float32, 
                device=self.device
            )
            logger.info("Normalization statistics loaded from checkpoint")
        else:
            self.mean = None
            self.std = None
            logger.warning("No normalization statistics provided")
        
        # 임베딩 캐시 초기화 (LRU 캐시)
        self.embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        
        # TorchScript 컴파일
        if use_torchscript:
            self._compile_models()
            logger.info("Models compiled with TorchScript")
        
        # 성능 메트릭
        self.inference_times = []
        
    def _load_embedding_model(self, model_path: Union[str, Path]) -> nn.Module:
        """
        임베딩 모델 로드
        
        Args:
            model_path: 모델 체크포인트 경로
            
        Returns:
            로드된 임베딩 모델
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 설정 추출
        config = checkpoint.get('config', {})
        
        # 모델 생성
        model = TradingEmbeddingModel(
            input_dim=config.get('input_dim', 60),
            embedding_dim=config.get('embedding_dim', 128),
            seq_len=config.get('seq_len', 60),
            num_heads=config.get('num_heads', 4),
            conv_channels=config.get('conv_channels', None)
        )
        
        # 가중치 로드
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        # 정규화 통계 추출 (제공되지 않은 경우)
        if self.normalization_stats is None and 'normalization_stats' in checkpoint:
            self.normalization_stats = checkpoint['normalization_stats']
        
        return model
    
    def _load_policy(self, policy_path: Union[str, Path]) -> nn.Module:
        """
        정책 로드
        
        Args:
            policy_path: 정책 체크포인트 경로
            
        Returns:
            로드된 정책
        """
        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        
        # 설정 추출
        config = checkpoint.get('config', {})
        
        # 정책 생성
        policy = GRPOPolicy(
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            action_dim=config.get('action_dim', 3)
        )
        
        # 가중치 로드
        policy.load_state_dict(checkpoint['state_dict'])
        policy.to(self.device)
        policy.eval()
        
        return policy
    
    def _compile_models(self):
        """
        TorchScript로 모델 컴파일하여 추론 속도 향상
        """
        try:
            # 임베딩 모델 컴파일
            example_input = torch.randn(
                1, 
                self.embedding_model.seq_len, 
                self.embedding_model.input_dim,
                device=self.device
            )
            self.embedding_model = torch.jit.trace(
                self.embedding_model, 
                example_input
            )
            
            # 정책 컴파일
            example_embedding = torch.randn(
                1, 
                self.policy.embedding_dim,
                device=self.device
            )
            self.policy = torch.jit.trace(
                self.policy,
                example_embedding
            )
            
            logger.info("Models successfully compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}. Using eager mode.")
            self.use_torchscript = False
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터 정규화
        
        Args:
            x: 입력 텐서 (batch, seq_len, input_dim) 또는 (seq_len, input_dim)
            
        Returns:
            정규화된 텐서
        """
        if self.mean is None or self.std is None:
            return x
        
        # 정규화: (x - mean) / std
        # Broadcasting을 위해 차원 맞추기
        if x.dim() == 2:
            # (seq_len, input_dim)
            return (x - self.mean) / (self.std + 1e-8)
        elif x.dim() == 3:
            # (batch, seq_len, input_dim)
            mean = self.mean.unsqueeze(0).unsqueeze(0)
            std = self.std.unsqueeze(0).unsqueeze(0)
            return (x - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")
    
    def _get_cache_key(self, sequence: torch.Tensor) -> str:
        """
        시퀀스에 대한 캐시 키 생성
        
        Args:
            sequence: 입력 시퀀스 텐서
            
        Returns:
            캐시 키 문자열
        """
        # 시퀀스의 해시를 캐시 키로 사용
        # 마지막 타임스텝의 특징 일부를 사용하여 키 생성
        last_step = sequence[-1, :5].cpu().numpy()  # 마지막 타임스텝의 처음 5개 특징
        key = ','.join([f"{x:.6f}" for x in last_step])
        return key
    
    def _get_cached_embedding(
        self, 
        sequence: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        캐시에서 임베딩 조회
        
        Args:
            sequence: 입력 시퀀스 텐서
            
        Returns:
            캐시된 임베딩 또는 None
        """
        cache_key = self._get_cache_key(sequence)
        
        if cache_key in self.embedding_cache:
            # LRU: 최근 사용된 항목을 끝으로 이동
            self.embedding_cache.move_to_end(cache_key)
            return self.embedding_cache[cache_key]
        
        return None
    
    def _cache_embedding(
        self, 
        sequence: torch.Tensor, 
        embedding: torch.Tensor
    ):
        """
        임베딩을 캐시에 저장
        
        Args:
            sequence: 입력 시퀀스 텐서
            embedding: 생성된 임베딩 텐서
        """
        cache_key = self._get_cache_key(sequence)
        
        # 캐시 크기 제한
        if len(self.embedding_cache) >= self.cache_size:
            # 가장 오래된 항목 제거 (FIFO)
            self.embedding_cache.popitem(last=False)
        
        # 새 임베딩 추가
        self.embedding_cache[cache_key] = embedding.detach()
    
    def _generate_embedding(
        self, 
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        시퀀스에 대한 임베딩 생성 (캐시 활용)
        
        Args:
            sequence: 입력 시퀀스 텐서 (seq_len, input_dim)
            
        Returns:
            임베딩 벡터 (embedding_dim,)
        """
        # 캐시 확인
        cached_embedding = self._get_cached_embedding(sequence)
        if cached_embedding is not None:
            return cached_embedding
        
        # 정규화
        normalized_seq = self._normalize_input(sequence)
        
        # 배치 차원 추가: (seq_len, input_dim) -> (1, seq_len, input_dim)
        normalized_seq = normalized_seq.unsqueeze(0)
        
        # 임베딩 생성
        with torch.no_grad():
            embedding = self.embedding_model(normalized_seq)  # (1, embedding_dim)
        
        # 배치 차원 제거: (1, embedding_dim) -> (embedding_dim,)
        embedding = embedding.squeeze(0)
        
        # 캐시에 저장
        self._cache_embedding(sequence, embedding)
        
        return embedding
    
    def predict(
        self, 
        sequence: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        입력 시퀀스에 대한 행동 예측
        
        이 메서드는 다음 단계를 수행합니다:
        1. 입력 시퀀스 정규화
        2. 임베딩 생성 (캐시 활용)
        3. 정책 실행 및 행동 반환
        
        목표 지연 시간: < 10ms
        
        Args:
            sequence: 입력 시퀀스 (seq_len, input_dim) 또는 numpy array
            deterministic: True면 최대 확률 행동 선택, False면 확률적 샘플링
            
        Returns:
            (action, confidence) 튜플
            - action: 선택된 행동 (0: 보유, 1: 매수, 2: 매도)
            - confidence: 행동에 대한 신뢰도 (확률)
        """
        start_time = time.time()
        
        # numpy array를 torch tensor로 변환
        if isinstance(sequence, np.ndarray):
            sequence = torch.tensor(
                sequence, 
                dtype=torch.float32, 
                device=self.device
            )
        elif not isinstance(sequence, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(sequence)}")
        
        # 디바이스로 이동
        if sequence.device != self.device:
            sequence = sequence.to(self.device)
        
        # 임베딩 생성 (캐시 활용)
        embedding = self._generate_embedding(sequence)
        
        # 배치 차원 추가: (embedding_dim,) -> (1, embedding_dim)
        embedding = embedding.unsqueeze(0)
        
        # 정책 실행
        with torch.no_grad():
            action_logits, _ = self.policy(embedding)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            if deterministic:
                # 최대 확률 행동 선택
                action = torch.argmax(action_probs, dim=-1)
                confidence = action_probs[0, action].item()
            else:
                # 확률적 샘플링
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                confidence = action_probs[0, action].item()
        
        # 결과 추출
        action = action.item()
        
        # 추론 시간 기록
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # 최근 100개 추론 시간만 유지
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        return action, confidence
    
    def predict_batch(
        self,
        sequences: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        여러 시퀀스에 대한 배치 예측
        
        Args:
            sequences: 입력 시퀀스 배치 (batch_size, seq_len, input_dim)
            deterministic: True면 최대 확률 행동 선택, False면 확률적 샘플링
            
        Returns:
            (actions, confidences) 튜플
            - actions: 선택된 행동들 (batch_size,)
            - confidences: 행동에 대한 신뢰도들 (batch_size,)
        """
        start_time = time.time()
        
        # numpy array를 torch tensor로 변환
        if isinstance(sequences, np.ndarray):
            sequences = torch.tensor(
                sequences,
                dtype=torch.float32,
                device=self.device
            )
        elif not isinstance(sequences, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(sequences)}")
        
        # 디바이스로 이동
        if sequences.device != self.device:
            sequences = sequences.to(self.device)
        
        # 정규화
        normalized_seqs = self._normalize_input(sequences)
        
        # 임베딩 생성 (배치)
        with torch.no_grad():
            embeddings = self.embedding_model(normalized_seqs)  # (batch_size, embedding_dim)
        
        # 정책 실행 (배치)
        with torch.no_grad():
            action_logits, _ = self.policy(embeddings)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            if deterministic:
                # 최대 확률 행동 선택
                actions = torch.argmax(action_probs, dim=-1)
                confidences = torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1)
            else:
                # 확률적 샘플링
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()
                confidences = torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1)
        
        # numpy로 변환
        actions = actions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # 추론 시간 기록
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        return actions, confidences
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        추론 성능 통계 반환
        
        Returns:
            성능 통계 딕셔너리
        """
        if not self.inference_times:
            return {
                'mean_inference_time_ms': 0.0,
                'median_inference_time_ms': 0.0,
                'p95_inference_time_ms': 0.0,
                'p99_inference_time_ms': 0.0
            }
        
        times = np.array(self.inference_times)
        return {
            'mean_inference_time_ms': float(np.mean(times)),
            'median_inference_time_ms': float(np.median(times)),
            'p95_inference_time_ms': float(np.percentile(times, 95)),
            'p99_inference_time_ms': float(np.percentile(times, 99)),
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """
        캐시 히트율 계산 (간단한 추정)
        
        Returns:
            캐시 히트율 (0.0 ~ 1.0)
        """
        # 실제 구현에서는 캐시 히트/미스 카운터를 유지해야 함
        # 여기서는 캐시 크기 기반 추정
        if self.cache_size == 0:
            return 0.0
        return min(1.0, len(self.embedding_cache) / self.cache_size)
    
    def clear_cache(self):
        """임베딩 캐시 초기화"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def reset_performance_stats(self):
        """성능 통계 초기화"""
        self.inference_times.clear()
        logger.info("Performance statistics reset")
