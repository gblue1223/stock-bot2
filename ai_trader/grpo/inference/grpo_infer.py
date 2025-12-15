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

from ai_trader.embedding import AutoEncoderEmbedding
from ai_trader.grpo.policies import GRPOPolicy
from ai_trader.grpo.policies.direct_feature_policy import DirectFeaturePolicy

logger = logging.getLogger(__name__)


class GRPOInference:
    """
    실시간 매매를 위한 GRPO 추론 엔진
    
    DirectFeaturePolicy와 GRPOPolicy 모두 지원 (기본값: DirectFeaturePolicy)
    
    최적화:
    - TorchScript 컴파일로 추론 속도 향상
    - 임베딩 캐시로 중복 계산 방지 (GRPOPolicy만)
    - 목표 지연 시간: < 10ms per sample
    
    Args:
        policy_path: 정책 체크포인트 경로 (필수)
        embedding_model_path: 임베딩 모델 체크포인트 경로 (GRPOPolicy 사용 시 필수)
        device: 디바이스 ('cuda' 또는 'cpu')
        use_torchscript: TorchScript 컴파일 사용 여부 (기본값: True)
        cache_size: 임베딩 캐시 크기 (기본값: 1000, GRPOPolicy만 사용)
        normalization_stats: 정규화 통계 (mean, std, GRPOPolicy만 사용)
    """
    
    def __init__(
        self,
        policy_path: Union[str, Path],
        embedding_model_path: Optional[Union[str, Path]] = None,
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
        
        # 정책 로드 (먼저 로드하여 타입 확인)
        self.policy, self.policy_type = self._load_policy(policy_path)
        logger.info(f"Policy loaded successfully: {self.policy_type}")
        
        # 임베딩 모델 로드 (DirectFeaturePolicy는 필요 없음)
        if self.policy_type == 'DirectFeaturePolicy':
            self.embedding_model = None
            logger.info("DirectFeaturePolicy: Skipping embedding model")
        else:
            if embedding_model_path is None:
                raise ValueError("embedding_model_path is required for GRPOPolicy")
            self.embedding_model = self._load_embedding_model(embedding_model_path)
            logger.info("Embedding model loaded successfully")
        
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
        
        # 설정 추출 (없으면 빈 dict)
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        
        # 가중치 탐지: 다양한 체크포맷 지원 및 차원 추론 준비
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'autoencoder_state_dict' in checkpoint:
                state_dict = checkpoint['autoencoder_state_dict']
            elif 'ae_state_dict' in checkpoint:
                state_dict = checkpoint['ae_state_dict']
            elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                state_dict = checkpoint['model'].state_dict()
            else:
                # raw state dict 형태인지 확인 (값이 텐서인 경우가 많음)
                try:
                    if all(hasattr(v, 'shape') for v in checkpoint.values()):
                        state_dict = checkpoint  # type: ignore[arg-type]
                except Exception:
                    state_dict = None
        
        if state_dict is None:
            raise KeyError("Embedding checkpoint does not contain a recognizable state dict ('state_dict', 'model_state_dict', etc.)")
        
        # 구성값 부족 시 state_dict로부터 차원 자동 추론
        inferred = {}
        try:
            if 'encoder.input_projection.weight' in state_dict:
                w = state_dict['encoder.input_projection.weight']
                inferred['hidden_dim'] = int(w.shape[0])
                inferred['input_dim'] = int(w.shape[1])
            if 'encoder.bottleneck.0.weight' in state_dict:
                # Linear( hidden_dim -> embedding_dim ) so weight is (embedding_dim, hidden_dim)
                bw = state_dict['encoder.bottleneck.0.weight']
                inferred['embedding_dim'] = int(bw.shape[0])
                inferred.setdefault('hidden_dim', int(bw.shape[1]))
            if 'decoder.embedding_expansion.weight' in state_dict:
                # Linear( embedding_dim -> hidden_dim * seq_len ) so weight is (hidden_dim*seq_len, embedding_dim)
                ew = state_dict['decoder.embedding_expansion.weight']
                if 'embedding_dim' not in inferred:
                    inferred['embedding_dim'] = int(ew.shape[1])
                if 'hidden_dim' in inferred:
                    inferred['seq_len'] = int(ew.shape[0] // inferred['hidden_dim'])
            if 'decoder.output_projection.weight' in state_dict and 'input_dim' not in inferred:
                # Linear( hidden_dim -> input_dim ) so weight is (input_dim, hidden_dim)
                ow = state_dict['decoder.output_projection.weight']
                inferred.setdefault('input_dim', int(ow.shape[0]))
                inferred.setdefault('hidden_dim', int(ow.shape[1]))
        except Exception as _:
            pass

        # 최종 하이퍼파라미터 결정: state_dict에서 추론된 값을 우선 사용하여 가중치와 일치 보장
        def pick(name: str, default_val):
            if name in inferred:
                return int(inferred[name])
            val = config.get(name, None)
            return int(val) if val is not None else default_val

        input_dim = pick('input_dim', 60)
        embedding_dim = pick('embedding_dim', 128)
        hidden_dim = pick('hidden_dim', 256)
        seq_len = pick('seq_len', 60)
        num_layers = int(config.get('num_layers', 3))
        dropout = float(config.get('dropout', 0.1))

        # 모델 생성 (추론값 반영)
        model = AutoEncoderEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
        logger.info(f"Embedding dims -> input_dim={input_dim}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, seq_len={seq_len}")
        
        # 가중치 로드 (호환성을 위해 strict=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Embedding state_dict missing keys: {list(missing)[:5]}{' …' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"Embedding state_dict unexpected keys: {list(unexpected)[:5]}{' …' if len(unexpected) > 5 else ''}")
        model.to(self.device)
        model.eval()
        
        # 모델 속성 저장 (TorchScript 컴파일에 필요)
        model.seq_len = seq_len
        model.input_dim = input_dim
        
        # 정규화 통계 추출 (제공되지 않은 경우)
        if self.normalization_stats is None and isinstance(checkpoint, dict):
            if 'normalization_stats' in checkpoint:
                self.normalization_stats = checkpoint['normalization_stats']
            elif 'norm' in checkpoint and isinstance(checkpoint['norm'], dict) and \
                 'mean' in checkpoint['norm'] and 'std' in checkpoint['norm']:
                self.normalization_stats = checkpoint['norm']
        
        return model
    
    def _load_policy(self, policy_path: Union[str, Path]) -> Tuple[nn.Module, str]:
        """
        정책 로드
        
        Args:
            policy_path: 정책 체크포인트 경로
            
        Returns:
            (로드된 정책, 정책 타입)
        """
        checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
        
        # 설정 추출
        config = checkpoint.get('config', {})
        
        # 정책 타입 감지
        policy_type = config.get('policy_type')
        
        # policy_type이 없으면 state_dict 키로 추론
        if policy_type is None:
            state_dict_key = 'policy_state_dict' if 'policy_state_dict' in checkpoint else 'state_dict'
            state_dict = checkpoint.get(state_dict_key, {})
            
            # input_projection이 있으면 DirectFeaturePolicy
            if any('input_projection' in k for k in state_dict.keys()):
                policy_type = 'DirectFeaturePolicy'
                logger.info("Auto-detected policy type: DirectFeaturePolicy")
            else:
                policy_type = 'GRPOPolicy'
                logger.info("Auto-detected policy type: GRPOPolicy")
        
        # state_dict 가져오기
        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise KeyError("Neither 'policy_state_dict' nor 'state_dict' found in checkpoint")
        
        # 정책 타입에 따라 생성 (config에 없으면 state_dict에서 차원 추론)
        if policy_type == 'DirectFeaturePolicy':
            # state_dict에서 실제 차원 추론
            if 'input_dim' not in config:
                # input_projection.weight: (hidden_dim, input_dim)
                input_dim = state_dict['input_projection.weight'].shape[1]
                hidden_dim = state_dict['input_projection.weight'].shape[0]
                logger.info(f"Inferred from state_dict: input_dim={input_dim}, hidden_dim={hidden_dim}")
            else:
                input_dim = config.get('input_dim', 3600)
                hidden_dim = config.get('hidden_dim', 128)
            
            policy = DirectFeaturePolicy(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=config.get('action_dim', 3)
            )
        else:
            # state_dict에서 실제 차원 추론
            if 'embedding_dim' not in config:
                # fc1.weight: (hidden_dim, embedding_dim)
                embedding_dim = state_dict['fc1.weight'].shape[1]
                hidden_dim = state_dict['fc1.weight'].shape[0]
                logger.info(f"Inferred from state_dict: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
            else:
                embedding_dim = config.get('embedding_dim', 128)
                hidden_dim = config.get('hidden_dim', 256)
            
            policy = GRPOPolicy(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                action_dim=config.get('action_dim', 3)
            )
        
        # 가중치 로드
        policy.load_state_dict(state_dict)
        
        policy.to(self.device)
        policy.eval()
        
        return policy, policy_type
    
    def _compile_models(self):
        """
        TorchScript로 모델 컴파일하여 추론 속도 향상
        """
        try:
            # 임베딩 모델 컴파일 (GRPOPolicy만)
            if self.policy_type == 'GRPOPolicy' and self.embedding_model is not None:
                example_input = torch.randn(
                    1, 
                    self.embedding_model.seq_len, 
                    self.embedding_model.input_dim,
                    device=self.device
                )
                # encoder만 trace (encode 메서드 사용)
                self.embedding_model.encoder = torch.jit.trace(
                    self.embedding_model.encoder, 
                    example_input
                )
            
            # 정책 컴파일
            if hasattr(self.policy, 'embedding_dim'):
                example_input = torch.randn(
                    1, 
                    self.policy.embedding_dim,
                    device=self.device
                )
            else:
                example_input = torch.randn(
                    1, 
                    self.policy.input_dim,
                    device=self.device
                )
            self.policy = torch.jit.trace(
                self.policy,
                example_input
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
            embedding = self.embedding_model.encode(normalized_seq)  # (1, embedding_dim)
        
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
        2. 임베딩 생성 (GRPOPolicy) 또는 평탄화 (DirectFeaturePolicy)
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
        
        # 정책 타입에 따라 입력 처리
        if self.policy_type == 'DirectFeaturePolicy':
            # 직접 특징 사용: 시퀀스를 평탄화
            if sequence.dim() == 2:
                # (seq_len, input_dim) -> (seq_len * input_dim,)
                policy_input = sequence.flatten()
            else:
                raise ValueError(f"Expected 2D sequence for DirectFeaturePolicy, got {sequence.dim()}D")
            
            # 배치 차원 추가: (input_dim,) -> (1, input_dim)
            policy_input = policy_input.unsqueeze(0)
        else:
            # GRPOPolicy: 임베딩 생성 (캐시 활용)
            embedding = self._generate_embedding(sequence)
            # 배치 차원 추가: (embedding_dim,) -> (1, embedding_dim)
            policy_input = embedding.unsqueeze(0)
        
        # 정책 실행
        with torch.no_grad():
            action_logits, _ = self.policy(policy_input)
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
        
        # 정책 타입에 따라 입력 처리
        if self.policy_type == 'DirectFeaturePolicy':
            # 직접 특징 사용: 시퀀스를 평탄화
            if sequences.dim() == 3:
                # (batch_size, seq_len, input_dim) -> (batch_size, seq_len * input_dim)
                policy_input = sequences.flatten(start_dim=1)
            else:
                raise ValueError(f"Expected 3D sequences for batch DirectFeaturePolicy, got {sequences.dim()}D")
        else:
            # GRPOPolicy: 정규화 및 임베딩 생성
            normalized_seqs = self._normalize_input(sequences)
            with torch.no_grad():
                policy_input = self.embedding_model.encode(normalized_seqs)  # (batch_size, embedding_dim)
        
        # 정책 실행 (배치)
        with torch.no_grad():
            action_logits, _ = self.policy(policy_input)
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
