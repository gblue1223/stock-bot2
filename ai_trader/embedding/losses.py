"""
대조 학습 손실 함수

이 모듈은 임베딩 모델 훈련을 위한 대조 학습 손실 함수를 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) 손실 함수
    
    대조 학습을 위한 손실 함수로, 긍정 쌍의 유사도는 최대화하고
    부정 쌍의 유사도는 최소화합니다.
    
    손실 계산:
    - 앵커와 긍정 샘플 간의 코사인 유사도를 최대화
    - 앵커와 부정 샘플 간의 코사인 유사도를 최소화
    - Temperature 파라미터로 손실 스케일 조정
    
    Args:
        temperature: 온도 파라미터 (기본값: 0.07)
                    낮을수록 더 날카로운 분포, 높을수록 더 부드러운 분포
        reduction: 손실 축소 방법 ('mean', 'sum', 'none') (기본값: 'mean')
    
    Example:
        >>> loss_fn = InfoNCELoss(temperature=0.07)
        >>> anchor = torch.randn(32, 128)  # (batch, embedding_dim)
        >>> positive = torch.randn(32, 128)
        >>> negatives = torch.randn(32, 10, 128)  # (batch, num_negatives, embedding_dim)
        >>> loss = loss_fn(anchor, positive, negatives)
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE 손실 계산
        
        Args:
            anchor: 앵커 임베딩 (batch, embedding_dim)
            positive: 긍정 샘플 임베딩 (batch, embedding_dim)
            negatives: 부정 샘플 임베딩 (batch, num_negatives, embedding_dim)
        
        Returns:
            손실 값 (스칼라 또는 배치별 손실)
        """
        batch_size = anchor.shape[0]
        embedding_dim = anchor.shape[1]
        num_negatives = negatives.shape[1]
        
        # 임베딩 정규화 (코사인 유사도 계산을 위해)
        anchor = F.normalize(anchor, p=2, dim=1)  # (batch, embedding_dim)
        positive = F.normalize(positive, p=2, dim=1)  # (batch, embedding_dim)
        negatives = F.normalize(negatives, p=2, dim=2)  # (batch, num_negatives, embedding_dim)
        
        # 긍정 쌍의 코사인 유사도 계산
        # (batch, embedding_dim) * (batch, embedding_dim) -> (batch,)
        positive_similarity = torch.sum(anchor * positive, dim=1)  # (batch,)
        
        # 부정 쌍의 코사인 유사도 계산
        # (batch, 1, embedding_dim) * (batch, num_negatives, embedding_dim) -> (batch, num_negatives)
        negative_similarity = torch.bmm(
            negatives,  # (batch, num_negatives, embedding_dim)
            anchor.unsqueeze(2)  # (batch, embedding_dim, 1)
        ).squeeze(2)  # (batch, num_negatives)
        
        # Temperature로 스케일링
        positive_similarity = positive_similarity / self.temperature  # (batch,)
        negative_similarity = negative_similarity / self.temperature  # (batch, num_negatives)
        
        # InfoNCE 손실 계산
        # log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = pos - log(exp(pos) + sum(exp(neg)))
        # = pos - logsumexp([pos, neg1, neg2, ...])
        
        # 긍정과 부정 유사도를 결합
        # (batch, 1 + num_negatives)
        logits = torch.cat([
            positive_similarity.unsqueeze(1),  # (batch, 1)
            negative_similarity  # (batch, num_negatives)
        ], dim=1)
        
        # 긍정 샘플의 인덱스는 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        # Cross-entropy 손실 계산 (InfoNCE와 동등)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss
    
    def forward_symmetric(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        대칭적 InfoNCE 손실 계산
        
        앵커->긍정과 긍정->앵커 방향 모두 계산하여 평균
        더 안정적인 학습을 위해 사용
        
        Args:
            anchor: 앵커 임베딩 (batch, embedding_dim)
            positive: 긍정 샘플 임베딩 (batch, embedding_dim)
            negatives: 부정 샘플 임베딩 (batch, num_negatives, embedding_dim)
        
        Returns:
            대칭적 손실 값
        """
        # 앵커 -> 긍정 방향 손실
        loss_forward = self.forward(anchor, positive, negatives)
        
        # 긍정 -> 앵커 방향 손실
        loss_backward = self.forward(positive, anchor, negatives)
        
        # 평균
        return (loss_forward + loss_backward) / 2.0


class TripletLoss(nn.Module):
    """
    Triplet Loss (대안적 대조 학습 손실)
    
    앵커, 긍정, 부정 샘플 간의 거리를 직접 최적화합니다.
    목표: d(anchor, positive) + margin < d(anchor, negative)
    
    Args:
        margin: 마진 값 (기본값: 1.0)
        distance_metric: 거리 메트릭 ('euclidean', 'cosine') (기본값: 'euclidean')
        reduction: 손실 축소 방법 ('mean', 'sum', 'none') (기본값: 'mean')
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = 'euclidean',
        reduction: str = 'mean'
    ):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction
        
        if distance_metric not in ['euclidean', 'cosine']:
            raise ValueError(f"distance_metric must be 'euclidean' or 'cosine', got {distance_metric}")
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def _compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """거리 계산"""
        if self.distance_metric == 'euclidean':
            return torch.norm(x1 - x2, p=2, dim=1)
        else:  # cosine
            x1_norm = F.normalize(x1, p=2, dim=1)
            x2_norm = F.normalize(x2, p=2, dim=1)
            # 코사인 거리 = 1 - 코사인 유사도
            return 1.0 - torch.sum(x1_norm * x2_norm, dim=1)
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Triplet Loss 계산
        
        Args:
            anchor: 앵커 임베딩 (batch, embedding_dim)
            positive: 긍정 샘플 임베딩 (batch, embedding_dim)
            negative: 부정 샘플 임베딩 (batch, embedding_dim)
        
        Returns:
            손실 값
        """
        # 거리 계산
        positive_distance = self._compute_distance(anchor, positive)
        negative_distance = self._compute_distance(anchor, negative)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = F.relu(positive_distance - negative_distance + self.margin)
        
        # 축소
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
