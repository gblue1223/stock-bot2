"""
추론 모듈 - 실시간 매매를 위한 추론 엔진

이 모듈은 훈련된 임베딩 모델과 GRPO 정책을 사용하여
실시간 매매 결정을 내리는 추론 파이프라인을 제공합니다.

주요 컴포넌트:
- GRPOInference: 기본 실시간 추론 엔진
- EnhancedGRPOInference: 개선된 추론 엔진 (필터링 + 리스크 관리)
- TradeLogger: 실시간 거래 로거
- Position: 포지션 정보 클래스
"""

from .enhanced_grpo_infer_xlstm import GRPOInferenceE2EXLSTM, EnhancedGRPOInferenceXLSTM, Position, Action
from .trade_logger import TradeLogger

__all__ = [
    'GRPOInferenceE2EXLSTM',
    'EnhancedGRPOInferenceXLSTM',
    'Position',
    'Action',
    'TradeLogger'
]
