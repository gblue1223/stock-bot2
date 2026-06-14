"""
GRPO 모듈 - Group Relative Policy Optimization

이 모듈은 그룹 상대 정책 최적화를 사용하여 스캘핑 전략을 학습하는
강화학습 에이전트를 제공합니다.

주요 컴포넌트:
- GRPOScalpingEnv: 스캘핑 환경 (Gymnasium)
- GRPOTrainer: GRPO 훈련기
- GRPOPolicy: 정책 네트워크
- GRPOEvaluationMetrics: 평가 메트릭 계산기
"""

from ai_trader.grpo.environments.scalping_env_e2e import GRPOScalpingEnv
from ai_trader.grpo.environments.scalping_env_xlstm import GRPOScalpingEnvXLSTM
from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM
from ai_trader.grpo.grpo import GRPOTrainer

__all__ = [
    'GRPOScalpingEnv',
    'GRPOScalpingEnvXLSTM',
    'GRPOPolicyE2E',
    'GRPOPolicyE2EXLSTM',
    'GRPOTrainer'
]
