"""
HTML 보고서 생성 모듈

훈련 진행 상황, 최종 메트릭, 샘플 거래 에피소드를 요약하는 HTML 보고서를 생성합니다.
"""

from ai_trader.reporting.report_generator import (
    generate_embedding_report,
    generate_grpo_report,
    generate_combined_report
)

__all__ = [
    'generate_embedding_report',
    'generate_grpo_report',
    'generate_combined_report'
]
