import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProgramTradingConfig:
    """실시간 프로그램 매매 10분할 트레이더 설정 클래스"""

    # 모니터링 및 매매 대상 종목 목록 (6자리 종목코드)
    target_stocks: List[str] = field(default_factory=lambda: ["005930", "000660"])

    # 분할 매매 설정
    split_count: int = 10  # 분할 횟수 (10분할)
    total_budget_per_stock: float = 1_000_000.0  # 종목당 총 할당 예산 (원)
    interval_seconds: int = 300  # 분할 주문 실행 체크 주기 (초 단위, 기본 5분)

    # 수급 시그널 설정
    min_net_buy_amount: float = 0.0  # 진입에 필요한 최소 프로그램 순매수 금액 (백만원 단위)
    min_net_buy_trend_irds: float = 0.0  # 매수 진입 시 필요한 순매수 증감 기준 (백만원 단위, >0 시 매수)
    sell_net_buy_trend_threshold: float = 100_000.0  # 프로그램 급격한 매도 시그널 이탈 기준 (백만원 단위, 100,000 = -1,000억원)

    # 10분 상승세 확인 매수 설정
    buy_trend_window_minutes: int = 10  # 매수 시 상승세 감지 분 창 (기본 10분)
    min_buy_trend_increase: float = 0.0  # 10분간 최소 순매수 상승 증가액 (백만원 단위, >0 시 10분간 상승으로 판단)

    # 10분 완만한 하락 추세 청산 설정
    trend_window_minutes: int = 10  # 추세 감지 분 창 (기본 10분)
    gentle_downward_threshold: float = 10_000.0  # 10분간 완만 하락 청산 감도 기준 (백만원 단위, 10,000 = -100억원)
    gentle_downward_pct: float = 10.0  # 1분 단위 스냅샷 기준 이전 10분 대비 완만 하락 비율 (기본 10.0%)
    consecutive_decrease_ratio: float = 0.1  # 1분 단위 하락 분 비율 기준 (기본 10% 이상)

    # 리스크 관리 및 재매수 설정
    stop_loss_pct: float = 2.0  # 손절 비율 (%) -> -2% 손실 시 전량 매도
    take_profit_pct: float = 3.0  # 익절 비율 (%) -> +3% 수익 시 전량 매도
    market_close_time: str = "15:15:00"  # 장 마감 전 강제 청산 시각 (HH:MM:SS)
    rebuy_cooldown_minutes: int = 10  # 매도 청산 완료 후 신규 재매수 금지 쿨다운 시간 (분 단위, 기본 10분)

    # 거래 모드 및 계좌 설정
    account_no: Optional[str] = None  # 계좌번호 (None일 경우 첫번째 계좌 자동 선택)
    dry_run: bool = False  # 드라이런 모드 (True일 경우 실제 주문을 내지 않고 로그만 출력)
    amount_quantity_type: str = "1"  # 키움 API 금액수량구분 (1: 금액 백만원, 2: 수량 천주)
    stock_exchange_type: str = "KRX"  # 거래소 구분 ("KRX": 한국거래소, "NXT": Nextrade 대체거래소, "SOR": 최유리 통합 주문)

    def get_budget_per_split(self) -> float:
        """1회 분할당 주문 예산 산출"""
        if self.split_count <= 0:
            return self.total_budget_per_stock
        return self.total_budget_per_stock / self.split_count
