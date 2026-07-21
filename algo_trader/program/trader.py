import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime

from koapys import KoapyRestSimple, OrderType, OrderBookType
from kiwoom_rest_api.koreanstock.market_condition import MarketCondition
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo

from .config import ProgramTradingConfig
from .strategy import ProgramTradingStrategy, ProgramSignal


@dataclass
class PositionState:
    """종목별 10분할 매매 포지션 관리 데이터클래스"""
    stock_code: str
    target_steps: int = 10  # 목표 분할 수 (10회)
    executed_steps: int = 0  # 현재 집행된 분할 횟수
    holding_qty: int = 0  # 현재 보유 수량
    total_cost: float = 0.0  # 총 매수 금액
    avg_price: float = 0.0  # 평균 매수가
    is_stopped_today: bool = False  # 손절/익절로 당일 매매가 종료되었는지 여부
    last_order_time: Optional[datetime] = None  # 마지막 주문 시간


class RealtimeProgramTrader:
    """실시간 프로그램 매매 10분할 트레이더 주 실행 엔진"""

    def __init__(self, config: ProgramTradingConfig):
        self.config = config
        self.logger = logging.getLogger("RealtimeProgramTrader")
        
        # Koapys REST 클라이언트 초기화
        self.logger.info("Initializing KoapyRestSimple client...")
        self.koapys = KoapyRestSimple(simulation=False, logger=self.logger)
        
        try:
            self.koapys.ensure_connected()
            self.logger.info("✅ KoapyRestSimple connected successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect KoapyRestSimple: {e}")
            if not config.dry_run:
                raise ConnectionError(f"API Connection Failed in Live Mode: {e}")
            else:
                self.logger.warning("⚠️ Dry-Run 모드로 계속 실행합니다 (API 연결 없이 시뮬레이션).")

        # API 토큰 및 컴포넌트
        base_url = "https://api.kiwoom.com"
        token_mgr = self.koapys._token_manager
        self.market_condition = MarketCondition(base_url=base_url, token_manager=token_mgr)
        self.stock_info = StockInfo(base_url=base_url, token_manager=token_mgr)

        # 전략 분석기 초기화
        self.strategy = ProgramTradingStrategy(
            config=self.config,
            market_condition_api=self.market_condition,
            stock_info_api=self.stock_info
        )

        # 계좌 선택 (API 연결 안되어있으면 모의 계좌 사용)
        if self.config.account_no:
            self.account_no = self.config.account_no
        else:
            if self.koapys.is_connected:
                try:
                    accounts = self.koapys.get_account_list()
                    self.account_no = accounts[0] if accounts else "0000000000"
                except Exception:
                    self.account_no = "0000000000"
            else:
                self.account_no = "0000000000"
        self.logger.info(f"Using account number: {self.account_no}")

        # 종목별 포지션 트래커 초기화
        self.positions: Dict[str, PositionState] = {
            code: PositionState(stock_code=code, target_steps=config.split_count)
            for code in config.target_stocks
        }

    def check_market_close_time(self) -> bool:
        """장 마감 전 강제 청산 시간이 도달했는지 확인합니다."""
        now_str = datetime.now().strftime("%H:%M:%S")
        return now_str >= self.config.market_close_time

    def calculate_order_quantity(self, current_price: float) -> int:
        """1회 분할당 주문 금액에 알맞은 매수 수량을 계산합니다."""
        if current_price <= 0:
            return 0
        budget_per_split = self.config.get_budget_per_split()
        qty = int(budget_per_split // current_price)
        return max(1, qty)

    def execute_split_buy(self, pos: PositionState, current_price: float) -> bool:
        """1회 분할 매수를 집행합니다."""
        if pos.executed_steps >= pos.target_steps:
            self.logger.info(f"[{pos.stock_code}] 10분할 매수가 이미 완료되었습니다 ({pos.executed_steps}/{pos.target_steps}).")
            return False

        qty = self.calculate_order_quantity(current_price)
        if qty <= 0:
            self.logger.warning(f"[{pos.stock_code}] 주문 수량이 0 이하입니다. (현재가: {current_price})")
            return False

        order_amount = qty * current_price
        step_num = pos.executed_steps + 1

        self.logger.info(
            f"🚀 [{pos.stock_code}] 10분할 매수 집행 #{step_num}/{pos.target_steps} "
            f"| 수량: {qty:,}주 | 단가: {current_price:,.0f}원 | 금액: {order_amount:,.0f}원 | 거래소: {self.config.stock_exchange_type}"
        )

        # NXT 모드에서는 지정가(LIMIT) 주문, KRX/SOR 모드에서는 시장가(MARKET) 주문 적용
        is_nxt = (self.config.stock_exchange_type.upper() == "NXT")
        hoga_type = OrderBookType.LIMIT if is_nxt else OrderBookType.MARKET
        order_price = int(current_price) if is_nxt else 0

        if not self.config.dry_run and self.koapys.is_connected:
            try:
                res = self.koapys.send_order(
                    rqname=f"PGM_SPLIT_BUY_{step_num}",
                    account_no=self.account_no,
                    order_type=OrderType.BUY,
                    code=pos.stock_code,
                    quantity=qty,
                    price=order_price,
                    hoga=hoga_type,
                    dmst_stex_tp=self.config.stock_exchange_type
                )
                self.logger.info(f"[{pos.stock_code}] 주문 응답: {res}")
            except Exception as e:
                self.logger.error(f"[{pos.stock_code}] 주문 체결 에러: {e}")
                return False
        else:
            self.logger.info(f"💡 [DRY-RUN] 실제 주문은 송신되지 않았습니다. (가상 매수 수량: {qty}주)")

        # 포지션 관리 갱신
        pos.executed_steps += 1
        pos.holding_qty += qty
        pos.total_cost += order_amount
        pos.avg_price = pos.total_cost / pos.holding_qty if pos.holding_qty > 0 else 0.0
        pos.last_order_time = datetime.now()

        self.logger.info(
            f"✅ [{pos.stock_code}] 보유 현황 | 총 수량: {pos.holding_qty:,}주 | "
            f"평단가: {pos.avg_price:,.0f}원 | 총 매수금액: {pos.total_cost:,.0f}원"
        )
        return True

    def liquidate_position(self, pos: PositionState, current_price: float, reason: str):
        """보유 포지션을 전량 매도(청산)합니다."""
        if pos.holding_qty <= 0:
            return

        pnl_amount = (current_price - pos.avg_price) * pos.holding_qty
        pnl_pct = ((current_price / pos.avg_price) - 1.0) * 100.0 if pos.avg_price > 0 else 0.0

        self.logger.info(
            f"🔥 [{pos.stock_code}] 포지션 전량 청산 [{reason}] | "
            f"수량: {pos.holding_qty:,}주 | 평단가: {pos.avg_price:,.0f}원 | "
            f"청산단가: {current_price:,.0f}원 | 손익: {pnl_amount:+,.0f}원 ({pnl_pct:+.2f}%)"
        )

        is_nxt = (self.config.stock_exchange_type.upper() == "NXT")
        hoga_type = OrderBookType.LIMIT if is_nxt else OrderBookType.MARKET
        order_price = int(current_price) if is_nxt else 0

        if not self.config.dry_run and self.koapys.is_connected:
            try:
                res = self.koapys.send_order(
                    rqname=f"PGM_LIQUIDATE_{reason}",
                    account_no=self.account_no,
                    order_type=OrderType.SELL,
                    code=pos.stock_code,
                    quantity=pos.holding_qty,
                    price=order_price,
                    hoga=hoga_type,
                    dmst_stex_tp=self.config.stock_exchange_type
                )
                self.logger.info(f"[{pos.stock_code}] 청산 주문 응답: {res}")
            except Exception as e:
                self.logger.error(f"[{pos.stock_code}] 청산 주문 에러: {e}")
        else:
            self.logger.info(f"💡 [DRY-RUN] 실제 청산 주문은 송신되지 않았습니다. (가상 청산 수량: {pos.holding_qty}주)")

        # 포지션 초기화 및 당일 매매 중단 설정
        pos.holding_qty = 0
        pos.total_cost = 0.0
        pos.avg_price = 0.0
        pos.is_stopped_today = True

    def check_risk_and_liquidation(self, pos: PositionState, current_price: float):
        """손절(-2%), 익절(+3%) 및 장마감 청산 조건을 검사합니다."""
        if pos.holding_qty <= 0 or current_price <= 0:
            return

        pnl_pct = ((current_price / pos.avg_price) - 1.0) * 100.0 if pos.avg_price > 0 else 0.0

        # 1. 손절 조건 검사
        if pnl_pct <= -abs(self.config.stop_loss_pct):
            self.liquidate_position(pos, current_price, reason=f"손절 기준 도달 ({pnl_pct:.2f}%)")
            return

        # 2. 익절 조건 검사
        if pnl_pct >= abs(self.config.take_profit_pct):
            self.liquidate_position(pos, current_price, reason=f"익절 기준 도달 ({pnl_pct:.2f}%)")
            return

        # 3. 장 마감 전 강제 청산 검사
        if self.check_market_close_time():
            self.liquidate_position(pos, current_price, reason="장 마감 전 자동 청산")
            return

    def run_cycle(self):
        """1회 모니터링 및 트레이딩 사이클을 수행합니다."""
        date_str = datetime.today().strftime("%Y%m%d")
        is_close_time = self.check_market_close_time()

        for code in self.config.target_stocks:
            pos = self.positions[code]
            if pos.is_stopped_today:
                continue

            # 수급 및 현재가 데이터 조회
            prog_data = self.strategy.fetch_program_data(code, date_str)
            current_price = prog_data.get("current_price", 0.0)

            # Dry-Run 모드이면서 API 접속이 불가할 때 시뮬레이션 더미가 설정
            if current_price <= 0 and self.config.dry_run:
                mock_prices = {"005930": 55000.0, "000660": 180000.0}
                current_price = mock_prices.get(code, 70000.0)
                prog_data["current_price"] = current_price
                prog_data["net_buy_amt"] = 1500.0  # +15억 양수 수급 예시
                prog_data["net_buy_irds"] = 300.0  # +3억 수급 확대 예시

            if current_price <= 0:
                self.logger.warning(f"[{code}] 현재가를 취득하지 못했습니다.")
                continue

            # 1. 손절/익절/장마감 청산 체크
            self.check_risk_and_liquidation(pos, current_price)
            if pos.is_stopped_today:
                continue

            # 2. 수급 시그널 분석
            signal = self.strategy.analyze_signal(code, prog_data)

            # 3. 신호에 따른 분할 주문 제어
            if signal == ProgramSignal.BUY:
                # 주문 주기 간격(interval_seconds) 체크
                can_order = True
                if pos.last_order_time:
                    elapsed = (datetime.now() - pos.last_order_time).total_seconds()
                    if elapsed < self.config.interval_seconds:
                        can_order = False

                if can_order:
                    self.execute_split_buy(pos, current_price)

            elif signal == ProgramSignal.SELL and pos.holding_qty > 0:
                self.liquidate_position(pos, current_price, reason="프로그램 매도 시그널 청산")

        # 장마감 시간이 지나면 전체 트레이더 중지 처리
        if is_close_time:
            self.logger.info("⏰ 장 마감 시각에 도달하여 당일 매매 모니터링을 종료합니다.")

    def start_loop(self, max_iterations: Optional[int] = None):
        """실시간 트레이딩 모니터링 루프를 실행합니다."""
        self.logger.info("=" * 60)
        self.logger.info(f"⚡ 실시간 프로그램 매매 10분할 트레이더 시작")
        self.logger.info(f" 대상 종목: {self.config.target_stocks}")
        self.logger.info(f" 거래소 구분: {self.config.stock_exchange_type}")
        self.logger.info(f" 분할 횟수: {self.config.split_count}회 | 분할 주기: {self.config.interval_seconds}초")
        self.logger.info(f" 종목당 예산: {self.config.total_budget_per_stock:,.0f}원 (1회당: {self.config.get_budget_per_split():,.0f}원)")
        self.logger.info(f" 손절: -{self.config.stop_loss_pct}% | 익절: +{self.config.take_profit_pct}% | 청산시각: {self.config.market_close_time}")
        self.logger.info(f" 드라이런 모드: {self.config.dry_run}")
        self.logger.info("=" * 60)

        iterations = 0
        while True:
            try:
                if self.check_market_close_time():
                    self.logger.info("장 마감 시간에 도달했습니다. 루프를 종료합니다.")
                    break

                self.run_cycle()
                iterations += 1

                if max_iterations and iterations >= max_iterations:
                    self.logger.info(f"최대 테스트 횟수({max_iterations}회)에 도달하여 루프를 종료합니다.")
                    break

                time.sleep(5)  # 5초 주기로 모니터링 폴링

            except KeyboardInterrupt:
                self.logger.info("사용자에 의해 트레이딩이 중단되었습니다.")
                break
            except Exception as e:
                self.logger.error(f"루프 실행 중 에러 발생: {e}", exc_info=True)
                time.sleep(5)
