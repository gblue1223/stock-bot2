import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

from .config import ProgramTradingConfig


class ProgramSignal(Enum):
    BUY = "BUY"        # 프로그램 매수세 유입 -> 분할 매수 진입
    SELL = "SELL"      # 프로그램 매도세 유입 -> 청산 또는 분할 매도
    HOLD = "HOLD"      # 수급 관망 / 변화 없음


def parse_amount(val: Any) -> float:
    """키움 API의 부호 기호(+, --, -) 포함 수치 문자열을 float으로 파싱합니다."""
    if val is None:
        return 0.0
    val_str = str(val).strip()
    if not val_str:
        return 0.0
    if val_str.startswith('--'):
        val_str = '-' + val_str[2:]
    val_str = val_str.replace('+-', '-').replace('-+', '-')
    if val_str.startswith('+'):
        val_str = val_str[1:]
    try:
        return float(val_str)
    except ValueError:
        return 0.0


class ProgramTradingStrategy:
    """실시간 프로그램 매매 동향 기반 시그널 생성기"""

    def __init__(self, config: ProgramTradingConfig, market_condition_api: Any, stock_info_api: Any):
        self.config = config
        self.market_condition = market_condition_api
        self.stock_info = stock_info_api
        self.logger = logging.getLogger("ProgramTradingStrategy")

    def fetch_program_data(self, stock_code: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """종목의 실시간/장중 시간대별 및 당일 프로그램 매매 동향을 조회합니다."""
        if date_str is None:
            date_str = datetime.today().strftime("%Y%m%d")

        result_data = {
            "stock_code": stock_code,
            "date": date_str,
            "net_buy_amt": 0.0,
            "net_buy_qty": 0.0,
            "net_buy_irds": 0.0,
            "sell_amt": 0.0,
            "buy_amt": 0.0,
            "current_price": 0.0,
            "time_str": "",
            "raw_time_data": None,
            "time_series_items": []
        }

        # 1. 시간대별 프로그램 매매 추이 요청 (ka90008)
        try:
            time_res = self.market_condition.stockwise_program_trading_by_hour_request_ka90008(
                amount_quantity_type=self.config.amount_quantity_type,
                stock_code=stock_code,
                date=date_str
            )
            if str(time_res.get("return_code", "-1")) == "0":
                items = time_res.get("stk_tm_prm_trde_trnsn", [])
                valid_items = [it for it in items if it.get("tm")]
                if valid_items:
                    latest = valid_items[0]  # 최신 체결 시각 데이터
                    result_data["time_str"] = latest.get("tm", "")
                    result_data["current_price"] = abs(parse_amount(latest.get("cur_prc")))
                    result_data["net_buy_amt"] = parse_amount(latest.get("prm_netprps_amt"))
                    result_data["net_buy_irds"] = parse_amount(latest.get("prm_netprps_amt_irds"))
                    result_data["sell_amt"] = parse_amount(latest.get("prm_sell_amt"))
                    result_data["buy_amt"] = parse_amount(latest.get("prm_buy_amt"))
                    result_data["net_buy_qty"] = parse_amount(latest.get("prm_netprps_qty"))
                    result_data["raw_time_data"] = latest
                    result_data["time_series_items"] = valid_items
        except Exception as e:
            self.logger.warning(f"[{stock_code}] ka90008 시간대별 프로그램 매매 조회 실패: {e}")

        # 2. 만약 시간대별 데이터가 아직 없거나 주말/장 시작 전인 경우, 당일 종목별 프로그램 매매 상태 (ka90004) 및 기본 정보 (ka10001) 활용
        if result_data["current_price"] == 0.0:
            try:
                stock_res = self.stock_info.basic_stock_information_request_ka10001(stock_code)
                if str(stock_res.get("return_code", "-1")) == "0":
                    price_str = stock_res.get("stk_prc") or stock_res.get("cur_prc") or "0"
                    result_data["current_price"] = abs(parse_amount(price_str))
            except Exception as e:
                self.logger.warning(f"[{stock_code}] ka10001 주식기본정보 조회 실패: {e}")

        return result_data

    def analyze_signal(self, stock_code: str, prog_data: Dict[str, Any]) -> ProgramSignal:
        """프로그램 매매 동향 수치를 분석하여 매수/매도/관망 시그널을 판정합니다."""
        net_buy_amt = prog_data.get("net_buy_amt", 0.0)
        net_buy_irds = prog_data.get("net_buy_irds", 0.0)

        # BUY 신호 조건:
        # 1) 프로그램 누적 순매수 금액이 최소 임계값(min_net_buy_amount) 이상이고
        # 2) 최근 시간대별 순매수 증감(net_buy_irds)이 지정한 양수(+) 기준 이상인 경우 (기본 0 초과)
        if net_buy_amt >= self.config.min_net_buy_amount and net_buy_irds > self.config.min_net_buy_trend_irds:
            self.logger.info(f"[{stock_code}] 🟢 BUY 시그널 감지 - 순매수: {net_buy_amt:,.0f}백만원 (증감: {net_buy_irds:,.0f})")
            return ProgramSignal.BUY

        # SELL 신호 조건 1: 최근 1분간 프로그램 순매수 급감(-1,000억원 이상) 또는 누적 순매수 대량 이탈 시
        sell_limit = -abs(self.config.sell_net_buy_trend_threshold)
        if net_buy_irds <= sell_limit or net_buy_amt <= sell_limit:
            self.logger.info(f"[{stock_code}] 🔴 SELL 시그널 감지 (급격한 이탈) - 순매수: {net_buy_amt:,.0f}백만원 (증감: {net_buy_irds:,.0f})")
            return ProgramSignal.SELL

        # SELL 신호 조건 2: 10분간 완만한 하락 추세 감지
        trend_items = prog_data.get("time_series_items", [])
        window_size = self.config.trend_window_minutes
        if len(trend_items) >= 5:  # 최소 5분 이상 시계열 데이터 누적 시
            idx_past = min(window_size, len(trend_items) - 1)
            net_buy_now = parse_amount(trend_items[0].get("prm_netprps_amt"))
            net_buy_past = parse_amount(trend_items[idx_past].get("prm_netprps_amt"))
            diff_trend = net_buy_now - net_buy_past

            # 시계열 내 감소 발생 구간 수 카운트
            check_count = min(window_size, len(trend_items) - 1)
            decreases = 0
            for k in range(check_count):
                irds = parse_amount(trend_items[k].get("prm_netprps_amt_irds"))
                if irds < 0:
                    decreases += 1

            decrease_ratio = decreases / check_count if check_count > 0 else 0.0
            gentle_threshold = -abs(self.config.gentle_downward_threshold)

            # 10분간 누적 순매수액 감소량이 완만 하락 감도 기준 이하이고, 하락 구간 비율이 60% 이상인 경우
            if diff_trend <= gentle_threshold and decrease_ratio >= self.config.consecutive_decrease_ratio:
                self.logger.info(
                    f"[{stock_code}] 🔴 SELL 시그널 감지 ({window_size}분 완만 하락 추세) - "
                    f"{window_size}분간 순매수 변동: {diff_trend:,.0f}백만원, "
                    f"하락비율: {decrease_ratio*100:.0f}% ({decreases}/{check_count}분)"
                )
                return ProgramSignal.SELL

        return ProgramSignal.HOLD
