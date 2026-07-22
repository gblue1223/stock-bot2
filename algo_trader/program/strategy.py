import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List
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


def parse_hhmmss_to_seconds(tm_str: str) -> int:
    """HHMMSS 포맷 시각 문자열을 하루 기준 초(seconds) 단위로 변환합니다."""
    tm_str = str(tm_str).zfill(6)
    try:
        h = int(tm_str[:2])
        m = int(tm_str[2:4])
        s = int(tm_str[4:6])
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def get_past_item_by_minutes(valid_items: List[Dict[str, Any]], minutes: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """현재 시각 대비 정확히 N분 전 시점의 아이템 및 해당 구간 시계열 목록을 추출합니다."""
    if not valid_items:
        return None, []
    
    now_sec = parse_hhmmss_to_seconds(valid_items[0].get("tm"))
    target_sec = now_sec - minutes * 60

    past_item = valid_items[-1]
    window_items = []
    
    for item in valid_items:
        sec = parse_hhmmss_to_seconds(item.get("tm"))
        if sec >= target_sec:
            window_items.append(item)
        if sec <= target_sec:
            past_item = item
            break

    return past_item, window_items


def get_1min_binned_items(valid_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """5초 단위 틱 데이터를 1분 단위(HHMM) 최신 틱으로 그룹화하여 1분별 스냅샷 리스트를 반환합니다."""
    binned = []
    seen_minutes = set()
    for item in valid_items:
        tm = str(item.get("tm", "")).zfill(6)
        if len(tm) >= 4:
            hhmm = tm[:4]
            if hhmm not in seen_minutes:
                seen_minutes.add(hhmm)
                binned.append(item)
    return binned


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
        trend_items = prog_data.get("time_series_items", [])

        # BUY 신호 조건:
        # 1) 최근 시간대별 순매수 증감(net_buy_irds)이 지정한 양수(+) 기준 이상이며 (기본: 0 초과)
        # 2) 실제 이전 10분 전 시점 대비 현재 프로그램 순매수 누적액이 상승(net_buy_now > net_buy_10m_ago)한 경우
        if net_buy_irds > self.config.min_net_buy_trend_irds:
            buy_trend_ok = True
            diff_buy_trend = 0.0
            if len(trend_items) >= 5:
                past_item, _ = get_past_item_by_minutes(trend_items, self.config.buy_trend_window_minutes)
                if past_item:
                    net_buy_now = parse_amount(trend_items[0].get("prm_netprps_amt"))
                    net_buy_past = parse_amount(past_item.get("prm_netprps_amt"))
                    diff_buy_trend = net_buy_now - net_buy_past

                    if diff_buy_trend <= self.config.min_buy_trend_increase:
                        buy_trend_ok = False

            if buy_trend_ok:
                self.logger.info(
                    f"[{stock_code}] 🟢 BUY 시그널 감지 (실제 {self.config.buy_trend_window_minutes}분간 상승세 확인: {diff_buy_trend:+,.0f}백만원) - "
                    f"순매수: {net_buy_amt:,.0f}백만원 (증감: {net_buy_irds:,.0f})"
                )
                return ProgramSignal.BUY

        # SELL 신호 조건 1: 최근 1분간 프로그램 순매수 급감(-1,000억원 이상) 또는 누적 순매수 대량 이탈 시
        sell_limit = -abs(self.config.sell_net_buy_trend_threshold)
        if net_buy_irds <= sell_limit or net_buy_amt <= sell_limit:
            self.logger.info(f"[{stock_code}] 🔴 SELL 시그널 감지 (급격한 이탈) - 순매수: {net_buy_amt:,.0f}백만원 (증감: {net_buy_irds:,.0f})")
            return ProgramSignal.SELL

        # SELL 신호 조건 2: 10분간 완만한 하락 추세 감지 (1분 단위 스냅샷 기준 이전 대비 10% 이상 낮아지는 경우)
        window_size = self.config.trend_window_minutes
        binned_1min = get_1min_binned_items(trend_items)

        if len(binned_1min) >= 5:
            idx_10m = min(window_size, len(binned_1min) - 1)
            net_buy_now = parse_amount(binned_1min[0].get("prm_netprps_amt"))
            net_buy_10m_ago = parse_amount(binned_1min[idx_10m].get("prm_netprps_amt"))
            diff_trend = net_buy_now - net_buy_10m_ago

            # 1분 단위 연속 변동 비교 (binned[k] vs binned[k+1])
            check_count = idx_10m
            decreases_1min = 0
            for k in range(check_count):
                amt_k = parse_amount(binned_1min[k].get("prm_netprps_amt"))
                amt_prev = parse_amount(binned_1min[k + 1].get("prm_netprps_amt"))
                if amt_k < amt_prev:
                    decreases_1min += 1

            decrease_ratio_1min = decreases_1min / check_count if check_count > 0 else 0.0

            # 10분 전 대비 순매수 하락 비율(%) 계산
            pct_drop = 0.0
            if abs(net_buy_10m_ago) > 0:
                pct_drop = ((net_buy_10m_ago - net_buy_now) / abs(net_buy_10m_ago)) * 100.0

            gentle_threshold = -abs(self.config.gentle_downward_threshold)

            # 완만한 하락 판정 조건:
            # (1) 1분 단위 스냅샷 기준 10분 전 대비 순매수가 설정 비율(기본 10.0%) 이상 낮아진 경우
            # (2) 또는 1분 단위 하락 비율이 50% 이상이면서 10분간 누적 순매수 감소액이 설정 임계값 이하인 경우
            if pct_drop >= self.config.gentle_downward_pct or (diff_trend <= gentle_threshold and decrease_ratio_1min >= self.config.consecutive_decrease_ratio):
                self.logger.info(
                    f"[{stock_code}] 🔴 SELL 시그널 감지 ({window_size}분 완만 하락 추세) - "
                    f"1분 스냅샷 10분전 대비 변동: {diff_trend:,.0f}백만원 ({pct_drop:.1f}% 하락), "
                    f"1분단위 하락비율: {decrease_ratio_1min*100:.0f}% ({decreases_1min}/{check_count}분)"
                )
                return ProgramSignal.SELL

        return ProgramSignal.HOLD
