from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class ForeignInstitution(KiwoomBaseAPI):
    """한국 주식 외인 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/frgnistt"
    ):
        """
        ForeignInstitution 클래스 초기화
        
        Args:
            base_url (str, optional): API 기본 URL
            token_manager: 토큰 관리자 객체
            use_async (bool): 비동기 클라이언트 사용 여부 (기본값: False)
        """
        super().__init__(
            base_url=base_url,
            token_manager=token_manager,
            use_async=use_async,
            resource_url=resource_url
        )
        
    def foreign_investor_stockwise_trading_trend_request_ka10008(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 외국인 종목별 매매 동향을 조회합니다.

        Args:
            stock_code (str): 종목코드 (거래소별 종목코드)
                - KRX: 039490
                - NXT: 039490_NX
                - SOR: 039490_AL
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식 외국인 종목별 매매 동향 데이터
                {
                    "stk_frgnr": [
                        {
                            "dt": str,  # 일자
                            "close_pric": str,  # 종가
                            "pred_pre": str,  # 전일대비
                            "trde_qty": str,  # 거래량
                            "chg_qty": str,  # 변동수량
                            "poss_stkcnt": str,  # 보유주식수
                            "wght": str,  # 비중
                            "gain_pos_stkcnt": str,  # 취득가능주식수
                            "frgnr_limit": str,  # 외국인한도
                            "frgnr_limit_irds": str,  # 외국인한도증감
                            "limit_exh_rt": str,  # 한도소진률
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.foreign_institution.foreign_investor_stockwise_trading_trend_request_ka10008(
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10008",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def institutional_stock_request_ka10009(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 기관 요청을 조회합니다.

        Args:
            stock_code (str): 종목코드 (거래소별 종목코드)
                - KRX: 039490
                - NXT: 039490_NX
                - SOR: 039490_AL
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식 기관 요청 데이터
                {
                    "date": str,  # 날짜
                    "close_pric": str,  # 종가
                    "pre": str,  # 대비
                    "orgn_dt_acc": str,  # 기관기간누적
                    "orgn_daly_nettrde": str,  # 기관일별순매매
                    "frgnr_daly_nettrde": str,  # 외국인일별순매매
                    "frgnr_qota_rt": str,  # 외국인지분율
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.foreign_institution.institutional_stock_request_ka10009(
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10009",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def institution_foreign_consecutive_trading_status_request_ka10131(
        self,
        dt: str,
        mrkt_tp: str,
        netslmt_tp: str = "2",
        stk_inds_tp: str = "0",
        amt_qty_tp: str = "0",
        stex_tp: str = "1",
        strt_dt: str = "",
        end_dt: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """기관외국인연속매매현황을 조회합니다.

        Args:
            dt (str): 기간 (1:최근일, 3:3일, 5:5일, 10:10일, 20:20일, 120:120일, 0:시작일자/종료일자로 조회)
            mrkt_tp (str): 장구분 (001:코스피, 101:코스닥)
            netslmt_tp (str, optional): 순매도수구분 (2:순매수(고정값)). Defaults to "2".
            stk_inds_tp (str, optional): 종목업종구분 (0:종목(주식),1:업종). Defaults to "0".
            amt_qty_tp (str, optional): 금액수량구분 (0:금액, 1:수량). Defaults to "0".
            stex_tp (str, optional): 거래소구분 (1:KRX, 2:NXT, 3:통합). Defaults to "1".
            strt_dt (str, optional): 시작일자 (YYYYMMDD). Defaults to "".
            end_dt (str, optional): 종료일자 (YYYYMMDD). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 기관외국인연속매매현황 데이터
                {
                    "orgn_frgnr_cont_trde_prst": [
                        {
                            "rank": str,  # 순위
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "prid_stkpc_flu_rt": str,  # 기간중주가등락률
                            "orgn_nettrde_amt": str,  # 기관순매매금액
                            "orgn_nettrde_qty": str,  # 기관순매매량
                            "orgn_cont_netprps_dys": str,  # 기관계연속순매수일수
                            "orgn_cont_netprps_qty": str,  # 기관계연속순매수량
                            "orgn_cont_netprps_amt": str,  # 기관계연속순매수금액
                            "frgnr_nettrde_qty": str,  # 외국인순매매량
                            "frgnr_nettrde_amt": str,  # 외국인순매매액
                            "frgnr_cont_netprps_dys": str,  # 외국인연속순매수일수
                            "frgnr_cont_netprps_qty": str,  # 외국인연속순매수량
                            "frgnr_cont_netprps_amt": str,  # 외국인연속순매수금액
                            "nettrde_qty": str,  # 순매매량
                            "nettrde_amt": str,  # 순매매액
                            "tot_cont_netprps_dys": str,  # 합계연속순매수일수
                            "tot_cont_nettrde_qty": str,  # 합계연속순매매수량
                            "tot_cont_netprps_amt": str,  # 합계연속순매수금액
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.foreign_institution.institution_foreign_consecutive_trading_status_request_ka10131(
            ...     dt="1",
            ...     mrkt_tp="001"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10131",
        }

        data = {
            "dt": dt,
            "strt_dt": strt_dt,
            "end_dt": end_dt,
            "mrkt_tp": mrkt_tp,
            "netslmt_tp": netslmt_tp,
            "stk_inds_tp": stk_inds_tp,
            "amt_qty_tp": amt_qty_tp,
            "stex_tp": stex_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )