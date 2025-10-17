from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class Account(KiwoomBaseAPI):
    """한국 주식 계좌 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/acnt"
    ):
        """
        Account 클래스 초기화
        
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
        
    def realized_profit_by_date_stock_request_ka10072(
        self,
        stock_code: str,
        start_date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        일자별종목별실현손익요청 (ka10072)

        Args:
            stock_code (str): 종목코드 (6자리)
            start_date (str): 시작일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 일자별종목별실현손익 데이터
                {
                    "dt_stk_div_rlzt_pl": [
                        {
                            "stk_nm": str,  # 종목명
                            "cntr_qty": str,  # 체결량
                            "buy_uv": str,  # 매입단가
                            "cntr_pric": str,  # 체결가
                            "tdy_sel_pl": str,  # 당일매도손익
                            "pl_rt": str,  # 손익율
                            "stk_cd": str,  # 종목코드
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "wthd_alowa": str,  # 인출가능금액
                            "loan_dt": str,  # 대출일
                            "crd_tp": str,  # 신용구분
                            "stk_cd_1": str,  # 종목코드1
                            "tdy_sel_pl_1": str,  # 당일매도손익1
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.realized_profit_by_date_stock_request_ka10072(
            ...     stock_code="005930",
            ...     start_date="20241128"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10072",
        }
        data = {
            "stk_cd": stock_code,
            "strt_dt": start_date,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def realized_profit_by_period_stock_request_ka10073(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        일자별종목별실현손익요청_기간 (ka10073)

        Args:
            stock_code (str): 종목코드 (6자리)
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 일자별종목별실현손익 데이터
                {
                    "dt_stk_rlzt_pl": [
                        {
                            "dt": str,  # 일자
                            "tdy_htssel_cmsn": str,  # 당일hts매도수수료
                            "stk_nm": str,  # 종목명
                            "cntr_qty": str,  # 체결량
                            "buy_uv": str,  # 매입단가
                            "cntr_pric": str,  # 체결가
                            "tdy_sel_pl": str,  # 당일매도손익
                            "pl_rt": str,  # 손익율
                            "stk_cd": str,  # 종목코드
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "wthd_alowa": str,  # 인출가능금액
                            "loan_dt": str,  # 대출일
                            "crd_tp": str,  # 신용구분
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.realized_profit_by_period_stock_request_ka10073(
            ...     stock_code="005930",
            ...     start_date="20241128",
            ...     end_date="20241128"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10073",
        }
        data = {
            "stk_cd": stock_code,
            "strt_dt": start_date,
            "end_dt": end_date,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def daily_realized_profit_request_ka10074(
        self,
        start_date: str,
        end_date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        일자별실현손익요청 (ka10074)

        Args:
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 일자별실현손익 데이터
                {
                    "tot_buy_amt": str,  # 총매수금액
                    "tot_sell_amt": str,  # 총매도금액
                    "rlzt_pl": str,  # 실현손익
                    "trde_cmsn": str,  # 매매수수료
                    "trde_tax": str,  # 매매세금
                    "dt_rlzt_pl": [
                        {
                            "dt": str,  # 일자
                            "buy_amt": str,  # 매수금액
                            "sell_amt": str,  # 매도금액
                            "tdy_sel_pl": str,  # 당일매도손익
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.daily_realized_profit_request_ka10074(
            ...     start_date="20241128",
            ...     end_date="20241128"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10074",
        }
        data = {
            "strt_dt": start_date,
            "end_dt": end_date,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def unfilled_orders_request_ka10075(
        self,
        all_stk_tp: str,
        trde_tp: str,
        stex_tp: str,
        stock_code: str = None,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        미체결요청 (ka10075)

        Args:
            all_stk_tp (str): 전체종목구분 (0:전체, 1:종목)
            trde_tp (str): 매매구분 (0:전체, 1:매도, 2:매수)
            stex_tp (str): 거래소구분 (0:통합, 1:KRX, 2:NXT)
            stock_code (str, optional): 종목코드 (6자리). Required when all_stk_tp is "1".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 미체결 데이터
                {
                    "oso": [
                        {
                            "acnt_no": str,  # 계좌번호
                            "ord_no": str,  # 주문번호
                            "mang_empno": str,  # 관리사번
                            "stk_cd": str,  # 종목코드
                            "tsk_tp": str,  # 업무구분
                            "ord_stt": str,  # 주문상태
                            "stk_nm": str,  # 종목명
                            "ord_qty": str,  # 주문수량
                            "ord_pric": str,  # 주문가격
                            "oso_qty": str,  # 미체결수량
                            "cntr_tot_amt": str,  # 체결누계금액
                            "orig_ord_no": str,  # 원주문번호
                            "io_tp_nm": str,  # 주문구분
                            "trde_tp": str,  # 매매구분
                            "tm": str,  # 시간
                            "cntr_no": str,  # 체결번호
                            "cntr_pric": str,  # 체결가
                            "cntr_qty": str,  # 체결량
                            "cur_prc": str,  # 현재가
                            "sel_bid": str,  # 매도호가
                            "buy_bid": str,  # 매수호가
                            "unit_cntr_pric": str,  # 단위체결가
                            "unit_cntr_qty": str,  # 단위체결량
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "ind_invsr": str,  # 개인투자자
                            "stex_tp": str,  # 거래소구분
                            "stex_tp_txt": str,  # 거래소구분텍스트
                            "sor_yn": str,  # SOR 여부값
                            "stop_pric": str,  # 스톱가
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.unfilled_orders_request_ka10075(
            ...     all_stk_tp="1",
            ...     trde_tp="0",
            ...     stex_tp="0",
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10075",
        }
        data = {
            "all_stk_tp": all_stk_tp,
            "trde_tp": trde_tp,
            "stex_tp": stex_tp,
        }
        if stock_code:
            data["stk_cd"] = stock_code
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def filled_orders_request_ka10076(
        self,
        qry_tp: str,
        sell_tp: str,
        stex_tp: str,
        stock_code: str = None,
        order_no: str = None,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        체결요청 (ka10076)

        Args:
            qry_tp (str): 조회구분 (0:전체, 1:종목)
            sell_tp (str): 매도수구분 (0:전체, 1:매도, 2:매수)
            stex_tp (str): 거래소구분 (0:통합, 1:KRX, 2:NXT)
            stock_code (str, optional): 종목코드 (6자리). Required when qry_tp is "1".
            order_no (str, optional): 주문번호. 검색 기준 값으로 입력한 주문번호 보다 과거에 체결된 내역이 조회됩니다.
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 체결 데이터
                {
                    "cntr": [
                        {
                            "ord_no": str,  # 주문번호
                            "stk_nm": str,  # 종목명
                            "io_tp_nm": str,  # 주문구분
                            "ord_pric": str,  # 주문가격
                            "ord_qty": str,  # 주문수량
                            "cntr_pric": str,  # 체결가
                            "cntr_qty": str,  # 체결량
                            "oso_qty": str,  # 미체결수량
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "ord_stt": str,  # 주문상태
                            "trde_tp": str,  # 매매구분
                            "orig_ord_no": str,  # 원주문번호
                            "ord_tm": str,  # 주문시간
                            "stk_cd": str,  # 종목코드
                            "stex_tp": str,  # 거래소구분
                            "stex_tp_txt": str,  # 거래소구분텍스트
                            "sor_yn": str,  # SOR 여부값
                            "stop_pric": str,  # 스톱가
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.filled_orders_request_ka10076(
            ...     qry_tp="1",
            ...     sell_tp="0",
            ...     stex_tp="0",
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10076",
        }
        data = {
            "qry_tp": qry_tp,
            "sell_tp": sell_tp,
            "stex_tp": stex_tp,
        }
        if stock_code:
            data["stk_cd"] = stock_code
        if order_no:
            data["ord_no"] = order_no
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def today_realized_profit_detail_request_ka10077(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        당일실현손익상세요청 (ka10077)

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 당일실현손익 상세 데이터
                {
                    "tdy_rlzt_pl": str,  # 당일실현손익
                    "tdy_rlzt_pl_dtl": [
                        {
                            "stk_nm": str,  # 종목명
                            "cntr_qty": str,  # 체결량
                            "buy_uv": str,  # 매입단가
                            "cntr_pric": str,  # 체결가
                            "tdy_sel_pl": str,  # 당일매도손익
                            "pl_rt": str,  # 손익율
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "stk_cd": str,  # 종목코드
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.today_realized_profit_detail_request_ka10077(
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10077",
        }
        data = {
            "stk_cd": stock_code,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def account_return_rate_request_ka10085(
        self,
        stex_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌수익률요청 (ka10085)

        Args:
            stex_tp (str): 거래소구분 (0:통합, 1:KRX, 2:NXT)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌수익률 데이터
                {
                    "acnt_prft_rt": [
                        {
                            "dt": str,  # 일자
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pur_pric": str,  # 매입가
                            "pur_amt": str,  # 매입금액
                            "rmnd_qty": str,  # 보유수량
                            "tdy_sel_pl": str,  # 당일매도손익
                            "tdy_trde_cmsn": str,  # 당일매매수수료
                            "tdy_trde_tax": str,  # 당일매매세금
                            "crd_tp": str,  # 신용구분
                            "loan_dt": str,  # 대출일
                            "setl_remn": str,  # 결제잔고
                            "clrn_alow_qty": str,  # 청산가능수량
                            "crd_amt": str,  # 신용금액
                            "crd_int": str,  # 신용이자
                            "expr_dt": str,  # 만기일
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.account_return_rate_request_ka10085(
            ...     stex_tp="0"  # 통합 거래소
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10085",
        }
        data = {
            "stex_tp": stex_tp,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def unfilled_split_order_detail_request_ka10088(
        self,
        order_no: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        미체결 분할주문 상세 요청 (ka10088)

        Args:
            order_no (str): 주문번호 (20자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 미체결 분할주문 상세 데이터
                {
                    "osop": [
                        {
                            "stk_cd": str,  # 종목코드
                            "acnt_no": str,  # 계좌번호
                            "stk_nm": str,  # 종목명
                            "ord_no": str,  # 주문번호
                            "ord_qty": str,  # 주문수량
                            "ord_pric": str,  # 주문가격
                            "osop_qty": str,  # 미체결수량
                            "io_tp_nm": str,  # 주문구분
                            "trde_tp": str,  # 매매구분
                            "sell_tp": str,  # 매도/수 구분
                            "cntr_qty": str,  # 체결량
                            "ord_stt": str,  # 주문상태
                            "cur_prc": str,  # 현재가
                            "stex_tp": str,  # 거래소구분 (0:통합, 1:KRX, 2:NXT)
                            "stex_tp_txt": str,  # 거래소구분텍스트 (통합,KRX,NXT)
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.unfilled_split_order_detail_request_ka10088(
            ...     order_no="8"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10088",
        }
        data = {
            "ord_no": order_no,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def today_trading_journal_request_ka10170(
        self,
        ottks_tp: str,
        ch_crd_tp: str,
        base_dt: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        당일매매일지 요청 (ka10170)

        Args:
            ottks_tp (str): 단주구분 (1:당일매수에 대한 당일매도, 2:당일매도 전체)
            ch_crd_tp (str): 현금신용구분 (0:전체, 1:현금매매만, 2:신용매매만)
            base_dt (str, optional): 기준일자 (YYYYMMDD). 공백입력시 금일데이터, 최근 2개월까지 제공. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 당일매매일지 데이터
                {
                    "tot_sell_amt": str,  # 총매도금액
                    "tot_buy_amt": str,  # 총매수금액
                    "tot_cmsn_tax": str,  # 총수수료_세금
                    "tot_exct_amt": str,  # 총정산금액
                    "tot_pl_amt": str,  # 총손익금액
                    "tot_prft_rt": str,  # 총수익률
                    "tdy_trde_diary": [  # 당일매매일지
                        {
                            "stk_nm": str,  # 종목명
                            "buy_avg_pric": str,  # 매수평균가
                            "buy_qty": str,  # 매수수량
                            "sel_avg_pric": str,  # 매도평균가
                            "sell_qty": str,  # 매도수량
                            "cmsn_alm_tax": str,  # 수수료_제세금
                            "pl_amt": str,  # 손익금액
                            "sell_amt": str,  # 매도금액
                            "buy_amt": str,  # 매수금액
                            "prft_rt": str,  # 수익률
                            "stk_cd": str,  # 종목코드
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.today_trading_journal_request_ka10170(
            ...     ottks_tp="1",
            ...     ch_crd_tp="0",
            ...     base_dt="20241120"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10170",
        }
        data = {
            "ottks_tp": ottks_tp,
            "ch_crd_tp": ch_crd_tp,
        }
        if base_dt:
            data["base_dt"] = base_dt
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
    
    def deposit_detail_status_request_kt00001(
        self,
        qry_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        예수금상세현황 요청 (kt00001)

        Args:
            qry_tp (str): 조회구분 (3:추정조회, 2:일반조회)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 예수금상세현황 데이터
                {
                    "entr": str,  # 예수금
                    "profa_ch": str,  # 주식증거금현금
                    "bncr_profa_ch": str,  # 수익증권증거금현금
                    "nxdy_bncr_sell_exct": str,  # 익일수익증권매도정산대금
                    "fc_stk_krw_repl_set_amt": str,  # 해외주식원화대용설정금
                    "crd_grnta_ch": str,  # 신용보증금현금
                    "crd_grnt_ch": str,  # 신용담보금현금
                    "add_grnt_ch": str,  # 추가담보금현금
                    "etc_profa": str,  # 기타증거금
                    "uncl_stk_amt": str,  # 미수확보금
                    "shrts_prica": str,  # 공매도대금
                    "crd_set_grnta": str,  # 신용설정평가금
                    "chck_ina_amt": str,  # 수표입금액
                    "etc_chck_ina_amt": str,  # 기타수표입금액
                    "crd_grnt_ruse": str,  # 신용담보재사용
                    "knx_asset_evltv": str,  # 코넥스기본예탁금
                    "elwdpst_evlta": str,  # ELW예탁평가금
                    "crd_ls_rght_frcs_amt": str,  # 신용대주권리예정금액
                    "lvlh_join_amt": str,  # 생계형가입금액
                    "lvlh_trns_alowa": str,  # 생계형입금가능금액
                    "repl_amt": str,  # 대용금평가금액(합계)
                    "remn_repl_evlta": str,  # 잔고대용평가금액
                    "trst_remn_repl_evlta": str,  # 위탁대용잔고평가금액
                    "bncr_remn_repl_evlta": str,  # 수익증권대용평가금액
                    "profa_repl": str,  # 위탁증거금대용
                    "crd_grnta_repl": str,  # 신용보증금대용
                    "crd_grnt_repl": str,  # 신용담보금대용
                    "add_grnt_repl": str,  # 추가담보금대용
                    "rght_repl_amt": str,  # 권리대용금
                    "pymn_alow_amt": str,  # 출금가능금액
                    "wrap_pymn_alow_amt": str,  # 랩출금가능금액
                    "ord_alow_amt": str,  # 주문가능금액
                    "bncr_buy_alowa": str,  # 수익증권매수가능금액
                    "20stk_ord_alow_amt": str,  # 20%종목주문가능금액
                    "30stk_ord_alow_amt": str,  # 30%종목주문가능금액
                    "40stk_ord_alow_amt": str,  # 40%종목주문가능금액
                    "100stk_ord_alow_amt": str,  # 100%종목주문가능금액
                    "ch_uncla": str,  # 현금미수금
                    "ch_uncla_dlfe": str,  # 현금미수연체료
                    "ch_uncla_tot": str,  # 현금미수금합계
                    "crd_int_npay": str,  # 신용이자미납
                    "int_npay_amt_dlfe": str,  # 신용이자미납연체료
                    "int_npay_amt_tot": str,  # 신용이자미납합계
                    "etc_loana": str,  # 기타대여금
                    "etc_loana_dlfe": str,  # 기타대여금연체료
                    "etc_loan_tot": str,  # 기타대여금합계
                    "nrpy_loan": str,  # 미상환융자금
                    "loan_sum": str,  # 융자금합계
                    "ls_sum": str,  # 대주금합계
                    "crd_grnt_rt": str,  # 신용담보비율
                    "mdstrm_usfe": str,  # 중도이용료
                    "min_ord_alow_yn": str,  # 최소주문가능금액
                    "loan_remn_evlt_amt": str,  # 대출총평가금액
                    "dpst_grntl_remn": str,  # 예탁담보대출잔고
                    "sell_grntl_remn": str,  # 매도담보대출잔고
                    "d1_entra": str,  # d+1추정예수금
                    "d1_slby_exct_amt": str,  # d+1매도매수정산금
                    "d1_buy_exct_amt": str,  # d+1매수정산금
                    "d1_out_rep_mor": str,  # d+1미수변제소요금
                    "d1_sel_exct_amt": str,  # d+1매도정산금
                    "d1_pymn_alow_amt": str,  # d+1출금가능금액
                    "d2_entra": str,  # d+2추정예수금
                    "d2_slby_exct_amt": str,  # d+2매도매수정산금
                    "d2_buy_exct_amt": str,  # d+2매수정산금
                    "d2_out_rep_mor": str,  # d+2미수변제소요금
                    "d2_sel_exct_amt": str,  # d+2매도정산금
                    "d2_pymn_alow_amt": str,  # d+2출금가능금액
                    "50stk_ord_alow_amt": str,  # 50%종목주문가능금액
                    "60stk_ord_alow_amt": str,  # 60%종목주문가능금액
                    "stk_entr_prst": [  # 종목별예수금
                        {
                            "crnc_cd": str,  # 통화코드
                            "fx_entr": str,  # 외화예수금
                            "fc_krw_repl_evlta": str,  # 원화대용평가금
                            "fc_trst_profa": str,  # 해외주식증거금
                            "pymn_alow_amt": str,  # 출금가능금액
                            "pymn_alow_amt_entr": str,  # 출금가능금액(예수금)
                            "ord_alow_amt_entr": str,  # 주문가능금액(예수금)
                            "fc_uncla": str,  # 외화미수(합계)
                            "fc_ch_uncla": str,  # 외화현금미수금
                            "dly_amt": str,  # 연체료
                            "d1_fx_entr": str,  # d+1외화예수금
                            "d2_fx_entr": str,  # d+2외화예수금
                            "d3_fx_entr": str,  # d+3외화예수금
                            "d4_fx_entr": str,  # d+4외화예수금
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.deposit_detail_status_request_kt00001(
            ...     qry_tp="3"  # 추정조회
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00001",
        }
        data = {
            "qry_tp": qry_tp,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def daily_estimated_deposit_asset_status_request_kt00002(
        self,
        start_dt: str,
        end_dt: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        일별추정예탁자산현황 요청 (kt00002)

        Args:
            start_dt (str): 시작조회기간 (YYYYMMDD)
            end_dt (str): 종료조회기간 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 일별추정예탁자산현황 데이터
                {
                    "daly_prsm_dpst_aset_amt_prst": [  # 일별추정예탁자산현황
                        {
                            "dt": str,  # 일자
                            "entr": str,  # 예수금
                            "grnt_use_amt": str,  # 담보대출금
                            "crd_loan": str,  # 신용융자금
                            "ls_grnt": str,  # 대주담보금
                            "repl_amt": str,  # 대용금
                            "prsm_dpst_aset_amt": str,  # 추정예탁자산
                            "prsm_dpst_aset_amt_bncr_skip": str,  # 추정예탁자산수익증권제외
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.daily_estimated_deposit_asset_status_request_kt00002(
            ...     start_dt="20241111",
            ...     end_dt="20241125"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00002",
        }
        data = {
            "start_dt": start_dt,
            "end_dt": end_dt,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def estimated_asset_inquiry_request_kt00003(
        self,
        qry_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        추정자산조회요청 (kt00003)

        Args:
            qry_tp (str): 상장폐지조회구분 (0:전체, 1:상장폐지종목제외)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 추정자산 데이터
                {
                    "prsm_dpst_aset_amt": str,  # 추정예탁자산
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.estimated_asset_inquiry_request_kt00003(
            ...     qry_tp="0"  # 전체 조회
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00003",
        }
        data = {
            "qry_tp": qry_tp,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def account_evaluation_status_request_kt00004(
        self,
        qry_tp: str,
        dmst_stex_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌평가현황요청 (kt00004)

        Args:
            qry_tp (str): 상장폐지조회구분 (0:전체, 1:상장폐지종목제외)
            dmst_stex_tp (str): 국내거래소구분 (KRX:한국거래소, NXT:넥스트트레이드)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌평가현황 데이터
                {
                    "acnt_nm": str,  # 계좌명
                    "brch_nm": str,  # 지점명
                    "entr": str,  # 예수금
                    "d2_entra": str,  # D+2추정예수금
                    "tot_est_amt": str,  # 유가잔고평가액
                    "aset_evlt_amt": str,  # 예탁자산평가액
                    "tot_pur_amt": str,  # 총매입금액
                    "prsm_dpst_aset_amt": str,  # 추정예탁자산
                    "tot_grnt_sella": str,  # 매도담보대출금
                    "tdy_lspft_amt": str,  # 당일투자원금
                    "invt_bsamt": str,  # 당월투자원금
                    "lspft_amt": str,  # 누적투자원금
                    "tdy_lspft": str,  # 당일투자손익
                    "lspft2": str,  # 당월투자손익
                    "lspft": str,  # 누적투자손익
                    "tdy_lspft_rt": str,  # 당일손익율
                    "lspft_ratio": str,  # 당월손익율
                    "lspft_rt": str,  # 누적손익율
                    "stk_acnt_evlt_prst": [  # 종목별계좌평가현황
                        {
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "rmnd_qty": str,  # 보유수량
                            "avg_prc": str,  # 평균단가
                            "cur_prc": str,  # 현재가
                            "evlt_amt": str,  # 평가금액
                            "pl_amt": str,  # 손익금액
                            "pl_rt": str,  # 손익율
                            "loan_dt": str,  # 대출일
                            "pur_amt": str,  # 매입금액
                            "setl_remn": str,  # 결제잔고
                            "pred_buyq": str,  # 전일매수수량
                            "pred_sellq": str,  # 전일매도수량
                            "tdy_buyq": str,  # 금일매수수량
                            "tdy_sellq": str,  # 금일매도수량
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.account_evaluation_status_request_kt00004(
            ...     qry_tp="0",  # 전체 조회
            ...     dmst_stex_tp="KRX"  # 한국거래소
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00004",
        }
        data = {
            "qry_tp": qry_tp,
            "dmst_stex_tp": dmst_stex_tp,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def filled_position_request_kt00005(
        self,
        dmst_stex_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        체결잔고요청 (kt00005)

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX:한국거래소, NXT:넥스트트레이드)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 체결잔고 데이터
                {
                    "entr": str,  # 예수금
                    "entr_d1": str,  # 예수금D+1
                    "entr_d2": str,  # 예수금D+2
                    "pymn_alow_amt": str,  # 출금가능금액
                    "uncl_stk_amt": str,  # 미수확보금
                    "repl_amt": str,  # 대용금
                    "rght_repl_amt": str,  # 권리대용금
                    "ord_alowa": str,  # 주문가능현금
                    "ch_uncla": str,  # 현금미수금
                    "crd_int_npay_gold": str,  # 신용이자미납금
                    "etc_loana": str,  # 기타대여금
                    "nrpy_loan": str,  # 미상환융자금
                    "profa_ch": str,  # 증거금현금
                    "repl_profa": str,  # 증거금대용
                    "stk_buy_tot_amt": str,  # 주식매수총액
                    "evlt_amt_tot": str,  # 평가금액합계
                    "tot_pl_tot": str,  # 총손익합계
                    "tot_pl_rt": str,  # 총손익률
                    "tot_re_buy_alowa": str,  # 총재매수가능금액
                    "20ord_alow_amt": str,  # 20%주문가능금액
                    "30ord_alow_amt": str,  # 30%주문가능금액
                    "40ord_alow_amt": str,  # 40%주문가능금액
                    "50ord_alow_amt": str,  # 50%주문가능금액
                    "60ord_alow_amt": str,  # 60%주문가능금액
                    "100ord_alow_amt": str,  # 100%주문가능금액
                    "crd_loan_tot": str,  # 신용융자합계
                    "crd_loan_ls_tot": str,  # 신용융자대주합계
                    "crd_grnt_rt": str,  # 신용담보비율
                    "dpst_grnt_use_amt_amt": str,  # 예탁담보대출금액
                    "grnt_loan_amt": str,  # 매도담보대출금액
                    "stk_cntr_remn": [  # 종목별체결잔고
                        {
                            "crd_tp": str,  # 신용구분
                            "loan_dt": str,  # 대출일
                            "expr_dt": str,  # 만기일
                            "stk_cd": str,  # 종목번호
                            "stk_nm": str,  # 종목명
                            "setl_remn": str,  # 결제잔고
                            "cur_qty": str,  # 현재잔고
                            "cur_prc": str,  # 현재가
                            "buy_uv": str,  # 매입단가
                            "pur_amt": str,  # 매입금액
                            "evlt_amt": str,  # 평가금액
                            "evltv_prft": str,  # 평가손익
                            "pl_rt": str,  # 손익률
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.filled_position_request_kt00005(
            ...     dmst_stex_tp="KRX"  # 한국거래소
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00005",
        }
        data = {
            "dmst_stex_tp": dmst_stex_tp,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def account_order_execution_detail_request_kt00007(
        self,
        qry_tp: str,
        stk_bond_tp: str,
        sell_tp: str,
        dmst_stex_tp: str,
        ord_dt: str = "",
        stock_code: str = "",
        fr_ord_no: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌별주문체결내역상세요청 (kt00007)

        Args:
            qry_tp (str): 조회구분 (1:주문순, 2:역순, 3:미체결, 4:체결내역만)
            stk_bond_tp (str): 주식채권구분 (0:전체, 1:주식, 2:채권)
            sell_tp (str): 매도수구분 (0:전체, 1:매도, 2:매수)
            dmst_stex_tp (str): 국내거래소구분 (%:전체, KRX:한국거래소, NXT:넥스트트레이드, SOR:최선주문집행)
            ord_dt (str, optional): 주문일자 (YYYYMMDD). Defaults to "".
            stock_code (str, optional): 종목코드 (12자리). 공백일때 전체종목. Defaults to "".
            fr_ord_no (str, optional): 시작주문번호 (7자리). 공백일때 전체주문. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌별주문체결내역상세 데이터
                {
                    "acnt_ord_cntr_prps_dtl": [  # 계좌별주문체결내역상세
                        {
                            "ord_no": str,  # 주문번호
                            "stk_cd": str,  # 종목번호
                            "trde_tp": str,  # 매매구분
                            "crd_tp": str,  # 신용구분
                            "ord_qty": str,  # 주문수량
                            "ord_uv": str,  # 주문단가
                            "cnfm_qty": str,  # 확인수량
                            "acpt_tp": str,  # 접수구분
                            "rsrv_tp": str,  # 반대여부
                            "ord_tm": str,  # 주문시간
                            "ori_ord": str,  # 원주문
                            "stk_nm": str,  # 종목명
                            "io_tp_nm": str,  # 주문구분
                            "loan_dt": str,  # 대출일
                            "cntr_qty": str,  # 체결수량
                            "cntr_uv": str,  # 체결단가
                            "ord_remnq": str,  # 주문잔량
                            "comm_ord_tp": str,  # 통신구분
                            "mdfy_cncl": str,  # 정정취소
                            "cnfm_tm": str,  # 확인시간
                            "dmst_stex_tp": str,  # 국내거래소구분
                            "cond_uv": str,  # 스톱가
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.account_order_execution_detail_request_kt00007(
            ...     qry_tp="1",  # 주문순
            ...     stk_bond_tp="0",  # 전체
            ...     sell_tp="0",  # 전체
            ...     dmst_stex_tp="%",  # 전체
            ...     stock_code="005930"  # 삼성전자
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00007",
        }
        data = {
            "qry_tp": qry_tp,
            "stk_bond_tp": stk_bond_tp,
            "sell_tp": sell_tp,
            "dmst_stex_tp": dmst_stex_tp,
        }
        
        if ord_dt:
            data["ord_dt"] = ord_dt
        if stock_code:
            data["stk_cd"] = stock_code
        if fr_ord_no:
            data["fr_ord_no"] = fr_ord_no
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def next_day_settlement_schedule_request_kt00008(
        self,
        strt_dcd_seq: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌별익일결제예정내역요청 (kt00008)

        Args:
            strt_dcd_seq (str, optional): 시작결제번호 (7자리). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌별익일결제예정내역 데이터
                {
                    "trde_dt": str,  # 매매일자
                    "setl_dt": str,  # 결제일자
                    "sell_amt_sum": str,  # 매도정산합
                    "buy_amt_sum": str,  # 매수정산합
                    "acnt_nxdy_setl_frcs_prps_array": [  # 계좌별익일결제예정내역배열
                        {
                            "seq": str,  # 일련번호
                            "stk_cd": str,  # 종목번호
                            "loan_dt": str,  # 대출일
                            "qty": str,  # 수량
                            "engg_amt": str,  # 약정금액
                            "cmsn": str,  # 수수료
                            "incm_tax": str,  # 소득세
                            "rstx": str,  # 농특세
                            "stk_nm": str,  # 종목명
                            "sell_tp": str,  # 매도수구분
                            "unp": str,  # 단가
                            "exct_amt": str,  # 정산금액
                            "trde_tax": str,  # 거래세
                            "resi_tax": str,  # 주민세
                            "crd_tp": str,  # 신용구분
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.next_day_settlement_schedule_request_kt00008()
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00008",
        }
        data = {}
        
        if strt_dcd_seq:
            data["strt_dcd_seq"] = strt_dcd_seq
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def account_order_execution_status_request_kt00009(
        self,
        stk_bond_tp: str,
        mrkt_tp: str,
        sell_tp: str,
        qry_tp: str,
        dmst_stex_tp: str,
        ord_dt: str = "",
        stock_code: str = "",
        fr_ord_no: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌별주문체결현황요청 (kt00009)

        Args:
            stk_bond_tp (str): 주식채권구분 (0:전체, 1:주식, 2:채권)
            mrkt_tp (str): 시장구분 (0:전체, 1:코스피, 2:코스닥, 3:OTCBB, 4:ECN)
            sell_tp (str): 매도수구분 (0:전체, 1:매도, 2:매수)
            qry_tp (str): 조회구분 (0:전체, 1:체결)
            dmst_stex_tp (str): 국내거래소구분 (%:전체, KRX:한국거래소, NXT:넥스트트레이드, SOR:최선주문집행)
            ord_dt (str, optional): 주문일자 (YYYYMMDD). Defaults to "".
            stock_code (str, optional): 종목코드 (12자리). Defaults to "".
            fr_ord_no (str, optional): 시작주문번호 (7자리). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌별주문체결현황 데이터
                {
                    "sell_grntl_engg_amt": str,  # 매도약정금액
                    "buy_engg_amt": str,  # 매수약정금액
                    "engg_amt": str,  # 약정금액
                    "acnt_ord_cntr_prst_array": [  # 계좌별주문체결현황배열
                        {
                            "stk_bond_tp": str,  # 주식채권구분
                            "ord_no": str,  # 주문번호
                            "stk_cd": str,  # 종목번호
                            "trde_tp": str,  # 매매구분
                            "io_tp_nm": str,  # 주문유형구분
                            "ord_qty": str,  # 주문수량
                            "ord_uv": str,  # 주문단가
                            "cnfm_qty": str,  # 확인수량
                            "rsrv_oppo": str,  # 예약/반대
                            "cntr_no": str,  # 체결번호
                            "acpt_tp": str,  # 접수구분
                            "orig_ord_no": str,  # 원주문번호
                            "stk_nm": str,  # 종목명
                            "setl_tp": str,  # 결제구분
                            "crd_deal_tp": str,  # 신용거래구분
                            "cntr_qty": str,  # 체결수량
                            "cntr_uv": str,  # 체결단가
                            "comm_ord_tp": str,  # 통신구분
                            "mdfy_cncl_tp": str,  # 정정/취소구분
                            "cntr_tm": str,  # 체결시간
                            "dmst_stex_tp": str,  # 국내거래소구분
                            "cond_uv": str,  # 스톱가
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.account_order_execution_status_request_kt00009(
            ...     stk_bond_tp="0",  # 전체
            ...     mrkt_tp="0",  # 전체
            ...     sell_tp="0",  # 전체
            ...     qry_tp="0",  # 전체
            ...     dmst_stex_tp="KRX"  # 한국거래소
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00009",
        }
        data = {
            "stk_bond_tp": stk_bond_tp,
            "mrkt_tp": mrkt_tp,
            "sell_tp": sell_tp,
            "qry_tp": qry_tp,
            "dmst_stex_tp": dmst_stex_tp,
        }
        
        if ord_dt:
            data["ord_dt"] = ord_dt
        if stock_code:
            data["stk_cd"] = stock_code
        if fr_ord_no:
            data["fr_ord_no"] = fr_ord_no
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def withdrawable_order_amount_request_kt00010(
        self,
        stock_code: str,
        trde_tp: str,
        uv: str,
        io_amt: str = "",
        trde_qty: str = "",
        exp_buy_unp: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주문인출가능금액요청 (kt00010)

        Args:
            stock_code (str): 종목번호 (12자리)
            trde_tp (str): 매매구분 (1:매도, 2:매수)
            uv (str): 매수가격 (10자리)
            io_amt (str, optional): 입출금액 (12자리). Defaults to "".
            trde_qty (str, optional): 매매수량 (10자리). Defaults to "".
            exp_buy_unp (str, optional): 예상매수단가 (10자리). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주문인출가능금액 데이터
                {
                    "profa_20ord_alow_amt": str,  # 증거금20%주문가능금액
                    "profa_20ord_alowq": str,  # 증거금20%주문가능수량
                    "profa_30ord_alow_amt": str,  # 증거금30%주문가능금액
                    "profa_30ord_alowq": str,  # 증거금30%주문가능수량
                    "profa_40ord_alow_amt": str,  # 증거금40%주문가능금액
                    "profa_40ord_alowq": str,  # 증거금40%주문가능수량
                    "profa_50ord_alow_amt": str,  # 증거금50%주문가능금액
                    "profa_50ord_alowq": str,  # 증거금50%주문가능수량
                    "profa_60ord_alow_amt": str,  # 증거금60%주문가능금액
                    "profa_60ord_alowq": str,  # 증거금60%주문가능수량
                    "profa_rdex_60ord_alow_amt": str,  # 증거금감면60%주문가능금
                    "profa_rdex_60ord_alowq": str,  # 증거금감면60%주문가능수
                    "profa_100ord_alow_amt": str,  # 증거금100%주문가능금액
                    "profa_100ord_alowq": str,  # 증거금100%주문가능수량
                    "pred_reu_alowa": str,  # 전일재사용가능금액
                    "tdy_reu_alowa": str,  # 금일재사용가능금액
                    "entr": str,  # 예수금
                    "repl_amt": str,  # 대용금
                    "uncla": str,  # 미수금
                    "ord_pos_repl": str,  # 주문가능대용
                    "ord_alowa": str,  # 주문가능현금
                    "wthd_alowa": str,  # 인출가능금액
                    "nxdy_wthd_alowa": str,  # 익일인출가능금액
                    "pur_amt": str,  # 매입금액
                    "cmsn": str,  # 수수료
                    "pur_exct_amt": str,  # 매입정산금
                    "d2entra": str,  # D2추정예수금
                    "profa_rdex_aplc_tp": str,  # 증거금감면적용구분 (0:일반,1:60%감면)
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.withdrawable_order_amount_request_kt00010(
            ...     stock_code="005930",  # 삼성전자
            ...     trde_tp="2",  # 매수
            ...     uv="267000"  # 매수가격
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00010",
        }
        data = {
            "stk_cd": stock_code,
            "trde_tp": trde_tp,
            "uv": uv,
        }
        
        if io_amt:
            data["io_amt"] = io_amt
        if trde_qty:
            data["trde_qty"] = trde_qty
        if exp_buy_unp:
            data["exp_buy_unp"] = exp_buy_unp
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def orderable_quantity_by_margin_ratio_request_kt00011(
        self,
        stock_code: str,
        uv: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        증거금율별주문가능수량조회요청 (kt00011)

        Args:
            stock_code (str): 종목번호 (12자리)
            uv (str, optional): 매수가격 (10자리). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 증거금율별주문가능수량 데이터
                {
                    "stk_profa_rt": str,  # 종목증거금율
                    "profa_rt": str,  # 계좌증거금율
                    "aplc_rt": str,  # 적용증거금율
                    "profa_20ord_alow_amt": str,  # 증거금20%주문가능금액
                    "profa_20ord_alowq": str,  # 증거금20%주문가능수량
                    "profa_20pred_reu_amt": str,  # 증거금20%전일재사용금액
                    "profa_20tdy_reu_amt": str,  # 증거금20%금일재사용금액
                    "profa_30ord_alow_amt": str,  # 증거금30%주문가능금액
                    "profa_30ord_alowq": str,  # 증거금30%주문가능수량
                    "profa_30pred_reu_amt": str,  # 증거금30%전일재사용금액
                    "profa_30tdy_reu_amt": str,  # 증거금30%금일재사용금액
                    "profa_40ord_alow_amt": str,  # 증거금40%주문가능금액
                    "profa_40ord_alowq": str,  # 증거금40%주문가능수량
                    "profa_40pred_reu_amt": str,  # 증거금40%전일재사용금액
                    "profa_40tdy_reu_amt": str,  # 증거금40%금일재사용금액
                    "profa_50ord_alow_amt": str,  # 증거금50%주문가능금액
                    "profa_50ord_alowq": str,  # 증거금50%주문가능수량
                    "profa_50pred_reu_amt": str,  # 증거금50%전일재사용금액
                    "profa_50tdy_reu_amt": str,  # 증거금50%금일재사용금액
                    "profa_60ord_alow_amt": str,  # 증거금60%주문가능금액
                    "profa_60ord_alowq": str,  # 증거금60%주문가능수량
                    "profa_60pred_reu_amt": str,  # 증거금60%전일재사용금액
                    "profa_60tdy_reu_amt": str,  # 증거금60%금일재사용금액
                    "profa_100ord_alow_amt": str,  # 증거금100%주문가능금액
                    "profa_100ord_alowq": str,  # 증거금100%주문가능수량
                    "profa_100pred_reu_amt": str,  # 증거금100%전일재사용금액
                    "profa_100tdy_reu_amt": str,  # 증거금100%금일재사용금액
                    "min_ord_alow_amt": str,  # 미수불가주문가능금액
                    "min_ord_alowq": str,  # 미수불가주문가능수량
                    "min_pred_reu_amt": str,  # 미수불가전일재사용금액
                    "min_tdy_reu_amt": str,  # 미수불가금일재사용금액
                    "entr": str,  # 예수금
                    "repl_amt": str,  # 대용금
                    "uncla": str,  # 미수금
                    "ord_pos_repl": str,  # 주문가능대용
                    "ord_alowa": str,  # 주문가능현금
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.orderable_quantity_by_margin_ratio_request_kt00011(
            ...     stock_code="005930"  # 삼성전자
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00011",
        }
        data = {
            "stk_cd": stock_code,
        }
        
        if uv:
            data["uv"] = uv
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def orderable_quantity_by_credit_guarantee_ratio_request_kt00012(
        self,
        stock_code: str,
        uv: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        신용보증금율별주문가능수량조회요청 (kt00012)

        Args:
            stock_code (str): 종목번호 (12자리)
            uv (str, optional): 매수가격 (10자리). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 신용보증금율별주문가능수량 데이터
                {
                    "stk_assr_rt": str,  # 종목보증금율
                    "stk_assr_rt_nm": str,  # 종목보증금율명
                    "assr_30ord_alow_amt": str,  # 보증금30%주문가능금액
                    "assr_30ord_alowq": str,  # 보증금30%주문가능수량
                    "assr_30pred_reu_amt": str,  # 보증금30%전일재사용금액
                    "assr_30tdy_reu_amt": str,  # 보증금30%금일재사용금액
                    "assr_40ord_alow_amt": str,  # 보증금40%주문가능금액
                    "assr_40ord_alowq": str,  # 보증금40%주문가능수량
                    "assr_40pred_reu_amt": str,  # 보증금40%전일재사용금액
                    "assr_40tdy_reu_amt": str,  # 보증금40%금일재사용금액
                    "assr_50ord_alow_amt": str,  # 보증금50%주문가능금액
                    "assr_50ord_alowq": str,  # 보증금50%주문가능수량
                    "assr_50pred_reu_amt": str,  # 보증금50%전일재사용금액
                    "assr_50tdy_reu_amt": str,  # 보증금50%금일재사용금액
                    "assr_60ord_alow_amt": str,  # 보증금60%주문가능금액
                    "assr_60ord_alowq": str,  # 보증금60%주문가능수량
                    "assr_60pred_reu_amt": str,  # 보증금60%전일재사용금액
                    "assr_60tdy_reu_amt": str,  # 보증금60%금일재사용금액
                    "entr": str,  # 예수금
                    "repl_amt": str,  # 대용금
                    "uncla": str,  # 미수금
                    "ord_pos_repl": str,  # 주문가능대용
                    "ord_alowa": str,  # 주문가능현금
                    "out_alowa": str,  # 미수가능금액
                    "out_pos_qty": str,  # 미수가능수량
                    "min_amt": str,  # 미수불가금액
                    "min_qty": str,  # 미수불가수량
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.orderable_quantity_by_credit_guarantee_ratio_request_kt00012(
            ...     stock_code="005930"  # 삼성전자
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00012",
        }
        data = {
            "stk_cd": stock_code,
        }
        
        if uv:
            data["uv"] = uv
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def margin_detail_inquiry_request_kt00013(
        self,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        증거금세부내역조회요청 (kt00013)

        Args:
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 증거금세부내역 데이터
                {
                    "tdy_reu_objt_amt": str,  # 금일재사용대상금액
                    "tdy_reu_use_amt": str,  # 금일재사용사용금액
                    "tdy_reu_alowa": str,  # 금일재사용가능금액
                    "tdy_reu_lmtt_amt": str,  # 금일재사용제한금액
                    "tdy_reu_alowa_fin": str,  # 금일재사용가능금액최종
                    "pred_reu_objt_amt": str,  # 전일재사용대상금액
                    "pred_reu_use_amt": str,  # 전일재사용사용금액
                    "pred_reu_alowa": str,  # 전일재사용가능금액
                    "pred_reu_lmtt_amt": str,  # 전일재사용제한금액
                    "pred_reu_alowa_fin": str,  # 전일재사용가능금액최종
                    "ch_amt": str,  # 현금금액
                    "ch_profa": str,  # 현금증거금
                    "use_pos_ch": str,  # 사용가능현금
                    "ch_use_lmtt_amt": str,  # 현금사용제한금액
                    "use_pos_ch_fin": str,  # 사용가능현금최종
                    "repl_amt_amt": str,  # 대용금액
                    "repl_profa": str,  # 대용증거금
                    "use_pos_repl": str,  # 사용가능대용
                    "repl_use_lmtt_amt": str,  # 대용사용제한금액
                    "use_pos_repl_fin": str,  # 사용가능대용최종
                    "crd_grnta_ch": str,  # 신용보증금현금
                    "crd_grnta_repl": str,  # 신용보증금대용
                    "crd_grnt_ch": str,  # 신용담보금현금
                    "crd_grnt_repl": str,  # 신용담보금대용
                    "uncla": str,  # 미수금
                    "ls_grnt_reu_gold": str,  # 대주담보금재사용금
                    "20ord_alow_amt": str,  # 20%주문가능금액
                    "30ord_alow_amt": str,  # 30%주문가능금액
                    "40ord_alow_amt": str,  # 40%주문가능금액
                    "50ord_alow_amt": str,  # 50%주문가능금액
                    "60ord_alow_amt": str,  # 60%주문가능금액
                    "100ord_alow_amt": str,  # 100%주문가능금액
                    "tdy_crd_rpya_loss_amt": str,  # 금일신용상환손실금액
                    "pred_crd_rpya_loss_amt": str,  # 전일신용상환손실금액
                    "tdy_ls_rpya_loss_repl_profa": str,  # 금일대주상환손실대용증거금
                    "pred_ls_rpya_loss_repl_profa": str,  # 전일대주상환손실대용증거금
                    "evlt_repl_amt_spg_use_skip": str,  # 평가대용금(현물사용제외)
                    "evlt_repl_rt": str,  # 평가대용비율
                    "crd_repl_profa": str,  # 신용대용증거금
                    "ch_ord_repl_profa": str,  # 현금주문대용증거금
                    "crd_ord_repl_profa": str,  # 신용주문대용증거금
                    "crd_repl_conv_gold": str,  # 신용대용환산금
                    "repl_alowa": str,  # 대용가능금액(현금제한)
                    "repl_alowa_2": str,  # 대용가능금액2(신용제한)
                    "ch_repl_lck_gold": str,  # 현금대용부족금
                    "crd_repl_lck_gold": str,  # 신용대용부족금
                    "ch_ord_alow_repla": str,  # 현금주문가능대용금
                    "crd_ord_alow_repla": str,  # 신용주문가능대용금
                    "d2vexct_entr": str,  # D2가정산예수금
                    "d2ch_ord_alow_amt": str,  # D2현금주문가능금액
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.margin_detail_inquiry_request_kt00013()
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00013",
        }
        data = {}
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def comprehensive_transaction_history_request_kt00015(
        self,
        start_date: str,
        end_date: str,
        transaction_type: str,
        stock_code: str = "",
        currency_code: str = "",
        goods_type: str = "0",
        foreign_exchange_code: str = "",
        domestic_exchange_type: str = "%",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        위탁종합거래내역요청 (kt00015)

        Args:
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            transaction_type (str): 구분 (0:전체,1:입출금,2:입출고,3:매매,4:매수,5:매도,6:입금,7:출금,A:예탁담보대출입금,B:매도담보대출입금,C:현금상환(융자,담보상환),F:환전,M:입출금+환전,G:외화매수,H:외화매도,I:환전정산입금,J:환전정산출금)
            stock_code (str, optional): 종목코드 (12자리). Defaults to "".
            currency_code (str, optional): 통화코드 (3자리). Defaults to "".
            goods_type (str, optional): 상품구분 (0:전체, 1:국내주식, 2:수익증권, 3:해외주식, 4:금융상품). Defaults to "0".
            foreign_exchange_code (str, optional): 해외거래소코드 (10자리). Defaults to "".
            domestic_exchange_type (str, optional): 국내거래소구분 (%:전체,KRX:한국거래소,NXT:넥스트트레이드). Defaults to "%".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 위탁종합거래내역 데이터
                {
                    "acnt_no": str,  # 계좌번호
                    "trst_ovrl_trde_prps_array": [  # 위탁종합거래내역배열
                        {
                            "trde_dt": str,  # 거래일자
                            "trde_no": str,  # 거래번호
                            "rmrk_nm": str,  # 적요명
                            "crd_deal_tp_nm": str,  # 신용거래구분명
                            "exct_amt": str,  # 정산금액
                            "loan_amt_rpya": str,  # 대출금상환
                            "fc_trde_amt": str,  # 거래금액(외)
                            "fc_exct_amt": str,  # 정산금액(외)
                            "entra_remn": str,  # 예수금잔고
                            "crnc_cd": str,  # 통화코드
                            "trde_ocr_tp": str,  # 거래종류구분
                            "trde_kind_nm": str,  # 거래종류명
                            "stk_nm": str,  # 종목명
                            "trde_amt": str,  # 거래금액
                            "trde_agri_tax": str,  # 거래및농특세
                            "rpy_diffa": str,  # 상환차금
                            "fc_trde_tax": str,  # 거래세(외)
                            "dly_sum": str,  # 연체합
                            "fc_entra": str,  # 외화예수금잔고
                            "mdia_tp_nm": str,  # 매체구분명
                            "io_tp": str,  # 입출구분
                            "io_tp_nm": str,  # 입출구분명
                            "orig_deal_no": str,  # 원거래번호
                            "stk_cd": str,  # 종목코드
                            "trde_qty_jwa_cnt": str,  # 거래수량/좌수
                            "cmsn": str,  # 수수료
                            "int_ls_usfe": str,  # 이자/대주이용
                            "fc_cmsn": str,  # 수수료(외)
                            "fc_dly_sum": str,  # 연체합(외)
                            "vlbl_nowrm": str,  # 유가금잔
                            "proc_tm": str,  # 처리시간
                            "isin_cd": str,  # ISIN코드
                            "stex_cd": str,  # 거래소코드
                            "stex_nm": str,  # 거래소명
                            "trde_unit": str,  # 거래단가/환율
                            "incm_resi_tax": str,  # 소득/주민세
                            "loan_dt": str,  # 대출일
                            "uncl_ocr": str,  # 미수(원/주)
                            "rpym_sum": str,  # 변제합
                            "cntr_dt": str,  # 체결일
                            "rcpy_no": str,  # 출납번호
                            "prcsr": str,  # 처리자
                            "proc_brch": str,  # 처리점
                            "trde_stle": str,  # 매매형태
                            "txon_base_pric": str,  # 과세기준가
                            "tax_sum_cmsn": str,  # 세금수수료합
                            "frgn_pay_txam": str,  # 외국납부세액(외)
                            "fc_uncl_ocr": str,  # 미수(외)
                            "rpym_sum_fr": str,  # 변제합(외)
                            "rcpmnyer": str,  # 입금자
                            "trde_prtc_tp": str,  # 거래내역구분
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.comprehensive_transaction_history_request_kt00015(
            ...     start_date="20241121",
            ...     end_date="20241125",
            ...     transaction_type="0"  # 전체
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00015",
        }
        data = {
            "strt_dt": start_date,
            "end_dt": end_date,
            "tp": transaction_type,
            "stk_cd": stock_code,
            "crnc_cd": currency_code,
            "gds_tp": goods_type,
            "frgn_stex_code": foreign_exchange_code,
            "dmst_stex_tp": domestic_exchange_type,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def daily_account_return_detail_status_request_kt00016(
        self,
        from_date: str,
        to_date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        일별계좌수익률상세현황요청 (kt00016)

        Args:
            from_date (str): 평가시작일 (YYYYMMDD)
            to_date (str): 평가종료일 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 일별계좌수익률상세현황 데이터
                {
                    "mang_empno": str,  # 관리사원번호
                    "mngr_nm": str,  # 관리자명
                    "dept_nm": str,  # 관리자지점
                    "entr_fr": str,  # 예수금_초
                    "entr_to": str,  # 예수금_말
                    "scrt_evlt_amt_fr": str,  # 유가증권평가금액_초
                    "scrt_evlt_amt_to": str,  # 유가증권평가금액_말
                    "ls_grnt_fr": str,  # 대주담보금_초
                    "ls_grnt_to": str,  # 대주담보금_말
                    "crd_loan_fr": str,  # 신용융자금_초
                    "crd_loan_to": str,  # 신용융자금_말
                    "ch_uncla_fr": str,  # 현금미수금_초
                    "ch_uncla_to": str,  # 현금미수금_말
                    "krw_asgna_fr": str,  # 원화대용금_초
                    "krw_asgna_to": str,  # 원화대용금_말
                    "ls_evlta_fr": str,  # 대주평가금_초
                    "ls_evlta_to": str,  # 대주평가금_말
                    "rght_evlta_fr": str,  # 권리평가금_초
                    "rght_evlta_to": str,  # 권리평가금_말
                    "loan_amt_fr": str,  # 대출금_초
                    "loan_amt_to": str,  # 대출금_말
                    "etc_loana_fr": str,  # 기타대여금_초
                    "etc_loana_to": str,  # 기타대여금_말
                    "crd_int_npay_gold_fr": str,  # 신용이자미납금_초
                    "crd_int_npay_gold_to": str,  # 신용이자미납금_말
                    "crd_int_fr": str,  # 신용이자_초
                    "crd_int_to": str,  # 신용이자_말
                    "tot_amt_fr": str,  # 순자산액계_초
                    "tot_amt_to": str,  # 순자산액계_말
                    "invt_bsamt": str,  # 투자원금평잔
                    "evltv_prft": str,  # 평가손익
                    "prft_rt": str,  # 수익률
                    "tern_rt": str,  # 회전율
                    "termin_tot_trns": str,  # 기간내총입금
                    "termin_tot_pymn": str,  # 기간내총출금
                    "termin_tot_inq": str,  # 기간내총입고
                    "termin_tot_outq": str,  # 기간내총출고
                    "futr_repl_sella": str,  # 선물대용매도금액
                    "trst_repl_sella": str,  # 위탁대용매도금액
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.daily_account_return_detail_status_request_kt00016(
            ...     from_date="20241111",
            ...     to_date="20241125"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00016",
        }
        data = {
            "fr_dt": from_date,
            "to_dt": to_date,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def today_account_status_by_account_request_kt00017(
        self,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌별당일현황요청 (kt00017)

        Args:
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌별당일현황 데이터
                {
                    "d2_entra": str,  # D+2추정예수금
                    "crd_int_npay_gold": str,  # 신용이자미납금
                    "etc_loana": str,  # 기타대여금
                    "gnrl_stk_evlt_amt_d2": str,  # 일반주식평가금액D+2
                    "dpst_grnt_use_amt_d2": str,  # 예탁담보대출금D+2
                    "crd_stk_evlt_amt_d2": str,  # 예탁담보주식평가금액D+2
                    "crd_loan_d2": str,  # 신용융자금D+2
                    "crd_loan_evlta_d2": str,  # 신용융자평가금D+2
                    "crd_ls_grnt_d2": str,  # 신용대주담보금D+2
                    "crd_ls_evlta_d2": str,  # 신용대주평가금D+2
                    "ina_amt": str,  # 입금금액
                    "outa": str,  # 출금금액
                    "inq_amt": str,  # 입고금액
                    "outq_amt": str,  # 출고금액
                    "sell_amt": str,  # 매도금액
                    "buy_amt": str,  # 매수금액
                    "cmsn": str,  # 수수료
                    "tax": str,  # 세금
                    "stk_pur_cptal_loan_amt": str,  # 주식매입자금대출금
                    "rp_evlt_amt": str,  # RP평가금액
                    "bd_evlt_amt": str,  # 채권평가금액
                    "elsevlt_amt": str,  # ELS평가금액
                    "crd_int_amt": str,  # 신용이자금액
                    "sel_prica_grnt_loan_int_amt_amt": str,  # 매도대금담보대출이자금액
                    "dvida_amt": str,  # 배당금액
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.today_account_status_by_account_request_kt00017()
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00017",
        }
        data = {}
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def account_evaluation_balance_detail_request_kt00018(
        self,
        query_type: str,
        domestic_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        계좌평가잔고내역요청 (kt00018)

        Args:
            query_type (str): 조회구분 (1:합산, 2:개별)
            domestic_exchange_type (str): 국내거래소구분 (KRX:한국거래소,NXT:넥스트트레이드)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 계좌평가잔고내역 데이터
                {
                    "tot_pur_amt": str,  # 총매입금액
                    "tot_evlt_amt": str,  # 총평가금액
                    "tot_evlt_pl": str,  # 총평가손익금액
                    "tot_prft_rt": str,  # 총수익률(%)
                    "prsm_dpst_aset_amt": str,  # 추정예탁자산
                    "tot_loan_amt": str,  # 총대출금
                    "tot_crd_loan_amt": str,  # 총융자금액
                    "tot_crd_ls_amt": str,  # 총대주금액
                    "acnt_evlt_remn_indv_tot": [  # 계좌평가잔고개별합산
                        {
                            "stk_cd": str,  # 종목번호
                            "stk_nm": str,  # 종목명
                            "evltv_prft": str,  # 평가손익
                            "prft_rt": str,  # 수익률(%)
                            "pur_pric": str,  # 매입가
                            "pred_close_pric": str,  # 전일종가
                            "rmnd_qty": str,  # 보유수량
                            "trde_able_qty": str,  # 매매가능수량
                            "cur_prc": str,  # 현재가
                            "pred_buyq": str,  # 전일매수수량
                            "pred_sellq": str,  # 전일매도수량
                            "tdy_buyq": str,  # 금일매수수량
                            "tdy_sellq": str,  # 금일매도수량
                            "pur_amt": str,  # 매입금액
                            "pur_cmsn": str,  # 매입수수료
                            "evlt_amt": str,  # 평가금액
                            "sell_cmsn": str,  # 평가수수료
                            "tax": str,  # 세금
                            "sum_cmsn": str,  # 수수료합
                            "poss_rt": str,  # 보유비중(%)
                            "crd_tp": str,  # 신용구분
                            "crd_tp_nm": str,  # 신용구분명
                            "crd_loan_dt": str,  # 대출일
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.account.account_evaluation_balance_detail_request_kt00018(
            ...     query_type="1",  # 합산
            ...     domestic_exchange_type="KRX"  # 한국거래소
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt00018",
        }
        data = {
            "qry_tp": query_type,
            "dmst_stex_tp": domestic_exchange_type,
        }
            
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )