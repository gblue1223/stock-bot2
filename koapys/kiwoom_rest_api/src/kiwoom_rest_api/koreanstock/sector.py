from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class Sector(KiwoomBaseAPI):
    """한국 주식 섹터 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/sect"
    ):
        """
        Sector 클래스 초기화
        
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
        
   
    def industry_program_trading_request_ka10010(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """업종프로그램매매를 조회합니다.

        Args:
            stock_code (str): 종목코드 (거래소별 종목코드)
                - KRX: 039490
                - NXT: 039490_NX
                - SOR: 039490_AL
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종프로그램매매 데이터
                {
                    "dfrt_trst_sell_qty": str,  # 차익위탁매도수량
                    "dfrt_trst_sell_amt": str,  # 차익위탁매도금액
                    "dfrt_trst_buy_qty": str,  # 차익위탁매수수량
                    "dfrt_trst_buy_amt": str,  # 차익위탁매수금액
                    "dfrt_trst_netprps_qty": str,  # 차익위탁순매수수량
                    "dfrt_trst_netprps_amt": str,  # 차익위탁순매수금액
                    "ndiffpro_trst_sell_qty": str,  # 비차익위탁매도수량
                    "ndiffpro_trst_sell_amt": str,  # 비차익위탁매도금액
                    "ndiffpro_trst_buy_qty": str,  # 비차익위탁매수수량
                    "ndiffpro_trst_buy_amt": str,  # 비차익위탁매수금액
                    "ndiffpro_trst_netprps_qty": str,  # 비차익위탁순매수수량
                    "ndiffpro_trst_netprps_amt": str,  # 비차익위탁순매수금액
                    "all_dfrt_trst_sell_qty": str,  # 전체차익위탁매도수량
                    "all_dfrt_trst_sell_amt": str,  # 전체차익위탁매도금액
                    "all_dfrt_trst_buy_qty": str,  # 전체차익위탁매수수량
                    "all_dfrt_trst_buy_amt": str,  # 전체차익위탁매수금액
                    "all_dfrt_trst_netprps_qty": str,  # 전체차익위탁순매수수량
                    "all_dfrt_trst_netprps_amt": str,  # 전체차익위탁순매수금액
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.industry_program_trading_request_ka10010(
            ...     stock_code="005930"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10010",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )

    def industrywise_investor_net_buy_request_ka10051(
        self,
        mrkt_tp: str,
        amt_qty_tp: str,
        stex_tp: str,
        base_dt: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """업종별투자자순매수를 조회합니다.

        Args:
            mrkt_tp (str): 시장구분 (코스피:0, 코스닥:1)
            amt_qty_tp (str): 금액수량구분 (금액:0, 수량:1)
            stex_tp (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            base_dt (str, optional): 기준일자 (YYYYMMDD). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종별투자자순매수 데이터
                {
                    "inds_netprps": [
                        {
                            "inds_cd": str,  # 업종코드
                            "inds_nm": str,  # 업종명
                            "cur_prc": str,  # 현재가
                            "pre_smbol": str,  # 대비부호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락율
                            "trde_qty": str,  # 거래량
                            "sc_netprps": str,  # 증권순매수
                            "insrnc_netprps": str,  # 보험순매수
                            "invtrt_netprps": str,  # 투신순매수
                            "bank_netprps": str,  # 은행순매수
                            "jnsinkm_netprps": str,  # 종신금순매수
                            "endw_netprps": str,  # 기금순매수
                            "etc_corp_netprps": str,  # 기타법인순매수
                            "ind_netprps": str,  # 개인순매수
                            "frgnr_netprps": str,  # 외국인순매수
                            "native_trmt_frgnr_netprps": str,  # 내국인대우외국인순매수
                            "natn_netprps": str,  # 국가순매수
                            "samo_fund_netprps": str,  # 사모펀드순매수
                            "orgn_netprps": str,  # 기관계순매수
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.industrywise_investor_net_buy_request_ka10051(
            ...     mrkt_tp="0",
            ...     amt_qty_tp="0",
            ...     stex_tp="3",
            ...     base_dt="20241107"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10051",
        }

        data = {
            "mrkt_tp": mrkt_tp,
            "amt_qty_tp": amt_qty_tp,
            "base_dt": base_dt,
            "stex_tp": stex_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_current_price_request_ka20001(
        self,
        mrkt_tp: str,
        inds_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """업종현재가를 조회합니다.

        Args:
            mrkt_tp (str): 시장구분 (0:코스피, 1:코스닥, 2:코스피200)
            inds_cd (str): 업종코드
                - 001: 종합(KOSPI)
                - 002: 대형주
                - 003: 중형주
                - 004: 소형주
                - 101: 종합(KOSDAQ)
                - 201: KOSPI200
                - 302: KOSTAR
                - 701: KRX100
                - 나머지 업종코드 참고
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종현재가 데이터
                {
                    "cur_prc": str,  # 현재가
                    "pred_pre_sig": str,  # 전일대비기호
                    "pred_pre": str,  # 전일대비
                    "flu_rt": str,  # 등락률
                    "trde_qty": str,  # 거래량
                    "trde_prica": str,  # 거래대금
                    "trde_frmatn_stk_num": str,  # 거래형성종목수
                    "trde_frmatn_rt": str,  # 거래형성비율
                    "open_pric": str,  # 시가
                    "high_pric": str,  # 고가
                    "low_pric": str,  # 저가
                    "upl": str,  # 상한
                    "rising": str,  # 상승
                    "stdns": str,  # 보합
                    "fall": str,  # 하락
                    "lst": str,  # 하한
                    "52wk_hgst_pric": str,  # 52주최고가
                    "52wk_hgst_pric_dt": str,  # 52주최고가일
                    "52wk_hgst_pric_pre_rt": str,  # 52주최고가대비율
                    "52wk_lwst_pric": str,  # 52주최저가
                    "52wk_lwst_pric_dt": str,  # 52주최저가일
                    "52wk_lwst_pric_pre_rt": str,  # 52주최저가대비율
                    "inds_cur_prc_tm": [  # 업종현재가_시간별
                        {
                            "tm_n": str,  # 시간n
                            "cur_prc_n": str,  # 현재가n
                            "pred_pre_sig_n": str,  # 전일대비기호n
                            "pred_pre_n": str,  # 전일대비n
                            "flu_rt_n": str,  # 등락률n
                            "trde_qty_n": str,  # 거래량n
                            "acc_trde_qty_n": str,  # 누적거래량n
                            "stex_tp": str,  # 거래소구분
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.industry_current_price_request_ka20001(
            ...     mrkt_tp="0",
            ...     inds_cd="001"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20001",
        }

        data = {
            "mrkt_tp": mrkt_tp,
            "inds_cd": inds_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industrywise_stock_price_request_ka20002(
        self,
        mrkt_tp: str,
        inds_cd: str,
        stex_tp: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """업종별주가를 조회합니다.

        Args:
            mrkt_tp (str): 시장구분 (0:코스피, 1:코스닥, 2:코스피200)
            inds_cd (str): 업종코드
                - 001: 종합(KOSPI)
                - 002: 대형주
                - 003: 중형주
                - 004: 소형주
                - 101: 종합(KOSDAQ)
                - 201: KOSPI200
                - 302: KOSTAR
                - 701: KRX100
                - 나머지 업종코드 참고
            stex_tp (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종별주가 데이터
                {
                    "inds_stkpc": [  # 업종별주가
                        {
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pred_pre_sig": str,  # 전일대비기호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락률
                            "now_trde_qty": str,  # 현재거래량
                            "sel_bid": str,  # 매도호가
                            "buy_bid": str,  # 매수호가
                            "open_pric": str,  # 시가
                            "high_pric": str,  # 고가
                            "low_pric": str,  # 저가
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.industrywise_stock_price_request_ka20002(
            ...     mrkt_tp="0",
            ...     inds_cd="001",
            ...     stex_tp="1"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20002",
        }

        data = {
            "mrkt_tp": mrkt_tp,
            "inds_cd": inds_cd,
            "stex_tp": stex_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def all_industries_index_request_ka20003(
        self,
        inds_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """전업종지수를 조회합니다.

        Args:
            inds_cd (str): 업종코드
                - 001: 종합(KOSPI)
                - 002: 대형주
                - 003: 중형주
                - 004: 소형주
                - 101: 종합(KOSDAQ)
                - 201: KOSPI200
                - 302: KOSTAR
                - 701: KRX100
                - 나머지 업종코드 참고
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 전업종지수 데이터
                {
                    "all_inds_idex": [  # 전업종지수
                        {
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pre_sig": str,  # 대비기호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락률
                            "trde_qty": str,  # 거래량
                            "wght": str,  # 비중
                            "trde_prica": str,  # 거래대금
                            "upl": str,  # 상한
                            "rising": str,  # 상승
                            "stdns": str,  # 보합
                            "fall": str,  # 하락
                            "lst": str,  # 하한
                            "flo_stk_num": str,  # 상장종목수
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.all_industries_index_request_ka20003(
            ...     inds_cd="001"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20003",
        }

        data = {
            "inds_cd": inds_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_daily_current_price_request_ka20009(
        self,
        mrkt_tp: str,
        inds_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """업종현재가일별을 조회합니다.

        Args:
            mrkt_tp (str): 시장구분 (0:코스피, 1:코스닥, 2:코스피200)
            inds_cd (str): 업종코드
                - 001: 종합(KOSPI)
                - 002: 대형주
                - 003: 중형주
                - 004: 소형주
                - 101: 종합(KOSDAQ)
                - 201: KOSPI200
                - 302: KOSTAR
                - 701: KRX100
                - 나머지 업종코드 참고
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종현재가일별 데이터
                {
                    "cur_prc": str,  # 현재가
                    "pred_pre_sig": str,  # 전일대비기호
                    "pred_pre": str,  # 전일대비
                    "flu_rt": str,  # 등락률
                    "trde_qty": str,  # 거래량
                    "trde_prica": str,  # 거래대금
                    "trde_frmatn_stk_num": str,  # 거래형성종목수
                    "trde_frmatn_rt": str,  # 거래형성비율
                    "open_pric": str,  # 시가
                    "high_pric": str,  # 고가
                    "low_pric": str,  # 저가
                    "upl": str,  # 상한
                    "rising": str,  # 상승
                    "stdns": str,  # 보합
                    "fall": str,  # 하락
                    "lst": str,  # 하한
                    "52wk_hgst_pric": str,  # 52주최고가
                    "52wk_hgst_pric_dt": str,  # 52주최고가일
                    "52wk_hgst_pric_pre_rt": str,  # 52주최고가대비율
                    "52wk_lwst_pric": str,  # 52주최저가
                    "52wk_lwst_pric_dt": str,  # 52주최저가일
                    "52wk_lwst_pric_pre_rt": str,  # 52주최저가대비율
                    "inds_cur_prc_daly_rept": [  # 업종현재가_일별반복
                        {
                            "dt_n": str,  # 일자n
                            "cur_prc_n": str,  # 현재가n
                            "pred_pre_sig_n": str,  # 전일대비기호n
                            "pred_pre_n": str,  # 전일대비n
                            "flu_rt_n": str,  # 등락률n
                            "acc_trde_qty_n": str,  # 누적거래량n
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.industry_daily_current_price_request_ka20009(
            ...     mrkt_tp="0",
            ...     inds_cd="001"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20009",
        }

        data = {
            "mrkt_tp": mrkt_tp,
            "inds_cd": inds_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )