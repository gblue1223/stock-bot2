from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class ETF(KiwoomBaseAPI):
    """한국 주식 ETF 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/etf"
    ):
        """
        ETF 클래스 초기화
        
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
        
   
    def etf_return_rate_request_ka40001(
        self,
        stock_code: str,
        etf_object_index_code: str,
        period: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 수익률을 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            etf_object_index_code (str): ETF대상지수코드 (3자리)
            period (str): 기간
                - "0": 1주
                - "1": 1달
                - "2": 6개월
                - "3": 1년
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 수익률 데이터
                {
                    "etfprft_rt_lst": list,  # ETF수익율 리스트
                        [
                            {
                                "etfprft_rt": str,  # ETF수익률
                                "cntr_prft_rt": str,  # 체결수익률
                                "for_netprps_qty": str,  # 외인순매수수량
                                "orgn_netprps_qty": str,  # 기관순매수수량
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_return_rate_request_ka40001(
            ...     stock_code="069500",
            ...     etf_object_index_code="207",
            ...     period="3"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40001",
        }

        data = {
            "stk_cd": stock_code,
            "etfobjt_idex_cd": etf_object_index_code,
            "dt": period,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_stock_info_request_ka40002(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 종목정보를 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 종목정보 데이터
                {
                    "stk_nm": str,  # 종목명
                    "etfobjt_idex_nm": str,  # ETF대상지수명
                    "wonju_pric": str,  # 원주가격
                    "etftxon_type": str,  # ETF과세유형
                    "etntxon_type": str,  # ETN과세유형
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_stock_info_request_ka40002(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40002",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_daily_trend_request_ka40003(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 일별추이를 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 일별추이 데이터
                {
                    "etfdaly_trnsn": list,  # ETF일별추이 리스트
                        [
                            {
                                "cntr_dt": str,  # 체결일자
                                "cur_prc": str,  # 현재가
                                "pre_sig": str,  # 대비기호
                                "pred_pre": str,  # 전일대비
                                "pre_rt": str,  # 대비율
                                "trde_qty": str,  # 거래량
                                "nav": str,  # NAV
                                "acc_trde_prica": str,  # 누적거래대금
                                "navidex_dispty_rt": str,  # NAV/지수괴리율
                                "navetfdispty_rt": str,  # NAV/ETF괴리율
                                "trace_eor_rt": str,  # 추적오차율
                                "trace_cur_prc": str,  # 추적현재가
                                "trace_pred_pre": str,  # 추적전일대비
                                "trace_pre_sig": str,  # 추적대비기호
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_daily_trend_request_ka40003(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40003",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_overall_market_price_request_ka40004(
        self,
        tax_type: str = "0",
        nav_pre: str = "0",
        management_company: str = "0000",
        tax_yn: str = "0",
        trace_index: str = "0",
        exchange_type: str = "1",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 전체시세를 조회합니다.

        Args:
            tax_type (str, optional): 과세유형. Defaults to "0".
                - "0": 전체
                - "1": 비과세
                - "2": 보유기간과세
                - "3": 회사형
                - "4": 외국
                - "5": 비과세해외(보유기간관세)
            nav_pre (str, optional): NAV대비. Defaults to "0".
                - "0": 전체
                - "1": NAV > 전일종가
                - "2": NAV < 전일종가
            management_company (str, optional): 운용사. Defaults to "0000".
                - "0000": 전체
                - "3020": KODEX(삼성)
                - "3027": KOSEF(키움)
                - "3191": TIGER(미래에셋)
                - "3228": KINDEX(한국투자)
                - "3023": KStar(KB)
                - "3022": 아리랑(한화)
                - "9999": 기타운용사
            tax_yn (str, optional): 과세여부. Defaults to "0".
                - "0": 전체
                - "1": 과세
                - "2": 비과세
            trace_index (str, optional): 추적지수. Defaults to "0".
                - "0": 전체
            exchange_type (str, optional): 거래소구분. Defaults to "1".
                - "1": KRX
                - "2": NXT
                - "3": 통합
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 전체시세 데이터
                {
                    "etfall_mrpr": list,  # ETF전체시세 리스트
                        [
                            {
                                "stk_cd": str,  # 종목코드
                                "stk_cls": str,  # 종목분류
                                "stk_nm": str,  # 종목명
                                "close_pric": str,  # 종가
                                "pre_sig": str,  # 대비기호
                                "pred_pre": str,  # 전일대비
                                "pre_rt": str,  # 대비율
                                "trde_qty": str,  # 거래량
                                "nav": str,  # NAV
                                "trace_eor_rt": str,  # 추적오차율
                                "txbs": str,  # 과표기준
                                "dvid_bf_base": str,  # 배당전기준
                                "pred_dvida": str,  # 전일배당금
                                "trace_idex_nm": str,  # 추적지수명
                                "drng": str,  # 배수
                                "trace_idex_cd": str,  # 추적지수코드
                                "trace_idex": str,  # 추적지수
                                "trace_flu_rt": str,  # 추적등락율
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_overall_market_price_request_ka40004(
            ...     tax_type="0",
            ...     nav_pre="0",
            ...     management_company="0000",
            ...     tax_yn="0",
            ...     trace_index="0",
            ...     exchange_type="1"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40004",
        }

        data = {
            "txon_type": tax_type,
            "navpre": nav_pre,
            "mngmcomp": management_company,
            "txon_yn": tax_yn,
            "trace_idex": trace_index,
            "stex_tp": exchange_type,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_time_segment_trend_request_ka40006(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 시간대별추이를 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 시간대별추이 데이터
                {
                    "stk_nm": str,  # 종목명
                    "etfobjt_idex_nm": str,  # ETF대상지수명
                    "wonju_pric": str,  # 원주가격
                    "etftxon_type": str,  # ETF과세유형
                    "etntxon_type": str,  # ETN과세유형
                    "etftisl_trnsn": list,  # ETF시간대별추이 리스트
                        [
                            {
                                "tm": str,  # 시간
                                "close_pric": str,  # 종가
                                "pre_sig": str,  # 대비기호
                                "pred_pre": str,  # 전일대비
                                "flu_rt": str,  # 등락율
                                "trde_qty": str,  # 거래량
                                "nav": str,  # NAV
                                "trde_prica": str,  # 거래대금
                                "navidex": str,  # NAV지수
                                "navetf": str,  # NAVETF
                                "trace": str,  # 추적
                                "trace_idex": str,  # 추적지수
                                "trace_idex_pred_pre": str,  # 추적지수전일대비
                                "trace_idex_pred_pre_sig": str,  # 추적지수전일대비기호
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_time_segment_trend_request_ka40006(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40006",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_time_segment_execution_request_ka40007(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 시간대별체결을 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 시간대별체결 데이터
                {
                    "stk_cls": str,  # 종목분류
                    "stk_nm": str,  # 종목명
                    "etfobjt_idex_nm": str,  # ETF대상지수명
                    "etfobjt_idex_cd": str,  # ETF대상지수코드
                    "objt_idex_pre_rt": str,  # 대상지수대비율
                    "wonju_pric": str,  # 원주가격
                    "etftisl_cntr_array": list,  # ETF시간대별체결배열
                        [
                            {
                                "cntr_tm": str,  # 체결시간
                                "cur_prc": str,  # 현재가
                                "pre_sig": str,  # 대비기호
                                "pred_pre": str,  # 전일대비
                                "trde_qty": str,  # 거래량
                                "stex_tp": str,  # 거래소구분 (KRX, NXT, 통합)
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_time_segment_execution_request_ka40007(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40007",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_datewise_execution_request_ka40008(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 일자별체결을 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 일자별체결 데이터
                {
                    "cntr_tm": str,  # 체결시간
                    "cur_prc": str,  # 현재가
                    "pre_sig": str,  # 대비기호
                    "pred_pre": str,  # 전일대비
                    "trde_qty": str,  # 거래량
                    "etfnetprps_qty_array": list,  # ETF순매수수량배열
                        [
                            {
                                "dt": str,  # 일자
                                "cur_prc_n": str,  # 현재가n
                                "pre_sig_n": str,  # 대비기호n
                                "pred_pre_n": str,  # 전일대비n
                                "acc_trde_qty": str,  # 누적거래량
                                "for_netprps_qty": str,  # 외인순매수수량
                                "orgn_netprps_qty": str,  # 기관순매수수량
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_datewise_execution_request_ka40008(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40008",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_timewise_execution_request_ka40009(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 시간대별NAV를 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 시간대별NAV 데이터
                {
                    "etfnavarray": list,  # ETFNAV배열
                        [
                            {
                                "nav": str,  # NAV
                                "navpred_pre": str,  # NAV전일대비
                                "navflu_rt": str,  # NAV등락율
                                "trace_eor_rt": str,  # 추적오차율
                                "dispty_rt": str,  # 괴리율
                                "stkcnt": str,  # 주식수
                                "base_pric": str,  # 기준가
                                "for_rmnd_qty": str,  # 외인보유수량
                                "repl_pric": str,  # 대용가
                                "conv_pric": str,  # 환산가격
                                "drstk": str,  # DR/주
                                "wonju_pric": str,  # 원주가격
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_timewise_execution_request_ka40009(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40009",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def etf_timewise_trend_request_ka40010(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ETF 시간대별추이를 조회합니다.

        Args:
            stock_code (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ETF 시간대별추이 데이터
                {
                    "etftisl_trnsn": list,  # ETF시간대별추이 리스트
                        [
                            {
                                "cur_prc": str,  # 현재가
                                "pre_sig": str,  # 대비기호
                                "pred_pre": str,  # 전일대비
                                "trde_qty": str,  # 거래량
                                "for_netprps": str,  # 외인순매수
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.etf_timewise_trend_request_ka40010(
            ...     stock_code="069500"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka40010",
        }

        data = {
            "stk_cd": stock_code,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )