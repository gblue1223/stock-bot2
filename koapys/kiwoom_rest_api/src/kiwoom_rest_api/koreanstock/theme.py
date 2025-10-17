from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class Theme(KiwoomBaseAPI):
    """한국 주식 테마 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/thme"
    ):
        """
        Theme 클래스 초기화
        
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
        
   
    def theme_group_list_request_ka90001(
        self,
        qry_tp: str,
        date_tp: str,
        flu_pl_amt_tp: str,
        stex_tp: str,
        stk_cd: str = "",
        thema_nm: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """테마그룹별 조회를 요청합니다.

        Args:
            qry_tp (str): 검색구분 (0:전체검색, 1:테마검색, 2:종목검색)
            date_tp (str): 날짜구분 n일전 (1일 ~ 99일 날짜입력)
            flu_pl_amt_tp (str): 등락수익구분 (1:상위기간수익률, 2:하위기간수익률, 3:상위등락률, 4:하위등락률)
            stex_tp (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            stk_cd (str, optional): 검색하려는 종목코드. Defaults to "".
            thema_nm (str, optional): 검색하려는 테마명. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 테마그룹별 데이터
                {
                    "thema_grp": list,  # 테마그룹별 리스트
                        [
                            {
                                "thema_grp_cd": str,  # 테마그룹코드
                                "thema_nm": str,  # 테마명
                                "stk_num": str,  # 종목수
                                "flu_sig": str,  # 등락기호
                                "flu_rt": str,  # 등락율
                                "rising_stk_num": str,  # 상승종목수
                                "fall_stk_num": str,  # 하락종목수
                                "dt_prft_rt": str,  # 기간수익률
                                "main_stk": str,  # 주요종목
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.theme_group_list_request_ka90001(
            ...     qry_tp="0",
            ...     date_tp="10",
            ...     flu_pl_amt_tp="1",
            ...     stex_tp="1"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90001",
        }

        data = {
            "qry_tp": qry_tp,
            "stk_cd": stk_cd,
            "date_tp": date_tp,
            "thema_nm": thema_nm,
            "flu_pl_amt_tp": flu_pl_amt_tp,
            "stex_tp": stex_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def theme_component_stocks_request_ka90002(
        self,
        thema_grp_cd: str,
        stex_tp: str,
        date_tp: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """테마구성종목 조회를 요청합니다.

        Args:
            thema_grp_cd (str): 테마그룹코드 번호
            stex_tp (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            date_tp (str, optional): 날짜구분 1일 ~ 99일 날짜입력. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 테마구성종목 데이터
                {
                    "flu_rt": str,  # 등락률
                    "dt_prft_rt": str,  # 기간수익률
                    "thema_comp_stk": list,  # 테마구성종목 리스트
                        [
                            {
                                "stk_cd": str,  # 종목코드
                                "stk_nm": str,  # 종목명
                                "cur_prc": str,  # 현재가
                                "flu_sig": str,  # 등락기호
                                "pred_pre": str,  # 전일대비
                                "flu_rt": str,  # 등락율
                                "acc_trde_qty": str,  # 누적거래량
                                "sel_bid": str,  # 매도호가
                                "sel_req": str,  # 매도잔량
                                "buy_bid": str,  # 매수호가
                                "buy_req": str,  # 매수잔량
                                "dt_prft_rt_n": str,  # 기간수익률n
                            }
                        ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.sector.theme_component_stocks_request_ka90002(
            ...     thema_grp_cd="100",
            ...     stex_tp="1",
            ...     date_tp="2"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90002",
        }

        data = {
            "date_tp": date_tp,
            "thema_grp_cd": thema_grp_cd,
            "stex_tp": stex_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )