from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class Order(KiwoomBaseAPI):
    """한국 주식 주문 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/ordr"
    ):
        """
        Order 클래스 초기화
        
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
        
    def stock_buy_order_request_kt10000(
        self,
        dmst_stex_tp: str,
        stk_cd: str,
        ord_qty: str,
        trde_tp: str,
        ord_uv: str = "",
        cond_uv: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 매수주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            stk_cd (str): 종목코드
            ord_qty (str): 주문수량
            trde_tp (str): 매매구분
                - 0: 보통
                - 3: 시장가
                - 5: 조건부지정가
                - 81: 장마감후시간외
                - 61: 장시작전시간외
                - 62: 시간외단일가
                - 6: 최유리지정가
                - 7: 최우선지정가
                - 10: 보통(IOC)
                - 13: 시장가(IOC)
                - 16: 최유리(IOC)
                - 20: 보통(FOK)
                - 23: 시장가(FOK)
                - 26: 최유리(FOK)
                - 28: 스톱지정가
                - 29: 중간가
                - 30: 중간가(IOC)
                - 31: 중간가(FOK)
            ord_uv (str, optional): 주문단가. Defaults to "".
            cond_uv (str, optional): 조건단가. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주문 응답 데이터
                {
                    "ord_no": str,  # 주문번호
                    "dmst_stex_tp": str,  # 국내거래소구분
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.order.stock_buy_order_request_kt10000(
            ...     dmst_stex_tp="KRX",
            ...     stk_cd="005930",
            ...     ord_qty="1",
            ...     trde_tp="3"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10000",
        }

        data = {
            "dmst_stex_tp": dmst_stex_tp,
            "stk_cd": stk_cd,
            "ord_qty": ord_qty,
            "ord_uv": ord_uv,
            "trde_tp": trde_tp,
            "cond_uv": cond_uv,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_sell_order_request_kt10001(
        self,
        dmst_stex_tp: str,
        stk_cd: str,
        ord_qty: str,
        trde_tp: str,
        ord_uv: str = "",
        cond_uv: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 매도주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            stk_cd (str): 종목코드
            ord_qty (str): 주문수량
            trde_tp (str): 매매구분
                - 0: 보통
                - 3: 시장가
                - 5: 조건부지정가
                - 81: 장마감후시간외
                - 61: 장시작전시간외
                - 62: 시간외단일가
                - 6: 최유리지정가
                - 7: 최우선지정가
                - 10: 보통(IOC)
                - 13: 시장가(IOC)
                - 16: 최유리(IOC)
                - 20: 보통(FOK)
                - 23: 시장가(FOK)
                - 26: 최유리(FOK)
                - 28: 스톱지정가
                - 29: 중간가
                - 30: 중간가(IOC)
                - 31: 중간가(FOK)
            ord_uv (str, optional): 주문단가. Defaults to "".
            cond_uv (str, optional): 조건단가. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주문 응답 데이터
                {
                    "ord_no": str,  # 주문번호
                    "dmst_stex_tp": str,  # 국내거래소구분
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.order.stock_sell_order_request_kt10001(
            ...     dmst_stex_tp="KRX",
            ...     stk_cd="005930",
            ...     ord_qty="1",
            ...     trde_tp="3"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10001",
        }

        data = {
            "dmst_stex_tp": dmst_stex_tp,
            "stk_cd": stk_cd,
            "ord_qty": ord_qty,
            "ord_uv": ord_uv,
            "trde_tp": trde_tp,
            "cond_uv": cond_uv,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_modify_order_request_kt10002(
        self,
        dmst_stex_tp: str,
        orig_ord_no: str,
        stk_cd: str,
        mdfy_qty: str,
        mdfy_uv: str,
        mdfy_cond_uv: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 정정주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            orig_ord_no (str): 원주문번호
            stk_cd (str): 종목코드
            mdfy_qty (str): 정정수량
            mdfy_uv (str): 정정단가
            mdfy_cond_uv (str, optional): 정정조건단가. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 정정주문 응답 데이터
                {
                    "ord_no": str,  # 주문번호
                    "base_orig_ord_no": str,  # 모주문번호
                    "mdfy_qty": str,  # 정정수량
                    "dmst_stex_tp": str,  # 국내거래소구분
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.order.stock_modify_order_request_kt10002(
            ...     dmst_stex_tp="KRX",
            ...     orig_ord_no="0000139",
            ...     stk_cd="005930",
            ...     mdfy_qty="1",
            ...     mdfy_uv="199700"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10002",
        }

        data = {
            "dmst_stex_tp": dmst_stex_tp,
            "orig_ord_no": orig_ord_no,
            "stk_cd": stk_cd,
            "mdfy_qty": mdfy_qty,
            "mdfy_uv": mdfy_uv,
            "mdfy_cond_uv": mdfy_cond_uv,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_cancel_order_request_kt10003(
        self,
        dmst_stex_tp: str,
        orig_ord_no: str,
        stk_cd: str,
        cncl_qty: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """주식 취소주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            orig_ord_no (str): 원주문번호
            stk_cd (str): 종목코드
            cncl_qty (str): 취소수량 ('0' 입력시 잔량 전부 취소)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 취소주문 응답 데이터
                {
                    "ord_no": str,  # 주문번호
                    "base_orig_ord_no": str,  # 모주문번호
                    "cncl_qty": str,  # 취소수량
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.order.stock_cancel_order_request_kt10003(
            ...     dmst_stex_tp="KRX",
            ...     orig_ord_no="0000140",
            ...     stk_cd="005930",
            ...     cncl_qty="1"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10003",
        }

        data = {
            "dmst_stex_tp": dmst_stex_tp,
            "orig_ord_no": orig_ord_no,
            "stk_cd": stk_cd,
            "cncl_qty": cncl_qty,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )