from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class CreditOrder(KiwoomBaseAPI):
    """한국 주식 신용매매 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/crdordr"
    ):
        """
        CreditOrder 클래스 초기화
        
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

        
    def margin_buy_order_request_kt10006(
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
        """신용 매수주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            stk_cd (str): 종목코드
            ord_qty (str): 주문수량
            trde_tp (str): 매매구분 (0:보통, 3:시장가, 5:조건부지정가, 81:장마감후시간외, 61:장시작전시간외, 62:시간외단일가, 6:최유리지정가, 7:최우선지정가, 10:보통(IOC), 13:시장가(IOC), 16:최유리(IOC), 20:보통(FOK), 23:시장가(FOK), 26:최유리(FOK), 28:스톱지정가, 29:중간가, 30:중간가(IOC), 31:중간가(FOK))
            ord_uv (str, optional): 주문단가. Defaults to "".
            cond_uv (str, optional): 조건단가. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 신용 매수주문 결과
                {
                    "ord_no": str,  # 주문번호
                    "dmst_stex_tp": str,  # 국내거래소구분
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.credit_order.margin_buy_order_request_kt10006(
            ...     dmst_stex_tp="KRX",
            ...     stk_cd="005930",
            ...     ord_qty="1",
            ...     ord_uv="2580",
            ...     trde_tp="0"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10006",
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
    
    def margin_sell_order_request_kt10007(
        self,
        dmst_stex_tp: str,
        stk_cd: str,
        ord_qty: str,
        trde_tp: str,
        crd_deal_tp: str,
        ord_uv: str = "",
        crd_loan_dt: str = "",
        cond_uv: str = "",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """신용 매도주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            stk_cd (str): 종목코드
            ord_qty (str): 주문수량
            trde_tp (str): 매매구분 (0:보통, 3:시장가, 5:조건부지정가, 81:장마감후시간외, 61:장시작전시간외, 62:시간외단일가, 6:최유리지정가, 7:최우선지정가, 10:보통(IOC), 13:시장가(IOC), 16:최유리(IOC), 20:보통(FOK), 23:시장가(FOK), 26:최유리(FOK), 28:스톱지정가, 29:중간가, 30:중간가(IOC), 31:중간가(FOK))
            crd_deal_tp (str): 신용거래구분 (33:융자, 99:융자합)
            ord_uv (str, optional): 주문단가. Defaults to "".
            crd_loan_dt (str, optional): 대출일 YYYYMMDD(융자일경우필수). Defaults to "".
            cond_uv (str, optional): 조건단가. Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 신용 매도주문 결과
                {
                    "ord_no": str,  # 주문번호
                    "dmst_stex_tp": str,  # 국내거래소구분
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.credit_order.margin_sell_order_request_kt10007(
            ...     dmst_stex_tp="KRX",
            ...     stk_cd="005930",
            ...     ord_qty="3",
            ...     ord_uv="6450",
            ...     trde_tp="0",
            ...     crd_deal_tp="99"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10007",
        }

        data = {
            "dmst_stex_tp": dmst_stex_tp,
            "stk_cd": stk_cd,
            "ord_qty": ord_qty,
            "ord_uv": ord_uv,
            "trde_tp": trde_tp,
            "crd_deal_tp": crd_deal_tp,
            "crd_loan_dt": crd_loan_dt,
            "cond_uv": cond_uv,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def margin_modify_order_request_kt10008(
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
        """신용 정정주문을 요청합니다.

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
            dict: 신용 정정주문 결과
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
            >>> result = api.credit_order.margin_modify_order_request_kt10008(
            ...     dmst_stex_tp="KRX",
            ...     orig_ord_no="0000455",
            ...     stk_cd="005930",
            ...     mdfy_qty="1",
            ...     mdfy_uv="2590"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10008",
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
        
    def margin_cancel_order_request_kt10009(
        self,
        dmst_stex_tp: str,
        orig_ord_no: str,
        stk_cd: str,
        cncl_qty: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """신용 취소주문을 요청합니다.

        Args:
            dmst_stex_tp (str): 국내거래소구분 (KRX, NXT, SOR)
            orig_ord_no (str): 원주문번호
            stk_cd (str): 종목코드
            cncl_qty (str): 취소수량 ('0' 입력시 잔량 전부 취소)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 신용 취소주문 결과
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
            >>> result = api.credit_order.margin_cancel_order_request_kt10009(
            ...     dmst_stex_tp="KRX",
            ...     orig_ord_no="0001615",
            ...     stk_cd="005930",
            ...     cncl_qty="1"
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "kt10009",
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