from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class Chart(KiwoomBaseAPI):
    """한국 주식 섹터 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/chart"
    ):
        """
        Chart 클래스 초기화
        
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
        
    def stockwise_investor_institution_chart_request_ka10060(
        self,
        dt: str,
        stk_cd: str,
        amt_qty_tp: str,
        trde_tp: str,
        unit_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        종목별투자자기관별차트요청 (ka10060)

        Args:
            dt (str): 일자 (YYYYMMDD)
            stk_cd (str): 종목코드
            amt_qty_tp (str): 금액수량구분 (1:금액, 2:수량)
            trde_tp (str): 매매구분 (0:순매수, 1:매수, 2:매도)
            unit_tp (str): 단위구분 (1000:천주, 1:단주)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 종목별투자자기관별차트 데이터
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10060",
        }
        data = {
            "dt": dt,
            "stk_cd": stk_cd,
            "amt_qty_tp": amt_qty_tp,
            "trde_tp": trde_tp,
            "unit_tp": unit_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def intraday_investor_trading_chart_request_ka10064(
        self,
        mrkt_tp: str,
        amt_qty_tp: str,
        trde_tp: str,
        stk_cd: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        장중투자자별매매차트요청 (ka10064)

        Args:
            mrkt_tp (str): 시장구분 (000:전체, 001:코스피, 101:코스닥)
            amt_qty_tp (str): 금액수량구분 (1:금액, 2:수량)
            trde_tp (str): 매매구분 (0:순매수, 1:매수, 2:매도)
            stk_cd (str): 종목코드
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 장중투자자별매매차트 데이터
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10064",
        }
        data = {
            "mrkt_tp": mrkt_tp,
            "amt_qty_tp": amt_qty_tp,
            "trde_tp": trde_tp,
            "stk_cd": stk_cd,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_tick_chart_request_ka10079(
        self,
        stk_cd: str,
        tic_scope: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식틱차트조회요청 (ka10079)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            tic_scope (str): 틱범위 (1:1틱, 3:3틱, 5:5틱, 10:10틱, 30:30틱)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식틱차트 데이터
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10079",
        }
        data = {
            "stk_cd": stk_cd,
            "tic_scope": tic_scope,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_minute_chart_request_ka10080(
        self,
        stk_cd: str,
        tic_scope: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식분봉차트조회요청 (ka10080)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            tic_scope (str): 틱범위 (1:1분, 3:3분, 5:5분, 10:10분, 15:15분, 30:30분, 45:45분, 60:60분)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식분봉차트 데이터
                - stk_cd (str): 종목코드
                - stk_min_pole_chart_qry (list): 주식분봉차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - cntr_tm (str): 체결시간
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - upd_stkpc_tp (str): 수정주가구분
                    - upd_rt (str): 수정비율
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - upd_stkpc_event (str): 수정주가이벤트
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10080",
        }
        data = {
            "stk_cd": stk_cd,
            "tic_scope": tic_scope,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_daily_chart_request_ka10081(
        self,
        stk_cd: str,
        base_dt: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식일봉차트조회요청 (ka10081)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            base_dt (str): 기준일자 (YYYYMMDD)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식일봉차트 데이터
                - stk_cd (str): 종목코드
                - stk_dt_pole_chart_qry (list): 주식일봉차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - trde_prica (str): 거래대금
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - upd_stkpc_tp (str): 수정주가구분
                    - upd_rt (str): 수정비율
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - upd_stkpc_event (str): 수정주가이벤트
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10081",
        }
        data = {
            "stk_cd": stk_cd,
            "base_dt": base_dt,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_weekly_chart_request_ka10082(
        self,
        stk_cd: str,
        base_dt: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식주봉차트조회요청 (ka10082)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            base_dt (str): 기준일자 (YYYYMMDD)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식주봉차트 데이터
                - stk_cd (str): 종목코드
                - stk_stk_pole_chart_qry (list): 주식주봉차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - trde_prica (str): 거래대금
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - upd_stkpc_tp (str): 수정주가구분 (1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락)
                    - upd_rt (str): 수정비율
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - upd_stkpc_event (str): 수정주가이벤트
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10082",
        }
        data = {
            "stk_cd": stk_cd,
            "base_dt": base_dt,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_monthly_chart_request_ka10083(
        self,
        stk_cd: str,
        base_dt: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식월봉차트조회요청 (ka10083)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            base_dt (str): 기준일자 (YYYYMMDD)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식월봉차트 데이터
                - stk_cd (str): 종목코드
                - stk_mth_pole_chart_qry (list): 주식월봉차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - trde_prica (str): 거래대금
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - upd_stkpc_tp (str): 수정주가구분 (1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락)
                    - upd_rt (str): 수정비율
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - upd_stkpc_event (str): 수정주가이벤트
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10083",
        }
        data = {
            "stk_cd": stk_cd,
            "base_dt": base_dt,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def stock_yearly_chart_request_ka10094(
        self,
        stk_cd: str,
        base_dt: str,
        upd_stkpc_tp: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        주식년봉차트조회요청 (ka10094)

        Args:
            stk_cd (str): 종목코드 (거래소별 종목코드 KRX:039490,NXT:039490_NX,SOR:039490_AL)
            base_dt (str): 기준일자 (YYYYMMDD)
            upd_stkpc_tp (str): 수정주가구분 (0 or 1)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 주식년봉차트 데이터
                - stk_cd (str): 종목코드
                - stk_yr_pole_chart_qry (list): 주식년봉차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - trde_prica (str): 거래대금
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - upd_stkpc_tp (str): 수정주가구분 (1:유상증자, 2:무상증자, 4:배당락, 8:액면분할, 16:액면병합, 32:기업합병, 64:감자, 256:권리락)
                    - upd_rt (str): 수정비율
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - upd_stkpc_event (str): 수정주가이벤트
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10094",
        }
        data = {
            "stk_cd": stk_cd,
            "base_dt": base_dt,
            "upd_stkpc_tp": upd_stkpc_tp,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_tick_chart_request_ka20004(
        self,
        inds_cd: str,
        tic_scope: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종틱차트조회요청 (ka20004)

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
            tic_scope (str): 틱범위 (1:1틱, 3:3틱, 5:5틱, 10:10틱, 30:30틱)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종틱차트 데이터
                - inds_cd (str): 업종코드
                - inds_tic_chart_qry (list): 업종틱차트조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - cntr_tm (str): 체결시간
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20004",
        }
        data = {
            "inds_cd": inds_cd,
            "tic_scope": tic_scope,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_minute_chart_request_ka20005(
        self,
        inds_cd: str,
        tic_scope: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종분봉조회요청 (ka20005)

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
            tic_scope (str): 틱범위 (1:1분, 3:3분, 5:5분, 10:10분, 15:15분, 30:30분, 45:45분, 60:60분)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종분봉차트 데이터
                - inds_cd (str): 업종코드
                - inds_min_pole_qry (list): 업종분봉조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - cntr_tm (str): 체결시간
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20005",
        }
        data = {
            "inds_cd": inds_cd,
            "tic_scope": tic_scope,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_daily_chart_request_ka20006(
        self,
        inds_cd: str,
        base_dt: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종일봉조회요청 (ka20006)

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
            base_dt (str): 기준일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종일봉차트 데이터
                - inds_cd (str): 업종코드
                - inds_dt_pole_qry (list): 업종일봉조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - trde_prica (str): 거래대금
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20006",
        }
        data = {
            "inds_cd": inds_cd,
            "base_dt": base_dt,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_weekly_chart_request_ka20007(
        self,
        inds_cd: str,
        base_dt: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종주봉조회요청 (ka20007)

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
            base_dt (str): 기준일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종주봉차트 데이터
                - inds_cd (str): 업종코드
                - inds_stk_pole_qry (list): 업종주봉조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - trde_prica (str): 거래대금
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20007",
        }
        data = {
            "inds_cd": inds_cd,
            "base_dt": base_dt,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_monthly_chart_request_ka20008(
        self,
        inds_cd: str,
        base_dt: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종월봉조회요청 (ka20008)

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
            base_dt (str): 기준일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종월봉차트 데이터
                - inds_cd (str): 업종코드
                - inds_mth_pole_qry (list): 업종월봉조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - trde_prica (str): 거래대금
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20008",
        }
        data = {
            "inds_cd": inds_cd,
            "base_dt": base_dt,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def industry_yearly_chart_request_ka20019(
        self,
        inds_cd: str,
        base_dt: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> dict:
        """
        업종년봉조회요청 (ka20019)

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
            base_dt (str): 기준일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 업종년봉차트 데이터
                - inds_cd (str): 업종코드
                - inds_yr_pole_qry (list): 업종년봉조회 데이터 리스트
                    - cur_prc (str): 현재가
                    - trde_qty (str): 거래량
                    - dt (str): 일자
                    - open_pric (str): 시가
                    - high_pric (str): 고가
                    - low_pric (str): 저가
                    - trde_prica (str): 거래대금
                    - bic_inds_tp (str): 대업종구분
                    - sm_inds_tp (str): 소업종구분
                    - stk_infr (str): 종목정보
                    - pred_close_pric (str): 전일종가
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka20019",
        }
        data = {
            "inds_cd": inds_cd,
            "base_dt": base_dt,
        }
        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )