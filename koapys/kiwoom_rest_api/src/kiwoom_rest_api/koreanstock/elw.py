from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class ELW(KiwoomBaseAPI):
    """한국 주식 ELW 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/elw"
    ):
        """
        ELW 클래스 초기화
        
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
        
          
    def elw_daily_sensitivity_indicator_request_ka10048(
        self,
        stk_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 일별 민감도 지표를 조회합니다.

        Args:
            stk_cd (str): 종목코드 (예: "57JBHH")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 일별 민감도 지표 데이터
                {
                    "elwdaly_snst_ix": [  # ELW일별민감도지표
                        {
                            "dt": str,  # 일자
                            "iv": str,  # IV (Implied Volatility)
                            "delta": str,  # 델타
                            "gam": str,  # 감마
                            "theta": str,  # 쎄타
                            "vega": str,  # 베가
                            "law": str,  # 로
                            "lp": str,  # LP
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_daily_sensitivity_indicator_request_ka10048(
            ...     stk_cd="57JBHH"  # ELW 종목코드
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10048",
        }

        data = {
            "stk_cd": stk_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_sensitivity_indicator_request_ka10050(
        self,
        stk_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 민감도 지표를 조회합니다.

        Args:
            stk_cd (str): 종목코드 (예: "57JBHH")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 민감도 지표 데이터
                {
                    "elwsnst_ix_array": [  # ELW민감도지표배열
                        {
                            "cntr_tm": str,  # 체결시간
                            "cur_prc": str,  # 현재가
                            "elwtheory_pric": str,  # ELW이론가
                            "iv": str,  # IV (Implied Volatility)
                            "delta": str,  # 델타
                            "gam": str,  # 감마
                            "theta": str,  # 쎄타
                            "vega": str,  # 베가
                            "law": str,  # 로
                            "lp": str,  # LP
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_sensitivity_indicator_request_ka10050(
            ...     stk_cd="57JBHH"  # ELW 종목코드
            ... )
            >>> print(result)
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10050",
        }

        data = {
            "stk_cd": stk_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_price_spike_request_ka30001(
        self,
        flu_tp: str,
        tm_tp: str,
        tm: str,
        trde_qty_tp: str,
        isscomp_cd: str,
        bsis_aset_cd: str,
        rght_tp: str,
        lpcd: str,
        trde_end_elwskip: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 가격 급등락 정보를 조회합니다.

        Args:
            flu_tp (str): 등락구분 (1:급등, 2:급락)
            tm_tp (str): 시간구분 (1:분전, 2:일전)
            tm (str): 시간 (분 혹은 일입력, 예: 1, 3, 5)
            trde_qty_tp (str): 거래량구분
                - 0: 전체
                - 10: 만주이상
                - 50: 5만주이상
                - 100: 10만주이상
                - 300: 30만주이상
                - 500: 50만주이상
                - 1000: 백만주이상
            isscomp_cd (str): 발행사코드
                - 000000000000: 전체
                - 3: 한국투자증권
                - 5: 미래대우
                - 6: 신영
                - 12: NK투자증권
                - 17: KB증권
            bsis_aset_cd (str): 기초자산코드
                - 000000000000: 전체
                - 201: KOSPI200
                - 150: KOSDAQ150
                - 005930: 삼성전자
                - 030200: KT
            rght_tp (str): 권리구분
                - 000: 전체
                - 001: 콜
                - 002: 풋
                - 003: DC
                - 004: DP
                - 005: EX
                - 006: 조기종료콜
                - 007: 조기종료풋
            lpcd (str): LP코드
                - 000000000000: 전체
                - 3: 한국투자증권
                - 5: 미래대우
                - 6: 신영
                - 12: NK투자증권
                - 17: KB증권
            trde_end_elwskip (str): 거래종료ELW제외 (0:포함, 1:제외)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 가격 급등락 데이터
                {
                    "base_pric_tm": str,  # 기준가시간
                    "elwpric_jmpflu": [  # ELW가격급등락
                        {
                            "stk_cd": str,  # 종목코드
                            "rank": str,  # 순위
                            "stk_nm": str,  # 종목명
                            "pre_sig": str,  # 대비기호
                            "pred_pre": str,  # 전일대비
                            "trde_end_elwbase_pric": str,  # 거래종료ELW기준가
                            "cur_prc": str,  # 현재가
                            "base_pre": str,  # 기준대비
                            "trde_qty": str,  # 거래량
                            "jmp_rt": str,  # 급등율
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_price_spike_request_ka30001(
            ...     flu_tp="1",  # 급등
            ...     tm_tp="2",  # 일전
            ...     tm="1",  # 1일
            ...     trde_qty_tp="0",  # 전체
            ...     isscomp_cd="000000000000",  # 전체
            ...     bsis_aset_cd="000000000000",  # 전체
            ...     rght_tp="000",  # 전체
            ...     lpcd="000000000000",  # 전체
            ...     trde_end_elwskip="0"  # 포함
            ... )
        """
        # 파라미터 유효성 검증
        if flu_tp not in ["1", "2"]:
            raise ValueError("flu_tp must be '1' (급등) or '2' (급락)")
        if tm_tp not in ["1", "2"]:
            raise ValueError("tm_tp must be '1' (분전) or '2' (일전)")
        if not tm.isdigit() or len(tm) > 2:
            raise ValueError("tm must be a 1-2 digit number")
        if trde_qty_tp not in ["0", "10", "50", "100", "300", "500", "1000"]:
            raise ValueError("Invalid trde_qty_tp value")
        if not isscomp_cd.isdigit() or len(isscomp_cd) != 12:
            raise ValueError("isscomp_cd must be a 12-digit number")
        if not bsis_aset_cd.isdigit() or len(bsis_aset_cd) != 12:
            raise ValueError("bsis_aset_cd must be a 12-digit number")
        if rght_tp not in ["000", "001", "002", "003", "004", "005", "006", "007"]:
            raise ValueError("Invalid rght_tp value")
        if not lpcd.isdigit() or len(lpcd) != 12:
            raise ValueError("lpcd must be a 12-digit number")
        if trde_end_elwskip not in ["0", "1"]:
            raise ValueError("trde_end_elwskip must be '0' (포함) or '1' (제외)")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30001",
        }

        data = {
            "flu_tp": flu_tp,
            "tm_tp": tm_tp,
            "tm": tm,
            "trde_qty_tp": trde_qty_tp,
            "isscomp_cd": isscomp_cd,
            "bsis_aset_cd": bsis_aset_cd,
            "rght_tp": rght_tp,
            "lpcd": lpcd,
            "trde_end_elwskip": trde_end_elwskip,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def top_elw_net_buying_by_broker_request_ka30002(
        self,
        isscomp_cd: str,
        trde_qty_tp: str,
        trde_tp: str,
        dt: str,
        trde_end_elwskip: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """거래원별 ELW 순매매 상위 정보를 조회합니다.

        Args:
            isscomp_cd (str): 발행사코드 (3자리)
                - 001: 교보
                - 002: 신한금융투자
                - 003: 한국투자증권
                - 004: 대신
                - 005: 미래대우
                - 기타: 영웅문4 0273화면 참조
            trde_qty_tp (str): 거래량구분
                - 0: 전체
                - 5: 5천주
                - 10: 만주
                - 50: 5만주
                - 100: 10만주
                - 500: 50만주
                - 1000: 백만주
            trde_tp (str): 매매구분
                - 1: 순매수
                - 2: 순매도
            dt (str): 기간
                - 1: 전일
                - 5: 5일
                - 10: 10일
                - 40: 40일
                - 60: 60일
            trde_end_elwskip (str): 거래종료ELW제외
                - 0: 포함
                - 1: 제외
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: 거래원별 ELW 순매매 상위 데이터
                {
                    "trde_ori_elwnettrde_upper": [  # 거래원별ELW순매매상위
                        {
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "stkpc_flu": str,  # 주가등락
                            "flu_rt": str,  # 등락율
                            "trde_qty": str,  # 거래량
                            "netprps": str,  # 순매수
                            "buy_trde_qty": str,  # 매수거래량
                            "sel_trde_qty": str,  # 매도거래량
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.top_elw_net_buying_by_broker_request_ka30002(
            ...     isscomp_cd="003",  # 한국투자증권
            ...     trde_qty_tp="0",  # 전체
            ...     trde_tp="2",  # 순매도
            ...     dt="60",  # 60일
            ...     trde_end_elwskip="0"  # 포함
            ... )
        """
        # 파라미터 유효성 검증
        if not isscomp_cd.isdigit() or len(isscomp_cd) != 3:
            raise ValueError("isscomp_cd must be a 3-digit number")
        if trde_qty_tp not in ["0", "5", "10", "50", "100", "500", "1000"]:
            raise ValueError("Invalid trde_qty_tp value")
        if trde_tp not in ["1", "2"]:
            raise ValueError("trde_tp must be '1' (순매수) or '2' (순매도)")
        if dt not in ["1", "5", "10", "40", "60"]:
            raise ValueError("dt must be one of: '1', '5', '10', '40', '60'")
        if trde_end_elwskip not in ["0", "1"]:
            raise ValueError("trde_end_elwskip must be '0' (포함) or '1' (제외)")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30002",
        }

        data = {
            "isscomp_cd": isscomp_cd,
            "trde_qty_tp": trde_qty_tp,
            "trde_tp": trde_tp,
            "dt": dt,
            "trde_end_elwskip": trde_end_elwskip,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_lp_daily_holding_trend_request_ka30003(
        self,
        bsis_aset_cd: str,
        base_dt: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW LP 보유 일별 추이 정보를 조회합니다.

        Args:
            bsis_aset_cd (str): 기초자산코드 (12자리)
            base_dt (str): 기준일자 (YYYYMMDD 형식)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW LP 보유 일별 추이 데이터
                {
                    "elwlpposs_daly_trnsn": [  # ELWLP보유일별추이
                        {
                            "dt": str,  # 일자
                            "cur_prc": str,  # 현재가
                            "pre_tp": str,  # 대비구분
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락율
                            "trde_qty": str,  # 거래량
                            "trde_prica": str,  # 거래대금
                            "chg_qty": str,  # 변동수량
                            "lprmnd_qty": str,  # LP보유수량
                            "wght": str,  # 비중
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_lp_daily_holding_trend_request_ka30003(
            ...     bsis_aset_cd="57KJ99",  # 기초자산코드
            ...     base_dt="20241122"  # 기준일자
            ... )
        """
        # 파라미터 유효성 검증
        if not bsis_aset_cd or len(bsis_aset_cd) != 12:
            raise ValueError("bsis_aset_cd must be a 12-character string")
        
        # base_dt 형식 검증 (YYYYMMDD)
        if not base_dt.isdigit() or len(base_dt) != 8:
            raise ValueError("base_dt must be in YYYYMMDD format")
        try:
            year = int(base_dt[:4])
            month = int(base_dt[4:6])
            day = int(base_dt[6:8])
            if not (1 <= month <= 12 and 1 <= day <= 31):
                raise ValueError
        except ValueError:
            raise ValueError("base_dt must be a valid date in YYYYMMDD format")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30003",
        }

        data = {
            "bsis_aset_cd": bsis_aset_cd,
            "base_dt": base_dt,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_premium_rate_request_ka30004(
        self,
        isscomp_cd: str,
        bsis_aset_cd: str,
        rght_tp: str,
        lpcd: str,
        trde_end_elwskip: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 괴리율 정보를 조회합니다.

        Args:
            isscomp_cd (str): 발행사코드 (12자리)
                - 000000000000: 전체
                - 3: 한국투자증권
                - 5: 미래대우
                - 6: 신영
                - 12: NK투자증권
                - 17: KB증권
            bsis_aset_cd (str): 기초자산코드 (12자리)
                - 000000000000: 전체
                - 201: KOSPI200
                - 150: KOSDAQ150
                - 005930: 삼성전자
                - 030200: KT
            rght_tp (str): 권리구분 (3자리)
                - 000: 전체
                - 001: 콜
                - 002: 풋
                - 003: DC
                - 004: DP
                - 005: EX
                - 006: 조기종료콜
                - 007: 조기종료풋
            lpcd (str): LP코드 (12자리)
                - 000000000000: 전체
                - 3: 한국투자증권
                - 5: 미래대우
                - 6: 신영
                - 12: NK투자증권
                - 17: KB증권
            trde_end_elwskip (str): 거래종료ELW제외
                - 0: 거래종료ELW포함
                - 1: 거래종료ELW제외
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 괴리율 데이터
                {
                    "elwdispty_rt": [  # ELW괴리율
                        {
                            "stk_cd": str,  # 종목코드
                            "isscomp_nm": str,  # 발행사명
                            "sqnc": str,  # 회차
                            "base_aset_nm": str,  # 기초자산명
                            "rght_tp": str,  # 권리구분
                            "dispty_rt": str,  # 괴리율
                            "basis": str,  # 베이시스
                            "srvive_dys": str,  # 잔존일수
                            "theory_pric": str,  # 이론가
                            "cur_prc": str,  # 현재가
                            "pre_tp": str,  # 대비구분
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락율
                            "trde_qty": str,  # 거래량
                            "stk_nm": str,  # 종목명
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_premium_rate_request_ka30004(
            ...     isscomp_cd="000000000000",  # 전체
            ...     bsis_aset_cd="000000000000",  # 전체
            ...     rght_tp="000",  # 전체
            ...     lpcd="000000000000",  # 전체
            ...     trde_end_elwskip="0"  # 거래종료ELW포함
            ... )
        """
        # 파라미터 유효성 검증
        if not isscomp_cd.isdigit() or len(isscomp_cd) != 12:
            raise ValueError("isscomp_cd must be a 12-digit number")
        if not bsis_aset_cd.isdigit() or len(bsis_aset_cd) != 12:
            raise ValueError("bsis_aset_cd must be a 12-digit number")
        if rght_tp not in ["000", "001", "002", "003", "004", "005", "006", "007"]:
            raise ValueError("Invalid rght_tp value")
        if not lpcd.isdigit() or len(lpcd) != 12:
            raise ValueError("lpcd must be a 12-digit number")
        if trde_end_elwskip not in ["0", "1"]:
            raise ValueError("trde_end_elwskip must be '0' (포함) or '1' (제외)")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30004",
        }

        data = {
            "isscomp_cd": isscomp_cd,
            "bsis_aset_cd": bsis_aset_cd,
            "rght_tp": rght_tp,
            "lpcd": lpcd,
            "trde_end_elwskip": trde_end_elwskip,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_condition_search_request_ka30005(
        self,
        isscomp_cd: str,
        bsis_aset_cd: str,
        rght_tp: str,
        lpcd: str,
        sort_tp: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 조건검색 정보를 조회합니다.

        Args:
            isscomp_cd (str): 발행사코드 (12자리)
                - 000000000000: 전체
                - 000000000003: 한국투자증권
                - 000000000005: 미래대우
                - 000000000006: 신영
                - 000000000012: NK투자증권
                - 000000000017: KB증권
            bsis_aset_cd (str): 기초자산코드
                - 000000000000: 전체
                - 201: KOSPI200
                - 150: KOSDAQ150
                - 005930: 삼성전자
                - 030200: KT
            rght_tp (str): 권리구분
                - 0: 전체
                - 1: 콜
                - 2: 풋
                - 3: DC
                - 4: DP
                - 5: EX
                - 6: 조기종료콜
                - 7: 조기종료풋
            lpcd (str): LP코드 (12자리)
                - 000000000000: 전체
                - 000000000003: 한국투자증권
                - 000000000005: 미래대우
                - 000000000006: 신영
                - 000000000012: NK투자증권
                - 000000000017: KB증권
            sort_tp (str): 정렬구분
                - 0: 정렬없음
                - 1: 상승율순
                - 2: 상승폭순
                - 3: 하락율순
                - 4: 하락폭순
                - 5: 거래량순
                - 6: 거래대금순
                - 7: 잔존일순
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 조건검색 데이터
                {
                    "elwcnd_qry": [  # ELW조건검색
                        {
                            "stk_cd": str,  # 종목코드
                            "isscomp_nm": str,  # 발행사명
                            "sqnc": str,  # 회차
                            "base_aset_nm": str,  # 기초자산명
                            "rght_tp": str,  # 권리구분
                            "expr_dt": str,  # 만기일
                            "cur_prc": str,  # 현재가
                            "pre_tp": str,  # 대비구분
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락율
                            "trde_qty": str,  # 거래량
                            "trde_qty_pre": str,  # 거래량대비
                            "trde_prica": str,  # 거래대금
                            "pred_trde_qty": str,  # 전일거래량
                            "sel_bid": str,  # 매도호가
                            "buy_bid": str,  # 매수호가
                            "prty": str,  # 패리티
                            "gear_rt": str,  # 기어링비율
                            "pl_qutr_rt": str,  # 손익분기율
                            "cfp": str,  # 자본지지점
                            "theory_pric": str,  # 이론가
                            "innr_vltl": str,  # 내재변동성
                            "delta": str,  # 델타
                            "lvrg": str,  # 레버리지
                            "exec_pric": str,  # 행사가격
                            "cnvt_rt": str,  # 전환비율
                            "lpposs_rt": str,  # LP보유비율
                            "pl_qutr_pt": str,  # 손익분기점
                            "fin_trde_dt": str,  # 최종거래일
                            "flo_dt": str,  # 상장일
                            "lpinitlast_suply_dt": str,  # LP초종공급일
                            "stk_nm": str,  # 종목명
                            "srvive_dys": str,  # 잔존일수
                            "dispty_rt": str,  # 괴리율
                            "lpmmcm_nm": str,  # LP회원사명
                            "lpmmcm_nm_1": str,  # LP회원사명1
                            "lpmmcm_nm_2": str,  # LP회원사명2
                            "xraymont_cntr_qty_arng_trde_tp": str,  # Xray순간체결량정리매매구분
                            "xraymont_cntr_qty_profa_100tp": str,  # Xray순간체결량증거금100구분
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_condition_search_request_ka30005(
            ...     isscomp_cd="000000000017",  # KB증권
            ...     bsis_aset_cd="201",  # KOSPI200
            ...     rght_tp="1",  # 콜
            ...     lpcd="000000000000",  # 전체
            ...     sort_tp="0"  # 정렬없음
            ... )
        """
        # 파라미터 유효성 검증
        if not isscomp_cd.isdigit() or len(isscomp_cd) != 12:
            raise ValueError("isscomp_cd must be a 12-digit number")
        if not bsis_aset_cd.isdigit():
            raise ValueError("bsis_aset_cd must be a number")
        if bsis_aset_cd == "000000000000" and len(bsis_aset_cd) != 12:
            raise ValueError("bsis_aset_cd must be 12 digits when '000000000000'")
        if rght_tp not in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            raise ValueError("Invalid rght_tp value")
        if not lpcd.isdigit() or len(lpcd) != 12:
            raise ValueError("lpcd must be a 12-digit number")
        if sort_tp not in ["0", "1", "2", "3", "4", "5", "6", "7"]:
            raise ValueError("Invalid sort_tp value")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30005",
        }

        data = {
            "isscomp_cd": isscomp_cd,
            "bsis_aset_cd": bsis_aset_cd,
            "rght_tp": rght_tp,
            "lpcd": lpcd,
            "sort_tp": sort_tp,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_price_change_rate_ranking_request_ka30009(
        self,
        sort_tp: str,
        rght_tp: str,
        trde_end_skip: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 등락율 순위 정보를 조회합니다.

        Args:
            sort_tp (str): 정렬구분
                - 1: 상승률
                - 2: 상승폭
                - 3: 하락률
                - 4: 하락폭
            rght_tp (str): 권리구분 (3자리)
                - 000: 전체
                - 001: 콜
                - 002: 풋
                - 003: DC
                - 004: DP
                - 006: 조기종료콜
                - 007: 조기종료풋
            trde_end_skip (str): 거래종료제외
                - 0: 거래종료포함
                - 1: 거래종료제외
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 등락율 순위 데이터
                {
                    "elwflu_rt_rank": [  # ELW등락율순위
                        {
                            "rank": str,  # 순위
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pre_sig": str,  # 대비기호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락률
                            "sel_req": str,  # 매도잔량
                            "buy_req": str,  # 매수잔량
                            "trde_qty": str,  # 거래량
                            "trde_prica": str,  # 거래대금
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_price_change_rate_ranking_request_ka30009(
            ...     sort_tp="1",  # 상승률
            ...     rght_tp="000",  # 전체
            ...     trde_end_skip="0"  # 거래종료포함
            ... )
        """
        # 파라미터 유효성 검증
        if sort_tp not in ["1", "2", "3", "4"]:
            raise ValueError("sort_tp must be one of: '1' (상승률), '2' (상승폭), '3' (하락률), '4' (하락폭)")
        if rght_tp not in ["000", "001", "002", "003", "004", "006", "007"]:
            raise ValueError("Invalid rght_tp value")
        if trde_end_skip not in ["0", "1"]:
            raise ValueError("trde_end_skip must be '0' (포함) or '1' (제외)")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30009",
        }

        data = {
            "sort_tp": sort_tp,
            "rght_tp": rght_tp,
            "trde_end_skip": trde_end_skip,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_order_volume_ranking_request_ka30010(
        self,
        sort_tp: str,
        rght_tp: str,
        trde_end_skip: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 잔량 순위 정보를 조회합니다.

        Args:
            sort_tp (str): 정렬구분
                - 1: 순매수잔량상위
                - 2: 순매도잔량상위
            rght_tp (str): 권리구분 (3자리)
                - 000: 전체
                - 001: 콜
                - 002: 풋
                - 003: DC
                - 004: DP
                - 006: 조기종료콜
                - 007: 조기종료풋
            trde_end_skip (str): 거래종료제외
                - 0: 거래종료포함
                - 1: 거래종료제외
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 잔량 순위 데이터
                {
                    "elwreq_rank": [  # ELW잔량순위
                        {
                            "stk_cd": str,  # 종목코드
                            "rank": str,  # 순위
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pre_sig": str,  # 대비기호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락률
                            "trde_qty": str,  # 거래량
                            "sel_req": str,  # 매도잔량
                            "buy_req": str,  # 매수잔량
                            "netprps_req": str,  # 순매수잔량
                            "trde_prica": str,  # 거래대금
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_order_volume_ranking_request_ka30010(
            ...     sort_tp="1",  # 순매수잔량상위
            ...     rght_tp="000",  # 전체
            ...     trde_end_skip="0"  # 거래종료포함
            ... )
        """
        # 파라미터 유효성 검증
        if sort_tp not in ["1", "2"]:
            raise ValueError("sort_tp must be '1' (순매수잔량상위) or '2' (순매도잔량상위)")
        if rght_tp not in ["000", "001", "002", "003", "004", "006", "007"]:
            raise ValueError("Invalid rght_tp value")
        if trde_end_skip not in ["0", "1"]:
            raise ValueError("trde_end_skip must be '0' (포함) or '1' (제외)")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30010",
        }

        data = {
            "sort_tp": sort_tp,
            "rght_tp": rght_tp,
            "trde_end_skip": trde_end_skip,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_proximity_rate_request_ka30011(
        self,
        stk_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 근접율 정보를 조회합니다.

        Args:
            stk_cd (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 근접율 데이터
                {
                    "elwalacc_rt": [  # ELW근접율
                        {
                            "stk_cd": str,  # 종목코드
                            "stk_nm": str,  # 종목명
                            "cur_prc": str,  # 현재가
                            "pre_sig": str,  # 대비기호
                            "pred_pre": str,  # 전일대비
                            "flu_rt": str,  # 등락율
                            "acc_trde_qty": str,  # 누적거래량
                            "alacc_rt": str,  # 근접율
                        },
                        ...
                    ],
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_proximity_rate_request_ka30011(
            ...     stk_cd="57JBHH"  # 종목코드
            ... )
        """
        # 파라미터 유효성 검증
        if not stk_cd or len(stk_cd) != 6:
            raise ValueError("stk_cd must be a 6-character string")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30011",
        }

        data = {
            "stk_cd": stk_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )
        
    def elw_detailed_stock_info_request_ka30012(
        self,
        stk_cd: str,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> dict:
        """ELW 종목 상세 정보를 조회합니다.

        Args:
            stk_cd (str): 종목코드 (6자리)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            dict: ELW 종목 상세 데이터
                {
                    "aset_cd": str,  # 자산코드
                    "cur_prc": str,  # 현재가
                    "pred_pre_sig": str,  # 전일대비기호
                    "pred_pre": str,  # 전일대비
                    "flu_rt": str,  # 등락율
                    "lpmmcm_nm": str,  # LP회원사명
                    "lpmmcm_nm_1": str,  # LP회원사명1
                    "lpmmcm_nm_2": str,  # LP회원사명2
                    "elwrght_cntn": str,  # ELW권리내용
                    "elwexpr_evlt_pric": str,  # ELW만기평가가격
                    "elwtheory_pric": str,  # ELW이론가
                    "dispty_rt": str,  # 괴리율
                    "elwinnr_vltl": str,  # ELW내재변동성
                    "exp_rght_pric": str,  # 예상권리가
                    "elwpl_qutr_rt": str,  # ELW손익분기율
                    "elwexec_pric": str,  # ELW행사가
                    "elwcnvt_rt": str,  # ELW전환비율
                    "elwcmpn_rt": str,  # ELW보상율
                    "elwpric_rising_part_rt": str,  # ELW가격상승참여율
                    "elwrght_type": str,  # ELW권리유형
                    "elwsrvive_dys": str,  # ELW잔존일수
                    "stkcnt": str,  # 주식수
                    "elwlpord_pos": str,  # ELWLP주문가능
                    "lpposs_rt": str,  # LP보유비율
                    "lprmnd_qty": str,  # LP보유수량
                    "elwspread": str,  # ELW스프레드
                    "elwprty": str,  # ELW패리티
                    "elwgear": str,  # ELW기어링
                    "elwflo_dt": str,  # ELW상장일
                    "elwfin_trde_dt": str,  # ELW최종거래일
                    "expr_dt": str,  # 만기일
                    "exec_dt": str,  # 행사일
                    "lpsuply_end_dt": str,  # LP공급종료일
                    "elwpay_dt": str,  # ELW지급일
                    "elwinvt_ix_comput": str,  # ELW투자지표산출
                    "elwpay_agnt": str,  # ELW지급대리인
                    "elwappr_way": str,  # ELW결재방법
                    "elwrght_exec_way": str,  # ELW권리행사방식
                    "elwpblicte_orgn": str,  # ELW발행기관
                    "dcsn_pay_amt": str,  # 확정지급액
                    "kobarr": str,  # KO베리어
                    "iv": str,  # IV
                    "clsprd_end_elwocr": str,  # 종기종료ELW발생
                    "bsis_aset_1": str,  # 기초자산1
                    "bsis_aset_comp_rt_1": str,  # 기초자산구성비율1
                    "bsis_aset_2": str,  # 기초자산2
                    "bsis_aset_comp_rt_2": str,  # 기초자산구성비율2
                    "bsis_aset_3": str,  # 기초자산3
                    "bsis_aset_comp_rt_3": str,  # 기초자산구성비율3
                    "bsis_aset_4": str,  # 기초자산4
                    "bsis_aset_comp_rt_4": str,  # 기초자산구성비율4
                    "bsis_aset_5": str,  # 기초자산5
                    "bsis_aset_comp_rt_5": str,  # 기초자산구성비율5
                    "fr_dt": str,  # 평가시작일자
                    "to_dt": str,  # 평가종료일자
                    "fr_tm": str,  # 평가시작시간
                    "evlt_end_tm": str,  # 평가종료시간
                    "evlt_pric": str,  # 평가가격
                    "evlt_fnsh_yn": str,  # 평가완료여부
                    "all_hgst_pric": str,  # 전체최고가
                    "all_lwst_pric": str,  # 전체최저가
                    "imaf_hgst_pric": str,  # 직후최고가
                    "imaf_lwst_pric": str,  # 직후최저가
                    "sndhalf_mrkt_hgst_pric": str,  # 후반장최고가
                    "sndhalf_mrkt_lwst_pric": str,  # 후반장최저가
                    "return_code": int,  # 응답코드
                    "return_msg": str,  # 응답메시지
                }

        Raises:
            ValueError: 필수 파라미터가 누락되었거나 유효하지 않은 경우

        Example:
            >>> from kiwoom_rest_api import KiwoomRestAPI
            >>> api = KiwoomRestAPI()
            >>> result = api.elw.elw_detailed_stock_info_request_ka30012(
            ...     stk_cd="57JBHH"  # 종목코드
            ... )
        """
        # 파라미터 유효성 검증
        if not stk_cd or len(stk_cd) != 6:
            raise ValueError("stk_cd must be a 6-character string")

        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka30012",
        }

        data = {
            "stk_cd": stk_cd,
        }

        return self._execute_request(
            "POST",
            json=data,
            headers=headers,
        )