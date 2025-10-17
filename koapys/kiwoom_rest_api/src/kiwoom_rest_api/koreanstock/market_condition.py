from kiwoom_rest_api.core.base_api import KiwoomBaseAPI
from typing import Union, Dict, Any, Awaitable

class MarketCondition(KiwoomBaseAPI):
    """한국 주식 시장 조건 관련 API를 제공하는 클래스"""
    
    def __init__(
        self, 
        base_url: str = None, 
        token_manager=None, 
        use_async: bool = False,
        resource_url: str = "/api/dostk/mrkcond"
    ):
        """
        MarketCondition 클래스 초기화
        
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
        
    def stock_quote_request_ka10004(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """주식호가요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "bid_req_base_tm": "162000",
                "sel_10th_pre_req_pre": "0",
                "sel_10th_pre_req": "0",
                "sel_10th_pre_bid": "0",
                ...
                "ovt_buy_req_pre": "0",
                "return_code": 0,
                "return_msg": "정상적으로 처리되었습니다"
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10004"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def stock_daily_weekly_monthly_time_request_ka10005(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """주식일주월시분요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "stk_ddwkmm": [
                    {
                        "date": "20241028",
                        "open_pric": "95400",
                        "high_pric": "95400",
                        "low_pric": "95400",
                        "close_pric": "95400",
                        "pre": "0",
                        "flu_rt": "0.00",
                        "trde_qty": "0",
                        "trde_prica": "0",
                        "for_poss": "+26.07",
                        "for_wght": "+26.07",
                        "for_netprps": "0",
                        "orgn_netprps": "",
                        "ind_netprps": "",
                        "crd_remn_rt": "",
                        "frgn": "",
                        "prm": ""
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10005"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def stock_minute_time_request_ka10006(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """주식시분요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "date": "20241105",
                "open_pric": "0",
                "high_pric": "0",
                "low_pric": "0",
                "close_pric": "135300",
                "pre": "0",
                "flu_rt": "0.00",
                "trde_qty": "0",
                "trde_prica": "0",
                "cntr_str": "0.00",
                "return_code": 0,
                "return_msg": "정상적으로 처리되었습니다"
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10006"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def market_price_table_info_request_ka10007(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """시세표성정보요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "stk_nm": "삼성전자",
                "stk_cd": "005930",
                "date": "20241105",
                "tm": "104000",
                "pred_close_pric": "135300",
                "pred_trde_qty": "88862",
                "upl_pric": "+175800",
                "lst_pric": "-94800",
                "pred_trde_prica": "11963",
                "flo_stkcnt": "25527",
                "cur_prc": "135300",
                "smbol": "3",
                "flu_rt": "0.00",
                "pred_rt": "0.00",
                "open_pric": "0",
                "high_pric": "0",
                "low_pric": "0",
                "cntr_qty": "",
                "trde_qty": "0",
                "trde_prica": "0",
                "exp_cntr_pric": "-0",
                "exp_cntr_qty": "0",
                "exp_sel_pri_bid": "0",
                "exp_buy_pri_bid": "0",
                "trde_strt_dt": "00000000",
                "exec_pric": "0",
                "hgst_pric": "",
                "lwst_pric": "",
                "hgst_pric_dt": "",
                "lwst_pric_dt": "",
                "sel_1bid": "0",
                "sel_2bid": "0",
                ...
                "buy_10bid_req": "0",
                "tot_buy_req": "0",
                "tot_sel_req": "0",
                "tot_buy_cnt": "",
                "tot_sel_cnt": "0",
                "return_code": 0,
                "return_msg": "정상적으로 처리되었습니다"
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10007"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def rights_issue_overall_price_request_ka10011(
        self,
        rights_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """신주인수권전체시세요청

        Args:
            rights_type (str): 신주인수권구분 (00:전체, 05:신주인수권증권, 07:신주인수권증서)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "newstk_recvrht_mrpr": [
                    {
                        "stk_cd": "J0036221D",
                        "stk_nm": "KG모빌리티 122WR",
                        "cur_prc": "988",
                        "pred_pre_sig": "3",
                        "pred_pre": "0",
                        "flu_rt": "0.00",
                        "fpr_sel_bid": "-0",
                        "fpr_buy_bid": "-0",
                        "acc_trde_qty": "0",
                        "open_pric": "-0",
                        "high_pric": "-0",
                        "low_pric": "-0"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10011"
        }
        data = {
            "newstk_recvrht_tp": rights_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def daily_institutional_trading_items_request_ka10044(
        self,
        start_date: str,
        end_date: str,
        trade_type: str,
        market_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """일별기관매매종목요청

        Args:
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            trade_type (str): 매매구분 (1:순매도, 2:순매수)
            market_type (str): 시장구분 (001:코스피, 101:코스닥)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "daly_orgn_trde_stk": [
                    {
                        "stk_cd": "005930",
                        "stk_nm": "삼성전자",
                        "netprps_qty": "-0",
                        "netprps_amt": "-1",
                        "prsm_avg_pric": "140000",
                        "cur_prc": "-95100",
                        "avg_pric_pre": "--44900",
                        "pre_rt": "-32.07"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10044"
        }
        data = {
            "strt_dt": start_date,
            "end_dt": end_date,
            "trde_tp": trade_type,
            "mrkt_tp": market_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def stockwise_institutional_trading_trend_request_ka10045(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        org_institution_price_type: str,
        foreign_institution_price_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """종목별기관매매추이요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            org_institution_price_type (str): 기관추정단가구분 (1:매수단가, 2:매도단가)
            foreign_institution_price_type (str): 외인추정단가구분 (1:매수단가, 2:매도단가)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "orgn_prsm_avg_pric": "117052",
                "for_prsm_avg_pric": "0",
                "stk_orgn_trde_trnsn": [
                    {
                        "dt": "20241107",
                        "close_pric": "133600",
                        "pre_sig": "0",
                        "pred_pre": "0",
                        "flu_rt": "0.00",
                        "trde_qty": "0",
                        "orgn_dt_acc": "158",
                        "orgn_daly_nettrde_qty": "0",
                        "for_dt_acc": "28315",
                        "for_daly_nettrde_qty": "0",
                        "limit_exh_rt": "+26.14"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10045"
        }
        data = {
            "stk_cd": stock_code,
            "strt_dt": start_date,
            "end_dt": end_date,
            "orgn_prsm_unp_tp": org_institution_price_type,
            "for_prsm_unp_tp": foreign_institution_price_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def execution_strength_by_hour_request_ka10046(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """체결강도추이시간별요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "cntr_str_tm": [
                    {
                        "cntr_tm": "163713",
                        "cur_prc": "+156600",
                        "pred_pre": "+34900",
                        "pred_pre_sig": "2",
                        "flu_rt": "+28.68",
                        "trde_qty": "-1",
                        "acc_trde_prica": "14449",
                        "acc_trde_qty": "113636",
                        "cntr_str": "172.01",
                        "cntr_str_5min": "172.01",
                        "cntr_str_20min": "172.01",
                        "cntr_str_60min": "170.67",
                        "stex_tp": "KRX"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10046"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def execution_strength_by_day_request_ka10047(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """체결강도추이일별요청

        Args:
            stock_code (str): 종목코드 (예: "005930", "KRX:039490")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "cntr_str_daly": [
                    {
                        "dt": "20241128",
                        "cur_prc": "+219000",
                        "pred_pre": "+14000",
                        "pred_pre_sig": "2",
                        "flu_rt": "+6.83",
                        "trde_qty": "",
                        "acc_trde_prica": "2",
                        "acc_trde_qty": "8",
                        "cntr_str": "0.00",
                        "cntr_str_5min": "201.54",
                        "cntr_str_20min": "139.37",
                        "cntr_str_60min": "172.06"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10047"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def intraday_investor_trading_request_ka10063(
        self,
        market_type: str,
        amount_quantity_type: str,
        investor_type: str,
        foreign_all: str,
        simultaneous_net_buy_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """장중투자자별매매요청

        Args:
            market_type (str): 시장구분 (000:전체, 001:코스피, 101:코스닥)
            amount_quantity_type (str): 금액수량구분 (1:금액, 2:수량)
            investor_type (str): 투자자별 (6:외국인, 7:기관계, 1:투신, 0:보험, 2:은행, 3:연기금, 4:국가, 5:기타법인)
            foreign_all (str): 외국계전체 (1:체크, 0:미체크)
            simultaneous_net_buy_type (str): 동시순매수구분 (1:체크, 0:미체크)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "opmr_invsr_trde": [
                    {
                        "stk_cd": "005930",
                        "stk_nm": "삼성전자",
                        "cur_prc": "64",
                        "pre_sig": "3",
                        "pred_pre": "0",
                        "flu_rt": "0.00",
                        "acc_trde_qty": "1",
                        "netprps_qty": "+1083000",
                        "prev_pot_netprps_qty": "+1083000",
                        "netprps_irds": "0",
                        "buy_qty": "+1113000",
                        "buy_qty_irds": "0",
                        "sell_qty": "--30000",
                        "sell_qty_irds": "0"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10063"
        }
        data = {
            "mrkt_tp": market_type,
            "amt_qty_tp": amount_quantity_type,
            "invsr": investor_type,
            "frgn_all": foreign_all,
            "smtm_netprps_tp": simultaneous_net_buy_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def post_market_investor_trading_request_ka10066(
        self,
        market_type: str,
        amount_quantity_type: str,
        trade_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """장마감후투자자별매매요청

        Args:
            market_type (str): 시장구분 (000:전체, 001:코스피, 101:코스닥)
            amount_quantity_type (str): 금액수량구분 (1:금액, 2:수량)
            trade_type (str): 매매구분 (0:순매수, 1:매수, 2:매도)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "opaf_invsr_trde": [
                    {
                        "stk_cd": "005930",
                        "stk_nm": "삼성전자",
                        "cur_prc": "-7410",
                        "pre_sig": "5",
                        "pred_pre": "-50",
                        "flu_rt": "-0.67",
                        "trde_qty": "8",
                        "ind_invsr": "0",
                        "frgnr_invsr": "0",
                        "orgn": "0",
                        "fnnc_invt": "0",
                        "insrnc": "0",
                        "invtrt": "0",
                        "etc_fnnc": "0",
                        "bank": "0",
                        "penfnd_etc": "0",
                        "samo_fund": "0",
                        "natn": "0",
                        "etc_corp": "0"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10066"
        }
        data = {
            "mrkt_tp": market_type,
            "amt_qty_tp": amount_quantity_type,
            "trde_tp": trade_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def brokerwise_stock_trading_trend_request_ka10078(
        self,
        member_company_code: str,
        stock_code: str,
        start_date: str,
        end_date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """증권사별종목매매동향요청

        Args:
            member_company_code (str): 회원사코드 (회원사 코드는 ka10102 조회)
            stock_code (str): 종목코드 (거래소별 종목코드, 예: KRX:039490, NXT:039490_NX, SOR:039490_AL)
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "sec_stk_trde_trend": [
                    {
                        "dt": "20241107",
                        "cur_prc": "10050",
                        "pre_sig": "0",
                        "pred_pre": "0",
                        "flu_rt": "0.00",
                        "acc_trde_qty": "0",
                        "netprps_qty": "0",
                        "buy_qty": "0",
                        "sell_qty": "0"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10078"
        }
        data = {
            "mmcm_cd": member_company_code,
            "stk_cd": stock_code,
            "strt_dt": start_date,
            "end_dt": end_date
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def daily_stock_price_request_ka10086(
        self,
        stock_code: str,
        query_date: str,
        indicator_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """일별주가요청

        Args:
            stock_code (str): 종목코드 (거래소별 종목코드, 예: KRX:039490, NXT:039490_NX, SOR:039490_AL)
            query_date (str): 조회일자 (YYYYMMDD)
            indicator_type (str): 표시구분 (0:수량, 1:금액(백만원))
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "daly_stkpc": [
                    {
                        "date": "20241125",
                        "open_pric": "+78800",
                        "high_pric": "+101100",
                        "low_pric": "-54500",
                        "close_pric": "-55000",
                        "pred_rt": "-22800",
                        "flu_rt": "-29.31",
                        "trde_qty": "20278",
                        "amt_mn": "1179",
                        "crd_rt": "0.00",
                        "ind": "--714",
                        "orgn": "+693",
                        "for_qty": "--266783",
                        "frgn": "0",
                        "prm": "0",
                        "for_rt": "+51.56",
                        "for_poss": "+51.56",
                        "for_wght": "+51.56",
                        "for_netprps": "--266783",
                        "orgn_netprps": "+693",
                        "ind_netprps": "--714",
                        "crd_remn_rt": "0.00"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10086"
        }
        data = {
            "stk_cd": stock_code,
            "qry_dt": query_date,
            "indc_tp": indicator_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def after_hours_single_price_request_ka10087(
        self,
        stock_code: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """시간외단일가요청

        Args:
            stock_code (str): 종목코드 (예: "005930")
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "bid_req_base_tm": "164000",
                "ovt_sigpric_sel_bid_jub_pre_5": "0",
                "ovt_sigpric_sel_bid_jub_pre_4": "0",
                "ovt_sigpric_sel_bid_jub_pre_3": "0",
                "ovt_sigpric_sel_bid_jub_pre_2": "0",
                "ovt_sigpric_sel_bid_jub_pre_1": "0",
                "ovt_sigpric_sel_bid_qty_5": "0",
                "ovt_sigpric_sel_bid_qty_4": "0",
                "ovt_sigpric_sel_bid_qty_3": "0",
                "ovt_sigpric_sel_bid_qty_2": "0",
                "ovt_sigpric_sel_bid_qty_1": "0",
                "ovt_sigpric_sel_bid_5": "-0",
                "ovt_sigpric_sel_bid_4": "-0",
                "ovt_sigpric_sel_bid_3": "-0",
                "ovt_sigpric_sel_bid_2": "-0",
                "ovt_sigpric_sel_bid_1": "-0",
                "ovt_sigpric_buy_bid_1": "-0",
                "ovt_sigpric_buy_bid_2": "-0",
                "ovt_sigpric_buy_bid_3": "-0",
                "ovt_sigpric_buy_bid_4": "-0",
                "ovt_sigpric_buy_bid_5": "-0",
                "ovt_sigpric_buy_bid_qty_1": "0",
                "ovt_sigpric_buy_bid_qty_2": "0",
                "ovt_sigpric_buy_bid_qty_3": "0",
                "ovt_sigpric_buy_bid_qty_4": "0",
                "ovt_sigpric_buy_bid_qty_5": "0",
                "ovt_sigpric_buy_bid_jub_pre_1": "0",
                "ovt_sigpric_buy_bid_jub_pre_2": "0",
                "ovt_sigpric_buy_bid_jub_pre_3": "0",
                "ovt_sigpric_buy_bid_jub_pre_4": "0",
                "ovt_sigpric_buy_bid_jub_pre_5": "0",
                "ovt_sigpric_sel_bid_tot_req": "0",
                "ovt_sigpric_buy_bid_tot_req": "0",
                "sel_bid_tot_req_jub_pre": "0",
                "sel_bid_tot_req": "24028",
                "buy_bid_tot_req": "26579",
                "buy_bid_tot_req_jub_pre": "0",
                "ovt_sel_bid_tot_req_jub_pre": "0",
                "ovt_sel_bid_tot_req": "0",
                "ovt_buy_bid_tot_req": "11",
                "ovt_buy_bid_tot_req_jub_pre": "0",
                "ovt_sigpric_cur_prc": "156600",
                "ovt_sigpric_pred_pre_sig": "0",
                "ovt_sigpric_pred_pre": "0",
                "ovt_sigpric_flu_rt": "0.00",
                "ovt_sigpric_acc_trde_qty": "0"
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka10087"
        }
        data = {
            "stk_cd": stock_code
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def program_trading_trend_by_time_request_ka90005(
        self,
        date: str,
        amount_quantity_type: str,
        market_type: str,
        minute_tick_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """프로그램매매추이요청 (시간대별)

        Args:
            date (str): 날짜 (YYYYMMDD)
            amount_quantity_type (str): 금액수량구분 (1:금액(백만원), 2:수량(천주))
            market_type (str): 시장구분 (코스피- 거래소구분값 1일경우:P00101, 2일경우:P001_NX01, 3일경우:P001_AL01, 코스닥- 거래소구분값 1일경우:P10102, 2일경우:P101_NX02, 3일경우:P001_AL02)
            minute_tick_type (str): 분틱구분 (0:틱, 1:분)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "prm_trde_trnsn": [
                    {
                        "cntr_tm": "170500",
                        "dfrt_trde_sel": "0",
                        "dfrt_trde_buy": "0",
                        "dfrt_trde_netprps": "0",
                        "ndiffpro_trde_sel": "1",
                        "ndiffpro_trde_buy": "17",
                        "ndiffpro_trde_netprps": "+17",
                        "dfrt_trde_sell_qty": "0",
                        "dfrt_trde_buy_qty": "0",
                        "dfrt_trde_netprps_qty": "0",
                        "ndiffpro_trde_sell_qty": "0",
                        "ndiffpro_trde_buy_qty": "0",
                        "ndiffpro_trde_netprps_qty": "+0",
                        "all_sel": "1",
                        "all_buy": "17",
                        "all_netprps": "+17",
                        "kospi200": "+47839",
                        "basis": "-146.59"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90005"
        }
        data = {
            "date": date,
            "amt_qty_tp": amount_quantity_type,
            "mrkt_tp": market_type,
            "min_tic_tp": minute_tick_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def program_trading_arbitrage_balance_trend_request_ka90006(
        self,
        date: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """프로그램매매차익잔고추이요청

        Args:
            date (str): 날짜 (YYYYMMDD)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "prm_trde_dfrt_remn_trnsn": [
                    {
                        "dt": "20241125",
                        "buy_dfrt_trde_qty": "0",
                        "buy_dfrt_trde_amt": "0",
                        "buy_dfrt_trde_irds_amt": "0",
                        "sel_dfrt_trde_qty": "0",
                        "sel_dfrt_trde_amt": "0",
                        "sel_dfrt_trde_irds_amt": "0"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90006"
        }
        data = {
            "date": date,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def cumulative_program_trading_trend_request_ka90007(
        self,
        date: str,
        amount_quantity_type: str,
        market_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """프로그램매매누적추이요청

        Args:
            date (str): 날짜 (YYYYMMDD, 종료일기준 1년간 데이터만 조회가능)
            amount_quantity_type (str): 금액수량구분 (1:금액, 2:수량)
            market_type (str): 시장구분 (0:코스피, 1:코스닥)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "prm_trde_acc_trnsn": [
                    {
                        "dt": "20241125",
                        "kospi200": "0.00",
                        "basis": "0.00",
                        "dfrt_trde_tdy": "0",
                        "dfrt_trde_acc": "+353665",
                        "ndiffpro_trde_tdy": "0",
                        "ndiffpro_trde_acc": "+671219",
                        "all_tdy": "0",
                        "all_acc": "+1024884"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90007"
        }
        data = {
            "date": date,
            "amt_qty_tp": amount_quantity_type,
            "mrkt_tp": market_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def stockwise_program_trading_by_hour_request_ka90008(
        self,
        amount_quantity_type: str,
        stock_code: str,
        date: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """종목시간별프로그램매매추이요청

        Args:
            amount_quantity_type (str): 금액수량구분 (1:금액, 2:수량)
            stock_code (str): 종목코드 (거래소별 종목코드, 예: KRX:039490, NXT:039490_NX, SOR:039490_AL)
            date (str): 날짜 (YYYYMMDD)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "stk_tm_prm_trde_trnsn": [
                    {
                        "tm": "153029",
                        "cur_prc": "+245500",
                        "pre_sig": "2",
                        "pred_pre": "+40000",
                        "flu_rt": "+19.46",
                        "trde_qty": "104006",
                        "prm_sell_amt": "14245",
                        "prm_buy_amt": "10773",
                        "prm_netprps_amt": "--3472",
                        "prm_netprps_amt_irds": "+771",
                        "prm_sell_qty": "58173",
                        "prm_buy_qty": "43933",
                        "prm_netprps_qty": "--14240",
                        "prm_netprps_qty_irds": "+3142",
                        "base_pric_tm": "",
                        "dbrt_trde_rpy_sum": "",
                        "remn_rcvord_sum": "",
                        "stex_tp": "KRX"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90008"
        }
        data = {
            "amt_qty_tp": amount_quantity_type,
            "stk_cd": stock_code,
            "date": date
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def program_trading_trend_by_date_request_ka90010(
        self,
        date: str,
        amount_quantity_type: str,
        market_type: str,
        minute_tick_type: str,
        stock_exchange_type: str,
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """프로그램매매추이요청 (일자별)

        Args:
            date (str): 날짜 (YYYYMMDD)
            amount_quantity_type (str): 금액수량구분 (1:금액(백만원), 2:수량(천주))
            market_type (str): 시장구분 (코스피- 거래소구분값 1일경우:P00101, 2일경우:P001_NX01, 3일경우:P001_AL01, 코스닥- 거래소구분값 1일경우:P10102, 2일경우:P101_NX02, 3일경우:P001_AL02)
            minute_tick_type (str): 분틱구분 (0:틱, 1:분)
            stock_exchange_type (str): 거래소구분 (1:KRX, 2:NXT, 3:통합)
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "prm_trde_trnsn": [
                    {
                        "cntr_tm": "20241125000000",
                        "dfrt_trde_sel": "0",
                        "dfrt_trde_buy": "0",
                        "dfrt_trde_netprps": "0",
                        "ndiffpro_trde_sel": "0",
                        "ndiffpro_trde_buy": "0",
                        "ndiffpro_trde_netprps": "0",
                        "dfrt_trde_sell_qty": "0",
                        "dfrt_trde_buy_qty": "0",
                        "dfrt_trde_netprps_qty": "0",
                        "ndiffpro_trde_sell_qty": "0",
                        "ndiffpro_trde_buy_qty": "0",
                        "ndiffpro_trde_netprps_qty": "0",
                        "all_sel": "0",
                        "all_buy": "0",
                        "all_netprps": "0",
                        "kospi200": "0.00",
                        "basis": ""
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90010"
        }
        data = {
            "date": date,
            "amt_qty_tp": amount_quantity_type,
            "mrkt_tp": market_type,
            "min_tic_tp": minute_tick_type,
            "stex_tp": stock_exchange_type
        }
        return self._execute_request("POST", json=data, headers=headers)
    
    def stockwise_program_trading_by_day_request_ka90013(
        self,
        stock_code: str,
        amount_quantity_type: str = "",
        date: str = "",
        cont_yn: str = "N",
        next_key: str = ""
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """종목일별프로그램매매추이요청

        Args:
            stock_code (str): 종목코드 (거래소별 종목코드, 예: KRX:039490, NXT:039490_NX, SOR:039490_AL)
            amount_quantity_type (str, optional): 금액수량구분 (1:금액, 2:수량). Defaults to "".
            date (str, optional): 날짜 (YYYYMMDD). Defaults to "".
            cont_yn (str, optional): 연속조회여부. Defaults to "N".
            next_key (str, optional): 연속조회키. Defaults to "".

        Returns:
            Union[Dict[str, Any], Awaitable[Dict[str, Any]]]: 응답 데이터
            {
                "stk_daly_prm_trde_trnsn": [
                    {
                        "dt": "20241125",
                        "cur_prc": "+267000",
                        "pre_sig": "2",
                        "pred_pre": "+60000",
                        "flu_rt": "+28.99",
                        "trde_qty": "3",
                        "prm_sell_amt": "0",
                        "prm_buy_amt": "0",
                        "prm_netprps_amt": "0",
                        "prm_netprps_amt_irds": "0",
                        "prm_sell_qty": "0",
                        "prm_buy_qty": "0",
                        "prm_netprps_qty": "0",
                        "prm_netprps_qty_irds": "0",
                        "base_pric_tm": "",
                        "dbrt_trde_rpy_sum": "",
                        "remn_rcvord_sum": "",
                        "stex_tp": "통합"
                    },
                    ...
                ]
            }
        """
        headers = {
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": "ka90013"
        }
        data = {
            "amt_qty_tp": amount_quantity_type,
            "stk_cd": stock_code,
            "date": date
        }
        return self._execute_request("POST", json=data, headers=headers)