#!/usr/bin/env python3
"""
프로그램 매매 내역 다운로드 유틸리티 스크립트

사용법:
    1. 일별 전 종목 프로그램 매매 현황 다운로드 (KOSPI & KOSDAQ)
       python scripts/data/download_program_trading.py --mode summary --date 20241125
       
    2. 특정 종목(들)의 시간대별(분 단위) 프로그램 매매 추이 다운로드 (당일/실시간 전용)
       python scripts/data/download_program_trading.py --mode time-trend --date 20260720 --stocks 005930,000660
       
    3. 특정 종목(들)의 일별 프로그램 매매 추이 다운로드 (역사적 데이터)
       python scripts/data/download_program_trading.py --mode daily-trend --stocks 005930,000660
"""

import os
import sys
import csv
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("program_trading_downloader")

# 프로젝트 루트 및 모듈 경로 설정
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "koapys" / "kiwoom_rest_api" / "src"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from kiwoom_rest_api.auth.token import TokenManager
from kiwoom_rest_api.koreanstock.market_condition import MarketCondition
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo


def get_api_clients():
    """API 토큰 매니저 및 클라이언트들을 초기화하여 반환합니다."""
    logger.info("키움 REST API 토큰 및 클라이언트 초기화 중...")
    token_manager = TokenManager()
    
    # 토큰 유효성 검사 겸 토큰 가져오기
    token = token_manager.get_token()
    if not token:
        logger.error("API 토큰 획득에 실패했습니다. .env 파일의 API 키와 시크릿을 확인해 주세요.")
        sys.exit(1)
        
    base_url = "https://api.kiwoom.com"
    market_condition = MarketCondition(base_url=base_url, token_manager=token_manager)
    stock_info = StockInfo(base_url=base_url, token_manager=token_manager)
    
    return market_condition, stock_info


def download_summary(stock_info: StockInfo, date: str, market: str, amount_qty_tp: str, output_dir: Path):
    """지정된 날짜의 코스피/코스닥 전 종목 프로그램 매매 현황을 수집하여 CSV로 저장합니다."""
    logger.info(f"[{date}] 전 종목 프로그램 매매 현황 수집을 시작합니다. (대상 시장: {market.upper()})")
    
    markets_to_fetch = []
    if market in ['kospi', 'all']:
        markets_to_fetch.append(('P00101', 'KOSPI'))
    if market in ['kosdaq', 'all']:
        markets_to_fetch.append(('P10102', 'KOSDAQ'))
        
    all_records = []
    
    for mkt_code, mkt_name in markets_to_fetch:
        logger.info(f"{mkt_name} 시장 데이터 수집 중...")
        cont_yn = "N"
        next_key = ""
        page = 1
        
        while True:
            logger.info(f"  페이지 {page} 조회 중 (next_key: {next_key or 'None'})...")
            try:
                res = stock_info.stock_wise_program_trading_status_request_ka90004(
                    date=date,
                    market_type=mkt_code,
                    stock_exchange_type="1",  # 1: KRX
                    cont_yn=cont_yn,
                    next_key=next_key
                )
                
                return_code = str(res.get("return_code", "-1"))
                return_msg = res.get("return_msg", "")
                
                if return_code != "0":
                    logger.error(f"API 호출 실패 (코드: {return_code}, 메시지: {return_msg})")
                    break
                    
                items = res.get("stk_prm_trde_prst", [])
                if not items:
                    logger.info("  수집된 데이터가 없습니다.")
                    break
                
                # 빈 값만 있는 더미 데이터 필터링
                valid_items = [item for item in items if item.get('stk_cd')]
                if not valid_items and page == 1:
                    logger.warning(f"  {mkt_name} 시장에 유효한 종목 데이터가 없습니다. (휴장일이거나 데이터 미제공일 수 있음)")
                    break
                
                for item in valid_items:
                    item['market'] = mkt_name
                    all_records.append(item)
                    
                logger.info(f"  페이지 {page} 완료: {len(valid_items)}개 종목 추가 (누적: {len(all_records)}개)")
                
                # 다음 페이지 확인
                next_key = res.get("next-key", "")
                if not next_key or next_key == "":
                    break
                    
                cont_yn = "Y"
                page += 1
                
            except Exception as e:
                logger.error(f"데이터 수집 중 오류 발생: {e}")
                break
                
    if not all_records:
        logger.warning("수집된 프로그램 매매 현황 데이터가 없어 CSV 파일을 생성하지 않습니다.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"program_summary_{date}.csv"
    
    headers = [
        'market', 'stk_cd', 'stk_nm', 'cur_prc', 'flu_sig', 'pred_pre', 
        'buy_cntr_qty', 'buy_cntr_amt', 'sel_cntr_qty', 'sel_cntr_amt', 
        'netprps_prica', 'all_trde_rt'
    ]
    
    # 한글 필드 매핑 정의
    header_mapping = {
        'market': '시장',
        'stk_cd': '종목코드',
        'stk_nm': '종목명',
        'cur_prc': '현재가',
        'flu_sig': '대비부호',
        'pred_pre': '전일대비',
        'buy_cntr_qty': '매수체결수량',
        'buy_cntr_amt': '매수체결금액',
        'sel_cntr_qty': '매도체결수량',
        'sel_cntr_amt': '매도체결금액',
        'netprps_prica': '순매수금액',
        'all_trde_rt': '전체거래비중'
    }
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            # 한글 헤더 작성
            writer.writerow(header_mapping)
            for record in all_records:
                # 불필요한 키 제외하고 headers 순서대로 쓰기
                filtered_record = {k: record.get(k, '') for k in headers}
                writer.writerow(filtered_record)
                
        logger.info(f"성공! 프로그램 매매 현황이 CSV로 저장되었습니다: {output_file}")
        logger.info(f"총 저장된 종목 수: {len(all_records)}개")
    except Exception as e:
        logger.error(f"CSV 파일 저장 중 오류 발생: {e}")


def download_time_trend(market_condition: MarketCondition, date: str, stocks: List[str], amount_qty_tp: str, output_dir: Path):
    """지정된 날짜 및 종목들에 대한 시간대별 프로그램 매매 추이를 수집하여 CSV로 저장합니다."""
    logger.info(f"[{date}] 지정된 {len(stocks)}개 종목에 대한 시간대별 프로그램 매매 추이 수집을 시작합니다. (당일 및 실시간 거래일 전용)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'tm', 'cur_prc', 'pre_sig', 'pred_pre', 'flu_rt', 'trde_qty', 
        'prm_sell_amt', 'prm_buy_amt', 'prm_netprps_amt', 'prm_netprps_amt_irds',
        'prm_sell_qty', 'prm_buy_qty', 'prm_netprps_qty', 'prm_netprps_qty_irds'
    ]
    
    header_mapping = {
        'tm': '시간',
        'cur_prc': '현재가',
        'pre_sig': '대비부호',
        'pred_pre': '전일대비',
        'flu_rt': '등락율',
        'trde_qty': '체결수량',
        'prm_sell_amt': '프로그램매도금액',
        'prm_buy_amt': '프로그램매수금액',
        'prm_netprps_amt': '프로그램순매수금액',
        'prm_netprps_amt_irds': '프로그램순매수금액증감',
        'prm_sell_qty': '프로그램매도수량',
        'prm_buy_qty': '프로그램매수수량',
        'prm_netprps_qty': '프로그램순매수수량',
        'prm_netprps_qty_irds': '프로그램순매수수량증감'
    }
    
    for code in stocks:
        code = code.strip()
        if not code:
            continue
            
        logger.info(f"종목 코드 [{code}] 시간대별 추이 수집 중...")
        cont_yn = "N"
        next_key = ""
        page = 1
        stock_records = []
        
        while True:
            logger.info(f"  페이지 {page} 조회 중...")
            try:
                res = market_condition.stockwise_program_trading_by_hour_request_ka90008(
                    amount_quantity_type=amount_qty_tp,
                    stock_code=code,
                    date=date,
                    cont_yn=cont_yn,
                    next_key=next_key
                )
                
                return_code = str(res.get("return_code", "-1"))
                return_msg = res.get("return_msg", "")
                
                if return_code != "0":
                    logger.error(f"  종목 {code} API 호출 실패 (코드: {return_code}, 메시지: {return_msg})")
                    break
                    
                items = res.get("stk_tm_prm_trde_trnsn", [])
                if not items:
                    logger.info("  수집된 시간대 데이터가 없습니다.")
                    break
                    
                valid_items = [item for item in items if item.get('tm')]
                if not valid_items:
                    break
                    
                stock_records.extend(valid_items)
                logger.info(f"  페이지 {page} 완료: {len(valid_items)}개 레코드 추가 (누적: {len(stock_records)}개)")
                
                next_key = res.get("next-key", "")
                if not next_key or next_key == "":
                    break
                    
                cont_yn = "Y"
                page += 1
                
            except Exception as e:
                logger.error(f"  종목 {code} 수집 중 오류 발생: {e}")
                break
                
        if stock_records:
            output_file = output_dir / f"program_time_trend_{code}_{date}.csv"
            try:
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(header_mapping)
                    for record in stock_records:
                        filtered_record = {k: record.get(k, '') for k in headers}
                        writer.writerow(filtered_record)
                logger.info(f"  성공! [{code}] 시간대별 추이가 CSV로 저장되었습니다: {output_file}")
            except Exception as e:
                logger.error(f"  CSV 파일 저장 중 오류 발생: {e}")
        else:
            logger.warning(f"  [{code}] 수집된 데이터가 없어 CSV 파일을 생성하지 않았습니다. (지나간 날짜는 조회가 되지 않으며 당일 실시간 장 중에만 조회가 가능할 수 있습니다.)")


def download_daily_trend(market_condition: MarketCondition, stocks: List[str], amount_qty_tp: str, output_dir: Path):
    """지정된 종목들에 대한 일별 프로그램 매매 추이(역사적 데이터)를 수집하여 CSV로 저장합니다."""
    logger.info(f"지정된 {len(stocks)}개 종목에 대한 일별 프로그램 매매 추이 수집을 시작합니다. (역사적 추이)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'dt', 'cur_prc', 'pre_sig', 'pred_pre', 'flu_rt', 'trde_qty', 
        'prm_sell_amt', 'prm_buy_amt', 'prm_netprps_amt', 'prm_netprps_amt_irds',
        'prm_sell_qty', 'prm_buy_qty', 'prm_netprps_qty', 'prm_netprps_qty_irds', 'stex_tp'
    ]
    
    header_mapping = {
        'dt': '일자',
        'cur_prc': '현재가',
        'pre_sig': '대비부호',
        'pred_pre': '전일대비',
        'flu_rt': '등락율',
        'trde_qty': '체결수량',
        'prm_sell_amt': '프로그램매도금액',
        'prm_buy_amt': '프로그램매수금액',
        'prm_netprps_amt': '프로그램순매수금액',
        'prm_netprps_amt_irds': '프로그램순매수금액증감',
        'prm_sell_qty': '프로그램매도수량',
        'prm_buy_qty': '프로그램매수수량',
        'prm_netprps_qty': '프로그램순매수수량',
        'prm_netprps_qty_irds': '프로그램순매수수량증감',
        'stex_tp': '거래소구분'
    }
    
    for code in stocks:
        code = code.strip()
        if not code:
            continue
            
        logger.info(f"종목 코드 [{code}] 일별 추이 수집 중...")
        cont_yn = "N"
        next_key = ""
        page = 1
        stock_records = []
        
        while True:
            logger.info(f"  페이지 {page} 조회 중 (next_key: {next_key or 'None'})...")
            try:
                res = market_condition.stockwise_program_trading_by_day_request_ka90013(
                    stock_code=code,
                    amount_quantity_type=amount_qty_tp,
                    cont_yn=cont_yn,
                    next_key=next_key
                )
                
                return_code = str(res.get("return_code", "-1"))
                return_msg = res.get("return_msg", "")
                
                if return_code != "0":
                    logger.error(f"  종목 {code} API 호출 실패 (코드: {return_code}, 메시지: {return_msg})")
                    break
                    
                items = res.get("stk_daly_prm_trde_trnsn", [])
                if not items:
                    logger.info("  수집된 일별 데이터가 없습니다.")
                    break
                    
                valid_items = [item for item in items if item.get('dt')]
                if not valid_items:
                    break
                    
                stock_records.extend(valid_items)
                logger.info(f"  페이지 {page} 완료: {len(valid_items)}개 레코드 추가 (누적: {len(stock_records)}개)")
                
                next_key = res.get("next-key", "")
                if not next_key or next_key == "":
                    break
                    
                cont_yn = "Y"
                page += 1
                
            except Exception as e:
                logger.error(f"  종목 {code} 수집 중 오류 발생: {e}")
                break
                
        if stock_records:
            output_file = output_dir / f"program_daily_trend_{code}.csv"
            try:
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(header_mapping)
                    for record in stock_records:
                        filtered_record = {k: record.get(k, '') for k in headers}
                        writer.writerow(filtered_record)
                logger.info(f"  성공! [{code}] 일별 추이가 CSV로 저장되었습니다: {output_file}")
            except Exception as e:
                logger.error(f"  CSV 파일 저장 중 오류 발생: {e}")
        else:
            logger.warning(f"  [{code}] 수집된 데이터가 없어 CSV 파일을 생성하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description="키움 REST API 프로그램 매매 내역 다운로드 도구")
    parser.add_argument(
        "--mode", "-m",
        choices=["summary", "time-trend", "daily-trend"],
        required=True,
        help="다운로드 모드 (summary: 전종목 요약 현황, time-trend: 특정종목 시간대별 추이, daily-trend: 특정종목 일별 추이)"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help="대상 날짜 (YYYYMMDD 형식, 기본값: 오늘 날짜, summary/time-trend 모드에 적용)"
    )
    parser.add_argument(
        "--market", "-k",
        choices=["kospi", "kosdaq", "all"],
        default="all",
        help="summary 모드 시 대상 시장 (기본값: all)"
    )
    parser.add_argument(
        "--stocks", "-s",
        type=str,
        help="time-trend/daily-trend 모드 시 조회할 종목코드 목록 (쉼표로 구분, 예: 005930,000660)"
    )
    parser.add_argument(
        "--amount-type", "-a",
        choices=["1", "2"],
        default="1",
        help="금액/수량 구분 (1: 금액(백만원), 2: 수량(천주), 기본값: 1)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(project_root / "data" / "program_trading"),
        help="출력 디렉토리 경로"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.mode in ["time-trend", "daily-trend"] and not args.stocks:
        parser.error(f"--mode {args.mode}를 선택한 경우 --stocks (-s) 종목코드 목록 인자가 필수입니다.")
        
    market_condition, stock_info = get_api_clients()
    
    if args.mode == "summary":
        download_summary(stock_info, args.date, args.market, args.amount_type, output_dir)
    elif args.mode == "time-trend":
        stock_list = args.stocks.split(",")
        download_time_trend(market_condition, args.date, stock_list, args.amount_type, output_dir)
    elif args.mode == "daily-trend":
        stock_list = args.stocks.split(",")
        download_daily_trend(market_condition, stock_list, args.amount_type, output_dir)


if __name__ == "__main__":
    main()
