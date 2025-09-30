import argparse
import os
import re
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures as _fut
import tempfile as _tempfile
import shutil as _shutil
import pickle as _pickle

import numpy as np
import pandas as pd

# Global sequential counter for '번호'
NO_COUNTER: int = 1

INPUT_TABLE = "datasets"  # 입력 DuckDB 테이블명
TEXT_COLUMNS = {"종목코드", "종목명", "시간", *{f"매도거래원{i}" for i in range(1, 6)}, *{f"매수거래원{i}" for i in range(1, 6)}}
DROP_COLUMNS = {"종류", "씨리얼"}
REQUIRED_TYPES = {"execution", "orderbook", "trader"}
IGNORING_STOCKS = "삼성전자, SK하이닉스, 삼성바이오로직스, LG에너지솔루션, 현대차, 셀트리온, 삼성전자우, 기아, NAVER, KB금융, HD현대중공업, 신한지주, 현대모비스, 한화에어로스페이스, 메리츠금융지주, 삼성물산, 알테오젠, POSCO홀딩스, 한화오션, 카카오, SK이노베이션, 크래프톤, 삼성생명, 삼성화재, 하나금융지주, 고려아연, HMM, HD한국조선해양, LG화학, 두산에너빌리티, 삼성SDI, HD현대일렉트릭, KT&G, 한국전력, LG전자, SK스퀘어, 우리금융지주, 기업은행, SK텔레콤, 에코프로비엠, KT, 카카오뱅크, 삼성중공업, 유한양행, 삼성전기, LG, SK, 하이브, 현대글로비스, SK바이오팜, HLB, 포스코퓨처엠, 삼성에스디에스, 한미반도체, KODEXCD금리액티브(합성), 대한항공, 현대로템, TIGER미국S&P500, 에코프로, 레인보우로보틱스, 포스코인터내셔널, DB손해보험, HD현대마린솔루션, 아모레퍼시픽, S-Oil, LSELECTRIC, 삼양식품, HD현대, KODEX200, TIGERCD금리투자KIS(합성), LIG넥스원, 현대차2우B, SKC, 코웨이, 한진칼, 맥쿼리인프라, LG씨엔에스, 두산, KODEX머니마켓액티브, 에코프로머티, 한국항공우주, 삼성카드, LG생활건강, 미래에셋증권, 한화시스템, NH투자증권, TIGER미국나스닥100, 한국타이어앤테크놀로지, 두산밥캣, LG디스플레이, 효성중공업, LG유플러스, 두산로보틱스, 한국금융지주, KODEXKOFR금리액티브(합성), 카카오페이, HD현대미포, 리가켐바이오, 삼천당제약, 삼성증권, 오리온, 현대오토에버, 넷마블, 엔씨소프트, KODEX미국S&P500, 시프트업, LG이노텍, 현대차우, SK바이오사이언스, GS, 삼성E&A, 현대건설, CJ제일제당, LS, BNK금융지주, 한화솔루션, 리노공업, JB금융지주, TIGERKOFR금리액티브(합성), 강원랜드, TIGER미국테크TOP10INDXX, 한미약품, 클래시스, 휴젤, 금호석유, 파마리서치, 키움증권, JYPEnt., 엘앤에프, KODEX종합채권(AA-이상)액티브, 포스코DX, 현대제철, 코오롱티슈진, 한온시스템, 한국가스공사, CJ, RISE머니마켓액티브, 한전기술, F&F, HPSP, TIGER미국필라델피아반도체나스닥, 신성델타테크, 더존비즈온, 이수페타시스, 한화, 펩트론, 대한전선, KODEX레버리지, KCC, KODEX200TR, 동서, 롯데케미칼, 휠라홀딩스, 엔켐, 에스원, 에스엠, 롯데지주, TIGER미국배당다우존스, 산일전기, 한화생명, 셀트리온제약, 현대엘리베이, 현대해상, 아시아나항공, 농심, 보로노이, SK가스, 펄어비스, 한화비전, TIGERMSCIKoreaTR, 한올바이오파마, HL만도, 제일기획, 한전KPS, CJ대한통운, 영원무역, 루닛, ACE미국30년국채액티브(H), KODEX미국나스닥100, 한미사이언스, TIGER200, 에이피알, DB하이텍, KODEXCD1년금리플러스액티브(합성), 이마트, 에이비엘바이오, ACE미국S&P500, TIGER차이나전기차SOLACTIVE, 한화엔진, 팬오션, 에스티팜, 이오테크닉스, 아모레G, BGF리테일, 코스맥스, KODEX코스닥150레버리지, 실리콘투, 씨에스윈드, SK아이이테크놀로지, 케어젠, KODEX단기채권PLUS, 파크시스템스, 녹십자, 코리안리, 코스모신소재, DGB금융지주, 롯데쇼핑, 대웅제약, 테크윙, ISC, 호텔신라, HD현대인프라코어, 대주전자재료, OCI홀딩스, 풍산, 오뚜기, 일진전기, 한국앤컴퍼니, 솔브레인, GS건설, 주성엔지니어링, 금호타이어, DL이앤씨, TIGER25-10회사채(A+이상)액티브, 카페24, 위메이드, 대우건설, ACE미국나스닥100, SOOP, 한국콜마, RISE종합채권(A-이상)액티브, 카카오게임즈, 동진쎄미켐, 동원산업, SK리츠, HD현대건설기계, 하이트진로, 에스디바이오센서, 에스엘, 제이앤티씨, 서진시스템, 브이티, 신영증권, 신세계, 고영, KODEX코스닥150, 원익IPS, LS에코에너지, 이수스페셜티케미컬, GS리테일, CJENM, DN오토모티브, 스튜디오드래곤, KODEX200선물인버스2X, TIGER미국30년국채커버드콜액티브(H), HLB생명과학, 대웅, 씨젠, HDC현대산업개발, 하이젠알앤엠, 현대백화점, 경동나비엔, TIGER25-12금융채(AA-이상), 롯데에너지머티리얼즈, 영원무역홀딩스, 종근당, 피에스케이홀딩스, 한솔케미칼, 세방전지, 금양, TIGERCD1년금리액티브(합성), 필옵틱스, 네이처셀, 메리츠KISCD금리투자ETN, 오스코텍, KODEX삼성그룹, RISE200, 더블유게임즈, 한샘, LX인터내셔널, 현대위아, HK이노엔, 한일시멘트, 롯데정밀화학, LX세미콘, 동원시스템즈, 롯데렌탈, 롯데웰푸드, KODEX25-12은행채(AAA)액티브, 두산퓨얼셀, 와이지엔터테인먼트, ESR켄달스퀘어리츠, 디어유, RISE미국나스닥100, KODEX25-11은행채(AA-이상)PLUS액티브, 티씨케이, 하나CD금리투자ETN, 효성티앤씨, 와이씨, 롯데칠성, ACEKRX금현물, TIGER미국배당다우존스타겟커버드콜2호, 오리온홀딩스, 레이크머티리얼즈, 젬백스, RISECD금리액티브(합성), 피엔티, 하나투어, LG화학우, HD현대마린엔진, 넥슨게임즈, SK네트웍스, 엠로, 솔루엠, 프레스티지바이오파마, 파라다이스, KB발해인프라, 롯데리츠, 국일제지, 대덕전자, 유진테크, 보령, SNT다이내믹스, DI동일, HS효성첨단소재, RISE미국S&P500, 메디톡스, 삼성화재우, KODEX2차전지산업, 빙그레, 세아제강지주, LS머트리얼즈, 메지온, 태성, 덕산네오룩스, 대신증권, CJCGV, 미원상사, TIGER2차전지테마, 안랩, 한국단자, 율촌화학, 다우기술, HLB테라퓨틱스, 쿠쿠홀딩스, SOL미국배당다우존스, 전진건설로봇, SOL종합채권(AA-이상)액티브, ACE테슬라밸류체인액티브, 덴티움, 현대지에프홀딩스, 미래에셋생명, HLB제약, 코스모화학, 코오롱인더, 효성, 위메이드맥스, HDC, 나노신소재, ACE종합채권(AA-이상)KIS액티브, 태광산업, 하나마이크론, TIGERTOP10, 영풍, SK오션플랜트, N2KISCD금리투자ETN, SK케미칼, 성광벤드, TIGER단기통안채, KIWOOM200TR, KODEXMSCIKoreaTR, 이노션, 티웨이항공, 동양생명, 대상, NICE평가정보, 파두, KODEXTop5PlusTR, GKL, 한화투자증권, KG모빌리티, TIGER종합채권(AA-이상)액티브, 덕산테코피아, 솔브레인홀딩스, TIGER미국30년국채스트립액티브(합성H), 미원에스씨, SOL조선TOP3플러스, 에스피지, 신한알파리츠, 중앙첨단소재, KODEX단기채권, 현대힘스, 동국제약, 교보증권, 제룡전기, 한국카본, 두산테스나, TIGER인도니프티50, ACE미국빅테크TOP7Plus, 바이넥스, 녹십자홀딩스, NHN, DL, 솔루스첨단소재, SK디스커버리, 에스에프에이, 태영건설, LG전자우, 케이씨텍, TKG휴켐스, TIGER미국달러단기채권액티브, 차바이오텍, 케이카, 티앤엘, 에스앤에스텍, TIGER미국나스닥100타겟데일리커버드콜, 대한유화, 세아베스틸지주, KODEX미국빅테크10(H), PI첨단소재, 컴투스, 올릭스, 이녹스첨단소재, 동아쏘시오홀딩스, 비에이치아이, 삼성레버리지WTI원유선물ETN, N2코스피변동성매칭형양매도ETN, TCC스틸, ACE미국배당다우존스, TIGER차이나항셍테크, 한화리츠, 카프로, 가온칩스, 제주반도체, 에코프로에이치엔, SOL초단기채권액티브, 화승엔터프라이즈, 지아이이노베이션, 동성화인텍, 엠앤씨솔루션, 비츠로셀, KG스틸, 동원F&B, 롯데관광개발, 넥스틴, 현대홈쇼핑, SFA반도체, 디앤디파마텍, 제주항공, TIGER리츠부동산인프라, TIGERFn반도체TOP10, 하림지주, 아이에스동서, PLUS200, 피에스케이, 하나머티리얼즈, 한글과컴퓨터, KODEX미국30년국채타겟커버드콜(합성H), KODEX반도체, 파미셀, 에스티큐브, 현대바이오, 씨아이에스, 비에이치, 우리기술투자, KODEX인버스, KCC글라스, 일진하이솔루스, 미래에셋증권2우B, 명신산업, OCI, 태광, 드림텍, 가온전선, 롯데손해보험, JW중외제약, TIGER미국달러SOFR금리액티브(합성), HJ중공업, 원익QnC, 유일로보틱스, 대한해운, 펌텍코리아, SNT에너지, KODEX국고채30년액티브, 삼양홀딩스, 키움CD금리투자ETN, 신풍제약, 시노펙스, DS단석, 코스메카코리아, 두산우, 쎄트렉아이, 유안타증권, TIGER200IT, 후성, STX엔진, LS마린솔루션, 아난티, 한세실업, 아세아, 넥센타이어, TIGER단기채권액티브, LX홀딩스, 진에어, KBKISCD금리투자ETN, 현대무벡스, 신한레버리지WTI원유선물ETN(H), 포스코엠텍, 쏘카, KODEX미국반도체MV".split(",")
IGNORING_STOCKS_SET = {s.strip() for s in IGNORING_STOCKS}

# Final column order required
FINAL_COLUMNS: List[str] = [
    "종목코드", "종목명", "시간", "현재가", "등락률", "거래량", "누적거래량", "누적거래대금", "시가", "고가", "저가",
    "전일거래량대비", "전일거래량대비비율", "거래회전율", "체결강도",
    # 매도호가/수량/직전대비 1~10
    *[f"매도호가{i}" for i in range(1, 11)],
    *[f"매도호가수량{i}" for i in range(1, 11)],
    *[f"매도호가직전대비{i}" for i in range(1, 11)],
    # 매수호가/수량/직전대비 1~10
    *[f"매수호가{i}" for i in range(1, 11)],
    *[f"매수호가수량{i}" for i in range(1, 11)],
    *[f"매수호가직전대비{i}" for i in range(1, 11)],
    "매도호가총잔량", "매도호가총잔량직전대비", "매수호가총잔량", "매수호가총잔량직전대비",
    # 거래원 관련 (문자열 컬럼들)
    *[f"매도거래원{i}" for i in range(1, 6)],
    *[f"매도거래원수량{i}" for i in range(1, 6)],
    *[f"매도거래원별증감{i}" for i in range(1, 6)],
    *[f"매수거래원{i}" for i in range(1, 6)],
    *[f"매수거래원수량{i}" for i in range(1, 6)],
    *[f"매수거래원별증감{i}" for i in range(1, 6)],
]


def find_duckdb_groups(input_db: str) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    """
    입력 DuckDB에서 데이터 그룹을 찾고 그룹화
    Returns: {group_key: {"merged": (db_path, code, name, date)}}
    """
    if not os.path.exists(input_db):
        return {}
    
    groups = {}
    try:
        conn = duckdb.connect(input_db, read_only=True)
        try:
            # 테이블 존재 확인
            try:
                conn.execute(f"DESCRIBE {INPUT_TABLE}")
            except Exception:
                print(f"경고: 입력 DB에 '{INPUT_TABLE}' 테이블이 없습니다.")
                return {}
            
            # 종목코드, 종목명, 날짜별로 그룹화
            query = f"""
                SELECT DISTINCT "종목코드", "종목명", "날짜"
                FROM {INPUT_TABLE}
                WHERE "종목코드" IS NOT NULL 
                  AND "종목명" IS NOT NULL 
                  AND "날짜" IS NOT NULL
                ORDER BY "날짜", "종목코드"
            """
            results = conn.execute(query).fetchall()
            
            for code, name, date in results:
                group_key = f"{code}_{name}_{date}"
                # 입력 DB에서는 이미 병합된 데이터이므로 단일 타입으로 처리
                groups[group_key] = {
                    "merged": (input_db, str(code), str(name), str(date))
                }
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 그룹 스캔 실패: {type(e).__name__}: {e}")
        return {}
    
    return groups


def _clean_column_name(col: str) -> str:
    # 내부 공백 제거 및 알려진 별칭 통일
    c = re.sub(r"\s+", "", col)
    if c == "스탬프":
        return "시간"
    return c


def load_and_clean_from_duckdb(db_path: str, code: str, name: str, date: str, 
                                time_start: int = 90000000, time_end: int = 110000000) -> pd.DataFrame:
    """
    DuckDB에서 특정 종목코드/날짜의 데이터를 로드하고 기본 정리
    time_start: 시작 시간 (기본: 90000000 = 오전 9시)
    time_end: 종료 시간 (기본: 110000000 = 오전 11시, 미포함)
    """
    try:
        conn = duckdb.connect(db_path, read_only=True)
        try:
            query = f"""
                SELECT *
                FROM {INPUT_TABLE}
                WHERE "종목코드" = ? AND "날짜" = ?
                ORDER BY "번호"
            """
            df = conn.execute(query, [code, date]).df()
            
            if df.empty:
                return pd.DataFrame()
            
            # 컬럼명 정규화 (이미 정규화되어 있을 수 있지만 안전장치)
            df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})

            # 종목명 필터링: IGNORING_STOCKS에 포함된 종목은 제외
            if '종목명' in df.columns:
                mask_ignore = df['종목명'].astype(str).str.strip().isin(IGNORING_STOCKS_SET)
                if mask_ignore.any():
                    df = df[~mask_ignore]
            
            # 시간 필터링 적용
            if '시간' in df.columns:
                # 시간 컬럼을 숫자로 변환
                df['시간_numeric'] = pd.to_numeric(df['시간'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0).astype(int)
                # 시간 범위 필터링: time_start <= 시간 < time_end
                df = df[(df['시간_numeric'] >= time_start) & (df['시간_numeric'] < time_end)]
                # 임시 컬럼 제거
                df = df.drop(columns=['시간_numeric'])
            
            # 불필요 컬럼 제거
            df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")
            
            # '번호' 숫자화 보정
            if '번호' in df.columns:
                df['번호'] = pd.to_numeric(df['번호'], errors='coerce')
            
            return df
        finally:
            conn.close()
    except Exception as e:
        print(f"경고: DuckDB 로드 실패 ({code}, {date}): {type(e).__name__}: {e}")
        return pd.DataFrame()


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    비어있는 데이터를 직전/직후 데이터로 채우기
    """
    # 최후 방어: 중복 컬럼 제거(이미 로드 시 처리했지만 합병 과정에서 생길 수 있음)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 번호 컬럼 기준으로 정렬
    df = df.sort_values('번호').reset_index(drop=True)
    
    # 텍스트 컬럼과 숫자 컬럼 분리
    text_cols = [col for col in df.columns if col in TEXT_COLUMNS]
    numeric_cols = [col for col in df.columns if col not in TEXT_COLUMNS and col != '번호']
    
    # 텍스트 컬럼: forward fill 후 backward fill, 그래도 없으면 빈 문자열
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna('').infer_objects(copy=False)
    
    # 숫자 컬럼: forward fill 후 backward fill, 그래도 없으면 0
    for col in numeric_cols:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                # 안전장치: 만약 여전히 DataFrame이면 첫 열 사용
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors='coerce')
            series = series.ffill().bfill().fillna(0)
            df[col] = series
    
    return df


def _coalesce_into_base(base: pd.DataFrame, temp: pd.DataFrame, overlap_cols: List[str]) -> pd.DataFrame:
    """
    base와 temp(번호로 병합된 상태)에서 동일 컬럼이 있을 때 base의 NaN을 temp의 값으로 보완
    overlap_cols에 대해 base[col] = base[col].combine_first(temp[f"{col}_new"]) 수행
    """
    for col in overlap_cols:
        new_col = f"{col}_new"
        if new_col in temp.columns:
            # 숫자/문자 모두 지원되는 combine_first 사용
            base[col] = base[col].combine_first(temp[new_col])
            temp = temp.drop(columns=[new_col])
    return base, temp


def merge_from_duckdb(group_info: Dict[str, Tuple[str, str, str, str]], code: str, name: str, date: str,
                       time_start: int = 90000000, time_end: int = 110000000) -> pd.DataFrame:
    """
    DuckDB에서 데이터를 로드하여 병합 (이미 병합된 데이터인 경우 그대로 반환)
    """
    # 입력 DB에서는 이미 병합된 상태이므로 단순히 로드만 수행
    if "merged" in group_info:
        db_path, code, name, date = group_info["merged"]
        df = load_and_clean_from_duckdb(db_path, code, name, date, time_start, time_end)
        
        if df.empty:
            return pd.DataFrame()
        
        # 결측값 채우기
        df = fill_missing_values(df)
        
        # 종목코드, 종목명 보정
        df['종목코드'] = df.get('종목코드', pd.Series(index=df.index, dtype=object)).fillna(code).replace({"": code})
        df['종목명'] = df.get('종목명', pd.Series(index=df.index, dtype=object)).fillna(name).replace({"": name})
        
        # 누락 컬럼 생성 (최종 스키마 강제)
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col in TEXT_COLUMNS else 0
        
        # 채우기(직전/직후)로 결측 제거
        df = fill_missing_values(df)
        
        # 최종 컬럼 순서 맞추기
        df = df[["번호", *FINAL_COLUMNS]]
        
        return df
    
    # 레거시: 여러 타입이 분리되어 있는 경우 (향후 확장용)
    return pd.DataFrame()


def _signed_log1p(arr: pd.Series) -> pd.Series:
    """Signed log1p that supports negative values: sign(x) * log1p(|x|)."""
    x = pd.to_numeric(arr, errors="coerce").fillna(0)
    return np.sign(x) * np.log1p(np.abs(x))


def _standard_scale(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    mean = float(x.mean())
    std = float(x.std(ddof=0))
    if std == 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - mean) / std


def _minmax_scale(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_v = float(x.min())
    max_v = float(x.max())
    rng = max_v - min_v
    if rng == 0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - min_v) / rng


def apply_feature_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns per rules:
    - Log + Standard: specified quantity and flow columns
    - Standard only: *_총잔량직전대비
    - Character-level scalar encoding for broker categorical columns (append scalar, keep originals)
    - Character-level scalar encoding for '종목명' (append scalar, keep original)
    - '시간' normalized to '시간_scalar' by dividing numeric value by 90000000.0 (keep original '시간')
    - Min-Max: remaining numeric columns (excluding '번호' and text columns and already-normalized columns)
    """
    out = df.copy()

    # Define column groups
    logstd_cols = set([
        "거래량", "누적거래량", "누적거래대금", "거래회전율",
        *[f"매도호가수량{i}" for i in range(1, 11)],
        *[f"매수호가수량{i}" for i in range(1, 11)],
        "매도호가총잔량", "매수호가총잔량",
        *[f"매도거래원수량{i}" for i in range(1, 6)],
        *[f"매수거래원수량{i}" for i in range(1, 6)],
        *[f"매도거래원별증감{i}" for i in range(1, 6)],
        *[f"매수거래원별증감{i}" for i in range(1, 6)],
        # 외국계 추정 관련 (없으면 생성 후 0)
        "외국계매도추정합", "외국계매수추정합", "외국계매도추정합변동", "외국계매수추정합변동",
    ])

    stdonly_cols = set(["매도호가총잔량직전대비", "매수호가총잔량직전대비", "등락률"])

    broker_cat_cols = [*[f"매도거래원{i}" for i in range(1, 6)], *[f"매수거래원{i}" for i in range(1, 6)]]

    # Ensure optional columns exist
    for col in list(logstd_cols | stdonly_cols):
        if col not in out.columns:
            out[col] = 0

    # Log + Standard scaling (signed log1p then z-score)
    for col in sorted(logstd_cols):
        if col in out.columns:
            out[col] = _standard_scale(_signed_log1p(out[col]))

    # Standard only
    for col in sorted(stdonly_cols):
        if col in out.columns:
            out[col] = _standard_scale(out[col])

    # Character-level scalar encoding for broker categorical columns (append scalar, keep originals)
    broker_scalar_cols: List[str] = []
    for col in broker_cat_cols:
        if col in out.columns:
            cats = out[col].astype(str).replace({"nan": ""})
            # Build per-column char dictionary
            unique_chars = sorted(set("".join(cats.tolist())))
            if len(unique_chars) == 0:
                # empty column -> scalar zeros
                out[f"{col}_scalar"] = 0.0
                broker_scalar_cols.append(f"{col}_scalar")
                continue
            char_to_id = {ch: i + 1 for i, ch in enumerate(unique_chars)}  # 1..N
            max_id = float(len(unique_chars))

            def _encode_scalar(s: str) -> float:
                if not s:
                    return 0.0
                ids = [char_to_id.get(ch, 0) for ch in s]
                if not ids:
                    return 0.0
                # mean of ids normalized by max id -> [0,1]
                return float(np.mean(ids)) / max_id

            out[f"{col}_scalar"] = cats.apply(_encode_scalar).astype(float)
            broker_scalar_cols.append(f"{col}_scalar")

    # Character-level scalar encoding for '종목명'
    if '종목명' in out.columns:
        cats = out['종목명'].astype(str).replace({"nan": ""})
        unique_chars = sorted(set("".join(cats.tolist())))
        if len(unique_chars) == 0:
            out["종목명_scalar"] = 0.0
        else:
            char_to_id = {ch: i + 1 for i, ch in enumerate(unique_chars)}
            max_id = float(len(unique_chars))

            def _encode_scalar_name(s: str) -> float:
                if not s:
                    return 0.0
                ids = [char_to_id.get(ch, 0) for ch in s]
                if not ids:
                    return 0.0
                return float(np.mean(ids)) / max_id

            out["종목명_scalar"] = cats.apply(_encode_scalar_name).astype(float)
        broker_scalar_cols.append("종목명_scalar")

    # '시간' -> '시간_scalar' (numeric normalization by 90000000.0)
    if '시간' in out.columns:
        def _to_num_time(v) -> float:
            try:
                # fast path for numeric
                val = float(v)
                return val
            except Exception:
                s = str(v)
                digits = re.sub(r"\D", "", s)
                if not digits:
                    return 0.0
                try:
                    return float(digits)
                except Exception:
                    return 0.0

        num_time = out['시간'].apply(_to_num_time).astype(float)
        denom = 90000000.0
        out['시간_scalar'] = (num_time / denom).astype(float)
        broker_scalar_cols.append('시간_scalar')

    # Min-Max for remaining numeric columns not already processed
    processed = set(["번호"]) | TEXT_COLUMNS | logstd_cols | stdonly_cols | set(broker_scalar_cols)
    numeric_rest = [c for c in out.columns if c not in processed and pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_rest:
        out[col] = _minmax_scale(out[col])

    # Final safety: fill any remaining NaNs
    for col in out.columns:
        if col in TEXT_COLUMNS:
            out[col] = out[col].astype(str).replace({"nan": ""}).fillna("")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


def ensure_datasets_table_duckdb(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    table = "datasets"
    # Create table if not exists
    try:
        conn.execute(f"DESCRIBE {table}")
        exists = True
    except Exception:
        exists = False
    if not exists:
        # Build explicit schema to preserve types (avoid df.head(0) inference)
        col_defs: list[str] = []
        for col in df.columns:
            series = df[col]
            if col == '번호' or pd.api.types.is_integer_dtype(series):
                duck_type = 'BIGINT'
            elif col in TEXT_COLUMNS or col in {'날짜', '종목명'} or series.dtype == object:
                duck_type = 'VARCHAR'
            else:
                duck_type = 'DOUBLE'
            col_defs.append(f'"{col}" {duck_type}')
        create_sql = f"CREATE TABLE {table} ({', '.join(col_defs)})"
        conn.execute(create_sql)
    else:
        # Use correct indices from PRAGMA table_info: (cid, name, type, null, default, pk)
        existing_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        existing_cols = [row[1] for row in existing_info]
        existing_types = {row[1]: (row[2] or "").upper() for row in existing_info}
        for col in df.columns:
            if col not in existing_cols:
                series = df[col]
                if col in TEXT_COLUMNS or col == '날짜' or series.dtype == object:
                    col_type = 'VARCHAR'
                elif pd.api.types.is_integer_dtype(series) or col == '번호':
                    col_type = 'BIGINT'
                else:
                    col_type = 'DOUBLE'
                conn.execute(f"ALTER TABLE {table} ADD COLUMN \"{col}\" {col_type}")
        # Enforce VARCHAR for known text columns if mismatched (e.g., '종목명' mistakenly INT)
        text_like = set(TEXT_COLUMNS) | {"날짜", "종목명"}
        # Drop dependent index before altering types to avoid catalog error
        try:
            conn.execute("DROP INDEX IF EXISTS idx_datasets_code_date")
        except Exception:
            pass
        for col in (c for c in df.columns if c in text_like and c in existing_types):
            ctype = existing_types.get(col, "")
            if "CHAR" not in ctype and "STRING" not in ctype and "VARCHAR" not in ctype:
                conn.execute(f"ALTER TABLE {table} ALTER COLUMN \"{col}\" TYPE VARCHAR")
    # Helpful index (recreate if dropped)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_code_date ON datasets(\"종목코드\", \"날짜\")")


def _month_key_from_yyyymmdd(date_str: str) -> str:
    """Extract YYYYMM from YYYYMMDD string."""
    return date_str[:6] if len(date_str) >= 6 else date_str


def _monthly_db_path(base_db_path: str, yyyymm: str) -> str:
    """Return a per-month DuckDB path based on base path and yyyymm.
    Example: base 'datasets.duckdb' -> 'datasets_YYYYMM.duckdb' in same directory.
    """
    p = Path(base_db_path)
    stem = p.stem
    suffix = p.suffix or ".duckdb"
    return str(p.with_name(f"{stem}_{yyyymm}{suffix}"))


def _ingest_pickle_into_db_path(db_path: str, pkl_path: str, code: str, date: str, group_key: str):
    """Open a DuckDB connection to db_path and ingest the pickle contents, then remove the pickle."""
    try:
        with open(pkl_path, 'rb') as f:
            merged_df = _pickle.load(f)
        # Overwrite/assign global sequential '번호'
        global NO_COUNTER
        n_rows = len(merged_df)
        if n_rows > 0:
            merged_df['번호'] = np.arange(NO_COUNTER, NO_COUNTER + n_rows, dtype=np.int64)
            NO_COUNTER += n_rows
        conn = duckdb.connect(db_path)
        try:
            ensure_datasets_table_duckdb(conn, merged_df)
            conn.register("_batch_df", merged_df)
            try:
                conn.execute("DELETE FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
            except Exception:
                pass
            conn.execute("INSERT INTO datasets SELECT * FROM _batch_df")
            conn.unregister("_batch_df")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        print(f"완료(즉시 반영): {group_key} - {len(merged_df)} 행 -> {db_path}")
    finally:
        try:
            os.remove(pkl_path)
        except Exception:
            pass


def _prep_pkl_metadata(pkl_path: str) -> Optional[Tuple[str, str, str, str]]:
    """Top-level helper: from a pickle file path, extract (group_key, code, date, pkl_path).
    Returns None if filename doesn't match expected pattern. Using only built-in types for pickling safety.
    """
    try:
        name = os.path.basename(pkl_path)
        group_key = os.path.splitext(name)[0]
        parts = group_key.split('_')
        if len(parts) < 2:
            return None
        code = parts[0]
        date = parts[-1]
        return group_key, code, date, pkl_path
    except Exception:
        return None


def _worker_process(group_key: str, group_info: Dict[str, Tuple[str, str, str, str]], tmp_root: str,
                    time_start: int = 90000000, time_end: int = 110000000) -> Tuple[str, str, str, str]:
    """Top-level worker for multiprocessing: merge + normalize + pickle dump.
    Returns (group_key, code, date, out_pickle_path).
    """
    parts = group_key.split('_')
    code = parts[0]
    date = parts[-1]
    name = '_'.join(parts[1:-1])
    
    merged_df = merge_from_duckdb(group_info, code, name, date, time_start, time_end)
    
    if merged_df.empty:
        return group_key, code, date, ""
    
    merged_df = apply_feature_normalization(merged_df)
    merged_df = merged_df.copy()
    merged_df['날짜'] = date
    merged_df = pd.concat([merged_df['날짜'], merged_df.drop(columns=['날짜'])], axis=1)
    out_path = Path(tmp_root) / f"{group_key}.pkl"
    # Write to a temp file then atomically replace to avoid partial reads
    out_tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(out_tmp, 'wb') as f:
        _pickle.dump(merged_df, f, protocol=_pickle.HIGHEST_PROTOCOL)
    try:
        os.replace(out_tmp, out_path)
    except Exception:
        # Best-effort fallback
        try:
            os.remove(out_tmp)
        except Exception:
            pass
    return group_key, code, date, str(out_path)


def _process_single_group_to_pickle(group_key: str, group_info: Dict[str, Tuple[str, str, str, str]], tmp_root: str,
                                     time_start: int = 90000000, time_end: int = 110000000) -> Optional[str]:
    """Process a single group and save to pickle file. Returns pickle path or None on error."""
    try:
        parts = group_key.split('_')
        code = parts[0]
        date = parts[-1]
        name = '_'.join(parts[1:-1])
        
        merged_df = merge_from_duckdb(group_info, code, name, date, time_start, time_end)
        
        if merged_df.empty:
            return None
        
        merged_df = apply_feature_normalization(merged_df)
        merged_df = merged_df.copy()
        merged_df['날짜'] = date
        merged_df = pd.concat([merged_df['날짜'], merged_df.drop(columns=['날짜'])], axis=1)
        
        # Save to pickle
        out_path = Path(tmp_root) / f"{group_key}.pkl"
        out_tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(out_tmp, 'wb') as f:
            _pickle.dump(merged_df, f, protocol=_pickle.HIGHEST_PROTOCOL)
        try:
            os.replace(out_tmp, out_path)
        except Exception:
            try:
                os.remove(out_tmp)
            except Exception:
                pass
            raise
        return str(out_path)
    except Exception as e:
        print(f"  경고: 그룹 처리 실패({group_key}): {type(e).__name__}: {e}")
        return None


def _ingest_pickles_to_db(pickle_paths: List[str], db_path: str, yyyymm: str, checkpoint_interval: int) -> int:
    """Ingest pickle files to DB sequentially with periodic checkpoints."""
    import duckdb
    processed_count = 0
    
    for pkl_path in pickle_paths:
        if not os.path.exists(pkl_path):
            continue
            
        try:
            # Load pickle and extract metadata
            with open(pkl_path, 'rb') as f:
                merged_df = _pickle.load(f)
            # Overwrite/assign global sequential '번호'
            global NO_COUNTER
            n_rows = len(merged_df)
            if n_rows > 0:
                merged_df['번호'] = np.arange(NO_COUNTER, NO_COUNTER + n_rows, dtype=np.int64)
                NO_COUNTER += n_rows
            
            group_key = Path(pkl_path).stem
            parts = group_key.split('_')
            code = parts[0]
            date = parts[-1]
            
            # Write to DB
            conn = duckdb.connect(db_path)
            try:
                ensure_datasets_table_duckdb(conn, merged_df)
                conn.register("_batch_df", merged_df)
                try:
                    conn.execute("DELETE FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=?", [code, date])
                except Exception:
                    pass
                conn.execute("INSERT INTO datasets SELECT * FROM _batch_df")
                conn.unregister("_batch_df")
            finally:
                conn.close()
            
            processed_count += 1
            print(f"  {yyyymm}: [{processed_count}] {group_key} - {len(merged_df)} 행 -> DB")
            
            # Remove pickle file after successful ingestion
            try:
                os.remove(pkl_path)
            except Exception:
                pass
            
            # Periodic checkpoint
            if checkpoint_interval > 0 and processed_count % checkpoint_interval == 0:
                try:
                    conn_ck = duckdb.connect(db_path)
                    try:
                        conn_ck.execute("CHECKPOINT")
                        print(f"  {yyyymm}: 체크포인트 실행 ({processed_count} 그룹 완료)")
                    finally:
                        conn_ck.close()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"  경고: {yyyymm} pickle 처리 실패({pkl_path}): {type(e).__name__}: {e}")
    
    # Final checkpoint
    if processed_count > 0:
        try:
            conn_ck = duckdb.connect(db_path)
            try:
                conn_ck.execute("CHECKPOINT")
                print(f"  {yyyymm}: 최종 체크포인트 실행 ({processed_count} 그룹 완료)")
            finally:
                conn_ck.close()
        except Exception:
            pass
    
    return processed_count


def _process_monthly_groups(month_groups: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]], 
                           db_path: str, tmp_root: str, yyyymm: str, 
                           checkpoint_interval: int, group_workers: int = 1,
                           time_start: int = 90000000, time_end: int = 110000000) -> int:
    """Process all groups for a specific month with parallel group processing.
    Returns the number of processed groups.
    """
    if not month_groups:
        return 0
    
    total_groups = len(month_groups)
    print(f"  {yyyymm}: {total_groups} 그룹 처리 시작 (group_workers={group_workers})", flush=True)
    
    # Step 1: Process groups to pickle files in parallel
    pickle_paths: List[str] = []
    max_group_workers = max(1, int(group_workers))
    used_group_workers = min(max_group_workers, total_groups)
    
    if used_group_workers > 1:
        print(f"  {yyyymm}: 병렬 그룹 처리 (workers={used_group_workers})", flush=True)
        with _fut.ProcessPoolExecutor(max_workers=used_group_workers) as ex:
            futs = []
            for group_key, group_info in month_groups:
                fut = ex.submit(_process_single_group_to_pickle, group_key, group_info, tmp_root, time_start, time_end)
                futs.append(fut)
            done = 0
            for fut in _fut.as_completed(futs):
                try:
                    pkl_path = fut.result()
                    if pkl_path:
                        pickle_paths.append(pkl_path)
                        # Incremental ingest when enough pickles are ready
                        if checkpoint_interval > 0 and len(pickle_paths) >= checkpoint_interval:
                            batch = pickle_paths[:checkpoint_interval]
                            print(f"  {yyyymm}: 증분 반영 시작 (batch={len(batch)}) -> {db_path}", flush=True)
                            try:
                                _ingest_pickles_to_db(batch, db_path, yyyymm, checkpoint_interval)
                                # remove ingested ones from buffer
                                pickle_paths = pickle_paths[checkpoint_interval:]
                                print(f"  {yyyymm}: 증분 반영 완료 (누적 대기 {len(pickle_paths)} pickle)", flush=True)
                            except Exception as e:
                                print(f"  {yyyymm}: 증분 반영 실패: {type(e).__name__}: {e}", flush=True)
                except Exception as e:
                    print(f"  {yyyymm}: 그룹 처리 중 오류: {type(e).__name__}: {e}", flush=True)
                finally:
                    done += 1
                    if done % 5 == 0 or done == total_groups:
                        print(f"  {yyyymm}: pickle 생성 진행률 [{done}/{total_groups}]", flush=True)
    else:
        print(f"  {yyyymm}: 순차 그룹 처리", flush=True)
        done = 0
        for group_key, group_info in month_groups:
            pkl_path = _process_single_group_to_pickle(group_key, group_info, tmp_root, time_start, time_end)
            if pkl_path:
                pickle_paths.append(pkl_path)
                # Incremental ingest when buffer reaches checkpoint size
                if checkpoint_interval > 0 and len(pickle_paths) >= checkpoint_interval:
                    batch = pickle_paths[:checkpoint_interval]
                    print(f"  {yyyymm}: 증분 반영 시작 (batch={len(batch)}) -> {db_path}", flush=True)
                    try:
                        _ingest_pickles_to_db(batch, db_path, yyyymm, checkpoint_interval)
                        pickle_paths = pickle_paths[checkpoint_interval:]
                        print(f"  {yyyymm}: 증분 반영 완료 (누적 대기 {len(pickle_paths)} pickle)", flush=True)
                    except Exception as e:
                        print(f"  {yyyymm}: 증분 반영 실패: {type(e).__name__}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == total_groups:
                print(f"  {yyyymm}: pickle 생성 진행률 [{done}/{total_groups}]", flush=True)
     
    # Step 2: Ingest pickle files to DB sequentially (to avoid DB write conflicts)
    if pickle_paths:
        print(f"  {yyyymm}: {len(pickle_paths)} pickle 파일을 DB에 순차 반영", flush=True)
        processed_count = _ingest_pickles_to_db(pickle_paths, db_path, yyyymm, checkpoint_interval)
    else:
        processed_count = 0
     
    return processed_count


def _checkpoint_db_once(db_path: str) -> Tuple[str, bool, str]:
    """Run DuckDB CHECKPOINT once for the given DB file. Returns (path, ok, msg)."""
    try:
        conn = duckdb.connect(db_path)
        try:
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
        return db_path, True, ""
    except Exception as e:
        return db_path, False, f"{type(e).__name__}: {e}"


def _parallel_checkpoint_months(base_db_path: str, months: List[str], workers: int) -> None:
    """Run CHECKPOINT across the given months' DB shards in parallel using up to `workers` processes."""
    if not months:
        return
    # Build existing paths only
    month_paths = []
    for m in months:
        p = _monthly_db_path(base_db_path, m)
        if os.path.exists(p):
            month_paths.append(p)
    if not month_paths:
        return
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(month_paths))
    
    print(f"최종 체크포인트 실행: {len(month_paths)}개 DB 파일")
    
    if used_workers > 1:
        print(f"  병렬 실행 (workers={used_workers})")
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {ex.submit(_checkpoint_db_once, path): path for path in month_paths}
            completed = 0
            for fut in _fut.as_completed(futs):
                path = futs[fut]
                try:
                    _, ok, msg = fut.result()
                    completed += 1
                    if ok:
                        print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 완료: {os.path.basename(path)}")
                    else:
                        print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {msg}")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {type(e).__name__}: {e}")
    else:
        print("  순차 실행")
        for i, path in enumerate(month_paths, 1):
            _, ok, msg = _checkpoint_db_once(path)
            if ok:
                print(f"  [{i}/{len(month_paths)}] CHECKPOINT 완료: {os.path.basename(path)}")
            else:
                print(f"  [{i}/{len(month_paths)}] CHECKPOINT 실패: {os.path.basename(path)} -> {msg}")


def _sweep_and_ingest_tmp(base_db_path: str, tmp_root: Path, workers: int = 1, checkpoint_interval: int = 20):
    """Scan tmp_root for any leftover .pkl files and ingest them in the current process.
    This supports resume-on-start and graceful Ctrl+C handling.
    """
    if not tmp_root.exists():
        return
    pkls = sorted(p for p in tmp_root.glob("*.pkl"))
    if not pkls:
        return
    print(f"임시 체크포인트 {len(pkls)}개를 DB에 반영합니다 (resume/cleanup)...")
    prepared: list[tuple[str, str, str, str]] = []
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(pkls))
    if used_workers > 1:
        print(f"- 메타데이터 병렬 준비: workers={used_workers}")
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = [ex.submit(_prep_pkl_metadata, str(p)) for p in pkls]
            done = 0
            for fut in _fut.as_completed(futs):
                try:
                    res = fut.result()
                    if res is not None:
                        prepared.append(res)
                    else:
                        # remove unknown naming
                        pass
                finally:
                    done += 1
    else:
        print("- 메타데이터 직렬 준비 (workers=1)")
        for p in pkls:
            res = _prep_pkl_metadata(str(p))
            if res is not None:
                prepared.append(res)
            else:
                try:
                    os.remove(p)
                except Exception:
                    pass

    print("- DuckDB 반영: 순차 처리 (쓰기 경합 방지)")
    month_counts: Dict[str, int] = {}
    for i, (group_key, code, date, pkl_path) in enumerate(prepared, 1):
        try:
            yyyymm = _month_key_from_yyyymmdd(date)
            db_path = _monthly_db_path(base_db_path, yyyymm)
            _ingest_pickle_into_db_path(db_path, pkl_path, code, date, group_key)
            # per-month checkpoint interval
            if checkpoint_interval > 0:
                month_counts[yyyymm] = month_counts.get(yyyymm, 0) + 1
                if month_counts[yyyymm] % checkpoint_interval == 0:
                    try:
                        conn_ck = duckdb.connect(db_path)
                        try:
                            conn_ck.execute("CHECKPOINT")
                        finally:
                            conn_ck.close()
                    except Exception:
                        pass
        finally:
            # Always try to remove the pickle to free disk space
            try:
                os.remove(pkl_path)
            except Exception:
                pass


def normalize_datasets(input_db: str, output_db: str, *,
                       skip_existing: bool = True,
                       compact_only: bool = False,
                       force_recreate: bool = False,
                       workers: int = 1,
                       group_workers: int = 1,
                       tmp_dir: Optional[str] = None,
                       checkpoint_interval: int = 20,
                       time_start: int = 90000000,
                       time_end: int = 110000000):
    """
    메인 정규화 함수 (DuckDB 전용)
    - 입력 DuckDB를 스캔하여 유효 데이터 그룹을 찾음
    - 그룹을 날짜(YYYYMMDD)에서 월(YYYYMM)로 묶어 월별 DuckDB 샤드에 기록
    - 최대 `workers`개의 월을 병렬로 처리
    - 각 월 내부에서는 최대 `group_workers`개의 그룹을 병렬 처리
    - `checkpoint-interval`마다 CHECKPOINT 실행
    - 작업 중단 복구를 위해 temp 디렉토리에 단계별 체크포인트(.pkl)를 사용하고 시작 시 반영
    """
    # compact-only 모드: 입력 DB를 읽지 않고 지정한 DB에 대해 최적화만 수행
    if compact_only:
        print("compact-only 모드: 입력 DB 처리 없이 출력 DB 최적화만 수행합니다.")
        p = Path(output_db)
        stem = p.stem
        suffix = p.suffix or ".duckdb"
        parent = p.parent
        month_files = sorted(parent.glob(f"{stem}_*{suffix}"))
        months: List[str] = []
        for f in month_files:
            m = f.stem.replace(f"{stem}_", "")
            if re.fullmatch(r"\d{6}", m):
                months.append(m)
        if not months and os.path.exists(output_db):
            _parallel_checkpoint_months(output_db, [], 1)
            try:
                conn = duckdb.connect(output_db)
                try:
                    conn.execute("CHECKPOINT")
                finally:
                    conn.close()
                print(f"DB 유지보수 완료: {output_db}")
            except Exception as e:
                print(f"경고: 단일 DB 체크포인트 실패: {type(e).__name__}: {e}")
            return
        _parallel_checkpoint_months(output_db, months, workers)
        print("월별 DB 유지보수 완료")
        return

    # 입력 DB 검증
    if not os.path.exists(input_db):
        print(f"입력 DuckDB 파일이 존재하지 않습니다: {input_db}")
        return

    # temp 디렉토리 준비 및 resume 처리
    p = Path(output_db)
    _tmp_base = Path(tmp_dir) if tmp_dir else p.with_suffix(p.suffix + ".tmp")
    _tmp_base.mkdir(parents=True, exist_ok=True)
    for part in _tmp_base.glob("*.pkl.part"):
        try:
            os.remove(part)
        except Exception:
            pass
    _sweep_and_ingest_tmp(output_db, _tmp_base, workers=group_workers, checkpoint_interval=checkpoint_interval)

    # DuckDB 그룹 스캔
    print("DuckDB 데이터 스캔 및 그룹화 중...")
    complete_groups = find_duckdb_groups(input_db)
    if not complete_groups:
        print("처리할 유효 데이터 그룹을 찾지 못했습니다.")
        return

    # 그룹을 월별로 묶기
    monthly_groups: Dict[str, List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]] = {}
    for group_key, group_info in complete_groups.items():
        parts = group_key.split("_")
        date = parts[-1]
        yyyymm = _month_key_from_yyyymmdd(date)
        monthly_groups.setdefault(yyyymm, []).append((group_key, group_info))

    # force-recreate: 대상 월 DB 삭제
    if force_recreate:
        for yyyymm in monthly_groups.keys():
            db_path = _monthly_db_path(output_db, yyyymm)
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                    print(f"삭제 후 재생성 예정: {db_path}")
                except Exception as e:
                    print(f"경고: DB 삭제 실패 {db_path}: {type(e).__name__}: {e}")

    # skip-existing: 각 월 DB에서 이미 존재하는 (종목코드, 날짜) 그룹 제거
    def _filter_skip_existing_for_month(yyyymm: str, groups: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]) -> List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]]:
        if not skip_existing:
            return groups
        db_path = _monthly_db_path(output_db, yyyymm)
        if not os.path.exists(db_path):
            return groups
        try:
            conn = duckdb.connect(db_path)
            try:
                try:
                    conn.execute("DESCRIBE datasets")
                    table_exists = True
                except Exception:
                    table_exists = False
                if not table_exists:
                    return groups
                keep: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]] = []
                for group_key, group_info in groups:
                    parts = group_key.split('_')
                    code = parts[0]
                    date = parts[-1]
                    try:
                        q = conn.execute("SELECT 1 FROM datasets WHERE \"종목코드\"=? AND \"날짜\"=? LIMIT 1", [code, date]).fetchone()
                    except Exception:
                        q = None
                    if q is None:
                        keep.append((group_key, group_info))
                return keep
            finally:
                conn.close()
        except Exception:
            return groups

    for yyyymm in list(monthly_groups.keys()):
        orig_n = len(monthly_groups[yyyymm])
        monthly_groups[yyyymm] = _filter_skip_existing_for_month(yyyymm, monthly_groups[yyyymm])
        if len(monthly_groups[yyyymm]) == 0:
            print(f"{yyyymm}: 스킵할 항목만 존재하여 건너뜁니다 (원래 {orig_n} 그룹)")
            del monthly_groups[yyyymm]

    if not monthly_groups:
        print("처리할 신규 그룹이 없습니다.")
        try:
            _shutil.rmtree(_tmp_base)
        except Exception:
            pass
        return

    # 월 목록 및 병렬 처리 설정
    months = sorted(monthly_groups.keys())
    max_workers = max(1, int(workers))
    used_workers = min(max_workers, len(months))

    print(f"월별 처리 시작: 대상 {len(months)}개월, 병렬 workers={used_workers}")

    # 병렬로 월별 처리 실행
    if used_workers > 1:
        with _fut.ProcessPoolExecutor(max_workers=used_workers) as ex:
            futs = {}
            for yyyymm in months:
                db_path = _monthly_db_path(output_db, yyyymm)
                groups = monthly_groups[yyyymm]
                fut = ex.submit(_process_monthly_groups, groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers), time_start, time_end)
                futs[fut] = (yyyymm, len(groups), db_path)
            done = 0
            total = len(futs)
            for fut in _fut.as_completed(futs):
                yyyymm, n_groups, db_path = futs[fut]
                try:
                    processed = fut.result()
                    print(f"월 처리 완료: {yyyymm} ({processed}/{n_groups}) -> {db_path}")
                except Exception as e:
                    print(f"경고: 월 처리 실패 {yyyymm}: {type(e).__name__}: {e}")
                finally:
                    done += 1
    else:
        for yyyymm in months:
            db_path = _monthly_db_path(output_db, yyyymm)
            groups = monthly_groups[yyyymm]
            processed = _process_monthly_groups(groups, db_path, str(_tmp_base), yyyymm, int(checkpoint_interval), int(group_workers), time_start, time_end)
            print(f"월 처리 완료: {yyyymm} ({processed}/{len(groups)}) -> {db_path}")

    # 처리된 월들에 대해 병렬 최종 CHECKPOINT 수행
    try:
        processed_months = months
        if processed_months:
            _parallel_checkpoint_months(output_db, processed_months, max_workers)
    except Exception:
        pass

    # 임시 디렉토리 정리
    try:
        _shutil.rmtree(_tmp_base)
    except Exception:
        pass

    print(f"정규화 완료: {output_db}")


def main():
    parser = argparse.ArgumentParser(description="DuckDB 데이터셋 정규화 스크립트")
    parser.add_argument("input_db", help="입력 DuckDB 파일 경로")
    parser.add_argument("-o", "--output", default="normalized.duckdb", help="출력 DuckDB 파일명")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                        help="이미 DB에 해당 (종목코드, 날짜) 그룹이 존재하면 스킵합니다 (기본: 활성화)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="이미 존재하는 그룹도 다시 처리합니다")
    parser.add_argument("--compact-only", action="store_true",
                        help="입력 DB 처리 없이 지정한 DuckDB에 대해 PRAGMA optimize/checkpoint만 수행합니다")
    parser.add_argument("--force-recreate", action="store_true",
                        help="출력 DuckDB 파일이 존재하면 삭제 후 새로 생성합니다 (손상/버전 문제 해결용)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="월별 병렬 처리에 사용할 프로세스 수 (기본: CPU 코어 수)")
    parser.add_argument("--group-workers", type=int, default=1,
                        help="각 월 내에서 그룹 병렬 처리에 사용할 프로세스 수 (기본: 1)")
    parser.add_argument("--tmp-dir", default=None,
                        help="임시 결과 저장 디렉토리 (기본: <output>.tmp)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="몇 개 그룹 처리마다 DuckDB CHECKPOINT를 실행할지 지정 (0이면 비활성화, 기본: 100)")
    parser.add_argument("--time-start", type=int, default=90000000,
                        help="시작 시간 (기본: 90000000 = 오전 9시)")
    parser.add_argument("--time-end", type=int, default=110000000,
                        help="종료 시간, 미포함 (기본: 110000000 = 오전 11시)")
     
    args = parser.parse_args()
     
    if not args.compact_only:
        if not os.path.exists(args.input_db):
            print(f"입력 DuckDB 파일이 존재하지 않습니다: {args.input_db}")
            return
     
    normalize_datasets(
        args.input_db,
        args.output,
        skip_existing=args.skip_existing,
        compact_only=args.compact_only,
        force_recreate=args.force_recreate,
        workers=args.workers,
        group_workers=args.group_workers,
        tmp_dir=args.tmp_dir,
        checkpoint_interval=args.checkpoint_interval,
        time_start=args.time_start,
        time_end=args.time_end,
    )


if __name__ == "__main__":
    main()
