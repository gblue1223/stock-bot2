# Koapys Client Migration Notes

## 개요
`koapys/client.py`가 `kiwoom_rest_api` 라이브러리를 내부적으로 사용하도록 리팩토링되었습니다.
**인터페이스는 그대로 유지**되므로 기존 코드 변경 없이 사용할 수 있습니다.

## 주요 변경사항

### 1. 내부 구현 변경
- 기존: `requests` 라이브러리를 직접 사용한 REST API 호출
- 변경: `kiwoom_rest_api` 라이브러리의 클래스들 사용
  - `TokenManager`: 인증 토큰 관리
  - `StockInfo`: 주식 정보 조회
  - `Account`: 계좌 정보 조회
  - `Order`: 주문 처리

### 2. 응답 데이터 한글화
- 모든 API 응답의 Element 이름을 한글명으로 자동 변환
- 기존 OpenAPI와 동일한 키 이름 사용으로 호환성 유지
- 변환 대상 메서드:
  - `get_deposit()`: 70+ 필드 변환
  - `get_equity_balance()`: total 및 stocks 배열 변환
  - `get_equity_balance2()`: 종목별체결잔고 포함 변환
  - `get_stock_basic_info()`: 주식 기본 정보 변환
- 예시:
  - `entr` → `예수금`
  - `cur_prc` → `현재가`
  - `stk_nm` → `종목명`
  - `pymn_alow_amt` → `출금가능금액`
  - `tot_pur_amt` → `총매입금액`

### 3. 생성자 파라미터
```python
KoapyRestSimple(
    timeout: int = 10,
    simulation: bool = False,
    logger: Optional[logging.Logger] = None,
)
```

### 4. API 매핑

#### 계좌 관련
- `get_deposit()` → `kt00001` (deposit_detail_status_request)
- `get_equity_balance()` → `kt00004` (account_evaluation_status_request)
- `get_equity_balance2()` → `kt00005` (filled_position_request)
- `get_account_list()` → 아직 미구현 (경고 로그 출력)

#### 시장 데이터
- `get_stock_basic_info()` → `ka10001` (basic_stock_information_request)
- `get_current_price()` → `ka10001`을 통해 조회
- `get_upper_limit_price()` → `ka10001`을 통해 조회
- `get_stocks_of_upper_limit_reached()` → 아직 미구현 (경고 로그 출력)

#### 주문
- `send_order()` 
  - 매수: `kt10000` (stock_buy_order_request)
  - 매도: `kt10001` (stock_sell_order_request)
- `send_order_modify()` → `kt10002` (stock_modify_order_request)
- `send_order_cancel()` → `kt10003` (stock_cancel_order_request)

#### 조건검색
- `get_condition_load()` → `ka10171` (condition_list_request) ✅ **지원**
- `get_condition_name_list()` → `ka10171` (condition_list_request) ✅ **지원**
- `get_stocks_by_condition()` → `ka10173` (condition_search_realtime_request) ✅ **지원**

### 5. 호가 타입 매핑
`OrderBookType` enum이 Kiwoom REST API의 `trde_tp` 파라미터로 자동 매핑됩니다:
- `LIMIT` → "0" (지정가)
- `MARKET` → "3" (시장가)
- `BEST_LIMIT` → "6" (최유리지정가)
- 등등...

## 알려진 제약사항

### 1. 미구현 API
- `get_account_list()`: 계좌 목록 조회가 아직 구현되지 않음
- `get_stocks_of_upper_limit_reached()`: 상한가 도달 종목 조회가 아직 구현되지 않음

**참고:**
- 조건검색 기능은 `ka10172`, `ka10173` API를 통해 지원됩니다
- HTS에서 미리 등록한 조건식을 사용할 수 있습니다

### 2. 주문 수정/취소 시 종목코드
`send_order_modify()`와 `send_order_cancel()`에서 종목코드(`stk_cd`)를 빈 문자열로 전달합니다.
실제 API가 종목코드를 필수로 요구한다면, 주문 정보를 따로 저장해서 사용해야 할 수 있습니다.

## 사용 예시

### 기본 사용법
기존 코드는 그대로 사용 가능합니다:

```python
from koapys import KoapyRestSimple

# 초기화
client = KoapyRestSimple(
    simulation=False
)

# 기존 인터페이스 그대로 사용
client.ensure_connected()
balance = client.get_equity_balance(account_no="1234567890")
price = client.get_current_price(code="005930")
```

### 조건검색 사용 예시
```python
from koapys import KoapyRestSimple

client = KoapyRestSimple(simulation=False)
client.ensure_connected()

# 1. 조건식 목록 로드
if client.get_condition_load():
    print("조건식 목록 로드 성공")

# 2. 조건식 목록 조회
conditions = client.get_condition_name_list()
print("등록된 조건식:", conditions)
# 예: [('000', '상승추세종목'), ('001', '거래량급증'), ...]

# 3. 조건검색 실행
if conditions:
    condition_name = conditions[0][1]  # 첫 번째 조건식 이름
    stocks = client.get_stocks_by_condition(condition_name)
    print(f"'{condition_name}' 조건에 맞는 종목:", stocks)
    # 예: ['005930', '000660', '035420', ...]
```

## 환경 변수 설정

`kiwoom_rest_api`는 환경 변수를 사용하므로, 생성자에서 자동으로 설정됩니다:
- `KIWOOM_API_KEY`
- `KIWOOM_API_SECRET`

별도로 환경 변수를 설정할 수도 있습니다.

## 의존성

이 코드는 `koapys/kiwoom_rest_api/src` 디렉토리를 자동으로 Python 경로에 추가합니다.
별도의 패키지 설치는 필요하지 않습니다.
