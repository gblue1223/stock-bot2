# Koapys Test Suite

실제 Kiwoom REST API 연결 및 기능 테스트를 위한 코드입니다.

## 사전 준비

환경 변수를 설정해야 합니다:

```bash
# Linux/macOS/Windows(git bash)
export KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
export KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"

# Windows (CMD)
set KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
set KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"

# Windows (PowerShell)
$env:KIWOOM_API_KEY="YOUR_ACTUAL_API_KEY"
$env:KIWOOM_API_SECRET="YOUR_ACTUAL_API_SECRET"
```

## 테스트 실행

**중요: 프로젝트 루트 디렉토리에서 실행하세요**

### 전체 테스트 실행

```bash
# 프로젝트 루트에서 실행
cd d:\Workspace\Project\stock-bot\stock-bot2
python koapys/test/test_real_connection.py
```

### 개별 테스트 실행

Python 스크립트로 직접 실행:

```python
from koapys.test.test_real_connection import test_connection, test_account_info, test_stock_info

# 연결 테스트만
test_connection()

# 계좌 정보 테스트만
test_account_info()

# 주식 정보 테스트만
test_stock_info()
```

## 테스트 내용

### 1. Connection Test (`test_connection`)
- API 키 확인
- 클라이언트 초기화
- 연결 상태 확인

### 2. Account Information Test (`test_account_info`)
- 계좌 목록 조회 (`get_account_list`)
- 예수금 조회 (`get_deposit`)
- 잔고 조회 (`get_equity_balance`)
- 상세 잔고 조회 (`get_equity_balance2`)

### 3. Stock Information Test (`test_stock_info`)
- 주식 기본 정보 조회 (`get_stock_basic_info`)
- 현재가 조회 (`get_current_price`)
- 상한가 조회 (`get_upper_limit_price`)

## 예상 출력

```
============================================================
Koapys Real API Connection Test Suite
============================================================
============================================================
Testing connection to Kiwoom REST API...
============================================================
✓ API Key: YOUR_KEY...
✓ API Secret: YOUR_SECRET...
✓ Client initialized
✓ Connection successful!
  - Connected: True
  - Simulation Mode: False

============================================================
Testing account information retrieval...
============================================================

[Test 1] Getting account list...
✓ Accounts: ['1234567890']

Using account: 1234567890

[Test 2] Getting deposit information...
✓ Deposit information retrieved:
  - 예수금: 10000000
  - 출금가능금액: 10000000
  ...

[Test 3] Getting equity balance...
✓ Equity balance retrieved:
  Total:
    - 총매입금액: 5000000
    - 총평가금액: 5500000
    ...

============================================================
TEST SUMMARY
============================================================
connection: ✅ PASSED
account_info: ✅ PASSED
stock_info: ✅ PASSED

============================================================
✅ All tests PASSED!
============================================================
```

## 주의사항

- **실제 API를 호출**하므로 API 사용량에 주의하세요.
- 일부 API가 아직 구현되지 않은 경우 경고 메시지가 출력됩니다.
- 시장 시간 외에는 일부 데이터를 조회할 수 없을 수 있습니다.
- API 키와 시크릿은 절대 공개 저장소에 커밋하지 마세요.

## 트러블슈팅

### "ModuleNotFoundError: No module named 'httpx'"
```bash
pip install httpx>=0.28.0
```

### "Connection failed: Unable to get access token"
- API 키와 시크릿이 올바른지 확인하세요.
- 환경 변수가 제대로 설정되었는지 확인하세요.
- Kiwoom API 서버가 정상 작동 중인지 확인하세요.

### API 응답 에러
- Kiwoom API 문서를 참조하여 올바른 파라미터를 사용하는지 확인하세요.
- API 사용량 제한에 도달했는지 확인하세요.
