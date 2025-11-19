# 잔고 조회 API 수정

## 문제

```
2025-11-18 12:29:41 - ERROR - [SELL SKIP] 0015G0: Failed to verify position:
'KoapyRestSimple' object has no attribute 'get_balance'
```

## 원인

`KoapyRestSimple` 클래스에 `get_balance` 메서드가 없음.  
올바른 메서드는 `get_equity_balance`입니다.

## 수정 사항

### Before (오류)

```python
balance = self.koapys.get_balance(account_no=self.account_no)
holdings = balance.get('holdings', [])
actual_qty = 0
for holding in holdings:
    if holding.get('code') == code:
        actual_qty = holding.get('quantity', 0)
        break
```

### After (수정)

```python
balance = self.koapys.get_equity_balance(account_no=self.account_no)
stocks = balance.get('stocks', [])
actual_qty = 0

# 종목 코드 매칭 (A 접두사 제거)
code_clean = code[1:] if code.startswith('A') else code

for stock in stocks:
    stock_code = stock.get('종목코드', '')
    # A 접두사 제거하여 비교
    stock_code_clean = stock_code[1:] if stock_code.startswith('A') else stock_code

    if stock_code_clean == code_clean:
        # 보유수량은 문자열로 반환될 수 있음
        qty_str = stock.get('보유수량', '0')
        actual_qty = int(qty_str) if qty_str else 0
        break
```

## get_equity_balance 반환 형식

```python
{
    "total": {
        "계좌명": "...",
        "예수금": "...",
        "총매입금액": "...",
        "추정예탁자산": "...",
        # ... 기타 계좌 정보
    },
    "stocks": [
        {
            "종목코드": "A005930",  # A 접두사 포함
            "종목명": "삼성전자",
            "보유수량": "10",       # 문자열
            "평균단가": "70000",
            "현재가": "71000",
            "평가금액": "710000",
            "손익금액": "10000",
            "손익율": "1.43",
            # ... 기타 종목 정보
        },
        # ... 기타 보유 종목
    ]
}
```

## 주요 변경 사항

1. **메서드 이름**: `get_balance` → `get_equity_balance`
2. **키 이름**: `holdings` → `stocks`
3. **종목 코드**: `code` → `종목코드` (한글)
4. **수량**: `quantity` → `보유수량` (한글, 문자열)
5. **코드 매칭**: A 접두사 제거하여 비교

## 테스트

실전 거래 중 체결 확인 시 정상 작동 확인 필요.

---

**수정일**: 2025-11-18  
**상태**: ✅ 수정 완료
