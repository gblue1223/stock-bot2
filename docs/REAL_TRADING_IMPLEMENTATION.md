# 실전 거래 시스템 구현 문서

## 개요

EnhancedGRPOInference와 Koapys REST API를 통합한 실시간 자동 매매 시스템

**구현 날짜**: 2025-10-17  
**버전**: 1.0.0  
**상태**: 구현 완료 (특징 추출 제외)

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                  Real Trading System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Koapys     │───▶│  Data Buffer │───▶│   Enhanced   │  │
│  │ REST Client  │    │  (Sequence)  │    │  Inference   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                         │          │
│         │                                         ▼          │
│         │                                  ┌──────────────┐  │
│         │                                  │   Position   │  │
│         │                                  │  Management  │  │
│         │                                  └──────────────┘  │
│         │                                         │          │
│         ▼                                         ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Order Execution                          │  │
│  │  - Market Orders (IOC)                                │  │
│  │  - Auto Stop Loss/Take Profit                         │  │
│  │  - Risk Management                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 구현된 컴포넌트

### 1. 핵심 스크립트

#### 1.1 live_trading.py
실시간 자동 매매 메인 시스템

**주요 클래스:**
- `TradingConfig`: 거래 설정 관리
- `MarketDataBuffer`: 시계열 데이터 버퍼 (deque 기반)
- `LiveTrader`: 거래 시스템 메인 클래스

**주요 기능:**
- 실시간 시장 데이터 수집
- GRPO 모델 기반 매매 신호 생성
- 자동 손절/익절 (Enhanced Inference)
- 포지션 및 리스크 관리
- 거래 로깅 및 통계

**실행 예시:**
```bash
python scripts/real/live_trading.py \
    --config config/trading_config.json \
    --stocks 005930 000660
```

#### 1.2 monitor_trading.py
실시간 거래 모니터링

**기능:**
- 로그 파일 tail 모니터링
- 매매 신호 및 체결 추적
- 실시간 통계 표시

**실행 예시:**
```bash
python scripts/real/monitor_trading.py --log logs/live_trading.log
```

#### 1.3 backtest_strategy.py
실전 전략 백테스팅

**기능:**
- Enhanced Inference 전략 검증
- 과거 데이터 기반 성과 분석
- 파라미터 최적화 지원

**실행 예시:**
```bash
python scripts/real/backtest_strategy.py \
    --model models/grpo_direct_features@20251016/direct_features_model.pt \
    --db datasets_raw_all.duckdb \
    --episodes 100
```

### 2. 설정 파일

#### 2.1 config/trading_config.json
거래 시스템 설정

**주요 파라미터:**
```json
{
  "model_path": "models/grpo_direct_features@20251016/direct_features_model.pt",
  "target_stocks": ["005930", "000660"],
  "max_position_size": 1000000,
  "max_positions": 3,
  "min_buy_confidence": 0.0,
  "stop_loss_rate": -2.0,
  "take_profit_rate": 5.0,
  "max_holding_period": 100,
  "trading_start": "09:00",
  "trading_end": "11:00",
  "simulation": true
}
```

#### 2.2 .env
환경 변수

```bash
KOAPYS_BASE_URL=http://localhost:5000
KOAPYS_API_KEY=your_api_key_here
SIMULATION=true
```

### 3. 문서

- `scripts/real/README.md`: 상세 문서
- `scripts/real/QUICKSTART.md`: 빠른 시작 가이드
- `docs/REAL_TRADING_IMPLEMENTATION.md`: 이 문서

## Enhanced Inference 통합

### 검증 결과 반영

`validate_enhanced.py` 검증 결과를 바탕으로 최적 설정 적용:

**최적 파라미터:**
- `min_buy_confidence`: 0.0 (필터링 없음)
- `stop_loss_rate`: -2.0%
- `take_profit_rate`: 5.0%
- `max_holding_period`: 100

**성과:**
- Mean Reward: 299.39
- Win Rate: 75%
- Buy Filter Rate: 88.96% (포지션 관리로 자동 필터링)

### 주요 특징

1. **신뢰도 필터링 없음**
   - 검증 결과: 필터링 없이 자동 청산만 사용하는 것이 최고 성과
   - 포지션 관리로 중복 진입 자동 방지

2. **자동 손절/익절**
   - Stop Loss: -2.0%
   - Take Profit: 5.0%
   - 최대 보유 기간: 100 틱

3. **포지션 관리**
   - 종목당 1개 포지션만 허용
   - 최대 동시 포지션 수 제한
   - 리스크 제어

## Koapys API 통합

### REST API 클라이언트

`koapys.client.KoapyRestSimple` 사용:

**주요 메서드:**
- `get_account_list()`: 계좌 목록 조회
- `get_stock_basic_info(code)`: 종목 정보 조회
- `get_current_price(code)`: 현재 가격 조회
- `send_order()`: 주문 실행
- `send_order_cancel()`: 주문 취소

**주문 타입:**
- `OrderType.BUY`: 매수
- `OrderType.SELL`: 매도
- `OrderBookType.MARKET_IOC`: 시장가 IOC (즉시 체결)

### 시뮬레이션 모드

실전 거래 전 테스트를 위한 시뮬레이션 모드 지원:

```python
koapys = KoapyRestSimple(
    base_url='http://localhost:5000',
    simulation=True  # 시뮬레이션 모드
)
```

## 데이터 파이프라인

### 1. 시장 데이터 수집

```python
# Koapys에서 시장 데이터 조회
market_data = koapys.get_stock_basic_info(code)

# 특징 추출 (TODO: 구현 필요)
features = extract_features(market_data)

# 버퍼에 추가
data_buffer.add(features)
```

### 2. 시퀀스 생성

```python
# 시퀀스 준비 확인
if data_buffer.is_ready():
    # 시퀀스 가져오기 (seq_len, num_features)
    sequence = data_buffer.get_sequence()
```

### 3. 추론 및 행동

```python
# Enhanced Inference 예측
action, confidence, pred_info = inference.predict(
    sequence=sequence,
    current_position=current_position,
    current_price=current_price,
    deterministic=True
)

# 행동 실행
if action == Action.BUY.value:
    execute_buy(code, confidence)
elif action == Action.SELL.value:
    execute_sell(code, confidence, pred_info['exit_reason'])
```

## 리스크 관리

### 1. 포지션 크기 제한

```python
# 종목당 최대 투자 금액
max_position_size = 1000000  # 100만원

# 매수 수량 계산
quantity = int(max_position_size / current_price)
```

### 2. 동시 포지션 수 제한

```python
# 최대 동시 보유 종목 수
max_positions = 3

# 포지션 수 확인
if len(positions) >= max_positions:
    return  # 추가 매수 불가
```

### 3. 자동 손절/익절

```python
# Enhanced Inference가 자동 처리
if position.profit_rate <= stop_loss_rate:
    return Action.SELL, "Stop loss"

if position.profit_rate >= take_profit_rate:
    return Action.SELL, "Take profit"
```

### 4. 거래 시간 제한

```python
# 스캘핑은 오전 장만 거래
trading_start = "09:00"
trading_end = "11:00"

if not is_trading_time():
    return  # 거래 불가
```

## 로깅 및 모니터링

### 1. 로그 파일

**logs/live_trading.log:**
- 시스템 이벤트
- 매매 신호 및 체결
- 오류 및 경고

**logs/orders_YYYYMMDD_HHMMSS.json:**
- 주문 기록 (종료 시 저장)
- 체결 가격 및 수량
- 수익률 및 보유 기간

### 2. 실시간 통계

```python
# 1분마다 통계 출력
stats = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_profit': 0.0,
    'active_positions': len(positions)
}
```

### 3. 추론 엔진 통계

```python
inference_stats = inference.get_stats()
# - total_predictions
# - buy_signal_rate
# - buy_filter_rate
# - auto_exit_rate
```

## 성능 목표

- **추론 속도**: < 10ms per prediction
- **데이터 지연**: < 1초
- **주문 체결**: < 100ms
- **시스템 가동률**: > 99%

## 구현 완료 항목

- [x] Koapys REST API 클라이언트 통합
- [x] Enhanced Inference 통합
- [x] 실시간 데이터 버퍼
- [x] 포지션 관리
- [x] 자동 손절/익절
- [x] 리스크 관리
- [x] 거래 로깅
- [x] 실시간 모니터링
- [x] 백테스팅 스크립트
- [x] 설정 파일 시스템
- [x] 시뮬레이션 모드
- [x] 문서 작성

## 미완성 항목

### 1. 특징 추출 구현 (중요!)

현재 `extract_features()` 메서드는 더미 데이터를 반환합니다.

**구현 필요:**
```python
def extract_features(self, market_data: Dict) -> np.ndarray:
    """
    시장 데이터에서 24개 특징 추출
    
    필요한 특징:
    1. 가격 변동률 (현재가, 시가, 고가, 저가)
    2. 거래량 변화
    3. 호가 정보 (매수/매도 호가)
    4. 기술적 지표 (이동평균, RSI 등)
    5. 시장 상태 (상한가 근접도 등)
    
    Returns:
        features: (24,) numpy array
    """
    # TODO: 실제 특징 추출 로직 구현
    pass
```

**참고:**
- 학습 데이터와 동일한 특징 사용
- 정규화 필요 (학습 시 사용한 통계)
- 실시간 계산 가능해야 함

### 2. 실시간 데이터 스트리밍

현재는 폴링 방식 (1초마다 조회):

**개선 방안:**
- WebSocket 기반 실시간 스트리밍
- 이벤트 기반 처리
- 지연 시간 최소화

### 3. 웹 대시보드

실시간 모니터링 UI:

**기능:**
- 실시간 차트
- 포지션 현황
- 수익률 그래프
- 알림 시스템

### 4. 알림 시스템

중요 이벤트 알림:

**채널:**
- Slack
- Email
- SMS

**이벤트:**
- 거래 체결
- 손절/익절
- 시스템 오류

## 테스트 계획

### 1. 단위 테스트

```bash
# 데이터 버퍼 테스트
pytest tests/test_market_data_buffer.py

# 포지션 관리 테스트
pytest tests/test_position_management.py
```

### 2. 통합 테스트

```bash
# 시뮬레이션 모드 통합 테스트
python scripts/real/live_trading.py --simulation --stocks 005930
```

### 3. 백테스팅

```bash
# 전략 검증
python scripts/real/backtest_strategy.py --episodes 100
```

### 4. 실전 테스트

1. 소액 테스트 (10만원)
2. 단일 종목 테스트
3. 오전 장만 테스트
4. 점진적 확대

## 배포 가이드

### 1. 환경 준비

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# 로그 디렉토리 생성
mkdir -p logs
```

### 2. Koapys 서버 실행

```bash
cd koapys
python server.py
```

### 3. 백테스팅

```bash
python scripts/real/backtest_strategy.py --episodes 100
```

### 4. 시뮬레이션 테스트

```bash
python scripts/real/live_trading.py --simulation --stocks 005930
```

### 5. 실전 거래

```bash
# .env에서 SIMULATION=false로 변경
python scripts/real/live_trading.py --config config/trading_config.json
```

## 유지보수

### 1. 일일 점검

- [ ] 로그 파일 확인
- [ ] 거래 기록 검토
- [ ] 성과 분석
- [ ] 시스템 상태 확인

### 2. 주간 점검

- [ ] 모델 성능 평가
- [ ] 파라미터 최적화
- [ ] 백테스팅 재실행
- [ ] 리스크 관리 검토

### 3. 월간 점검

- [ ] 전체 성과 분석
- [ ] 모델 재학습 검토
- [ ] 시스템 업그레이드
- [ ] 문서 업데이트

## 참고 자료

- [Enhanced Inference 검증](../scripts/grpo/direct/validate_enhanced.py)
- [GRPO 추론 엔진](../ai_trader/grpo/inference/infer_grpo.py)
- [Koapys API](../koapys/README.md)
- [GRPO 문서](GRPO_*.md)

## 변경 이력

### v1.0.0 (2025-10-17)
- 초기 구현 완료
- Enhanced Inference 통합
- Koapys API 통합
- 백테스팅 스크립트
- 모니터링 스크립트
- 문서 작성

## 라이선스

프로젝트 라이선스 참조

## 기여

이슈 및 풀 리퀘스트 환영
