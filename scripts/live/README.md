# 실전 거래 스크립트

EnhancedGRPOInference와 Koapys API를 사용한 실시간 자동 매매 시스템

## 구조

```
scripts/real/
├── live_trading.py      # 실전 거래 메인 스크립트
├── monitor_trading.py   # 거래 모니터링 스크립트
└── README.md           # 이 파일
```

## 주요 기능

### 1. live_trading.py

실시간 자동 매매 시스템

**기능:**
- 실시간 시장 데이터 수집
- GRPO 모델 기반 매매 신호 생성
- 자동 손절/익절 관리 (Enhanced Inference)
- 포지션 및 리스크 관리
- 거래 로깅 및 통계

**사용법:**
```bash
# 기본 실행 (설정 파일 사용)
python scripts/real/live_trading.py --config config/trading_config.json

# 특정 종목 지정
python scripts/real/live_trading.py --stocks 005930 000660

# 시뮬레이션 모드
python scripts/real/live_trading.py --simulation --stocks 005930
```

**설정 파일 (config/trading_config.json):**
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

### 2. monitor_trading.py

실시간 거래 모니터링

**기능:**
- 로그 파일 실시간 모니터링
- 매매 신호 및 체결 추적
- 통계 및 성과 표시

**사용법:**
```bash
# 기본 실행
python scripts/real/monitor_trading.py

# 특정 로그 파일 모니터링
python scripts/real/monitor_trading.py --log logs/live_trading.log
```

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Live Trading System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Koapys     │───▶│  Data Buffer │───▶│   Enhanced   │  │
│  │   Client     │    │  (Sequence)  │    │  Inference   │  │
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
│  │  - Buy/Sell Orders                                    │  │
│  │  - Auto Stop Loss/Take Profit                         │  │
│  │  - Risk Management                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Enhanced Inference 특징

검증 결과를 반영한 최적 설정:

1. **신뢰도 필터링 없음 (min_buy_confidence=0.0)**
   - 검증 결과: 필터링 없이 자동 청산만 사용하는 것이 최고 성과
   - Mean Reward: 299.39 (필터링 사용 시 217.21)
   - 포지션 관리로 이미 88.96% 자동 필터링됨

2. **자동 손절/익절**
   - Stop Loss: -2.0% (기본값)
   - Take Profit: 5.0% (기본값)
   - 75% 승률 달성

3. **포지션 관리**
   - 중복 진입 방지
   - 최대 보유 기간 제한
   - 리스크 제어

## 환경 변수

`.env` 파일에 다음 변수 설정:

```bash
# Koapys API 설정
KOAPYS_BASE_URL=http://localhost:5000
KOAPYS_API_KEY=your_api_key_here

# 실행 모드
SIMULATION=true  # true: 시뮬레이션, false: 실전
```

## 로그 파일

- `logs/live_trading.log`: 거래 시스템 로그
- `logs/orders_YYYYMMDD_HHMMSS.json`: 주문 기록 (종료 시 저장)

## 주의사항

### 1. 실전 거래 전 필수 확인

- [ ] 시뮬레이션 모드에서 충분히 테스트
- [ ] 모델 성능 검증 완료
- [ ] 리스크 관리 파라미터 확인
- [ ] Koapys API 연결 테스트
- [ ] 계좌 잔고 및 권한 확인

### 2. 리스크 관리

- 최대 포지션 크기 제한 (`max_position_size`)
- 최대 동시 포지션 수 제한 (`max_positions`)
- 손절/익절 비율 설정
- 거래 시간 제한 (오전 장만 거래)

### 3. 모니터링

- 실시간 로그 모니터링 필수
- 통계 및 성과 주기적 확인
- 비정상 동작 시 즉시 중단

### 4. 특징 추출 구현 필요

현재 `extract_features()` 메서드는 더미 데이터를 반환합니다.
실제 사용 전에 다음을 구현해야 합니다:

```python
def extract_features(self, market_data: Dict) -> np.ndarray:
    """
    시장 데이터에서 24개 특징 추출
    
    필요한 특징:
    - 가격 변동률
    - 거래량 변화
    - 호가 정보
    - 기술적 지표 등
    """
    # TODO: 실제 특징 추출 로직 구현
    pass
```

## 성능 목표

- **추론 속도**: < 10ms per prediction
- **데이터 지연**: < 1초
- **주문 체결**: < 100ms

## 문제 해결

### Koapys 연결 실패

```bash
# Koapys 서버 상태 확인
curl http://localhost:5000/ping

# 로그 확인
tail -f logs/live_trading.log
```

### 모델 로드 실패

```bash
# 모델 파일 확인
ls -lh models/grpo_direct_features@20251016/

# 디바이스 확인 (CUDA)
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족

- 시퀀스 길이 줄이기 (`seq_len`)
- 캐시 크기 줄이기
- 동시 처리 종목 수 줄이기

## 향후 개선 사항

- [ ] 실시간 특징 추출 구현
- [ ] 다중 종목 병렬 처리
- [ ] 웹 대시보드 추가
- [ ] 백테스팅 결과 연동
- [ ] 알림 시스템 (Slack, Email)
- [ ] 성과 분석 리포트 자동 생성

## 참고 문서

- [Enhanced Inference 검증 결과](../../scripts/grpo/direct/validate_enhanced.py)
- [GRPO 추론 엔진](../../ai_trader/grpo/inference/infer_grpo.py)
- [Koapys API 문서](../../koapys/README.md)
