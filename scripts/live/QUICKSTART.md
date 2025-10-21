# 실전 거래 빠른 시작 가이드

## 1. 환경 설정

### 1.1 의존성 설치

```bash
pip install -r requirements.txt
```

### 1.2 환경 변수 설정

`.env` 파일 생성:

```bash
# Koapys API 설정
KOAPYS_BASE_URL=http://localhost:5000
KOAPYS_API_KEY=your_api_key_here

# 실행 모드
SIMULATION=true  # 시뮬레이션 모드로 시작
```

### 1.3 Koapys 서버 실행

```bash
# Koapys REST API 서버 실행 (별도 터미널)
cd koapys
python server.py
```

## 2. 백테스팅 (필수)

실전 거래 전에 반드시 백테스팅으로 전략 검증:

```bash
# 기본 백테스팅 (100 에피소드)
python scripts/real/backtest_strategy.py \
    --model models/grpo_direct_features@20251016/direct_features_model.pt \
    --db datasets_raw_all.duckdb \
    --episodes 100

# 파라미터 조정 테스트
python scripts/real/backtest_strategy.py \
    --episodes 50 \
    --min-buy-confidence 0.0 \
    --stop-loss -2.0 \
    --take-profit 5.0
```

**기대 결과:**
- Mean Reward: > 200
- Win Rate: > 70%
- Auto Exits: 적절한 손절/익절 발생

## 3. 시뮬레이션 모드 테스트

실전 거래 전에 시뮬레이션 모드로 테스트:

```bash
# 시뮬레이션 모드 실행
python scripts/live/live_trading.py \
    --simulation \
    --config config/trading_config.json
```

**확인 사항:**
- [ ] Koapys 연결 성공
- [ ] 모델 로드 성공
- [ ] 시장 데이터 수집 정상
- [ ] 매매 신호 생성 정상
- [ ] 로그 기록 정상

## 4. 모니터링 설정

별도 터미널에서 모니터링 실행:

```bash
# 실시간 모니터링
python scripts/real/monitor_trading.py --log logs/live_trading.log
```

## 5. 실전 거래 (주의!)

### 5.1 사전 체크리스트

- [ ] 백테스팅 결과 만족스러움
- [ ] 시뮬레이션 모드 정상 동작 확인
- [ ] 리스크 관리 파라미터 설정 완료
- [ ] 계좌 잔고 및 권한 확인
- [ ] 모니터링 시스템 준비 완료
- [ ] 긴급 중단 방법 숙지

### 5.2 실전 거래 시작

```bash
# .env 파일에서 SIMULATION=false로 변경
# 또는 명령줄에서 직접 실행

python scripts/real/live_trading.py \
    --stocks 005930 \
    --config config/trading_config.json
```

### 5.3 실시간 모니터링

```bash
# 별도 터미널에서
python scripts/real/monitor_trading.py
```

## 6. 설정 파일 커스터마이징

`config/trading_config.json` 편집:

```json
{
  "target_stocks": ["005930", "000660"],
  "max_position_size": 1000000,
  "max_positions": 3,
  "min_buy_confidence": 0.0,
  "stop_loss_rate": -2.0,
  "take_profit_rate": 5.0,
  "trading_start": "09:00",
  "trading_end": "11:00"
}
```

**주요 파라미터:**

- `target_stocks`: 거래 대상 종목 (삼성전자: 005930, SK하이닉스: 000660)
- `max_position_size`: 종목당 최대 투자 금액 (원)
- `max_positions`: 최대 동시 보유 종목 수
- `min_buy_confidence`: Buy 신뢰도 임계값 (0.0 권장)
- `stop_loss_rate`: 손절 비율 (%)
- `take_profit_rate`: 익절 비율 (%)
- `trading_start/end`: 거래 시간 (스캘핑은 오전만 권장)

## 7. 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/live_trading.log

# 주문 기록 확인
cat logs/orders_YYYYMMDD_HHMMSS.json
```

## 8. 긴급 중단

거래 시스템 중단:

```bash
# Ctrl+C 입력
# 또는 프로세스 종료
pkill -f live_trading.py
```

**자동 처리:**
- 모든 열린 포지션 자동 청산
- 주문 기록 저장
- 통계 출력

## 9. 문제 해결

### Koapys 연결 실패

```bash
# Koapys 서버 상태 확인
curl http://localhost:5000/ping

# 서버 재시작
cd koapys
python server.py
```

### 모델 로드 실패

```bash
# 모델 파일 확인
ls -lh models/grpo_direct_features@20251016/

# CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 특징 추출 오류

현재 `extract_features()` 메서드는 더미 데이터를 반환합니다.
실제 사용 전에 구현 필요:

```python
# scripts/real/live_trading.py의 extract_features() 메서드 수정
def extract_features(self, market_data: Dict) -> np.ndarray:
    # TODO: 실제 특징 추출 로직 구현
    # - 가격 변동률
    # - 거래량 변화
    # - 호가 정보
    # - 기술적 지표
    pass
```

## 10. 성과 분석

거래 종료 후:

```bash
# 주문 기록 분석
python scripts/analysis/analyze_orders.py logs/orders_*.json

# 백테스팅과 비교
python scripts/real/backtest_strategy.py --episodes 100
```

## 11. 안전 수칙

1. **소액으로 시작**: 처음에는 최소 금액으로 테스트
2. **실시간 모니터링**: 거래 중 항상 모니터링
3. **손절 준수**: 자동 손절 설정 필수
4. **거래 시간 제한**: 오전 장만 거래 (09:00-11:00)
5. **최대 포지션 제한**: 동시 보유 종목 수 제한
6. **정기 점검**: 매일 성과 및 로그 확인

## 12. 추가 리소스

- [상세 문서](README.md)
- [Enhanced Inference 검증](../../scripts/grpo/direct/validate_enhanced.py)
- [GRPO 문서](../../docs/GRPO_*.md)
- [Koapys API](../../koapys/README.md)

## 지원

문제가 발생하면:
1. 로그 파일 확인 (`logs/live_trading.log`)
2. 시뮬레이션 모드로 재현
3. 백테스팅으로 검증
4. GitHub Issues에 보고
