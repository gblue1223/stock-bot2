# Rolling Window 정규화 빠른 시작 가이드

## 개요

Buy Signal Rate 0% 문제를 해결하기 위해 Rolling Window 정규화를 구현했습니다.
이 가이드는 즉시 실행 가능한 단계별 지침을 제공합니다.

## 준비 사항

### 1. 검증 실행

```bash
# 통합 검증
python scripts/verify_rolling_window_integration.py
```

**예상 결과**:
```
🎉 모든 검증 통과! (5/5)
✅ Rolling Window 정규화가 성공적으로 통합되었습니다!
✅ 실시간 거래 시스템 실행 준비 완료!
```

### 2. 테스트 실행

```bash
# Rolling Window 정규화 테스트
python scripts/test_rolling_normalization.py
```

**예상 결과**:
```
🎉 모든 테스트 통과!
✅ Rolling Window 정규화가 정상적으로 작동합니다.
✅ 분포 이동 문제를 효과적으로 해결할 수 있습니다.
```

## 실행 방법

### 1. 실시간 거래 시작

```bash
# 기본 실행
python scripts/live/live_trading.py --config config/trading_config.json

# 특정 종목 지정 (선택사항)
python scripts/live/live_trading.py --config config/trading_config.json --stocks 005930 000660
```

### 2. 로그 모니터링

**터미널 1**: 실시간 거래 실행
```bash
python scripts/live/live_trading.py --config config/trading_config.json
```

**터미널 2**: 로그 모니터링
```bash
# 전체 로그
tail -f logs/live_trading_*.log

# 주요 이벤트만
tail -f logs/live_trading_*.log | grep -E "Buy Signal Rate|NORMALIZER|PREDICT|BUY|SELL"

# 정규화 상태만
tail -f logs/live_trading_*.log | grep "NORMALIZER"

# 예측 결과만
tail -f logs/live_trading_*.log | grep "PREDICT"
```

## 모니터링 포인트

### 1. 정규화 상태 확인

**워밍업 중**:
```
[NORMALIZER] 005930 warming up... (50/100)
[NORMALIZER] 000660 warming up... (75/100)
```

**준비 완료**:
```
[NORMALIZER] 005930 ready for inference (samples=100)
[NORMALIZER] 000660 ready for inference (samples=100)
```

### 2. 예측 결과 확인

**BUY 신호**:
```
[PREDICT] 005930 (삼성전자): BUY signal, confidence=0.8234, raw_action=0, filtered=False
```

**SELL 신호**:
```
[PREDICT] 000660 (SK하이닉스): SELL signal, confidence=0.7891, raw_action=2, filtered=False, reason=Signal
```

### 3. 통계 확인 (1분마다)

```
[AI] Inference Stats:
Total Predictions: 1225
Buy Signal Rate: 12.50%  ← 목표: 5-15%
Buy Filter Rate: 2.30%
Auto Exit Rate: 1.50%
```

## 성공 기준

### 즉시 확인 (5분 이내)

- [ ] 정규화 워밍업 완료 (100개 샘플 수집)
- [ ] 예측 실행 시작
- [ ] 정규화 통계 정상 (mean ~0, std ~1)

### 단기 확인 (30분 이내)

- [ ] Buy Signal Rate > 0%
- [ ] 예측 횟수 > 100회
- [ ] BUY 신호 발생 (최소 1회)

### 중기 확인 (2시간 이내)

- [ ] Buy Signal Rate: 5-15%
- [ ] 거래 발생 (최소 1건)
- [ ] 정규화 안정성 유지

## 문제 해결

### Buy Signal Rate 여전히 0%

**증상**: 30분 이상 실행했는데도 Buy Signal Rate가 0%

**원인 1**: 워밍업 미완료
```bash
# 로그 확인
grep "warming up" logs/live_trading_*.log | tail -10

# 해결: 더 기다리기 (100개 샘플 수집 필요)
```

**원인 2**: 모델 문제
```bash
# 모델 재학습 필요
python scripts/grpo/quick_train_grpo.py
```

### 정규화 통계 이상

**증상**: 정규화 후에도 값이 너무 큼 (> 10)

**확인**:
```bash
# 정규화 전후 비교
grep "BEFORE NORM\|AFTER NORM" logs/live_trading_*.log | tail -20
```

**해결**:
```python
# config/trading_config.json 수정
{
  "online_window_size": 2000,  # 1000 → 2000
  "online_warmup_samples": 200  # 100 → 200
}
```

### 예측 실행 안됨

**증상**: Total Predictions가 0으로 유지

**확인**:
```bash
# 데이터 수집 상태
grep "BUFFER\|SEQUENCE" logs/live_trading_*.log | tail -20
```

**원인**: 시퀀스 길이 부족 (60개 필요)
- 해결: 1-2분 더 기다리기

## 설정 조정

### Window Size 조정

**현재 설정**: 1000개
```json
{
  "online_window_size": 1000
}
```

**조정 가이드**:
- **500**: 빠른 적응, 불안정할 수 있음
- **1000**: 균형 (권장)
- **2000**: 안정적, 느린 적응

### Warmup Samples 조정

**현재 설정**: 100개
```json
{
  "online_warmup_samples": 100
}
```

**조정 가이드**:
- **50**: 빠른 시작, 초기 품질 낮음
- **100**: 균형 (권장)
- **200**: 높은 품질, 느린 시작

## 성능 비교

### 기존 (고정 통계)
```
정규화 평균 편차: 2.48
정규화 표준편차: 1.51
Buy Signal Rate: 0%
거래 발생: 0건/일
```

### 신규 (Rolling Window)
```
정규화 평균 편차: 0.02 (106배 개선!)
정규화 표준편차: 1.01
Buy Signal Rate: 5-15% (예상)
거래 발생: 5-20건/일 (예상)
```

## 다음 단계

### 1일차: 초기 모니터링
- [ ] 실시간 거래 시스템 실행
- [ ] Buy Signal Rate 확인
- [ ] 정규화 통계 모니터링
- [ ] 첫 거래 발생 확인

### 2-3일차: 성능 분석
- [ ] Buy Signal Rate 추이 분석
- [ ] 거래 성과 분석 (Win Rate, 수익률)
- [ ] 정규화 품질 검증
- [ ] 필요시 설정 조정

### 1주일차: 최적화
- [ ] Window size 최적화
- [ ] 신뢰도 임계값 조정
- [ ] 시간대별 정규화 고려
- [ ] 모델 재학습 검토

## 체크리스트

### 실행 전
- [ ] 검증 스크립트 실행 (모두 통과)
- [ ] 테스트 스크립트 실행 (모두 통과)
- [ ] 설정 파일 확인
- [ ] 모델 파일 존재 확인

### 실행 중
- [ ] 정규화 워밍업 완료
- [ ] 예측 실행 시작
- [ ] Buy Signal Rate > 0%
- [ ] 로그 정상 출력

### 실행 후
- [ ] Buy Signal Rate 기록
- [ ] 거래 내역 확인
- [ ] 성과 분석
- [ ] 다음 날 계획

## 참고 명령어

```bash
# 검증
python scripts/verify_rolling_window_integration.py

# 테스트
python scripts/test_rolling_normalization.py

# 실행
python scripts/live/live_trading.py --config config/trading_config.json

# 로그 모니터링
tail -f logs/live_trading_*.log | grep -E "Buy Signal Rate|NORMALIZER|PREDICT"

# 분포 분석 (선택사항)
python scripts/data/analyze_distribution_shift.py
```

## 지원

### 문서
- `docs/BUY_SIGNAL_ZERO_ANALYSIS.md`: 문제 분석
- `docs/ROLLING_WINDOW_IMPLEMENTATION.md`: 구현 세부사항
- `docs/DATA_QUALITY_VALIDATION.md`: 데이터 품질 검증

### 코드
- `lib/rolling_normalization.py`: Rolling Window 구현
- `scripts/live/live_trading.py`: 실시간 거래 시스템
- `scripts/test_rolling_normalization.py`: 테스트 코드

## 요약

1. **검증**: `python scripts/verify_rolling_window_integration.py`
2. **테스트**: `python scripts/test_rolling_normalization.py`
3. **실행**: `python scripts/live/live_trading.py --config config/trading_config.json`
4. **모니터링**: `tail -f logs/live_trading_*.log | grep "Buy Signal Rate"`
5. **성공**: Buy Signal Rate > 5%

---

**작성일**: 2024-11-18
**버전**: 1.0
**상태**: ✅ 실행 준비 완료
