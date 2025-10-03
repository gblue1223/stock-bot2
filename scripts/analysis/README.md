# Training Analysis Tools

훈련 결과를 분석하기 위한 도구들입니다.

## 📊 training_analyzer.py

종합적인 훈련 분석 도구입니다. 모델 파일, 설정, 성능 지표, 체크포인트, TensorBoard 로그를 모두 분석합니다.

### 사용법

```bash
# 기본 분석
python scripts/analysis/training_analyzer.py models/supervised_sample_test

# 다른 모델 분석
python scripts/analysis/training_analyzer.py models/supervised_stable
```

### 분석 항목

1. **파일 분석**: 필수 파일들의 존재 여부 확인
2. **훈련 설정**: 하이퍼파라미터 및 데이터 설정 검토
3. **성능 지표**: 손실 값 및 성능 평가
4. **체크포인트**: 저장된 체크포인트 정보
5. **TensorBoard**: 로그 파일 상태 확인
6. **권장사항**: 다음 훈련을 위한 개선 제안

### 출력 예시

```
🔍 훈련 결과 종합 분석
============================================================
📁 파일 분석
============================================================
모델 디렉토리: models\supervised_sample_test
✅ Config 파일: True
✅ 모델 파일: True
✅ 체크포인트: True
✅ TensorBoard 로그: True
📊 모델 크기: 14.96 MB

============================================================
⚙️  훈련 설정
============================================================
데이터셋: 000100
타겟 날짜: 20241128
시퀀스 길이: 60
예측 구간: 20
타겟 컬럼: 현재가
보조 작업: direction3
피처 수: 62

============================================================
📈 성능 지표
============================================================
최고 검증 손실: 0.414762
✅ 양호한 성능 (< 0.5)
```

## 🎯 사용 시나리오

### 1. 훈련 완료 후 즉시 분석
```bash
python -m ai_trader.ml.train_supervised --out models/my_model ...
python scripts/analysis/training_analyzer.py models/my_model
```

### 2. 기존 모델 재평가
```bash
python scripts/analysis/training_analyzer.py models/old_model
```

### 3. 여러 모델 비교
```bash
python scripts/analysis/training_analyzer.py models/model_v1
python scripts/analysis/training_analyzer.py models/model_v2
```

## 📋 분석 결과 해석

### 성능 등급
- **우수 (< 0.3)**: 🎉 매우 좋은 성능
- **양호 (< 0.5)**: ✅ 사용 가능한 성능  
- **보통 (0.5-1.0)**: ⚠️ 개선 필요
- **불량 (> 1.0)**: ❌ 재훈련 권장

### 권장사항 적용
분석 결과에서 제공하는 권장 설정을 복사하여 다음 훈련에 사용하세요.

## 🔧 확장 가능성

이 도구는 다음과 같이 확장할 수 있습니다:

1. **자동 비교**: 여러 모델 자동 비교 기능
2. **시각화**: matplotlib을 이용한 그래프 생성
3. **리포트**: HTML/PDF 리포트 생성
4. **알림**: 성능 임계값 기반 알림 시스템