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
## 🔧 확장 가능성

이 도구는 다음과 같이 확장할 수 있습니다:

  1. **자동 비교**: 여러 모델 자동 비교 기능
  2. **시각화**: matplotlib을 이용한 그래프 생성
  3. **리포트**: HTML/PDF 리포트 생성
  4. **알림**: 성능 임계값 기반 알림 시스템

## 🧰 도구 통합 안내

중복되는 스크립트를 정리하여 핵심 도구만 유지했습니다.

- **유지**
  - `scripts/analysis/training_analyzer.py`: 모델 디렉토리 종합 분석
  - `scripts/analysis/diagnose_training_issues.py`: 데이터/정규화/누수 등 종합 진단
  - `scripts/check_tensorboard.py`: TensorBoard 로그 분석(워치 모드, 차트/리포트)
- **보관(Archive)**: 중복/부분기능 스크립트는 `scripts/analysis/_archive/`로 이동
  - `analyze_training_loss.py`
  - `check_group_normalization.py`
  - `deep_normalization_check.py`
  - `verify_normalization.py`
  - `check_data_quality.py`
  
## 🧪 diagnose_training_issues.py
  
  데이터셋(DuckDB)에 대한 종합 진단을 수행합니다. 데이터 누수, 그룹별 정규화 여부, 타겟 분포, 시계열 연속성, 피처 분산 등을 점검합니다.

### 사용법

```bash
python scripts/analysis/diagnose_training_issues.py path/to/dataset.duckdb \\
{{ ... }}
  --date-col 날짜 --code-col 종목명_scalar --target-col direction3
```

### 주요 출력

- **누수 탐지**: 학습/검증 구간 교차, 미래 정보 포함 여부
- **정규화 점검**: 그룹별 통계 분포 비교(날짜/종목)
- **타겟 분포**: 클래스 불균형, 편향
- **시퀀스 연속성**: 결측 구간, 샘플 간 간격
- **피처 분산**: 상수화/폭발(Inf/NaN) 여부

## 📈 TensorBoard 로그 점검

학습 로그와 메트릭 이상 탐지는 `scripts/check_tensorboard.py`를 사용하세요.

### 사용법 예시

```bash
python scripts/check_tensorboard.py runs/sb3/ppo_scalp_cnn_1 --watch --export report.html
```

### 기능

- **스칼라 요약**: 주요 태그의 최종/최고값, 추세
- **이상 탐지**: NaN/Inf, 수렴 정체, 검증 손실 비감소 등
- **리포트 출력**: 콘솔 요약, HTML/PNG 저장