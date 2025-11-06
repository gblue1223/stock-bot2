# 학습 데이터 정규화 통계 통합 완료

## ✅ 수정 완료

### 문제
- `OnlineNormalizer`가 실시간 데이터만으로 통계 계산 (200샘플 rolling window)
- 학습 데이터는 전체 데이터셋(수백만 샘플)의 통계로 정규화됨
- **분포 불일치** → 모델이 제대로 작동하지 않음 → Buy Signal Rate 0%

### 해결
학습 데이터 정규화 통계를 실시간 추론에서 사용하도록 통합

## 📝 수정된 파일

### 1. `scripts/live/online_normalizer.py`

#### OnlineNormalizer 클래스
```python
def __init__(
    self,
    num_features: int = 28,
    window_size: int = 200,
    warmup_samples: int = 50,
    clip_range: tuple = (-5.0, 5.0),
    use_training_stats: bool = False,  # ✅ 추가
    training_mean: Optional[np.ndarray] = None,  # ✅ 추가
    training_std: Optional[np.ndarray] = None  # ✅ 추가
):
```

**주요 변경사항**:
- `use_training_stats`: 학습 통계 사용 여부
- `training_mean`, `training_std`: 학습 데이터 통계
- 학습 통계 사용 시 워밍업 불필요 (즉시 준비됨)
- `is_ready()`: 학습 통계 사용 시 항상 True 반환

#### PerStockNormalizer 클래스
```python
def __init__(
    self,
    num_features: int,
    window_size: int = 200,
    warmup_samples: int = 50,
    use_training_stats: bool = False,  # ✅ 추가
    training_mean: Optional[np.ndarray] = None,  # ✅ 추가
    training_std: Optional[np.ndarray] = None  # ✅ 추가
):
```

**주요 변경사항**:
- 학습 통계를 저장하고 각 종목의 normalizer에 전달
- 종목별 normalizer 생성 시 학습 통계 사용

### 2. `scripts/live/live_trading.py`

#### 정규화 통계 로드
```python
# ✅ 학습 데이터 정규화 통계 로드
training_mean = None
training_std = None
use_training_stats = False

if config.normalization_stats_path:
    norm_stats_path = Path(config.normalization_stats_path)
    if norm_stats_path.exists():
        # JSON 또는 PKL 파일 로드
        if norm_stats_path.suffix == '.json':
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)
            training_mean = np.array(stats['mean'], dtype=np.float32)
            training_std = np.array(stats['std'], dtype=np.float32)
        elif norm_stats_path.suffix == '.pkl':
            with open(norm_stats_path, 'rb') as f:
                stats = pickle.load(f)
            training_mean = stats['mean']
            training_std = stats['std']
        
        use_training_stats = True
```

#### PerStockNormalizer 초기화
```python
self.normalizer = PerStockNormalizer(
    num_features=config.num_features,
    window_size=config.online_window_size,
    warmup_samples=config.online_warmup_samples,
    use_training_stats=use_training_stats,  # ✅ 추가
    training_mean=training_mean,  # ✅ 추가
    training_std=training_std  # ✅ 추가
)
```

## 🎯 작동 방식

### Before (문제)
```
실시간 데이터 → OnlineNormalizer (rolling 200샘플) → 정규화
                 ↓
                mean ≈ 3,318,042 (원본 값)
                std ≈ 18,240,332
                 ↓
                모델 입력 (학습 데이터와 분포 불일치)
                 ↓
                Buy Signal Rate: 0%
```

### After (해결)
```
실시간 데이터 → OnlineNormalizer (학습 통계 사용) → 정규화
                 ↓
                mean ≈ 0 (학습 데이터와 동일)
                std ≈ 1
                 ↓
                모델 입력 (학습 데이터와 분포 일치)
                 ↓
                Buy Signal Rate: 정상 (예상 30-50%)
```

## 📊 예상 로그

### 시작 시
```
✅ Loaded training normalization stats from models/grpo_scalping@2025120/normalization_stats.json
   Mean range: [-0.8312, 0.6242]
   Std range: [0.0515, 1.0000]
✅ PerStockNormalizer initialized with TRAINING STATISTICS
```

### 종목 추가 시
```
Created normalizer for stock: 000230 (using training stats)
[NORMALIZER] 000230 ready for inference  # 즉시 준비됨 (워밍업 불필요)
```

### 정규화 전후
```
[BEFORE NORM] 000230: mean=3318042.25, std=18240332.00, range=[-1.75, 131973528.00]
[AFTER NORM] 000230: mean=0.0234, std=1.2345, range=[-2.3456, 3.4567]  # ✅ 정규화됨!
```

### 예측 결과
```
[INFERENCE] Input sequence stats: mean=0.0234, std=1.2345, range=[-2.3456, 3.4567]
[PREDICT] 000230: BUY signal, confidence=0.7234  # ✅ BUY 신호 발생!
```

## 🚀 테스트 방법

### 1. 설정 확인
```bash
# config/trading_config.json
{
    "normalization_stats_path": "models/grpo_scalping@2025120/normalization_stats.json"
}
```

### 2. 실행
```bash
python scripts/live/live_trading.py
```

### 3. 로그 확인
다음 로그들이 나타나야 함:
- ✅ "Loaded training normalization stats"
- ✅ "PerStockNormalizer initialized with TRAINING STATISTICS"
- ✅ "Created normalizer for stock: XXX (using training stats)"
- ✅ "[AFTER NORM]" 로그에서 mean≈0, std≈1, range≈[-5, 5]

### 4. Buy Signal Rate 확인
```
[AI] Inference Stats:
Total Predictions: 74
Buy Signal Rate: 35.14%  # ✅ 0%에서 증가!
```

## 🔍 문제 해결

### 문제 1: 여전히 정규화가 안됨
**증상**:
```
[AFTER NORM] 000230: mean=3318042.25, ...
```

**원인**: 학습 통계가 로드되지 않음

**해결**:
1. `config.normalization_stats_path` 확인
2. 파일 경로가 올바른지 확인
3. 로그에서 "Loaded training normalization stats" 확인

### 문제 2: 학습 통계 로드 실패
**증상**:
```
Failed to load normalization stats: ...
Falling back to online normalization
```

**원인**: 파일 형식 오류 또는 경로 문제

**해결**:
1. JSON 파일 형식 확인
2. 파일 권한 확인
3. 절대 경로 사용 시도

### 문제 3: 여전히 Buy Signal Rate 0%
**증상**: 정규화는 되지만 여전히 BUY 신호 없음

**원인**: 다른 문제 (호가 데이터 누락, 모델 문제 등)

**해결**:
1. 호가 데이터 품질 확인
2. 신뢰도 임계값 낮추기
3. 다른 조건 검색식 사용

## ✅ 체크리스트

- [x] OnlineNormalizer에 학습 통계 사용 기능 추가
- [x] PerStockNormalizer에 학습 통계 전달 기능 추가
- [x] live_trading.py에서 학습 통계 로드
- [x] 진단 오류 없음
- [ ] 실시간 거래 테스트
- [ ] Buy Signal Rate 정상화 확인

## 🎉 기대 효과

1. ✅ **정규화 일관성**: 학습 데이터와 동일한 분포
2. ✅ **즉시 준비**: 워밍업 불필요 (학습 통계 사용 시)
3. ✅ **Buy Signal Rate 정상화**: 0% → 30-50%
4. ✅ **모델 성능 향상**: 올바른 입력 분포로 정확한 예측

## 📚 관련 문서

- `docs/BUY_SIGNAL_ANALYSIS_20251106.md`: 문제 분석
- `docs/FIX_PROPOSAL_BUY_SIGNAL.md`: 수정 제안
- `docs/FIX_APPLIED_BUY_SIGNAL.md`: 초기 수정
- `docs/NORMALIZATION_STATS_COMPARISON.md`: 통계 파일 비교
- `docs/TRAINING_STATS_INTEGRATION.md`: 학습 통계 통합 (현재 문서)
