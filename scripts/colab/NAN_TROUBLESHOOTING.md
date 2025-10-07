# NaN Loss 문제 해결 가이드

## 🔍 문제 상황

```
Epoch [1] Batch [10/13659] Loss: nan
Epoch [1] Batch [20/13659] Loss: nan
```

훈련 시작 직후 Loss가 NaN이 되는 문제

## 🎯 적용된 해결책

### 1. **Gradient Clipping 추가** ✅
```python
# Gradient explosion 방지
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. **데이터 정규화 강화** ✅
```python
# NaN/Inf 체크 및 처리
if np.any(np.isnan(data)):
    data = np.nan_to_num(data, nan=0.0)

if np.any(np.isinf(data)):
    data = np.nan_to_num(data, posinf=1e10, neginf=-1e10)

# 너무 작은 std 처리
std = np.where(std < 1e-6, 1.0, std)
```

### 3. **학습률 조정** ✅
```python
# 이전: INITIAL_LR = 3e-4 (너무 높음)
# 이후: INITIAL_LR = 1e-4 (안정적)
```

### 4. **NaN 조기 감지** ✅
```python
if np.isnan(loss_value) or np.isinf(loss_value):
    logger.error(f"NaN/Inf loss detected!")
    raise ValueError(f"NaN/Inf loss detected: {loss_value}")
```

### 5. **API 업데이트** ✅
```python
# Deprecated API 수정
torch.amp.GradScaler('cuda')  # 이전: torch.cuda.amp.GradScaler()
torch.amp.autocast('cuda')    # 이전: torch.cuda.amp.autocast()
```

## 🚀 재훈련 방법

### 1. 코드 업데이트
```bash
# Colab에서
%cd /content/stock-bot2
!git pull origin main
```

### 2. 런타임 재시작
```
런타임 → 런타임 다시 시작
```

### 3. 새 설정으로 훈련
```python
# 노트북 설정 셀
MAX_SAMPLES = 10000000  # 1천만 샘플
BATCH_SIZE = 512
TEMPERATURE = 0.10
INITIAL_LR = 1e-4       # 안정적인 학습률 (중요!)
INITIAL_EPOCHS = 20
VAL_EVERY = 1
```

## 📊 예상 결과

### 정상 훈련
```
Epoch [1] Batch [10/13659] Loss: 0.0856
Epoch [1] Batch [20/13659] Loss: 0.0723
Epoch [1] Batch [30/13659] Loss: 0.0645
```

Loss가 점진적으로 감소해야 합니다.

## 🔧 여전히 NaN 발생 시

### 추가 조치 1: 학습률 더 낮추기
```python
INITIAL_LR = 5e-5  # 1e-4 → 5e-5
```

### 추가 조치 2: Batch Size 줄이기
```python
BATCH_SIZE = 256  # 512 → 256
```

### 추가 조치 3: Mixed Precision 비활성화
```python
# train_embedding.py에서
scaler = None  # Mixed Precision 사용 안 함
```

### 추가 조치 4: Temperature 높이기
```python
TEMPERATURE = 0.15  # 0.10 → 0.15 (더 부드러운 학습)
```

### 추가 조치 5: 데이터 확인
```python
# 노트북에서 데이터 샘플 확인
import duckdb
conn = duckdb.connect(DB_PATH)
df = conn.execute("SELECT * FROM datasets LIMIT 1000").df()

# NaN 체크
print(f"NaN count: {df.isna().sum().sum()}")
print(f"Inf count: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

# 통계 확인
print(df.describe())
```

## 🎯 NaN 발생 원인별 해결책

### 원인 1: 학습률이 너무 높음
**증상**: 첫 배치부터 NaN
**해결**: `INITIAL_LR = 1e-4` → `5e-5`

### 원인 2: Gradient Explosion
**증상**: 몇 배치 후 갑자기 NaN
**해결**: Gradient Clipping (이미 적용됨)

### 원인 3: 데이터에 NaN/Inf 포함
**증상**: 특정 배치에서만 NaN
**해결**: 데이터 정규화 강화 (이미 적용됨)

### 원인 4: Numerical Instability
**증상**: Mixed Precision 사용 시 NaN
**해결**: Mixed Precision 비활성화 또는 더 낮은 학습률

### 원인 5: Temperature가 너무 낮음
**증상**: InfoNCE Loss에서 exp() overflow
**해결**: `TEMPERATURE = 0.15` 이상

## 💡 권장 설정 (안정성 우선)

```python
# 가장 안정적인 설정
MAX_SAMPLES = 5000000   # 500만 (메모리 안전)
BATCH_SIZE = 256        # 작은 배치
TEMPERATURE = 0.15      # 높은 temperature
INITIAL_LR = 5e-5       # 낮은 학습률
INITIAL_EPOCHS = 20
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
```

## 📈 단계별 디버깅

### Step 1: 최소 설정으로 테스트
```python
MAX_SAMPLES = 100000    # 10만 샘플
BATCH_SIZE = 64
INITIAL_LR = 1e-5       # 매우 낮은 학습률
INITIAL_EPOCHS = 1
```

→ 성공하면 점진적으로 증가

### Step 2: 샘플 수 증가
```python
MAX_SAMPLES = 1000000   # 100만
BATCH_SIZE = 128
INITIAL_LR = 5e-5
```

### Step 3: 최종 설정
```python
MAX_SAMPLES = 10000000  # 1천만
BATCH_SIZE = 512
INITIAL_LR = 1e-4
```

## 🆘 긴급 해결책

NaN이 계속 발생하면:

```python
# 1. Mixed Precision 끄기
scaler = None

# 2. 매우 낮은 학습률
INITIAL_LR = 1e-5

# 3. 작은 배치
BATCH_SIZE = 64

# 4. 높은 Temperature
TEMPERATURE = 0.20

# 5. 강한 Gradient Clipping
max_norm = 0.5  # 1.0 → 0.5
```

## ✅ 체크리스트

훈련 시작 전 확인:

- [ ] 코드 최신 버전으로 업데이트
- [ ] 학습률 1e-4 이하
- [ ] Gradient Clipping 활성화 확인
- [ ] 데이터베이스 파일 정상 확인
- [ ] GPU 메모리 충분한지 확인
- [ ] 첫 10 배치 Loss 모니터링

---

**모든 수정 사항이 적용되었으니 재훈련을 시작하세요!** 🚀
