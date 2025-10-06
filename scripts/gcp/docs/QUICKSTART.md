# 🚀 빠른 시작 가이드

GCP에서 5분 안에 훈련을 시작하는 방법입니다.

## 📋 사전 준비 (한 번만)

### 1. Google Cloud SDK 설치

**Windows:**
```powershell
# PowerShell에서 실행
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

**macOS:**
```bash
brew install --cask google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. gcloud 초기화

```bash
gcloud init
gcloud auth login
```

### 3. 프로젝트 설정

```bash
# 프로젝트 ID 확인
gcloud projects list

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID
```

## ⚡ 5분 시작 가이드

### Step 1: 환경 변수 설정 (1분)

**Windows (PowerShell):**
```powershell
$env:GCP_PROJECT_ID = "your-project-id"
$env:DB_PATH = "C:\path\to\datasets_norm_all.duckdb"
```

**Linux/macOS:**
```bash
export GCP_PROJECT_ID="your-project-id"
export DB_PATH="/path/to/datasets_norm_all.duckdb"
```

또는 설정 파일 사용:
```bash
cd scripts/gcp
cp config.example.sh config.sh
# config.sh 편집
source config.sh
```

### Step 2: 인스턴스 생성 (2분)

```bash
cd scripts/gcp
./create_instance.sh
```

대기 시간: 약 2-3분

### Step 3: 환경 설정 (1분)

```bash
./setup_instance.sh
```

### Step 4: 훈련 시작 (1분)

```bash
./upload_and_train.sh
```

이제 훈련이 백그라운드에서 실행됩니다! 🎉

## 📊 훈련 모니터링

### 실시간 로그 보기

```bash
./monitor_training.sh
# 메뉴에서 1번 선택
```

### TensorBoard 보기

```bash
./monitor_training.sh
# 메뉴에서 6번 선택
# 브라우저에서 http://localhost:6006 열기
```

### GPU 사용률 확인

```bash
./monitor_training.sh
# 메뉴에서 2번 선택
```

## 💾 결과 다운로드

훈련 완료 후:

```bash
./download_results.sh
```

다운로드 위치:
- 모델: `models/embedding_gcp/`
- 로그: `logs/gcp/`

## 🗑️ 인스턴스 삭제 (중요!)

**비용 절감을 위해 반드시 삭제하세요:**

```bash
./delete_instance.sh
```

## 🔧 커스터마이징

### 더 강력한 GPU 사용

```bash
export GPU_TYPE="nvidia-tesla-v100"
export MACHINE_TYPE="n1-standard-8"
./create_instance.sh
```

### 더 긴 훈련

```bash
export EPOCHS="100"
export BATCH_SIZE="256"
./upload_and_train.sh
```

### 더 큰 모델

```bash
export EMBEDDING_DIM="256"
export SEQ_LEN="120"
./upload_and_train.sh
```

## 💰 예상 비용

**기본 설정 (T4 GPU):**
- 시간당: ~$0.54
- 50 에포크 (약 3시간): ~$1.62

**고성능 설정 (V100 GPU):**
- 시간당: ~$2.86
- 50 에포크 (약 2시간): ~$5.72

**비용 절감 팁:**
```bash
# Preemptible VM 사용 (80% 할인)
export USE_PREEMPTIBLE="true"
./create_instance.sh
```

## 🆘 문제 해결

### GPU 할당량 초과

```
ERROR: Quota 'NVIDIA_T4_GPUS' exceeded
```

**해결:**
1. [GCP 콘솔](https://console.cloud.google.com/iam-admin/quotas)에서 할당량 증가 요청
2. 또는 다른 Zone 시도:
   ```bash
   export GCP_ZONE="us-west1-b"
   ./create_instance.sh
   ```

### SSH 연결 실패

```bash
# SSH 키 재설정
gcloud compute config-ssh

# 수동 연결
gcloud compute ssh ai-trader-training --zone=us-central1-a
```

### 메모리 부족 (CUDA OOM)

```bash
# 배치 크기 줄이기
export BATCH_SIZE="64"
./upload_and_train.sh
```

### 데이터베이스 업로드 느림

큰 파일은 GCS 사용:
```bash
# 1. GCS에 업로드
gsutil cp datasets.duckdb gs://your-bucket/

# 2. 인스턴스에서 다운로드
gcloud compute ssh ai-trader-training --command="gsutil cp gs://your-bucket/datasets.duckdb ~/"
```

## 📚 다음 단계

- [전체 문서 읽기](README.md)
- [비용 최적화 가이드](README.md#비용-관리)
- [고급 설정](README.md#환경-변수)

## ✅ 체크리스트

훈련 시작 전:
- [ ] gcloud 설치 및 로그인
- [ ] 프로젝트 ID 설정
- [ ] 데이터베이스 파일 준비
- [ ] GPU 할당량 확인

훈련 중:
- [ ] 로그 모니터링
- [ ] GPU 사용률 확인
- [ ] 비용 모니터링

훈련 완료 후:
- [ ] 결과 다운로드
- [ ] 인스턴스 삭제 ⚠️

## 🎯 전체 워크플로우 (한 줄로)

```bash
source config.sh && ./create_instance.sh && ./setup_instance.sh && ./upload_and_train.sh
```

훈련 완료 후:
```bash
./download_results.sh && ./delete_instance.sh
```

끝! 🎉
