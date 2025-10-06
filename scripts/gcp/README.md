# GCP 훈련 스크립트

Google Cloud Platform에서 GPU 인스턴스를 생성하고 딥러닝 모델을 훈련하기 위한 bash 스크립트 모음입니다.

## 📋 목차

- [사전 준비](#사전-준비)
- [빠른 시작](#빠른-시작)
- [스크립트 설명](#스크립트-설명)
- [환경 변수](#환경-변수)
- [비용 관리](#비용-관리)
- [문제 해결](#문제-해결)

## 📚 추가 문서

- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 5분 빠른 시작 가이드
- **[docs/REFERENCE.md](docs/REFERENCE.md)** - 스크립트 레퍼런스

## 🔧 사전 준비

### 1. Google Cloud SDK 설치

```bash
# Windows (PowerShell)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe

# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
```

### 2. gcloud 초기화

```bash
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. GPU 할당량 확인

GCP 콘솔에서 GPU 할당량을 확인하고 필요시 증가 요청:
- https://console.cloud.google.com/iam-admin/quotas

필요한 할당량:
- **NVIDIA T4 GPUs**: 최소 1개
- **CPUs**: 최소 4개
- **In-use IP addresses**: 최소 1개

### 4. 환경 변수 설정

```bash
# Windows (PowerShell)
$env:GCP_PROJECT_ID = "your-project-id"
$env:GCP_ZONE = "us-central1-a"
$env:DB_PATH = "C:\path\to\datasets_norm_all.duckdb"

# Linux/macOS
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"
export DB_PATH="/path/to/datasets_norm_all.duckdb"
```

## 🚀 빠른 시작

### 전체 워크플로우

```bash
# 1. 인스턴스 생성
./create_instance.sh

# 2. 환경 설정
./setup_instance.sh

# 3. 코드 업로드 및 훈련 시작
./upload_and_train.sh

# 4. 훈련 모니터링 (선택사항)
./monitor_training.sh

# 5. 결과 다운로드
./download_results.sh

# 6. 인스턴스 삭제 (비용 절감)
./delete_instance.sh
```

### Windows에서 실행

Windows에서는 Git Bash 또는 WSL을 사용하세요:

```bash
# Git Bash 사용
bash create_instance.sh

# WSL 사용
wsl bash create_instance.sh
```

## 📜 스크립트 설명

### 1. `create_instance.sh`

GPU 인스턴스를 생성합니다.

**기본 설정:**
- 머신 타입: `n1-standard-4` (4 vCPU, 15GB RAM)
- GPU: `nvidia-tesla-t4` x 1
- 디스크: 100GB
- 이미지: PyTorch GPU (Deep Learning VM)

**커스터마이징:**

```bash
# 더 강력한 GPU 사용
export GPU_TYPE="nvidia-tesla-v100"
export GPU_COUNT="2"

# 더 큰 머신 타입
export MACHINE_TYPE="n1-standard-8"

# 다른 Zone (가격 확인 필요)
export GCP_ZONE="us-west1-b"

./create_instance.sh
```

**예상 비용 (us-central1-a):**
- n1-standard-4: $0.19/시간
- NVIDIA T4 GPU: $0.35/시간
- **총: ~$0.54/시간** (~$13/일)

### 2. `setup_instance.sh`

인스턴스에 필요한 패키지를 설치합니다.

**수행 작업:**
- 시스템 패키지 업데이트
- Git, tmux, htop 등 유틸리티 설치
- NVIDIA 드라이버 확인
- PyTorch 및 CUDA 확인

### 3. `upload_and_train.sh`

훈련 코드와 데이터를 업로드하고 훈련을 시작합니다.

**수행 작업:**
1. 로컬 코드 압축 및 업로드
2. 데이터베이스 파일 업로드
3. Python 의존성 설치
4. tmux 세션에서 훈련 시작

**훈련 파라미터 커스터마이징:**

```bash
export SEQ_LEN=120
export EMBEDDING_DIM=256
export BATCH_SIZE=256
export EPOCHS=100
export LEARNING_RATE=5e-5

./upload_and_train.sh
```

### 4. `monitor_training.sh`

훈련 진행 상황을 모니터링합니다.

**기능:**
- 실시간 로그 확인
- GPU 사용률 모니터링
- TensorBoard 포트 포워딩

### 5. `download_results.sh`

훈련된 모델과 로그를 다운로드합니다.

**다운로드 항목:**
- 체크포인트 파일 (`.pt`)
- TensorBoard 로그
- 훈련 로그 파일

### 6. `delete_instance.sh`

인스턴스를 삭제하여 비용을 절감합니다.

**주의사항:**
- 삭제 전 결과를 반드시 백업하세요!
- 삭제는 되돌릴 수 없습니다.

## 🔐 환경 변수

### 필수 변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `GCP_PROJECT_ID` | GCP 프로젝트 ID | `my-project-123` |
| `DB_PATH` | 로컬 데이터베이스 경로 | `C:\data\datasets.duckdb` |

### 선택적 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GCP_ZONE` | `us-central1-a` | GCP Zone |
| `INSTANCE_NAME` | `ai-trader-training` | 인스턴스 이름 |
| `MACHINE_TYPE` | `n1-standard-4` | 머신 타입 |
| `GPU_TYPE` | `nvidia-tesla-t4` | GPU 타입 |
| `GPU_COUNT` | `1` | GPU 개수 |
| `BOOT_DISK_SIZE` | `100GB` | 부트 디스크 크기 |
| `SEQ_LEN` | `60` | 시퀀스 길이 |
| `EMBEDDING_DIM` | `128` | 임베딩 차원 |
| `BATCH_SIZE` | `128` | 배치 크기 |
| `EPOCHS` | `50` | 에포크 수 |
| `LEARNING_RATE` | `1e-4` | 학습률 |

## 💰 비용 관리

### 비용 추정

**기본 설정 (n1-standard-4 + T4 GPU):**
- 시간당: ~$0.54
- 일일 (24시간): ~$13
- 주간 (7일): ~$91

**훈련 시간 예측:**
- 50 에포크, 배치 크기 128: 약 2-4시간
- 100 에포크, 배치 크기 256: 약 6-10시간

### 비용 절감 팁

1. **Preemptible VM 사용** (최대 80% 절감):
   ```bash
   # create_instance.sh 수정
   --provisioning-model=SPOT
   ```
   주의: 24시간 내에 중단될 수 있음

2. **훈련 완료 후 즉시 삭제**:
   ```bash
   ./delete_instance.sh
   ```

3. **저렴한 Zone 선택**:
   - `us-central1-a` (저렴)
   - `asia-northeast1-a` (한국과 가까움, 약간 비쌈)

4. **필요한 만큼만 훈련**:
   - 작은 에포크로 시작 (10-20)
   - 결과 확인 후 추가 훈련

### 비용 모니터링

GCP 콘솔에서 실시간 비용 확인:
- https://console.cloud.google.com/billing

예산 알림 설정:
```bash
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="AI Training Budget" \
  --budget-amount=100USD
```

## 🔍 문제 해결

### 인스턴스 생성 실패

**문제:** GPU 할당량 초과
```
ERROR: Quota 'NVIDIA_T4_GPUS' exceeded
```

**해결:**
1. GCP 콘솔에서 할당량 증가 요청
2. 다른 Zone 시도
3. 다른 GPU 타입 시도 (K80, P4 등)

### SSH 연결 실패

**문제:** 인스턴스에 연결할 수 없음

**해결:**
```bash
# 방화벽 규칙 확인
gcloud compute firewall-rules list

# SSH 키 재설정
gcloud compute config-ssh

# 수동 연결 시도
gcloud compute ssh ai-trader-training --zone=us-central1-a
```

### GPU 인식 안 됨

**문제:** `nvidia-smi` 실행 실패

**해결:**
```bash
# 인스턴스 SSH 접속 후
sudo /opt/deeplearning/install-driver.sh

# 재부팅
sudo reboot
```

### 데이터베이스 업로드 느림

**문제:** 큰 데이터베이스 파일 업로드 시간 초과

**해결:**
1. **압축 후 업로드:**
   ```bash
   gzip -c datasets.duckdb > datasets.duckdb.gz
   gcloud compute scp datasets.duckdb.gz instance:~/
   # 원격에서 압축 해제
   gcloud compute ssh instance --command="gunzip datasets.duckdb.gz"
   ```

2. **Google Cloud Storage 사용:**
   ```bash
   # 로컬에서 GCS로 업로드
   gsutil cp datasets.duckdb gs://your-bucket/
   
   # 인스턴스에서 다운로드 (훨씬 빠름)
   gcloud compute ssh instance --command="gsutil cp gs://your-bucket/datasets.duckdb ~/"
   ```

### 훈련 중 메모리 부족

**문제:** CUDA out of memory

**해결:**
```bash
# 배치 크기 줄이기
export BATCH_SIZE=64
./upload_and_train.sh

# 또는 더 큰 GPU 사용
export GPU_TYPE="nvidia-tesla-v100"
./create_instance.sh
```

## 📊 TensorBoard 사용

### 로컬에서 TensorBoard 보기

```bash
# SSH 터널링으로 포트 포워딩
gcloud compute ssh ai-trader-training \
  --project=$GCP_PROJECT_ID \
  --zone=$GCP_ZONE \
  -- -L 6006:localhost:6006

# 원격 인스턴스에서 TensorBoard 실행
tensorboard --logdir=~/ai-trader/models/embedding/tensorboard_logs --host=0.0.0.0

# 로컬 브라우저에서 접속
# http://localhost:6006
```

## 🔄 재훈련 (Resume)

체크포인트에서 훈련 재개:

```bash
# 원격 인스턴스에 SSH 접속
gcloud compute ssh ai-trader-training

# tmux 세션 생성
tmux new -s training

# 체크포인트에서 재개
cd ~/ai-trader
python3 -m ai_trader.embedding.train_embedding \
  --db datasets.duckdb \
  --table datasets \
  --out models/embedding \
  --resume models/embedding/checkpoint_epoch30.pt \
  --epochs 100 \
  --device cuda
```

## 📝 체크리스트

### 훈련 시작 전
- [ ] GCP 프로젝트 ID 설정
- [ ] GPU 할당량 확인
- [ ] 데이터베이스 파일 준비
- [ ] 환경 변수 설정
- [ ] 예산 설정 (선택사항)

### 훈련 중
- [ ] 훈련 로그 모니터링
- [ ] GPU 사용률 확인
- [ ] TensorBoard로 메트릭 확인
- [ ] 비용 모니터링

### 훈련 완료 후
- [ ] 결과 다운로드
- [ ] 모델 성능 검증
- [ ] 인스턴스 삭제
- [ ] 비용 확인

## 🆘 지원

문제가 발생하면:
1. 이 README의 문제 해결 섹션 확인
2. GCP 문서 참조: https://cloud.google.com/compute/docs
3. 프로젝트 이슈 트래커에 문의

## 📚 추가 자료

- [GCP Compute Engine 가격](https://cloud.google.com/compute/pricing)
- [Deep Learning VM 이미지](https://cloud.google.com/deep-learning-vm)
- [GPU 가격 비교](https://cloud.google.com/compute/gpus-pricing)
- [Preemptible VM 가이드](https://cloud.google.com/compute/docs/instances/preemptible)
