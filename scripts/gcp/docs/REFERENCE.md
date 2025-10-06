# GCP 스크립트 레퍼런스

빠른 참조를 위한 스크립트 목록 및 사용법입니다.

## 📑 스크립트 목록

### 🚀 핵심 워크플로우

| 스크립트 | 설명 | 사용 예시 |
|---------|------|----------|
| `create_instance.sh` | GPU 인스턴스 생성 | `./create_instance.sh` |
| `setup_instance.sh` | 환경 설정 및 패키지 설치 | `./setup_instance.sh` |
| `upload_and_train.sh` | 코드/데이터 업로드 및 훈련 시작 | `./upload_and_train.sh` |
| `monitor_training.sh` | 훈련 모니터링 (대화형) | `./monitor_training.sh` |
| `download_results.sh` | 훈련 결과 다운로드 | `./download_results.sh` |
| `delete_instance.sh` | 인스턴스 삭제 | `./delete_instance.sh` |

### 🎯 자동화 & 유틸리티

| 스크립트 | 설명 | 사용 예시 |
|---------|------|----------|
| `run_training.sh` | 전체 워크플로우 자동 실행 | `./run_training.sh` |
| `check_costs.sh` | 비용 확인 및 추정 | `./check_costs.sh` |
| `make_executable.sh` | 실행 권한 부여 | `./make_executable.sh` |

### 🔧 고급 기능

| 스크립트 | 설명 | 사용 예시 |
|---------|------|----------|
| `train_distributed.sh` | 분산 훈련 (다중 GPU) | `./train_distributed.sh` |
| `monthly_train.sh` | 월간 자동 훈련 | `./monthly_train.sh` |

### ⚙️ 설정

| 파일 | 설명 |
|------|------|
| `config.example.sh` | 환경 변수 설정 예시 |

## 🎬 사용 시나리오

### 시나리오 1: 처음 사용 (자동)

```bash
source config.sh
./run_training.sh
```

### 시나리오 2: 단계별 실행

```bash
source config.sh
./create_instance.sh      # 1. 인스턴스 생성
./setup_instance.sh       # 2. 환경 설정
./upload_and_train.sh     # 3. 훈련 시작
./monitor_training.sh     # 4. 모니터링
./download_results.sh     # 5. 결과 다운로드
./delete_instance.sh      # 6. 인스턴스 삭제
```

### 시나리오 3: 빠른 테스트

```bash
export GCP_PROJECT_ID="my-project"
export DB_PATH="/path/to/data.duckdb"
export EPOCHS="10"
./run_training.sh
```

## 🔑 주요 환경 변수

### 필수
```bash
export GCP_PROJECT_ID="your-project-id"
export DB_PATH="/path/to/data.duckdb"
```

### 선택적
```bash
export GCP_ZONE="us-central1-a"
export INSTANCE_NAME="ai-trader-training"
export GPU_TYPE="nvidia-tesla-t4"
export EPOCHS="50"
export BATCH_SIZE="128"
```

## 💰 비용 참고

| 설정 | 시간당 | 3시간 훈련 |
|------|--------|-----------|
| T4 GPU | ~$0.54 | ~$1.62 |
| V100 GPU | ~$2.86 | ~$8.58 |
| Preemptible T4 | ~$0.11 | ~$0.33 |

## 📚 추가 문서

- **[README.md](../README.md)** - 전체 상세 가이드
- **[QUICKSTART.md](QUICKSTART.md)** - 5분 빠른 시작
