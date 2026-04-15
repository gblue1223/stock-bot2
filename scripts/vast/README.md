# Vast.ai 자동화 스크립트 사용 가이드

## 개요

Vast.ai 인스턴스에서 SSH 접속부터 Google Drive 파일 다운로드, 프로젝트 설정, 훈련 실행까지 모든 과정을 자동화하는 스크립트입니다.

## 사전 준비

### 1. Google Drive 파일 준비

Google Drive에 다음 파일들을 업로드하고 공유 링크를 생성하세요:

- `checkpoint_iter770.pt`: 사전 훈련된 모델 체크포인트
- `datasets_raw_09_11.duckdb`: 훈련용 데이터셋

**공유 링크에서 파일 ID 추출 방법:**
```
https://drive.google.com/file/d/1ABC123DEF456GHI789JKL/view?usp=sharing
                            ↑ 이 부분이 파일 ID
```

### 2. SSH 키 설정

Vast.ai 인스턴스에 SSH 키를 등록하거나, 기본 SSH 키 경로를 확인하세요:
```bash
# SSH 키 생성 (없는 경우)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 공개 키 확인
cat ~/.ssh/id_rsa.pub
```

### 3. Vast.ai 인스턴스 정보 확인

- 인스턴스 IP 주소
- SSH 포트 번호 (보통 22가 아닌 다른 포트)

## 사용법

### 기본 사용법

```bash
# 실행 권한 부여
chmod +x scripts/vast/setup_and_train.sh

# 스크립트 실행
./scripts/vast/setup_and_train.sh <vast_ip> <ssh_port> <checkpoint_file_id> <dataset_file_id>
```

### 예시

```bash
# Google Drive 파일 ID와 함께 실행
./scripts/vast/setup_and_train.sh 123.45.67.89 12345 1ABC123DEF456GHI789JKL 2MNO456PQR789STU012VWX

# 파일 ID 없이 실행 (기본값 사용)
./scripts/vast/setup_and_train.sh 123.45.67.89 12345
```

## 스크립트 동작 과정

### 1. SSH 연결 테스트
- Vast.ai 인스턴스에 SSH 연결 가능 여부 확인
- 연결 실패 시 오류 메시지와 함께 종료

### 2. 원격 환경 설정
- 시스템 패키지 업데이트
- Python 의존성 설치 (PyTorch, DuckDB, Stable-Baselines3 등)
- 프로젝트 디렉토리 구조 생성

### 3. Google Drive 파일 다운로드
- `gdown`을 사용하여 체크포인트와 데이터셋 다운로드
- 파일 크기 및 다운로드 상태 확인

### 4. 프로젝트 코드 업로드
- `ai_trader` 패키지
- `config` 설정 파일
- `lib` 라이브러리 파일

### 5. 훈련 실행
- 환경 변수 설정
- GPU 상태 확인
- E2E 훈련 스크립트 실행

## 훈련 파라미터

기본 훈련 파라미터는 다음과 같습니다:

```python
--db /workspace/datasets_raw_09_11.duckdb
--load_policy /workspace/models/grpo_scalping_v10/checkpoints/checkpoint_iter770.pt
--seq_len 3000
--features 28
--total_timesteps 200000
--output_dir /workspace/models/grpo_scalping_v10
--early_exit_penalty 1.2
--num_workers 2
--min_holding 2
--max_holding 300
--entropy_coef 0.15
--transaction_cost 0.00215
--no_trade_penalty 30.0
--lr 0.00001
```

## 모니터링

### 훈련 로그 확인

```bash
# SSH로 접속
ssh -p <ssh_port> root@<vast_ip>

# 로그 파일 확인
tail -f /workspace/logs/training.log
```

### TensorBoard 실행

```bash
# 원격 인스턴스에서 실행
cd /workspace
tensorboard --logdir runs --host 0.0.0.0 --port 6006

# 로컬에서 접속 (포트 포워딩)
ssh -p <ssh_port> -L 6006:localhost:6006 root@<vast_ip>
# 브라우저에서 http://localhost:6006 접속
```

## 트러블슈팅

### SSH 연결 실패

```bash
# 연결 테스트
ssh -p <ssh_port> root@<vast_ip>

# SSH 키 확인
ls -la ~/.ssh/
cat ~/.ssh/id_rsa.pub
```

### Google Drive 다운로드 실패

1. 파일 ID가 올바른지 확인
2. Google Drive 파일이 공개 또는 링크 공유로 설정되어 있는지 확인
3. 파일 크기가 너무 큰 경우 다운로드 시간이 오래 걸릴 수 있음

### GPU 인식 실패

```bash
# GPU 상태 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족

- `--num_workers` 값을 줄여보세요 (기본값: 2)
- `--seq_len` 값을 줄여보세요 (기본값: 3000)
- Vast.ai 인스턴스의 RAM 용량을 확인하세요

## 파일 구조

스크립트 실행 후 원격 인스턴스의 파일 구조:

```
/workspace/
├── ai_trader/           # 메인 패키지
├── config/              # 설정 파일
├── lib/                 # 라이브러리
├── models/              # 모델 체크포인트
│   └── grpo_scalping_v10/
│       └── checkpoints/
│           └── checkpoint_iter770.pt
├── logs/                # 로그 파일
│   └── training.log
├── runs/                # TensorBoard 로그
├── datasets_raw_09_11.duckdb  # 데이터셋
├── train_e2e_vast.py    # 훈련 스크립트
└── .env                 # 환경 변수
```

## 비용 최적화 팁

1. **인스턴스 선택**: GPU 메모리 8GB 이상, RAM 16GB 이상 권장
2. **자동 종료**: 훈련 완료 후 자동으로 인스턴스를 종료하도록 설정
3. **스팟 인스턴스**: 비용 절약을 위해 스팟 인스턴스 사용 고려
4. **모니터링**: 훈련 진행 상황을 주기적으로 확인하여 불필요한 비용 방지