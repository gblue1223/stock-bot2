#!/bin/bash
# Google Cloud Platform 분산 훈련 스크립트
# A100 40GB x12 GPU 활용

set -e

# 설정
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=12  # GPU 개수
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# 데이터베이스 경로 (GCS에서 다운로드 또는 로컬 경로)
DB_PATH="${DB_PATH:-/home/user/data/datasets_norm_all.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/user/models/embedding}"
START_DATE="${START_DATE:-20240901}"
END_DATE="${END_DATE:-20240930}"

# A100 40GB x12 최적화 설정
BATCH_SIZE=512        # GPU당 배치 크기 (총 512 x 12 = 6144)
NUM_WORKERS=16        # 데이터 로더 워커
MAX_SAMPLES=50000000  # 5천만 샘플 (A100 메모리 활용)

echo "=========================================="
echo "분산 훈련 시작"
echo "=========================================="
echo "GPU 개수: ${WORLD_SIZE}"
echo "배치 크기: ${BATCH_SIZE} (총: $((BATCH_SIZE * WORLD_SIZE)))"
echo "데이터베이스: ${DB_PATH}"
echo "출력 디렉토리: ${OUTPUT_DIR}"
echo "날짜 범위: ${START_DATE} ~ ${END_DATE}"
echo "최대 샘플: ${MAX_SAMPLES}"
echo "=========================================="

# PyTorch 분산 훈련 실행
python3 -m torch.distributed.launch \
    --nproc_per_node=${WORLD_SIZE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/gcp/train_embedding_distributed.py \
    --db "${DB_PATH}" \
    --table datasets \
    --out "${OUTPUT_DIR}" \
    --seq-len 60 \
    --embedding-dim 128 \
    --batch-size ${BATCH_SIZE} \
    --epochs 30 \
    --lr 1e-4 \
    --device cuda \
    --positive-time-threshold 10 \
    --negative-time-threshold 60 \
    --temperature 0.07 \
    --num-heads 4 \
    --val-every 3 \
    --checkpoint-interval 5 \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --num-workers ${NUM_WORKERS} \
    --max-samples ${MAX_SAMPLES}

echo ""
echo "=========================================="
echo "분산 훈련 완료!"
echo "=========================================="
