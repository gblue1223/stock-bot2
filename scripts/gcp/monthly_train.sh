#!/bin/bash
# Google Cloud Platform 월별 자동 훈련 스크립트
# A100 40GB x12 GPU 활용

set -e

# 설정
DB_PATH="/home/user/data/datasets_norm_all.duckdb"
OUTPUT_BASE="/home/user/models/embedding"
YEAR=2024
START_MONTH=9
END_MONTH=12

# A100 x12 최적화 설정
export WORLD_SIZE=12
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

BATCH_SIZE=512        # GPU당 배치 크기
NUM_WORKERS=16
MAX_SAMPLES=50000000  # 5천만 샘플

echo "=========================================="
echo "월별 자동 훈련 시작"
echo "=========================================="
echo "기간: ${YEAR}년 ${START_MONTH}월 ~ ${END_MONTH}월"
echo "GPU: ${WORLD_SIZE}개"
echo "배치 크기: ${BATCH_SIZE} x ${WORLD_SIZE} = $((BATCH_SIZE * WORLD_SIZE))"
echo "=========================================="

# 월별 날짜 계산 함수
get_month_dates() {
    local year=$1
    local month=$2
    
    # 시작일
    start_date=$(printf "%04d%02d01" $year $month)
    
    # 종료일 (다음 달 1일 - 1일)
    if [ $month -eq 12 ]; then
        end_year=$((year + 1))
        end_month=1
    else
        end_year=$year
        end_month=$((month + 1))
    fi
    
    # 마지막 날 계산
    end_date=$(date -d "${end_year}-${end_month}-01 -1 day" +%Y%m%d)
    
    echo "${start_date} ${end_date}"
}

# 초기 훈련 (첫 달)
train_initial() {
    local year=$1
    local month=$2
    local output_dir="${OUTPUT_BASE}_${year}_$(printf "%02d" $month)"
    
    read start_date end_date <<< $(get_month_dates $year $month)
    
    echo ""
    echo "=========================================="
    echo "초기 훈련: ${year}년 ${month}월"
    echo "날짜: ${start_date} ~ ${end_date}"
    echo "출력: ${output_dir}"
    echo "=========================================="
    
    DB_PATH="${DB_PATH}" \
    OUTPUT_DIR="${output_dir}" \
    START_DATE="${start_date}" \
    END_DATE="${end_date}" \
    bash scripts/gcp/train_distributed.sh
    
    echo "✓ ${year}년 ${month}월 초기 훈련 완료"
}

# Fine-tuning (이후 달)
train_finetune() {
    local year=$1
    local month=$2
    local prev_checkpoint=$3
    local output_dir="${OUTPUT_BASE}_${year}_$(printf "%02d" $month)"
    
    read start_date end_date <<< $(get_month_dates $year $month)
    
    echo ""
    echo "=========================================="
    echo "Fine-tuning: ${year}년 ${month}월"
    echo "날짜: ${start_date} ~ ${end_date}"
    echo "이전 모델: ${prev_checkpoint}"
    echo "출력: ${output_dir}"
    echo "=========================================="
    
    # Fine-tuning 설정 (에포크 절반, 학습률 절반)
    python3 -m torch.distributed.launch \
        --nproc_per_node=${WORLD_SIZE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        scripts/gcp/train_embedding_distributed.py \
        --db "${DB_PATH}" \
        --table datasets \
        --out "${output_dir}" \
        --seq-len 60 \
        --embedding-dim 128 \
        --batch-size ${BATCH_SIZE} \
        --epochs 15 \
        --lr 5e-5 \
        --device cuda \
        --positive-time-threshold 10 \
        --negative-time-threshold 60 \
        --temperature 0.07 \
        --num-heads 4 \
        --val-every 3 \
        --checkpoint-interval 5 \
        --start-date "${start_date}" \
        --end-date "${end_date}" \
        --num-workers ${NUM_WORKERS} \
        --max-samples ${MAX_SAMPLES} \
        --resume "${prev_checkpoint}"
    
    echo "✓ ${year}년 ${month}월 Fine-tuning 완료"
}

# 월별 순차 훈련
prev_checkpoint=""

for month in $(seq $START_MONTH $END_MONTH); do
    if [ -z "$prev_checkpoint" ]; then
        # 첫 달: 초기 훈련
        train_initial $YEAR $month
        prev_checkpoint="${OUTPUT_BASE}_${YEAR}_$(printf "%02d" $month)/checkpoint_epoch30.pt"
    else
        # 이후 달: Fine-tuning
        train_finetune $YEAR $month "$prev_checkpoint"
        prev_checkpoint="${OUTPUT_BASE}_${YEAR}_$(printf "%02d" $month)/checkpoint_epoch15.pt"
    fi
done

echo ""
echo "=========================================="
echo "✓ 전체 월별 훈련 완료!"
echo "=========================================="
echo "훈련된 모델:"
for month in $(seq $START_MONTH $END_MONTH); do
    echo "  - ${OUTPUT_BASE}_${YEAR}_$(printf "%02d" $month)"
done
echo "=========================================="
