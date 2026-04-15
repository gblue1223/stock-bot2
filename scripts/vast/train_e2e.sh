#!/bin/bash
# Vast.ai E2E 훈련 스크립트
# 사용법: ./train_e2e.sh [db_path]

cd "$(dirname "$0")/../.."
ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

DB_PATH="${1:-datasets_raw_09_11.duckdb}"
LOAD_POLICY="models/grpo_scalping_v10/checkpoints/checkpoint_iter770.pt"

if [ ! -f "$DB_PATH" ]; then
    echo "⚠️ DB 파일을 찾을 수 없습니다: $DB_PATH"
    echo "사용법: $0 [/path/to/datasets.duckdb]"
    exit 1
fi

echo "======================================"
echo "🚀 훈련 시작"
echo "DB: $DB_PATH"
echo "체크포인트: $LOAD_POLICY"
echo "======================================"

python ai_trader/grpo/train_e2e.py \
    --db "$DB_PATH" \
    --load_policy "$LOAD_POLICY" \
    --seq_len 3000 \
    --features 28 \
    --total_timesteps 200000 \
    --output_dir "models/grpo_scalping_v10" \
    --early_exit_penalty 1.2 \
    --num_workers 2 \
    --min_holding 2 \
    --max_holding 300 \
    --entropy_coef 0.15 \
    --transaction_cost 0.00215 \
    --no_trade_penalty 30.0 \
    --lr 0.00001
