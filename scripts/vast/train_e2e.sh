#!/bin/bash
# Vast.ai E2E 훈련 스크립트
# 사용법: ./train_e2e.sh [db_path]

# 스크립트 위치 기준으로 ROOT_DIR 결정
# - 로컬(프로젝트 루트/scripts/vast/): ../../ 로 올라감
# - 원격(/workspace/): 스크립트가 /workspace에 있으므로 그대로 사용
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../../ai_trader/grpo/train_e2e.py" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

DB_PATH="${1:-$ROOT_DIR/datasets_raw_09_11.duckdb}"
LOAD_POLICY="$ROOT_DIR/models/grpo_scalping_v10/checkpoints/checkpoint_iter770.pt"

if [ ! -f "$DB_PATH" ]; then
    echo "⚠️ DB 파일을 찾을 수 없습니다: $DB_PATH"
    echo "사용법: $0 [/path/to/datasets.duckdb]"
    exit 1
fi

echo "======================================"
echo "🚀 훈련 시작"
echo "ROOT: $ROOT_DIR"
echo "DB: $DB_PATH"
echo "체크포인트: $LOAD_POLICY"
echo "======================================"

python3 "$ROOT_DIR/ai_trader/grpo/train_e2e.py" \
    --db "$DB_PATH" \
    --load_policy "$LOAD_POLICY" \
    --seq_len 3000 \
    --features 28 \
    --total_timesteps 200000 \
    --output_dir "$ROOT_DIR/models/grpo_scalping_v10" \
    --early_exit_penalty 1.2 \
    --num_workers 2 \
    --min_holding 2 \
    --max_holding 300 \
    --entropy_coef 0.15 \
    --transaction_cost 0.00215 \
    --no_trade_penalty 30.0 \
    --lr 0.00001
