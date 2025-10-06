#!/bin/bash
# 모든 스크립트에 실행 권한 부여

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "GCP 스크립트에 실행 권한 부여 중..."

# 새로운 스크립트들
chmod +x "$SCRIPT_DIR/create_instance.sh"
chmod +x "$SCRIPT_DIR/setup_instance.sh"
chmod +x "$SCRIPT_DIR/upload_and_train.sh"
chmod +x "$SCRIPT_DIR/delete_instance.sh"
chmod +x "$SCRIPT_DIR/monitor_training.sh"
chmod +x "$SCRIPT_DIR/download_results.sh"
chmod +x "$SCRIPT_DIR/check_costs.sh"
chmod +x "$SCRIPT_DIR/run_training.sh"

# 기존 스크립트들
chmod +x "$SCRIPT_DIR/setup.sh"
chmod +x "$SCRIPT_DIR/train_distributed.sh"
chmod +x "$SCRIPT_DIR/monthly_train.sh"
chmod +x "$SCRIPT_DIR/upload_download.sh"
chmod +x "$SCRIPT_DIR/train_embedding_distributed.py"

echo "✓ 모든 스크립트에 실행 권한 부여 완료"
echo ""
echo "사용 가능한 스크립트:"
ls -lh "$SCRIPT_DIR"/*.sh | awk '{print "  " $9}'
