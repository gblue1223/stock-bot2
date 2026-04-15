#!/bin/bash
# Vast.ai 완전 자동화 스크립트 - SSH 접속 → GDrive 다운로드 → 코드 업로드 → 훈련
# 사용법: ./setup_and_train.sh <ssh_port> <user@ip> <checkpoint_gdrive_url_or_id> <dataset_gdrive_url_or_id>
# 예시:   ./setup_and_train.sh 20212 root@96.3.252.118 "https://drive.google.com/file/d/1ABC.../view" "https://drive.google.com/file/d/2DEF.../view"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Google Drive URL 또는 ID에서 파일 ID 추출
extract_gdrive_id() {
    local input="$1"
    # URL 형식: https://drive.google.com/file/d/FILE_ID/view...
    if [[ "$input" =~ /file/d/([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        # 이미 ID만 입력된 경우
        echo "$input"
    fi
}

# 파라미터 검증
if [ $# -lt 4 ]; then
    log_error "사용법: $0 <ssh_port> <user@ip> <checkpoint_gdrive_url_or_id> <dataset_gdrive_url_or_id>"
    echo ""
    echo "예시:"
    echo "  $0 20212 root@96.3.252.118 \"https://drive.google.com/file/d/1ABC.../view\" \"https://drive.google.com/file/d/2DEF.../view\""
    echo "  $0 20212 root@96.3.252.118 1ABC123DEF456GHI 2JKL789MNO012PQR"
    exit 1
fi

SSH_PORT="$1"
SSH_TARGET="$2"   # user@ip 형식
CHECKPOINT_ID="$(extract_gdrive_id "$3")"
DATASET_ID="$(extract_gdrive_id "$4")"
REMOTE_WORKSPACE="/workspace"
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no"

log_info "======================================"
log_info "🚀 Vast.ai 자동화 스크립트 시작"
log_info "대상: $SSH_TARGET  Port: $SSH_PORT"
log_info "체크포인트 ID: $CHECKPOINT_ID"
log_info "데이터셋 ID:   $DATASET_ID"
log_info "======================================"

# 1. SSH 연결 테스트
log_info "📡 SSH 연결 테스트 중..."
if ! ssh $SSH_OPTS "$SSH_TARGET" "echo ok" &>/dev/null; then
    log_error "SSH 연결 실패. 포트, 대상 주소, SSH 키를 확인해주세요."
    log_info "수동 연결 시도: ssh $SSH_OPTS $SSH_TARGET"
    exit 1
fi
log_success "SSH 연결 성공"

# 2. 원격 설정 스크립트 업로드 및 실행
log_info "📤 설정 스크립트 업로드 중..."
scp $SSH_OPTS "$SCRIPT_DIR/remote_setup.sh" "$SSH_TARGET:/tmp/"

log_info "� 원격 환경 설정 및 GDrive 파일 다운로드 중..."
ssh $SSH_OPTS "$SSH_TARGET" "chmod +x /tmp/remote_setup.sh && /tmp/remote_setup.sh '$CHECKPOINT_ID' '$DATASET_ID'"

# 3. 프로젝트 코드 업로드
log_info "📤 프로젝트 코드 업로드 중..."
for dir in ai_trader config lib; do
    log_info "  → $dir/"
    scp -r $SSH_OPTS "$PROJECT_ROOT/$dir" "$SSH_TARGET:$REMOTE_WORKSPACE/"
done

# 4. 훈련 스크립트 업로드 및 실행
log_info "📤 훈련 스크립트 업로드 중..."
scp $SSH_OPTS "$SCRIPT_DIR/train_e2e.sh" "$SSH_TARGET:$REMOTE_WORKSPACE/"

log_info "🚀 훈련 시작..."
ssh $SSH_OPTS "$SSH_TARGET" \
    "cd $REMOTE_WORKSPACE && chmod +x train_e2e.sh && ./train_e2e.sh $REMOTE_WORKSPACE/datasets_raw_09_11.duckdb"

log_success "🎉 완료!"
log_info "SSH 접속:    ssh $SSH_OPTS $SSH_TARGET"
log_info "TensorBoard: ssh $SSH_OPTS -L 6006:localhost:6006 $SSH_TARGET"
