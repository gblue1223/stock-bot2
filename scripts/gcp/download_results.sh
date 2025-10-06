#!/bin/bash
# 훈련 결과 다운로드 스크립트
# GCP 인스턴스에서 훈련된 모델과 로그를 다운로드합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"

# 원격 경로
REMOTE_DIR="~/ai-trader"
REMOTE_MODEL_DIR="$REMOTE_DIR/models/embedding"
REMOTE_LOG_FILE="$REMOTE_DIR/training.log"

# 로컬 경로
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_MODEL_DIR="$PROJECT_ROOT/models/embedding_gcp"
LOCAL_LOG_DIR="$PROJECT_ROOT/logs/gcp"

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

create_local_dirs() {
    print_header "로컬 디렉토리 생성"
    
    mkdir -p "$LOCAL_MODEL_DIR"
    mkdir -p "$LOCAL_LOG_DIR"
    
    echo "모델 저장 경로: $LOCAL_MODEL_DIR"
    echo "로그 저장 경로: $LOCAL_LOG_DIR"
    echo ""
}

check_remote_files() {
    print_header "원격 파일 확인"
    
    echo "체크포인트 파일:"
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="ls -lh $REMOTE_MODEL_DIR/*.pt 2>/dev/null || echo '체크포인트 없음'"
    
    echo ""
    echo "TensorBoard 로그:"
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="du -sh $REMOTE_MODEL_DIR/tensorboard_logs 2>/dev/null || echo 'TensorBoard 로그 없음'"
    
    echo ""
    echo "훈련 로그:"
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="ls -lh $REMOTE_LOG_FILE 2>/dev/null || echo '훈련 로그 없음'"
    
    echo ""
}

download_models() {
    print_header "모델 체크포인트 다운로드"
    
    echo "다운로드 중..."
    gcloud compute scp --recurse \
        "$INSTANCE_NAME:$REMOTE_MODEL_DIR/*.pt" \
        "$LOCAL_MODEL_DIR/" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" 2>/dev/null || echo "체크포인트 파일이 없거나 다운로드 실패"
    
    echo "✓ 모델 다운로드 완료"
}

download_tensorboard_logs() {
    print_header "TensorBoard 로그 다운로드"
    
    echo "다운로드 중..."
    gcloud compute scp --recurse \
        "$INSTANCE_NAME:$REMOTE_MODEL_DIR/tensorboard_logs" \
        "$LOCAL_MODEL_DIR/" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" 2>/dev/null || echo "TensorBoard 로그가 없거나 다운로드 실패"
    
    echo "✓ TensorBoard 로그 다운로드 완료"
}

download_training_log() {
    print_header "훈련 로그 다운로드"
    
    echo "다운로드 중..."
    gcloud compute scp \
        "$INSTANCE_NAME:$REMOTE_LOG_FILE" \
        "$LOCAL_LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" 2>/dev/null || echo "훈련 로그가 없거나 다운로드 실패"
    
    echo "✓ 훈련 로그 다운로드 완료"
}

show_summary() {
    print_header "다운로드 요약"
    
    echo "로컬 저장 위치:"
    echo "  모델: $LOCAL_MODEL_DIR"
    echo "  로그: $LOCAL_LOG_DIR"
    echo ""
    
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        echo "다운로드된 파일:"
        ls -lh "$LOCAL_MODEL_DIR"/*.pt 2>/dev/null || echo "  (체크포인트 없음)"
        
        if [ -d "$LOCAL_MODEL_DIR/tensorboard_logs" ]; then
            echo ""
            echo "TensorBoard 로그 크기:"
            du -sh "$LOCAL_MODEL_DIR/tensorboard_logs"
        fi
    fi
    
    echo ""
    echo "TensorBoard 실행 방법:"
    echo "  tensorboard --logdir=$LOCAL_MODEL_DIR/tensorboard_logs"
    echo ""
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "훈련 결과 다운로드"
    
    echo "인스턴스: $INSTANCE_NAME"
    echo "프로젝트: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo ""
    
    # 로컬 디렉토리 생성
    create_local_dirs
    
    # 원격 파일 확인
    check_remote_files
    
    # 다운로드 확인
    read -p "다운로드를 시작하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 0
    fi
    
    # 다운로드 실행
    download_models
    download_tensorboard_logs
    download_training_log
    
    # 요약 출력
    show_summary
    
    echo ""
    print_header "완료"
    echo "모든 결과가 다운로드되었습니다."
    echo "이제 인스턴스를 삭제하여 비용을 절감할 수 있습니다:"
    echo "  ./delete_instance.sh"
}

main "$@"
