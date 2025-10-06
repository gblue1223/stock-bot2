#!/bin/bash
# 훈련 코드 업로드 및 실행 스크립트
# 로컬 코드를 GCP 인스턴스에 업로드하고 훈련을 시작합니다.

set -e  # 에러 발생 시 스크립트 중단

# =============================================================================
# 설정 변수
# =============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-ai-trader-training}"

# 로컬 프로젝트 경로 (이 스크립트의 상위 디렉토리)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 원격 작업 디렉토리
REMOTE_DIR="~/ai-trader"

# 훈련 설정
DB_PATH="${DB_PATH:-}"  # 로컬 데이터베이스 경로
REMOTE_DB_PATH="$REMOTE_DIR/datasets.duckdb"
TABLE_NAME="${TABLE_NAME:-datasets}"
OUTPUT_DIR="$REMOTE_DIR/models/embedding"
SEQ_LEN="${SEQ_LEN:-60}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# =============================================================================
# 함수 정의
# =============================================================================

print_header() {
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

check_database() {
    if [ -z "$DB_PATH" ]; then
        echo "Error: DB_PATH 환경 변수를 설정해주세요."
        echo "예시: export DB_PATH=/path/to/datasets_norm_all.duckdb"
        exit 1
    fi
    
    if [ ! -f "$DB_PATH" ]; then
        echo "Error: 데이터베이스 파일을 찾을 수 없습니다: $DB_PATH"
        exit 1
    fi
}

upload_code() {
    print_header "코드 업로드 중..."
    
    echo "프로젝트 루트: $PROJECT_ROOT"
    echo "원격 디렉토리: $REMOTE_DIR"
    echo ""
    
    # 임시 디렉토리에 업로드할 파일 준비
    TEMP_DIR=$(mktemp -d)
    echo "임시 디렉토리: $TEMP_DIR"
    
    # 필요한 파일만 복사
    echo "필요한 파일 복사 중..."
    mkdir -p "$TEMP_DIR/ai_trader"
    cp -r "$PROJECT_ROOT/ai_trader/embedding" "$TEMP_DIR/ai_trader/"
    cp "$PROJECT_ROOT/ai_trader/__init__.py" "$TEMP_DIR/ai_trader/" 2>/dev/null || touch "$TEMP_DIR/ai_trader/__init__.py"
    
    # requirements.txt 복사
    cp "$PROJECT_ROOT/requirements.txt" "$TEMP_DIR/"
    
    # 압축
    echo "파일 압축 중..."
    cd "$TEMP_DIR"
    tar -czf code.tar.gz ai_trader requirements.txt
    
    # 업로드
    echo "GCP 인스턴스로 업로드 중..."
    gcloud compute scp code.tar.gz \
        "$INSTANCE_NAME:$REMOTE_DIR/" \
        --project="$PROJECT_ID" \
        --zone="$ZONE"
    
    # 원격에서 압축 해제
    echo "원격에서 압축 해제 중..."
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="cd $REMOTE_DIR && tar -xzf code.tar.gz && rm code.tar.gz"
    
    # 임시 파일 정리
    rm -rf "$TEMP_DIR"
    
    echo "✓ 코드 업로드 완료!"
}

upload_database() {
    print_header "데이터베이스 업로드 중..."
    
    echo "로컬 DB: $DB_PATH"
    echo "원격 DB: $REMOTE_DB_PATH"
    echo ""
    
    # 데이터베이스 파일 크기 확인
    DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
    echo "데이터베이스 크기: $DB_SIZE"
    echo "업로드에 시간이 걸릴 수 있습니다..."
    echo ""
    
    # 업로드
    gcloud compute scp "$DB_PATH" \
        "$INSTANCE_NAME:$REMOTE_DB_PATH" \
        --project="$PROJECT_ID" \
        --zone="$ZONE"
    
    echo "✓ 데이터베이스 업로드 완료!"
}

install_dependencies() {
    print_header "의존성 설치 중..."
    
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="cd $REMOTE_DIR && pip3 install -r requirements.txt"
    
    echo "✓ 의존성 설치 완료!"
}

start_training() {
    print_header "훈련 시작"
    
    echo "훈련 설정:"
    echo "  데이터베이스: $REMOTE_DB_PATH"
    echo "  테이블: $TABLE_NAME"
    echo "  출력 디렉토리: $OUTPUT_DIR"
    echo "  시퀀스 길이: $SEQ_LEN"
    echo "  임베딩 차원: $EMBEDDING_DIM"
    echo "  배치 크기: $BATCH_SIZE"
    echo "  에포크: $EPOCHS"
    echo "  학습률: $LEARNING_RATE"
    echo ""
    
    # tmux 세션에서 훈련 실행 (백그라운드)
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="bash -s" << REMOTE_SCRIPT

set -e

# tmux 세션 생성 및 훈련 시작
tmux new-session -d -s training "
    cd $REMOTE_DIR && \
    export PYTHONPATH=\$PYTHONPATH:$REMOTE_DIR && \
    python3 -m ai_trader.embedding.train_embedding \
        --db $REMOTE_DB_PATH \
        --table $TABLE_NAME \
        --out $OUTPUT_DIR \
        --seq-len $SEQ_LEN \
        --embedding-dim $EMBEDDING_DIM \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --device cuda \
        2>&1 | tee training.log
"

echo ""
echo "✓ 훈련이 tmux 세션 'training'에서 시작되었습니다!"
echo ""
echo "훈련 모니터링 방법:"
echo "  1. SSH 접속: gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
echo "  2. tmux 세션 연결: tmux attach -t training"
echo "  3. 세션 나가기: Ctrl+B, D"
echo ""
echo "로그 확인:"
echo "  tail -f $REMOTE_DIR/training.log"
echo ""
echo "TensorBoard 실행 (로컬에서):"
echo "  gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE -- -L 6006:localhost:6006"
echo "  (원격) tensorboard --logdir=$OUTPUT_DIR/tensorboard_logs"
echo "  (로컬 브라우저) http://localhost:6006"

REMOTE_SCRIPT

    echo ""
    echo "✓ 훈련이 시작되었습니다!"
}

download_results() {
    print_header "결과 다운로드"
    
    echo "훈련이 완료된 후 이 명령어로 결과를 다운로드하세요:"
    echo ""
    echo "  gcloud compute scp --recurse \\"
    echo "    $INSTANCE_NAME:$OUTPUT_DIR \\"
    echo "    ./models/ \\"
    echo "    --project=$PROJECT_ID \\"
    echo "    --zone=$ZONE"
    echo ""
}

# =============================================================================
# 메인 실행
# =============================================================================

main() {
    print_header "훈련 코드 업로드 및 실행"
    
    echo "인스턴스: $INSTANCE_NAME"
    echo "프로젝트: $PROJECT_ID"
    echo "Zone: $ZONE"
    echo ""
    
    # 데이터베이스 확인
    check_database
    
    # 코드 업로드
    upload_code
    
    # 데이터베이스 업로드
    upload_database
    
    # 의존성 설치
    install_dependencies
    
    # 훈련 시작
    start_training
    
    # 결과 다운로드 안내
    download_results
    
    echo ""
    print_header "모든 작업 완료"
    echo "훈련이 백그라운드에서 실행 중입니다."
    echo "위의 안내를 참고하여 훈련을 모니터링하세요."
}

main "$@"
