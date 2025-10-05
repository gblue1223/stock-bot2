"""
임베딩 모델 훈련 스크립트

대조 학습을 사용하여 매매 데이터로부터 임베딩 모델을 훈련합니다.

사용 예시:
    python -m ai_trader.embedding.train_embedding \
        --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \
        --table datasets \
        --out models/embedding \
        --seq-len 60 \
        --embedding-dim 128 \
        --batch-size 128 \
        --epochs 50 \
        --lr 1e-4 \
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    """CLI 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='임베딩 모델 훈련 - 대조 학습을 통한 매매 데이터 임베딩',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 데이터 관련 인수
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로 (예: datasets_norm_all.duckdb)'
    )
    
    parser.add_argument(
        '--table',
        type=str,
        required=True,
        help='데이터베이스 테이블명 (예: datasets)'
    )
    
    # 출력 관련 인수
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='모델 체크포인트 저장 디렉토리 (예: models/embedding)'
    )
    
    # 모델 하이퍼파라미터
    parser.add_argument(
        '--seq-len',
        type=int,
        default=60,
        help='입력 시퀀스 길이 (타임스텝 수)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='임베딩 벡터 차원'
    )
    
    # 훈련 하이퍼파라미터
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='배치 크기'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='훈련 에포크 수'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='학습률 (learning rate)'
    )
    
    # 디바이스 설정
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='훈련에 사용할 디바이스'
    )
    
    return parser.parse_args()


def validate_args(args):
    """인수 유효성 검증"""
    # 데이터베이스 파일 존재 확인
    if not os.path.exists(args.db):
        raise FileNotFoundError(f"데이터베이스 파일을 찾을 수 없습니다: {args.db}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 하이퍼파라미터 범위 검증
    if args.seq_len <= 0:
        raise ValueError(f"seq-len은 양수여야 합니다: {args.seq_len}")
    
    if args.embedding_dim <= 0:
        raise ValueError(f"embedding-dim은 양수여야 합니다: {args.embedding_dim}")
    
    if args.batch_size <= 0:
        raise ValueError(f"batch-size는 양수여야 합니다: {args.batch_size}")
    
    if args.epochs <= 0:
        raise ValueError(f"epochs는 양수여야 합니다: {args.epochs}")
    
    if args.lr <= 0:
        raise ValueError(f"lr은 양수여야 합니다: {args.lr}")
    
    # CUDA 사용 가능 여부 확인
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                print("경고: CUDA를 사용할 수 없습니다. CPU로 대체합니다.")
                args.device = 'cpu'
        except ImportError:
            print("경고: PyTorch를 가져올 수 없습니다. 나중에 확인됩니다.")
    
    return args


def main():
    """메인 훈련 함수"""
    # CLI 인수 파싱 및 검증
    args = parse_args()
    args = validate_args(args)
    
    # 설정 출력
    print("=" * 80)
    print("임베딩 모델 훈련 시작")
    print("=" * 80)
    print(f"데이터베이스: {args.db}")
    print(f"테이블: {args.table}")
    print(f"출력 디렉토리: {args.out}")
    print(f"시퀀스 길이: {args.seq_len}")
    print(f"임베딩 차원: {args.embedding_dim}")
    print(f"배치 크기: {args.batch_size}")
    print(f"에포크: {args.epochs}")
    print(f"학습률: {args.lr}")
    print(f"디바이스: {args.device}")
    print("=" * 80)
    
    # TODO: 실제 훈련 로직은 task 5.2에서 구현
    print("\n[TODO] 훈련 루프 구현 예정 (task 5.2)")
    print("현재는 CLI 인터페이스만 구현되었습니다.")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        sys.exit(1)
