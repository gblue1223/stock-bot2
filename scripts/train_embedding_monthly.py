"""
월별 임베딩 모델 훈련 스크립트

12개월 5억 개 데이터를 효율적으로 훈련하기 위한 자동화 스크립트
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta


def get_month_range(year: int, month: int):
    """월의 시작일과 종료일 반환"""
    start_date = datetime(year, month, 1)
    
    # 다음 달 1일 - 1일 = 이번 달 마지막 날
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")


def train_month(
    db_path: str,
    year: int,
    month: int,
    base_output_dir: str,
    resume_from: str = None,
    is_initial: bool = False,
    max_samples: int = 5000000
):
    """
    특정 월 데이터로 훈련
    
    Args:
        db_path: 데이터베이스 경로
        year: 연도
        month: 월
        base_output_dir: 기본 출력 디렉토리
        resume_from: 재개할 체크포인트 (선택사항)
        is_initial: 초기 훈련 여부
        max_samples: 최대 샘플 수
    """
    start_date, end_date = get_month_range(year, month)
    output_dir = f"{base_output_dir}_{year}_{month:02d}"
    
    # 기본 인수
    cmd = [
        "python", "-m", "ai_trader.embedding.train_embedding",
        "--db", db_path,
        "--table", "datasets",
        "--out", output_dir,
        "--seq-len", "60",
        "--embedding-dim", "128",
        "--batch-size", "256",
        "--device", "cuda",
        "--positive-time-threshold", "10",
        "--negative-time-threshold", "60",
        "--temperature", "0.07",
        "--num-heads", "4",
        "--val-every", "3",
        "--checkpoint-interval", "5",
        "--start-date", start_date,
        "--end-date", end_date,
        "--num-workers", "8",
        "--max-samples", str(max_samples)
    ]
    
    # 초기 훈련 vs Fine-tuning
    if is_initial:
        cmd.extend([
            "--epochs", "30",
            "--lr", "1e-4"
        ])
        print(f"\n{'='*80}")
        print(f"초기 훈련: {year}년 {month}월 ({start_date} ~ {end_date})")
        print(f"{'='*80}\n")
    else:
        cmd.extend([
            "--epochs", "15",
            "--lr", "5e-5"
        ])
        if resume_from:
            cmd.extend(["--resume", resume_from])
        print(f"\n{'='*80}")
        print(f"Fine-tuning: {year}년 {month}월 ({start_date} ~ {end_date})")
        if resume_from:
            print(f"이전 모델: {resume_from}")
        print(f"{'='*80}\n")
    
    # 훈련 실행
    print(f"명령어: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n오류: {year}년 {month}월 훈련 실패")
        sys.exit(1)
    
    print(f"\n✓ {year}년 {month}월 훈련 완료: {output_dir}")
    return output_dir


def train_quarterly(
    db_path: str,
    year: int,
    quarter: int,
    base_output_dir: str,
    max_samples: int = 15000000
):
    """
    분기별 전체 재훈련
    
    Args:
        db_path: 데이터베이스 경로
        year: 연도
        quarter: 분기 (1-4)
        base_output_dir: 기본 출력 디렉토리
        max_samples: 최대 샘플 수
    """
    # 분기별 월 범위
    quarter_months = {
        1: (1, 3),   # Q1: 1-3월
        2: (4, 6),   # Q2: 4-6월
        3: (7, 9),   # Q3: 7-9월
        4: (10, 12)  # Q4: 10-12월
    }
    
    start_month, end_month = quarter_months[quarter]
    start_date, _ = get_month_range(year, start_month)
    _, end_date = get_month_range(year, end_month)
    
    output_dir = f"{base_output_dir}_{year}_Q{quarter}"
    
    cmd = [
        "python", "-m", "ai_trader.embedding.train_embedding",
        "--db", db_path,
        "--table", "datasets",
        "--out", output_dir,
        "--seq-len", "60",
        "--embedding-dim", "128",
        "--batch-size", "256",
        "--epochs", "20",
        "--lr", "1e-4",
        "--device", "cuda",
        "--positive-time-threshold", "10",
        "--negative-time-threshold", "60",
        "--temperature", "0.07",
        "--num-heads", "4",
        "--val-every", "2",
        "--checkpoint-interval", "5",
        "--start-date", start_date,
        "--end-date", end_date,
        "--num-workers", "8",
        "--max-samples", str(max_samples)
    ]
    
    print(f"\n{'='*80}")
    print(f"분기별 재훈련: {year}년 Q{quarter} ({start_date} ~ {end_date})")
    print(f"{'='*80}\n")
    
    print(f"명령어: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n오류: {year}년 Q{quarter} 훈련 실패")
        sys.exit(1)
    
    print(f"\n✓ {year}년 Q{quarter} 훈련 완료: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='월별 임베딩 모델 훈련 자동화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:

  # 9월부터 12월까지 순차 훈련
  python scripts/train_embedding_monthly.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --start-year 2024 --start-month 9 \\
    --end-year 2024 --end-month 12 \\
    --output models/embedding

  # 특정 월만 훈련
  python scripts/train_embedding_monthly.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --start-year 2024 --start-month 10 \\
    --end-year 2024 --end-month 10 \\
    --output models/embedding \\
    --resume models/embedding_2024_09/checkpoint_epoch30.pt

  # 분기별 재훈련
  python scripts/train_embedding_monthly.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --quarterly \\
    --year 2024 --quarter 4 \\
    --output models/embedding
        """
    )
    
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/embedding',
        help='출력 디렉토리 (기본값: models/embedding)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        help='시작 연도 (예: 2024)'
    )
    
    parser.add_argument(
        '--start-month',
        type=int,
        help='시작 월 (1-12)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        help='종료 연도 (예: 2024)'
    )
    
    parser.add_argument(
        '--end-month',
        type=int,
        help='종료 월 (1-12)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='재개할 체크포인트 경로 (첫 달에만 적용)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000000,
        help='월별 최대 샘플 수 (기본값: 5000000)'
    )
    
    # 분기별 훈련 옵션
    parser.add_argument(
        '--quarterly',
        action='store_true',
        help='분기별 재훈련 모드'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='분기별 훈련 연도'
    )
    
    parser.add_argument(
        '--quarter',
        type=int,
        choices=[1, 2, 3, 4],
        help='분기 (1-4)'
    )
    
    args = parser.parse_args()
    
    # 분기별 훈련
    if args.quarterly:
        if not args.year or not args.quarter:
            print("오류: 분기별 훈련은 --year와 --quarter가 필요합니다")
            sys.exit(1)
        
        train_quarterly(
            db_path=args.db,
            year=args.year,
            quarter=args.quarter,
            base_output_dir=args.output,
            max_samples=args.max_samples * 3  # 3개월치
        )
        return
    
    # 월별 훈련
    if not all([args.start_year, args.start_month, args.end_year, args.end_month]):
        print("오류: 월별 훈련은 --start-year, --start-month, --end-year, --end-month가 필요합니다")
        sys.exit(1)
    
    # 월 범위 생성
    current = datetime(args.start_year, args.start_month, 1)
    end = datetime(args.end_year, args.end_month, 1)
    
    months = []
    while current <= end:
        months.append((current.year, current.month))
        # 다음 달로 이동
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    print(f"\n총 {len(months)}개월 훈련 예정: {months}\n")
    
    # 월별 순차 훈련
    prev_checkpoint = args.resume
    
    for idx, (year, month) in enumerate(months):
        is_initial = (idx == 0 and prev_checkpoint is None)
        
        output_dir = train_month(
            db_path=args.db,
            year=year,
            month=month,
            base_output_dir=args.output,
            resume_from=prev_checkpoint,
            is_initial=is_initial,
            max_samples=args.max_samples
        )
        
        # 다음 달을 위한 체크포인트 경로 설정
        if is_initial:
            prev_checkpoint = f"{output_dir}/checkpoint_epoch30.pt"
        else:
            prev_checkpoint = f"{output_dir}/checkpoint_epoch15.pt"
    
    print(f"\n{'='*80}")
    print(f"✓ 전체 훈련 완료!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
