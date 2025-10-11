"""
병렬 데이터 전처리 스크립트

멀티프로세싱을 사용하여 대용량 데이터를 빠르게 전처리합니다.
월별/분기별 데이터를 병렬로 처리하여 시간을 단축시킵니다.
"""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def process_month_data(args_tuple: Tuple) -> dict:
    """
    단일 월 데이터 처리
    
    Args:
        args_tuple: (db_path, year, month, output_base_dir, seq_len, batch_size, max_samples)
    
    Returns:
        처리 결과 딕셔너리
    """
    db_path, year, month, output_base_dir, seq_len, batch_size, max_samples = args_tuple
    
    # 월별 출력 디렉토리
    output_dir = f"{output_base_dir}_{year}_{month:02d}"
    
    # 날짜 범위 계산
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    try:
        # 전처리 명령어 구성
        cmd = [
            "python", "scripts/preprocess_for_autoencoder.py",
            "--db", db_path,
            "--output-dir", output_dir,
            "--seq-len", str(seq_len),
            "--batch-size", str(batch_size),
            "--start-date", start_date_str,
            "--end-date", end_date_str,
            "--compute-norm-params",
            "--create-batches"
        ]
        
        if max_samples:
            cmd.extend(["--max-samples", str(max_samples)])
        
        logger.info(f"Processing {year}-{month:02d}: {start_date_str} ~ {end_date_str}")
        
        # 프로세스 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                'year': year,
                'month': month,
                'status': 'success',
                'output_dir': output_dir,
                'message': f"Successfully processed {year}-{month:02d}"
            }
        else:
            return {
                'year': year,
                'month': month,
                'status': 'error',
                'output_dir': output_dir,
                'message': f"Error: {result.stderr}"
            }
    
    except Exception as e:
        return {
            'year': year,
            'month': month,
            'status': 'error',
            'output_dir': output_dir,
            'message': f"Exception: {str(e)}"
        }


def parallel_preprocess_months(
    db_path: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    output_base_dir: str,
    seq_len: int = 60,
    batch_size: int = 10000,
    max_samples: int = None,
    max_workers: int = None
) -> List[dict]:
    """
    여러 월 데이터를 병렬로 전처리
    
    Args:
        db_path: 데이터베이스 경로
        start_year: 시작 연도
        start_month: 시작 월
        end_year: 종료 연도
        end_month: 종료 월
        output_base_dir: 출력 기본 디렉토리
        seq_len: 시퀀스 길이
        batch_size: 배치 크기
        max_samples: 월별 최대 샘플 수
        max_workers: 최대 워커 수
    
    Returns:
        처리 결과 리스트
    """
    # 월 범위 생성
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    
    months = []
    while current <= end:
        months.append((current.year, current.month))
        # 다음 달로 이동
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    logger.info(f"Processing {len(months)} months in parallel: {months}")
    
    # 워커 수 결정
    if max_workers is None:
        max_workers = min(len(months), mp.cpu_count() - 1)
    
    logger.info(f"Using {max_workers} parallel workers")
    
    # 작업 인수 준비
    task_args = [
        (db_path, year, month, output_base_dir, seq_len, batch_size, max_samples)
        for year, month in months
    ]
    
    # 병렬 처리 실행
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 작업 제출
        future_to_month = {
            executor.submit(process_month_data, args): (args[1], args[2])
            for args in task_args
        }
        
        # 결과 수집
        for future in as_completed(future_to_month):
            year, month = future_to_month[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"✓ {result['message']}")
                else:
                    logger.error(f"✗ {result['message']}")
                    
            except Exception as e:
                error_result = {
                    'year': year,
                    'month': month,
                    'status': 'error',
                    'message': f"Future exception: {str(e)}"
                }
                results.append(error_result)
                logger.error(f"✗ {error_result['message']}")
    
    return results


def merge_preprocessed_batches(
    input_dirs: List[str],
    output_dir: str,
    max_workers: int = 4
):
    """
    여러 전처리된 디렉토리의 배치를 병합
    
    Args:
        input_dirs: 입력 디렉토리 리스트
        output_dir: 출력 디렉토리
        max_workers: 최대 워커 수
    """
    import h5py
    import json
    from tqdm import tqdm
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Merging {len(input_dirs)} preprocessed directories")
    
    # 모든 배치 파일 수집
    all_batch_files = []
    total_sequences = 0
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        
        # 배치 정보 로드
        with open(input_path / 'batch_info.json', 'r') as f:
            batch_info = json.load(f)
        
        # 배치 파일 추가
        for i in range(batch_info['num_batches']):
            batch_file = input_path / f'batch_{i:06d}.h5'
            if batch_file.exists():
                all_batch_files.append(batch_file)
        
        total_sequences += batch_info['total_sequences']
    
    logger.info(f"Found {len(all_batch_files)} batch files, {total_sequences} total sequences")
    
    # 배치 파일 복사 및 재번호
    merged_batch_count = 0
    
    for i, batch_file in enumerate(tqdm(all_batch_files, desc="Merging batches")):
        # 새 파일명
        new_batch_file = output_path / f'batch_{merged_batch_count:06d}.h5'
        
        # 배치 데이터 복사
        with h5py.File(batch_file, 'r') as src:
            with h5py.File(new_batch_file, 'w') as dst:
                # 데이터셋 복사
                for key in src.keys():
                    src.copy(key, dst)
        
        merged_batch_count += 1
    
    # 병합된 배치 정보 저장
    merged_batch_info = {
        'num_batches': merged_batch_count,
        'batch_size': 10000,  # 기본값
        'seq_len': 60,        # 기본값
        'total_sequences': total_sequences
    }
    
    with open(output_path / 'batch_info.json', 'w') as f:
        json.dump(merged_batch_info, f, indent=2)
    
    logger.info(f"Merged preprocessing completed: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='병렬 데이터 전처리',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:

  # 9월부터 12월까지 병렬 전처리 (4개 프로세스)
  python scripts/parallel_preprocessing.py \\
    --db "datasets_norm_all.duckdb" \\
    --start-year 2024 --start-month 9 \\
    --end-year 2024 --end-month 12 \\
    --output preprocessed_data \\
    --max-workers 4

  # 전처리된 월별 데이터 병합
  python scripts/parallel_preprocessing.py \\
    --merge-only \\
    --input-dirs preprocessed_data_2024_09 preprocessed_data_2024_10 \\
    --output preprocessed_data_merged
        """
    )
    
    parser.add_argument('--db', help='DuckDB 데이터베이스 경로')
    parser.add_argument('--output', required=True, help='출력 기본 디렉토리')
    
    parser.add_argument('--start-year', type=int, help='시작 연도')
    parser.add_argument('--start-month', type=int, help='시작 월')
    parser.add_argument('--end-year', type=int, help='종료 연도')
    parser.add_argument('--end-month', type=int, help='종료 월')
    
    parser.add_argument('--seq-len', type=int, default=60, help='시퀀스 길이')
    parser.add_argument('--batch-size', type=int, default=10000, help='배치 크기')
    parser.add_argument('--max-samples', type=int, help='월별 최대 샘플 수')
    parser.add_argument('--max-workers', type=int, help='최대 워커 수')
    
    # 병합 전용 모드
    parser.add_argument('--merge-only', action='store_true', help='병합만 수행')
    parser.add_argument('--input-dirs', nargs='+', help='병합할 입력 디렉토리들')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.merge_only:
        if not args.input_dirs:
            print("오류: --merge-only 모드에서는 --input-dirs가 필요합니다")
            sys.exit(1)
        
        merge_preprocessed_batches(
            input_dirs=args.input_dirs,
            output_dir=args.output,
            max_workers=args.max_workers or 4
        )
    else:
        if not all([args.db, args.start_year, args.start_month, args.end_year, args.end_month]):
            print("오류: 전처리 모드에서는 --db, --start-year, --start-month, --end-year, --end-month가 필요합니다")
            sys.exit(1)
        
        results = parallel_preprocess_months(
            db_path=args.db,
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
            output_base_dir=args.output,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_workers=args.max_workers
        )
        
        # 결과 요약
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = len(results) - success_count
        
        print(f"\n{'='*80}")
        print(f"병렬 전처리 완료!")
        print(f"성공: {success_count}개월, 실패: {error_count}개월")
        print(f"{'='*80}")
        
        if error_count > 0:
            print("\n실패한 월:")
            for result in results:
                if result['status'] == 'error':
                    print(f"  - {result['year']}-{result['month']:02d}: {result['message']}")


if __name__ == '__main__':
    main()