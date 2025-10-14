"""
실제 데이터를 사용한 전처리 성능 벤치마크

실제 DuckDB 데이터베이스를 사용하여 전처리 스크립트들의 성능을 측정합니다.
다양한 데이터 크기와 설정에서 성능을 비교합니다.
"""

import argparse
import time
import psutil
import subprocess
import sys
from pathlib import Path
import json
import tempfile
import shutil
from typing import Dict, List, Tuple
import logging
import duckdb
import numpy as np

logger = logging.getLogger(__name__)


class RealDataBenchmark:
    """실제 데이터 성능 벤치마크"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.temp_dirs = []
        self.results = {}
        
        # 데이터베이스 연결 테스트
        try:
            conn = duckdb.connect(db_path, read_only=True)
            
            # 테이블 존재 확인
            tables = conn.execute("SHOW TABLES").fetchall()
            if not any('datasets' in str(table) for table in tables):
                raise ValueError("datasets 테이블을 찾을 수 없습니다")
            
            # 데이터 크기 확인
            count_result = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()
            self.total_rows = count_result[0] if count_result else 0
            
            # 날짜 범위 확인
            date_result = conn.execute("SELECT MIN(날짜), MAX(날짜) FROM datasets").fetchone()
            self.date_range = date_result if date_result else (None, None)
            
            conn.close()
            
            logger.info(f"데이터베이스 연결 성공: {db_path}")
            logger.info(f"총 행 수: {self.total_rows:,}")
            logger.info(f"날짜 범위: {self.date_range[0]} ~ {self.date_range[1]}")
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
    
    def cleanup(self):
        """임시 디렉토리 정리"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"임시 디렉토리 정리: {temp_dir}")
    
    def get_system_info(self) -> Dict:
        """시스템 정보 수집"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'memory_available': psutil.virtual_memory().available / 1024**3,  # GB
            'disk_free': shutil.disk_usage('.').free / 1024**3  # GB
        }
    
    def measure_process_performance(self, cmd: List[str], timeout: int = 3600) -> Dict:
        """프로세스 성능 측정"""
        logger.info(f"실행 명령어: {' '.join(cmd)}")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024**2  # MB
        
        try:
            # 프로세스 실행
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 메모리 사용량 모니터링
            max_memory = start_memory
            memory_samples = []
            
            while process.poll() is None:
                try:
                    current_memory = psutil.virtual_memory().used / 1024**2  # MB
                    memory_samples.append(current_memory - start_memory)
                    max_memory = max(max_memory, current_memory)
                    time.sleep(1)  # 1초마다 샘플링
                    
                    # 타임아웃 체크
                    if time.time() - start_time > timeout:
                        process.terminate()
                        raise TimeoutError(f"프로세스가 {timeout}초 내에 완료되지 않음")
                        
                except psutil.NoSuchProcess:
                    break
            
            # 프로세스 완료 대기
            stdout, stderr = process.communicate()
            end_time = time.time()
            
            # 결과 수집
            result = {
                'success': process.returncode == 0,
                'elapsed_time': end_time - start_time,
                'max_memory_used': max_memory - start_memory,
                'avg_memory_used': np.mean(memory_samples) if memory_samples else 0,
                'return_code': process.returncode,
                'stdout': stdout,
                'stderr': stderr
            }
            
            if result['success']:
                logger.info(f"✓ 완료: {result['elapsed_time']:.1f}초, 메모리: {result['max_memory_used']:.1f}MB")
            else:
                logger.error(f"✗ 실패: 반환코드 {result['return_code']}")
                if stderr:
                    logger.error(f"오류: {stderr[:500]}")
            
            return result
            
        except Exception as e:
            logger.error(f"프로세스 실행 중 오류: {e}")
            return {
                'success': False,
                'elapsed_time': time.time() - start_time,
                'max_memory_used': 0,
                'avg_memory_used': 0,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def benchmark_preprocessing(self, sample_sizes: List[int], date_ranges: List[Tuple[str, str]]) -> Dict:
        """전처리 성능 벤치마크"""
        results = {}
        
        for sample_size in sample_sizes:
            for start_date, end_date in date_ranges:
                test_name = f"preprocess_{sample_size}_{start_date}_{end_date}"
                logger.info(f"\n{'='*60}")
                logger.info(f"전처리 테스트: {test_name}")
                logger.info(f"샘플 크기: {sample_size:,}, 날짜: {start_date} ~ {end_date}")
                logger.info(f"{'='*60}")
                
                # 임시 출력 디렉토리
                temp_dir = Path(tempfile.mkdtemp())
                self.temp_dirs.append(temp_dir)
                
                # 전처리 명령어
                cmd = [
                    "python", "scripts/preprocess_for_autoencoder.py",
                    "--db", self.db_path,
                    "--output-dir", str(temp_dir),
                    "--seq-len", "60",
                    "--batch-size", "10000",
                    "--max-samples", str(sample_size),
                    "--compute-norm-params",
                    "--create-batches"
                ]
                
                if start_date and end_date:
                    cmd.extend(["--start-date", start_date, "--end-date", end_date])
                
                # 성능 측정
                result = self.measure_process_performance(cmd, timeout=1800)  # 30분 타임아웃
                
                # 결과 저장
                results[test_name] = {
                    'sample_size': sample_size,
                    'date_range': (start_date, end_date),
                    'performance': result
                }
                
                # 생성된 파일 정보 수집
                if result['success'] and temp_dir.exists():
                    batch_info_file = temp_dir / 'batch_info.json'
                    if batch_info_file.exists():
                        with open(batch_info_file, 'r') as f:
                            batch_info = json.load(f)
                        results[test_name]['batch_info'] = batch_info
                        
                        # 처리량 계산
                        if result['elapsed_time'] > 0:
                            throughput = batch_info['total_sequences'] / result['elapsed_time']
                            results[test_name]['throughput'] = throughput
                            logger.info(f"처리량: {throughput:.0f} sequences/sec")
        
        return results
    
    def benchmark_parallel_preprocessing(self, month_ranges: List[Tuple[int, int, int, int]], max_workers_list: List[int]) -> Dict:
        """병렬 전처리 성능 벤치마크"""
        results = {}
        
        for start_year, start_month, end_year, end_month in month_ranges:
            for max_workers in max_workers_list:
                test_name = f"parallel_{start_year}{start_month:02d}_{end_year}{end_month:02d}_w{max_workers}"
                logger.info(f"\n{'='*60}")
                logger.info(f"병렬 전처리 테스트: {test_name}")
                logger.info(f"기간: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d}, 워커: {max_workers}")
                logger.info(f"{'='*60}")
                
                # 임시 출력 디렉토리
                temp_dir = Path(tempfile.mkdtemp())
                self.temp_dirs.append(temp_dir)
                
                # 병렬 전처리 명령어
                cmd = [
                    "python", "scripts/parallel_preprocessing.py",
                    "--db", self.db_path,
                    "--start-year", str(start_year),
                    "--start-month", str(start_month),
                    "--end-year", str(end_year),
                    "--end-month", str(end_month),
                    "--output", str(temp_dir / "preprocessed"),
                    "--max-workers", str(max_workers),
                    "--max-samples", "1000000"  # 월당 100만 샘플로 제한
                ]
                
                # 성능 측정
                result = self.measure_process_performance(cmd, timeout=3600)  # 1시간 타임아웃
                
                # 결과 저장
                results[test_name] = {
                    'month_range': (start_year, start_month, end_year, end_month),
                    'max_workers': max_workers,
                    'performance': result
                }
                
                # 생성된 디렉토리 수 확인
                if result['success']:
                    preprocessed_dirs = list(temp_dir.glob("preprocessed_*"))
                    results[test_name]['output_dirs'] = len(preprocessed_dirs)
                    logger.info(f"생성된 디렉토리 수: {len(preprocessed_dirs)}")
        
        return results
    
    def benchmark_data_loading(self, preprocessed_dirs: List[str], batch_sizes: List[int]) -> Dict:
        """데이터 로딩 성능 벤치마크"""
        results = {}
        
        for preprocessed_dir in preprocessed_dirs:
            if not Path(preprocessed_dir).exists():
                logger.warning(f"전처리 디렉토리가 존재하지 않음: {preprocessed_dir}")
                continue
            
            for batch_size in batch_sizes:
                test_name = f"loading_{Path(preprocessed_dir).name}_bs{batch_size}"
                logger.info(f"\n데이터 로딩 테스트: {test_name}")
                
                # 로딩 테스트 스크립트 생성
                test_script = f"""
import time
import sys
sys.path.append('.')

from ai_trader.embedding.fast_data_loader import FastAutoEncoderDataLoader

try:
    # 고속 로더 생성
    fast_loader = FastAutoEncoderDataLoader(
        preprocessed_dir="{preprocessed_dir}",
        use_cached_batches=True
    )
    
    # DataLoader 생성
    dataloader = fast_loader.get_dataloader(
        split='train',
        batch_size={batch_size},
        shuffle=False,
        num_workers=0
    )
    
    # 로딩 테스트
    start_time = time.time()
    batch_count = 0
    total_samples = 0
    
    for batch in dataloader:
        batch_count += 1
        total_samples += batch.shape[0]
        if batch_count >= 50:  # 50 배치만 테스트
            break
    
    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
    
    print(f"SUCCESS,{{batch_count}},{{total_samples}},{{elapsed_time:.3f}},{{throughput:.0f}}")
    
except Exception as e:
    print(f"ERROR,{{str(e)}}")
"""
                
                # 임시 스크립트 파일 생성
                temp_script = Path(tempfile.mktemp(suffix='.py'))
                with open(temp_script, 'w') as f:
                    f.write(test_script)
                
                try:
                    # 스크립트 실행
                    cmd = ["python", str(temp_script)]
                    result = self.measure_process_performance(cmd, timeout=300)  # 5분 타임아웃
                    
                    # 결과 파싱
                    if result['success'] and result['stdout']:
                        output_line = result['stdout'].strip().split('\n')[-1]
                        if output_line.startswith('SUCCESS'):
                            parts = output_line.split(',')
                            batch_count = int(parts[1])
                            total_samples = int(parts[2])
                            elapsed_time = float(parts[3])
                            throughput = float(parts[4])
                            
                            results[test_name] = {
                                'preprocessed_dir': preprocessed_dir,
                                'batch_size': batch_size,
                                'batch_count': batch_count,
                                'total_samples': total_samples,
                                'elapsed_time': elapsed_time,
                                'throughput': throughput,
                                'memory_used': result['max_memory_used']
                            }
                            
                            logger.info(f"처리량: {throughput:.0f} samples/sec, 메모리: {result['max_memory_used']:.1f}MB")
                        else:
                            logger.error(f"스크립트 실행 오류: {output_line}")
                    else:
                        logger.error(f"데이터 로딩 테스트 실패: {result['stderr']}")
                
                finally:
                    # 임시 스크립트 삭제
                    if temp_script.exists():
                        temp_script.unlink()
        
        return results
    
    def generate_report(self, all_results: Dict):
        """성능 테스트 보고서 생성"""
        report = {
            'system_info': self.get_system_info(),
            'database_info': {
                'path': self.db_path,
                'total_rows': self.total_rows,
                'date_range': self.date_range
            },
            'results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # JSON 보고서 저장
        report_file = f"benchmark_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"성능 보고서 저장: {report_file}")
        
        # 요약 출력
        print("\n" + "="*80)
        print("성능 벤치마크 요약")
        print("="*80)
        
        # 시스템 정보
        sys_info = report['system_info']
        print(f"시스템: CPU {sys_info['cpu_count']}코어, RAM {sys_info['memory_total']:.1f}GB")
        print(f"데이터베이스: {self.total_rows:,}행, {self.date_range[0]} ~ {self.date_range[1]}")
        
        # 전처리 성능
        if 'preprocessing' in all_results:
            print(f"\n📊 전처리 성능:")
            for test_name, result in all_results['preprocessing'].items():
                if result['performance']['success']:
                    throughput = result.get('throughput', 0)
                    elapsed = result['performance']['elapsed_time']
                    memory = result['performance']['max_memory_used']
                    print(f"  {test_name}: {throughput:.0f} seq/sec, {elapsed:.1f}초, {memory:.1f}MB")
        
        # 병렬 처리 성능
        if 'parallel_preprocessing' in all_results:
            print(f"\n🚀 병렬 처리 성능:")
            for test_name, result in all_results['parallel_preprocessing'].items():
                if result['performance']['success']:
                    elapsed = result['performance']['elapsed_time']
                    workers = result['max_workers']
                    dirs = result.get('output_dirs', 0)
                    print(f"  {test_name}: {elapsed:.1f}초, {workers}워커, {dirs}개 디렉토리")
        
        # 데이터 로딩 성능
        if 'data_loading' in all_results:
            print(f"\n⚡ 데이터 로딩 성능:")
            for test_name, result in all_results['data_loading'].items():
                throughput = result['throughput']
                batch_size = result['batch_size']
                memory = result['memory_used']
                print(f"  배치크기 {batch_size}: {throughput:.0f} samples/sec, {memory:.1f}MB")
        
        print("="*80)
        return report_file


def main():
    parser = argparse.ArgumentParser(
        description='실제 데이터를 사용한 전처리 성능 벤치마크',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예제:

  # 기본 벤치마크 (소규모)
  python scripts/benchmark_real_data.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --quick

  # 전체 벤치마크
  python scripts/benchmark_real_data.py \\
    --db "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb" \\
    --full

  # 커스텀 벤치마크
  python scripts/benchmark_real_data.py \\
    --db "datasets_norm_all.duckdb" \\
    --sample-sizes 10000,50000,100000 \\
    --date-ranges "2024-10-01,2024-10-07" "2024-11-01,2024-11-07" \\
    --batch-sizes 64,128,256,512
        """
    )
    
    parser.add_argument(
        '--db',
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='빠른 테스트 (소규모 데이터)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='전체 테스트 (대규모 데이터)'
    )
    
    parser.add_argument(
        '--sample-sizes',
        help='테스트할 샘플 크기 (쉼표 구분, 예: 10000,50000,100000)'
    )
    
    parser.add_argument(
        '--date-ranges',
        nargs='+',
        help='테스트할 날짜 범위 (예: "2024-10-01,2024-10-07")'
    )
    
    parser.add_argument(
        '--batch-sizes',
        help='테스트할 배치 크기 (쉼표 구분, 예: 64,128,256)'
    )
    
    parser.add_argument(
        '--max-workers',
        help='병렬 처리 워커 수 (쉼표 구분, 예: 2,4,8)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='전처리 테스트 건너뛰기'
    )
    
    parser.add_argument(
        '--skip-parallel',
        action='store_true',
        help='병렬 처리 테스트 건너뛰기'
    )
    
    parser.add_argument(
        '--skip-loading',
        action='store_true',
        help='데이터 로딩 테스트 건너뛰기'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 벤치마크 초기화
    try:
        benchmark = RealDataBenchmark(args.db)
    except Exception as e:
        logger.error(f"벤치마크 초기화 실패: {e}")
        sys.exit(1)
    
    try:
        all_results = {}
        
        # 테스트 설정
        if args.quick:
            sample_sizes = [10000, 50000]
            date_ranges = [("2024-10-01", "2024-10-07")]
            batch_sizes = [64, 256]
            max_workers_list = [2, 4]
            month_ranges = [(2024, 10, 2024, 10)]
        elif args.full:
            sample_sizes = [50000, 100000, 500000, 1000000]
            date_ranges = [
                ("2024-09-01", "2024-09-07"),
                ("2024-10-01", "2024-10-31"),
                ("2024-11-01", "2024-11-30")
            ]
            batch_sizes = [32, 64, 128, 256, 512, 1024]
            max_workers_list = [1, 2, 4, 8]
            month_ranges = [
                (2024, 9, 2024, 10),
                (2024, 10, 2024, 12)
            ]
        else:
            # 커스텀 설정
            sample_sizes = [int(x) for x in args.sample_sizes.split(',')] if args.sample_sizes else [50000]
            
            date_ranges = []
            if args.date_ranges:
                for dr in args.date_ranges:
                    start, end = dr.split(',')
                    date_ranges.append((start, end))
            else:
                date_ranges = [("2024-10-01", "2024-10-07")]
            
            batch_sizes = [int(x) for x in args.batch_sizes.split(',')] if args.batch_sizes else [64, 256]
            max_workers_list = [int(x) for x in args.max_workers.split(',')] if args.max_workers else [2, 4]
            month_ranges = [(2024, 10, 2024, 10)]
        
        # 1. 전처리 성능 테스트
        if not args.skip_preprocessing:
            logger.info("전처리 성능 테스트 시작...")
            all_results['preprocessing'] = benchmark.benchmark_preprocessing(sample_sizes, date_ranges)
        
        # 2. 병렬 처리 성능 테스트
        if not args.skip_parallel:
            logger.info("병렬 처리 성능 테스트 시작...")
            all_results['parallel_preprocessing'] = benchmark.benchmark_parallel_preprocessing(month_ranges, max_workers_list)
        
        # 3. 데이터 로딩 성능 테스트
        if not args.skip_loading:
            logger.info("데이터 로딩 성능 테스트 시작...")
            # 전처리된 디렉토리 찾기
            preprocessed_dirs = []
            for temp_dir in benchmark.temp_dirs:
                if temp_dir.exists():
                    # batch_info.json이 있는 디렉토리 찾기
                    if (temp_dir / 'batch_info.json').exists():
                        preprocessed_dirs.append(str(temp_dir))
                    
                    # 하위 디렉토리도 확인
                    for subdir in temp_dir.iterdir():
                        if subdir.is_dir() and (subdir / 'batch_info.json').exists():
                            preprocessed_dirs.append(str(subdir))
            
            if preprocessed_dirs:
                all_results['data_loading'] = benchmark.benchmark_data_loading(preprocessed_dirs[:2], batch_sizes)  # 최대 2개 디렉토리만
            else:
                logger.warning("전처리된 디렉토리를 찾을 수 없어 데이터 로딩 테스트를 건너뜁니다")
        
        # 보고서 생성
        report_file = benchmark.generate_report(all_results)
        print(f"\n📄 상세 보고서: {report_file}")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"벤치마크 실행 중 오류: {e}")
        raise
    finally:
        benchmark.cleanup()


if __name__ == '__main__':
    main()