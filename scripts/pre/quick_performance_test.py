"""
빠른 성능 테스트 스크립트

실제 환경에서 전처리 최적화의 효과를 빠르게 확인할 수 있는 테스트입니다.
작은 데이터셋으로 기본 방식 vs 최적화 방식을 비교합니다.
"""

import time
import tempfile
import shutil
from pathlib import Path
import logging
import sys
import argparse
import json
import numpy as np
import psutil

# 프로젝트 모듈 import
try:
    from ai_trader.embedding.data import AutoEncoderDataLoader
    from ai_trader.embedding.fast_data_loader import FastAutoEncoderDataLoader
    from scripts.preprocess_for_autoencoder import AutoEncoderPreprocessor
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("프로젝트 루트 디렉토리에서 실행해주세요")
    sys.exit(1)

logger = logging.getLogger(__name__)


def measure_performance(func, *args, **kwargs):
    """함수 실행 시간과 메모리 사용량 측정"""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'success': success,
        'result': result,
        'error': error,
        'elapsed_time': end_time - start_time,
        'memory_used': end_memory - start_memory,
        'peak_memory': end_memory
    }


def test_basic_loading(db_path: str, max_samples: int = 10000):
    """기본 방식 데이터 로딩 테스트"""
    print("🔄 기본 방식 (DuckDB 직접 로딩) 테스트 중...")
    
    def basic_load():
        # 데이터 로더 생성
        loader = AutoEncoderDataLoader(
            db_path=db_path,
            seq_len=60,
            max_samples=max_samples,
            skip_clipping=True  # 빠른 테스트를 위해 클리핑 건너뛰기
        )
        
        # 데이터 로드 및 분할
        loader.load_and_split_data()
        
        # 훈련 데이터셋 생성
        train_dataset = loader.get_dataset('train', stride=1)
        
        # 샘플 로딩 테스트
        sample_count = min(1000, len(train_dataset))
        samples = []
        
        for i in range(sample_count):
            sample = train_dataset[i]
            samples.append(sample)
        
        return {
            'dataset_size': len(train_dataset),
            'samples_loaded': len(samples),
            'sample_shape': samples[0].shape if samples else None
        }
    
    return measure_performance(basic_load)


def test_preprocessed_loading(db_path: str, max_samples: int = 10000):
    """전처리 방식 데이터 로딩 테스트"""
    print("🚀 전처리 방식 테스트 중...")
    
    # 임시 디렉토리 생성
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 1단계: 전처리 수행
        def preprocess_data():
            preprocessor = AutoEncoderPreprocessor(db_path, str(temp_dir))
            preprocessor.connect()
            
            try:
                # 정규화 파라미터 계산
                preprocessor.compute_normalization_params(sample_size=min(max_samples, 10000))
                
                # 배치 생성
                batch_info = preprocessor.create_sequence_batches(
                    seq_len=60,
                    stride=1,
                    batch_size=1000,
                    max_samples=max_samples
                )
                
                return batch_info
            finally:
                preprocessor.close()
        
        preprocess_result = measure_performance(preprocess_data)
        
        if not preprocess_result['success']:
            return {
                'preprocess': preprocess_result,
                'loading': {'success': False, 'error': '전처리 실패'}
            }
        
        # 2단계: 전처리된 데이터 로딩
        def preprocessed_load():
            # 고속 로더 생성
            fast_loader = FastAutoEncoderDataLoader(
                preprocessed_dir=str(temp_dir),
                use_cached_batches=True
            )
            
            # 훈련 데이터셋 생성
            train_dataset = fast_loader.get_dataset('train')
            
            # 샘플 로딩 테스트
            sample_count = min(1000, len(train_dataset))
            samples = []
            
            for i in range(sample_count):
                sample = train_dataset[i]
                samples.append(sample)
            
            return {
                'dataset_size': len(train_dataset),
                'samples_loaded': len(samples),
                'sample_shape': samples[0].shape if samples else None
            }
        
        loading_result = measure_performance(preprocessed_load)
        
        return {
            'preprocess': preprocess_result,
            'loading': loading_result
        }
    
    finally:
        # 임시 디렉토리 정리
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_dataloader_performance(db_path: str, max_samples: int = 5000):
    """DataLoader 성능 비교 테스트"""
    print("⚡ DataLoader 성능 비교 테스트 중...")
    
    # 전처리 수행
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 전처리
        preprocessor = AutoEncoderPreprocessor(db_path, str(temp_dir))
        preprocessor.connect()
        
        try:
            preprocessor.compute_normalization_params(sample_size=5000)
            preprocessor.create_sequence_batches(
                seq_len=60,
                stride=1,
                batch_size=500,
                max_samples=max_samples
            )
        finally:
            preprocessor.close()
        
        # 1. 기본 DataLoader 테스트
        def basic_dataloader_test():
            loader = AutoEncoderDataLoader(
                db_path=db_path,
                seq_len=60,
                max_samples=max_samples,
                skip_clipping=True
            )
            loader.load_and_split_data()
            
            dataloader = loader.get_dataloader(
                split='train',
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            batch_count = 0
            total_samples = 0
            
            for batch in dataloader:
                batch_count += 1
                total_samples += batch.shape[0]
                if batch_count >= 10:  # 10 배치만 테스트
                    break
            
            return {'batch_count': batch_count, 'total_samples': total_samples}
        
        basic_result = measure_performance(basic_dataloader_test)
        
        # 2. 고속 DataLoader 테스트
        def fast_dataloader_test():
            fast_loader = FastAutoEncoderDataLoader(
                preprocessed_dir=str(temp_dir),
                use_cached_batches=True
            )
            
            dataloader = fast_loader.get_dataloader(
                split='train',
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            batch_count = 0
            total_samples = 0
            
            for batch in dataloader:
                batch_count += 1
                total_samples += batch.shape[0]
                if batch_count >= 10:  # 10 배치만 테스트
                    break
            
            return {'batch_count': batch_count, 'total_samples': total_samples}
        
        fast_result = measure_performance(fast_dataloader_test)
        
        return {
            'basic': basic_result,
            'fast': fast_result
        }
    
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def print_results(results: dict):
    """결과 출력"""
    print("\n" + "="*80)
    print("🎯 성능 테스트 결과")
    print("="*80)
    
    # 1. 데이터 로딩 비교
    if 'basic_loading' in results and 'preprocessed_loading' in results:
        basic = results['basic_loading']
        preprocessed = results['preprocessed_loading']
        
        print("\n📊 데이터 로딩 성능 비교:")
        
        if basic['success']:
            basic_time = basic['elapsed_time']
            basic_memory = basic['memory_used']
            basic_samples = basic['result']['samples_loaded']
            basic_throughput = basic_samples / basic_time if basic_time > 0 else 0
            
            print(f"  기본 방식:")
            print(f"    - 시간: {basic_time:.2f}초")
            print(f"    - 메모리: {basic_memory:.1f}MB")
            print(f"    - 처리량: {basic_throughput:.0f} samples/sec")
        else:
            print(f"  기본 방식: 실패 - {basic['error']}")
        
        if preprocessed['loading']['success']:
            # 전처리 시간 포함
            preprocess_time = preprocessed['preprocess']['elapsed_time']
            loading_time = preprocessed['loading']['elapsed_time']
            total_time = preprocess_time + loading_time
            
            loading_memory = preprocessed['loading']['memory_used']
            loading_samples = preprocessed['loading']['result']['samples_loaded']
            loading_throughput = loading_samples / loading_time if loading_time > 0 else 0
            
            print(f"  전처리 방식:")
            print(f"    - 전처리 시간: {preprocess_time:.2f}초 (1회만)")
            print(f"    - 로딩 시간: {loading_time:.2f}초")
            print(f"    - 총 시간: {total_time:.2f}초")
            print(f"    - 메모리: {loading_memory:.1f}MB")
            print(f"    - 처리량: {loading_throughput:.0f} samples/sec")
            
            # 성능 개선도 계산
            if basic['success']:
                speedup = basic_time / loading_time
                memory_reduction = (basic_memory - loading_memory) / basic_memory * 100 if basic_memory > 0 else 0
                
                print(f"\n  📈 개선도:")
                print(f"    - 로딩 속도: {speedup:.1f}배 빠름")
                print(f"    - 메모리 절약: {memory_reduction:.1f}%")
                
                # 전처리 비용 회수 계산
                if loading_time > 0:
                    break_even = preprocess_time / (basic_time - loading_time)
                    if break_even > 0:
                        print(f"    - {break_even:.0f}회 이상 사용 시 전처리 비용 회수")
        else:
            print(f"  전처리 방식: 실패 - {preprocessed['loading']['error']}")
    
    # 2. DataLoader 성능 비교
    if 'dataloader' in results:
        dataloader_results = results['dataloader']
        
        print(f"\n⚡ DataLoader 성능 비교:")
        
        if dataloader_results['basic']['success'] and dataloader_results['fast']['success']:
            basic_dl = dataloader_results['basic']
            fast_dl = dataloader_results['fast']
            
            basic_time = basic_dl['elapsed_time']
            fast_time = fast_dl['elapsed_time']
            
            basic_samples = basic_dl['result']['total_samples']
            fast_samples = fast_dl['result']['total_samples']
            
            basic_throughput = basic_samples / basic_time if basic_time > 0 else 0
            fast_throughput = fast_samples / fast_time if fast_time > 0 else 0
            
            print(f"  기본 DataLoader: {basic_throughput:.0f} samples/sec ({basic_time:.2f}초)")
            print(f"  고속 DataLoader: {fast_throughput:.0f} samples/sec ({fast_time:.2f}초)")
            
            if basic_time > 0 and fast_time > 0:
                speedup = basic_time / fast_time
                print(f"  성능 개선: {speedup:.1f}배 빠름")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='빠른 성능 테스트')
    
    parser.add_argument(
        '--db',
        required=True,
        help='DuckDB 데이터베이스 경로'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10000,
        help='테스트할 최대 샘플 수 (기본값: 10000)'
    )
    
    parser.add_argument(
        '--skip-basic',
        action='store_true',
        help='기본 방식 테스트 건너뛰기'
    )
    
    parser.add_argument(
        '--skip-preprocessed',
        action='store_true',
        help='전처리 방식 테스트 건너뛰기'
    )
    
    parser.add_argument(
        '--skip-dataloader',
        action='store_true',
        help='DataLoader 테스트 건너뛰기'
    )
    
    parser.add_argument(
        '--output',
        help='결과를 JSON 파일로 저장할 경로'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 AutoEncoder 전처리 성능 테스트 시작")
    print(f"데이터베이스: {args.db}")
    print(f"테스트 샘플 수: {args.samples:,}")
    
    results = {}
    
    try:
        # 1. 기본 방식 테스트
        if not args.skip_basic:
            results['basic_loading'] = test_basic_loading(args.db, args.samples)
        
        # 2. 전처리 방식 테스트
        if not args.skip_preprocessed:
            results['preprocessed_loading'] = test_preprocessed_loading(args.db, args.samples)
        
        # 3. DataLoader 성능 테스트
        if not args.skip_dataloader:
            results['dataloader'] = test_dataloader_performance(args.db, args.samples // 2)
        
        # 결과 출력
        print_results(results)
        
        # JSON 파일로 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📄 결과 저장: {args.output}")
        
        print("\n✅ 성능 테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        logger.exception("상세 오류:")
        sys.exit(1)


if __name__ == '__main__':
    main()