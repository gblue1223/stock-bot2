"""
1년치 데이터 전처리 스크립트 (2024.09 ~ 2025.09)

월별로 순차적으로 전처리를 진행하여 메모리 효율성을 확보합니다.
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import json


def get_month_info(year: int, month: int):
    """월 정보 반환"""
    start_date = datetime(year, month, 1)
    
    # 다음 달 1일 - 1일 = 이번 달 마지막 날
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def preprocess_month(year: int, month: int, base_output_dir: str, db_path: str, max_samples: int = 10000000):
    """단일 월 전처리"""
    start_date, end_date = get_month_info(year, month)
    output_dir = f"{base_output_dir}\\{year}_{month:02d}"
    
    print(f"\n{'='*80}")
    print(f"📅 {year}년 {month}월 전처리 시작 ({start_date} ~ {end_date})")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"📊 최대 샘플: {max_samples:,}")
    print(f"{'='*80}")
    
    # 전처리 명령어
    cmd = [
        "python", "scripts/pre/preprocess_for_autoencoder.py",
        "--db", db_path,
        "--output-dir", output_dir,
        "--seq-len", "60",
        "--batch-size", "20000",
        "--start-date", start_date,
        "--end-date", end_date,
        "--max-samples", str(max_samples),
        "--create-batches"
    ]
    
    # 첫 번째 월에만 정규화 파라미터 계산
    if year == 2024 and month == 9:
        cmd.append("--compute-norm-params")
        print("🔧 정규화 파라미터 계산 포함")
    
    start_time = time.time()
    
    try:
        print(f"🚀 실행 명령어: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2시간 타임아웃
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {year}년 {month}월 전처리 완료!")
            print(f"⏱️ 소요 시간: {elapsed_time/60:.1f}분")
            
            # 결과 정보 확인
            batch_info_file = Path(output_dir) / 'batch_info.json'
            if batch_info_file.exists():
                with open(batch_info_file, 'r') as f:
                    batch_info = json.load(f)
                
                print(f"📊 결과:")
                print(f"  - 총 배치: {batch_info['num_batches']:,}")
                print(f"  - 총 시퀀스: {batch_info['total_sequences']:,}")
                print(f"  - 처리량: {batch_info['total_sequences']/elapsed_time:.0f} sequences/sec")
            
            return True, elapsed_time, batch_info.get('total_sequences', 0)
        else:
            print(f"❌ {year}년 {month}월 전처리 실패!")
            print(f"오류: {result.stderr}")
            return False, elapsed_time, 0
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {year}년 {month}월 전처리 타임아웃 (2시간 초과)")
        return False, 7200, 0
    except Exception as e:
        print(f"💥 {year}년 {month}월 전처리 중 예외 발생: {e}")
        return False, time.time() - start_time, 0


def main():
    # 설정
    db_path = "C:\\Users\\user\\Workspace\\datasets@20251005\\datasets_norm_all.duckdb"
    base_output_dir = "C:\\Users\\user\\Workspace\\datasets@20251005\\pre_training_data"
    
    # 월별 최대 샘플 수 (메모리 고려)
    max_samples_per_month = 15000000  # 1500만 샘플
    
    # 처리할 월 목록 생성 (2024.09 ~ 2025.09)
    months_to_process = []
    
    # 2024년 9-12월
    for month in range(9, 13):
        months_to_process.append((2024, month))
    
    # 2025년 1-9월
    for month in range(1, 10):
        months_to_process.append((2025, month))
    
    print("🚀 1년치 데이터 전처리 시작")
    print(f"📅 처리 기간: 2024년 9월 ~ 2025년 9월 ({len(months_to_process)}개월)")
    print(f"💾 데이터베이스: {db_path}")
    print(f"📁 출력 디렉토리: {base_output_dir}")
    print(f"📊 월별 최대 샘플: {max_samples_per_month:,}")
    
    # 결과 추적
    results = []
    total_start_time = time.time()
    total_sequences = 0
    successful_months = 0
    
    for i, (year, month) in enumerate(months_to_process, 1):
        print(f"\n🔄 진행률: {i}/{len(months_to_process)} ({i/len(months_to_process)*100:.1f}%)")
        
        success, elapsed_time, sequences = preprocess_month(
            year, month, base_output_dir, db_path, max_samples_per_month
        )
        
        results.append({
            'year': year,
            'month': month,
            'success': success,
            'elapsed_time': elapsed_time,
            'sequences': sequences
        })
        
        if success:
            successful_months += 1
            total_sequences += sequences
        
        # 중간 결과 출력
        print(f"\n📈 중간 결과:")
        print(f"  - 완료된 월: {successful_months}/{i}")
        print(f"  - 총 시퀀스: {total_sequences:,}")
        print(f"  - 평균 처리 시간: {sum(r['elapsed_time'] for r in results)/len(results)/60:.1f}분")
        
        # 잠시 대기 (시스템 안정화)
        if i < len(months_to_process):
            print("⏸️ 시스템 안정화를 위해 10초 대기...")
            time.sleep(10)
    
    # 최종 결과
    total_elapsed_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 1년치 데이터 전처리 완료!")
    print(f"{'='*80}")
    print(f"📊 최종 결과:")
    print(f"  - 성공한 월: {successful_months}/{len(months_to_process)}")
    print(f"  - 총 처리 시간: {total_elapsed_time/3600:.1f}시간")
    print(f"  - 총 시퀀스: {total_sequences:,}")
    print(f"  - 평균 처리량: {total_sequences/total_elapsed_time:.0f} sequences/sec")
    
    # 실패한 월 출력
    failed_months = [r for r in results if not r['success']]
    if failed_months:
        print(f"\n❌ 실패한 월:")
        for r in failed_months:
            print(f"  - {r['year']}-{r['month']:02d}: {r['elapsed_time']/60:.1f}분 후 실패")
    
    # 결과를 JSON으로 저장
    results_file = f"{base_output_dir}\\preprocessing_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_months': len(months_to_process),
                'successful_months': successful_months,
                'total_elapsed_time': total_elapsed_time,
                'total_sequences': total_sequences,
                'average_throughput': total_sequences/total_elapsed_time if total_elapsed_time > 0 else 0
            },
            'monthly_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 상세 결과 저장: {results_file}")
    print("✅ 전체 작업 완료!")


if __name__ == '__main__':
    main()