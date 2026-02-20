#!/usr/bin/env python3
"""
DuckDB 데이터를 Parquet 임베딩 파일에 병합하는 스크립트

현재 Parquet 컬럼: date, time, code, embedding
DuckDB 컬럼: 날짜, 번호, 종목코드, 종목명, 시간, 종목명_scalar, 시간_sin, 시간_cos,
             시간_scalar, 등락률, 누적거래대금, 거래회전율, 체결강도,
             매도대기금액1~10, 매수대기금액1~10

결과: 각 parquet 파일에 DuckDB 컬럼이 추가되어 덮어쓰기
Join key: parquet.date == DuckDB.날짜, parquet.code == DuckDB.종목코드, parquet.time == DuckDB.시간

사용법:
  python scripts/data/merge_embeddings_with_duckdb.py \\
    --parquet_dir "C:/Users/user/Workspace/datasets@20260117/embeddings_v2" \\
    --db_path "C:/Users/user/Workspace/datasets@20260117/datasets_raw_09_11.duckdb" \\
    --table_name datasets \\
    --workers 4
"""

import os
import sys
import glob
import logging
import argparse
import time
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import duckdb

# ──────────────── 로깅 ────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('merge_embeddings.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

DONE_SUFFIX = '.merged'  # 완료된 파일 표시용 마커


def get_done_marker(parquet_path: str) -> str:
    return parquet_path + DONE_SUFFIX


def is_done(parquet_path: str) -> bool:
    return os.path.exists(get_done_marker(parquet_path))


def mark_done(parquet_path: str):
    Path(get_done_marker(parquet_path)).touch()


def merge_one_file(args_tuple):
    """
    단일 parquet 파일에 DuckDB 데이터를 병합.
    ProcessPoolExecutor와 호환되도록 튜플 인자 형태 사용.
    """
    parquet_path, db_path, table_name, output_dir = args_tuple

    file_name = os.path.basename(parquet_path)
    out_path = os.path.join(output_dir, file_name)

    # 완료 마커 확인 (output_dir 기준)
    done_marker = out_path + DONE_SUFFIX
    if os.path.exists(done_marker):
        return ('skipped', parquet_path, 0)

    t0 = time.time()
    try:
        # 1. Parquet 로드
        pq_df = pd.read_parquet(parquet_path)
        if pq_df.empty:
            return ('empty', parquet_path, 0)

        # parquet의 고유 (date, code) 조합 파악 → DuckDB 쿼리 범위 최소화
        unique_pairs = pq_df[['date', 'code']].drop_duplicates()
        dates = unique_pairs['date'].unique().tolist()
        codes = unique_pairs['code'].unique().tolist()

        # 2. DuckDB에서 해당 날짜/종목만 조회 (IN 절 사용)
        dates_str = ', '.join(f"'{d}'" for d in dates)
        codes_str = ', '.join(f"'{c}'" for c in codes)

        query = f"""
            SELECT
                날짜 AS date,
                종목코드 AS code,
                시간 AS time,
                번호,
                종목명,
                종목명_scalar,
                시간_sin,
                시간_cos,
                시간_scalar,
                등락률,
                누적거래대금,
                거래회전율,
                체결강도,
                매도대기금액1, 매도대기금액2, 매도대기금액3, 매도대기금액4, 매도대기금액5,
                매도대기금액6, 매도대기금액7, 매도대기금액8, 매도대기금액9, 매도대기금액10,
                매수대기금액1, 매수대기금액2, 매수대기금액3, 매수대기금액4, 매수대기금액5,
                매수대기금액6, 매수대기금액7, 매수대기금액8, 매수대기금액9, 매수대기금액10
            FROM {table_name}
            WHERE 날짜 IN ({dates_str})
              AND 종목코드 IN ({codes_str})
        """

        # 프로세스마다 별도 연결 (read_only)
        conn = duckdb.connect(database=db_path, read_only=True)
        db_df = conn.execute(query).fetchdf()
        conn.close()

        if db_df.empty:
            logger.warning(f"[WARN] No DuckDB data for {file_name}")
            return ('no_db_data', parquet_path, 0)

        # 3. Merge on (date, code, time)
        merged = pq_df.merge(db_df, on=['date', 'code', 'time'], how='left')

        matched = merged['등락률'].notna().sum()
        total = len(merged)
        match_rate = matched / total * 100 if total > 0 else 0

        if match_rate < 50:
            logger.warning(
                f"[WARN] Low match rate {match_rate:.1f}% in {file_name} "
                f"({matched}/{total}). 시간 포맷 불일치 가능성."
            )

        # 4. 출력 경로에 저장 (parquet snappy 압축)
        os.makedirs(output_dir, exist_ok=True)
        merged.to_parquet(out_path, index=False, compression='snappy')

        # 완료 마커 생성
        Path(done_marker).touch()

        elapsed = time.time() - t0
        return ('ok', parquet_path, elapsed, matched, total)

    except Exception as e:
        return ('error', parquet_path, str(e))


def main():
    parser = argparse.ArgumentParser(description='DuckDB → Parquet 병합')
    parser.add_argument(
        '--parquet_dir',
        default=r'C:\Users\user\Workspace\datasets@20260117\embeddings_v2',
        help='원본 parquet 폴더 경로'
    )
    parser.add_argument(
        '--db_path',
        default=r'C:\Users\user\Workspace\datasets@20260117\datasets_raw_09_11.duckdb',
        help='DuckDB 파일 경로'
    )
    parser.add_argument('--table_name', default='datasets', help='DuckDB 테이블명')
    parser.add_argument(
        '--output_dir',
        default=None,
        help='출력 디렉토리. 기본값: parquet_dir 과 동일 (덮어쓰기)'
    )
    parser.add_argument('--workers', type=int, default=4, help='병렬 프로세스 수')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='이미 완료된 파일도 재처리'
    )
    args = parser.parse_args()

    parquet_dir = args.parquet_dir
    output_dir = args.output_dir if args.output_dir else parquet_dir
    db_path = args.db_path
    table_name = args.table_name

    # DuckDB 존재 확인
    if not os.path.exists(db_path):
        logger.error(f"DuckDB not found: {db_path}")
        sys.exit(1)

    # Parquet 파일 목록
    pq_files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
    if not pq_files:
        logger.error(f"No parquet files in {parquet_dir}")
        sys.exit(1)

    logger.info(f"파일 수: {len(pq_files)}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"병렬 Workers: {args.workers}")

    # 완료된 파일 제외
    if not args.overwrite:
        pending = []
        for f in pq_files:
            out_path = os.path.join(output_dir, os.path.basename(f))
            done_marker = out_path + DONE_SUFFIX
            if not os.path.exists(done_marker):
                pending.append(f)
        skipped_count = len(pq_files) - len(pending)
        logger.info(f"이미 완료: {skipped_count}개 (건너뜀), 처리 대상: {len(pending)}개")
        pq_files = pending
    else:
        logger.info("--overwrite: 모든 파일 재처리")

    if not pq_files:
        logger.info("처리할 파일이 없습니다. 완료!")
        return

    # 작업 인자 리스트
    tasks = [(f, db_path, table_name, output_dir) for f in pq_files]

    # 병렬 실행
    ok_count = error_count = 0
    total_rows = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(merge_one_file, t): t[0] for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result[0]
            path = result[1]
            fname = os.path.basename(path)

            if status == 'ok':
                _, _, elapsed, matched, total = result
                ok_count += 1
                total_rows += total
                eta_per_file = (time.time() - t_start) / i
                remaining = (len(tasks) - i) * eta_per_file
                logger.info(
                    f"[{i}/{len(tasks)}] OK {fname} "
                    f"({matched}/{total} matched, {elapsed:.1f}s) "
                    f"ETA: {remaining/60:.1f}m"
                )
            elif status == 'skipped':
                ok_count += 1
                logger.debug(f"[{i}/{len(tasks)}] SKIP {fname}")
            elif status == 'error':
                error_count += 1
                logger.error(f"[{i}/{len(tasks)}] ERROR {fname}: {result[2]}")
            else:
                logger.warning(f"[{i}/{len(tasks)}] {status.upper()} {fname}")

    total_elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"완료: {ok_count}개 성공, {error_count}개 실패")
    logger.info(f"총 행수: {total_rows:,}")
    logger.info(f"총 소요 시간: {total_elapsed/60:.1f}분")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
