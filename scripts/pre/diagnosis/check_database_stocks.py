"""
데이터베이스 내 종목 정보 확인 스크립트
"""

import sys
from pathlib import Path
import duckdb
import argparse

# 프로젝트 루트 추가
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_database_stocks(db_path: str, table_name: str = "datasets"):
    """데이터베이스 내 종목 정보 확인"""

    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print("=" * 60)

    try:
        conn = duckdb.connect(db_path, read_only=True)

        # 1. 전체 레코드 수
        total_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        print(f"\n총 레코드 수: {total_count:,}")

        # 2. 종목 수
        stock_count = conn.execute(
            f"SELECT COUNT(DISTINCT 종목코드) FROM {table_name}"
        ).fetchone()[0]
        print(f"종목 수: {stock_count:,}")

        # 3. 날짜 범위
        date_range = conn.execute(
            f"SELECT MIN(날짜) as min_date, MAX(날짜) as max_date FROM {table_name}"
        ).fetchone()
        print(f"날짜 범위: {date_range[0]} ~ {date_range[1]}")

        # 4. 상위 10개 종목 (레코드 수 기준)
        print(f"\n상위 10개 종목 (레코드 수 기준):")
        print("-" * 60)
        top_stocks = conn.execute(
            f"""
            SELECT 종목코드, 종목명, COUNT(*) as cnt
            FROM {table_name}
            GROUP BY 종목코드, 종목명
            ORDER BY cnt DESC
            LIMIT 10
            """
        ).fetchall()

        for i, (code, name, cnt) in enumerate(top_stocks, 1):
            print(f"{i:2d}. {code} ({name}): {cnt:,} records")

        # 5. 종목별 날짜 범위
        print(f"\n종목별 날짜 범위 (상위 5개):")
        print("-" * 60)
        stock_dates = conn.execute(
            f"""
            SELECT 
                종목코드, 
                종목명,
                MIN(날짜) as min_date, 
                MAX(날짜) as max_date,
                COUNT(DISTINCT 날짜) as date_count
            FROM {table_name}
            GROUP BY 종목코드, 종목명
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
        ).fetchall()

        for code, name, min_date, max_date, date_cnt in stock_dates:
            print(f"{code} ({name}): {min_date} ~ {max_date} ({date_cnt} days)")

        # 6. 특징 컬럼 정보
        print(f"\n특징 컬럼 정보:")
        print("-" * 60)
        columns = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        exclude_cols = {"날짜", "종목코드", "번호", "종목명", "시간"}
        feature_cols = [
            col for col in columns["column_name"].tolist() if col not in exclude_cols
        ]
        print(f"특징 컬럼 수: {len(feature_cols)}")
        print(f"특징 컬럼: {', '.join(feature_cols[:10])}...")

        # 7. 샘플 데이터 (첫 3개 레코드)
        print(f"\n샘플 데이터 (첫 3개 레코드):")
        print("-" * 60)
        sample = conn.execute(
            f"""
            SELECT 종목코드, 종목명, 날짜, 시간
            FROM {table_name}
            ORDER BY 날짜, 종목코드, 시간
            LIMIT 3
            """
        ).fetchall()

        for code, name, date, time in sample:
            print(f"{code} ({name}) - {date} {time}")

        conn.close()

        print("\n" + "=" * 60)
        print("✓ 데이터베이스 확인 완료")

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="데이터베이스 내 종목 정보 확인")
    parser.add_argument(
        "--db",
        type=str,
        default=r"C:\Users\user\Workspace\datasets@20251013\datasets_norm_all.duckdb",
        help="DuckDB 데이터베이스 경로",
    )
    parser.add_argument(
        "--table", type=str, default="datasets", help="테이블명 (기본값: datasets)"
    )

    args = parser.parse_args()

    check_database_stocks(args.db, args.table)


if __name__ == "__main__":
    main()
