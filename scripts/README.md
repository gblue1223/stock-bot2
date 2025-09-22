# scripts

## normalize_datasets: 데이터셋 정규화

`scripts/normalize_datasets.py`를 사용하여 원본 CSV 데이터들을 하나의 DuckDB 테이블로 정규화합니다.

- 입력 폴더 아래의 CSV 파일들을 재귀적으로 탐색합니다.
- 파일명 패턴을 기반으로 종목코드/종목명/날짜별로 그룹화하고, 각 그룹에 `execution`, `orderbook`, `trader` 3종이 모두 있을 때만 처리합니다. (`after_hours`는 자동 제외)
- 세 CSV를 `번호` 기준으로 병합하고, 피처 정규화를 수행한 뒤, `datasets` 테이블에 저장합니다.

### 파일명 패턴
```
^(?P<code>\d{6})_(?P<name>.+?)_(?P<type>[^_]+)_(?P<date>\d{8})\.csv$
```
예: `005930_SAMSUNG_execution_20240115.csv`

### 사용법
```bash
python scripts/normalize_datasets.py <input_folder> \
  -o <output_db_path.duckdb> \
  [--skip-existing | --no-skip-existing] \
  [--compact-only]
```

### 인자 설명
- **input_folder**: CSV 파일들이 들어있는 루트 폴더(하위 폴더까지 재귀 탐색)
- **-o, --output**: 출력 DuckDB 파일 경로 (기본: `normalized.duckdb`)
- **--skip-existing**: 이미 DB에 해당 (`종목코드`, `날짜`) 그룹이 존재하면 스킵(기본 활성화)
- **--no-skip-existing**: 이미 존재하는 그룹도 다시 처리
- **--compact-only**: CSV를 읽지 않고 지정한 DuckDB에 대해 `PRAGMA optimize`/`PRAGMA checkpoint`만 수행

### 예시
- 기본 실행(DuckDB):
```bash
python scripts/normalize_datasets.py "D:\Workspace\Project\stock-bot\hoga-crawler\data" -o "C:\Users\user\Workspace\datasets\datasets.duckdb"

python scripts/normalize_datasets.py models/test_datasets -o "C:\Users\user\Workspace\datasets\test_datasets.duckdb"
```
- 이미 있는 그룹 스킵 해제:
```bash
python scripts/normalize_datasets.py "D:\...\data" -o "C:\...\datasets.duckdb" --no-skip-existing
```
- 유지보수 전용(입력 CSV 무시):
```bash
python scripts/normalize_datasets.py "D:\anything" -o "C:\path\to\datasets.duckdb" --compact-only
```

### 출력
- DuckDB 파일(`.duckdb`) 내 `datasets` 테이블 생성/갱신
- 컬럼: 기본 스키마 + 정규화/파생 피처(`*_scalar` 등) 포함
