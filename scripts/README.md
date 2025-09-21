# scripts

## normalize_datasets: 데이터셋 정규화

`scripts/normalize_datasets.py`를 사용하여 원본 CSV 데이터들을 하나의 SQLite 데이터베이스 테이블로 정규화합니다.

- 입력 폴더 아래의 CSV 파일들을 재귀적으로 탐색합니다.
- 파일명 패턴을 기반으로 종목코드/종목명/날짜별로 그룹화하고, 각 그룹에 `execution`, `orderbook`, `trader` 3종이 모두 있을 때만 처리합니다. (`after_hours`는 자동 제외)
- 세 CSV를 `번호` 기준으로 병합하고, 피처 정규화를 수행한 뒤, 기본키가 (`날짜`, `종목코드`, `번호`)인 SQLite `datasets` 테이블에 upsert합니다.

### 파일명 패턴
```
^(?P<code>\d{6})_(?P<name>.+?)_(?P<type>[^_]+)_(?P<date>\d{8})\.csv$
```
예: `005930_SAMSUNG_execution_20240115.csv`

### 사용법
```bash
python scripts/normalize_datasets.py <input_folder> \
  -o <output_db_path> \
  [--compact] \
  [--page-size 4096|8192|16384] \
  [--journal-mode DELETE|TRUNCATE|PERSIST|MEMORY|WAL|OFF] \
  [--synchronous OFF|NORMAL|FULL|EXTRA] \
  [--auto-vacuum none|full|incremental] \
  [--vacuum-into <compact_copy.db>]
```

### 인자 설명
- **input_folder**: CSV 파일들이 들어있는 루트 폴더(하위 폴더까지 재귀 탐색)
- **-o, --output**: 출력 SQLite DB 파일 경로 (기본: `normalized.db`)
- **--compact**: 실행 종료 시 `VACUUM`으로 DB 파일을 컴팩트하게 만듭니다
- **--page-size**: 페이지 크기(바이트). 테이블 생성 전에 설정하는 것을 권장합니다. 예: 4096, 8192, 16384 (기본: 4096)
- **--journal-mode**: SQLite 저널 모드 (기본: `WAL`)
- **--synchronous**: 동기화 수준 (기본: `NORMAL`)
- **--auto-vacuum**: 자동 VACUUM 모드 (기본: `full`)
- **--vacuum-into**: 지원 시 `VACUUM INTO`로 컴팩트 복사본 생성. 지정 시 `--compact`보다 우선합니다

### 예시
- 기본 실행:
```bash
python scripts/normalize_datasets.py "D:\Workspace\Project\stock-bot\hoga-crawler\data" -o models/datasets.db
```
- 대용량 처리 최적화:
```bash
python scripts/normalize_datasets.py "D:\Workspace\Project\stock-bot\hoga-crawler\data" -o models/datasets.db \
  --page-size 32768 --journal-mode WAL --synchronous NORMAL --auto-vacuum none \
  --vacuum-into models/datasets_compact.db
```
- 컴팩트화
```bash
python scripts/normalize_datasets.py "D:\anything" -o "C:\path\to\datasets.db" --compact-only --compact

python scripts/normalize_datasets.py "D:\anything" -o "C:\path\to\datasets.db" ^
  --compact-only --vacuum-into "C:\path\to\datasets_compact.db"
```

### 출력
- SQLite DB 테이블: `datasets`
- 기본키: (`날짜`, `종목코드`, `번호`)
- 컬럼: 기본 스키마 + 정규화/파생 피처(`*_scalar` 등) 포함
