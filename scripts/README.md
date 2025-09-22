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
  [--compact-only] \
  [--force-recreate] \
  [--workers N] \
  [--tmp-dir <path>] \
  [--checkpoint-interval N]
```

### 인자 설명
- **input_folder**: CSV 파일들이 들어있는 루트 폴더(하위 폴더까지 재귀 탐색)
- **-o, --output**: 출력 DuckDB 파일 경로 (기본: `normalized.duckdb`)
- **--skip-existing**: 이미 DB에 해당 (`종목코드`, `날짜`) 그룹이 존재하면 스킵(기본 활성화)
- **--no-skip-existing**: 이미 존재하는 그룹도 다시 처리
- **--compact-only**: CSV를 읽지 않고 지정한 DuckDB에 대해 `PRAGMA optimize`/`PRAGMA checkpoint`만 수행
- **--force-recreate**: 출력 DuckDB 파일이 존재하면 삭제 후 새로 생성 (파일 손상/버전 불일치 문제 해결용)
- **--workers N**: 병렬 처리에 사용할 프로세스 수. 기본값은 시스템 CPU 코어 수
- **--tmp-dir <path>**: 병렬 처리 중간 결과(피클) 저장 디렉토리. 기본은 `<output>.tmp` 폴더 생성
- **--checkpoint-interval N**: N개 그룹 처리마다 DuckDB `CHECKPOINT` 수행. `0`이면 비활성화 (기본: `100`)

### 예시
- 기본 실행(DuckDB):
```bash
python scripts/normalize_datasets.py "D:\Workspace\Project\stock-bot\hoga-crawler\data" -o "C:\Users\user\Workspace\datasets\datasets.duckdb" --workers 8 --tmp-dir "C:\Users\user\Workspace\datasets\tmp" --checkpoint-interval 50

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
- 병렬 처리로 속도 향상(예: 8 프로세스):
```bash
python scripts/normalize_datasets.py "D:\...\data" -o "C:\...\datasets.duckdb" --workers 8
```
- 임시 디렉토리 지정(여유 공간이 큰 드라이브 권장):
```bash
python scripts/normalize_datasets.py "D:\...\data" -o "E:\datasets.duckdb" --workers 8 --tmp-dir "E:\normalize_tmp"
```
- 손상된 DB 재생성을 강제:
```bash
python scripts/normalize_datasets.py "D:\...\data" -o "C:\...\datasets.duckdb" --force-recreate
```

### 동작 및 성능 노트
- **병렬 처리**: 그룹 단위로 CSV 병합과 정규화를 멀티프로세스로 처리하여 CPU/I/O 병목을 완화합니다.
- **즉시 반영과 임시 파일 정리**: 각 그룹의 처리가 끝나면 결과를 곧바로 DB에 반영하고, 해당 임시 피클 파일을 즉시 삭제하여 디스크 사용량을 최소화합니다.
- **주기적 CHECKPOINT**: 설정한 간격(기본 100)마다 자동 `CHECKPOINT`를 수행합니다. `--checkpoint-interval 0`이면 주기적 체크포인트를 비활성화합니다.
- **DB 쓰기 순차화**: DuckDB 파일 쓰기는 순차적으로 수행하여 동시 쓰기 경합을 방지합니다.
- **임시 파일**: 각 그룹 결과는 중간에 피클(`.pkl`)로 저장되며, 반영 후 삭제됩니다. 종료 시 임시 폴더도 정리됩니다.
- **메모리/디스크 고려**: 워커 수를 늘리면 메모리 사용량이 증가합니다. 대용량 데이터는 임시 폴더에 충분한 디스크 공간이 필요합니다.
- **Windows 호환**: 멀티프로세싱 워커는 모듈 최상위에 정의되어 Windows spawn 모드와 호환됩니다.

### 재시작/복구(Resume) 노트
- **temp 폴더 유지**: 실행 시작 시 temp 폴더가 이미 존재하면 삭제하지 않고 유지합니다.
- **미완료 파일 정리**: 이전 실행에서 남았을 수 있는 부분 파일(`*.pkl.part`)은 자동 삭제합니다.
- **남은 체크포인트 반영**: temp 폴더에 완료된 체크포인트(`*.pkl`)가 있으면, 새로운 작업을 시작하기 전에 전부 DuckDB에 병합합니다.
- **Ctrl+C 처리**: 중단 시에도 완료된 체크포인트는 최대한 반영되며, 다음 실행 시 temp 폴더의 `.pkl`이 자동 병합되어 작업을 이어갑니다.

### 출력
- DuckDB 파일(`.duckdb`) 내 `datasets` 테이블 생성/갱신
- 컬럼: 기본 스키마 + 정규화/파생 피처(`*_scalar` 등) 포함
