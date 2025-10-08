# Stock Bot

## 목차

- [데이터 정규화](#데이터-정규화)

```bash
날짜 번호 종목코드 종목명 시간 등락률 누적거래대금 거래회전율 체결강도 매도대기금액1 매도대기금액2 매도대기금액3 매도대기금액4 매도대기금액5 매도대기금액6 매도대기금액7 매도대기금액8 매도대기금액9 매도대기금액10 매수대기금액1 매수대기금액2 매수대기금액3 매수대기금액4 매수대기금액5 매수대기금액6 매수대기금액7 매수대기금액8 매수대기금액9 매수대기금액10 종목명_scalar 시간_sin 시간_cos 시간_scalar
```

## 데이터 정규화

```bash
python scripts/generate_datasets.py \
  "D:\Workspace\Project\stock-bot\hoga-crawler\data" \
  -o "C:\Users\user\Workspace\datasets@raw\datasets.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --checkpoint-interval 50

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets@raw" \
  --out "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets@raw\tmp" \
  --threads 4 \
  --memory-limit 64GB

python scripts/normalize_datasets.py \
  "C:\Users\user\Workspace\datasets@raw\datasets_all.duckdb" \
  -o "C:\Users\user\Workspace\datasets\datasets_norm.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --checkpoint-interval 50 \
  --ignoring-stocks-csv scripts/ignoring_stocks.csv

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets" \
  --out "C:\Users\user\Workspace\datasets\datasets_norm_all.duckdb" \
  --temp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --threads 4 \
  --memory-limit 64GB
```
