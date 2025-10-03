# Stock Bot

### Features
```bash
날짜, 등락률, 누적거래대금, 거래회전율, 체결강도, 매도호가수량1, 매도호가수량2, 매도호가수량3, 매도호가수량4, 매도호가수량5, 매도호가수량6, 매도호가수량7, 매도호가수량8, 매도호가수량9, 매도호가수량10, 매수호가수량1, 매수호가수량2, 매수호가수량3, 매수호가수량4, 매수호가수량5, 매수호가수량6, 매수호가수량7, 매수호가수량8, 매수호가수량9, 매수호가수량10, 매도호가총잔량, 매수호가총잔량, 매도거래원수량1, 매도거래원수량2, 매도거래원수량3, 매도거래원수량4, 매도거래원수량5, 매도거래원별증감1, 매도거래원별증감2, 매도거래원별증감3, 매도거래원별증감4, 매도거래원별증감5, 매수거래원수량1, 매수거래원수량2, 매수거래원수량3, 매수거래원수량4, 매수거래원수량5, 매수거래원별증감1, 매수거래원별증감2, 매수거래원별증감3, 매수거래원별증감4, 매수거래원별증감5, 매도거래원1_scalar, 매도거래원2_scalar, 매도거래원3_scalar, 매도거래원4_scalar, 매도거래원5_scalar, 매수거래원1_scalar, 매수거래원2_scalar, 매수거래원3_scalar, 매수거래원4_scalar, 매수거래원5_scalar, 종목명_scalar, 시간_scalar, 시간_sin, 시간_cos 
```

```bash
날짜, 등락률, 누적거래대금, 거래회전율, 체결강도, 매도호가수량1, 매도호가수량2, 매도호가수량3, 매도호가수량4, 매도호가수량5, 매도호가수량6, 매도호가수량7, 매도호가수량8, 매도호가수량9, 매도호가수량10, 매도호가직전대비1, 매도호가직전대비2, 매도호가직전대비3, 매도호가직전대비4, 매도호가직전대비5, 매도호가직전대비6, 매도호가직전대비7, 매도호가직전대비8, 매도호가직전대비9, 매도호가직전대비10, 매수호가수량1, 매수호가수량2, 매수호가수량3, 매수호가수량4, 매수호가수량5, 매수호가수량6, 매수호가수량7, 매수호가수량8, 매수호가수량9, 매수호가수량10, 매수호가직전대비1, 매수호가직전대비2, 매수호가직전대비3, 매수호가직전대비4, 매수호가직전대비5, 매수호가직전대비6, 매수호가직전대비7, 매수호가직전대비8, 매수호가직전대비9, 매수호가직전대비10, 매도호가총잔량, 매도호가총잔량직전대비, 매수호가총잔량, 매수호가총잔량직전대비, 매도거래원1, 매도거래원2, 매도거래원3, 매도거래원4, 매도거래원5, 매도거래원수량1, 매도거래원수량2, 매도거래원수량3, 매도거래원수량4, 매도거래원수량5, 매도거래원별증감1, 매도거래원별증감2, 매도거래원별증감3, 매도거래원별증감4, 매도거래원별증감5, 매수거래원1, 매수거래원2, 매수거래원3, 매수거래원4, 매수거래원5, 매수거래원수량1, 매수거래원수량2, 매수거래원수량3, 매수거래원수량4, 매수거래원수량5, 매수거래원별증감1, 매수거래원별증감2, 매수거래원별증감3, 매수거래원별증감4, 매수거래원별증감5, 외국계매수추정합변동, 외국계매도추정합변동, 외국계매수추정합, 외국계매도추정합, 매도거래원1_scalar, 매도거래원2_scalar, 매도거래원3_scalar, 매도거래원4_scalar, 매도거래원5_scalar, 매수거래원1_scalar, 매수거래원2_scalar, 매수거래원3_scalar, 매수거래원4_scalar, 매수거래원5_scalar, 종목명_scalar, 시간_scalar, 시간_sin, 시간_cos
```

### 정규화
```bash
python scripts/generate_datasets.py \
  "D:\Workspace\Project\stock-bot\hoga-crawler\data" \
  -o "C:\Users\user\Workspace\datasets\datasets.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --checkpoint-interval 50

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets" --out "C:\Users\user\Workspace\datasets\datasets_all.duckdb" --temp-dir "C:\Users\user\Workspace\datasets\tmp" --threads 2 --memory-limit 64GB

python scripts/normalize_datasets.py \
  "C:\Users\user\Workspace\datasets\datasets_all.duckdb" \
  -o "C:\Users\user\Workspace\datasets\datasets_norm.duckdb" \
  --workers 12 \
  --tmp-dir "C:\Users\user\Workspace\datasets\tmp" \
  --checkpoint-interval 50 \
  --ignoring-stocks-csv scripts/ignoring_stocks.csv

python scripts/merge_datasets.py "C:\Users\user\Workspace\datasets" --out "C:\Users\user\Workspace\datasets\datasets_norm_all.duckdb" --temp-dir "C:\Users\user\Workspace\datasets\tmp" --threads 2 --memory-limit 64GB

@train_supervised.py 에서 폴더 지정시 해당 폴더 안의 모든 duckdb 파일을 로드하고, 다음 컬럼을 features 로 사용하되 "날짜"는 month만 (예. 20240902 일때 09 만) 이용해서 훈련할 수 있도록 수정.
```

### 훈련
- 러닝레이트 2e-4 ~ 5e-4
- 훈련 방식
    - “상승/하락 분류” 같은 분류 문제라면 → CrossEntropyLoss가 무조건 더 좋습니다.
        - direction3-threshold → 0.015~0.02
    - “가격 수치 예측” 같은 회귀 문제라면 → Huber가 더 낫습니다.

##### 분류(Focal, CrossEntropy)방식
```bash
python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251002\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised_sample_test \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 500 \
  --device cuda \
  --batch-size 64 \
  --epochs 8 \
  --lr 3e-5 \
  --weight-decay 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised_sample_test/checkpoints \
  --target-col 현재가 \
  --code 000100 \
  --date 20241128 \
  \
  --aux-task direction3 \
  --loss focal \
  --focal-gamma 2.0 \
  --use-weighted-sampler \
  --direction3-threshold 0.015

python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251002\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --target-col 현재가 \
  \
  --aux-task direction3 \
  --loss focal \
  --focal-gamma 2.0 \
  --use-weighted-sampler \
  --direction3-threshold 0.015 \
  \
  --auto-diagnosis abort \
  --diag-warmup-chunks 10 \
  --diag-warmup-epochs 1 \
  --diag-min-val-samples 512 \
  --diag-require-consecutive 2

python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251002\datasets_norm_all.duckdb" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 20 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --target-col 현재가 \
  \
  --aux-task direction3 \
  --loss ce \
  --label-smoothing 0.05 \
  --use-weighted-sampler \
  --direction3-threshold 0.015 \
  \
  --auto-diagnosis abort \
  --diag-warmup-chunks 50 \
  --diag-warmup-epochs 1 \
  --diag-min-val-samples 4096 \
  --diag-require-consecutive 2
```

##### 회기방식
```bash
python -m ai_trader.ml.train_supervised \
  --db "C:\Users\user\Workspace\datasets@20251002" \
  --table datasets \
  --out models/supervised \
  --seq-len 60 \
  --horizon 10 \
  --chunk-size 50000 \
  --device cuda \
  --batch-size 128 \
  --epochs 10 \
  --lr 1e-4 \
  --progress-every 100 \
  --ckpt-every-chunks 50 \
  --resume-from models/supervised/checkpoints \
  --target-col 등락률 \
  --aux-task regression \
  --loss huber \
  --huber-delta 1.0
```

### 로스 보기
```bash
$ tensorboard --logdir models/supervised_sample_test/tensorboard_logs/

$ python scripts/check_tensorboard.py \
  --logdir "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/tensorboard_logs" \
  --watch --interval 10 --idle-seconds 180 --verbose \
  --plot-last 400 \
  --save-png "D:/Workspace/Project/stock-bot/stock-bot2/models/supervised/plots" \
  --out-html ./train_result-2025-10-03.html
```

### 트러블슈팅
- GPU OOM 발생 시: batch-size를 단계적으로 낮추세요(256 → 128 → 64).
- CPU/RAM 압박 시: chunk-size를 10~20k 단위로 줄이세요.
- 학습 속도: chunk-size를 키우면 I/O 오버헤드가 줄고 학습이 다소 빨라질 수 있으나, RAM 여유를 꼭 확인하세요.
