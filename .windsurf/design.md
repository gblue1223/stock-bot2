# Stock-Bot 시스템 설계서 (design.md)

## 1. 아키텍처 개요
- 프로세스
  - Ingestor: `koapys` 시세 수신 → MarketDataBus(PyZMQ/asyncio Queue)
  - FeatureEngine: 초 단위 윈도우링/라그/롤링 계산 → Feature Store(Parquet/Feather)
  - Backtester: 오프라인 재생 엔진(틱/초)
  - Trainer: 모델별 학습/튜닝 파이프라인
  - Realtime Inference: 저지연 예측 서비스(모델 핫스왑)
  - Strategy/OMS: 시그널→포지션/리스크→주문(`koapys` 어댑터)
  - UI(Qt5): 대시보드/컨트롤/로그/알람
- 통신
  - 내부 이벤트 버스: in-memory Queue(멀티스레드), 필요 시 ZMQ
  - 메시지 포맷: pydantic dataclass(타임스탬프, 심볼, 타입 등)

## 2. 모듈 구조 (폴더 제안)
- `ai_trader/`
  - `data/`: loader, writer, schema, validators
  - `features/`: transformers, aggregators, encoders
  - `models/`: clmha/, lgbm/, transformer/, baseline/
  - `backtest/`: engine, simulator, metrics
  - `realtime/`: infer_server, throttler, recovery
  - `strategy/`: position_sizing, risk, execution
  - `ui/`: qt_app, widgets, views, viewmodels
  - `adapters/`: koapys_adapter, time_sync, config
  - `utils/`: logger, time, math, io

## 3. 데이터 스키마
- 원천 틱/호가: `timestamp`, `symbol`, `bid/ask price/size(levels)`, `last_price`, `last_size`, `agg_trade` 등
- 초 집계: `open/high/low/close(mid)`, `vwap`, `spread`, `imbalance`, `buy/sell pressure`, `volatility`
- Feature Store: 파티셔닝(`symbol`, `date`), Parquet + Arrow, 컬럼형 압축

## 4. 피처 엔지니어링
- 윈도우링: 5/10/30/60초 롤링 통계(평균/표준편차/변동성/RSI 변형)
- 마이크로마켓: 마이크로프라이스, Order Book Imbalance(OBI), Queue Imbalance, Trade Sign(이벤트 기반)
- 정규화: RobustScaler/Quantile 변환(오프라인 피팅 → 온라인 적용)
- 피처 선택: SHAP(LightGBM), Permutation, 누설 방지

## 5. 모델 설계
### 5.1 CNN+LSTM+Multi-Head Attention (CLMHA)
- 입력: X ∈ (B, T, F) → Conv1D(채널=F, 커널=3~5) 다중 스택 → LSTM(양방향 선택) → Multi-Head Attention(Q=K=V=LSTM 출력)
- 출력: 회귀(단기 수익률) 또는 분류(상/하/중립 Softmax)
- 손실: MSE 또는 Focal/CE, 비용민감(거래비용 반영)
- 정규화: Dropout, LayerNorm, EarlyStopping
- 추론: TorchScript/ONNX 변환, FP16, 배치=1 스트리밍

### 5.2 LightGBM
- 입력: 집계된 수치 피처(카테고리 X), 라그/차분/변형
- 목적함수: 회귀(L2/L1/HUBER) 또는 이진/다중 분류
- 튜닝: num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction, min_data_in_leaf
- 추론: 수 μs~ms 단위, 모델 스냅샷/버전 관리

### 5.3 Transformer
- 입력: 시퀀스 임베딩(Linear) + 포지셔널 인코딩(학습형 권장)
- 구조: N층 인코더 블록(MHA + FFN), Pre-LN, DropPath(선택)
- 출력: 마지막 토큰 또는 풀링, 회귀/분류 헤드
- 최적화: AdamW, Cosine LR, Label Smoothing, AMP

## 6. 학습/검증 전략
- 시계열 CV: expanding window, Purged K-Fold(TimeSeriesSplit + embargo)
- 메트릭: 회귀(RMSE, MAE), 분류(Logloss, AUC, Balanced Accuracy), 트레이딩(PnL, 샤프, MDD, Turnover, Cost)
- 데이터 분할: 날짜 기반(train/val/test), 심볼별 롤링 검증
- 드리프트 감지: PSI/JS Divergence, 성능 모니터링

## 7. 백테스트/시뮬레이션
- 이벤트 드리븐 엔진: 틱/초 이벤트, 지연/슬리피지/수수료 모형
- 체결 로직: 마켓/리밋/IOC/FOK, 큐 포지션 추정(간단화 모델)
- 재현성: 시드 관리, 로그/체결 리플레이, 결과 레포트

## 8. 실시간 추론/운영
- 파이프라인: MarketData → Feature → Predict → Signal → Risk → Order
- 지연 최적화: 고정 크기 버퍼, 전처리 numba, 멀티스레딩(Producer-Consumer), Zero-Copy
- 내결함성: 워치독, 상태머신, 자동 재연결, 콜드/웜 스타트
- 배포: 모델 핫스왑(버전 토글), 안전 스위치, 실시간 피처 동기화

## 9. Qt5 UI 설계
- 메인 뷰: 심볼/포지션/PNL/오더북/체결/전략상태/알람
- 상세: 전략 파라미터 편집, 모델 선택/버전, 리스크 한도 설정
- 기술: `PyQt5` + `pyqtgraph` 차트, 비동기 업데이트(qtimer, worker thread), ViewModel로 상태 분리

## 10. koapys 연동 설계
- 어댑터: 시세/주문/계좌 표준 인터페이스 정의(emit 이벤트, 재시도/backoff)
- 에러 처리: 인증/세션만료/거래시간 외/호가제한/체결거부 코드 표준화
- 속도 제한: 초당 주문 수 제한, 동일 가격/수량 드리프트 제한

## 11. 설정/로깅/모니터링
- 설정: yaml + pydantic 검증, 런타임 리로드(일부)
- 로깅: 구조화 로그(jsonlines), 레벨/채널 분리, 파일 롤링
- 모니터링: 전략/모델 KPI, 지연, 에러율, 체결 히트맵

## 12. 보안/권한/컴플라이언스
- API Key/계정 보호, 기록/감사 로그, 전략 비상 중지 버튼

## 13. 성능 목표 요약
- 추론 파이프라인 end-to-end 20ms 내외, 주문 ACK 50ms 내외, GC 튜닝/메모리 풀링
