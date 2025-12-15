# 구현 계획

- [x] 1. 프로젝트 구조 및 기본 인터페이스 설정

  - ai_trader/embedding/, ai_trader/grpo/, ai_trader/inference/ 디렉토리 생성
  - 각 모듈의 **init**.py 파일 생성
  - _요구사항: 6.1, 6.2_

- [x] 2. AutoEncoder 모델 아키텍처 구현

  - [x] 2.1 AutoEncoderEmbedding 클래스 구현

    - ai_trader/embedding/autoencoder_model.py에 기본 AutoEncoder 모델 정의
    - Transformer 기반 인코더/디코더 아키텍처
    - Positional Encoding 적용
    - 입력: (batch, seq_len, 60+), 출력: (batch, embedding_dim)
    - _요구사항: 1.1, 1.2, 1.4_

  - [x] 2.2 MaskedAutoEncoder 클래스 구현

    - 마스킹 기법을 적용한 자기지도 학습 모델
    - 설정 가능한 마스킹 비율 (기본값 15%)
    - 마스킹된 위치에서만 손실 계산
    - _요구사항: 5.3, 5.4_

- [x] 3. AutoEncoder 훈련 파이프라인 구현

  - [x] 3.1 AutoEncoderTrainer 클래스 구현

    - ai_trader/embedding/autoencoder_trainer.py에 훈련기 정의
    - 효율적인 데이터 로딩 및 배치 처리
    - 재구성 손실 계산 및 최적화
    - 체크포인트 저장 및 로딩 기능
    - _요구사항: 5.1, 5.2, 5.5, 5.6_

  - [x] 3.2 TimeSeriesDataset 클래스 구현

    - 시계열 데이터를 위한 효율적인 Dataset
    - 슬라이딩 윈도우 방식으로 시퀀스 생성
    - 설정 가능한 stride 파라미터
    - _요구사항: 5.1, 5.2_

- [x] 4. Fine-tuning 시스템 구현

  - [x] 4.1 FineTunedEmbedding 클래스 구현

    - ai_trader/embedding/fine_tuning.py에 Fine-tuning 모델 정의
    - 사전 훈련된 AutoEncoder + Task-specific Head
    - 차별적 학습률 적용 (인코더 vs 헤드)
    - _요구사항: 5.8_

  - [x] 4.2 TradingTaskHead 클래스 구현

    - 트레이딩 특화 태스크별 헤드 (분류/회귀/랭킹)
    - 설정 가능한 히든 차원 및 드롭아웃
    - _요구사항: 5.8_

- [x] 5. AutoEncoder 훈련 스크립트 구현

  - [x] 5.1 CLI 인터페이스 구현

    - examples/autoencoder_training_example.py 생성
    - argparse로 --db, --output-dir, --seq-len, --embedding-dim, --batch-size, --epochs, --model-type, --fine-tune, --device 인수 처리
    - _요구사항: 6.2_

  - [x] 5.2 훈련 루프 구현

    - 데이터 로더에서 배치 로드
    - AutoEncoder forward pass 및 재구성 손실 계산
    - Optimizer 업데이트 (AdamW)
    - 체크포인트 저장 (설정 가능한 간격)
    - _요구사항: 5.5, 5.6, 6.1_

  - [x] 5.3 Fine-tuning 파이프라인 구현

    - 사전 훈련된 모델 로드
    - 트레이딩 태스크별 라벨 생성
    - Fine-tuning 훈련 루프
    - 성능 메트릭 계산 (정확도, R² 등)
    - _요구사항: 5.7, 5.8_

- [x] 6. GRPO 환경 구현

  - [x] 6.1 GRPOScalpingEnv 클래스 기본 구조 구현

    - ai_trader/grpo/env.py에 Gymnasium 환경 정의
    - 관측 공간: Box(embedding_dim,)
    - 행동 공간: Discrete(3) - 보유/매수/매도
    - 임베딩 모델 로드 및 추론 통합
    - _요구사항: 3.1, 6.4_

  - [x] 6.2 보상 계산 로직 구현

    - 거래 비용 계산 (수수료 0.015% + 세금 0.2% = 0.215%, 왕복 0.43%)
    - 수익률 기반 보상: (청산가 - 진입가) / 진입가 - 거래비용
    - 양의 보상 조건: 수익률 > 0.43%
    - 장기 보유 페널티: -0.001 \* (보유시간 - 60초)
    - _요구사항: 3.2, 3.2.1, 3.2.2_

  - [x] 6.3 빠른 손절 룰 구현

    - 매수 후 설정 가능한 시간 임계값(기본값 1.5초) 체크
    - 임계값 내 가격 미상승 시 자동 매도 및 페널티 적용
    - 위반 횟수 카운터 및 메타데이터 기록
    - CLI 인수로 임계값 및 페널티 조정 가능
    - _요구사항: 3.3_

  - [x] 6.4 에피소드 관리 및 메타데이터 기록

    - reset() 메서드: DuckDB에서 다양한 시장 상황의 시작 지점 샘플링
    - step() 메서드: 행동 실행, 보상 계산, 다음 관측 반환
    - 에피소드 종료 시 메타데이터 기록 (총 수익, 거래 횟수, 평균 보유 시간, 샤프 비율, 빠른 손절 룰 위반 횟수)
    - _요구사항: 3.4, 3.5, 3.6, 3.7_

- [x] 7. GRPO 알고리즘 구현

  - [x] 7.1 GRPOTrainer 클래스 기본 구조 구현

    - ai_trader/grpo/grpo.py에 훈련기 정의
    - 정책, 환경, 하이퍼파라미터 초기화
    - _요구사항: 4.1_

  - [x] 7.2 롤아웃 수집 구현

    - 현재 정책으로 여러 에피소드 실행 (그룹당 8-16 에피소드)
    - 상태, 행동, 보상, 다음 상태 저장
    - 에피소드 메타데이터 수집
    - _요구사항: 4.1_

  - [x] 7.3 에피소드 그룹화 구현

    - 시장 체제 지표 추출 (변동성, 추세 강도)
    - K-means 클러스터링 또는 규칙 기반 그룹화
    - 그룹별 에피소드 분류
    - _요구사항: 4.2_

  - [x] 7.4 그룹 상대 어드밴티지 계산 구현

    - 각 그룹의 평균 수익 계산
    - 그룹 내 상대 어드밴티지: A_i = R_i - mean(R_group)
    - _요구사항: 4.3_

  - [x] 7.5 정책 업데이트 구현

    - PPO 스타일 클리핑된 목적 함수
    - KL 발산 제약 적용
    - Optimizer 업데이트
    - _요구사항: 4.4, 4.5_

  - [x] 7.6 TensorBoard 로깅 구현

    - 그룹 수준 메트릭 로깅 (그룹당 평균 수익, 정책 엔트로피, KL 발산)
    - 전체 메트릭 로깅 (평균 보상, 승률, 평균 보유 시간, 빠른 손절 룰 위반 횟수)
    - _요구사항: 4.6, 6.6_

- [x] 8. GRPO 정책 네트워크 구현

  - [x] 8.1 GRPOPolicy 클래스 구현

    - ai_trader/grpo/policy.py에 정책 네트워크 정의
    - 입력: 임베딩 벡터 (embedding_dim,)
    - 출력: 행동 확률 분포 (3,)
    - get_action() 메서드: deterministic 및 stochastic 모드 지원
    - _요구사항: 4.4_

- [x] 9. GRPO 훈련 스크립트 구현

  - [x] 9.1 CLI 인터페이스 구현

    - ai_trader/grpo/train_grpo.py 생성
    - argparse로 --embedding-model, --db, --table, --out, --seq-len, --episodes-per-group, --num-groups, --total-timesteps, --quick-exit-threshold, --quick-exit-penalty, --device 인수 처리
    - _요구사항: 6.2_

  - [x] 9.2 훈련 루프 구현

    - GRPOTrainer 초기화
    - 롤아웃 수집 → 그룹화 → 어드밴티지 계산 → 정책 업데이트 반복
    - 체크포인트 저장 (설정 가능한 간격)
    - _요구사항: 6.1, 6.6_

- [x] 10. 추론 파이프라인 구현

  - [x] 10.1 GRPOInference 클래스 구현

    - ai_trader/inference/grpo_infer.py에 추론 엔진 정의
    - 임베딩 모델 및 정책 로드
    - TorchScript 컴파일로 최적화
    - 임베딩 캐시 구현
    - _요구사항: 2.1, 2.2, 2.3, 2.4, 2.5, 6.5_

  - [x] 10.2 predict() 메서드 구현

    - 입력 시퀀스 정규화
    - 임베딩 생성 (캐시 활용)
    - 정책 실행 및 행동 반환
    - 목표 지연 시간: < 10ms
    - _요구사항: 1.5_

- [x] 11. 평가 및 백테스팅 구현

  - [x] 11.1 AutoEncoder 모델 평가 메트릭 구현

    - 재구성 정확도 계산
    - Fine-tuning 성능 메트릭 (정확도, R², 손실)
    - 임베딩 품질 평가 (다운스트림 태스크 성능)
    - _요구사항: 7.1_

  - [x] 11.2 GRPO 에이전트 평가 메트릭 구현

    - 승률, 거래당 평균 수익, 최대 낙폭, 샤프 비율, 평균 보유 시간 계산
    - _요구사항: 7.2_

  - [x] 11.3 백테스팅 시뮬레이터 구현

    - 홀드아웃 테스트 데이터에서 거래 시뮬레이션
    - 현실적인 거래 비용 및 슬리피지 적용
    - _요구사항: 7.3_

  - [x] 11.4 실시간 거래 로깅 구현

    - 모든 거래 기록 (타임스탬프, 진입/청산 가격, 보유 기간, 손익)
    - _요구사항: 7.4_

- - [x] 11.5 성능 알림 시스템 구현

    - 설정 가능한 임계값 기반 알림 (낙폭 > 5%, 승률 < 45%)
    - _요구사항: 7.5_

  - [x] 11.6 HTML 보고서 생성 구현

    - 훈련 진행 상황, 최종 메트릭, 샘플 거래 에피소드 요약
    - _요구사항: 7.6_

- [x] 12. 단위 테스트 작성

  - [x] 12.1 AutoEncoder 모델 테스트

    - tests/test_autoencoder_model.py 생성
    - AutoEncoder 출력 형태 검증
    - 마스킹 기법 동작 검증
    - 재구성 손실 계산 검증
    - Fine-tuning 모델 동작 검증
    - _요구사항: 1.1, 1.2, 1.3, 1.4_

  - [x] 12.2 GRPO 환경 테스트

    - tests/test_grpo_env.py 생성
    - 빠른 손절 룰 동작 검증
    - 보상 계산 검증
    - 설정 가능한 임계값 검증
    - _요구사항: 3.2, 3.3, 3.5, 3.6_

  - [x] 12.3 GRPO 알고리즘 테스트

    - tests/test_grpo.py 생성
    - 그룹화 로직 검증
    - 어드밴티지 계산 검증
    - _요구사항: 4.2, 4.3_

- [x] 13. 통합 테스트 작성

  - [x] 13.1 전체 파이프라인 테스트

    - tests/test_integration.py 생성
    - AutoEncoder 사전 훈련 → Fine-tuning → GRPO 훈련 → 추론 전체 흐름 검증
    - _요구사항: 6.3, 6.4, 6.5_

- [x] 14. 백테스팅 테스트 작성

  - [x] 14.1 성능 기준 검증

    - tests/test_backtest.py 생성
    - 승률 > 50%, 샤프 비율 > 1.0, 최대 낙폭 < 10%, 평균 보유 시간 < 30초 검증
    - _요구사항: 7.2, 7.3_

- [x] 15. 문서화 및 예제 작성

  - [x] 15.1 README 업데이트

    - AutoEncoder 및 Fine-tuning 훈련 예제 추가
    - GRPO 훈련 예제 추가
    - CLI 인수 설명 추가
    - 성능 벤치마크 결과 추가
    - _요구사항: 6.2_

  - [x] 15.2 사용 예제 스크립트 작성

    - examples/autoencoder_training_example.py: AutoEncoder 훈련 예제
    - scripts/benchmark_autoencoder.py: 성능 벤치마크 스크립트
    - _요구사항: 6.4, 6.5_

- [x] 16. 성능 최적화 및 벤치마킹

  - [x] 16.1 AutoEncoder vs 대조 학습 성능 비교

    - 훈련 속도 벤치마크 (10-100배 개선 검증)
    - 메모리 사용량 비교
    - 임베딩 품질 비교
    - _요구사항: 1.3, 5.4, 5.5_

  - [x] 16.2 대용량 데이터 처리 최적화

    - 점진적 학습 지원
    - 메모리 효율적인 데이터 로딩
    - 분산 학습 준비 (멀티 GPU 지원)
    - _요구사항: 5.1, 5.2, 5.6_
