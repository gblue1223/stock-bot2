# GRPO 성능 알림 시스템 빠른 참조 가이드

## 개요

GRPO 성능 알림 시스템은 에이전트의 성능 메트릭을 모니터링하고 설정 가능한 임계값을 기반으로 알림을 트리거합니다.

**요구사항 7.5**: 설정 가능한 임계값 기반 알림 (낙폭 > 5%, 승률 < 45%)

## 주요 기능

- ✅ 설정 가능한 임계값 기반 알림
- ✅ 다양한 알림 레벨 (INFO, WARNING, CRITICAL)
- ✅ 커스텀 알림 핸들러 지원
- ✅ 알림 히스토리 관리
- ✅ JSON 로그 파일 저장
- ✅ 설정 저장/로드

## 기본 사용법

### 1. 기본 임계값으로 사용

```python
from ai_trader.grpo.alerts import PerformanceAlertSystem

# 기본 임계값으로 초기화
alert_system = PerformanceAlertSystem()

# 성능 메트릭
metrics = {
    'win_rate': 0.42,
    'avg_profit_per_trade': -0.001,
    'max_drawdown': 0.08,
    'sharpe_ratio': 0.3,
    'avg_holding_time': 45.0
}

# 메트릭 체크 및 알림 트리거
alerts = alert_system.check_metrics(metrics)

# 알림 요약 출력
alert_system.print_alert_summary()
```

### 2. 편의 함수 사용

```python
from ai_trader.grpo.alerts import monitor_performance

metrics = {
    'win_rate': 0.55,
    'avg_profit_per_trade': 0.003,
    'max_drawdown': 0.03,
    'sharpe_ratio': 1.2,
    'avg_holding_time': 45.0
}

# 간단하게 모니터링
alerts = monitor_performance(
    metrics=metrics,
    log_file='logs/alerts.jsonl',
    print_summary=True
)
```

## 기본 임계값

시스템은 다음과 같은 기본 임계값을 제공합니다:

| 메트릭 | 임계값 | 비교 | 레벨 | 설명 |
|--------|--------|------|------|------|
| `max_drawdown` | 0.05 (5%) | > | CRITICAL | 최대 낙폭이 5%를 초과 |
| `win_rate` | 0.45 (45%) | < | WARNING | 승률이 45% 미만 |
| `sharpe_ratio` | 0.5 | < | WARNING | 샤프 비율이 0.5 미만 |
| `avg_profit_per_trade` | 0.0 | < | CRITICAL | 평균 거래당 수익이 음수 |
| `avg_holding_time` | 60.0 (초) | > | INFO | 평균 보유 시간이 60초 초과 |

## 커스텀 임계값

### 임계값 정의

```python
from ai_trader.grpo.alerts import AlertThreshold, AlertLevel, AlertType

# 커스텀 임계값 생성
custom_threshold = AlertThreshold(
    metric_name=AlertType.WIN_RATE.value,
    threshold=0.50,
    comparison='lt',  # 'gt', 'lt', 'gte', 'lte', 'eq'
    level=AlertLevel.CRITICAL,
    message_template="[{level}] Win rate {value:.2%} is critically low"
)

# 알림 시스템에 추가
alert_system = PerformanceAlertSystem()
alert_system.add_threshold(custom_threshold)
```

### 비교 연산자

- `'gt'`: greater than (>)
- `'lt'`: less than (<)
- `'gte'`: greater than or equal (>=)
- `'lte'`: less than or equal (<=)
- `'eq'`: equal (==)

## 커스텀 알림 핸들러

### 이메일 알림 핸들러

```python
def email_handler(alert):
    """이메일 알림 핸들러"""
    if alert.level == AlertLevel.CRITICAL:
        # 이메일 전송 로직
        send_email(
            subject=f"CRITICAL ALERT: {alert.alert_type}",
            body=alert.message
        )

alert_system.add_handler(email_handler)
```

### Slack 알림 핸들러

```python
def slack_handler(alert):
    """Slack 알림 핸들러"""
    if alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
        # Slack 메시지 전송 로직
        send_slack_message(
            channel='#trading-alerts',
            text=alert.message
        )

alert_system.add_handler(slack_handler)
```

## 백테스팅과 통합

### 백테스팅 시 알림 활성화

```python
from ai_trader.grpo.backtest import run_backtest
from ai_trader.grpo.alerts import AlertThreshold, AlertLevel

# 커스텀 임계값 정의
alert_thresholds = [
    AlertThreshold(
        metric_name='max_drawdown',
        threshold=0.03,  # 3%
        comparison='gt',
        level=AlertLevel.WARNING
    ),
    AlertThreshold(
        metric_name='win_rate',
        threshold=0.50,  # 50%
        comparison='lt',
        level=AlertLevel.CRITICAL
    )
]

# 백테스팅 실행 (알림 활성화)
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    alert_thresholds=alert_thresholds,
    alert_log_file='logs/backtest_alerts.jsonl',
    enable_alerts=True
)

# 알림 확인
if results['alerts']:
    print(f"⚠️  {len(results['alerts'])} alerts triggered")
    for alert in results['alerts']:
        print(f"  - {alert.message}")
```

## 알림 히스토리 관리

### 히스토리 조회

```python
# 전체 히스토리
all_alerts = alert_system.get_alert_history()

# 레벨별 필터링
critical_alerts = alert_system.get_alert_history(level=AlertLevel.CRITICAL)

# 타입별 필터링
drawdown_alerts = alert_system.get_alert_history(alert_type='max_drawdown')

# 최근 N개 알림
recent_alerts = alert_system.get_alert_history(limit=10)
```

### 히스토리 초기화

```python
alert_system.clear_history()
```

## 설정 저장 및 로드

### 설정 저장

```python
# 설정을 JSON 파일로 저장
alert_system.save_config('config/alert_config.json')
```

### 설정 로드

```python
# JSON 파일에서 설정 로드
alert_system = PerformanceAlertSystem.load_config('config/alert_config.json')
```

### 설정 파일 형식

```json
{
  "thresholds": [
    {
      "metric_name": "max_drawdown",
      "threshold": 0.05,
      "comparison": "gt",
      "level": "CRITICAL",
      "message_template": "[{level}] Maximum drawdown is {value:.2%}"
    },
    {
      "metric_name": "win_rate",
      "threshold": 0.45,
      "comparison": "lt",
      "level": "WARNING",
      "message_template": "[{level}] Win rate is {value:.2%}"
    }
  ],
  "log_file": "logs/alerts.jsonl"
}
```

## 알림 로그 파일

### JSON Lines 형식

알림은 JSON Lines 형식으로 저장됩니다 (각 라인이 하나의 JSON 객체):

```json
{"alert_type": "max_drawdown", "level": "CRITICAL", "message": "...", "metric_value": 0.08, "threshold": 0.05, "timestamp": "2024-10-05T12:00:00", "metadata": {}}
{"alert_type": "win_rate", "level": "WARNING", "message": "...", "metric_value": 0.42, "threshold": 0.45, "timestamp": "2024-10-05T12:00:01", "metadata": {}}
```

### 로그 파일 읽기

```python
import json

with open('logs/alerts.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        alert_data = json.loads(line)
        print(f"{alert_data['timestamp']}: {alert_data['message']}")
```

## 알림 레벨

### AlertLevel Enum

```python
from ai_trader.grpo.alerts import AlertLevel

AlertLevel.INFO       # 정보성 알림
AlertLevel.WARNING    # 경고 알림
AlertLevel.CRITICAL   # 심각한 알림
```

## 알림 타입

### AlertType Enum

```python
from ai_trader.grpo.alerts import AlertType

AlertType.MAX_DRAWDOWN              # 최대 낙폭
AlertType.WIN_RATE                  # 승률
AlertType.SHARPE_RATIO              # 샤프 비율
AlertType.AVG_PROFIT                # 평균 거래당 수익
AlertType.AVG_HOLDING_TIME          # 평균 보유 시간
AlertType.QUICK_EXIT_VIOLATIONS     # 빠른 손절 룰 위반
AlertType.TOTAL_RETURN              # 총 수익률
```

## 실전 예제

### 1. 프로덕션 환경 모니터링

```python
from ai_trader.grpo.alerts import PerformanceAlertSystem, AlertThreshold, AlertLevel
import smtplib
from email.mime.text import MIMEText

# 이메일 핸들러
def send_email_alert(alert):
    if alert.level == AlertLevel.CRITICAL:
        msg = MIMEText(alert.message)
        msg['Subject'] = f'CRITICAL: Trading Alert - {alert.alert_type}'
        msg['From'] = 'alerts@trading.com'
        msg['To'] = 'admin@trading.com'
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('user', 'password')
            server.send_message(msg)

# 알림 시스템 설정
alert_system = PerformanceAlertSystem(
    log_file='logs/production_alerts.jsonl'
)
alert_system.add_handler(send_email_alert)

# 실시간 모니터링
while True:
    # 성능 메트릭 수집
    metrics = collect_performance_metrics()
    
    # 알림 체크
    alerts = alert_system.check_metrics(metrics)
    
    if alerts:
        logger.warning(f"Triggered {len(alerts)} alerts")
    
    time.sleep(60)  # 1분마다 체크
```

### 2. 백테스팅 후 자동 보고서

```python
from ai_trader.grpo.backtest import run_backtest
from ai_trader.grpo.alerts import PerformanceAlertSystem

# 백테스팅 실행
results = run_backtest(
    embedding_model_path='models/embedding/checkpoint.pt',
    policy_path='models/grpo/checkpoint.pt',
    db_path='datasets.duckdb',
    enable_alerts=True,
    alert_log_file='logs/backtest_alerts.jsonl'
)

# 알림 요약 생성
if results['alerts']:
    print("\n⚠️  PERFORMANCE ISSUES DETECTED")
    print("=" * 70)
    
    critical_alerts = [a for a in results['alerts'] if a.level == AlertLevel.CRITICAL]
    warning_alerts = [a for a in results['alerts'] if a.level == AlertLevel.WARNING]
    
    if critical_alerts:
        print(f"\n🚨 {len(critical_alerts)} CRITICAL ALERTS:")
        for alert in critical_alerts:
            print(f"  - {alert.message}")
    
    if warning_alerts:
        print(f"\n⚠️  {len(warning_alerts)} WARNING ALERTS:")
        for alert in warning_alerts:
            print(f"  - {alert.message}")
    
    print("\n❌ RECOMMENDATION: Review and adjust strategy before deployment")
else:
    print("\n✅ All performance metrics within acceptable thresholds")
    print("✅ Strategy is ready for deployment")
```

## 문제 해결

### 알림이 트리거되지 않음

1. 메트릭 이름이 정확한지 확인
2. 임계값이 올바르게 설정되었는지 확인
3. 비교 연산자가 올바른지 확인

```python
# 디버깅
print(f"Thresholds: {len(alert_system.thresholds)}")
for threshold in alert_system.thresholds:
    print(f"  {threshold.metric_name} {threshold.comparison} {threshold.threshold}")

print(f"Metrics: {metrics.keys()}")
```

### 알림이 너무 많이 트리거됨

1. 임계값을 조정
2. 알림 레벨을 변경
3. 특정 알림을 비활성화

```python
# 임계값 조정
alert_system.thresholds = [
    t for t in alert_system.thresholds 
    if t.metric_name != 'avg_holding_time'  # 특정 알림 제거
]
```

## 참고 자료

- **구현 파일**: `ai_trader/grpo/alerts.py`
- **테스트 파일**: `tests/test_grpo_alerts.py`
- **예제 스크립트**: `examples/grpo_alert_example.py`
- **요구사항**: 요구사항 7.5 (설정 가능한 임계값 기반 알림)

## 요약

GRPO 성능 알림 시스템은 에이전트의 성능을 실시간으로 모니터링하고 문제를 조기에 감지할 수 있게 해줍니다. 설정 가능한 임계값, 커스텀 핸들러, 그리고 백테스팅 통합을 통해 프로덕션 환경에서 안전하게 거래 전략을 운영할 수 있습니다.
