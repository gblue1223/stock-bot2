"""
GRPO 성능 알림 시스템 사용 예제

이 스크립트는 GRPO 에이전트의 성능을 모니터링하고 알림을 트리거하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.alerts import (
    PerformanceAlertSystem,
    AlertThreshold,
    AlertLevel,
    AlertType,
    monitor_performance
)


def example_basic_usage():
    """기본 사용 예제"""
    print("\n" + "=" * 70)
    print("Example 1: Basic Usage with Default Thresholds")
    print("=" * 70)
    
    # 기본 임계값으로 알림 시스템 초기화
    alert_system = PerformanceAlertSystem()
    
    # 성능 메트릭 (예시)
    metrics = {
        'win_rate': 0.42,  # 45% 미만 -> WARNING
        'avg_profit_per_trade': -0.001,  # 0 미만 -> CRITICAL
        'max_drawdown': 0.08,  # 5% 초과 -> CRITICAL
        'sharpe_ratio': 0.3,  # 0.5 미만 -> WARNING
        'avg_holding_time': 45.0,  # 60초 미만 -> OK
        'total_trades': 100,
        'total_return': -0.1
    }
    
    # 메트릭 체크 및 알림 트리거
    alerts = alert_system.check_metrics(metrics)
    
    print(f"\nTriggered {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert.message}")
    
    # 알림 요약 출력
    alert_system.print_alert_summary()


def example_custom_thresholds():
    """커스텀 임계값 예제"""
    print("\n" + "=" * 70)
    print("Example 2: Custom Thresholds")
    print("=" * 70)
    
    # 커스텀 임계값 정의
    custom_thresholds = [
        # 승률 < 50% (CRITICAL)
        AlertThreshold(
            metric_name=AlertType.WIN_RATE.value,
            threshold=0.50,
            comparison='lt',
            level=AlertLevel.CRITICAL,
            message_template="[{level}] Win rate {value:.2%} is critically low (threshold: {threshold:.2%})"
        ),
        
        # 최대 낙폭 > 3% (WARNING)
        AlertThreshold(
            metric_name=AlertType.MAX_DRAWDOWN.value,
            threshold=0.03,
            comparison='gt',
            level=AlertLevel.WARNING,
            message_template="[{level}] Drawdown {value:.2%} exceeds warning threshold {threshold:.2%}"
        ),
        
        # 샤프 비율 < 1.0 (INFO)
        AlertThreshold(
            metric_name=AlertType.SHARPE_RATIO.value,
            threshold=1.0,
            comparison='lt',
            level=AlertLevel.INFO,
            message_template="[{level}] Sharpe ratio {value:.4f} is below target {threshold:.4f}"
        ),
    ]
    
    # 커스텀 임계값으로 알림 시스템 초기화
    alert_system = PerformanceAlertSystem(
        thresholds=custom_thresholds,
        log_file='logs/alerts.jsonl'
    )
    
    # 성능 메트릭
    metrics = {
        'win_rate': 0.48,  # 50% 미만 -> CRITICAL
        'max_drawdown': 0.04,  # 3% 초과 -> WARNING
        'sharpe_ratio': 0.8,  # 1.0 미만 -> INFO
        'avg_profit_per_trade': 0.002,
        'avg_holding_time': 35.0
    }
    
    # 메트릭 체크
    alerts = alert_system.check_metrics(metrics)
    
    print(f"\nTriggered {len(alerts)} alerts with custom thresholds")
    alert_system.print_alert_summary()


def example_custom_handler():
    """커스텀 알림 핸들러 예제"""
    print("\n" + "=" * 70)
    print("Example 3: Custom Alert Handler")
    print("=" * 70)
    
    # 커스텀 알림 핸들러 정의
    def email_handler(alert):
        """이메일 알림 핸들러 (시뮬레이션)"""
        if alert.level == AlertLevel.CRITICAL:
            print(f"  📧 [EMAIL SENT] CRITICAL ALERT: {alert.message}")
    
    def slack_handler(alert):
        """Slack 알림 핸들러 (시뮬레이션)"""
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
            print(f"  💬 [SLACK NOTIFICATION] {alert.level.value}: {alert.message}")
    
    # 알림 시스템 초기화
    alert_system = PerformanceAlertSystem()
    
    # 커스텀 핸들러 추가
    alert_system.add_handler(email_handler)
    alert_system.add_handler(slack_handler)
    
    # 성능 메트릭
    metrics = {
        'win_rate': 0.40,  # WARNING
        'avg_profit_per_trade': -0.005,  # CRITICAL
        'max_drawdown': 0.06,  # CRITICAL
        'sharpe_ratio': 0.2,  # WARNING
        'avg_holding_time': 50.0
    }
    
    # 메트릭 체크 (커스텀 핸들러가 자동으로 호출됨)
    print("\nChecking metrics with custom handlers...")
    alerts = alert_system.check_metrics(metrics)
    
    print(f"\nTotal alerts triggered: {len(alerts)}")


def example_with_backtest_results():
    """백테스팅 결과와 함께 사용하는 예제"""
    print("\n" + "=" * 70)
    print("Example 4: Integration with Backtest Results")
    print("=" * 70)
    
    # 백테스팅 결과 시뮬레이션
    backtest_metrics = {
        'win_rate': 0.52,
        'avg_profit_per_trade': 0.0015,
        'max_drawdown': 0.04,
        'sharpe_ratio': 1.2,
        'avg_holding_time': 28.5,
        'total_trades': 250,
        'total_return': 0.375
    }
    
    # 메타데이터 추가
    metadata = {
        'test_period': '2024-01-01 to 2024-03-31',
        'num_stocks': 50,
        'num_episodes': 150
    }
    
    # 알림 시스템으로 모니터링
    alert_system = PerformanceAlertSystem(
        log_file='logs/backtest_alerts.jsonl'
    )
    
    alerts = alert_system.check_metrics(backtest_metrics, metadata=metadata)
    
    if alerts:
        print(f"\n⚠️  {len(alerts)} alerts triggered during backtest")
        alert_system.print_alert_summary()
    else:
        print("\n✅ All metrics within acceptable thresholds")
        print(f"   Win Rate: {backtest_metrics['win_rate']:.2%}")
        print(f"   Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")


def example_convenience_function():
    """편의 함수 사용 예제"""
    print("\n" + "=" * 70)
    print("Example 5: Using Convenience Function")
    print("=" * 70)
    
    # 성능 메트릭
    metrics = {
        'win_rate': 0.55,
        'avg_profit_per_trade': 0.003,
        'max_drawdown': 0.02,
        'sharpe_ratio': 1.5,
        'avg_holding_time': 25.0
    }
    
    # 편의 함수로 간단하게 모니터링
    alerts = monitor_performance(
        metrics=metrics,
        log_file='logs/quick_check.jsonl',
        print_summary=True
    )
    
    if not alerts:
        print("\n✅ Performance looks good!")


def example_save_load_config():
    """설정 저장/로드 예제"""
    print("\n" + "=" * 70)
    print("Example 6: Save and Load Configuration")
    print("=" * 70)
    
    # 커스텀 임계값으로 알림 시스템 생성
    alert_system = PerformanceAlertSystem()
    
    # 추가 임계값 설정
    alert_system.add_threshold(
        AlertThreshold(
            metric_name=AlertType.TOTAL_RETURN.value,
            threshold=-0.05,
            comparison='lt',
            level=AlertLevel.CRITICAL,
            message_template="[{level}] Total return {value:.2%} is critically negative"
        )
    )
    
    # 설정 저장
    config_path = 'config/alert_config.json'
    alert_system.save_config(config_path)
    print(f"\n✅ Configuration saved to {config_path}")
    
    # 설정 로드
    loaded_system = PerformanceAlertSystem.load_config(config_path)
    print(f"✅ Configuration loaded: {len(loaded_system.thresholds)} thresholds")
    
    # 로드된 설정으로 메트릭 체크
    metrics = {
        'win_rate': 0.48,
        'total_return': -0.08,
        'max_drawdown': 0.03,
        'sharpe_ratio': 0.6,
        'avg_profit_per_trade': 0.001,
        'avg_holding_time': 40.0
    }
    
    alerts = loaded_system.check_metrics(metrics)
    print(f"\nTriggered {len(alerts)} alerts with loaded configuration")


def main():
    """모든 예제 실행"""
    print("\n" + "=" * 70)
    print("GRPO Performance Alert System Examples")
    print("=" * 70)
    
    # 예제 1: 기본 사용법
    example_basic_usage()
    
    # 예제 2: 커스텀 임계값
    example_custom_thresholds()
    
    # 예제 3: 커스텀 핸들러
    example_custom_handler()
    
    # 예제 4: 백테스팅 결과와 통합
    example_with_backtest_results()
    
    # 예제 5: 편의 함수
    example_convenience_function()
    
    # 예제 6: 설정 저장/로드
    example_save_load_config()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
