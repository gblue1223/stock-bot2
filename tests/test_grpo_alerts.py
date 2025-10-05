"""
GRPO 성능 알림 시스템 테스트

이 모듈은 성능 알림 시스템의 기능을 테스트합니다.
"""

import pytest
import tempfile
import json
from pathlib import Path

from ai_trader.grpo.alerts import (
    PerformanceAlertSystem,
    AlertThreshold,
    AlertLevel,
    AlertType,
    Alert,
    monitor_performance
)


class TestAlertThreshold:
    """AlertThreshold 클래스 테스트"""
    
    def test_check_greater_than(self):
        """gt 비교 연산자 테스트"""
        threshold = AlertThreshold(
            metric_name='max_drawdown',
            threshold=0.05,
            comparison='gt',
            level=AlertLevel.CRITICAL
        )
        
        assert threshold.check(0.06) is True
        assert threshold.check(0.05) is False
        assert threshold.check(0.04) is False
    
    def test_check_less_than(self):
        """lt 비교 연산자 테스트"""
        threshold = AlertThreshold(
            metric_name='win_rate',
            threshold=0.45,
            comparison='lt',
            level=AlertLevel.WARNING
        )
        
        assert threshold.check(0.44) is True
        assert threshold.check(0.45) is False
        assert threshold.check(0.46) is False
    
    def test_check_greater_than_or_equal(self):
        """gte 비교 연산자 테스트"""
        threshold = AlertThreshold(
            metric_name='sharpe_ratio',
            threshold=1.0,
            comparison='gte',
            level=AlertLevel.INFO
        )
        
        assert threshold.check(1.1) is True
        assert threshold.check(1.0) is True
        assert threshold.check(0.9) is False
    
    def test_check_less_than_or_equal(self):
        """lte 비교 연산자 테스트"""
        threshold = AlertThreshold(
            metric_name='avg_profit',
            threshold=0.0,
            comparison='lte',
            level=AlertLevel.CRITICAL
        )
        
        assert threshold.check(-0.01) is True
        assert threshold.check(0.0) is True
        assert threshold.check(0.01) is False
    
    def test_format_message_default(self):
        """기본 메시지 포맷팅 테스트"""
        threshold = AlertThreshold(
            metric_name='max_drawdown',
            threshold=0.05,
            comparison='gt',
            level=AlertLevel.CRITICAL
        )
        
        message = threshold.format_message(0.08)
        
        assert 'max_drawdown' in message
        assert '0.08' in message
        assert '0.05' in message
        assert 'CRITICAL' in message
    
    def test_format_message_custom(self):
        """커스텀 메시지 포맷팅 테스트"""
        threshold = AlertThreshold(
            metric_name='win_rate',
            threshold=0.45,
            comparison='lt',
            level=AlertLevel.WARNING,
            message_template="Win rate {value:.2%} is below {threshold:.2%}"
        )
        
        message = threshold.format_message(0.42)
        
        assert message == "Win rate 42.00% is below 45.00%"


class TestPerformanceAlertSystem:
    """PerformanceAlertSystem 클래스 테스트"""
    
    def test_initialization_default(self):
        """기본 초기화 테스트"""
        alert_system = PerformanceAlertSystem()
        
        assert len(alert_system.thresholds) > 0
        assert len(alert_system.alert_handlers) > 0
        assert len(alert_system.alert_history) == 0
    
    def test_initialization_custom(self):
        """커스텀 초기화 테스트"""
        custom_thresholds = [
            AlertThreshold(
                metric_name='win_rate',
                threshold=0.50,
                comparison='lt',
                level=AlertLevel.CRITICAL
            )
        ]
        
        alert_system = PerformanceAlertSystem(thresholds=custom_thresholds)
        
        assert len(alert_system.thresholds) == 1
        assert alert_system.thresholds[0].threshold == 0.50
    
    def test_add_threshold(self):
        """임계값 추가 테스트"""
        alert_system = PerformanceAlertSystem(thresholds=[])
        
        initial_count = len(alert_system.thresholds)
        
        alert_system.add_threshold(
            AlertThreshold(
                metric_name='sharpe_ratio',
                threshold=1.0,
                comparison='lt',
                level=AlertLevel.INFO
            )
        )
        
        assert len(alert_system.thresholds) == initial_count + 1
    
    def test_check_metrics_no_alerts(self):
        """알림이 트리거되지 않는 경우 테스트"""
        alert_system = PerformanceAlertSystem()
        
        # 모든 메트릭이 정상 범위 내
        metrics = {
            'win_rate': 0.55,
            'avg_profit_per_trade': 0.003,
            'max_drawdown': 0.03,
            'sharpe_ratio': 1.2,
            'avg_holding_time': 45.0
        }
        
        alerts = alert_system.check_metrics(metrics)
        
        assert len(alerts) == 0
    
    def test_check_metrics_with_alerts(self):
        """알림이 트리거되는 경우 테스트"""
        alert_system = PerformanceAlertSystem()
        
        # 여러 메트릭이 임계값 위반
        metrics = {
            'win_rate': 0.40,  # < 0.45 -> WARNING
            'avg_profit_per_trade': -0.001,  # < 0 -> CRITICAL
            'max_drawdown': 0.08,  # > 0.05 -> CRITICAL
            'sharpe_ratio': 0.3,  # < 0.5 -> WARNING
            'avg_holding_time': 70.0  # > 60 -> INFO
        }
        
        alerts = alert_system.check_metrics(metrics)
        
        assert len(alerts) > 0
        
        # 레벨별 알림 확인
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        info_alerts = [a for a in alerts if a.level == AlertLevel.INFO]
        
        assert len(critical_alerts) >= 2  # max_drawdown, avg_profit
        assert len(warning_alerts) >= 2  # win_rate, sharpe_ratio
        assert len(info_alerts) >= 1  # avg_holding_time
    
    def test_check_metrics_with_metadata(self):
        """메타데이터와 함께 메트릭 체크 테스트"""
        alert_system = PerformanceAlertSystem()
        
        metrics = {
            'win_rate': 0.40,
            'avg_profit_per_trade': 0.001,
            'max_drawdown': 0.03,
            'sharpe_ratio': 0.8,
            'avg_holding_time': 50.0
        }
        
        metadata = {
            'test_period': '2024-01-01 to 2024-03-31',
            'num_stocks': 50
        }
        
        alerts = alert_system.check_metrics(metrics, metadata=metadata)
        
        for alert in alerts:
            assert alert.metadata == metadata
    
    def test_custom_handler(self):
        """커스텀 핸들러 테스트"""
        handler_called = []
        
        def custom_handler(alert):
            handler_called.append(alert)
        
        alert_system = PerformanceAlertSystem()
        alert_system.add_handler(custom_handler)
        
        metrics = {
            'win_rate': 0.40,
            'avg_profit_per_trade': 0.001,
            'max_drawdown': 0.03,
            'sharpe_ratio': 0.8,
            'avg_holding_time': 50.0
        }
        
        alerts = alert_system.check_metrics(metrics)
        
        # 커스텀 핸들러가 호출되었는지 확인
        assert len(handler_called) == len(alerts)
    
    def test_get_alert_history(self):
        """알림 히스토리 조회 테스트"""
        alert_system = PerformanceAlertSystem()
        
        # 여러 번 메트릭 체크
        for i in range(3):
            metrics = {
                'win_rate': 0.40 + i * 0.01,
                'avg_profit_per_trade': -0.001,
                'max_drawdown': 0.08,
                'sharpe_ratio': 0.3,
                'avg_holding_time': 50.0
            }
            alert_system.check_metrics(metrics)
        
        # 전체 히스토리
        all_alerts = alert_system.get_alert_history()
        assert len(all_alerts) > 0
        
        # 레벨별 필터링
        critical_alerts = alert_system.get_alert_history(level=AlertLevel.CRITICAL)
        assert all(a.level == AlertLevel.CRITICAL for a in critical_alerts)
        
        # 타입별 필터링
        drawdown_alerts = alert_system.get_alert_history(alert_type='max_drawdown')
        assert all(a.alert_type == 'max_drawdown' for a in drawdown_alerts)
        
        # 제한
        limited_alerts = alert_system.get_alert_history(limit=2)
        assert len(limited_alerts) <= 2
    
    def test_clear_history(self):
        """히스토리 초기화 테스트"""
        alert_system = PerformanceAlertSystem()
        
        metrics = {
            'win_rate': 0.40,
            'avg_profit_per_trade': -0.001,
            'max_drawdown': 0.08,
            'sharpe_ratio': 0.3,
            'avg_holding_time': 50.0
        }
        
        alert_system.check_metrics(metrics)
        assert len(alert_system.alert_history) > 0
        
        alert_system.clear_history()
        assert len(alert_system.alert_history) == 0
    
    def test_save_and_load_config(self):
        """설정 저장 및 로드 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'alert_config.json'
            
            # 설정 저장
            alert_system = PerformanceAlertSystem()
            alert_system.add_threshold(
                AlertThreshold(
                    metric_name='total_return',
                    threshold=-0.05,
                    comparison='lt',
                    level=AlertLevel.CRITICAL
                )
            )
            
            original_threshold_count = len(alert_system.thresholds)
            alert_system.save_config(str(config_path))
            
            assert config_path.exists()
            
            # 설정 로드
            loaded_system = PerformanceAlertSystem.load_config(str(config_path))
            
            assert len(loaded_system.thresholds) == original_threshold_count
            
            # 로드된 설정으로 메트릭 체크
            metrics = {
                'total_return': -0.08,
                'win_rate': 0.50,
                'avg_profit_per_trade': 0.001,
                'max_drawdown': 0.03,
                'sharpe_ratio': 1.0,
                'avg_holding_time': 50.0
            }
            
            alerts = loaded_system.check_metrics(metrics)
            
            # total_return 알림이 트리거되어야 함
            total_return_alerts = [a for a in alerts if a.alert_type == 'total_return']
            assert len(total_return_alerts) > 0
    
    def test_alert_log_file(self):
        """알림 로그 파일 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'alerts.jsonl'
            
            alert_system = PerformanceAlertSystem(log_file=str(log_file))
            
            metrics = {
                'win_rate': 0.40,
                'avg_profit_per_trade': -0.001,
                'max_drawdown': 0.08,
                'sharpe_ratio': 0.3,
                'avg_holding_time': 50.0
            }
            
            alerts = alert_system.check_metrics(metrics)
            
            # 로그 파일이 생성되었는지 확인
            assert log_file.exists()
            
            # 로그 파일 내용 확인
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) == len(alerts)
            
            # 각 라인이 유효한 JSON인지 확인
            for line in lines:
                alert_data = json.loads(line)
                assert 'alert_type' in alert_data
                assert 'level' in alert_data
                assert 'message' in alert_data


class TestMonitorPerformance:
    """monitor_performance 편의 함수 테스트"""
    
    def test_monitor_performance_no_alerts(self):
        """알림이 없는 경우 테스트"""
        metrics = {
            'win_rate': 0.55,
            'avg_profit_per_trade': 0.003,
            'max_drawdown': 0.03,
            'sharpe_ratio': 1.2,
            'avg_holding_time': 45.0
        }
        
        alerts = monitor_performance(metrics, print_summary=False)
        
        assert len(alerts) == 0
    
    def test_monitor_performance_with_alerts(self):
        """알림이 있는 경우 테스트"""
        metrics = {
            'win_rate': 0.40,
            'avg_profit_per_trade': -0.001,
            'max_drawdown': 0.08,
            'sharpe_ratio': 0.3,
            'avg_holding_time': 50.0
        }
        
        alerts = monitor_performance(metrics, print_summary=False)
        
        assert len(alerts) > 0
    
    def test_monitor_performance_custom_thresholds(self):
        """커스텀 임계값 테스트"""
        custom_thresholds = [
            AlertThreshold(
                metric_name='win_rate',
                threshold=0.60,
                comparison='lt',
                level=AlertLevel.CRITICAL
            )
        ]
        
        metrics = {
            'win_rate': 0.55,
            'avg_profit_per_trade': 0.003,
            'max_drawdown': 0.03,
            'sharpe_ratio': 1.2,
            'avg_holding_time': 45.0
        }
        
        alerts = monitor_performance(
            metrics,
            thresholds=custom_thresholds,
            print_summary=False
        )
        
        # 커스텀 임계값에 의해 알림이 트리거되어야 함
        assert len(alerts) > 0
        assert alerts[0].alert_type == 'win_rate'


class TestAlert:
    """Alert 클래스 테스트"""
    
    def test_alert_creation(self):
        """알림 생성 테스트"""
        alert = Alert(
            alert_type='max_drawdown',
            level=AlertLevel.CRITICAL,
            message='Test alert',
            metric_value=0.08,
            threshold=0.05
        )
        
        assert alert.alert_type == 'max_drawdown'
        assert alert.level == AlertLevel.CRITICAL
        assert alert.message == 'Test alert'
        assert alert.metric_value == 0.08
        assert alert.threshold == 0.05
        assert alert.timestamp is not None
    
    def test_alert_to_dict(self):
        """알림 딕셔너리 변환 테스트"""
        alert = Alert(
            alert_type='win_rate',
            level=AlertLevel.WARNING,
            message='Win rate too low',
            metric_value=0.40,
            threshold=0.45,
            metadata={'test': 'data'}
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_type'] == 'win_rate'
        assert alert_dict['level'] == 'WARNING'
        assert alert_dict['message'] == 'Win rate too low'
        assert alert_dict['metric_value'] == 0.40
        assert alert_dict['threshold'] == 0.45
        assert alert_dict['metadata'] == {'test': 'data'}
        assert 'timestamp' in alert_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
