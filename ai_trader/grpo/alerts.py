"""
성능 알림 시스템

이 모듈은 GRPO 에이전트의 성능을 모니터링하고 설정 가능한 임계값을 기반으로 알림을 트리거합니다.
요구사항 7.5에 따라 낙폭 > 5%, 승률 < 45% 등의 조건에서 알림을 발생시킵니다.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """알림 타입"""
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    AVG_PROFIT = "avg_profit_per_trade"
    AVG_HOLDING_TIME = "avg_holding_time"
    QUICK_EXIT_VIOLATIONS = "quick_exit_violations"
    TOTAL_RETURN = "total_return"


@dataclass
class AlertThreshold:
    """
    알림 임계값 설정
    
    Args:
        metric_name: 메트릭 이름 (AlertType enum 값)
        threshold: 임계값
        comparison: 비교 연산자 ('gt', 'lt', 'gte', 'lte', 'eq')
        level: 알림 레벨 (AlertLevel enum 값)
        message_template: 알림 메시지 템플릿 (선택적)
    """
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    level: AlertLevel = AlertLevel.WARNING
    message_template: Optional[str] = None
    
    def check(self, value: float) -> bool:
        """
        임계값 체크
        
        Args:
            value: 체크할 값
            
        Returns:
            임계값을 위반했는지 여부
        """
        if self.comparison == 'gt':
            return value > self.threshold
        elif self.comparison == 'lt':
            return value < self.threshold
        elif self.comparison == 'gte':
            return value >= self.threshold
        elif self.comparison == 'lte':
            return value <= self.threshold
        elif self.comparison == 'eq':
            return abs(value - self.threshold) < 1e-6
        else:
            raise ValueError(f"Unknown comparison operator: {self.comparison}")
    
    def format_message(self, value: float) -> str:
        """
        알림 메시지 포맷팅
        
        Args:
            value: 현재 값
            
        Returns:
            포맷팅된 메시지
        """
        if self.message_template:
            return self.message_template.format(
                metric=self.metric_name,
                value=value,
                threshold=self.threshold,
                level=self.level.value
            )
        else:
            # 기본 메시지
            comparison_text = {
                'gt': 'greater than',
                'lt': 'less than',
                'gte': 'greater than or equal to',
                'lte': 'less than or equal to',
                'eq': 'equal to'
            }
            return (f"[{self.level.value}] {self.metric_name} is {value:.4f}, "
                   f"which is {comparison_text[self.comparison]} threshold {self.threshold:.4f}")


@dataclass
class Alert:
    """
    알림 데이터
    
    Args:
        alert_type: 알림 타입
        level: 알림 레벨
        message: 알림 메시지
        metric_value: 메트릭 현재 값
        threshold: 임계값
        timestamp: 알림 발생 시간
        metadata: 추가 메타데이터
    """
    alert_type: str
    level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'alert_type': self.alert_type,
            'level': self.level.value,
            'message': self.message,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class PerformanceAlertSystem:
    """
    성능 알림 시스템
    
    GRPO 에이전트의 성능 메트릭을 모니터링하고,
    설정 가능한 임계값을 기반으로 알림을 트리거합니다.
    
    요구사항 7.5에 따라 다음과 같은 알림을 지원합니다:
    - 최대 낙폭 > 5%
    - 승률 < 45%
    - 샤프 비율 < 0.5
    - 평균 거래당 수익 < 0
    - 빠른 손절 룰 위반 비율 > 20%
    
    Args:
        thresholds: 알림 임계값 리스트
        alert_handlers: 알림 핸들러 함수 리스트 (선택적)
        log_file: 알림 로그 파일 경로 (선택적)
    """
    
    def __init__(
        self,
        thresholds: Optional[List[AlertThreshold]] = None,
        alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
        log_file: Optional[str] = None
    ):
        self.thresholds = thresholds or self._default_thresholds()
        self.alert_handlers = alert_handlers or []
        self.log_file = log_file
        
        # 알림 히스토리
        self.alert_history: List[Alert] = []
        
        # 기본 핸들러 추가
        self.alert_handlers.append(self._log_alert)
        
        if self.log_file:
            self.alert_handlers.append(self._write_alert_to_file)
        
        logger.info(f"PerformanceAlertSystem initialized with {len(self.thresholds)} thresholds")
    
    @staticmethod
    def _default_thresholds() -> List[AlertThreshold]:
        """
        기본 알림 임계값 설정
        
        요구사항 7.5에 따른 기본 임계값:
        - 최대 낙폭 > 5% (CRITICAL)
        - 승률 < 45% (WARNING)
        - 샤프 비율 < 0.5 (WARNING)
        - 평균 거래당 수익 < 0 (CRITICAL)
        - 빠른 손절 룰 위반 비율 > 20% (WARNING)
        
        Returns:
            기본 알림 임계값 리스트
        """
        return [
            # 최대 낙폭 > 5% (CRITICAL)
            AlertThreshold(
                metric_name=AlertType.MAX_DRAWDOWN.value,
                threshold=0.05,
                comparison='gt',
                level=AlertLevel.CRITICAL,
                message_template="[{level}] Maximum drawdown is {value:.2%}, exceeding threshold of {threshold:.2%}"
            ),
            
            # 승률 < 45% (WARNING)
            AlertThreshold(
                metric_name=AlertType.WIN_RATE.value,
                threshold=0.45,
                comparison='lt',
                level=AlertLevel.WARNING,
                message_template="[{level}] Win rate is {value:.2%}, below threshold of {threshold:.2%}"
            ),
            
            # 샤프 비율 < 0.5 (WARNING)
            AlertThreshold(
                metric_name=AlertType.SHARPE_RATIO.value,
                threshold=0.5,
                comparison='lt',
                level=AlertLevel.WARNING,
                message_template="[{level}] Sharpe ratio is {value:.4f}, below threshold of {threshold:.4f}"
            ),
            
            # 평균 거래당 수익 < 0 (CRITICAL)
            AlertThreshold(
                metric_name=AlertType.AVG_PROFIT.value,
                threshold=0.0,
                comparison='lt',
                level=AlertLevel.CRITICAL,
                message_template="[{level}] Average profit per trade is {value:.4f}, below threshold of {threshold:.4f}"
            ),
            
            # 평균 보유 시간 > 60초 (INFO)
            AlertThreshold(
                metric_name=AlertType.AVG_HOLDING_TIME.value,
                threshold=60.0,
                comparison='gt',
                level=AlertLevel.INFO,
                message_template="[{level}] Average holding time is {value:.2f}s, exceeding threshold of {threshold:.2f}s"
            ),
        ]
    
    def add_threshold(self, threshold: AlertThreshold):
        """
        알림 임계값 추가
        
        Args:
            threshold: 추가할 알림 임계값
        """
        self.thresholds.append(threshold)
        logger.info(f"Added threshold: {threshold.metric_name} {threshold.comparison} {threshold.threshold}")
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """
        알림 핸들러 추가
        
        Args:
            handler: 알림 핸들러 함수 (Alert 객체를 인자로 받음)
        """
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def check_metrics(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        메트릭 체크 및 알림 트리거
        
        Args:
            metrics: 체크할 메트릭 딕셔너리
            metadata: 추가 메타데이터 (선택적)
            
        Returns:
            트리거된 알림 리스트
        """
        triggered_alerts = []
        
        for threshold in self.thresholds:
            # 메트릭 값 가져오기
            if threshold.metric_name not in metrics:
                continue
            
            metric_value = metrics[threshold.metric_name]
            
            # 임계값 체크
            if threshold.check(metric_value):
                # 알림 생성
                alert = Alert(
                    alert_type=threshold.metric_name,
                    level=threshold.level,
                    message=threshold.format_message(metric_value),
                    metric_value=metric_value,
                    threshold=threshold.threshold,
                    metadata=metadata or {}
                )
                
                triggered_alerts.append(alert)
                self.alert_history.append(alert)
                
                # 알림 핸들러 실행
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in alert handler {handler.__name__}: {e}")
        
        return triggered_alerts
    
    def _log_alert(self, alert: Alert):
        """
        알림을 로그에 기록
        
        Args:
            alert: 알림 객체
        """
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(alert.message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(alert.message)
        else:
            logger.info(alert.message)
    
    def _write_alert_to_file(self, alert: Alert):
        """
        알림을 파일에 기록
        
        Args:
            alert: 알림 객체
        """
        if not self.log_file:
            return
        
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON Lines 형식으로 추가
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    def get_alert_history(
        self,
        level: Optional[AlertLevel] = None,
        alert_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """
        알림 히스토리 조회
        
        Args:
            level: 필터링할 알림 레벨 (선택적)
            alert_type: 필터링할 알림 타입 (선택적)
            limit: 반환할 최대 알림 수 (선택적)
            
        Returns:
            필터링된 알림 리스트
        """
        filtered_alerts = self.alert_history
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
        
        if limit:
            filtered_alerts = filtered_alerts[-limit:]
        
        return filtered_alerts
    
    def clear_history(self):
        """알림 히스토리 초기화"""
        self.alert_history = []
        logger.info("Alert history cleared")
    
    def print_alert_summary(self):
        """알림 요약 출력"""
        if not self.alert_history:
            print("\nNo alerts triggered")
            return
        
        print("\n" + "=" * 70)
        print("ALERT SUMMARY")
        print("=" * 70)
        
        # 레벨별 카운트
        level_counts = {}
        for alert in self.alert_history:
            level_counts[alert.level] = level_counts.get(alert.level, 0) + 1
        
        print(f"Total Alerts:            {len(self.alert_history)}")
        for level in AlertLevel:
            count = level_counts.get(level, 0)
            print(f"  {level.value:12s}:      {count}")
        
        print("-" * 70)
        print("RECENT ALERTS")
        print("-" * 70)
        
        # 최근 10개 알림 출력
        recent_alerts = self.alert_history[-10:]
        for alert in recent_alerts:
            print(f"[{alert.timestamp}] {alert.message}")
        
        print("=" * 70 + "\n")
    
    def save_config(self, config_path: str):
        """
        알림 설정 저장
        
        Args:
            config_path: 설정 파일 경로 (.json)
        """
        config = {
            'thresholds': [
                {
                    'metric_name': t.metric_name,
                    'threshold': t.threshold,
                    'comparison': t.comparison,
                    'level': t.level.value,
                    'message_template': t.message_template
                }
                for t in self.thresholds
            ],
            'log_file': self.log_file
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Alert config saved to {config_path}")
    
    @classmethod
    def load_config(cls, config_path: str) -> 'PerformanceAlertSystem':
        """
        알림 설정 로드
        
        Args:
            config_path: 설정 파일 경로 (.json)
            
        Returns:
            PerformanceAlertSystem 인스턴스
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        thresholds = [
            AlertThreshold(
                metric_name=t['metric_name'],
                threshold=t['threshold'],
                comparison=t['comparison'],
                level=AlertLevel(t['level']),
                message_template=t.get('message_template')
            )
            for t in config['thresholds']
        ]
        
        return cls(
            thresholds=thresholds,
            log_file=config.get('log_file')
        )


def monitor_performance(
    metrics: Dict[str, float],
    thresholds: Optional[List[AlertThreshold]] = None,
    log_file: Optional[str] = None,
    print_summary: bool = True
) -> List[Alert]:
    """
    성능 모니터링 (편의 함수)
    
    Args:
        metrics: 체크할 메트릭 딕셔너리
        thresholds: 알림 임계값 리스트 (None이면 기본값 사용)
        log_file: 알림 로그 파일 경로 (선택적)
        print_summary: 알림 요약 출력 여부
        
    Returns:
        트리거된 알림 리스트
    """
    alert_system = PerformanceAlertSystem(
        thresholds=thresholds,
        log_file=log_file
    )
    
    alerts = alert_system.check_metrics(metrics)
    
    if print_summary and alerts:
        alert_system.print_alert_summary()
    
    return alerts
