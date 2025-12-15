"""
백테스팅 시뮬레이터

이 모듈은 홀드아웃 테스트 데이터에서 GRPO 에이전트의 거래를 시뮬레이션합니다.
요구사항 7.3에 따라 현실적인 거래 비용 및 슬리피지를 적용합니다.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import duckdb
from datetime import datetime

from ai_trader.inference.grpo_infer import GRPOInference
from ai_trader.grpo.evaluation import GRPOEvaluationMetrics
from ai_trader.grpo.alerts import PerformanceAlertSystem, AlertThreshold

logger = logging.getLogger(__name__)


class BacktestSimulator:
    """
    백테스팅 시뮬레이터
    
    홀드아웃 테스트 데이터에서 거래를 시뮬레이션하고,
    현실적인 거래 비용 및 슬리피지를 적용합니다.
    
    Args:
        inference_engine: GRPO 추론 엔진
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명 (기본값: 'datasets')
        seq_len: 시퀀스 길이 (기본값: 60)
        transaction_cost_rate: 거래 비용 비율 (기본값: 0.00215 = 0.215%)
        slippage_rate: 슬리피지 비율 (기본값: 0.0001 = 0.01%)
        quick_exit_threshold: 빠른 손절 시간 임계값 (초, 기본값: 1.5)
        quick_exit_penalty: 빠른 손절 룰 위반 페널티 (기본값: 0.01)
        max_holding_time: 최대 보유 시간 (초, 기본값: 60)
        holding_penalty_rate: 장기 보유 페널티 비율 (기본값: 0.001)
    """
    
    def __init__(
        self,
        inference_engine: GRPOInference,
        db_path: str,
        table_name: str = 'datasets',
        seq_len: int = 60,
        transaction_cost_rate: float = 0.00215,
        slippage_rate: float = 0.0001,
        quick_exit_threshold: float = 1.5,
        quick_exit_penalty: float = 0.01,
        max_holding_time: float = 60.0,
        holding_penalty_rate: float = 0.001,
        alert_system: Optional[PerformanceAlertSystem] = None,
        enable_alerts: bool = True
    ):
        self.inference_engine = inference_engine
        self.db_path = db_path
        self.table_name = table_name
        self.seq_len = seq_len
        
        # 거래 비용 설정
        self.transaction_cost_rate = transaction_cost_rate
        self.round_trip_cost = transaction_cost_rate * 2  # 왕복 거래비용: 0.43%
        
        # 슬리피지 설정
        self.slippage_rate = slippage_rate
        
        # 빠른 손절 룰 설정
        self.quick_exit_threshold = quick_exit_threshold
        self.quick_exit_penalty = quick_exit_penalty
        
        # 보유 시간 페널티 설정
        self.max_holding_time = max_holding_time
        self.holding_penalty_rate = holding_penalty_rate
        
        # 알림 시스템 설정
        self.enable_alerts = enable_alerts
        if enable_alerts and alert_system is None:
            self.alert_system = PerformanceAlertSystem()
        else:
            self.alert_system = alert_system
        
        # 데이터베이스 연결
        self.conn = None
        self._connect_db()
        
        logger.info(f"BacktestSimulator initialized: "
                   f"transaction_cost={transaction_cost_rate*100:.3f}%, "
                   f"slippage={slippage_rate*100:.3f}%, "
                   f"quick_exit_threshold={quick_exit_threshold}s, "
                   f"alerts_enabled={enable_alerts}")
    
    def _connect_db(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise RuntimeError(f"Cannot connect to database {self.db_path}: {e}")
    
    def _get_feature_columns(self) -> List[str]:
        """특징 컬럼 목록 가져오기"""
        try:
            query = f"DESCRIBE {self.table_name}"
            columns_df = self.conn.execute(query).fetchdf()
            all_columns = columns_df['column_name'].tolist()
            
            # 메타데이터 컬럼 제외
            exclude_columns = {'날짜', '종목코드', '시간'}
            feature_columns = [col for col in all_columns if col not in exclude_columns]
            
            return feature_columns
        except Exception as e:
            logger.error(f"Failed to get feature columns: {e}")
            raise RuntimeError(f"Cannot get feature columns: {e}")
    
    def _load_test_data(
        self,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
        stock_codes: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        테스트 데이터 로드
        
        Args:
            test_start_date: 테스트 시작 날짜 (YYYY-MM-DD)
            test_end_date: 테스트 종료 날짜 (YYYY-MM-DD)
            stock_codes: 테스트할 종목 코드 리스트 (None이면 모든 종목)
            
        Returns:
            (data, metadata) 튜플
            - data: (n_samples, n_features)
            - metadata: (n_samples, 3) - [종목코드, 날짜, 시간]
        """
        feature_cols = self._get_feature_columns()
        
        try:
            # 쿼리 구성
            query = f"""
                SELECT 종목코드, 날짜, 시간, {', '.join(feature_cols)}
                FROM {self.table_name}
                WHERE 1=1
            """
            params = []
            
            # 날짜 필터
            if test_start_date:
                query += " AND 날짜 >= ?"
                params.append(test_start_date)
            
            if test_end_date:
                query += " AND 날짜 <= ?"
                params.append(test_end_date)
            
            # 종목 필터
            if stock_codes:
                placeholders = ','.join(['?' for _ in stock_codes])
                query += f" AND 종목코드 IN ({placeholders})"
                params.extend(stock_codes)
            
            query += " ORDER BY 종목코드, 날짜, 시간"
            
            # 데이터 로드
            df = self.conn.execute(query, params).fetchdf()
            
            if len(df) == 0:
                raise RuntimeError("No test data found")
            
            # 메타데이터와 특징 분리
            metadata = df[['종목코드', '날짜', '시간']].values
            features = df[feature_cols].values.astype(np.float32)
            
            logger.info(f"Loaded test data: {len(features)} samples, "
                       f"{len(df['종목코드'].unique())} stocks, "
                       f"{len(df['날짜'].unique())} days")
            
            return features, metadata
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise RuntimeError(f"Cannot load test data: {e}")
    
    def _apply_slippage(self, price: float, action: int) -> float:
        """
        슬리피지 적용
        
        슬리피지는 주문 실행 시 예상 가격과 실제 체결 가격의 차이입니다.
        매수 시에는 가격이 상승하고, 매도 시에는 가격이 하락합니다.
        
        Args:
            price: 원래 가격
            action: 행동 (1: 매수, 2: 매도)
            
        Returns:
            슬리피지가 적용된 가격
        """
        if action == 1:  # 매수
            # 매수 시 가격 상승 (불리한 방향)
            return price * (1 + self.slippage_rate)
        elif action == 2:  # 매도
            # 매도 시 가격 하락 (불리한 방향)
            return price * (1 - self.slippage_rate)
        else:
            return price
    
    def _calculate_reward(
        self,
        entry_price: float,
        exit_price: float,
        holding_time: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        보상 계산
        
        보상 구조:
        - 수익률 기반 보상: (청산가 - 진입가) / 진입가 - 거래비용
        - 거래비용: 수수료(0.015%) + 세금(0.2%) = 0.215% (매수/매도 각각 적용)
        - 총 거래비용: 0.43% (왕복)
        - 양의 보상 조건: 수익률 > 0.43%
        - 장기 보유 페널티: -0.001 * (보유시간 - 60초)
        
        Args:
            entry_price: 진입 가격
            exit_price: 청산 가격
            holding_time: 보유 시간 (초)
            
        Returns:
            (total_reward, reward_components) 튜플
        """
        # 수익률 계산
        profit_rate = (exit_price - entry_price) / entry_price
        
        # 거래 비용 차감
        reward = profit_rate - self.round_trip_cost
        
        # 장기 보유 페널티
        holding_penalty = 0.0
        if holding_time > self.max_holding_time:
            holding_penalty = self.holding_penalty_rate * (holding_time - self.max_holding_time)
            reward -= holding_penalty
        
        reward_components = {
            'profit_rate': profit_rate,
            'transaction_cost': -self.round_trip_cost,
            'holding_penalty': -holding_penalty,
            'net_reward': reward
        }
        
        return reward, reward_components

    
    def run_backtest(
        self,
        test_start_date: Optional[str] = None,
        test_end_date: Optional[str] = None,
        stock_codes: Optional[List[str]] = None,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        백테스팅 실행
        
        홀드아웃 테스트 데이터에서 거래를 시뮬레이션하고,
        현실적인 거래 비용 및 슬리피지를 적용합니다.
        
        Args:
            test_start_date: 테스트 시작 날짜 (YYYY-MM-DD)
            test_end_date: 테스트 종료 날짜 (YYYY-MM-DD)
            stock_codes: 테스트할 종목 코드 리스트 (None이면 모든 종목)
            deterministic: True면 최대 확률 행동 선택, False면 확률적 샘플링
            verbose: 진행 상황 출력 여부
            
        Returns:
            백테스팅 결과 딕셔너리:
            - episodes: 에피소드 데이터 리스트
            - metrics: 평가 메트릭
            - trades: 모든 거래 데이터
            - summary: 요약 통계
        """
        logger.info("Starting backtest...")
        
        # 테스트 데이터 로드
        test_data, test_metadata = self._load_test_data(
            test_start_date, test_end_date, stock_codes
        )
        
        # 종목별로 데이터 그룹화
        episodes = self._run_episodes_by_stock(
            test_data, test_metadata, deterministic, verbose
        )
        
        # 평가 메트릭 계산
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 모든 거래 데이터 수집
        all_trades = []
        for episode in episodes:
            if 'trades' in episode:
                all_trades.extend(episode['trades'])
        
        # 요약 통계
        summary = self._generate_summary(episodes, metrics, all_trades)
        
        # 성능 알림 체크
        alerts = []
        if self.enable_alerts and self.alert_system:
            metadata = {
                'test_start_date': test_start_date,
                'test_end_date': test_end_date,
                'num_stocks': summary['num_stocks'],
                'num_episodes': summary['num_episodes'],
                'total_trades': summary['total_trades']
            }
            alerts = self.alert_system.check_metrics(metrics, metadata=metadata)
            
            if alerts and verbose:
                self.alert_system.print_alert_summary()
        
        if verbose:
            self._print_backtest_results(summary, metrics)
        
        logger.info("Backtest completed")
        
        return {
            'episodes': episodes,
            'metrics': metrics,
            'trades': all_trades,
            'summary': summary,
            'alerts': alerts
        }
    
    def _run_episodes_by_stock(
        self,
        data: np.ndarray,
        metadata: np.ndarray,
        deterministic: bool,
        verbose: bool
    ) -> List[Dict[str, Any]]:
        """
        종목별로 에피소드 실행
        
        Args:
            data: 특징 데이터 (n_samples, n_features)
            metadata: 메타데이터 (n_samples, 3) - [종목코드, 날짜, 시간]
            deterministic: 결정적 행동 선택 여부
            verbose: 진행 상황 출력 여부
            
        Returns:
            에피소드 데이터 리스트
        """
        episodes = []
        
        # 종목별로 그룹화
        unique_stocks = np.unique(metadata[:, 0])
        
        for stock_idx, stock_code in enumerate(unique_stocks):
            # 해당 종목의 데이터 추출
            stock_mask = metadata[:, 0] == stock_code
            stock_data = data[stock_mask]
            stock_metadata = metadata[stock_mask]
            
            # 날짜별로 그룹화
            unique_dates = np.unique(stock_metadata[:, 1])
            
            for date in unique_dates:
                # 해당 날짜의 데이터 추출
                date_mask = stock_metadata[:, 1] == date
                episode_data = stock_data[date_mask]
                episode_metadata = stock_metadata[date_mask]
                
                # 에피소드 길이 체크
                if len(episode_data) < self.seq_len + 10:
                    continue  # 너무 짧은 에피소드는 스킵
                
                # 에피소드 실행
                episode_result = self._run_single_episode(
                    episode_data,
                    episode_metadata,
                    deterministic
                )
                
                episodes.append(episode_result)
                
                if verbose and (len(episodes) % 10 == 0):
                    logger.info(f"Processed {len(episodes)} episodes "
                               f"({stock_idx+1}/{len(unique_stocks)} stocks)")
        
        return episodes
    
    def _run_single_episode(
        self,
        episode_data: np.ndarray,
        episode_metadata: np.ndarray,
        deterministic: bool
    ) -> Dict[str, Any]:
        """
        단일 에피소드 실행
        
        Args:
            episode_data: 에피소드 특징 데이터 (episode_length, n_features)
            episode_metadata: 에피소드 메타데이터 (episode_length, 3)
            deterministic: 결정적 행동 선택 여부
            
        Returns:
            에피소드 결과 딕셔너리
        """
        episode_length = len(episode_data)
        current_step = self.seq_len - 1  # 최소 seq_len만큼의 히스토리 필요
        
        # 에피소드 상태
        position = 0  # 0: 포지션 없음, 1: 매수 포지션
        entry_price = 0.0
        entry_time = 0.0
        
        # 에피소드 기록
        trades = []
        rewards = []
        quick_exit_violations = 0
        
        # 에피소드 실행
        while current_step < episode_length - 1:
            # 현재 관측값 (시퀀스) 추출
            start_idx = max(0, current_step - self.seq_len + 1)
            end_idx = current_step + 1
            sequence = episode_data[start_idx:end_idx]
            
            # 시퀀스가 seq_len보다 짧으면 패딩
            if len(sequence) < self.seq_len:
                padding = np.zeros((self.seq_len - len(sequence), sequence.shape[1]), dtype=np.float32)
                sequence = np.vstack([padding, sequence])
            
            # 행동 예측
            action, confidence = self.inference_engine.predict(
                sequence, deterministic=deterministic
            )
            
            # 현재 가격 및 시간
            current_price = float(episode_data[current_step, 0])  # 등락률을 가격으로 사용
            current_time = float(episode_metadata[current_step, 2])
            
            # 행동 실행
            reward = 0.0
            quick_exit_triggered = False
            
            if action == 1:  # 매수
                if position == 0:
                    # 슬리피지 적용
                    entry_price = self._apply_slippage(current_price, action)
                    entry_time = current_time
                    position = 1
            
            elif action == 2:  # 매도
                if position == 1:
                    # 슬리피지 적용
                    exit_price = self._apply_slippage(current_price, action)
                    holding_time = current_time - entry_time
                    
                    # 보상 계산
                    reward, reward_components = self._calculate_reward(
                        entry_price, exit_price, holding_time
                    )
                    
                    # 거래 기록
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'holding_time': holding_time,
                        'profit_rate': reward_components['profit_rate'],
                        'reward': reward,
                        'reward_components': reward_components,
                        'stock_code': str(episode_metadata[current_step, 0]),
                        'date': str(episode_metadata[current_step, 1]),
                        'time': current_time,
                        'confidence': confidence
                    })
                    
                    # 포지션 청산
                    position = 0
                    entry_price = 0.0
                    entry_time = 0.0
            
            elif action == 0:  # 보유
                # 빠른 손절 룰 체크
                if position == 1:
                    holding_time = current_time - entry_time
                    
                    # 임계값 이내이고 가격이 상승하지 않았는지 체크
                    if holding_time <= self.quick_exit_threshold and current_price <= entry_price:
                        # 빠른 손절 룰 위반
                        quick_exit_triggered = True
                        quick_exit_violations += 1
                        
                        # 자동 매도 및 페널티 적용
                        exit_price = self._apply_slippage(current_price, 2)
                        reward, reward_components = self._calculate_reward(
                            entry_price, exit_price, holding_time
                        )
                        
                        # 빠른 손절 룰 위반 페널티 추가
                        reward -= self.quick_exit_penalty
                        
                        # 거래 기록
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'holding_time': holding_time,
                            'profit_rate': reward_components['profit_rate'],
                            'reward': reward,
                            'reward_components': reward_components,
                            'quick_exit_violation': True,
                            'quick_exit_penalty': self.quick_exit_penalty,
                            'stock_code': str(episode_metadata[current_step, 0]),
                            'date': str(episode_metadata[current_step, 1]),
                            'time': current_time,
                            'confidence': confidence
                        })
                        
                        # 포지션 청산
                        position = 0
                        entry_price = 0.0
                        entry_time = 0.0
            
            # 보상 기록
            rewards.append(reward)
            
            # 다음 스텝으로 이동
            current_step += 1
        
        # 포지션이 남아있으면 강제 청산
        if position == 1:
            current_price = float(episode_data[current_step, 0])
            exit_price = self._apply_slippage(current_price, 2)
            holding_time = float(episode_metadata[current_step, 2]) - entry_time
            
            final_reward, final_components = self._calculate_reward(
                entry_price, exit_price, holding_time
            )
            rewards.append(final_reward)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'holding_time': holding_time,
                'profit_rate': final_components['profit_rate'],
                'reward': final_reward,
                'reward_components': final_components,
                'forced_liquidation': True,
                'stock_code': str(episode_metadata[current_step, 0]),
                'date': str(episode_metadata[current_step, 1]),
                'time': float(episode_metadata[current_step, 2]),
                'confidence': 0.0
            })
        
        # 에피소드 메타데이터 계산
        episode_result = self._calculate_episode_metadata(
            trades, rewards, quick_exit_violations,
            episode_metadata[0, 0], episode_metadata[0, 1]
        )
        
        return episode_result
    
    def _calculate_episode_metadata(
        self,
        trades: List[Dict[str, Any]],
        rewards: List[float],
        quick_exit_violations: int,
        stock_code: str,
        date: str
    ) -> Dict[str, Any]:
        """
        에피소드 메타데이터 계산
        
        Args:
            trades: 거래 데이터 리스트
            rewards: 보상 리스트
            quick_exit_violations: 빠른 손절 룰 위반 횟수
            stock_code: 종목 코드
            date: 날짜
            
        Returns:
            에피소드 메타데이터 딕셔너리
        """
        # 총 수익
        total_return = sum(rewards)
        
        # 거래 횟수
        num_trades = len(trades)
        
        # 평균 보유 시간
        if trades:
            avg_holding_time = np.mean([t['holding_time'] for t in trades])
        else:
            avg_holding_time = 0.0
        
        # 샤프 비율
        if len(rewards) > 1:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            sharpe_ratio = mean_reward / std_reward if std_reward > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 승률
        if trades:
            win_rate = np.mean([1 if t['reward'] > 0 else 0 for t in trades])
        else:
            win_rate = 0.0
        
        # 평균 거래당 수익
        if trades:
            avg_profit_per_trade = np.mean([t['reward'] for t in trades])
        else:
            avg_profit_per_trade = 0.0
        
        metadata = {
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'avg_holding_time': float(avg_holding_time),
            'sharpe_ratio': float(sharpe_ratio),
            'quick_exit_violations': int(quick_exit_violations),
            'win_rate': float(win_rate),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'trades': trades,
            'stock_code': str(stock_code),
            'date': str(date)
        }
        
        return metadata
    
    def _generate_summary(
        self,
        episodes: List[Dict[str, Any]],
        metrics: Dict[str, float],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        백테스팅 요약 통계 생성
        
        Args:
            episodes: 에피소드 데이터 리스트
            metrics: 평가 메트릭
            trades: 모든 거래 데이터
            
        Returns:
            요약 통계 딕셔너리
        """
        # 종목 수
        unique_stocks = set(ep['stock_code'] for ep in episodes)
        num_stocks = len(unique_stocks)
        
        # 거래일 수
        unique_dates = set(ep['date'] for ep in episodes)
        num_days = len(unique_dates)
        
        # 슬리피지 영향 계산
        total_slippage_cost = 0.0
        for trade in trades:
            # 슬리피지는 매수/매도 각각 slippage_rate만큼 발생
            # 총 슬리피지 비용 = 2 * slippage_rate
            total_slippage_cost += 2 * self.slippage_rate
        
        avg_slippage_per_trade = total_slippage_cost / len(trades) if trades else 0.0
        
        # 거래 비용 영향 계산
        total_transaction_cost = len(trades) * self.round_trip_cost
        avg_transaction_cost_per_trade = self.round_trip_cost
        
        # 빠른 손절 룰 위반 통계
        quick_exit_violations = sum(ep['quick_exit_violations'] for ep in episodes)
        quick_exit_violation_rate = quick_exit_violations / len(trades) if trades else 0.0
        
        summary = {
            'num_episodes': len(episodes),
            'num_stocks': num_stocks,
            'num_days': num_days,
            'total_trades': len(trades),
            'total_slippage_cost': float(total_slippage_cost),
            'avg_slippage_per_trade': float(avg_slippage_per_trade),
            'total_transaction_cost': float(total_transaction_cost),
            'avg_transaction_cost_per_trade': float(avg_transaction_cost_per_trade),
            'quick_exit_violations': int(quick_exit_violations),
            'quick_exit_violation_rate': float(quick_exit_violation_rate),
            'metrics': metrics
        }
        
        return summary
    
    def _print_backtest_results(
        self,
        summary: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        백테스팅 결과 출력
        
        Args:
            summary: 요약 통계
            metrics: 평가 메트릭
        """
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Test Period:             {summary['num_days']} days")
        print(f"Stocks Tested:           {summary['num_stocks']}")
        print(f"Total Episodes:          {summary['num_episodes']}")
        print(f"Total Trades:            {summary['total_trades']}")
        print("-" * 70)
        print("PERFORMANCE METRICS")
        print("-" * 70)
        print(f"Total Return:            {metrics['total_return']:.4f}")
        print(f"Avg Episode Return:      {metrics['avg_episode_return']:.4f}")
        print(f"Win Rate:                {metrics['win_rate']:.2%}")
        print(f"Avg Profit per Trade:    {metrics['avg_profit_per_trade']:.4f}")
        print(f"Max Drawdown:            {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio:            {metrics['sharpe_ratio']:.4f}")
        print(f"Avg Holding Time:        {metrics['avg_holding_time']:.2f} seconds")
        print("-" * 70)
        print("COST ANALYSIS")
        print("-" * 70)
        print(f"Total Transaction Cost:  {summary['total_transaction_cost']:.4f} "
              f"({summary['avg_transaction_cost_per_trade']*100:.3f}% per trade)")
        print(f"Total Slippage Cost:     {summary['total_slippage_cost']:.4f} "
              f"({summary['avg_slippage_per_trade']*100:.3f}% per trade)")
        print(f"Quick Exit Violations:   {summary['quick_exit_violations']} "
              f"({summary['quick_exit_violation_rate']:.2%} of trades)")
        print("=" * 70 + "\n")
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        백테스팅 결과 저장
        
        Args:
            results: 백테스팅 결과
            output_path: 출력 파일 경로 (.json)
        """
        import json
        
        # numpy 타입을 Python 기본 타입으로 변환
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        # JSON 파일로 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Backtest results saved to {output_path}")
    
    def close(self):
        """리소스 정리"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def run_backtest(
    embedding_model_path: str,
    policy_path: str,
    db_path: str,
    table_name: str = 'datasets',
    test_start_date: Optional[str] = None,
    test_end_date: Optional[str] = None,
    stock_codes: Optional[List[str]] = None,
    seq_len: int = 60,
    transaction_cost_rate: float = 0.00215,
    slippage_rate: float = 0.0001,
    quick_exit_threshold: float = 1.5,
    quick_exit_penalty: float = 0.01,
    device: str = 'cuda',
    output_path: Optional[str] = None,
    verbose: bool = True,
    alert_thresholds: Optional[List[AlertThreshold]] = None,
    alert_log_file: Optional[str] = None,
    enable_alerts: bool = True
) -> Dict[str, Any]:
    """
    백테스팅 실행 (편의 함수)
    
    Args:
        embedding_model_path: 임베딩 모델 체크포인트 경로
        policy_path: 정책 체크포인트 경로
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명
        test_start_date: 테스트 시작 날짜 (YYYY-MM-DD)
        test_end_date: 테스트 종료 날짜 (YYYY-MM-DD)
        stock_codes: 테스트할 종목 코드 리스트
        seq_len: 시퀀스 길이
        transaction_cost_rate: 거래 비용 비율
        slippage_rate: 슬리피지 비율
        quick_exit_threshold: 빠른 손절 시간 임계값 (초)
        quick_exit_penalty: 빠른 손절 룰 위반 페널티
        device: 디바이스 ('cuda' 또는 'cpu')
        output_path: 결과 저장 경로 (.json)
        verbose: 진행 상황 출력 여부
        alert_thresholds: 알림 임계값 리스트 (None이면 기본값 사용)
        alert_log_file: 알림 로그 파일 경로 (선택적)
        enable_alerts: 알림 시스템 활성화 여부
        
    Returns:
        백테스팅 결과 딕셔너리
    """
    # 추론 엔진 초기화
    inference_engine = GRPOInference(
        embedding_model_path=embedding_model_path,
        policy_path=policy_path,
        device=device
    )
    
    # 알림 시스템 초기화
    alert_system = None
    if enable_alerts:
        alert_system = PerformanceAlertSystem(
            thresholds=alert_thresholds,
            log_file=alert_log_file
        )
    
    # 백테스팅 시뮬레이터 초기화
    simulator = BacktestSimulator(
        inference_engine=inference_engine,
        db_path=db_path,
        table_name=table_name,
        seq_len=seq_len,
        transaction_cost_rate=transaction_cost_rate,
        slippage_rate=slippage_rate,
        quick_exit_threshold=quick_exit_threshold,
        quick_exit_penalty=quick_exit_penalty,
        alert_system=alert_system,
        enable_alerts=enable_alerts
    )
    
    try:
        # 백테스팅 실행
        results = simulator.run_backtest(
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            stock_codes=stock_codes,
            deterministic=True,
            verbose=verbose
        )
        
        # 결과 저장
        if output_path:
            simulator.save_results(results, output_path)
        
        return results
    
    finally:
        simulator.close()
