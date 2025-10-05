"""
GRPO 에이전트 평가 메트릭

이 모듈은 GRPO 에이전트의 성능을 평가하기 위한 메트릭을 계산합니다.
요구사항 7.2에 따라 승률, 거래당 평균 수익, 최대 낙폭, 샤프 비율, 평균 보유 시간을 계산합니다.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GRPOEvaluationMetrics:
    """
    GRPO 에이전트 평가 메트릭 계산기
    
    요구사항 7.2에 따라 다음 메트릭을 계산합니다:
    - 승률 (win_rate): 수익이 발생한 거래의 비율
    - 거래당 평균 수익 (avg_profit_per_trade): 모든 거래의 평균 수익률
    - 최대 낙폭 (max_drawdown): 누적 수익의 최대 하락폭
    - 샤프 비율 (sharpe_ratio): 위험 대비 수익률 (평균 수익 / 수익 표준편차)
    - 평균 보유 시간 (avg_holding_time): 거래당 평균 포지션 보유 시간 (초)
    
    Args:
        episodes: 에피소드 데이터 리스트 (환경에서 수집된 데이터)
    """
    
    def __init__(self, episodes: Optional[List[Dict[str, Any]]] = None):
        self.episodes = episodes or []
        self._metrics_cache = None
    
    def add_episode(self, episode: Dict[str, Any]):
        """
        평가할 에피소드 추가
        
        Args:
            episode: 에피소드 데이터 (환경의 step 메서드에서 반환된 info['episode'])
        """
        self.episodes.append(episode)
        self._metrics_cache = None  # 캐시 무효화
    
    def add_episodes(self, episodes: List[Dict[str, Any]]):
        """
        여러 에피소드 추가
        
        Args:
            episodes: 에피소드 데이터 리스트
        """
        self.episodes.extend(episodes)
        self._metrics_cache = None  # 캐시 무효화
    
    def clear(self):
        """에피소드 데이터 초기화"""
        self.episodes = []
        self._metrics_cache = None
    
    def compute_metrics(self, force_recompute: bool = False) -> Dict[str, float]:
        """
        모든 평가 메트릭 계산
        
        Args:
            force_recompute: 캐시를 무시하고 강제로 재계산
            
        Returns:
            평가 메트릭 딕셔너리:
            - win_rate: 승률 (0.0 ~ 1.0)
            - avg_profit_per_trade: 거래당 평균 수익률
            - max_drawdown: 최대 낙폭 (0.0 ~ 1.0)
            - sharpe_ratio: 샤프 비율
            - avg_holding_time: 평균 보유 시간 (초)
            - total_trades: 총 거래 횟수
            - total_episodes: 총 에피소드 수
            - total_return: 총 수익률
            - avg_episode_return: 에피소드당 평균 수익률
        """
        # 캐시된 메트릭 반환
        if not force_recompute and self._metrics_cache is not None:
            return self._metrics_cache
        
        if not self.episodes:
            logger.warning("No episodes to evaluate")
            return self._empty_metrics()
        
        # 모든 거래 데이터 수집
        all_trades = self._collect_all_trades()
        
        if not all_trades:
            logger.warning("No trades found in episodes")
            return self._empty_metrics()
        
        # 각 메트릭 계산
        metrics = {
            'win_rate': self._compute_win_rate(all_trades),
            'avg_profit_per_trade': self._compute_avg_profit_per_trade(all_trades),
            'max_drawdown': self._compute_max_drawdown(all_trades),
            'sharpe_ratio': self._compute_sharpe_ratio(all_trades),
            'avg_holding_time': self._compute_avg_holding_time(all_trades),
            'total_trades': len(all_trades),
            'total_episodes': len(self.episodes),
            'total_return': self._compute_total_return(all_trades),
            'avg_episode_return': self._compute_avg_episode_return()
        }
        
        # 캐시 저장
        self._metrics_cache = metrics
        
        logger.info(f"Computed evaluation metrics: "
                   f"win_rate={metrics['win_rate']:.2%}, "
                   f"avg_profit={metrics['avg_profit_per_trade']:.4f}, "
                   f"max_drawdown={metrics['max_drawdown']:.2%}, "
                   f"sharpe_ratio={metrics['sharpe_ratio']:.4f}, "
                   f"avg_holding_time={metrics['avg_holding_time']:.2f}s")
        
        return metrics
    
    def _collect_all_trades(self) -> List[Dict[str, Any]]:
        """
        모든 에피소드에서 거래 데이터 수집
        
        Returns:
            거래 데이터 리스트
            각 거래는 다음을 포함:
            - entry_price: 진입 가격
            - exit_price: 청산 가격
            - holding_time: 보유 시간 (초)
            - profit_rate: 수익률
            - reward: 보상 (거래 비용 포함)
        """
        all_trades = []
        
        for episode in self.episodes:
            # 에피소드 메타데이터에서 거래 정보 추출
            # 환경의 _calculate_episode_metadata에서 생성된 데이터 사용
            
            # 방법 1: 에피소드에 직접 저장된 거래 데이터 사용
            if 'trades' in episode:
                all_trades.extend(episode['trades'])
            
            # 방법 2: 에피소드 메타데이터에서 거래 정보 재구성
            # (환경에서 거래 데이터를 직접 제공하지 않는 경우)
            elif 'num_trades' in episode and episode['num_trades'] > 0:
                # 에피소드 수준의 집계 데이터만 있는 경우
                # 개별 거래 데이터를 재구성할 수 없으므로 에피소드 평균 사용
                avg_profit = episode.get('avg_profit_per_trade', 0.0)
                avg_holding_time = episode.get('avg_holding_time', 0.0)
                num_trades = episode['num_trades']
                
                # 에피소드의 평균 데이터로 가상 거래 생성
                # 이는 근사치이며, 실제 거래 데이터가 있는 것이 더 정확합니다
                for _ in range(num_trades):
                    all_trades.append({
                        'profit_rate': avg_profit,
                        'reward': avg_profit,
                        'holding_time': avg_holding_time,
                        'entry_price': 100.0,  # 더미 값
                        'exit_price': 100.0 * (1 + avg_profit)  # 더미 값
                    })
        
        return all_trades
    
    def _compute_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """
        승률 계산
        
        승률 = (수익이 발생한 거래 수) / (총 거래 수)
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            승률 (0.0 ~ 1.0)
        """
        if not trades:
            return 0.0
        
        # 수익이 발생한 거래 수 (reward > 0)
        winning_trades = sum(1 for trade in trades if trade.get('reward', 0.0) > 0)
        
        win_rate = winning_trades / len(trades)
        
        return float(win_rate)
    
    def _compute_avg_profit_per_trade(self, trades: List[Dict[str, Any]]) -> float:
        """
        거래당 평균 수익 계산
        
        거래당 평균 수익 = sum(rewards) / num_trades
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            거래당 평균 수익률
        """
        if not trades:
            return 0.0
        
        # 모든 거래의 보상 합계
        total_profit = sum(trade.get('reward', 0.0) for trade in trades)
        
        avg_profit = total_profit / len(trades)
        
        return float(avg_profit)
    
    def _compute_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """
        최대 낙폭 계산
        
        최대 낙폭 = max((peak - trough) / peak)
        
        누적 수익의 최고점에서 최저점까지의 최대 하락폭을 계산합니다.
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            최대 낙폭 (0.0 ~ 1.0)
        """
        if not trades:
            return 0.0
        
        # 누적 수익 계산 (초기 자본 1.0에서 시작)
        cumulative_returns = []
        cumulative_return = 1.0  # 초기 자본
        
        for trade in trades:
            # 수익률을 곱셈으로 누적 (복리 효과)
            reward = trade.get('reward', 0.0)
            cumulative_return *= (1.0 + reward)
            cumulative_returns.append(cumulative_return)
        
        cumulative_returns = np.array(cumulative_returns)
        
        # 최대 낙폭 계산
        # 각 시점에서의 최고점 (running maximum)
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # 낙폭 = (최고점 - 현재값) / 최고점
        drawdowns = (running_max - cumulative_returns) / running_max
        
        # 최대 낙폭
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return float(max_drawdown)
    
    def _compute_sharpe_ratio(self, trades: List[Dict[str, Any]]) -> float:
        """
        샤프 비율 계산
        
        샤프 비율 = (평균 수익 - 무위험 수익률) / 수익 표준편차
        
        스캘핑의 경우 무위험 수익률은 0으로 가정합니다.
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            샤프 비율
        """
        if not trades or len(trades) < 2:
            return 0.0
        
        # 거래별 수익률
        returns = np.array([trade.get('reward', 0.0) for trade in trades])
        
        # 평균 수익률
        mean_return = np.mean(returns)
        
        # 수익률 표준편차
        std_return = np.std(returns, ddof=1)  # 표본 표준편차
        
        # 표준편차가 0이면 샤프 비율은 0
        if std_return == 0:
            return 0.0
        
        # 샤프 비율 = 평균 수익 / 표준편차
        # (무위험 수익률 = 0 가정)
        sharpe_ratio = mean_return / std_return
        
        return float(sharpe_ratio)
    
    def _compute_avg_holding_time(self, trades: List[Dict[str, Any]]) -> float:
        """
        평균 보유 시간 계산
        
        평균 보유 시간 = sum(holding_times) / num_trades
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            평균 보유 시간 (초)
        """
        if not trades:
            return 0.0
        
        # 모든 거래의 보유 시간 합계
        total_holding_time = sum(trade.get('holding_time', 0.0) for trade in trades)
        
        avg_holding_time = total_holding_time / len(trades)
        
        return float(avg_holding_time)
    
    def _compute_total_return(self, trades: List[Dict[str, Any]]) -> float:
        """
        총 수익률 계산
        
        Args:
            trades: 거래 데이터 리스트
            
        Returns:
            총 수익률
        """
        if not trades:
            return 0.0
        
        total_return = sum(trade.get('reward', 0.0) for trade in trades)
        
        return float(total_return)
    
    def _compute_avg_episode_return(self) -> float:
        """
        에피소드당 평균 수익률 계산
        
        Returns:
            에피소드당 평균 수익률
        """
        if not self.episodes:
            return 0.0
        
        # 에피소드별 총 수익
        episode_returns = []
        
        for episode in self.episodes:
            # 에피소드 메타데이터에서 총 수익 추출
            if 'total_return' in episode:
                episode_returns.append(episode['total_return'])
            elif 'episode_reward' in episode:
                episode_returns.append(episode['episode_reward'])
            else:
                # 거래 데이터에서 계산
                if 'trades' in episode:
                    episode_return = sum(trade.get('reward', 0.0) for trade in episode['trades'])
                    episode_returns.append(episode_return)
        
        if not episode_returns:
            return 0.0
        
        avg_episode_return = np.mean(episode_returns)
        
        return float(avg_episode_return)
    
    def _empty_metrics(self) -> Dict[str, float]:
        """
        빈 메트릭 딕셔너리 반환
        
        Returns:
            모든 메트릭이 0인 딕셔너리
        """
        return {
            'win_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_holding_time': 0.0,
            'total_trades': 0,
            'total_episodes': 0,
            'total_return': 0.0,
            'avg_episode_return': 0.0
        }
    
    def print_metrics(self, metrics: Optional[Dict[str, float]] = None):
        """
        메트릭을 보기 좋게 출력
        
        Args:
            metrics: 출력할 메트릭 (None이면 compute_metrics 호출)
        """
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "=" * 60)
        print("GRPO Agent Evaluation Metrics")
        print("=" * 60)
        print(f"Total Episodes:          {metrics['total_episodes']}")
        print(f"Total Trades:            {metrics['total_trades']}")
        print(f"Total Return:            {metrics['total_return']:.4f}")
        print(f"Avg Episode Return:      {metrics['avg_episode_return']:.4f}")
        print("-" * 60)
        print(f"Win Rate:                {metrics['win_rate']:.2%}")
        print(f"Avg Profit per Trade:    {metrics['avg_profit_per_trade']:.4f}")
        print(f"Max Drawdown:            {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio:            {metrics['sharpe_ratio']:.4f}")
        print(f"Avg Holding Time:        {metrics['avg_holding_time']:.2f} seconds")
        print("=" * 60 + "\n")


def evaluate_grpo_agent(
    episodes: List[Dict[str, Any]],
    print_results: bool = True
) -> Dict[str, float]:
    """
    GRPO 에이전트 평가 (편의 함수)
    
    Args:
        episodes: 에피소드 데이터 리스트
        print_results: 결과를 출력할지 여부
        
    Returns:
        평가 메트릭 딕셔너리
    """
    evaluator = GRPOEvaluationMetrics(episodes)
    metrics = evaluator.compute_metrics()
    
    if print_results:
        evaluator.print_metrics(metrics)
    
    return metrics
