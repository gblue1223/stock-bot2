"""
GRPO 평가 메트릭 테스트

요구사항 7.2에 따른 GRPO 에이전트 평가 메트릭 구현을 테스트합니다.
"""

import pytest
import numpy as np
from ai_trader.grpo.evaluation import GRPOEvaluationMetrics, evaluate_grpo_agent


class TestGRPOEvaluationMetrics:
    """GRPO 평가 메트릭 테스트"""
    
    def test_empty_episodes(self):
        """빈 에피소드 리스트 처리 테스트"""
        evaluator = GRPOEvaluationMetrics([])
        metrics = evaluator.compute_metrics()
        
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_profit_per_trade'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['avg_holding_time'] == 0.0
        assert metrics['total_trades'] == 0
        assert metrics['total_episodes'] == 0
    
    def test_win_rate_calculation(self):
        """승률 계산 테스트"""
        # 5개 거래: 3개 수익, 2개 손실
        episodes = [{
            'trades': [
                {'reward': 0.01, 'holding_time': 10.0},  # 승
                {'reward': -0.005, 'holding_time': 5.0},  # 패
                {'reward': 0.02, 'holding_time': 15.0},  # 승
                {'reward': -0.01, 'holding_time': 8.0},  # 패
                {'reward': 0.015, 'holding_time': 12.0},  # 승
            ],
            'total_return': 0.03,
            'num_trades': 5
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 승률 = 3/5 = 0.6
        assert metrics['win_rate'] == pytest.approx(0.6, abs=1e-6)
        assert metrics['total_trades'] == 5
    
    def test_avg_profit_per_trade(self):
        """거래당 평균 수익 계산 테스트"""
        episodes = [{
            'trades': [
                {'reward': 0.01, 'holding_time': 10.0},
                {'reward': -0.005, 'holding_time': 5.0},
                {'reward': 0.02, 'holding_time': 15.0},
            ],
            'total_return': 0.025,
            'num_trades': 3
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 평균 수익 = (0.01 - 0.005 + 0.02) / 3 = 0.025 / 3
        expected_avg = 0.025 / 3
        assert metrics['avg_profit_per_trade'] == pytest.approx(expected_avg, abs=1e-6)
    
    def test_max_drawdown_calculation(self):
        """최대 낙폭 계산 테스트"""
        # 복리 누적 수익 계산 (초기 자본 1.0)
        # 1.0 * 1.1 = 1.1
        # 1.1 * 1.05 = 1.155
        # 1.155 * 0.9 = 1.0395
        # 1.0395 * 1.05 = 1.091475
        # 1.091475 * 1.1 = 1.2006225
        episodes = [{
            'trades': [
                {'reward': 0.1, 'holding_time': 10.0},
                {'reward': 0.05, 'holding_time': 5.0},
                {'reward': -0.1, 'holding_time': 8.0},  # 낙폭 발생
                {'reward': 0.05, 'holding_time': 7.0},
                {'reward': 0.1, 'holding_time': 12.0},
            ],
            'total_return': 0.2,
            'num_trades': 5
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 최대 낙폭 계산 (복리 기준)
        # 누적: 1.1, 1.155, 1.0395, 1.091475, 1.2006225
        # 최고점: 1.1, 1.155, 1.155, 1.155, 1.2006225
        # 낙폭: 0, 0, (1.155-1.0395)/1.155=0.1, (1.155-1.091475)/1.155=0.055, 0
        # 최대 낙폭: 0.1
        expected_max_dd = (1.155 - 1.0395) / 1.155
        assert metrics['max_drawdown'] == pytest.approx(expected_max_dd, abs=1e-4)
    
    def test_sharpe_ratio_calculation(self):
        """샤프 비율 계산 테스트"""
        # 수익률: 0.01, 0.02, 0.015, 0.01, 0.02
        # 평균: 0.015
        # 표준편차: sqrt(sum((x - mean)^2) / (n-1))
        rewards = [0.01, 0.02, 0.015, 0.01, 0.02]
        episodes = [{
            'trades': [
                {'reward': r, 'holding_time': 10.0}
                for r in rewards
            ],
            'total_return': sum(rewards),
            'num_trades': len(rewards)
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 샤프 비율 = 평균 / 표준편차
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards, ddof=1)
        expected_sharpe = mean_reward / std_reward
        
        assert metrics['sharpe_ratio'] == pytest.approx(expected_sharpe, abs=1e-4)
    
    def test_sharpe_ratio_zero_std(self):
        """표준편차가 0일 때 샤프 비율 테스트"""
        # 모든 거래가 동일한 수익
        episodes = [{
            'trades': [
                {'reward': 0.01, 'holding_time': 10.0},
                {'reward': 0.01, 'holding_time': 10.0},
                {'reward': 0.01, 'holding_time': 10.0},
            ],
            'total_return': 0.03,
            'num_trades': 3
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 표준편차가 0이면 샤프 비율은 0
        assert metrics['sharpe_ratio'] == 0.0
    
    def test_avg_holding_time_calculation(self):
        """평균 보유 시간 계산 테스트"""
        episodes = [{
            'trades': [
                {'reward': 0.01, 'holding_time': 10.0},
                {'reward': 0.02, 'holding_time': 20.0},
                {'reward': 0.015, 'holding_time': 15.0},
            ],
            'total_return': 0.045,
            'num_trades': 3
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 평균 보유 시간 = (10 + 20 + 15) / 3 = 15.0
        assert metrics['avg_holding_time'] == pytest.approx(15.0, abs=1e-6)
    
    def test_multiple_episodes(self):
        """여러 에피소드 평가 테스트"""
        episodes = [
            {
                'trades': [
                    {'reward': 0.01, 'holding_time': 10.0},
                    {'reward': 0.02, 'holding_time': 15.0},
                ],
                'total_return': 0.03,
                'num_trades': 2
            },
            {
                'trades': [
                    {'reward': -0.005, 'holding_time': 5.0},
                    {'reward': 0.015, 'holding_time': 12.0},
                    {'reward': 0.01, 'holding_time': 8.0},
                ],
                'total_return': 0.02,
                'num_trades': 3
            }
        ]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 총 거래: 5개
        assert metrics['total_trades'] == 5
        assert metrics['total_episodes'] == 2
        
        # 총 수익: 0.03 + 0.02 = 0.05
        assert metrics['total_return'] == pytest.approx(0.05, abs=1e-6)
        
        # 에피소드당 평균 수익: 0.05 / 2 = 0.025
        assert metrics['avg_episode_return'] == pytest.approx(0.025, abs=1e-6)
        
        # 승률: 4/5 = 0.8
        assert metrics['win_rate'] == pytest.approx(0.8, abs=1e-6)
    
    def test_add_episode(self):
        """에피소드 추가 테스트"""
        evaluator = GRPOEvaluationMetrics()
        
        # 초기 상태
        metrics = evaluator.compute_metrics()
        assert metrics['total_episodes'] == 0
        
        # 에피소드 추가
        episode = {
            'trades': [
                {'reward': 0.01, 'holding_time': 10.0},
            ],
            'total_return': 0.01,
            'num_trades': 1
        }
        evaluator.add_episode(episode)
        
        # 메트릭 재계산
        metrics = evaluator.compute_metrics()
        assert metrics['total_episodes'] == 1
        assert metrics['total_trades'] == 1
    
    def test_add_episodes(self):
        """여러 에피소드 추가 테스트"""
        evaluator = GRPOEvaluationMetrics()
        
        episodes = [
            {
                'trades': [{'reward': 0.01, 'holding_time': 10.0}],
                'total_return': 0.01,
                'num_trades': 1
            },
            {
                'trades': [{'reward': 0.02, 'holding_time': 15.0}],
                'total_return': 0.02,
                'num_trades': 1
            }
        ]
        
        evaluator.add_episodes(episodes)
        
        metrics = evaluator.compute_metrics()
        assert metrics['total_episodes'] == 2
        assert metrics['total_trades'] == 2
    
    def test_clear(self):
        """에피소드 초기화 테스트"""
        episodes = [{
            'trades': [{'reward': 0.01, 'holding_time': 10.0}],
            'total_return': 0.01,
            'num_trades': 1
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        
        # 초기 메트릭
        metrics = evaluator.compute_metrics()
        assert metrics['total_episodes'] == 1
        
        # 초기화
        evaluator.clear()
        
        # 초기화 후 메트릭
        metrics = evaluator.compute_metrics()
        assert metrics['total_episodes'] == 0
        assert metrics['total_trades'] == 0
    
    def test_metrics_caching(self):
        """메트릭 캐싱 테스트"""
        episodes = [{
            'trades': [{'reward': 0.01, 'holding_time': 10.0}],
            'total_return': 0.01,
            'num_trades': 1
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        
        # 첫 번째 계산
        metrics1 = evaluator.compute_metrics()
        
        # 두 번째 계산 (캐시 사용)
        metrics2 = evaluator.compute_metrics()
        
        # 동일한 결과
        assert metrics1 == metrics2
        
        # 강제 재계산
        metrics3 = evaluator.compute_metrics(force_recompute=True)
        assert metrics1 == metrics3
    
    def test_episode_metadata_format(self):
        """에피소드 메타데이터 형식 테스트 (환경과의 호환성)"""
        # 환경에서 생성하는 메타데이터 형식
        episodes = [{
            'total_return': 0.05,
            'num_trades': 3,
            'avg_holding_time': 12.5,
            'sharpe_ratio': 1.2,
            'quick_exit_violations': 1,
            'win_rate': 0.667,
            'avg_profit_per_trade': 0.0167,
            'episode_length': 100,
            'steps_taken': 95,
            'trades': [
                {'reward': 0.02, 'holding_time': 10.0, 'profit_rate': 0.025},
                {'reward': 0.01, 'holding_time': 15.0, 'profit_rate': 0.015},
                {'reward': 0.02, 'holding_time': 12.5, 'profit_rate': 0.025},
            ]
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 메트릭이 올바르게 계산되는지 확인
        assert metrics['total_episodes'] == 1
        assert metrics['total_trades'] == 3
        assert metrics['total_return'] == pytest.approx(0.05, abs=1e-6)
        assert metrics['win_rate'] == pytest.approx(1.0, abs=1e-6)  # 모든 거래가 수익
    
    def test_negative_returns_max_drawdown(self):
        """음수 수익에서 최대 낙폭 테스트"""
        # 모든 거래가 손실인 경우
        episodes = [{
            'trades': [
                {'reward': -0.01, 'holding_time': 10.0},
                {'reward': -0.02, 'holding_time': 15.0},
                {'reward': -0.015, 'holding_time': 12.0},
            ],
            'total_return': -0.045,
            'num_trades': 3
        }]
        
        evaluator = GRPOEvaluationMetrics(episodes)
        metrics = evaluator.compute_metrics()
        
        # 승률 = 0
        assert metrics['win_rate'] == 0.0
        
        # 평균 수익 < 0
        assert metrics['avg_profit_per_trade'] < 0
        
        # 최대 낙폭 계산 (복리 기준)
        # 누적: 1.0 * 0.99 = 0.99, 0.99 * 0.98 = 0.9702, 0.9702 * 0.985 = 0.9556
        # 최고점: 1.0 (초기), 1.0, 1.0
        # 낙폭: (1.0-0.99)/1.0=0.01, (1.0-0.9702)/1.0=0.0298, (1.0-0.9556)/1.0=0.0444
        # 하지만 running_max는 누적 수익의 최대값이므로:
        # running_max: 0.99, 0.99, 0.99
        # 낙폭: 0, (0.99-0.9702)/0.99=0.02, (0.99-0.9556)/0.99=0.0347
        # 최대 낙폭: 0.0347
        cumulative = [0.99, 0.99 * 0.98, 0.99 * 0.98 * 0.985]
        running_max = [cumulative[0], cumulative[0], cumulative[0]]
        expected_max_dd = (running_max[2] - cumulative[2]) / running_max[2]
        assert metrics['max_drawdown'] == pytest.approx(expected_max_dd, abs=1e-4)


def test_evaluate_grpo_agent_convenience_function():
    """편의 함수 테스트"""
    episodes = [{
        'trades': [
            {'reward': 0.01, 'holding_time': 10.0},
            {'reward': 0.02, 'holding_time': 15.0},
        ],
        'total_return': 0.03,
        'num_trades': 2
    }]
    
    # 출력 없이 평가
    metrics = evaluate_grpo_agent(episodes, print_results=False)
    
    assert metrics['total_episodes'] == 1
    assert metrics['total_trades'] == 2
    assert metrics['win_rate'] == 1.0
