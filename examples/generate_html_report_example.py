"""
HTML 보고서 생성 예제

임베딩 모델과 GRPO 훈련 결과를 HTML 보고서로 생성하는 예제입니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.reporting.html_report import (
    generate_embedding_report,
    generate_grpo_report,
    generate_combined_report
)


def example_embedding_report():
    """임베딩 모델 보고서 생성 예제"""
    print("=" * 80)
    print("임베딩 모델 보고서 생성 예제")
    print("=" * 80)
    
    # 예제 데이터
    training_metadata = {
        'timestamp': '2025-10-05 14:30:00',
        'epoch': 50,
    }
    
    final_metrics = {
        'train_loss': 0.234,
        'val_loss': 0.267,
        'silhouette_score': 0.623,
        'temporal_coherence': 0.781
    }
    
    config = {
        'input_dim': 60,
        'embedding_dim': 128,
        'seq_len': 60,
        'num_heads': 4,
        'batch_size': 128,
        'learning_rate': 0.0001
    }
    
    # 보고서 생성
    output_path = 'models/embedding/training_report.html'
    tensorboard_dir = 'models/embedding/tensorboard_logs'
    
    report_path = generate_embedding_report(
        output_path=output_path,
        training_metadata=training_metadata,
        final_metrics=final_metrics,
        config=config,
        tensorboard_dir=tensorboard_dir
    )
    
    print(f"\n✅ 임베딩 보고서 생성 완료: {report_path}")
    print(f"   브라우저에서 열어보세요: file:///{Path(report_path).absolute()}")


def example_grpo_report():
    """GRPO 훈련 보고서 생성 예제"""
    print("\n" + "=" * 80)
    print("GRPO 훈련 보고서 생성 예제")
    print("=" * 80)
    
    # 예제 데이터
    training_metadata = {
        'timestamp': '2025-10-05 18:45:00',
        'total_timesteps': 1000000,
        'num_updates': 5000,
        'quick_exit_threshold': 1.5,
        'quick_exit_penalty': 0.01
    }
    
    final_metrics = {
        'mean_reward': 0.0234,
        'win_rate': 0.58,
        'sharpe_ratio': 1.42,
        'max_drawdown': 0.067,
        'avg_holding_time': 24.3,
        'avg_profit_per_trade': 0.0067,
        'quick_exit_violations': 234,
        'total_trades': 1500
    }
    
    config = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'clip_epsilon': 0.2,
        'episodes_per_group': 12,
        'num_groups': 4
    }
    
    # 샘플 에피소드
    sample_episodes = [
        {
            'total_profit': 0.0123,
            'num_trades': 5,
            'avg_holding_time': 18.5,
            'sharpe_ratio': 1.67,
            'quick_exit_violations': 0,
            'stock_code': '005930',
            'start_time': '2025-10-05 09:00:00'
        },
        {
            'total_profit': -0.0045,
            'num_trades': 3,
            'avg_holding_time': 32.1,
            'sharpe_ratio': 0.89,
            'quick_exit_violations': 1,
            'stock_code': '000660',
            'start_time': '2025-10-05 10:15:00'
        },
        {
            'total_profit': 0.0089,
            'num_trades': 4,
            'avg_holding_time': 21.3,
            'sharpe_ratio': 1.45,
            'quick_exit_violations': 0,
            'stock_code': '035720',
            'start_time': '2025-10-05 11:30:00'
        }
    ]
    
    # 보고서 생성
    output_path = 'models/grpo_scalping/training_report.html'
    tensorboard_dir = 'models/grpo_scalping/tensorboard_logs'
    
    report_path = generate_grpo_report(
        output_path=output_path,
        training_metadata=training_metadata,
        final_metrics=final_metrics,
        config=config,
        sample_episodes=sample_episodes,
        tensorboard_dir=tensorboard_dir
    )
    
    print(f"\n✅ GRPO 보고서 생성 완료: {report_path}")
    print(f"   브라우저에서 열어보세요: file:///{Path(report_path).absolute()}")


def example_combined_report():
    """통합 보고서 생성 예제"""
    print("\n" + "=" * 80)
    print("통합 보고서 생성 예제")
    print("=" * 80)
    
    # 임베딩 데이터
    embedding_metadata = {
        'timestamp': '2025-10-05 14:30:00',
        'epoch': 50,
    }
    
    embedding_metrics = {
        'val_loss': 0.267,
        'silhouette_score': 0.623,
        'temporal_coherence': 0.781
    }
    
    # GRPO 데이터
    grpo_metadata = {
        'timestamp': '2025-10-05 18:45:00',
        'total_timesteps': 1000000,
        'num_updates': 5000
    }
    
    grpo_metrics = {
        'win_rate': 0.58,
        'sharpe_ratio': 1.42,
        'avg_holding_time': 24.3,
        'avg_profit_per_trade': 0.0067
    }
    
    # 샘플 에피소드
    sample_episodes = [
        {
            'total_profit': 0.0123,
            'num_trades': 5,
            'avg_holding_time': 18.5
        },
        {
            'total_profit': 0.0089,
            'num_trades': 4,
            'avg_holding_time': 21.3
        }
    ]
    
    # 보고서 생성
    output_path = 'models/combined_training_report.html'
    
    report_path = generate_combined_report(
        output_path=output_path,
        embedding_metadata=embedding_metadata,
        embedding_metrics=embedding_metrics,
        grpo_metadata=grpo_metadata,
        grpo_metrics=grpo_metrics,
        sample_episodes=sample_episodes
    )
    
    print(f"\n✅ 통합 보고서 생성 완료: {report_path}")
    print(f"   브라우저에서 열어보세요: file:///{Path(report_path).absolute()}")


if __name__ == '__main__':
    # 각 예제 실행
    example_embedding_report()
    example_grpo_report()
    example_combined_report()
    
    print("\n" + "=" * 80)
    print("모든 보고서 생성 완료!")
    print("=" * 80)
