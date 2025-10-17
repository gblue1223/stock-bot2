#!/usr/bin/env python3
"""
실시간 데이터 수집 테스트

시퀀스 버퍼 및 추론 파이프라인 테스트
"""

import os
import sys
import logging
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from koapys.client import KoapyRestSimple
from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Position, Action

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeDataCollector:
    """실시간 데이터 수집기"""
    
    def __init__(
        self,
        stock_code: str,
        seq_len: int = 30,
        num_features: int = 24,
        update_interval: float = 1.0
    ):
        self.stock_code = stock_code
        self.seq_len = seq_len
        self.num_features = num_features
        self.update_interval = update_interval
        
        # 데이터 버퍼
        self.buffer = deque(maxlen=seq_len)
        
        # Koapys 클라이언트
        self.koapys = KoapyRestSimple(
            base_url=os.getenv('KOAPYS_BASE_URL', 'http://localhost:5000'),
            simulation=True
        )
        
        # 통계
        self.stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'extraction_errors': 0
        }
    
    def extract_features(self, market_data: dict) -> np.ndarray:
        """특징 추출"""
        try:
            features = []
            
            # 1. 등락률
            current_price = float(market_data.get('현재가', 0))
            base_price = float(market_data.get('기준가', current_price))
            change_rate = (current_price - base_price) / base_price * 100 if base_price > 0 else 0.0
            features.append(change_rate)
            
            # 2. 누적거래대금
            volume_amount = float(market_data.get('누적거래대금', 0))
            features.append(volume_amount / 1e9)
            
            # 3. 거래회전율
            volume = float(market_data.get('거래량', 0))
            listed_shares = float(market_data.get('상장주식수', 1))
            turnover_rate = (volume / listed_shares * 100) if listed_shares > 0 else 0.0
            features.append(turnover_rate)
            
            # 4. 체결강도
            buy_volume = float(market_data.get('매수체결량', 0))
            sell_volume = float(market_data.get('매도체결량', 1))
            strength = (buy_volume / sell_volume * 100) if sell_volume > 0 else 100.0
            features.append(strength)
            
            # 5-14. 매도대기금액
            for i in range(1, 11):
                sell_price = float(market_data.get(f'매도호가{i}', 0))
                sell_qty = float(market_data.get(f'매도호가수량{i}', 0))
                features.append(sell_price * sell_qty / 1e6)
            
            # 15-24. 매수대기금액
            for i in range(1, 11):
                buy_price = float(market_data.get(f'매수호가{i}', 0))
                buy_qty = float(market_data.get(f'매수호가수량{i}', 0))
                features.append(buy_price * buy_qty / 1e6)
            
            features_array = np.array(features, dtype=np.float32)
            
            # 길이 확인
            if len(features_array) != self.num_features:
                if len(features_array) < self.num_features:
                    features_array = np.pad(
                        features_array,
                        (0, self.num_features - len(features_array)),
                        mode='constant'
                    )
                else:
                    features_array = features_array[:self.num_features]
            
            return features_array
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            self.stats['extraction_errors'] += 1
            return np.zeros(self.num_features, dtype=np.float32)
    
    def update(self):
        """데이터 업데이트"""
        self.stats['total_updates'] += 1
        
        try:
            # 시장 데이터 조회
            market_data = self.koapys.get_stock_basic_info(self.stock_code)
            
            # 특징 추출
            features = self.extract_features(market_data)
            
            # 버퍼에 추가
            self.buffer.append(features)
            
            self.stats['successful_updates'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            self.stats['failed_updates'] += 1
            return False
    
    def get_sequence(self) -> np.ndarray:
        """현재 시퀀스 반환"""
        if len(self.buffer) < self.seq_len:
            # 패딩
            padding = np.zeros((self.seq_len - len(self.buffer), self.num_features), dtype=np.float32)
            sequence = np.vstack([padding, np.array(list(self.buffer))])
        else:
            sequence = np.array(list(self.buffer))
        
        return sequence
    
    def is_ready(self) -> bool:
        """시퀀스 준비 여부"""
        return len(self.buffer) >= self.seq_len
    
    def close(self):
        """리소스 정리"""
        self.koapys.close()


def test_data_collection(stock_code: str = '005930', duration: int = 60):
    """데이터 수집 테스트"""
    logger.info("=" * 80)
    logger.info(f"[TEST] Realtime Data Collection ({stock_code})")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration} seconds")
    logger.info(f"Update Interval: 1.0 seconds")
    logger.info("=" * 80)
    
    collector = RealtimeDataCollector(
        stock_code=stock_code,
        seq_len=30,
        num_features=24,
        update_interval=1.0
    )
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # 데이터 업데이트
            success = collector.update()
            
            # 상태 출력
            elapsed = int(time.time() - start_time)
            buffer_size = len(collector.buffer)
            is_ready = collector.is_ready()
            
            logger.info(
                f"[{elapsed:3d}s] Buffer: {buffer_size:2d}/30, "
                f"Ready: {is_ready}, "
                f"Success: {success}"
            )
            
            # 시퀀스 준비되면 샘플 출력
            if is_ready and elapsed % 10 == 0:
                sequence = collector.get_sequence()
                logger.info(f"  Sequence shape: {sequence.shape}")
                logger.info(f"  Last features: {sequence[-1, :5]}")
            
            # 대기
            time.sleep(1.0)
        
        # 최종 통계
        logger.info("\n" + "=" * 80)
        logger.info("[STATS] Collection Statistics")
        logger.info("=" * 80)
        logger.info(f"Total Updates: {collector.stats['total_updates']}")
        logger.info(f"Successful: {collector.stats['successful_updates']}")
        logger.info(f"Failed: {collector.stats['failed_updates']}")
        logger.info(f"Extraction Errors: {collector.stats['extraction_errors']}")
        
        success_rate = collector.stats['successful_updates'] / collector.stats['total_updates'] * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        collector.close()
        return True
        
    except KeyboardInterrupt:
        logger.info("\n[STOP] Interrupted by user")
        collector.close()
        return False
    except Exception as e:
        logger.error(f"[ERROR] Test failed: {e}", exc_info=True)
        collector.close()
        return False


def test_inference_pipeline(stock_code: str = '005930'):
    """추론 파이프라인 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info(f"[TEST] Inference Pipeline ({stock_code})")
    logger.info("=" * 80)
    
    try:
        # 데이터 수집기
        collector = RealtimeDataCollector(stock_code=stock_code)
        
        # 추론 엔진
        logger.info("Loading model...")
        base_inference = GRPOInference(
            policy_path='models/grpo_direct_features@20251016/direct_features_model.pt',
            embedding_model_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_torchscript=False
        )
        
        inference = EnhancedGRPOInference(
            base_inference=base_inference,
            min_buy_confidence=0.0,
            stop_loss_rate=-2.0,
            take_profit_rate=5.0,
            max_holding_period=100,
            enable_auto_exit=True
        )
        
        logger.info("[OK] Model loaded")
        
        # 시퀀스 수집 (30개)
        logger.info("\nCollecting sequence...")
        for i in range(30):
            collector.update()
            logger.info(f"  [{i+1}/30] Collected")
            time.sleep(0.5)
        
        # 추론 테스트
        logger.info("\n[TEST] Running inference...")
        sequence = collector.get_sequence()
        current_price = sequence[-1, 0]  # 등락률
        
        action, confidence, pred_info = inference.predict(
            sequence=sequence,
            current_position=None,
            current_price=current_price,
            deterministic=True
        )
        
        action_name = Action(action).name
        logger.info(f"\n[RESULT] Prediction:")
        logger.info(f"  Action: {action_name}")
        logger.info(f"  Confidence: {confidence:.3f}")
        logger.info(f"  Raw Action: {Action(pred_info['raw_action']).name}")
        logger.info(f"  Raw Confidence: {pred_info['raw_confidence']:.3f}")
        logger.info(f"  Filtered: {pred_info['filtered']}")
        
        # 성능 통계
        perf_stats = base_inference.get_performance_stats()
        logger.info(f"\n[PERF] Performance:")
        logger.info(f"  Inference Time: {perf_stats['mean_inference_time_ms']:.2f}ms")
        
        collector.close()
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline test failed: {e}", exc_info=True)
        return False


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("[START] Realtime Data Test Suite")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. 데이터 수집 테스트 (30초)
    logger.info("\n[1/2] Testing data collection (30 seconds)...")
    results['data_collection'] = test_data_collection('005930', duration=30)
    
    # 2. 추론 파이프라인 테스트
    logger.info("\n[2/2] Testing inference pipeline...")
    results['inference_pipeline'] = test_inference_pipeline('005930')
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("[SUMMARY] Test Results")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        logger.info(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n[SUCCESS] All tests passed!")
        logger.info("System is ready for live trading!")
    else:
        logger.error("\n[FAILURE] Some tests failed!")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
