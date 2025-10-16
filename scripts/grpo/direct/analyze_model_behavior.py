#!/usr/bin/env python3
"""
DirectFeaturePolicy 심층 행동 분석

모델이 어떤 시장 상황에서 어떤 행동을 선택하는지 분석합니다.
"""

import os
import sys
import logging
import time
from pathlib import Path
import duckdb
import numpy as np
import torch
from collections import defaultdict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_analysis_data(db_path, year_month, seq_len=30, limit=2000):
    """
    분석용 데이터 로드 (더 많은 샘플, 더 다양한 종목)
    """
    logger.info(f"Loading analysis data from {year_month}...")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        query = f"""
        SELECT *
        FROM datasets_raw
        WHERE 날짜 >= '{year_month}01' AND 날짜 < '{year_month}32'
        ORDER BY 종목코드, 날짜, 시간
        """
        
        df = conn.execute(query).fetchdf()
        
        if len(df) == 0:
            logger.error(f"No data found for {year_month}")
            return None
        
        logger.info(f"Loaded {len(df)} rows")
        
        # 특징 컬럼
        feature_cols = [
            '등락률',
            '누적거래대금',
            '거래회전율',
            '체결강도',
            '매도대기금액1', '매도대기금액2', '매도대기금액3', '매도대기금액4', '매도대기금액5',
            '매도대기금액6', '매도대기금액7', '매도대기금액8', '매도대기금액9', '매도대기금액10',
            '매수대기금액1', '매수대기금액2', '매수대기금액3', '매수대기금액4', '매수대기금액5',
            '매수대기금액6', '매수대기금액7', '매수대기금액8', '매수대기금액9', '매수대기금액10'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # 데이터 구조
        data = {
            'sequences': [],
            'dates': [],
            'codes': [],
            'times': [],
            'raw_features': [],  # 원본 특징값 (분석용)
        }
        
        grouped = df.groupby(['종목코드', '날짜'])
        total_groups = len(grouped)
        samples_per_group = max(1, limit // total_groups)
        
        logger.info(f"Found {total_groups} stock-date combinations")
        logger.info(f"Sampling up to {samples_per_group} sequences per combination")
        
        for (code, date), group in grouped:
            if len(data['sequences']) >= limit:
                break
            
            group = group.sort_values('시간')
            features = group[available_cols].fillna(0.0).values.astype(np.float32)
            
            max_sequences = len(features) - seq_len
            if max_sequences <= 0:
                continue
            
            # 균등 샘플링
            if max_sequences <= samples_per_group:
                sample_indices = list(range(seq_len, len(features)))
            else:
                step = max_sequences // samples_per_group
                sample_indices = list(range(seq_len, len(features), step))[:samples_per_group]
            
            for i in sample_indices:
                if len(data['sequences']) >= limit:
                    break
                    
                seq = features[i-seq_len:i]
                data['sequences'].append(seq)
                data['dates'].append(date)
                data['codes'].append(code)
                data['times'].append(group.iloc[i]['시간'] if '시간' in group.columns else 0)
                
                # 현재 시점의 특징값 저장 (분석용)
                current_features = features[i]
                data['raw_features'].append(current_features)
        
        # numpy 배열로 변환
        data['sequences'] = np.array(data['sequences'], dtype=np.float32)
        data['raw_features'] = np.array(data['raw_features'], dtype=np.float32)
        
        logger.info(f"Created {len(data['sequences'])} sequences")
        logger.info(f"Unique stocks: {len(set(data['codes']))}, Unique dates: {len(set(data['dates']))}")
        
        return data
        
    finally:
        conn.close()


def analyze_action_patterns(data, actions, confidences):
    """
    행동 패턴 분석: 어떤 시장 상황에서 어떤 행동을 선택하는가?
    """
    logger.info("=" * 80)
    logger.info("📊 Action Pattern Analysis")
    logger.info("=" * 80)
    
    # 행동별로 특징 분류
    action_features = {
        0: [],  # Hold
        1: [],  # Buy
        2: []   # Sell
    }
    
    action_names = ['Hold', 'Buy', 'Sell']
    
    for i, action in enumerate(actions):
        action_features[action].append(data['raw_features'][i])
    
    # 각 행동에 대한 특징 통계
    feature_names = [
        '등락률', '누적거래대금', '거래회전율', '체결강도'
    ]
    
    logger.info("\n특징별 평균값 (주요 4개 특징):")
    logger.info(f"{'행동':<10} {'등락률':<12} {'누적거래대금':<15} {'거래회전율':<12} {'체결강도':<12}")
    logger.info("-" * 70)
    
    for action in range(3):
        if len(action_features[action]) == 0:
            logger.info(f"{action_names[action]:<10} No samples")
            continue
        
        features_array = np.array(action_features[action])
        means = features_array.mean(axis=0)
        
        logger.info(
            f"{action_names[action]:<10} "
            f"{means[0]:>11.4f} "
            f"{means[1]:>14.2e} "
            f"{means[2]:>11.4f} "
            f"{means[3]:>11.4f}"
        )
    
    # 등락률 분포 분석
    logger.info("\n등락률 분포 분석:")
    for action in range(3):
        if len(action_features[action]) == 0:
            continue
        
        features_array = np.array(action_features[action])
        price_changes = features_array[:, 0]  # 등락률
        
        logger.info(f"\n{action_names[action]}:")
        logger.info(f"  평균: {price_changes.mean():.4f}")
        logger.info(f"  표준편차: {price_changes.std():.4f}")
        logger.info(f"  최소: {price_changes.min():.4f}")
        logger.info(f"  최대: {price_changes.max():.4f}")
        logger.info(f"  중앙값: {np.median(price_changes):.4f}")
        
        # 등락률 구간별 분포
        positive = (price_changes > 0).sum()
        negative = (price_changes < 0).sum()
        neutral = (price_changes == 0).sum()
        
        logger.info(f"  상승(+): {positive} ({positive/len(price_changes)*100:.1f}%)")
        logger.info(f"  하락(-): {negative} ({negative/len(price_changes)*100:.1f}%)")
        logger.info(f"  보합(0): {neutral} ({neutral/len(price_changes)*100:.1f}%)")


def analyze_confidence_patterns(data, actions, confidences):
    """
    신뢰도 패턴 분석: 높은 신뢰도 예측의 특징
    """
    logger.info("\n" + "=" * 80)
    logger.info("🎯 Confidence Pattern Analysis")
    logger.info("=" * 80)
    
    # 신뢰도 구간별 분류
    high_conf = confidences >= 0.7
    medium_conf = (confidences >= 0.4) & (confidences < 0.7)
    low_conf = confidences < 0.4
    
    logger.info(f"\n신뢰도 구간별 분포:")
    logger.info(f"  높음 (≥0.7): {high_conf.sum()} ({high_conf.sum()/len(confidences)*100:.1f}%)")
    logger.info(f"  중간 (0.4-0.7): {medium_conf.sum()} ({medium_conf.sum()/len(confidences)*100:.1f}%)")
    logger.info(f"  낮음 (<0.4): {low_conf.sum()} ({low_conf.sum()/len(confidences)*100:.1f}%)")
    
    # 각 신뢰도 구간의 행동 분포
    action_names = ['Hold', 'Buy', 'Sell']
    
    logger.info("\n신뢰도 구간별 행동 분포:")
    for conf_name, conf_mask in [('높음', high_conf), ('중간', medium_conf), ('낮음', low_conf)]:
        if conf_mask.sum() == 0:
            continue
        
        logger.info(f"\n{conf_name}:")
        conf_actions = actions[conf_mask]
        for action in range(3):
            count = (conf_actions == action).sum()
            logger.info(f"  {action_names[action]}: {count} ({count/len(conf_actions)*100:.1f}%)")
    
    # 높은 신뢰도 샘플의 특징 분석
    if high_conf.sum() > 0:
        logger.info(f"\n높은 신뢰도 샘플의 특징 (n={high_conf.sum()}):")
        high_conf_features = data['raw_features'][high_conf]
        
        logger.info(f"  평균 등락률: {high_conf_features[:, 0].mean():.4f}")
        logger.info(f"  등락률 표준편차: {high_conf_features[:, 0].std():.4f}")
        
        # 극단값 비율
        extreme_change = np.abs(high_conf_features[:, 0]) > 2.0
        logger.info(f"  극단적 등락률 (|x|>2.0): {extreme_change.sum()} ({extreme_change.sum()/len(high_conf_features)*100:.1f}%)")


def analyze_temporal_patterns(data, actions, confidences):
    """
    시간적 패턴 분석: 날짜별, 종목별 행동 분포
    """
    logger.info("\n" + "=" * 80)
    logger.info("📅 Temporal Pattern Analysis")
    logger.info("=" * 80)
    
    # 날짜별 행동 분포
    date_actions = defaultdict(lambda: {'hold': 0, 'buy': 0, 'sell': 0, 'total': 0})
    
    for i, action in enumerate(actions):
        date = data['dates'][i]
        date_actions[date]['total'] += 1
        if action == 0:
            date_actions[date]['hold'] += 1
        elif action == 1:
            date_actions[date]['buy'] += 1
        else:
            date_actions[date]['sell'] += 1
    
    logger.info("\n날짜별 행동 분포 (상위 10개 날짜):")
    logger.info(f"{'날짜':<12} {'Hold':<8} {'Buy':<8} {'Sell':<8} {'Total':<8}")
    logger.info("-" * 50)
    
    sorted_dates = sorted(date_actions.keys())[:10]
    for date in sorted_dates:
        d = date_actions[date]
        logger.info(
            f"{date:<12} "
            f"{d['hold']:<8} "
            f"{d['buy']:<8} "
            f"{d['sell']:<8} "
            f"{d['total']:<8}"
        )
    
    # 종목별 행동 다양성
    stock_actions = defaultdict(set)
    stock_counts = defaultdict(int)
    
    for i, action in enumerate(actions):
        code = data['codes'][i]
        stock_actions[code].add(action)
        stock_counts[code] += 1
    
    # 3가지 행동을 모두 선택한 종목
    diverse_stocks = [code for code, acts in stock_actions.items() if len(acts) == 3]
    
    logger.info(f"\n종목 분석:")
    logger.info(f"  총 종목 수: {len(stock_actions)}")
    logger.info(f"  3가지 행동 모두 선택한 종목: {len(diverse_stocks)} ({len(diverse_stocks)/len(stock_actions)*100:.1f}%)")
    logger.info(f"  2가지 행동 선택한 종목: {sum(1 for acts in stock_actions.values() if len(acts) == 2)}")
    logger.info(f"  1가지 행동만 선택한 종목: {sum(1 for acts in stock_actions.values() if len(acts) == 1)}")
    
    if diverse_stocks:
        logger.info(f"\n다양한 행동을 보인 종목 예시 (최대 5개):")
        for code in diverse_stocks[:5]:
            logger.info(f"  {code}: {stock_counts[code]} 샘플")


def simulate_trading(data, actions, confidences):
    """
    간단한 거래 시뮬레이션 및 Buy→Sell 간격 분석
    """
    logger.info("\n" + "=" * 80)
    logger.info("💰 Trading Simulation & Buy-Sell Gap Analysis")
    logger.info("=" * 80)
    
    # 전체 데이터에서 Buy와 Sell의 거리 분석
    buy_indices = np.where(actions == 1)[0]
    sell_indices = np.where(actions == 2)[0]
    
    logger.info(f"\n전체 통계:")
    logger.info(f"  총 샘플 수: {len(actions)}")
    logger.info(f"  Buy 발생: {len(buy_indices)} ({len(buy_indices)/len(actions)*100:.1f}%)")
    logger.info(f"  Sell 발생: {len(sell_indices)} ({len(sell_indices)/len(actions)*100:.1f}%)")
    logger.info(f"  Hold 발생: {(actions == 0).sum()} ({(actions == 0).sum()/len(actions)*100:.1f}%)")
    
    # 종목별로 Buy-Sell 패턴 분석
    logger.info(f"\n종목별 Buy-Sell 패턴 분석:")
    
    stock_patterns = defaultdict(lambda: {'buys': [], 'sells': [], 'samples': []})
    
    for i, (code, date) in enumerate(zip(data['codes'], data['dates'])):
        action = actions[i]
        stock_patterns[code]['samples'].append(i)
        if action == 1:
            stock_patterns[code]['buys'].append(i)
        elif action == 2:
            stock_patterns[code]['sells'].append(i)
    
    # Buy와 Sell이 모두 있는 종목
    tradable_stocks = []
    for code, pattern in stock_patterns.items():
        if len(pattern['buys']) > 0 and len(pattern['sells']) > 0:
            tradable_stocks.append(code)
    
    logger.info(f"  Buy와 Sell이 모두 있는 종목: {len(tradable_stocks)} / {len(stock_patterns)}")
    
    if tradable_stocks:
        logger.info(f"\n거래 가능 종목 예시 (최대 10개):")
        for code in tradable_stocks[:10]:
            pattern = stock_patterns[code]
            logger.info(f"  {code}: Buy {len(pattern['buys'])}회, Sell {len(pattern['sells'])}회")
        
        # 간단한 거래 시뮬레이션 (종목별로 시간순 정렬 후 Buy→Sell)
        results = []
        
        for code in tradable_stocks[:50]:  # 샘플로 50개만
            pattern = stock_patterns[code]
            samples = sorted(pattern['samples'])
            
            position = False
            entry_price_change = 0.0
            entry_idx = 0
            trades = []
            
            for idx in samples:
                action = actions[idx]
                price_change = data['raw_features'][idx][0]
                
                if action == 1 and not position:  # Buy
                    position = True
                    entry_price_change = price_change
                    entry_idx = idx
                elif action == 2 and position:  # Sell
                    position = False
                    profit = price_change - entry_price_change
                    holding_period = idx - entry_idx
                    trades.append({
                        'profit': profit,
                        'holding_period': holding_period,
                        'entry_price': entry_price_change,
                        'exit_price': price_change
                    })
            
            if trades:
                results.append({
                    'code': code,
                    'num_trades': len(trades),
                    'total_profit': sum(t['profit'] for t in trades),
                    'avg_profit': np.mean([t['profit'] for t in trades]),
                    'avg_holding': np.mean([t['holding_period'] for t in trades]),
                    'win_rate': sum(1 for t in trades if t['profit'] > 0) / len(trades),
                    'trades': trades
                })
        
        if results:
            logger.info(f"\n시뮬레이션 결과 (n={len(results)} 종목):")
            
            total_trades = sum(r['num_trades'] for r in results)
            total_profit = sum(r['total_profit'] for r in results)
            avg_profit = np.mean([r['avg_profit'] for r in results])
            avg_holding = np.mean([r['avg_holding'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            
            logger.info(f"  총 거래 수: {total_trades}")
            logger.info(f"  총 수익률: {total_profit:.4f}%")
            logger.info(f"  평균 거래당 수익률: {avg_profit:.4f}%")
            logger.info(f"  평균 보유 기간: {avg_holding:.1f} 샘플")
            logger.info(f"  평균 승률: {avg_win_rate:.2%}")
            
            # 상위 5개 수익 종목
            top_performers = sorted(results, key=lambda x: x['total_profit'], reverse=True)[:5]
            logger.info("\n상위 수익 종목:")
            for r in top_performers:
                logger.info(f"  {r['code']}: {r['total_profit']:.4f}% ({r['num_trades']} 거래, 평균 보유 {r['avg_holding']:.1f})")
            
            # 하위 5개 손실 종목
            bottom_performers = sorted(results, key=lambda x: x['total_profit'])[:5]
            logger.info("\n하위 손실 종목:")
            for r in bottom_performers:
                logger.info(f"  {r['code']}: {r['total_profit']:.4f}% ({r['num_trades']} 거래)")
        else:
            logger.info("\n시뮬레이션 결과가 없습니다.")
    else:
        logger.info("\n동일 종목 내에서 Buy→Sell 사이클을 찾을 수 없습니다.")
        logger.info("모델이 종목별로 일관된 전략을 사용하지 않는 것으로 보입니다.")


def main():
    """메인 함수"""
    
    model_path = r"d:\Workspace\Project\stock-bot\stock-bot2\models\grpo_direct_features@20251016\direct_features_model.pt"
    db_path = r"C:\Users\user\Workspace\datasets@20251016\datasets_raw_all.duckdb"
    year_month = "202509"
    limit = 2000
    
    logger.info("DirectFeaturePolicy 심층 분석")
    logger.info(f"Model: {model_path}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Period: {year_month}")
    logger.info(f"Samples: {limit}")
    logger.info("=" * 80)
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 추론 엔진 초기화
    logger.info("Initializing inference engine...")
    inference = GRPOInference(
        policy_path=model_path,
        embedding_model_path=None,
        device=device,
        use_torchscript=False,
        cache_size=100
    )
    logger.info("✅ Inference engine initialized")
    
    # 데이터 로드
    data = load_analysis_data(db_path, year_month, seq_len=30, limit=limit)
    
    if data is None:
        logger.error("Failed to load data")
        return False
    
    # 배치 추론
    logger.info("\nRunning inference on all samples...")
    actions, confidences = inference.predict_batch(data['sequences'], deterministic=True)
    logger.info("✅ Inference completed")
    
    # 분석 수행
    analyze_action_patterns(data, actions, confidences)
    analyze_confidence_patterns(data, actions, confidences)
    analyze_temporal_patterns(data, actions, confidences)
    simulate_trading(data, actions, confidences)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 심층 분석 완료!")
    logger.info("=" * 80)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
