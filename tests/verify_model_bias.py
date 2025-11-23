
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.inference.infer_grpo import GRPOInference
from ai_trader.grpo.inference.enhanced_inference import EnhancedGRPOInference, Action

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_synthetic_data(num_samples, seq_len, num_features, pattern='random'):
    data = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    
    for i in range(num_samples):
        if pattern == 'random':
            # Normal distribution
            seq = np.random.randn(seq_len, num_features)
        elif pattern == 'bullish':
            # Price (feature 0 usually?) increasing
            # Let's assume feature 4 is return/change
            seq = np.random.randn(seq_len, num_features)
            # Add positive trend to some features
            trend = np.linspace(0, 2, seq_len).reshape(-1, 1)
            seq[:, 4] += trend[:, 0] # Feature 4 is often return/close
        elif pattern == 'bearish':
            seq = np.random.randn(seq_len, num_features)
            trend = np.linspace(0, -2, seq_len).reshape(-1, 1)
            seq[:, 4] += trend[:, 0]
            
        data[i] = seq
        
    return data

def test_bias():
    config_path = os.path.join(project_root, 'config', 'trading_config.json')
    config = load_config(config_path)
    
    model_path = os.path.join(project_root, config['model_path'])
    embedding_path = os.path.join(project_root, config['embedding_model_path'])
    
    print(f"Loading model from {model_path}")
    
    # Initialize inference
    base_inference = GRPOInference(
        policy_path=model_path,
        embedding_model_path=embedding_path,
        device='cpu', # Use CPU for test
        use_torchscript=False,
        normalization_stats=None
    )
    
    # Use the config thresholds
    inference = EnhancedGRPOInference(
        base_inference=base_inference,
        min_buy_confidence=config.get('min_buy_confidence', 0.62),
        min_sell_confidence=config.get('min_sell_confidence', 0.0), # This might have been changed by user
        stop_loss_rate=-2.0,
        take_profit_rate=5.0,
        enable_auto_exit=False 
    )
    
    print(f"Thresholds: Buy={inference.min_buy_confidence}, Sell={inference.min_sell_confidence}")
    
    patterns = ['random', 'bullish', 'bearish']
    results = {}
    
    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        data = generate_synthetic_data(1000, 60, 28, pattern)
        
        actions = []
        confidences = []
        raw_actions = []
        
        buy_cnt = 0
        sell_cnt = 0
        hold_cnt = 0
        
        high_conf_buy = 0
        high_conf_sell = 0
        
        for i in range(len(data)):
            seq = data[i]
            # Predict
            action, conf, info = inference.predict(seq, deterministic=True)
            
            actions.append(action)
            confidences.append(conf)
            raw_actions.append(info['raw_action'])
            
            if info['raw_action'] == 1: # BUY
                buy_cnt += 1
                if conf > 0.62:
                    high_conf_buy += 1
            elif info['raw_action'] == 2: # SELL
                sell_cnt += 1
                if conf > 0.62:
                    high_conf_sell += 1
            else:
                hold_cnt += 1
                
        avg_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        
        print(f"  Total: 1000")
        print(f"  Raw Actions: BUY={buy_cnt}, SELL={sell_cnt}, HOLD={hold_cnt}")
        print(f"  High Conf (>0.62): BUY={high_conf_buy}, SELL={high_conf_sell}")
        print(f"  Avg Confidence: {avg_conf:.4f}")
        print(f"  Max Confidence: {max_conf:.4f}")
        
        # Check max confidence for BUY specifically
        buy_confs = [c for a, c in zip(raw_actions, confidences) if a == 1]
        max_buy_conf = max(buy_confs) if buy_confs else 0.0
        print(f"  Max BUY Confidence: {max_buy_conf:.4f}")

if __name__ == "__main__":
    test_bias()
