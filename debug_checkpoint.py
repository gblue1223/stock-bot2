#!/usr/bin/env python3
"""
체크포인트 파일의 내용을 확인하는 디버그 스크립트
"""
import torch
import sys

if len(sys.argv) != 2:
    print("Usage: python debug_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Keys: {list(checkpoint.keys())}")
    print()
    
    for key, value in checkpoint.items():
        if key == 'state_dict':
            print(f"{key}: <model state dict with {len(value)} parameters>")
        elif isinstance(value, (int, float, str)):
            print(f"{key}: {value}")
        elif isinstance(value, list):
            print(f"{key}: list with {len(value)} items")
        else:
            print(f"{key}: {type(value)}")
            
except Exception as e:
    print(f"Error loading checkpoint: {e}")