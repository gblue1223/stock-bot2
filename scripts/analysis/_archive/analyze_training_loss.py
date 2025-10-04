"""
TensorBoard 로그를 분석하여 학습 문제를 진단하는 스크립트
"""
import os
import sys
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_tensorboard_logs(log_dir: str):
    """TensorBoard 로그 분석"""
    
    # Find event files
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return
    
    event_file = os.path.join(log_dir, event_files[0])
    print(f"Analyzing: {event_file}\n")
    
    # Load events
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Available tags
    print("=" * 80)
    print("Available Metrics:")
    print("=" * 80)
    for tag in sorted(ea.Tags()['scalars']):
        print(f"  - {tag}")
    print()
    
    # Analyze training loss
    if 'Loss/Train_Chunk' in ea.Tags()['scalars']:
        train_loss = ea.Scalars('Loss/Train_Chunk')
        steps = [s.step for s in train_loss]
        values = [s.value for s in train_loss]
        
        print("=" * 80)
        print("Training Loss Analysis (Chunk-level)")
        print("=" * 80)
        print(f"Total chunks: {len(values)}")
        print(f"Loss range: [{min(values):.4f}, {max(values):.4f}]")
        print(f"Loss mean: {np.mean(values):.4f}")
        print(f"Loss std: {np.std(values):.4f}")
        
        # Check for convergence
        if len(values) >= 100:
            first_100 = np.mean(values[:100])
            last_100 = np.mean(values[-100:])
            improvement = (first_100 - last_100) / first_100 * 100
            print(f"\nFirst 100 chunks avg: {first_100:.4f}")
            print(f"Last 100 chunks avg: {last_100:.4f}")
            print(f"Improvement: {improvement:.2f}%")
            
            if improvement < 5:
                print("⚠️  WARNING: Loss not improving significantly!")
        
        # Check oscillation
        if len(values) >= 50:
            recent_std = np.std(values[-50:])
            recent_mean = np.mean(values[-50:])
            cv = recent_std / recent_mean if recent_mean > 0 else 0
            print(f"\nRecent 50 chunks:")
            print(f"  Mean: {recent_mean:.4f}")
            print(f"  Std: {recent_std:.4f}")
            print(f"  Coefficient of Variation: {cv:.4f}")
            
            if cv > 0.15:
                print("⚠️  WARNING: High oscillation detected!")
        print()
    
    # Analyze validation loss
    if 'Loss/Validation_Chunk' in ea.Tags()['scalars']:
        val_loss = ea.Scalars('Loss/Validation_Chunk')
        values = [s.value for s in val_loss]
        
        print("=" * 80)
        print("Validation Loss Analysis (Chunk-level)")
        print("=" * 80)
        print(f"Total chunks: {len(values)}")
        print(f"Loss range: [{min(values):.4f}, {max(values):.4f}]")
        print(f"Loss mean: {np.mean(values):.4f}")
        print(f"Loss std: {np.std(values):.4f}")
        
        if len(values) >= 50:
            recent_std = np.std(values[-50:])
            recent_mean = np.mean(values[-50:])
            cv = recent_std / recent_mean if recent_mean > 0 else 0
            print(f"\nRecent 50 chunks:")
            print(f"  Mean: {recent_mean:.4f}")
            print(f"  Std: {recent_std:.4f}")
            print(f"  Coefficient of Variation: {cv:.4f}")
        print()
    
    # Analyze accuracy
    if 'Metrics/Val_Accuracy_Chunk' in ea.Tags()['scalars']:
        acc = ea.Scalars('Metrics/Val_Accuracy_Chunk')
        values = [s.value for s in acc]
        
        print("=" * 80)
        print("Validation Accuracy Analysis")
        print("=" * 80)
        print(f"Total chunks: {len(values)}")
        print(f"Accuracy range: [{min(values):.4f}, {max(values):.4f}]")
        print(f"Accuracy mean: {np.mean(values):.4f}")
        print(f"Accuracy std: {np.std(values):.4f}")
        
        if len(values) >= 50:
            recent_mean = np.mean(values[-50:])
            print(f"\nRecent 50 chunks avg accuracy: {recent_mean:.4f}")
            
            # Check for majority baseline
            if recent_mean < 0.40:
                print("⚠️  WARNING: Accuracy below 40% - possible class collapse!")
        print()
    
    # Analyze class distribution
    print("=" * 80)
    print("Class Distribution Analysis (Recent)")
    print("=" * 80)
    
    for cls in [0, 1, 2]:
        pred_tag = f'Metrics/Val_Pred_Class{cls}_Count_Chunk'
        true_tag = f'Metrics/Val_True_Class{cls}_Count_Chunk'
        
        if pred_tag in ea.Tags()['scalars'] and true_tag in ea.Tags()['scalars']:
            pred_counts = ea.Scalars(pred_tag)
            true_counts = ea.Scalars(true_tag)
            
            if pred_counts and true_counts:
                # Get last 10 chunks
                recent_pred = [s.value for s in pred_counts[-10:]]
                recent_true = [s.value for s in true_counts[-10:]]
                
                avg_pred = np.mean(recent_pred)
                avg_true = np.mean(recent_true)
                
                print(f"Class {cls}:")
                print(f"  True (avg last 10): {avg_true:.1f}")
                print(f"  Pred (avg last 10): {avg_pred:.1f}")
    
    print()
    
    # Diagnose issues
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    issues = []
    
    # Check if loss is stuck
    if 'Loss/Validation_Chunk' in ea.Tags()['scalars']:
        val_loss = ea.Scalars('Loss/Validation_Chunk')
        values = [s.value for s in val_loss]
        
        if len(values) >= 100:
            first_50 = np.mean(values[:50])
            last_50 = np.mean(values[-50:])
            improvement = (first_50 - last_50) / first_50 * 100
            
            if improvement < 3:
                issues.append("Loss가 거의 개선되지 않음 (< 3%)")
            
            # Check if stuck at baseline
            if 1.05 < last_50 < 1.15:
                issues.append("Loss가 3-class baseline (ln(3)≈1.099) 근처에 정체")
    
    # Check class collapse
    if 'Metrics/Val_Pred_Class0_Count_Chunk' in ea.Tags()['scalars']:
        pred_counts = []
        for cls in [0, 1, 2]:
            tag = f'Metrics/Val_Pred_Class{cls}_Count_Chunk'
            if tag in ea.Tags()['scalars']:
                counts = ea.Scalars(tag)
                if counts:
                    pred_counts.append(np.mean([s.value for s in counts[-10:]]))
        
        if pred_counts:
            total = sum(pred_counts)
            if total > 0:
                props = [c / total for c in pred_counts]
                max_prop = max(props)
                
                if max_prop > 0.90:
                    issues.append(f"예측 붕괴: 단일 클래스가 {max_prop*100:.1f}% 차지")
    
    if issues:
        print("발견된 문제:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 주요 문제가 발견되지 않았습니다.")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "models/supervised_sample_test/tensorboard_logs"
    
    if not os.path.exists(log_dir):
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)
    
    analyze_tensorboard_logs(log_dir)
