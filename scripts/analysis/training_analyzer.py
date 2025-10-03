#!/usr/bin/env python3
"""
Comprehensive Training Analysis Tool
통합된 훈련 분석 도구 - 모델 파일, 로그, 그래프 패턴을 종합적으로 분석
"""

import os
import json
import torch
from pathlib import Path
import sys

class TrainingAnalyzer:
    """훈련 결과를 종합적으로 분석하는 클래스"""
    
    def __init__(self, model_dir: str):
        self.model_path = Path(model_dir)
        self.config = None
        self.model_data = None
        
    def load_model_info(self):
        """모델 정보 로드"""
        config_file = self.model_path / "config.json"
        model_file = self.model_path / "model.pt"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        if model_file.exists():
            try:
                self.model_data = torch.load(model_file, map_location='cpu')
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                
    def analyze_files(self):
        """파일 존재 여부 및 기본 정보 분석"""
        print("=" * 60)
        print("📁 파일 분석")
        print("=" * 60)
        
        config_file = self.model_path / "config.json"
        model_file = self.model_path / "model.pt"
        checkpoints_dir = self.model_path / "checkpoints"
        tensorboard_dir = self.model_path / "tensorboard_logs"
        
        print(f"모델 디렉토리: {self.model_path}")
        print(f"✅ Config 파일: {config_file.exists()}")
        print(f"✅ 모델 파일: {model_file.exists()}")
        print(f"✅ 체크포인트: {checkpoints_dir.exists()}")
        print(f"✅ TensorBoard 로그: {tensorboard_dir.exists()}")
        
        if model_file.exists():
            size_mb = model_file.stat().st_size / 1024 / 1024
            print(f"📊 모델 크기: {size_mb:.2f} MB")
            
        return all([config_file.exists(), model_file.exists()])
    
    def analyze_config(self):
        """훈련 설정 분석"""
        if not self.config:
            return
            
        print("\n" + "=" * 60)
        print("⚙️  훈련 설정")
        print("=" * 60)
        
        print(f"데이터셋: {self.config.get('code', 'N/A')}")
        print(f"타겟 날짜: {self.config.get('date', 'N/A')}")
        print(f"시퀀스 길이: {self.config.get('seq_len', 'N/A')}")
        print(f"예측 구간: {self.config.get('horizon', 'N/A')}")
        print(f"타겟 컬럼: {self.config.get('target_col', 'N/A')}")
        print(f"보조 작업: {self.config.get('aux_task', 'N/A')}")
        print(f"피처 수: {len(self.config.get('feature_names', []))}")
        
    def analyze_performance(self):
        """성능 지표 분석"""
        if not self.config:
            return
            
        print("\n" + "=" * 60)
        print("📈 성능 지표")
        print("=" * 60)
        
        best_val_loss = self.config.get('best_val_mse')
        if best_val_loss is not None:
            print(f"최고 검증 손실: {best_val_loss:.6f}")
            
            if best_val_loss < 0.3:
                print("🎉 우수한 성능 (< 0.3)")
            elif best_val_loss < 0.5:
                print("✅ 양호한 성능 (< 0.5)")
            elif best_val_loss < 1.0:
                print("⚠️  보통 성능 (0.5-1.0)")
            else:
                print("❌ 개선 필요 (> 1.0)")
        
        if self.model_data and isinstance(self.model_data, dict):
            if 'epoch' in self.model_data:
                print(f"완료 에포크: {self.model_data['epoch']}")
            if 'loss' in self.model_data:
                print(f"최종 손실: {self.model_data['loss']:.6f}")
    
    def analyze_checkpoints(self):
        """체크포인트 분석"""
        checkpoints_dir = self.model_path / "checkpoints"
        if not checkpoints_dir.exists():
            return
            
        print("\n" + "=" * 60)
        print("💾 체크포인트")
        print("=" * 60)
        
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        print(f"저장된 체크포인트: {len(checkpoints)}개")
        
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    
    def analyze_tensorboard(self):
        """TensorBoard 로그 분석"""
        tensorboard_dir = self.model_path / "tensorboard_logs"
        if not tensorboard_dir.exists():
            return
            
        print("\n" + "=" * 60)
        print("📊 TensorBoard")
        print("=" * 60)
        
        tb_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
        print(f"이벤트 파일: {len(tb_files)}개")
        
        if tb_files:
            print("✅ 훈련 메트릭이 기록됨")
            print(f"확인 명령: tensorboard --logdir {tensorboard_dir}")
        
    def analyze_loss_patterns(self):
        """손실 패턴 분석 (일반적인 문제점들)"""
        print("\n" + "=" * 60)
        print("🔍 손실 패턴 분석")
        print("=" * 60)
        
        print("일반적인 문제 패턴들:")
        print("1. 높은 변동성 → 학습률 감소, 배치 크기 증가")
        print("2. 수렴 실패 → 더 많은 에포크, 학습률 스케줄러")
        print("3. 과적합 → 드롭아웃 증가, 정규화 강화")
        print("4. 클래스 불균형 → 가중 샘플링, Focal Loss")
        
        print("\n권장 개선사항:")
        print("- 학습률 스케줄러 (ReduceLROnPlateau)")
        print("- 그래디언트 클리핑 (max_norm=1.0)")
        print("- 조기 종료 (Early Stopping)")
        print("- 배치 정규화 또는 레이어 정규화")
    
    def generate_recommendations(self):
        """개선 권장사항 생성"""
        print("\n" + "=" * 60)
        print("💡 권장사항")
        print("=" * 60)
        
        if not self.config:
            print("❌ 설정 파일이 없어 권장사항을 생성할 수 없습니다.")
            return
            
        best_val_loss = self.config.get('best_val_mse', float('inf'))
        
        print("다음 훈련을 위한 권장 설정:")
        print("```bash")
        print("python -m ai_trader.ml.train_supervised \\")
        print(f"  --db \"{self.config.get('db_path', '')}\" \\")
        print(f"  --table {self.config.get('table', 'datasets')} \\")
        print("  --out models/supervised_improved \\")
        print(f"  --seq-len {self.config.get('seq_len', 60)} \\")
        print(f"  --horizon {self.config.get('horizon', 20)} \\")
        print("  --chunk-size 500 \\")
        print("  --device cuda \\")
        
        # 성능에 따른 동적 권장사항
        if best_val_loss > 0.5:
            print("  --batch-size 128 \\")
            print("  --epochs 10 \\")
            print("  --lr 1e-5 \\")
        elif best_val_loss > 0.3:
            print("  --batch-size 64 \\")
            print("  --epochs 8 \\")
            print("  --lr 3e-5 \\")
        else:
            print("  --batch-size 64 \\")
            print("  --epochs 5 \\")
            print("  --lr 5e-5 \\")
            
        print("  --weight-decay 1e-4 \\")
        print(f"  --target-col {self.config.get('target_col', '현재가')} \\")
        print(f"  --code {self.config.get('code', '000100')} \\")
        print(f"  --aux-task {self.config.get('aux_task', 'direction3')} \\")
        print("  --loss focal \\")
        print("  --use-weighted-sampler")
        print("```")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("🔍 훈련 결과 종합 분석")
        print("=" * 60)
        
        self.load_model_info()
        
        if not self.analyze_files():
            print("❌ 필수 파일이 없습니다. 훈련이 실패했을 수 있습니다.")
            return False
            
        self.analyze_config()
        self.analyze_performance()
        self.analyze_checkpoints()
        self.analyze_tensorboard()
        self.analyze_loss_patterns()
        self.generate_recommendations()
        
        print("\n" + "=" * 60)
        print("✅ 분석 완료")
        return True

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python training_analyzer.py <model_directory>")
        print("예시: python training_analyzer.py models/supervised_sample_test")
        return
        
    model_dir = sys.argv[1]
    analyzer = TrainingAnalyzer(model_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()