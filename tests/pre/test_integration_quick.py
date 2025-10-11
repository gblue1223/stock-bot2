#!/usr/bin/env python3
"""
빠른 통합 테스트 - 실제 스크립트 실행 검증
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import json
import h5py
import numpy as np

def create_minimal_test_data():
    """최소한의 테스트 데이터 생성"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test data structure
    month_dir = Path(temp_dir) / '2024_09'
    month_dir.mkdir(parents=True, exist_ok=True)
    
    # Create batch_info.json
    batch_info = {
        'num_batches': 2,
        'sequences_per_batch': 100,
        'seq_len': 60,
        'num_features': 28
    }
    with open(month_dir / 'batch_info.json', 'w') as f:
        json.dump(batch_info, f)
    
    # Create test batch files
    for i in range(2):
        batch_file = month_dir / f'batch_{i:06d}.h5'
        with h5py.File(batch_file, 'w') as f:
            # Create random sequences data
            sequences = np.random.randn(100, 60, 28).astype(np.float32)
            f.create_dataset('sequences', data=sequences)
    
    return temp_dir

def test_script_execution():
    """스크립트 실행 테스트"""
    print("🧪 Creating test data...")
    data_dir = create_minimal_test_data()
    output_dir = tempfile.mkdtemp()
    
    try:
        print("🚀 Running training script...")
        
        # Run the script with minimal parameters
        cmd = [
            sys.executable, '-m', 'scripts.pre.train_autoencoder_preprocessed',
            '--data-dir', data_dir,
            '--output-dir', output_dir,
            '--train-months', '2024_09',
            '--val-months', '2024_09',
            '--max-epochs', '1',
            '--batch-size', '1',
            '--max-sequences', '50',
            '--max-batches-per-month', '1',
            '--embedding-dim', '32',
            '--hidden-dim', '64'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        print(f"📊 Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Script executed successfully!")
            
            # Check output files
            output_path = Path(output_dir)
            expected_files = [
                'training_config.json',
                'training_history.json',
                'training.log'
            ]
            
            for file_name in expected_files:
                if (output_path / file_name).exists():
                    print(f"✅ {file_name} created")
                else:
                    print(f"❌ {file_name} missing")
            
            # Check training history
            history_file = output_path / 'training_history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    print(f"📈 Training completed with {len(history)} epochs")
                    if history:
                        final_loss = history[-1].get('train_loss', 'N/A')
                        print(f"📉 Final training loss: {final_loss}")
            
        else:
            print("❌ Script execution failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Script execution timed out")
        return False
    except Exception as e:
        print(f"💥 Error during execution: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(data_dir)
        shutil.rmtree(output_dir)

def test_import_verification():
    """Import 검증 테스트"""
    print("🔍 Verifying imports...")
    
    try:
        # Add project root to path
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        # Test individual imports
        from scripts.pre.train_autoencoder_preprocessed import (
            create_data_loaders,
            PreprocessedAutoEncoderTrainer,
            main
        )
        print("✅ Script imports successful")
        
        from ai_trader.embedding.autoencoder_model import create_autoencoder_model
        print("✅ Model creation import successful")
        
        from ai_trader.embedding.data import PreprocessedDataset, BatchCollator
        print("✅ Data classes import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Integration Test for train_autoencoder_preprocessed.py")
    print("=" * 60)
    
    # Test 1: Import verification
    import_success = test_import_verification()
    
    # Test 2: Script execution (only if imports work)
    if import_success:
        execution_success = test_script_execution()
    else:
        execution_success = False
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Import Test: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"  Execution Test: {'✅ PASS' if execution_success else '❌ FAIL'}")
    
    if import_success and execution_success:
        print("\n🎉 All integration tests passed!")
        exit(0)
    else:
        print("\n💥 Some tests failed!")
        exit(1)