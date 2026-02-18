"""
임포트 테스트 스크립트
평가 스크립트가 필요한 모든 모듈을 임포트할 수 있는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
# 스크립트: scripts/pre/autoencoder/diagnosis/test_imports.py
# 루트: 3단계 위
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Testing imports...")

try:
    from ai_trader.embedding.autoencoder_model import (
        AutoEncoderEmbedding,
        MaskedAutoEncoder,
    )
    print("✓ AutoEncoder models imported successfully")
except ImportError as e:
    print(f"✗ Failed to import AutoEncoder models: {e}")
    sys.exit(1)

try:
    from ai_trader.embedding.data import AutoEncoderDataLoader
    print("✓ Data loader imported successfully")
except ImportError as e:
    print(f"✗ Failed to import data loader: {e}")
    sys.exit(1)

try:
    from ai_trader.embedding.evaluation import (
        compute_silhouette_score,
        compute_temporal_coherence,
        evaluate_embedding_quality,
    )
    print("✓ Evaluation functions imported successfully")
except ImportError as e:
    print(f"✗ Failed to import evaluation functions: {e}")
    sys.exit(1)

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    print("✓ External dependencies imported successfully")
except ImportError as e:
    print(f"✗ Failed to import external dependencies: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("The evaluation script should work correctly.")
