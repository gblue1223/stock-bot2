#!/usr/bin/env python3
"""
xLSTM 정책 사전 훈련 스크립트 (Knowledge Distillation / Behavior Cloning)

이미 학습된 GRU E2E 에이전트(Teacher)의 행동 확률 분포와 상태 가치를
새로운 xLSTM E2E 에이전트(Student)에게 모방 학습(지식 증류)시켜 빠르게 수렴시킵니다.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BehaviorCloning")


class OfflineEpisodeDataset(Dataset):
    """
    사전 추출된 npz 에피소드 파일들로부터 시퀀스를 생성하는 Dataset
    """
    def __init__(self, extracted_dir: str, seq_len: int, step_size: int = 50):
        self.extracted_dir = Path(extracted_dir)
        self.seq_len = seq_len
        self.step_size = step_size  # 데이터를 촘촘하게 혹은 듬성듬성 샘플링
        
        manifest_path = self.extracted_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {extracted_dir}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        self.episodes = manifest_data.get("episodes", [])
        self.samples = []
        
        # 각 에피소드별로 가능한 시퀀스 인덱스 빌드
        for ep_idx, ep in enumerate(self.episodes):
            length = ep["length"]
            # seq_len 이상의 데이터가 확보되는 지점들을 인덱싱
            for start_idx in range(0, length - seq_len, step_size):
                self.samples.append((ep_idx, start_idx))
                
        # 에피소드 데이터를 메모리에 캐싱하여 로딩 속도 극대화
        self.cached_features = {}
        self.cached_metadata = {}
        logger.info(f"Indexing completed: {len(self.samples)} sequence samples from {len(self.episodes)} episodes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, start_idx = self.samples[idx]
        ep = self.episodes[ep_idx]
        file_path = self.extracted_dir / ep["file_path"]
        
        if ep_idx not in self.cached_features:
            data = np.load(file_path, allow_pickle=True)
            self.cached_features[ep_idx] = data['features'].astype(np.float32)
            self.cached_metadata[ep_idx] = data['metadata']
            
        features = self.cached_features[ep_idx]
        metadata = self.cached_metadata[ep_idx]
        
        # seq_len 만큼 추출
        seq_features = features[start_idx : start_idx + self.seq_len]
        
        # 환경(scalping_env)과 동일하게 포지션 메타데이터(3차원) 추가
        # 사전 학습 단계이므로 포지션이 없는 상태(0, 0, 0) 및 보유 상태(1, 0, 0)를 반반씩 랜덤 시뮬레이션
        # 이를 통해 모델이 포지션 상태 변화에 대응할 수 있도록 합니다.
        has_position = np.random.choice([0.0, 1.0])
        position_meta = np.array([has_position, 0.0, 0.0], dtype=np.float32)
        position_meta_expanded = np.tile(position_meta, (self.seq_len, 1))
        
        # 최종 입력 형태: (seq_len, obs_dim) (예: 28 피처 + 3 포지션 피처 = 31)
        state = np.concatenate([seq_features, position_meta_expanded], axis=-1)
        return torch.tensor(state, dtype=torch.float32)


def pretrain(
    teacher_path: str,
    extracted_dir: str,
    output_path: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    temperature: float = 2.0,
    device: str = 'cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Dataset 및 DataLoader 준비
    logger.info("Preparing dataset...")
    dataset = OfflineEpisodeDataset(extracted_dir, seq_len=3000, step_size=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Teacher (GRU) 모델 로드 및 차원 추론
    logger.info(f"Loading Teacher (GRU) model from: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device)
    state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # 구조 감지
    obs_dim = state_dict['conv1.weight'].shape[1] if 'conv1.weight' in state_dict else 31
    cnn_channels = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 64
    rnn_hidden_dim = state_dict['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in state_dict else 128
    fc_hidden_dim = state_dict['fc1.weight'].shape[0] if 'fc1.weight' in state_dict else 256
    action_dim = state_dict['policy_head.weight'].shape[0] if 'policy_head.weight' in state_dict else 3
    
    teacher = GRPOPolicyE2E(
        obs_dim=obs_dim,
        cnn_channels=cnn_channels,
        rnn_hidden_dim=rnn_hidden_dim,
        fc_hidden_dim=fc_hidden_dim,
        action_dim=action_dim
    )
    teacher.load_state_dict(state_dict, strict=False)
    teacher.to(device)
    teacher.eval()  # Teacher는 고정
    
    # 3. Student (xLSTM) 모델 생성 및 CNN 가중치 전이
    logger.info("Initializing Student (xLSTM) model and transferring CNN layers...")
    student = GRPOPolicyE2EXLSTM(
        obs_dim=obs_dim,
        cnn_channels=cnn_channels,
        rnn_hidden_dim=rnn_hidden_dim,
        fc_hidden_dim=fc_hidden_dim,
        action_dim=action_dim
    )
    
    # CNN 가중치 카피 (conv1, bn1, conv2, bn2)
    cnn_state = {k: v for k, v in state_dict.items() if k.startswith(('conv', 'bn'))}
    student.load_state_dict(cnn_state, strict=False)
    student.to(device)
    student.train()
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    logger.info("Start Behavior Cloning (Knowledge Distillation)...")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch_states in progress_bar:
            batch_states = batch_states.to(device)
            
            # Teacher 예측 (확률 분포 및 가치)
            with torch.no_grad():
                teacher_logits, teacher_values = teacher(batch_states)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                
            # Student 예측
            student_logits, student_values = student(batch_states)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            
            # KL Divergence Loss (Policy 모방)
            # F.kl_div는 target이 log-space가 아닐 때 log_target=False(기본값)로 줍니다.
            loss_policy = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            
            # MSE Loss (Value Head 모방)
            loss_value = F.mse_loss(student_values, teacher_values)
            
            # Total Loss
            loss = loss_policy + 0.5 * loss_value
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            policy_loss_sum += loss_policy.item()
            value_loss_sum += loss_value.item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Pi": f"{loss_policy.item():.4f}",
                "V": f"{loss_value.item():.4f}"
            })
            
        avg_loss = epoch_loss / len(dataloader)
        avg_pi = policy_loss_sum / len(dataloader)
        avg_v = value_loss_sum / len(dataloader)
        logger.info(f"Epoch {epoch} Completed. Avg Loss: {avg_loss:.4f} (Policy: {avg_pi:.4f}, Value: {avg_v:.4f})")
        
    # 4. 사전 훈련된 모델 체크포인트 저장
    output_path_pt = Path(output_path)
    output_path_pt.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_to_save = {
        'policy_state_dict': student.state_dict(),
        'config': {
            'policy_type': student.__class__.__name__,
            'action_dim': action_dim,
            'hidden_dim': fc_hidden_dim,
            'cnn_channels': cnn_channels,
            'rnn_hidden_dim': rnn_hidden_dim,
        }
    }
    
    torch.save(checkpoint_to_save, output_path_pt)
    logger.info(f"🎉 Pre-training finished! Saved student model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xLSTM E2E 에이전트 행동 복제 및 사전 학습")
    parser.add_argument("--teacher_policy", type=str, required=True, help="기존 GRU E2E 모델 체크포인트 경로 (.pt)")
    parser.add_argument("--extracted_dir", type=str, default="data/extracted_episodes", help="추출된 npz 에피소드 폴더")
    parser.add_argument("--output_path", type=str, default="models/pretrain/xlstm_distilled.pt", help="결과 xLSTM 저장 경로")
    parser.add_argument("--epochs", type=int, default=5, help="훈련 에포크 수")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--temperature", type=float, default=2.0, help="증류 온도(Temperature)")
    parser.add_argument("--device", type=str, default="cuda", help="장치 (cuda 또는 cpu)")
    
    args = parser.parse_args()
    
    pretrain(
        teacher_path=args.teacher_policy,
        extracted_dir=args.extracted_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device
    )
