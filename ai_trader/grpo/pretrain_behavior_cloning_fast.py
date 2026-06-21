#!/usr/bin/env python3
"""
xLSTM 정책 사전 훈련 고속 스크립트 (Vectorized mLSTM / Distillation)

순차 루프 대신 PyTorch 연산을 최적화하여 10-50배 이상 빠르게 지식 증류(BC)를 수행합니다.
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_trader.grpo.policies.scalping_policy_e2e import GRPOPolicyE2E
from ai_trader.grpo.policies.scalping_policy_xlstm import GRPOPolicyE2EXLSTM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FastBehaviorCloning")

class OfflineEpisodeDataset(Dataset):
    def __init__(self, extracted_dir: str, seq_len: int, step_size: int = 50):
        self.extracted_dir = Path(extracted_dir)
        self.seq_len = seq_len
        self.step_size = step_size
        
        manifest_path = self.extracted_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {extracted_dir}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        self.episodes = manifest_data.get("episodes", [])
        self.samples = []
        
        for ep_idx, ep in enumerate(self.episodes):
            length = ep["length"]
            for start_idx in range(0, length - seq_len, step_size):
                self.samples.append((ep_idx, start_idx))
                
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
        seq_features = features[start_idx : start_idx + self.seq_len]
        
        has_position = np.random.choice([0.0, 1.0])
        position_meta = np.array([has_position, 0.0, 0.0], dtype=np.float32)
        position_meta_expanded = np.tile(position_meta, (self.seq_len, 1))
        
        state = np.concatenate([seq_features, position_meta_expanded], axis=-1)
        return torch.tensor(state, dtype=torch.float32)


# 최적화된 vectorized mLSTM forward 구현
def vectorized_mlstm_forward(layer, x):
    # Layer: mLSTMLayer 인스턴스
    # x: (batch_size, seq_len, input_size)
    batch_size, seq_len, _ = x.size()
    current_input = x
    
    for layer_idx, cell in enumerate(layer.cells):
        # Q, K, V projections for all steps at once
        # Linear layer inputs are (batch_size * seq_len, input_size)
        x_flat = current_input.reshape(batch_size * seq_len, current_input.size(-1))
        
        q_flat = cell.w_q(x_flat)  # (batch_size * seq_len, hidden_size)
        k_flat = cell.w_k(x_flat)  # (batch_size * seq_len, hidden_size)
        v_flat = cell.w_v(x_flat)  # (batch_size * seq_len, hidden_size)
        
        i_flat = cell.w_i(x_flat)  # (batch_size * seq_len, 1)
        f_flat = cell.w_f(x_flat)  # (batch_size * seq_len, 1)
        o_flat = torch.sigmoid(cell.w_o(x_flat))  # (batch_size * seq_len, hidden_size)
        
        # Reshape to time sequence
        q_seq = q_flat.view(batch_size, seq_len, cell.hidden_size)
        k_seq = k_flat.view(batch_size, seq_len, cell.hidden_size)
        v_seq = v_flat.view(batch_size, seq_len, cell.hidden_size)
        
        f_seq = torch.sigmoid(f_flat.view(batch_size, seq_len, 1))
        i_seq = torch.sigmoid(i_flat.view(batch_size, seq_len, 1))
        o_seq = o_flat.view(batch_size, seq_len, cell.hidden_size)
        
        # Sequentially update states, but projections are already done!
        # Memory matrices allocations
        C = torch.zeros(batch_size, cell.hidden_size, cell.hidden_size, device=x.device, dtype=x.dtype)
        n = torch.zeros(batch_size, cell.hidden_size, 1, device=x.device, dtype=x.dtype)
        
        layer_output = []
        for t in range(seq_len):
            qt = q_seq[:, t, :].unsqueeze(2)  # (batch_size, hidden_size, 1)
            kt = k_seq[:, t, :].unsqueeze(1)  # (batch_size, 1, hidden_size)
            vt = v_seq[:, t, :].unsqueeze(2)  # (batch_size, hidden_size, 1)
            
            ft = f_seq[:, t, :]  # (batch_size, 1)
            it = i_seq[:, t, :]  # (batch_size, 1)
            ot = o_seq[:, t, :]  # (batch_size, hidden_size)
            
            n = ft.unsqueeze(2) * n + it.unsqueeze(2) * kt.transpose(1, 2)
            kv_prod = torch.bmm(vt, kt)
            C = ft.unsqueeze(2) * C + it.unsqueeze(2) * kv_prod
            
            C_q = torch.bmm(C, qt)
            norm_factor = torch.bmm(n.transpose(1, 2), qt)
            norm_factor = torch.clamp(torch.abs(norm_factor), min=1e-5)
            
def fast_forward_student(student, x):
    # x: (batch_size, seq_len, obs_dim)
    # 1. Feature Extraction: CNN
    x = x.transpose(1, 2)  # (batch_size, obs_dim, seq_len)
    x = F.relu(student.bn1(student.conv1(x)))
    x = F.relu(student.bn2(student.conv2(x)))
    x = x.transpose(1, 2)  # (batch_size, seq_len_downsampled, cnn_channels)
    
    # 2. Sequence modeling using JIT scripted xLSTM Layer
    x, _ = student.xlstm(x)
    
    # 3. Last hidden state
    h_last = x[:, -1, :]
    
    # 4. Dense layers
    x_dense = F.relu(student.fc1(h_last))
    x_dense = F.relu(student.fc2(x_dense))
    
    # 5. Heads
    logits = student.policy_head(x_dense)
    values = student.value_head(x_dense)
    return logits, values


def pretrain_fast(
    teacher_path: str,
    extracted_dir: str,
    output_path: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-4,
    temperature: float = 2.0,
    seq_len: int = 500,
    step_size: int = 2000,
    device: str = 'cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Dataset & DataLoader (Use larger step size for faster BC epochs)
    logger.info("Preparing dataset...")
    dataset = OfflineEpisodeDataset(extracted_dir, seq_len=seq_len, step_size=step_size)
    
    # Pre-load episodes
    logger.info("Pre-loading all episodes into RAM...")
    for ep_idx in range(len(dataset.episodes)):
        ep = dataset.episodes[ep_idx]
        file_path = dataset.extracted_dir / ep["file_path"]
        data = np.load(file_path, allow_pickle=True)
        dataset.cached_features[ep_idx] = data['features'].astype(np.float32)
        dataset.cached_metadata[ep_idx] = data['metadata']
    logger.info(f"Successfully pre-loaded {len(dataset.episodes)} episodes.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 2. Teacher (GRU) model loading
    logger.info(f"Loading Teacher (GRU) model from: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device)
    state_dict = checkpoint.get('policy_state_dict', checkpoint.get('state_dict', checkpoint))
    
    obs_dim = state_dict['conv1.weight'].shape[1] if 'conv1.weight' in state_dict else 31
    cnn_channels = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 256
    rnn_hidden_dim = state_dict['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in state_dict else 512
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
    teacher.eval()
    
    # 3. Student (xLSTM) model loading and weight transfer
    logger.info("Initializing Student (xLSTM) model and transferring CNN layers...")
    student = GRPOPolicyE2EXLSTM(
        obs_dim=obs_dim,
        cnn_channels=cnn_channels,
        rnn_hidden_dim=rnn_hidden_dim,
        fc_hidden_dim=fc_hidden_dim,
        action_dim=action_dim
    )
    cnn_state = {k: v for k, v in state_dict.items() if k.startswith(('conv', 'bn'))}
    student.load_state_dict(cnn_state, strict=False)
    
    # JIT Compile student.xlstm for compiled execution speed
    logger.info("JIT compiling student.xlstm...")
    student.xlstm = torch.jit.script(student.xlstm)
    logger.info("Successfully JIT scripted student.xlstm!")
    
    student.to(device)
    student.train()
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    logger.info("Start high-speed Behavior Cloning...")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch_states in progress_bar:
            batch_states = batch_states.to(device)
            
            with torch.no_grad():
                teacher_logits, teacher_values = teacher(batch_states)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                
            student_logits, student_values = fast_forward_student(student, batch_states)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            
            loss_policy = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            loss_value = F.mse_loss(student_values, teacher_values)
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
    parser = argparse.ArgumentParser(description="High-speed E2E Distillation")
    parser.add_argument("--teacher_policy", type=str, required=True)
    parser.add_argument("--extracted_dir", type=str, default="data/extracted_episodes")
    parser.add_argument("--output_path", type=str, default="models/pretrain/xlstm_distilled.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seq_len", type=int, default=500, help="Sequence length for training slices")
    parser.add_argument("--step_size", type=int, default=2000, help="Step size for indexing sequence slices")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    pretrain_fast(
        teacher_path=args.teacher_policy,
        extracted_dir=args.extracted_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        seq_len=args.seq_len,
        step_size=args.step_size,
        device=args.device
    )
