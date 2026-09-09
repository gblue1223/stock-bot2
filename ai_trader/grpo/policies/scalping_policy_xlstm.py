"""
GRPO 정책 네트워크 구현 (xLSTM 기반 End-to-End 강화학습용)

이 모듈은 원본 시계열(seq_len, features)을 직접 입력받아 행동을 결정하는 
xLSTM 기반의 E2E 정책 네트워크를 정의합니다.
"""

import logging
from typing import Tuple, Optional, Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.distributions import Categorical
from lib.observations import validate_max_stages

logger = logging.getLogger(__name__)


class mLSTMCell(nn.Module):
    """
    matrix memory와 exponential gating을 사용하는 mLSTMCell
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Q, K, V Projections
        self.w_q = nn.Linear(input_size, hidden_size, bias=False)
        self.w_k = nn.Linear(input_size, hidden_size, bias=False)
        self.w_v = nn.Linear(input_size, hidden_size, bias=False)
        
        # Gates: input, forget, output
        self.w_i = nn.Linear(input_size, 1)
        self.w_f = nn.Linear(input_size, 1)
        self.w_o = nn.Linear(input_size, hidden_size)
        
        # Output Group Normalization (stabilization)
        self.gn = nn.GroupNorm(1, hidden_size)

    def forward(
        self, 
        x: torch.Tensor, 
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch_size, input_size)
        # prev_state: (C_prev, n_prev) or None
        #   - C_prev: (batch_size, hidden_size, hidden_size)
        #   - n_prev: (batch_size, hidden_size, 1)
        batch_size = x.size(0)
        
        q = self.w_q(x)  # (batch_size, hidden_size)
        k = self.w_k(x)  # (batch_size, hidden_size)
        v = self.w_v(x)  # (batch_size, hidden_size)
        
        # Gate pre-activations
        i_gate = self.w_i(x)  # (batch_size, 1)
        f_gate = self.w_f(x)  # (batch_size, 1)
        o_gate = torch.sigmoid(self.w_o(x))  # (batch_size, hidden_size)
        
        # Bounded gating to prevent exponential explosion and NaNs
        f_act = torch.sigmoid(f_gate)
        i_act = torch.sigmoid(i_gate)
        
        # Reshape for matrix algebra
        q = q.unsqueeze(2)  # (batch_size, hidden_size, 1)
        k = k.unsqueeze(1)  # (batch_size, 1, hidden_size)
        v = v.unsqueeze(2)  # (batch_size, hidden_size, 1)
        
        if prev_state is None:
            C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=x.device, dtype=x.dtype)
            n = torch.zeros(batch_size, self.hidden_size, 1, device=x.device, dtype=x.dtype)
        else:
            C, n = prev_state
            
        # Update normalizer state: n_t = f_t * n_{t-1} + i_t * k_t^T
        n_next = f_act.unsqueeze(2) * n + i_act.unsqueeze(2) * k.transpose(1, 2)
        
        # Update matrix memory: C_t = f_t * C_{t-1} + i_t * (v_t * k_t)
        kv_prod = torch.bmm(v, k)  # (batch_size, hidden_size, hidden_size)
        C_next = f_act.unsqueeze(2) * C + i_act.unsqueeze(2) * kv_prod
        
        # Retrieve: C_t * q_t / (n_t^T * q_t)
        C_q = torch.bmm(C_next, q)  # (batch_size, hidden_size, 1)
        norm_factor = torch.bmm(n_next.transpose(1, 2), q)  # (batch_size, 1, 1)
        
        # Stabilized division to prevent division by zero or extremely small values
        norm_factor = torch.clamp(torch.abs(norm_factor), min=1e-5)
        
        h_tilde = C_q / norm_factor  # (batch_size, hidden_size, 1)
        h_tilde = h_tilde.squeeze(2)  # (batch_size, hidden_size)
        
        # Apply output gate and group norm
        h_out = self.gn(h_tilde) * o_gate
        
        return h_out, (C_next, n_next)


class mLSTMLayer(nn.Module):
    """
    nn.GRU를 대체하는 다층 mLSTM 레이어
    
    Gradient Checkpointing:
    시퀀스를 checkpoint_segments 개의 세그먼트로 나누어,
    세그먼트 경계의 state(C, n)만 보관하고 내부 중간결과는
    backward 시 재계산합니다. 이를 통해 VRAM 사용량을
    O(seq_len)에서 O(seq_len / checkpoint_segments)로 줄입니다.
    """
    def __init__(
        self, input_size: int, hidden_size: int,
        num_layers: int = 1, dropout: float = 0.0,
        checkpoint_segments: int = 16
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.checkpoint_segments = checkpoint_segments
        
        self.cells = nn.ModuleList([
            mLSTMCell(input_size if l == 0 else hidden_size, hidden_size) 
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _run_segment(cell: 'mLSTMCell', segment_input: torch.Tensor,
                     C: torch.Tensor, n: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        세그먼트 내 타임스텝을 순차 실행 (checkpoint 내부 함수).
        checkpoint 에 전달되려면 입출력이 모두 Tensor여야 하므로
        state tuple을 풀어서 받고 풀어서 반환합니다.
        """
        state = (C, n)
        outputs = []
        for t in range(segment_input.size(1)):
            x_t = segment_input[:, t, :]
            h_t, state = cell(x_t, state)
            outputs.append(h_t)
        seg_out = torch.stack(outputs, dim=1)  # (batch, seg_len, hidden)
        return seg_out, state[0], state[1]

    def forward(
        self, 
        x: torch.Tensor, 
        prev_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        # x: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        if prev_states is None:
            for _ in range(self.num_layers):
                states.append(None)
        else:
            states = list(prev_states)
            
        next_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        current_input = x
        
        for layer_idx, cell in enumerate(self.cells):
            # --- State 초기화 (checkpoint에 Tensor가 필요) ---
            raw_state = states[layer_idx]
            if raw_state is None:
                C = torch.zeros(batch_size, cell.hidden_size, cell.hidden_size,
                                device=x.device, dtype=x.dtype)
                n = torch.zeros(batch_size, cell.hidden_size, 1,
                                device=x.device, dtype=x.dtype)
            else:
                C, n = raw_state

            if self.training and seq_len > 1:
                # --- Gradient Checkpointing 모드 ---
                num_segs = min(self.checkpoint_segments, seq_len)
                seg_size = max(1, (seq_len + num_segs - 1) // num_segs)
                
                all_outputs: List[torch.Tensor] = []
                for seg_start in range(0, seq_len, seg_size):
                    seg_end = min(seg_start + seg_size, seq_len)
                    seg_input = current_input[:, seg_start:seg_end, :]

                    # _run_segment을 cell에 대해 바인딩한 래퍼
                    # cell 참조를 closure로 캡처하면 loop variable 문제가 있으므로
                    # 즉시 바인딩합니다.
                    def _make_fn(_cell):
                        def _fn(seg_in, _C, _n):
                            return mLSTMLayer._run_segment(_cell, seg_in, _C, _n)
                        return _fn

                    seg_out, C, n = cp.checkpoint(
                        _make_fn(cell),
                        seg_input, C, n,
                        use_reentrant=False
                    )
                    all_outputs.append(seg_out)
                
                layer_output = torch.cat(all_outputs, dim=1)
            else:
                # --- 추론 모드: checkpointing 불필요 ---
                layer_output_list = []
                state = (C, n)
                for t in range(seq_len):
                    x_t = current_input[:, t, :]
                    h_t, state = cell(x_t, state)
                    layer_output_list.append(h_t)
                layer_output = torch.stack(layer_output_list, dim=1)
                C, n = state

            # (batch_size, seq_len, hidden_size)
            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            current_input = layer_output
            next_states.append((C, n))
            
        return current_input, next_states


class GRPOPolicyE2EXLSTM(nn.Module):
    """
    GRPO를 위한 xLSTM 기반 End-to-End 정책 네트워크
    
    입력: 원본 틱 시계열 데이터 (batch_size, seq_len, obs_dim)
    출력: 행동 확률 분포 (3,) - 보유/매수/매도
    
    아키텍처:
    - Feature Extraction: 1D CNN
    - Sequence Modeling: mLSTM (matrix memory)
    - Dense Layers: mLSTM의 마지막 은닉 상태 -> MLP
    - 정책 헤드: hidden_dim -> action_dim
    - 가치 헤드: hidden_dim -> 1
    """
    def __init__(
        self,
        obs_dim: int = 43,
        cnn_channels: int = 64,
        rnn_hidden_dim: int = 128,
        fc_hidden_dim: int = 256,
        action_dim: int = 3,
        checkpoint_segments: int = 16,
        max_stages: int = 1
    ):
        super().__init__()
        self.max_stages = validate_max_stages(max_stages)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cnn_channels = cnn_channels
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        
        # 1. 1D CNN for local feature extraction and downsampling
        self.conv1 = nn.Conv1d(in_channels=obs_dim, out_channels=cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        
        # 2. Sequence Modeling with mLSTM (with gradient checkpointing)
        self.xlstm = mLSTMLayer(
            input_size=cnn_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            dropout=0.2,
            checkpoint_segments=checkpoint_segments
        )
        
        # 3. 공유 Dense 레이어
        self.fc1 = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        
        # 4. 정책 및 가치 헤드
        self.policy_head = nn.Linear(fc_hidden_dim, action_dim)
        self.value_head = nn.Linear(fc_hidden_dim, 1)
        # A PPO likelihood must depend on the observation and parameters only.
        # Keep checkpoint-compatible BatchNorm tensors, but freeze their statistics
        # and disable recurrent dropout in both rollout and gradient evaluation.
        self.train()
        
        logger.info(f"GRPOPolicyE2EXLSTM initialized: obs_dim={obs_dim}, "
                    f"cnn_channels={cnn_channels}, rnn_hidden_dim={rnn_hidden_dim}, "
                    f"action_dim={action_dim}")
                    
    def train(self, mode: bool = True):
        super().train(mode)
        self.bn1.eval()
        self.bn2.eval()
        self.xlstm.dropout.eval()
        return self

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책 네트워크 forward pass
        """
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if state.dim() == 2:
            state = state.unsqueeze(0)
            
        batch_size = state.size(0)
        
        # CNN 입력 형태: (batch_size, channels, seq_len)
        x = state.transpose(1, 2).contiguous()
        
        # CNN Layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # RNN 입력 형태: (batch_size, seq_len, channels)
        x = x.transpose(1, 2).contiguous()
        
        # xLSTM Layer
        lstm_out, _ = self.xlstm(x)
        # 마지막 타임스텝 사용
        xlstm_last = lstm_out[:, -1, :]
        
        # Dense Layers
        fc_out = F.relu(self.fc1(xlstm_last))
        fc_out = F.relu(self.fc2(fc_out))
        
        # 정책 및 가치 헤드
        action_logits = self.policy_head(fc_out)
        if self.obs_dim > 15 and self.action_dim == 3:
            active_stages = (state[:, -1, -15::3] > 0.5).sum(dim=-1)
            valid_actions = torch.stack([
                torch.ones_like(active_stages, dtype=torch.bool),
                active_stages < self.max_stages,
                active_stages > 0,
            ], dim=-1)
            action_logits = action_logits.masked_fill(
                ~valid_actions, torch.finfo(action_logits.dtype).min)
        state_value = self.value_head(fc_out)
        
        return action_logits, state_value
        
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        정책에서 행동 샘플링 (GRPO 호환)
        """
        action, dist = self._select_action(obs, deterministic)
        return action, dist.log_prob(action)

    def get_action_with_probabilities(self, obs: torch.Tensor, deterministic: bool = False):
        """Return the same action plus masked probabilities in one forward pass."""
        action, dist = self._select_action(obs, deterministic)
        return action, dist.log_prob(action), dist.probs

    def _select_action(self, obs: torch.Tensor, deterministic: bool):
        action_logits, _ = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
            
        return action, dist
        
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        주어진 상태와 행동에 대한 로그 확률, 엔트로피, 가치 계산
        """
        action_logits, state_values = self.forward(states)
        
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(state_values).any() or torch.isinf(state_values).any():
            state_values = torch.nan_to_num(state_values, nan=0.0, posinf=1.0, neginf=-1.0)
            
        dist = Categorical(logits=action_logits)
        
        actions_safe = torch.clamp(actions.long(), 0, self.action_dim - 1)
        log_probs = dist.log_prob(actions_safe)
        entropy = dist.entropy()
        values = state_values.squeeze(-1)
        
        return log_probs, entropy, values
