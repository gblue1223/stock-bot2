from typing import Dict, Tuple

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box


class TimeSeriesCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for time-series windows.
    Accepts observations shaped either as flat (T*F,) or matrix (T, F).
    Internally reshapes to [B, 1, T, F] and applies small Conv2d stack.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256, seq_len: int | None = None, num_features: int | None = None):
        # Determine input shape
        if len(observation_space.shape) == 1:
            # Flat: (T*F,)
            super().__init__(observation_space, features_dim)
            self.flat = True
            self.seq_len = seq_len
            self.num_features = num_features
            if self.seq_len is None or self.num_features is None:
                raise ValueError("For flat observations, seq_len and num_features must be provided in features_extractor_kwargs")
            in_shape = (1, self.seq_len, self.num_features)
        elif len(observation_space.shape) == 2:
            # Matrix: (T, F)
            super().__init__(observation_space, features_dim)
            self.flat = False
            T, F = observation_space.shape
            in_shape = (1, T, F)
        else:
            raise ValueError(f"Unsupported observation shape: {observation_space.shape}")

        C, T, F = in_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((T // 2 if T >= 2 else 1, F // 2 if F >= 2 else 1)),
        )
        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, T, F)
            conv_out = self.cnn(dummy)
            conv_dim = conv_out.view(1, -1).shape[1]
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.flat:
            # obs: [B, T*F] -> [B, 1, T, F]
            b, d = obs.shape
            obs = obs.view(b, 1, self.seq_len, self.num_features)
        else:
            # obs: [B, T, F] -> [B, 1, T, F]
            obs = obs.unsqueeze(1)
        h = self.cnn(obs)
        return self.linear(h)
