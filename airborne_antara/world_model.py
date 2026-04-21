"""
Airborne-Antara V9.0 - World Modeling Module (Synthetic Intuition)
==============================================================
Implements Joint-Embedding Predictive Architecture (I-JEPA) components
to enable the framework to forecast its own future latent states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class JEPAPredictor(nn.Module):
    """
    Predicts the next latent state z_{t+1} given the current latent z_{t}
    and an optional action context. This is the core of 'Intuitive Foresight'.
    """
    def __init__(self, model_dim: int = 256, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = model_dim * 2
            
        # The Predictor is a small Transformer-style block or MLP-Mixer
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim)
        )
        
        # Sentient action context projection (Scale, Shift)
        self.action_proj = nn.Linear(2, model_dim)
        
    def forward(self, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: current latent embedding [B, Seq, Dim] or [B, Dim]
        action: current action/context [B, Dim]
        """
        x = z
        if action is not None:
            # Inject action context via addition or cross-attention
            x = x + self.action_proj(action).unsqueeze(1) if x.dim() == 3 else x + self.action_proj(action)
            
        return self.net(x)

class WorldModel(nn.Module):
    """
    Aggregates predictive logic and computes 'Predictive Surprise'.
    """
    def __init__(self, config: any):
        super().__init__()
        self.model_dim = config.model_dim
        self.predictor = JEPAPredictor(model_dim=self.model_dim)
        
        # Exponential moving average for predictive error (baseline)
        self.register_buffer("error_ema", torch.tensor(0.0))
        self.alpha = 0.95

    def forward(self, z_t: torch.Tensor, action_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forecasts z_{t+1}"""
        return self.predictor(z_t, action_t)

    def compute_surprise(self, z_pred: torch.Tensor, z_actual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the delta between predicted and actual latent states.
        Returns: (surprise_signal, prediction_loss)
        """
        # Mean Squared Error in latent space
        # z might be [B, Seq, Dim], so we average over Seq and Dim
        mse = F.mse_loss(z_pred, z_actual, reduction='none')
        if mse.dim() == 3:
            mse = mse.mean(dim=(1, 2))
        else:
            mse = mse.mean(dim=1)
            
        # Predictive Surprise: How much worse we did than our EMA
        surprise = torch.clamp(mse - self.error_ema, min=0.0)
        
        # Update EMA
        batch_avg_error = mse.mean().detach()
        self.error_ema = self.alpha * self.error_ema + (1 - self.alpha) * batch_avg_error
        
        return surprise, mse.mean()
