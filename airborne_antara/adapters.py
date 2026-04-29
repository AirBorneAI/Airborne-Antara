import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Iterator
import torch.nn.functional as F

class FiLMAdapter(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        # Initializing on CPU by default if device is None, framework will move it
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        if device: self.to(device)

    def forward(self, x):
        return x * self.scale + self.shift

class BottleneckAdapter(nn.Module):
    def __init__(self, out_dim: int, r: int, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.r = r
        
        # Kaiming Init for Down projection
        self.Wdown = nn.Parameter(torch.randn(out_dim, r) * (2 / out_dim)**0.5)
        # ZERO Init for Up projection (Identity start)
        self.Wup = nn.Parameter(torch.zeros(r, out_dim))
        
        self.bdown = nn.Parameter(torch.zeros(r))
        self.bup = nn.Parameter(torch.zeros(out_dim))
        
        if device: self.to(device)

    def forward(self, activation, module_type):
        orig_dtype = activation.dtype
        
        # Case A: Convolutional Layers (Channel-First)
        is_conv = module_type in [nn.Conv1d, nn.Conv2d]
        if is_conv and activation.dim() > 2:
            orig_shape = activation.shape
            # (B, C, H, W) -> (B, C, N) -> (B, N, C)
            x_flat = activation.flatten(2).permute(0, 2, 1)
            
            z = F.silu(F.linear(x_flat.to(self.Wdown.dtype), self.Wdown.t(), self.bdown))
            res = F.linear(z, self.Wup.t(), self.bup)
            
            # (B, N, C) -> (B, C, N) -> (B, C, H, W)
            res = res.permute(0, 2, 1).view(*orig_shape)
            return activation + res.to(orig_dtype)

        # Case B: General Purpose
        else:
            z = F.silu(F.linear(activation.to(self.Wdown.dtype), self.Wdown.t(), self.bdown))
            res = F.linear(z, self.Wup.t(), self.bup)
            return activation + res.to(orig_dtype)

class AdapterBank(nn.Module):
    """
    Parameter-efficient FiLM-style adapters per tracked layer.
    V26.1: Strictly registered as nn.Module for perfect device affinity.
    """
    def __init__(self, num_layers: int = 0, device: torch.device = None):
        super().__init__()
        self.logger = logging.getLogger('AdapterBank')
        self.num_layers = num_layers
        self.adapters = nn.ModuleDict()
        
        # Pre-allocate slots
        for i in range(num_layers):
            self.adapters[str(i)] = nn.Identity() # Placeholder

    def ensure_index(self, idx: int, out_dim: int = None, force_upgrade: bool = False):
        idx_str = str(idx)
        current_adapter = self.adapters.get(idx_str, None)
        
        # Determine current type
        current_type = 'empty'
        if isinstance(current_adapter, FiLMAdapter): current_type = 'film'
        elif isinstance(current_adapter, BottleneckAdapter): current_type = 'bneck'
        elif isinstance(current_adapter, nn.Identity): current_type = 'empty'

        # Upgrade Logic
        if out_dim is None or out_dim <= 8:
            if current_type == 'empty':
                self.adapters[idx_str] = FiLMAdapter()
            return

        if current_type in ['empty', 'film'] or force_upgrade:
            if current_type == 'bneck' and not force_upgrade: return
            
            r = max(4, min(64, out_dim // 8))
            self.adapters[idx_str] = BottleneckAdapter(out_dim, r)
        
        elif current_type == 'bneck':
            if current_adapter.out_dim != out_dim:
                r = max(4, min(64, out_dim // 8))
                self.adapters[idx_str] = BottleneckAdapter(out_dim, r)

    def apply(self, idx: int, activation: torch.Tensor, module_type: type) -> torch.Tensor:
        idx_str = str(idx)
        if idx_str not in self.adapters:
            return activation
        
        adapter = self.adapters[idx_str]
        if isinstance(adapter, nn.Identity):
            return activation
            
        try:
            if isinstance(adapter, FiLMAdapter):
                return adapter(activation)
            elif isinstance(adapter, BottleneckAdapter):
                return adapter(activation, module_type)
        except Exception:
            return activation
            
        return activation

    def state_dict(self, *args, **kwargs):
        # We want a format compatible with previous version's state_dict if possible
        # but the structure has changed. Let's provide a clean state_dict.
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        # [V26.1] Custom load to handle structural shifts from V1 to V2
        # If we see 'Wdown' or 'scale' in the first level, it's the old format.
        # But usually AdaptiveFramework saves 'adapters' which is our state_dict.
        super().load_state_dict(state_dict, strict=strict)