import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Iterator
import torch.nn.functional as F

class AdapterBank:
    """
    Parameter-efficient FiLM-style adapters per tracked layer.
    
    FIXES V2:
    - Zero-Init: Adapters start as Identity (no noise injection).
    - Robust Reshaping: Handles arbitrary spatial dimensions safely.
    - Strict Parameter Registration: Ensures optimizer can find params.
    """
    def __init__(self, num_layers: int = 0, device: torch.device = None):
        self.logger = logging.getLogger('AdapterBank')
        self.device = device if device else torch.device('cpu')
        self.num_layers = num_layers
        self.adapters: Dict[int, Dict[str, Any]] = {}
        
        # Pre-allocate slots (empty) to track existence
        for i in range(num_layers):
            self.adapters[i] = {'type': 'empty'}

    def ensure_index(self, idx: int, out_dim: int = None, force_upgrade: bool = False):
        """
        Ensure adapter exists for idx. 
        Promotes 'empty' or 'film' adapters to 'bneck' if out_dim is sufficiently large.
        [V9.4] force_upgrade: Triggered during Mind-Space expansion.
        """
        if idx not in self.adapters:
            self.adapters[idx] = {'type': 'empty'}
            
        entry = self.adapters[idx]
        current_type = entry.get('type', 'empty')

        # Case 1: Dimension unknown or small -> Use Scalar FiLM (Lightweight)
        if out_dim is None or out_dim <= 8:
            if current_type == 'empty':
                self.adapters[idx] = {
                    'type': 'film',
                    # Initialize to Identity: Scale=1.0, Shift=0.0
                    'scale': nn.Parameter(torch.ones(1, device=self.device, dtype=torch.float32)),
                    'shift': nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))
                }
            # If already film or bneck, leave as is (don't downgrade bneck)
            return

        # Case 2: Dimension known & large -> Use Bottleneck Adapter (High Capacity)
        # We upgrade if it's currently empty OR film (feature upgrade)
        # [V9.4] If force_upgrade is True, we promote even existing film adapters.
        if current_type in ['empty', 'film'] or force_upgrade:
            # Skip if it's already a bottleneck and we aren't resizing
            if current_type == 'bneck' and not force_upgrade: return
            
            r = max(4, min(64, out_dim // 8)) # Bottleneck ratio
            
            # Kaiming Init for Down projection (Information extraction)
            Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (2 / out_dim)**0.5)
            
            # ZERO Init for Up projection (Identity start)
            # This ensures the adapter output is 0.0 initially, so f(x) + adapter(x) = f(x)
            Wup = nn.Parameter(torch.zeros(r, out_dim, device=self.device))
            
            bdown = nn.Parameter(torch.zeros(r, device=self.device))
            bup = nn.Parameter(torch.zeros(out_dim, device=self.device))
            
            self.adapters[idx] = {
                'type': 'bneck',
                'Wdown': Wdown,
                'Wup': Wup,
                'bdown': bdown,
                'bup': bup,
                'r': r,
                'out_dim': out_dim
            }
        
        # Case 3: Resize existing bottleneck if dimension changed (rare but possible)
        elif current_type == 'bneck':
            if entry.get('out_dim') != out_dim:
                # Re-initialize to match new shape
                r = max(4, min(64, out_dim // 8))
                Wdown = nn.Parameter(torch.randn(out_dim, r, device=self.device) * (2 / out_dim)**0.5)
                Wup = nn.Parameter(torch.zeros(r, out_dim, device=self.device)) # Zero init
                bdown = nn.Parameter(torch.zeros(r, device=self.device))
                bup = nn.Parameter(torch.zeros(out_dim, device=self.device))
                
                self.adapters[idx] = {
                    'type': 'bneck',
                    'Wdown': Wdown,
                    'Wup': Wup,
                    'bdown': bdown,
                    'bup': bup,
                    'r': r,
                    'out_dim': out_dim
                }

    def apply(self, idx: int, activation: torch.Tensor, module_type: type) -> torch.Tensor:
        """Apply adapter to activation, using module_type to handle tensor shapes correctly."""
        if idx not in self.adapters:
            return activation

        entry = self.adapters[idx]
        adapter_type = entry.get('type')

        if adapter_type == 'empty':
            return activation

        try:
            # === FiLM ADAPTER (Scalar Scale & Shift) ===
            if adapter_type == 'film':
                return activation * entry['scale'] + entry['shift']

            # === BOTTLENECK ADAPTER (Low-Rank MLP) ===
            elif adapter_type == 'bneck':
                orig_dtype = activation.dtype
                Wdown = entry['Wdown']
                Wup = entry['Wup']
                bdown = entry['bdown']
                bup = entry['bup']

                # Case A: Convolutional Layers (Channel-First)
                is_conv = module_type in [nn.Conv1d, nn.Conv2d]
                if is_conv and activation.dim() > 2:
                    orig_shape = activation.shape
                    # Flatten spatial/temporal dims: (B, C, H, W) -> (B, C, N)
                    x_flat = activation.flatten(2)
                    # Permute for linear layer: (B, C, N) -> (B, N, C)
                    x_flat = x_flat.permute(0, 2, 1)
                    
                    # Apply adapter
                    z = F.silu(F.linear(x_flat.to(Wdown.dtype), Wdown.t(), bdown))
                    res = F.linear(z, Wup.t(), bup)
                    
                    # Restore original shape: (B, N, C) -> (B, C, N) -> (B, C, H, W)
                    res = res.permute(0, 2, 1).view(*orig_shape)
                    return activation + res.to(orig_dtype)

                # Case B: General Purpose (for Linear, LSTM, MHA, etc.)
                # Assumes feature dimension is the last dimension.
                else:
                    z = F.silu(F.linear(activation.to(Wdown.dtype), Wdown.t(), bdown))
                    res = F.linear(z, Wup.t(), bup)
                    return activation + res.to(orig_dtype)

        except Exception as e:
            # Failsafe to prevent crashes
            return activation

        return activation

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over adapter parameters for optimizers."""
        for v in self.adapters.values():
            if v.get('type') == 'film':
                yield v['scale']
                yield v['shift']
            elif v.get('type') == 'bneck':
                yield v['Wdown']
                yield v['Wup']
                yield v['bdown']
                yield v['bup']

    def state_dict(self):
        """Serializable state."""
        return {
            k: {
                key: val.cpu() if isinstance(val, torch.Tensor) else val 
                for key, val in v.items() 
                if key != 'type'
            } | {'type': v['type']}
            for k, v in self.adapters.items()
        }

    def load_state_dict(self, state_dict):
        """Restore state."""
        for k, v in state_dict.items():
            idx = int(k)
            atype = v.get('type', 'empty')
            
            if atype == 'film':
                self.adapters[idx] = {
                    'type': 'film',
                    'scale': nn.Parameter(v['scale'].to(self.device)),
                    'shift': nn.Parameter(v['shift'].to(self.device))
                }
            elif atype == 'bneck':
                self.adapters[idx] = {
                    'type': 'bneck',
                    'Wdown': nn.Parameter(v['Wdown'].to(self.device)),
                    'Wup': nn.Parameter(v['Wup'].to(self.device)),
                    'bdown': nn.Parameter(v['bdown'].to(self.device)),
                    'bup': nn.Parameter(v['bup'].to(self.device)),
                    'r': v['r'],
                    'out_dim': v['out_dim']
                }
            else:
                self.adapters[idx] = {'type': 'empty'}