"""
Perception Module: Multi-Modal Perception Interface (Universal v1.1.1 "Sentient")
=============================================================================
Modular encoders for Vision, Audio, and Text streams with unified latent fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Any, Union

class VisionEncoder(nn.Module):
    """
    SOTA Vision Encoder: Implements a Vision Transformer (ViT) style backbone.
    Processes images as sequences of patches, enabling global dependency modeling
    across the entire visual field from the earliest layers.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, model_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        
        # Patch Projection
        self.projection = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dim_feedforward=model_dim*4, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional Embeddings (Learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, model_dim) * 0.02)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.projection(x) # [B, model_dim, H', W']
        h_p, w_p = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2) # [B, Seq, model_dim]
        
        # [V27] Robust 2D-Aware Positional Encoding
        # Always treat the embedding as a 2D grid and interpolate to the current feature map size (h_p, w_p).
        # This prevents 'Visual Dyslexia' when moving between resolutions.
        grid_size = int(math.sqrt(self.pos_embedding.size(1)))
        pos_tokens = self.pos_embedding.view(1, grid_size, grid_size, self.model_dim).permute(0, 3, 1, 2)
        
        # Scaling to current patch grid via bicubic interpolation
        interpolated_pos = F.interpolate(pos_tokens, size=(h_p, w_p), mode='bicubic', align_corners=False)
        x = x + interpolated_pos.permute(0, 2, 3, 1).flatten(1, 2)
        
        x = self.transformer(x)
        return self.norm(x)

class AudioEncoder(nn.Module):
    """
    SOTA Audio Encoder: Implements a Transformer-based Temporal Processing backbone.
    Projects multi-band spectrogram features into a high-dimensional latent space
    where temporal dependencies are captured via multi-head self-attention.
    """
    def __init__(self, in_features: int = 80, model_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(in_features, model_dim)
        
        # Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True, dim_feedforward=model_dim*4, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional Embeddings (Learned)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2048, model_dim) * 0.02)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, T] -> [B, T, F]
        x = x.transpose(1, 2)
        x = self.projection(x) # [B, T, model_dim]
        
        # Add positional embedding
        seq_len = x.size(1)
        if seq_len <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :seq_len, :]
            
        x = self.transformer(x)
        return self.norm(x)

class ModalityFuser(nn.Module):
    """
    SOTA Multi-Modal Fuser: Implements Cross-Modality Attention (XMA).
    Instead of simple concatenation, this module treats all modality tokens as part
    of a global recursive workspace, allowing dynamic interaction and fusion
    across Vision, Audio, and Text streams based on relevance and surprise.
    """
    def __init__(self, model_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.model_dim = model_dim
        # Learned modality-specific tokens
        self.modality_tokens = nn.ParameterDict({
            'vision': nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
            'audio': nn.Parameter(torch.randn(1, 1, model_dim) * 0.02),
            'text': nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        })
        
        # Cross-modality attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim)
        )

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_list = []
        for key, x in modalities.items():
            if key in self.modality_tokens:
                # Prepend modality token
                token = self.modality_tokens[key].expand(x.size(0), -1, -1)
                x = torch.cat([token, x], dim=1)
            fused_list.append(x)
        
        # Concatenate along sequence dimension
        fused = torch.cat(fused_list, dim=1) # [B, Seq_Total, model_dim]
        
        # Apply Cross-Modality Attention (Self-Attention over the concatenated sequence)
        attn_out, _ = self.cross_attention(fused, fused, fused)
        fused = self.norm(fused + attn_out)
        
        # Feed-forward refinement
        fused = self.norm(fused + self.ff(fused))
        
        return fused

class PerceptionGateway(nn.Module):
    """
    The main entry point for raw multi-modal inputs.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.model_dim = config.model_dim
        
        self.encoders = nn.ModuleDict()
        if hasattr(config, 'vision_dim') and config.vision_dim > 0:
            self.encoders['vision'] = VisionEncoder(
                in_channels=config.vision_dim, 
                model_dim=self.model_dim,
                num_layers=getattr(config, 'perception_layers', 2),
                num_heads=getattr(config, 'perception_heads', 4)
            )
        
        if hasattr(config, 'audio_dim') and config.audio_dim > 0:
            self.encoders['audio'] = AudioEncoder(
                in_features=config.audio_dim, 
                model_dim=self.model_dim,
                num_layers=getattr(config, 'perception_layers', 2),
                num_heads=getattr(config, 'perception_heads', 4)
            )
            
        self.fuser = ModalityFuser(
            model_dim=self.model_dim,
            num_heads=getattr(config, 'perception_heads', 4)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {}
        for key, x in inputs.items():
            if key in self.encoders:
                encoded[key] = self.encoders[key](x)
            elif key == 'text' and x.dim() == 3 and x.size(-1) == self.model_dim:
                # Text is already projected
                encoded[key] = x
            elif key == 'text' and x.dim() == 2:
                # Text is token IDs, handle elsewhere or add Embedding layer here
                pass
                
        if not encoded:
            return None
            
        fused = self.fuser(encoded)
        
        # [V31.8] STRATEGIC MODE: Holographic Feature Shunting
        # If we have a holographic anchor from Task 0, we dampen similar features
        # to force Task 1 to learn its own neural space.
        if hasattr(self, 'holographic_anchor') and self.holographic_anchor is not None:
            with torch.no_grad():
                # [B, S, D] -> [B, D]
                curr_feat = fused.mean(dim=1)
                anch_feat = self.holographic_anchor.mean(dim=1).to(fused.device)
                
                # Broad similarity across the batch
                sim = F.cosine_similarity(curr_feat, anch_feat.mean(dim=0, keepdim=True), dim=-1)
                
                # Damping curve: High similarity (>0.7) leads to exponential shunting
                # This makes Task 0 space 'expensive' for new gradients.
                shunt = torch.exp(-F.relu(sim - 0.7) * 4.0).view(-1, 1, 1)
                fused = fused * shunt
                
        return fused
