import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Any, List, Tuple, Optional

class ExpertBlock(nn.Module):
    """
    A wrapper for any neural module to act as an expert.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, task_id=None):
        return self.model(x)

class AdaptiveExpertBlock(nn.Module):
    """
    [V9.4] Memory-Efficient Expert.
    Instead of copying the base model, it uses a shared backbone
    with lightweight Task-Adaptive Adapters (FiLM/LoRA).
    Ensures 8GB GPU stability with many experts.
    """
    def __init__(self, shared_backbone, input_dim, adapter_dim=32):
        super().__init__()
        self.backbone = shared_backbone # Reference to shared weights
        # Expert-specific modulation
        self.modulation = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, input_dim * 2) # For Scale and Shift
        )

    def forward(self, x, task_id=None):
        # 1. Generate local modulation from input
        # Pool seq if needed
        if x.dim() == 3: 
            x_flat = x.mean(dim=1)
        else: 
            x_flat = x.view(x.size(0), -1)
        
        # Verify alignment
        if x_flat.size(-1) != self.modulation[0].in_features:
            x_flat = F.adaptive_avg_pool1d(x_flat.unsqueeze(1), self.modulation[0].in_features).squeeze(1)

        mod = self.modulation(x_flat)
        scale, shift = torch.chunk(mod, 2, dim=-1)
        
        # [V31.7] Bounded Scale: Prevent FiLM explosion
        scale = torch.tanh(scale)
        
        # 2. Reshape for broadcasting
        view_shape = [x.size(0)] + [1] * (x.dim() - 1)
        scale = scale.view(*view_shape)
        shift = shift.view(*view_shape)
        
        # 3. Apply shared backbone with local modulation
        # This is essentially a dynamic FiLM-Expert
        return self.backbone(x * (1 + scale) + shift)

class GatingNetwork(nn.Module):
    """
    Router that decides which experts to activate.
    """
    def __init__(self, input_dim, num_experts, top_k=2, temperature=1.0):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, x, task_id=None):
        # [V17] Hard Reset of local cache to prevent graph leakage
        self.aux_loss = torch.tensor(0.0, device=x.device)
        
        # x: [batch_size, input_dim] or [batch_size, seq, input_dim]
        if x.dim() == 3:
            # SOTA Sequence Pooling: [B, S, D] -> [B, D]
            x_flat = x.mean(dim=1)
        elif x.dim() == 4:
            # [V31.7] Image Spatial Pooling: [B, C, H, W] -> [B, C]
            # This handles Bug #4 by ensuring image features are pooled before gating.
            x_flat = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        elif x.dim() > 4:
            # Flatten multi-dim inputs
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        # Verify alignment with gate input dimension
        if x_flat.size(-1) != self.gate.in_features:
            # Fallback: Force resize if possible, or raise clear error
            if x_flat.numel() % self.gate.in_features == 0:
                x_flat = x_flat.view(x.size(0), self.gate.in_features)
            else:
                raise ValueError(f"SOTA MoE Shape Mismatch: Got {x_flat.shape[1]}, expected {self.gate.in_features}. "
                                 f"Original input: {x.shape}")

        logits = self.gate(x_flat) / max(self.temperature, 1e-8)
        
        # [V27] Autonomous Feature-Based Routing
        # We removed the hard task-id mask to ensure the router learns to 
        # map input features to experts without relying on extrinsic indicators.
        # This is critical for Class-IL compliance and structural integrity.
        
        # Top-k gating
        # Keep top k values, set others to -inf
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Softmax over top-k
        weights = F.softmax(top_k_logits, dim=1)
        
        # [V9.0] Load Balancing Loss (Auxiliary)
        batch_size = top_k_indices.size(0)
        mask = torch.zeros(batch_size, self.gate.out_features, device=x.device)
        mask.scatter_(1, top_k_indices, weights)
        importance = mask.sum(dim=0)
        mean_imp = importance.mean() + 1e-6
        var_imp = importance.var()
        self.aux_loss = (var_imp / (mean_imp ** 2)) * 1.0 
        
        return weights, top_k_indices

    def get_aux_loss(self):
        val = getattr(self, 'aux_loss', 0.0)
        return val if val is not None else 0.0

class SparseMoE(nn.Module):
    """
    The Sparse Mixture of Experts Container.
    """
    def __init__(self, base_model, input_dim, num_experts=4, top_k=2, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_adaptive = getattr(base_model, '_moe_adaptive', False)
        
        if self.use_adaptive:
            self.experts = nn.ModuleList([
                AdaptiveExpertBlock(base_model, input_dim) for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                ExpertBlock(copy.deepcopy(base_model)) for _ in range(num_experts)
            ])
        
        self.gate = GatingNetwork(input_dim, num_experts, top_k, temperature)
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def set_temperature(self, temperature):
        self.gate.temperature = temperature

    def get_usage_stats(self):
        return self.expert_usage.cpu().numpy()
        
    def get_aux_loss(self):
        return self.gate.get_aux_loss()

    def forward(self, x, task_id=None):
        weights, indices = self.gate(x, task_id=task_id)
        
        with torch.no_grad():
            # [V26.0] Vectorized usage update
            flat_indices = indices.view(-1)
            self.expert_usage.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=self.expert_usage.dtype))
        
        batch_size = x.size(0)
        final_output = None
        
        for i in range(self.num_experts):
            mask = (indices == i)
            batch_idx, k_idx = torch.where(mask)
            if len(batch_idx) == 0: continue
            
            selected_inputs = x[batch_idx]
            expert_out = self.experts[i](selected_inputs, task_id=task_id)
            
            if final_output is None:
                out_shape = list(expert_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=expert_out.dtype)
            
            selected_weights = weights[batch_idx, k_idx]
            view_shape = [len(batch_idx)] + [1] * (expert_out.dim() - 1)
            selected_weights = selected_weights.view(*view_shape)
            final_output = final_output.index_add(0, batch_idx, expert_out * selected_weights)
            
        return final_output, indices

class HierarchicalMoE(nn.Module):
    """
    [V9.0] Hierarchical Mixture of Experts.
    """
    def __init__(self, base_model, input_dim, num_domains=2, experts_per_domain=2, top_k=1, temperature=1.0):
        super().__init__()
        self.num_domains = num_domains
        self.experts_per_domain = experts_per_domain
        self.top_k = top_k
        self.domain_router = GatingNetwork(input_dim, num_domains, top_k=1, temperature=temperature)
        self.domains = nn.ModuleList([
            SparseMoE(base_model, input_dim, num_experts=experts_per_domain, top_k=top_k, temperature=temperature)
            for _ in range(num_domains)
        ])

    def set_temperature(self, temperature):
        self.domain_router.temperature = temperature
        for domain in self.domains:
            domain.set_temperature(temperature)
        
    def get_expert_usage(self):
        usage = {}
        idx = 0
        for domain in self.domains:
            stats = domain.get_usage_stats()
            for s in stats:
                usage[idx] = float(s)
                idx += 1
        return usage
        
    def reset_usage(self):
        for domain in self.domains:
            domain.expert_usage.zero_()
    
    def forward(self, x, task_id=None):
        domain_weights, domain_indices = self.domain_router(x, task_id=task_id)
        batch_size = x.size(0)
        final_output = None
        
        for i in range(self.num_domains):
            mask = (domain_indices == i).any(dim=1)
            batch_idx = torch.where(mask)[0]
            if len(batch_idx) == 0: continue
                
            selected_inputs = x[batch_idx]
            domain_out, _ = self.domains[i](selected_inputs, task_id=task_id)
            
            if final_output is None:
                out_shape = list(domain_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=domain_out.dtype)
            
            w = domain_weights[batch_idx, 0].view(len(batch_idx), *([1] * (domain_out.dim() - 1)))
            final_output = final_output.index_add(0, batch_idx, domain_out * w)
            
        return final_output, domain_indices

    def get_aux_loss(self):
        total_loss = 0.0
        if hasattr(self.domain_router, 'get_aux_loss'):
            val = self.domain_router.get_aux_loss()
            if val is not None:
                total_loss += val
        for domain in self.domains:
            if hasattr(domain, 'get_aux_loss'):
                val = domain.get_aux_loss()
                if val is not None:
                    total_loss += val
        return total_loss
