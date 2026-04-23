import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class ExpertBlock(nn.Module):
    """
    A wrapper for any neural module to act as an expert.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
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
    
    def forward(self, x):
        # 1. Generate local modulation from input
        # Pool seq if needed
        if x.dim() == 3: x_flat = x.mean(dim=1)
        else: x_flat = x.view(x.size(0), -1)
        
        # Verify alignment
        if x_flat.size(-1) != self.modulation[0].in_features:
            x_flat = F.adaptive_avg_pool1d(x_flat.unsqueeze(1), self.modulation[0].in_features).squeeze(1)

        mod = self.modulation(x_flat)
        scale, shift = torch.chunk(mod, 2, dim=-1)
        
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
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch_size, input_dim] or [batch_size, seq, input_dim]
        if x.dim() == 3:
            # SOTA Sequence Pooling: [B, S, D] -> [B, D]
            x_flat = x.mean(dim=1)
        elif x.dim() > 3:
            # Flatten multi-dim inputs (Images, etc)
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

        logits = self.gate(x_flat)
        
        # Top-k gating
        # Keep top k values, set others to -inf
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Softmax over top-k
        weights = F.softmax(top_k_logits, dim=1)
        
        # [V9.0] Load Balancing Loss (Auxiliary)
        # Goal: Minimize Coefficient of Variation (CV) of expert usage
        # Importance = Sum of gate probabilities for each expert
        # Since we only have top-k logits, we approximate importance using the selected weights
        
        batch_size = top_k_indices.size(0)

        # Create full probability vector (scatter)
        # [B, NumExperts]
        mask = torch.zeros(batch_size, self.gate.out_features, device=x.device)
        mask.scatter_(1, top_k_indices, weights)
        
        # Importance: [NumExperts]
        importance = mask.sum(dim=0)
        
        # CV squared = (std / mean)^2
        # std = sqrt(E[x^2] - (E[x])^2)
        # CV^2 = var / mean^2
        
        # Add small epsilon to mean to prevent div by zero
        mean_imp = importance.mean() + 1e-6
        var_imp = importance.var()
        self.aux_loss = (var_imp / (mean_imp ** 2)) * 1.0 # Scaling factor
        
        return weights, top_k_indices

    def get_aux_loss(self):
        return getattr(self, 'aux_loss', 0.0)

class SparseMoE(nn.Module):
    """
    The Sparse Mixture of Experts Container.
    """
    def __init__(self, base_model, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # [V9.4] Adaptive Mode: Share weights to save VRAM
        self.use_adaptive = getattr(base_model, '_moe_adaptive', False)
        
        if self.use_adaptive:
            # Shared backbone experts
            self.experts = nn.ModuleList([
                AdaptiveExpertBlock(base_model, input_dim) for _ in range(num_experts)
            ])
        else:
            # Full model experts (Heavy)
            self.experts = nn.ModuleList([
                ExpertBlock(copy.deepcopy(base_model)) for _ in range(num_experts)
            ])
        
        self.gate = GatingNetwork(input_dim, num_experts, top_k)
        
        # [V9.0] Analysis: Track expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def get_usage_stats(self):
        return self.expert_usage.cpu().numpy()
        
    def get_aux_loss(self):
        return self.gate.get_aux_loss()

    def forward(self, x):
        # x: [batch_size, ...]
        
        # 1. Gating
        # We need a feature vector for gating. 
        # If x is image [B, C, H, W], we flatten.
        # If x is sequence [B, S, D], we might gate per token or per sequence.
        # For simplicity V7.0: Gate per sample (using flattened input).
        
        # [V9.0] Let GatingNetwork handle Pooling/Flattening
        weights, indices = self.gate(x) # weights: [B, k], indices: [B, k]
        
        # [V9.0] Track usage
        with torch.no_grad():
            flat_indices = indices.view(-1)
            for idx in flat_indices:
                if idx < self.num_experts:
                    self.expert_usage[idx] += 1
        
        batch_size = x.size(0)
        
        # Initialize output
        # We need to know the output shape of the expert.
        # We run one expert on a dummy input or the first sample to get shape?
        # Or just rely on the loop.
        
        final_output = None
        
        # 2. Dispatch & Execute
        for i in range(self.num_experts):
            # Find batch indices where expert i is selected
            mask = (indices == i) # [B, k]
            batch_idx, k_idx = torch.where(mask)
            
            if len(batch_idx) == 0:
                continue
            
            # Get inputs for this expert
            selected_inputs = x[batch_idx]
            
            # Run expert
            expert_out = self.experts[i](selected_inputs)
            
            # Initialize final_output on first valid execution
            if final_output is None:
                out_shape = list(expert_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=expert_out.dtype)
            
            # Get weights
            selected_weights = weights[batch_idx, k_idx] # [N_selected]
            
            # Reshape weights to match output dimensions for broadcasting
            # e.g. if output is [N, 10], weights need to be [N, 1]
            view_shape = [len(batch_idx)] + [1] * (expert_out.dim() - 1)
            selected_weights = selected_weights.view(*view_shape)
            
            # Accumulate
            final_output = final_output.index_add(0, batch_idx, expert_out * selected_weights)
            
        return final_output, indices

class HierarchicalMoE(nn.Module):
    """
    [V9.0] Hierarchical Mixture of Experts.
    Uses tree-based routing for multi-domain specialization.
    Level 1 Router -> Domain Clusters -> Level 2 Experts.
    """
    def __init__(self, base_model, input_dim, num_domains=2, experts_per_domain=2, top_k=1):
        super().__init__()
        self.num_domains = num_domains
        self.experts_per_domain = experts_per_domain
        self.top_k = top_k
        
        # Level 1 Router: Selects which Domain Cluster to use
        self.domain_router = GatingNetwork(input_dim, num_domains, top_k=1)
        
        # Level 2: Domain Clusters (each is a SparseMoE)
        self.domains = nn.ModuleList([
            SparseMoE(base_model, input_dim, num_experts=experts_per_domain, top_k=top_k)
            for _ in range(num_domains)
        ])
        
    def get_expert_usage(self):
        """Returns flattened usage counts across all domains."""
        usage = {}
        idx = 0
        for domain in self.domains:
            stats = domain.get_usage_stats()
            for s in stats:
                usage[idx] = float(s)
                idx += 1
        return usage
        
    def reset_usage(self):
        """Reset usage stats for windowed monitoring."""
        for domain in self.domains:
            domain.expert_usage.zero_()
    
    def forward(self, x):
        """
        Hierarchical Routing:
        1. Select Domain Cluster (Level 1)
        2. Within Domain, Select Experts (Level 2)
        """
        # [V9.0] Let GatingNetwork handle Pooling/Flattening
        # 1. Level 1 Gating
        domain_weights, domain_indices = self.domain_router(x) # [B, 1]
        
        batch_size = x.size(0)
        final_output = None
        
        # 2. Sequential Dispatch to Domain Clusters
        for i in range(self.num_domains):
            # Mask for samples belonging to domain i
            mask = (domain_indices == i).any(dim=1)
            batch_idx = torch.where(mask)[0]
            
            if len(batch_idx) == 0:
                continue
                
            selected_inputs = x[batch_idx]
            
            # 3. Level 2 Gating happens inside the SparseMoE domain
            domain_out, _ = self.domains[i](selected_inputs)
            
            if final_output is None:
                out_shape = list(domain_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=domain_out.dtype)
            
            # Apply Level 1 weight (importance of this domain)
            # Find the weight for domain i among the top-k (which is top-1 here)
            d_w_mask = (domain_indices[batch_idx] == i)
            # Get the weight where the index matched
            # This is simplified because top_k=1
            w = domain_weights[batch_idx, 0].view(len(batch_idx), *([1] * (domain_out.dim() - 1)))
            
            final_output = final_output.index_add(0, batch_idx, domain_out * w)
            
        return final_output, domain_indices

    def get_aux_loss(self):
        """Aggregate load balancing loss from all levels."""
        total_loss = 0.0
        # Level 1 Router
        if hasattr(self.domain_router, 'get_aux_loss'):
            total_loss += self.domain_router.get_aux_loss()
            
        # Level 2 Experts
        for domain in self.domains:
            if hasattr(domain, 'get_aux_loss'):
                total_loss += domain.get_aux_loss()
                
        return total_loss
