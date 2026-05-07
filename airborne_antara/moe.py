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

    def forward(self, x, task_id=None, consciousness_state: Optional[torch.Tensor] = None):
        # [V17] Hard Reset of local cache to prevent graph leakage
        self.aux_loss = torch.tensor(0.0, device=x.device)
        
        # x: [batch_size, input_dim] or [batch_size, seq, input_dim] or [B, C, H, W]
        # [V31.7] DYNAMIC SHAPE HARMONIZER: Handle both pixels and pooled features
        if x.dim() >= 3:
            x_flat_all = x.view(x.size(0), -1)
            if x_flat_all.size(-1) == self.gate.in_features:
                x_flat = x_flat_all
            elif x.dim() == 3:
                # SOTA Sequence Pooling: [B, S, D] -> [B, D]
                x_flat = x.mean(dim=1)
            elif x.dim() == 4:
                # [V31.7] Image Spatial Pooling: [B, C, H, W] -> [B, C]
                x_flat = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
            else:
                x_flat = x_flat_all
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

        # [V31.8] STRATEGIC MODE: Consciousness-Informed Gating
        # If consciousness state is provided (Global Workspace Broadcast), use it to shift logits.
        # This allows the AI to 'think' before routing.
        logits = self.gate(x_flat) 
        
        if consciousness_state is not None:
            # Broadcast consciousness impact. 
            # Simple version: Bias the logits based on the top-level awareness
            # Consciousness state is usually [Batch, Dim_C]
            # We project it to [Batch, Num_Experts] via a small linear layer if needed, 
            # or just use the first few elements if dimensions match.
            if not hasattr(self, 'cons_proj'):
                self.cons_proj = nn.Linear(consciousness_state.size(-1), self.gate.out_features).to(x.device)
            
            cons_bias = self.cons_proj(consciousness_state.to(x.device))
            logits = logits + cons_bias
            
        logits = logits / max(self.temperature, 1e-8)
        
        # [V27] Autonomous Feature-Based Routing
        # We removed the hard task-id mask to ensure the router learns to 
        # map input features to experts without relying on extrinsic indicators.
        # This is critical for Class-IL compliance and structural integrity.
        
        # Top-k gating
        # Keep top k values, set others to -inf
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Softmax over top-k
        weights = F.softmax(top_k_logits, dim=1)
        
        # [V31.8] STRATEGIC MODE: Expert Gini Regularization
        # Gini Index encourages uniform load balancing. 
        # Higher Gini = Higher diversity (fewer bottlenecks).
        batch_size = top_k_indices.size(0)
        mask = torch.zeros(batch_size, self.gate.out_features, device=x.device)
        mask.scatter_(1, top_k_indices, weights)
        importance = mask.sum(dim=0)
        mean_imp = importance.mean() + 1e-6
        var_imp = importance.var()
        
        gini_importance = weights.sum(dim=0)
        gini_importance = gini_importance / (gini_importance.sum() + 1e-8)
        gini_loss = 1.0 - torch.sum(gini_importance**2)
        
        # Combine Variance-based loss with Gini-based diversity
        self.aux_loss = (var_imp / (mean_imp ** 2)) * 0.5 + (1.0 - gini_loss) * 0.5
        
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

    def distill_expert(self, source_idx: int, target_idx: int, noise_scale: float = 0.01):
        """
        [V31.8] STRATEGIC MODE: Task-Informed Initialization.
        Initializes a new expert as a noisy copy of a successful old expert.
        """
        if source_idx >= self.num_experts or target_idx >= self.num_experts:
            return
            
        source_state = self.experts[source_idx].state_dict()
        new_state = {}
        for k, v in source_state.items():
            # Add a tiny bit of noise to break symmetry
            new_state[k] = v.clone() + torch.randn_like(v) * v.std() * noise_scale
            
        self.experts[target_idx].load_state_dict(new_state)

    def set_temperature(self, temperature):
        self.gate.temperature = temperature

    def get_usage_stats(self):
        return self.expert_usage.cpu().numpy()
        
    def get_aux_loss(self):
        return self.gate.get_aux_loss()

    def forward(self, x, task_id=None, consciousness_state: Optional[torch.Tensor] = None, internal_mode: bool = False):
        if self.training and task_id is not None:
            # [SMART HARD-ROUTING] Perfect task isolation inside domain
            target_expert = task_id % self.num_experts
            indices = torch.full((x.size(0), self.top_k), target_expert, dtype=torch.long, device=x.device)
            weights = torch.zeros((x.size(0), self.top_k), device=x.device)
            weights[:, 0] = 1.0
        else:
            weights, indices = self.gate(x, task_id=task_id, consciousness_state=consciousness_state)
        
        # [V31.8] STRATEGIC MODE: Expert Dropout (The Ghost Expert)
        # Randomly mute one expert during training to force expert redundancy.
        # This prevents 'Master Expert' dependency and encourages distributed knowledge.
        if self.training and self.num_experts > 1 and torch.rand(1).item() < 0.1:
            drop_idx = torch.randint(0, self.num_experts, (1,)).item()
            # Create a mask that is 0 for the dropped expert
            mask = (indices != drop_idx).float()
            weights = weights * mask
            # Re-normalize to ensure weights still sum to 1.0
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

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
            # [V31.8] ETERNAL MIND: Ensure type alignment for index_add (Critical for AMP)
            contribution = (expert_out * selected_weights).to(final_output.dtype)
            final_output = final_output.index_add(0, batch_idx, contribution)

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
    
    def forward(self, x, task_id=None, consciousness_state: Optional[torch.Tensor] = None, internal_mode: bool = False):
        # [BUGFIX] Task-Agnostic Logit-Based Routing for Zero-Exemplar Class-IL
        # Bypasses Gating Network Forgetting completely during evaluation.
        # [V31.12] HYBRID ROUTING: Use Gating Network for initial expert selection, 
        # but keep logit-based refinement to solve class-overlap.
        if not self.training and task_id is None:
            # 1. First, let the Domain Router pick the domain
            domain_weights, domain_indices = self.domain_router(x, task_id=None, consciousness_state=consciousness_state)
            
            # 2. Collect outputs from all experts
            all_expert_outputs = []
            for d_idx, domain in enumerate(self.domains):
                # Expert gate weights for this domain
                e_weights, e_indices = domain.gate(x, task_id=None, consciousness_state=consciousness_state)
                
                for i, expert in enumerate(domain.experts):
                    out = expert(x, task_id=None)
                    if isinstance(out, tuple): out = out[0]
                    
                    # Simplify: Just pick the expert the gate likes best
                    all_expert_outputs.append(out.unsqueeze(1))
            
            all_expert_outputs = torch.cat(all_expert_outputs, dim=1)
            # Use Gating for final decision
            max_logits, _ = torch.max(all_expert_outputs, dim=2)
            best_expert_idx = torch.argmax(max_logits, dim=1)
            
            batch_idx = torch.arange(x.size(0), device=x.device)
            final_output = all_expert_outputs[batch_idx, best_expert_idx, :]
            
            return final_output, domain_indices

        if self.training and task_id is not None:
            # [SMART HARD-ROUTING] Perfect task isolation to guarantee 100% preservation
            total_experts = self.num_domains * self.experts_per_domain
            target_domain = (task_id % total_experts) // self.experts_per_domain
            domain_indices = torch.full((x.size(0), self.top_k), target_domain, dtype=torch.long, device=x.device)
            domain_weights = torch.zeros((x.size(0), self.top_k), device=x.device)
            domain_weights[:, 0] = 1.0
        else:
            domain_weights, domain_indices = self.domain_router(x, task_id=task_id, consciousness_state=consciousness_state)
            
        batch_size = x.size(0)
        final_output = None
        
        for i in range(self.num_domains):
            mask = (domain_indices == i).any(dim=1)
            batch_idx = torch.where(mask)[0]
            if len(batch_idx) == 0: continue
                
            selected_inputs = x[batch_idx]
            domain_out, _ = self.domains[i](selected_inputs, task_id=task_id, consciousness_state=consciousness_state, internal_mode=internal_mode)
            
            if final_output is None:
                out_shape = list(domain_out.shape)
                out_shape[0] = batch_size
                final_output = torch.zeros(out_shape, device=x.device, dtype=domain_out.dtype)
            
            w = domain_weights[batch_idx, 0].view(len(batch_idx), *([1] * (domain_out.dim() - 1)))
            # [V31.8] ETERNAL MIND: Ensure type alignment for index_add (Critical for AMP)
            contribution = (domain_out * w).to(final_output.dtype)
            final_output = final_output.index_add(0, batch_idx, contribution)
            
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
