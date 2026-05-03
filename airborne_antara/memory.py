import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import gc
from typing import Dict, List, Any, Optional

class MemoryManager:
    """
    [V26] Unified Memory System (hybrid EWC/SI/OGD).
    [V30.11] Hardened for 8GB VRAM (Half-Precision CPU Storage).
    """
    def __init__(self, config: Any, models: List[nn.Module]):
        self.config = config
        self.models = models
        self.method = config.memory_method
        self.device = torch.device(config.device)
        self.logger = logging.getLogger('MemoryManager')
        
        # 1. Importance Matrices (Stored on CPU to save VRAM)
        self.omega = {}       # SI weights
        self.fisher_dict = {} # EWC weights
        self.anchor = {}      # SI anchors
        self.opt_param_dict = {} # EWC anchors
        
        # 2. Path Integrals (For SI)
        self.w = {n: torch.zeros_like(p).cpu() for m in models for n, p in m.named_parameters() if p.requires_grad}
        
        # 3. Governance
        self.sacred_mask = {} 
        self.saturation_level = 0.0
        
        # [V26.1] OGD Projector
        self.projector = None
        if self.method == 'ogd':
            from .ogd import OGDProjector
            self.projector = OGDProjector(config)

        self.si_lambda = config.si_lambda
        self.ewc_lambda = config.ewc_lambda
        
    def is_enabled(self) -> bool:
        return self.method != 'none'

    def before_step_snapshot(self) -> Dict[str, torch.Tensor]:
        """Capture parameters before optimizer.step() for SI accumulation."""
        if self.method not in ['si', 'hybrid']:
            return {}
        results = {}
        for model in self.models:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    results[n] = p.data.clone().detach()
        return results
    
    def accumulate_path(self, param_before: Dict[str, torch.Tensor]) -> None:
        """SI path-integral accumulation: s_i += -g_i * delta_theta_i"""
        if self.method not in ['si', 'hybrid'] or not param_before:
            return
        
        try:
            with torch.no_grad():
                for model in self.models:
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in param_before and p.grad is not None:
                            # Update path integral: delta_w = -grad * delta_param
                            delta = (p.detach() - param_before[n].to(p.device))
                            self.w[n] += (-p.grad.detach().cpu() * delta.cpu())
        except Exception as e:
            self.logger.error(f"SI Accumulation error: {e}")

    def consolidate(self, task_id: int, feedback_buffer: Any):
        """Finalize importance metrics for the current task."""
        if not self.is_enabled():
            return

        self.logger.info(f"Consolidating Memory for Task {task_id}...")
        
        # 1. Update SI (Omega)
        if self.method in ['si', 'hybrid']:
            self._consolidate_si_omega()
            
        # 2. Update EWC (Fisher)
        if self.method in ['ewc', 'hybrid']:
            self._consolidate_ewc_fisher_vectorized(feedback_buffer)
            
        # 3. Update OGD
        if self.method == 'ogd':
            self._consolidate_ogd(feedback_buffer)

        # 4. Final Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _consolidate_si_omega(self, epsilon: float = 1e-3):
        """Finalize SI Omega importance scores."""
        for model in self.models:
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.w:
                    # delta_theta^2
                    delta_theta = (p.detach().cpu() - self.anchor.get(n, p.detach().cpu()))
                    # Omega = w / (delta_theta^2 + epsilon)
                    new_omega = self.w[n] / (delta_theta**2 + epsilon)
                    
                    if n not in self.omega:
                        self.omega[n] = new_omega.half().cpu()
                    else:
                        # EMA update for Omega
                        self.omega[n] = (self.omega[n] * 0.7 + new_omega.half().cpu() * 0.3).half()
                    
                    # Reset path integral and anchor
                    self.w[n].zero_()
                    self.anchor[n] = p.detach().clone().half().cpu()

    def update_saturation(self):
        """Calculates architectural saturation based on sacred mask."""
        if not self.sacred_mask:
            self.saturation_level = 0.0
            return
            
        with torch.no_grad():
            num_total = 0
            total_sacred = 0
            for model in self.models:
                for name, p in model.named_parameters():
                    if not p.requires_grad: continue
                    num_total += p.numel()
                    if name in self.sacred_mask:
                        total_sacred += self.sacred_mask[name].sum().item()
                
                self.saturation_level = total_sacred / num_total

    def _consolidate_ewc_fisher_vectorized(self, feedback_buffer, sample_limit: int = 128, batch_size: int = 8):
        """
        Vectorized Fisher computation. 
        [V31.1] ROLLING ANCHORS: Anchor to the most recent task state.
        """
        if not feedback_buffer.buffer:
            return
            
        # 1. Update Anchors (Standard EWC: Anchor to state after Task N-1)
        for model in self.models:
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                # [V31.1] Rolling Anchor: plasticity is prioritized
                self.opt_param_dict[n] = p.data.clone().detach().half().cpu()
        
        # 2. Compute Fisher
        fisher = {n: torch.zeros_like(p).float().cpu() for m in self.models for n, p in m.named_parameters() if p.requires_grad}
        samples = list(feedback_buffer.buffer)[-sample_limit:]
        
        orig_training = {m: m.training for m in self.models}
        for m in self.models: m.eval()
        
        try:
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]
                if not batch_samples: continue
                
                batch_x = torch.stack([s.x for s in batch_samples]).to(self.device)
                
                for model in self.models:
                    model.zero_grad()
                    outputs = model(batch_x)
                    
                    # Multi-Task Fisher: Average over likely classes
                    log_probs = F.log_softmax(outputs, dim=1)
                    # Use sample weights if available
                    for b in range(len(batch_samples)):
                        label = outputs[b].argmax().item()
                        loss = log_probs[b, label]
                        loss.backward(retain_graph=True if b < len(batch_samples)-1 else False)
                    
                    # Accumulate Fisher
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            fisher[n] += (p.grad.data.detach().cpu()**2) / len(samples)
        finally:
            for m, mode in orig_training.items(): m.train(mode)
            
        # [V30.11] EMA Update with explicit Half casting to prevent VRAM leak
        for n in fisher:
            f_val = fisher[n].clamp(min=1e-8, max=60000.0).half()
            if n not in self.fisher_dict:
                self.fisher_dict[n] = f_val
            else:
                curr = self.fisher_dict[n].half()
                self.fisher_dict[n] = ((curr * 0.9) + (f_val * 0.1)).half()
        
        self.logger.info("   [EWC] Rolling Fisher Matrix stabilized.")
        gc.collect()

    def compute_penalty(self, adaptive_mode: str = 'NORMAL', step_in_mode: int = 0) -> torch.Tensor:
        """Compute total regularization loss."""
        if not self.is_enabled():
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        # [V31.1] Novelty Release: If in NOVELTY mode, significantly lower constraints
        base = {'BOOTSTRAP': 0.0, 'PANIC': 0.0, 'SURVIVAL': 0.1, 'NOVELTY': 0.2, 'NORMAL': 0.4}.get(adaptive_mode, 0.4)
        decay = np.exp(-0.01 * step_in_mode)
        lamb = base * decay
        
        if lamb < 1e-4: return torch.tensor(0.0, device=self.device)

        # SI Penalty
        if self.method in ['si', 'hybrid']:
            si_loss = 0.0
            for model in self.models:
                for name, p in model.named_parameters():
                    if name in self.omega:
                        o_dev = self.omega[name].to(p.device)
                        a_dev = self.anchor.get(name, p).to(p.device)
                        si_loss += (o_dev * (p - a_dev).pow(2)).sum()
            loss += (self.si_lambda * lamb * si_loss)

        # EWC Penalty
        if self.method in ['ewc', 'hybrid']:
            ewc_total = 0.0
            for model in self.models:
                for name, p in model.named_parameters():
                    if name in self.fisher_dict:
                        fisher = self.fisher_dict[name].to(p.device)
                        anchor = self.opt_param_dict.get(name)
                        if anchor is not None:
                            anchor = anchor.to(p.device)
                            # [V31.2] Shape Resilience Guard
                            if p.shape == anchor.shape and p.shape == fisher.shape:
                                ewc_total += (fisher * (p - anchor).pow(2)).sum()
            loss += (self.ewc_lambda * lamb * ewc_total)

        return loss