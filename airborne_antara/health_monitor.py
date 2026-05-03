import torch
import torch.nn as nn
import logging
from typing import Dict, Any

class NeuralHealthMonitor:
    """
    [V9.1] Autonomic Self-Repair Module.
    Monitors the 'vital signs' of the neural network:
    - Dead Neurons (zero gradients)
    - Gradient Explosions
    - Mode Collapse
    - Weight Saturation
    - [NEW] Repair Patience: Avoids reset loops during convergence noise.
    """
    def __init__(self, model: nn.Module, dead_threshold: float = 1e-12, explosion_threshold: float = 1e2):
        self.model = model
        self.dead_threshold = dead_threshold
        self.explosion_threshold = explosion_threshold
        self.logger = logging.getLogger("NeuralHealthMonitor")
        self.health_history = []

    def check_vital_signs(self) -> Dict[str, str]:
        """
        Analyzes gradients and weights to detect issues.
        Returns: Dict[parameter_name, status]
        """
        report = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    status = "IDLE"
                else:
                    grad_mean = param.grad.abs().mean().item()
                    if grad_mean < self.dead_threshold:
                        status = "DEAD"
                    elif grad_mean > self.explosion_threshold:
                        status = "CRITICAL"
                    else:
                        weight_std = param.data.std().item()
                        if weight_std < 1e-4:
                            status = "SATURATED"
                        else:
                            status = "HEALTHY"
                report[name] = status
        return report

    def autonomic_repair(self, report: Dict[str, str], projector=None, expert_usage: Dict[int, float] = None, memory=None):
        """
        [V9.3] MoE-Aware Soft Jitter Rejuvenation.
        Preserves knowledge in idle experts while fixing truly dead neurons.
        """
        repairs_made = 0
        for name, status in report.items():
            if status in ["DEAD", "SATURATED"]:
                # [V9.3] MoE Awareness: Skip if layer belongs to an idle expert
                if expert_usage is not None and "experts." in name:
                    try:
                        # Extract expert index (e.g., 'experts.0.model...')
                        parts = name.split("experts.")
                        if len(parts) > 1:
                            expert_idx = int(parts[1].split(".")[0])
                            # If expert hasn't been used in this window, skip repair
                            if expert_usage.get(expert_idx, 0) == 0:
                                continue
                    except (ValueError, IndexError):
                        pass

                # [V9.2] Neural Shivering: Small Gaussian noise to break equilibrium
                self.logger.info(f"[REPAIR] Autonomic Shiver: Rejuvenating {status} layer '{name}'")
                
                param = dict(self.model.named_parameters())[name]
                noise_scale = 1e-4 # Conservative jitter
                
                noise = torch.randn_like(param.data) * noise_scale
                if projector is not None:
                    noise = projector.project_gradient(name, noise)

                # [V17] UNYIELDING SOUL: Apply Sacred Mask protection to repairs
                mem = memory if memory is not None else getattr(self.model, 'memory', None)
                if mem and hasattr(mem, 'sacred_mask'):
                    mask = mem.sacred_mask.get(name, None)
                    if mask is not None and mask.any():
                        noise = noise * (~mask.to(noise.device))
                
                param.data.add_(noise)
                repairs_made += 1
                
            elif status == "CRITICAL":
                # Critical explosions: aggressive scaling
                self.logger.warning(f"[WARNING] Autonomic Scaler: Protecting exploding layer '{name}'")
                param = dict(self.model.named_parameters())[name]
                if param.grad is not None:
                    param.grad.data.copy_(torch.clamp(param.grad.data, -5.0, 5.0))
                repairs_made += 1
                
        return repairs_made
