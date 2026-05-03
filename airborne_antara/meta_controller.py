"""
Meta-Controller: Reptile-Based Dynamic Adaptation (Production V2)
=================================================================

Implements the "Reptile" meta-learning algorithm (OpenAI) adapted for
continuous online learning. This replaces brittle second-order MAML
with a stable "Lookahead" optimization strategy.

FEATURES:
- Reptile "Lookahead" Optimizer (Fast/Slow weight interpolation)
- Z-Score based Dynamic Learning Rate
- Automated Curriculum Difficulty scaling
- Learned Optimizer (LSTM-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
from collections import deque
import copy
from dataclasses import dataclass
import logging

# ==================== CONFIGURATION ====================

@dataclass
class MetaControllerConfig:
    """Configuration for the meta-controller"""
    # Learning rate scheduling
    base_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Gradient analysis
    gradient_clip_norm: float = 1.0
    
    # Reptile Meta-Learning (New Optimization Style)
    use_reptile: bool = True
    reptile_learning_rate: float = 0.1  # Epsilon (Interpolation rate)
    reptile_update_interval: int = 5    # k steps (Inner loop length)
    
    # Curriculum strategy
    curriculum_start_difficulty: float = 0.1
    curriculum_increase_rate: float = 0.01

    # Learned Optimizer
    use_learned_optimizer: bool = True
    learned_optimizer_hidden_dim: int = 32


# ==================== GRADIENT ANALYZER ====================

class GradientAnalyzer:
    """
    Analyzes gradient statistics to make adaptation decisions.
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger('GradientAnalyzer')
        self.gradient_history = deque(maxlen=100)
        
    def analyze(self) -> Dict[str, float]:
        """[V26.0] Vectorized gradient analysis."""
        stats = {'mean_norm': 0.0, 'max_norm': 0.0, 'variance': 0.0, 'sparsity': 0.0}
        
        # 1. Collect all valid norms in a single vectorized pass
        norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                norms.append(param.grad.detach().norm(2).view(1))
        
        if not norms:
            return stats
            
        # 2. Vectorized computation
        with torch.no_grad():
            all_norms = torch.cat(norms)
            finite_mask = torch.isfinite(all_norms)
            all_norms = all_norms[finite_mask]
            
            if all_norms.numel() == 0:
                return stats
                
            stats['mean_norm'] = all_norms.mean().item()
            stats['max_norm'] = all_norms.max().item()
            stats['variance'] = all_norms.var().item()
            stats['sparsity'] = (all_norms < 1e-6).float().mean().item()
        
        self.gradient_history.append(stats)
        return stats
    
    def get_trajectory(self) -> List[Dict[str, float]]:
        return list(self.gradient_history)


# ==================== LEARNED OPTIMIZER (LSTM) ====================

class LearnedOptimizerPolicy(nn.Module):
    """
    Neural Network that learns to control the learning rate.
    Inputs: [Loss, GradNorm, CurrentLR]
    Output: LR Multiplier
    """
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.hidden = None
        
    def forward(self, x):
        # x: [B, Seq, Dim] or [B, Dim]
        if x.dim() == 2: x = x.unsqueeze(1)
        
        out, self.hidden = self.lstm(x, self.hidden)
        # Output range: [0.5, 2.0] (Safe modifiers)
        multiplier = torch.sigmoid(self.head(out[:, -1, :])) * 1.5 + 0.5
        return multiplier

    def reset_state(self):
        self.hidden = None


# ==================== ROBUST LR SCHEDULER ====================

class DynamicLearningRateScheduler(nn.Module):
    """
    Statistically Adaptive Scheduler (No Magic Numbers).
    Uses Z-Scores to detect anomalies relative to the model's own history.
    Can optionally use a Learned Policy.
    """
    
    def __init__(self, optimizer, config):
        super().__init__()
        self.optimizer = optimizer
        self.config = config
        self.current_lr = config.base_lr
        
        # History buffers for statistical analysis
        self.loss_history = deque(maxlen=100)
        self.grad_history = deque(maxlen=100)
        
        # Learned Policy
        self.policy = None
        if config.use_learned_optimizer:
            self.policy = LearnedOptimizerPolicy(hidden_dim=config.learned_optimizer_hidden_dim)
            # Simple meta-optimizer for the policy itself
            self.policy_optim = torch.optim.SGD(self.policy.parameters(), lr=0.01)
            self.last_action = None
            self.last_state = None
        
        # Safety floor/ceiling
        self.min_lr = config.min_lr
        self.max_lr = config.max_lr

    def step(self, loss: float, gradient_stats: Dict[str, float]) -> float:
        self.loss_history.append(loss)
        grad_norm = gradient_stats.get('mean_norm', 0.0)
        self.grad_history.append(grad_norm)
        
        # Need enough data to establish a baseline
        if len(self.grad_history) < 10:
            return self.current_lr

        # --- LEARNED OPTIMIZER STEP ---
        if self.policy:
            return self._step_learned(loss, grad_norm)
            
        # --- HEURISTIC STEP (Fallback) ---
        return self._step_heuristic(loss, grad_norm)

    def _step_learned(self, loss, grad_norm):
        # 1. Prepare State
        # Normalize inputs roughly
        state = torch.tensor([[loss, grad_norm, self.current_lr]], dtype=torch.float32)
        
        # 2. Meta-Update (Train Policy)
        # If we took an action last step, did it reduce loss?
        if self.last_action is not None and len(self.loss_history) > 1:
            prev_loss = self.loss_history[-2]
            # Reward: Positive if loss decreased
            reward = (prev_loss - loss) 
            
            # Simple Policy Gradient: Maximize Reward * LogProb(Action)
            # But here output is deterministic multiplier.
            # We treat it as regression to "Optimal Multiplier".
            # Heuristic: If reward > 0, we wanted this action. If < 0, we wanted opposite.
            # [V8.1] Activated: Train the learned optimizer policy
            if abs(reward) > 1e-6 and self.last_state is not None:
                try:
                    self.policy_optim.zero_grad()
                    # Recompute action with gradients enabled
                    action = self.policy(self.last_state)
                    # Simple reward-weighted loss: push multiplier toward 1.0 if bad, keep if good
                    # If reward > 0 (loss decreased), reinforce current action
                    # If reward < 0 (loss increased), push toward 1.0 (neutral)
                    target = torch.tensor([[1.0]]) if reward < 0 else action.detach()
                    pg_loss = torch.nn.functional.mse_loss(action, target) * abs(reward)
                    pg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.policy_optim.step()
                    self.policy.reset_state()  # Reset LSTM hidden after update
                except Exception:
                    pass  # Graceful fallback if training fails
            
        # 3. Forward
        with torch.no_grad(): # Don't backprop through main loop yet
             multiplier = self.policy(state).item()
             
        self.current_lr *= multiplier
        self.current_lr = np.clip(self.current_lr, self.min_lr, self.max_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
            
        self.last_action = multiplier
        self.last_state = state
        return self.current_lr

    def _step_heuristic(self, loss, grad_norm):
        # 1. CALCULATE STATISTICS (Self-Baseline)
        grad_mean = np.mean(self.grad_history)
        grad_std = np.std(self.grad_history) + 1e-6 
        
        loss_mean = np.mean(self.loss_history)
        loss_std = np.std(self.loss_history) + 1e-6 
        
        # 2. DETECT ANOMALIES (Z-Score)
        grad_z_score = (grad_norm - grad_mean) / grad_std
        loss_z_score = (loss - loss_mean) / loss_std
        
        # 3. ADAPTIVE LOGIC (Relative, not Absolute)

        # A. SURPRISE DETECTED (Loss Spike > 1.5 Sigma) -> BOOST PLASTICITY
        if loss_z_score > 1.5:
            # Boost proportional to surprise
            boost = 1.0 + (loss_z_score * 0.1) 
            self.current_lr *= min(2.0, boost)

        # B. EXPLOSION DETECTED (Grads > 2 Sigma) -> CUT LR
        # Only brake if Gradients are high but Loss is STABLE (Numerical Instability)
        elif grad_z_score > 2.0:
            reduction = 0.5 * (1.0 / grad_z_score) 
            self.current_lr *= max(0.1, reduction)
            
        # C. STAGNATION (Low Variance) -> CONVERGENCE DECAY
        elif abs(loss_z_score) < 0.1 and abs(grad_z_score) < 0.1:
            self.current_lr *= 0.98

        # 4. Apply & Clamp
        self.current_lr = np.clip(self.current_lr, self.min_lr, self.max_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
            
        return self.current_lr
        
    def get_lr(self) -> float:
        return self.current_lr


# ==================== CURRICULUM STRATEGY ====================

class CurriculumStrategy:
    """
    Manages task difficulty with multi-modal safety checks.
    """
    
    def __init__(self, config: MetaControllerConfig):
        self.config = config
        self.current_difficulty = config.curriculum_start_difficulty
        self.logger = logging.getLogger('Curriculum')
        
    def get_difficulty(self) -> float:
        return np.clip(self.current_difficulty, 0.0, 1.0)
    
    def step(self, loss_improvement: float):
        if loss_improvement > 0.01:
            self.current_difficulty += self.config.curriculum_increase_rate
    
    def sample_task_batch(self, batch: torch.Tensor, batch_targets: torch.Tensor):
        difficulty = self.get_difficulty()
        
        # SAFETY CHECK: Only add noise to Floats (Images/Audio)
        if torch.is_floating_point(batch):
            noise_level = difficulty * 0.1
            perturbed_batch = batch + torch.randn_like(batch) * noise_level
            return perturbed_batch, batch_targets
        
        return batch, batch_targets


# ==================== REPTILE OPTIMIZER ====================

class ReptileOptimizer:
    """
    Implements the Reptile 'Lookahead' update rule.
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig):
        self.model = model
        self.config = config
        self.anchor_weights = None
        self.step_counter = 0
        self.logger = logging.getLogger('ReptileOptimizer')
        
        # [V17] Hardening: Link to memory for CAS protection
        self.memory = getattr(model, 'memory', None)

        
    def step(self):
        """
        Called every training step. Performs meta-update every k steps.
        """
        self.step_counter += 1
        
        # 1. Initialization: Set Anchor (Slow Weights)
        if self.anchor_weights is None:
            self.anchor_weights = self._clone_weights()
            return
            
        # 2. Check interval
        if self.step_counter % self.config.reptile_update_interval == 0:
            self._perform_update()

    def to(self, device):
        """[V26.4] Device Affinity: Move anchor weights."""
        if self.anchor_weights:
            self.anchor_weights = {k: v.to(device) for k, v in self.anchor_weights.items()}
        return self
            
    def _clone_weights(self) -> Dict[str, torch.Tensor]:
        """Deep copy current model weights to CPU."""
        """Deep copy current model weights."""
        target_model = self.model
        if hasattr(self.model, '_orig_mod'):
            target_model = self.model._orig_mod
            
        return {
            k: v.clone().detach() 
            for k, v in target_model.state_dict().items()
        }
        
    def _perform_update(self):
        """Performs the Reptile interpolation"""
        target_model = self.model
        if hasattr(self.model, '_orig_mod'):
            target_model = self.model._orig_mod

        current_weights = target_model.state_dict()
        new_state_dict = {}
        
        epsilon = self.config.reptile_learning_rate
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.anchor_weights:
                    anchor = self.anchor_weights[name]
                    # Interpolated target
                    target = anchor + epsilon * (param.data - anchor)
                    
                    # [V26.3] Pure Device-Local Sacred Mask protection
                    if self.memory and hasattr(self.memory, 'sacred_mask'):
                        mask = self.memory.sacred_mask.get(name, None)
                        if mask is not None and mask.any():
                            # [V31.6] META-LEAKAGE: Allow meta-learning to refine sacred weights (Positive BWT)
                            # Instead of a hard-lock (0%), we allow a 5% dampened update.
                            # This enables the backbone to stay aligned with previous tasks as it evolves.
                            target_sacred = anchor + (epsilon * 0.05) * (param.data - anchor)
                            param.data.copy_(torch.where(mask.to(param.device), target_sacred.to(param.device), target.to(param.device)))
                            continue
                            
                    param.data.copy_(target)
            
            # Update Buffers (Running Mean/Var) - handled via state_dict load if needed,
            # but usually Reptile keeps buffers from fast path or interpolates.
            # Here we let buffers stick to the Fast Weights by NOT forcing anchor buffers back
            # unless we explicitly loaded a state_dict. 
            # The loop above only updates `named_parameters`. 
            # This is standard Reptile behavior: Weights interpolate, Buffers track latest.
        
        # Update Anchor for next cycle
        self.anchor_weights = self._clone_weights()


# ==================== META-CONTROLLER ====================

class MetaController(nn.Module):
    """
    Orchestrates the optimization cycle using Reptile logic.
    """
    
    def __init__(self, 
                 framework: Any, 
                 config: Optional[MetaControllerConfig] = None):
        super().__init__()
        if config is None:
            config = MetaControllerConfig()
        
        # Removed self.framework = framework to prevent RecursionError in state_dict (circular ref)
        self.config = config
        
        # Components
        self.gradient_analyzer = GradientAnalyzer(framework.model, config)
        self.lr_scheduler = DynamicLearningRateScheduler(framework.optimizer, config)
        self.curriculum = CurriculumStrategy(config)
        self.reptile = ReptileOptimizer(framework.model, config)
        
        self.current_mode = 'NORMAL'
        self.step_count = 0
        
    def adapt(self,
              loss: float,
              gradients: Optional[Dict[str, torch.Tensor]] = None,
              performance_metrics: Optional[Dict[str, float]] = None,
              external_grad_stats: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        
        metrics = {}
        
        # 1. Analyze Gradients
        if external_grad_stats is not None:
             grad_stats = external_grad_stats
        else:
             grad_stats = self.gradient_analyzer.analyze()
             
        metrics['gradient_stats'] = grad_stats
        
        # 2. Schedule Learning Rate
        new_lr = self.lr_scheduler.step(loss, grad_stats)
        metrics['learning_rate'] = new_lr
        
        # 3. Update Curriculum
        if performance_metrics:
            loss_imp = performance_metrics.get('loss_improvement', 0.0)
            self.curriculum.step(loss_imp)
            metrics['curriculum_difficulty'] = self.curriculum.get_difficulty()
            
        # 4. REPTILE META-UPDATE
        if self.config.use_reptile:
            self.reptile.step()
        
        self.step_count += 1
        return metrics

    def to(self, device):
        """[V26.4] Device Affinity: Propagate to sub-components."""
        super().to(device)
        if hasattr(self, 'reptile'):
            self.reptile.to(device)
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'current_lr': self.lr_scheduler.get_lr(),
            'reptile_active': self.config.use_reptile
        }