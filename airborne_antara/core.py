"""
Core Adaptive Meta-Learning Framework (Universal v1.1.1 - "Sentient" Edition)
=============================================================================
The Universal Wrapper that turns ANY PyTorch model into a Self-Learning System.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import random
from collections import deque
from pathlib import Path
import logging
import sys
import os
import platform
import shutil
from datetime import datetime
import time
import math
from enum import Enum
import json

# Import Unified Memory, Meta-Controller, and Consciousness
# NOTE: We use .consciousness_v2 as requested for the SOTA module
from .memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, AdaptiveRegularization, DynamicConsolidationScheduler
from .meta_controller import MetaController, MetaControllerConfig
from .consciousness_v2 import ConsciousnessCore
from .adapters import AdapterBank
from .moe import SparseMoE
from .perception import PerceptionGateway
from .world_model import WorldModel
from .governance import KnowledgeGovernor

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the Universal Framework (V8.0).
    """
    # Architecture
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # Learning parameters
    learning_rate: float = 2e-3
    meta_learning_rate: float = 1e-4
    
    # Plasticity: How much the model can 'edit' itself directly
    weight_adaptation_lr: float = 1e-5 
    bias_adaptation_lr: float = 1e-5
    adaptation_threshold: float = 0.05
    
    # Introspection
    telemetry_dim: int = 4 
    feedback_buffer_size: int = 10000
    evaluation_frequency: int = 10
    # How often to run dreaming/replay (in steps).
    dream_interval: int = 10 
    dream_batch_size: int = 0 # [V26.5] Zero-Exemplar Protocol
    
    # Optimization
    compile_model: bool = True 
    use_amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_frequency: int = 50
    checkpoint_frequency: int = 500
    gradient_clip_norm: float = 1.0
    adapter_max_norm: float = 2.0
    
    # --- HIERARCHICAL REFLEX ---
    enable_active_shield: bool = True 
    active_shield_threshold: float = 0.05 
    active_shield_slope: float = 10.0   
    panic_threshold: float = 0.2
    warmup_steps: int = 50
    
    # Z-Score Thresholds
    novelty_z_threshold: float = 2.0
    survival_z_threshold: float = 4.0
    enable_dreaming: bool = False # [V26.5] Zero-Exemplar Protocol
    enable_tracing: bool = False
    trace_max_records: int = 1000
    
    # SOTA Unified Memory System (V7.0)
    memory_type: str = 'hybrid'  # 'ewc', 'si', or 'hybrid'
    consolidation_criterion: str = 'hybrid'
    consolidation_min_interval: int = 30
    consolidation_max_interval: int = 100
    consolidation_surprise_threshold: float = 2.5
    adaptive_lambda: bool = True
    use_prioritized_replay: bool = False # [V26.5] Zero-Exemplar Protocol
    replay_priority_temperature: float = 0.6
    
    # --- V7.0: CONSCIOUSNESS LAYER ---
    enable_consciousness: bool = True # [NeurIPS-Best] Enabled for superior task-aware adaptation
    use_attention: bool = True
    use_intrinsic_motivation: bool = False 
    consciousness_buffer_size: int = 5000
    novelty_threshold: float = 2.0
    
    # SI Parameters (Restored)
    importance_method: str = 'ewc'  # 'ewc', 'si', or 'hybrid'
    si_lambda: float = 1.5 
    ewc_lambda: float = 0.0 
    si_xi: float = 1e-3
    use_graph_memory: bool = False 
    graph_memory_threshold: float = 0.85
    use_ogd: bool = False 
    ogd_max_basis_size: int = 1024
    enable_holographic_compression: bool = True 

    # [V15] IRON MIND PROTOCOL
    use_iron_mind: bool = True
    iron_mind_quota: float = 0.25
    use_elastic_quota: bool = True # [V9.5] ENA: Elastic Neural Allocation

    # [V8.0] Optimization
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    use_gradient_centralization: bool = True

    # --- V7.1: CORTEX ENGINE (MoE) ---
    use_moe: bool = True
    use_hierarchical_moe: bool = True
    num_experts: int = 10
    top_k_experts: int = 2
    num_domains: int = 2
    experts_per_domain: int = 4
    input_dim: int = 3072 
    classes_per_task: int = 10 # [V28] Dynamic task density
    moe_temperature: float = 1.0
    moe_temp_decay: float = 0.90

    # Meta-Controller / Reptile Configuration
    use_reptile: bool = True
    reptile_learning_rate: float = 0.1
    reptile_update_interval: int = 5
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    curriculum_start_difficulty: float = 0.1
    curriculum_increase_rate: float = 0.01
    use_learned_optimizer: bool = False
    learned_optimizer_hidden_dim: int = 32

    # --- V8.0: PERCEPTION INTERFACE ---
    enable_perception: bool = False
    vision_dim: int = 3 # Channels
    audio_dim: int = 80 # Mel bins
    text_dim: int = 0   # Optional projection
    perception_layers: int = 2
    perception_heads: int = 4
    
    # --- V9.0: SYNTHETIC INTUITION ---
    enable_world_model: bool = True # [NeurIPS-Best] Enabled for predictive stability
    world_model_loss_weight: float = 0.1
    world_model_plasticity_gamma: float = 1.0 
    enable_health_monitor: bool = False # [CLEAN STREAM] Disabled to prevent mask overrides
    health_check_interval: int = 100 
    enable_performance_monitor: bool = False  

    @classmethod
    def production(cls):
        return cls(
            model_dim=512, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=torch.cuda.is_available(),
            compile_model=True,
            memory_type='hybrid',
            use_prioritized_replay=True,
            adaptive_lambda=True,
            enable_consciousness=True,
            ewc_lambda=150.0,
            dream_interval=2,
            dream_batch_size=32
        )


# ==================== DATA STRUCTURES ====================

@dataclass
class PerformanceSnapshot:
    """Standard container for experience replay"""
    input_args: tuple
    input_kwargs: dict
    output: torch.Tensor
    target: torch.Tensor
    reward: float
    loss: float
    timestamp: float
    episode: int
    task_id: int = -1  # [V17] Track which task produced this experience
    
    def to_device(self, device):
        def _to_device(x):
            if isinstance(x, torch.Tensor): return x.to(device)
            if isinstance(x, dict): return {k: _to_device(v) for k, v in x.items()}
            if isinstance(x, list): return [_to_device(v) for v in x]
            return x

        self.input_args = tuple(_to_device(arg) for arg in self.input_args)
        self.input_kwargs = {k: _to_device(v) for k, v in self.input_kwargs.items()}
        self.output = self.output.to(device)
        self.target = self.target.to(device)
        return self


# ==================== UNIVERSAL COMPONENTS ====================

class FeedbackBuffer:
    """Robust Experience Replay Buffer using Reservoir Sampling."""
    def __init__(self, config: AdaptiveFrameworkConfig, device):
        self.capacity = config.feedback_buffer_size
        self.device = device
        self.buffer: List[PerformanceSnapshot] = []
        self.total_seen = 0
        
    def add(self, input_args: tuple, input_kwargs: dict, output: torch.Tensor, target: torch.Tensor, reward: float, loss: float, task_id: int = -1):
        # Move to CPU immediately to save VRAM
        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                # [V9.2] Holographic Saliency Pooling for "Bigger Images"
                # If tensor is a high-res image (4D), downsample to prevent CPU RAM exhaustion
                if x.dim() == 4 and x.size(2) > 128:
                    # Adaptive pooling to 128x128 for memory safety
                    x = F.adaptive_avg_pool2d(x, (128, 128))
                return x.detach().cpu()
            if isinstance(x, dict): return {k: _to_cpu(v) for k, v in x.items()}
            if isinstance(x, list): return [_to_cpu(v) for v in x]
            return x

        snapshot = PerformanceSnapshot(
            input_args=tuple(_to_cpu(arg) for arg in input_args),
            input_kwargs={k: _to_cpu(v) for k, v in input_kwargs.items()},
            output=_to_cpu(output),
            target=_to_cpu(target),
            reward=reward,
            loss=loss,
            timestamp=datetime.now().timestamp(),
            episode=self.total_seen,
            task_id=task_id
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(snapshot)
        else:
            replace_idx = random.randint(0, self.total_seen)
            if replace_idx < self.capacity:
                old_snapshot = self.buffer[replace_idx]
                self.buffer[replace_idx] = snapshot
                del old_snapshot # Explicitly release memory
        self.total_seen += 1


class IntrospectionEngine(nn.Module):
    """
    The 'Meta-Brain' (Policy Network).
    Outputs a DISTRIBUTION of Affine Modifiers to enable REINFORCE training.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        
        # 1. State Monitor (Consciousness/Uncertainty)
        self.state_monitor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Output: Log Variance
        )
        
        # 2. Hyper-Policy (Outputs Mu and Sigma for Modifiers)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4) 
        )
        
        # [V9.2 STABILIZATION] Zero-initialize the policy head
        # This ensures sentient modifiers start at Identity (Scale=1, Shift=0)
        # preventing signal destruction during early backbone learning.
        nn.init.zeros_(self.policy_net[-1].weight)
        nn.init.zeros_(self.policy_net[-1].bias)
        
    def forward(self, global_state):
        log_var = torch.tanh(self.state_monitor(global_state))
        policy_out = self.policy_net(global_state)
        
        # Guard against NaNs
        policy_out = torch.nan_to_num(policy_out, nan=0.0, posinf=10.0, neginf=-10.0)

        # Split into Mu and Log-Sigma
        try:
            mu, log_sigma = policy_out.chunk(2, dim=-1)
        except Exception:
            mu = torch.zeros(1, 2, device=global_state.device)
            log_sigma = torch.zeros(1, 2, device=global_state.device)

        # Clamp log_sigma
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=5.0)
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)

        try:
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        except Exception:
            action = torch.zeros_like(mu)
            log_prob = torch.zeros(mu.size(0), device=mu.device)

        return log_var, action, log_prob


class PerformanceMonitor:
    """
    The 'Cortex' that governs adaptation via direct weight editing.
    """
    def __init__(self, model: nn.Module, config: AdaptiveFrameworkConfig, device):
        self.model = model
        self.config = config
        self.device = device


    def adapt_weights(self, 
                      current_loss: float, 
                      previous_loss: float,
                      activations: Dict[str, Any]) -> float:
        
        affine_modifiers = activations.get('affine_modifiers', None)
        telemetry_buffer = activations.get('telemetry_buffer', None) 
        layer_map = activations.get('layer_map', {}) 
        
        if affine_modifiers is None: return 0.0
        
        if affine_modifiers.ndim > 1: affine_modifiers = affine_modifiers.mean(dim=0)
        raw_scale = affine_modifiers[0].item()
        raw_shift = affine_modifiers[1].item()

        if abs(raw_scale) < 1e-4 and abs(raw_shift) < 1e-4:
            return 0.0


        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_importance = 0.1
                    
                    # Find layer index
                    for layer_name, idx in layer_map.items():
                        if layer_name in name and telemetry_buffer is not None:
                            stats = telemetry_buffer[idx]
                            mean_act = stats[0].abs()
                            var_act = stats[1]
                            param_importance = (mean_act * var_act).item()
                            break
                    
                    # Apply updates
                    scale_factor = raw_scale * self.config.weight_adaptation_lr * param_importance
                    shift_factor = raw_shift * self.config.weight_adaptation_lr * param_importance
                    
                    # [V17] UNYIELDING SOUL: Apply Sacred Mask protection to direct adaptation
                    memory = getattr(self.model, 'memory', None)
                    mask = None
                    if memory and hasattr(memory, 'sacred_mask'):
                        mask = memory.sacred_mask.get(name, None)
                        if mask is not None:
                            mask = mask.to(param.device)

                    if param.ndim == 1:
                        if mask is not None:
                            param.data.mul_(1.0 + scale_factor * (~mask))
                            param.data.add_(shift_factor * (~mask))
                        else:
                            param.mul_(1.0 + scale_factor)
                            param.add_(shift_factor)
                    elif param.ndim >= 2:
                        if mask is not None:
                            param.data.mul_(1.0 + scale_factor * (~mask))
                        else:
                            param.mul_(1.0 + scale_factor)

        return abs(raw_scale) + abs(raw_shift)


# ==================== UNIVERSAL FRAMEWORK ====================

class CognitiveRegime(Enum):
    """[V9.4] The Cognitive State of the model."""
    SCRATCH = "scratch"         # Random weights, no experience
    TRANSFER = "transfer"       # Pretrained weights, no experience
    CONTINUOUS = "continuous"   # Pretrained weights, existing memory
    GHOST = "ghost"             # Random weights, existing memory (Distillation/Ghost state)

class AdaptiveFramework(nn.Module):
    """
    The Universal Wrapper (V8.0).
    Pass ANY PyTorch model here, and it becomes self-learning.
    """
    
    def __init__(self, user_model: nn.Module, config: AdaptiveFrameworkConfig = None, device=None):
        super().__init__()
        
        if config is None: config = AdaptiveFrameworkConfig()
        if device is None: device = torch.device(config.device)
             
        self.config = config
        self.device = device
        self.logger = self._setup_logging()
        
        # [V8.3] Maintenance Mode Flag
        self._internal_consolidation_mode = False

        # [V9.2] Mixed Precision Support for scaling to high resolutions
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_amp and self.device.type == 'cuda')
        
        # 1. The "Body" (Base Model)
        self.model = user_model.to(self.device)
        
        # 4. Perception Gateway (V8.0)
        self.perception = None
        if self.config.enable_perception:
            self.perception = PerceptionGateway(self.config)
            self.logger.info("Perception Interface Enabled")
        
        if getattr(config, 'use_moe', False):
            from .moe import SparseMoE
            moe_input_dim = config.input_dim if config.input_dim > 0 else config.model_dim
            if getattr(config, 'use_hierarchical_moe', False):
                from .moe import HierarchicalMoE
                self.logger.info("Transforming Cortex into Hierarchical MoE...")
                self.model = HierarchicalMoE(
                    base_model=self.model,
                    input_dim=moe_input_dim,
                    num_domains=config.num_domains,
                    experts_per_domain=config.experts_per_domain,
                    top_k=config.top_k_experts,
                    temperature=config.moe_temperature
                ).to(self.device)
                
                # [V26.5] Critical: Force recursive device sync after deepcopying experts
                for p in self.model.parameters():
                    p.data = p.data.to(self.device)
                for b in self.model.buffers():
                    b.data = b.data.to(self.device)
            else:
                self.logger.info("Transforming Cortex into Sparse MoE...")
                self.model = SparseMoE(
                    base_model=self.model,
                    input_dim=moe_input_dim,
                    num_experts=config.num_experts,
                    top_k=config.top_k_experts,
                    temperature=config.moe_temperature
                ).to(self.device)
            self.logger.info("   [OK] Transformation Complete. The Mind is now distributed.")
        
        # [V9.0] World Model (I-JEPA) - Init BEFORE Memory to allow tracking
        self.world_model = None
        if getattr(self.config, 'enable_world_model', False):
            self.world_model = WorldModel(self.config).to(self.device)
            self._last_z_pred = None
            self.logger.info("[SENSORY] World Model (I-JEPA) Enabled")

        # [V8.0] Introspection Engine (System 2 Policy)
        self.introspection_engine = IntrospectionEngine(
            input_dim=config.telemetry_dim, 
            hidden_dim=config.model_dim // 4
        ).to(self.device)

        # 5. Memory System (Unified Handler V9.4)
        # [V15.1] FULL SPECTRUM PROTECTION
        # Now that backward() is unified, we can safely track ALL modules 
        # to ensure the Perception and Introspection policies are also preserved.
        tracked_models = [self.model]
        if self.world_model:
            tracked_models.append(self.world_model)
        if self.introspection_engine:
            tracked_models.append(self.introspection_engine)
        if self.perception:
            tracked_models.append(self.perception)

        self.memory = UnifiedMemoryHandler(
            models=tracked_models,
            method=getattr(config, 'memory_type', 'hybrid'),
            si_lambda=getattr(config, 'si_lambda', 1.0),
            si_xi=getattr(config, 'si_xi', 1e-3),
            ewc_lambda=getattr(config, 'ewc_lambda', 0.4),
            consolidation_criterion=getattr(config, 'consolidation_criterion', 'hybrid'),
            use_graph_memory=getattr(config, 'use_graph_memory', False),
            use_ogd=getattr(config, 'use_ogd', False),
            ogd_max_basis_size=getattr(config, 'ogd_max_basis_size', 1024),
            graph_threshold=getattr(config, 'graph_memory_threshold', 0.85),
            feature_dim=config.model_dim
        )
        self.logger.info(
            f"[BRAIN] Unified Memory System Online "
            f"({getattr(config, 'memory_type', 'hybrid')}, Tracking {len(tracked_models)} Models)"
        )
        
        # [V15] Governance Engine
        if getattr(config, 'use_iron_mind', True):
            self.governor = KnowledgeGovernor(
                quota=getattr(config, 'iron_mind_quota', 0.15), 
                device=self.device
            )
            # Attach Governor directly to Memory
            self.memory.governor = self.governor
            # Disable dynamic health/consolidation to prevent mask overrides
            self.config.enable_health_monitor = False
            self.logger.info(f"Iron Mind Active. Absolute {self.config.iron_mind_quota*100}% Mathematical Quota Set.")
        else:
            self.governor = None
        
        # 6. Experience Replay
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        if getattr(config, 'use_prioritized_replay', True):
            self.prioritized_buffer = PrioritizedReplayBuffer(
                capacity=config.feedback_buffer_size,
                temperature=getattr(config, 'replay_priority_temperature', 0.6)
            )
            # FIX: Link framework buffer to memory handler so train_step can save data
            self.memory.replay_buffer = self.prioritized_buffer
        else:
            self.prioritized_buffer = None
        
        # 7. Adaptive Regularization & Consolidation
        self.adaptive_reg = AdaptiveRegularization(base_lambda=0.4)
        self.consolidation_scheduler = DynamicConsolidationScheduler(
            min_interval=getattr(config, 'consolidation_min_interval', 30),
            max_interval=getattr(config, 'consolidation_max_interval', 100)
        )
        
        if self.governor is not None:
            # Iron Mind requires manual task boundaries, not dynamic steps
            self.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External Control")
        
        # 8. Consciousness Layer
        if getattr(config, 'enable_consciousness', False):
            self.consciousness = ConsciousnessCore(
                feature_dim=config.model_dim,
                num_heads=getattr(config, 'num_heads', 4),
                awareness_buffer_size=getattr(config, 'consciousness_buffer_size', 5000),
                novelty_threshold=getattr(config, 'novelty_threshold', 2.0)
            )
            self.logger.info("[CONSCIOUSNESS] Self-Awareness Module Active")
        else:
            self.consciousness = None
        
        self.current_modifiers = None
        self.meta_log_probs = []
        self.loss_history = []
        self.reward_baseline = 0.0
        self.alpha = 0.1
        self.step_count = 0
        self._cached_sacred_params = [] # [V26.0] Optimization: Cached references to sacred params

        # 4. Initialize Adapters & Hooks (Must run BEFORE optimizer creation)
        self._init_adapters_and_hooks()
        
        # 9. Optimizers
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Adapter Optimizer (CRITICAL FIX: Now sees parameters because _init_adapters_and_hooks ran first)
        if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
            adapter_params = list(self.adapter_bank.parameters())
            if adapter_params:
                self.adapter_optimizer = AdamW(adapter_params, lr=config.weight_adaptation_lr)
                self.logger.info(f"[ADAPTER] Optimizer attached to {len(adapter_params)//4} adapters.")
            else:
                self.adapter_optimizer = None
        else:
            self.adapter_optimizer = None

        self.meta_optimizer = AdamW(self.introspection_engine.parameters(), 
                                   lr=config.meta_learning_rate,
                                   weight_decay=1e-2)

        # [V9.0] World Model Optimizer
        if self.world_model:
            self.world_model_optimizer = AdamW(self.world_model.parameters(), lr=config.learning_rate)
            
        # [V9.0] Neural Health Monitor
        self.health_monitor = None
        if self.config.enable_health_monitor:
            from .health_monitor import NeuralHealthMonitor
            self.health_monitor = NeuralHealthMonitor(self.model)
            self.logger.info("[AUTONOMIC] Neural Health Monitor Active")

        # [V8.1] Performance Monitor for direct weight adaptation
        self.performance_monitor = None
        if getattr(self.config, 'enable_performance_monitor', False):
            self.performance_monitor = PerformanceMonitor(self.model, self.config, self.device)
            self.logger.info("[CORTEX] Performance Monitor Active (Direct Weight Editing)")

        # Meta-Controller (Reptile)
        self.meta_controller = MetaController(self, MetaControllerConfig(
            use_reptile=config.use_reptile,
            reptile_learning_rate=config.reptile_learning_rate,
            reptile_update_interval=config.reptile_update_interval,
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            curriculum_start_difficulty=config.curriculum_start_difficulty,
            curriculum_increase_rate=config.curriculum_increase_rate,
            use_learned_optimizer=config.use_learned_optimizer,
            learned_optimizer_hidden_dim=config.learned_optimizer_hidden_dim
        ))
        
        # Compilation
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                if platform.system() != 'Windows': # Compilation often fails on Windows
                    self.logger.info("Compiling model for speed...")
                    self.model = torch.compile(self.model)
            except Exception as e:
                self.logger.warning(f"Compilation failed: {e}")

        # [V8.0] Optimization: Lookahead Wrapper
        if self.config.use_lookahead:
            # Simple Lookahead implementation wrapper
            self.lookahead_k = getattr(config, 'lookahead_k', 5)
            self.lookahead_alpha = getattr(config, 'lookahead_alpha', 0.5)
            self.lookahead_step = 0
            self.slow_weights = {n: p.data.clone().detach() 
                                 for n, p in self.model.named_parameters() 
                                 if p.requires_grad}

        # [V9.4] Autonomous Regime Awareness
        self._perform_self_assessment()

        # [V9.4] CAS Protocol: Apply Gradient Shunting Hooks
        self.apply_cas_protection()

        # [V26.1] Titan Soul: Strict Device Affinity
        self.to(self.device)
        self.logger.info(f"[TITAN] Cognitive Device Sync: {self.device}")

        self.logger.info("Airborne-Antara Framework Initialized (V9.4 Eternal Edition)")

    def to(self, device=None, *args, **kwargs):
        """Override to ensure internal caches and weights are rebuilt on the new device."""
        if device is not None:
            self.device = torch.device(device)
        super().to(device, *args, **kwargs)
        
        # [V26.4] Device Affinity: Move slow weights
        if hasattr(self, 'slow_weights') and self.slow_weights:
            self.slow_weights = {k: v.to(self.device) for k, v in self.slow_weights.items()}
            
        # Rebuild caches after migration
        self._rebuild_restoration_cache()
        return self

    def apply_cas_protection(self):
        """
        [V9.4] Attaches gradient shunting hooks to all 'Sacred' parameters in all models.
        Ensures BWT 0 by zeroing gradients at protected coordinates.
        """
        self.cas_hooks = []
        # Support for multi-model protection (Backbone + World Model)
        models_to_protect = [self.model]
        if self.world_model:
            models_to_protect.append(self.world_model)
            
        for model in models_to_protect:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    def get_hook(p_name):
                        def hook(grad):
                            if self.memory and hasattr(self.memory, 'sacred_mask'):
                                mask = self.memory.sacred_mask.get(p_name, None)
                                if mask is not None:
                                    # Optimization: Skip if mask is all False
                                    if not mask.any():
                                        return grad
                                    # Shunt gradients
                                    return grad * (~mask.to(grad.device))
                            return grad
                        return hook
                    
                    h = param.register_hook(get_hook(name))
                    self.cas_hooks.append(h)
        
        self.logger.info(f"[CAS] Protection Active: {len(self.cas_hooks)} Gradient Shunts installed across {len(models_to_protect)} models.")

    def sync_lookahead_weights(self):
        """[V17] Resets Lookahead slow_weights to current model state.
        Call after consolidation to prevent stale slow weights from overwriting sacred coordinates.
        """
        if hasattr(self, 'slow_weights') and self.config.use_lookahead:
            self.slow_weights = {
                n: p.data.clone().detach()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }
            self.logger.info("[CORTEX] Lookahead weights synchronized with anchored state.")

    def on_task_complete(self, task_id: int):
        """[V9.4] Handles native framework logic for task boundaries."""
        print(f"\n[ANTARA] Task {task_id} complete. Anchoring Knowledge...")
        # 1. Consolidate Memory (EWC/SI)
        self.memory.consolidate(task_id=task_id, feedback_buffer=self.feedback_buffer)
        
        # 2. Update Governance (Iron Mind Quota)
        if self.governor:
            self.governor.update_sacred_mask(self.memory, task_id, self.model)
            # [V26.5] Rebuild restoration cache for immediate protection of new knowledge
            self._rebuild_restoration_cache()
            
        # 3. Entropy-Driven Expert Sharpening (Temperature Decay)
        if getattr(self.config, 'use_moe', False):
            new_temp = self.config.moe_temperature * (self.config.moe_temp_decay ** (task_id + 1))
            # Clamp temp to prevent numerical instability, but allow it to go low for "Hard Max"
            new_temp = max(new_temp, 0.05) 
            if hasattr(self.model, 'set_temperature'):
                self.model.set_temperature(new_temp)
            self.logger.info(f"🔥 MoE Sharpening: Temperature adjusted to {new_temp:.4f}")

        # 4. Reset Optimization State (Lookahead)
        if self.config.use_lookahead:
            self.sync_lookahead_weights()

    def _setup_logging(self):
        logger = logging.getLogger('AdaptiveFramework')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _expand_cognitive_capacity(self):
        """
        [V9.4] HEE: Holographic Expert Expansion.
        Spawns new neural real estate to allow learning when backbone is full.
        """
        self.logger.info("⚡ Expanding Neural Expert Bank...")
        
        # 1. Offload 'Ancient' state to Holographic Vault before expansion
        if self.memory and hasattr(self.memory, 'holographic_vault'):
             current_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
             # Use a virtual task ID to index the snapshot
             snapshot_id = getattr(self, 'task_id_counter', self.step_count // 100)
             self.memory.holographic_vault.deposit(snapshot_id, current_params)
        
        # 2. Refresh/Upgrade all adapters
        self._init_adapters_and_hooks(force_upgrade=True)
        
        # 2. Re-attach Optimizer to include new parameters
        if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
            adapter_params = list(self.adapter_bank.parameters())
            if adapter_params:
                # Use current learning rate or config default
                lr = getattr(self.config, 'weight_adaptation_lr', 1e-3)
                self.adapter_optimizer = AdamW(adapter_params, lr=lr)
                self.logger.info(f"✨ Expansion Complete. {len(adapter_params)//4} experts now online.")

    def _init_adapters_and_hooks(self, force_upgrade: bool = False):
        """
        Initialize adapters by inspecting layer dimensions upfront.
        [V9.4] Support for Force Upgrade during expansion.
        """
        valid_types = (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU, nn.MultiheadAttention)
        self.layer_map = {}
        
        # 1. Initialize Bank if not already present
        num_potential = sum(1 for _ in self.model.modules())
        if not hasattr(self, 'adapter_bank') or self.adapter_bank is None:
            try:
                self.adapter_bank = AdapterBank(num_layers=num_potential, device=self.device)
            except Exception:
                self.adapter_bank = None
        
        # 2. Attach hooks and pre-allocate adapters
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, valid_types):
                self.layer_map[name] = idx
                
                # Pre-allocate adapter if possible
                if self.adapter_bank:
                    out_dim = getattr(module, 'out_features', getattr(module, 'out_channels', getattr(module, 'hidden_size', None)))
                    if out_dim:
                        self.adapter_bank.ensure_index(idx, out_dim=int(out_dim), force_upgrade=force_upgrade)
                
                # Only register hook if not already present (prevents duplicates during expansion)
                if not hasattr(module, '_antara_hook_installed'):
                    module.register_forward_hook(self._generate_fast_hook(idx, type(module)))
                    module._antara_hook_installed = True
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros((idx, 4), device=self.device, dtype=torch.float32, requires_grad=False)

    def _perform_self_assessment(self):
        """
        [V9.4] Autonomous Cognitive Assessment.
        Detects the current training regime and adjusts meta-parameters.
        """
        # 1. Inspect Weight Distribution
        weights = []
        for p in self.model.parameters():
            if p.requires_grad and p.dim() >= 2:
                weights.append(p.data.view(-1))
        
        is_pretrained = False
        if weights:
            all_weights = torch.cat(weights)
            std = all_weights.std().item()
            mean = all_weights.mean().item()
            # [V9.4] Refined Heuristic: Scratch init (Kaiming) can have std up to 0.15.
            # Pretrained models usually have more skewed distributions or shifted means.
            if abs(mean) > 0.05 or std > 0.2: 
                is_pretrained = True

        # 2. Inspect Memory State
        has_memory = False
        if self.memory:
            if hasattr(self.memory, 'omega') and self.memory.omega:
                if any(v.abs().sum() > 0 for v in self.memory.omega.values()):
                    has_memory = True
            
        # 3. Classify Regime
        if not is_pretrained and not has_memory:
            self.regime = CognitiveRegime.SCRATCH
        elif is_pretrained and not has_memory:
            self.regime = CognitiveRegime.TRANSFER
        elif is_pretrained and has_memory:
            self.regime = CognitiveRegime.CONTINUOUS
        else:
            self.regime = CognitiveRegime.GHOST

        self.logger.info(f"[SENTIENT] Self-Assessment Complete. Regime: {self.regime.value.upper()}")
        
        # 4. Adaptive Policy Injection
        if self.regime == CognitiveRegime.SCRATCH:
            self.config.meta_learning_rate *= 1.5
            self.config.novelty_threshold = 1.0 
        elif self.regime == CognitiveRegime.TRANSFER:
            self.config.si_lambda *= 2.0
            self.config.novelty_threshold = 2.0
        elif self.regime == CognitiveRegime.CONTINUOUS:
            self.config.si_lambda *= 5.0
            self.config.novelty_threshold = 4.0
        elif self.regime == CognitiveRegime.GHOST:
            self.config.si_lambda *= 10.0
            self.config.novelty_threshold = 0.5

    def _generate_fast_hook(self, layer_idx, module_type):
        def hook(module, inputs, output):
            try:
                inp = output
                if isinstance(inp, torch.Tensor):
                    # Fast Telemetry
                    with torch.no_grad():
                        if inp.numel() > 0:
                            # Use simple stats to avoid sync overhead
                            # [V26.5] Robust device sync for telemetry
                            if inp.device != self.telemetry_buffer.device:
                                inp = inp.to(self.telemetry_buffer.device)
                                
                            mean_val = inp.mean().detach()
                            var_val = inp.var(unbiased=False).detach()
                            
                            # Double check buffer device right before assignment
                            self.telemetry_buffer[layer_idx, 0] = mean_val
                            self.telemetry_buffer[layer_idx, 1] = var_val
                            self.telemetry_buffer[layer_idx, 2] = 0 # Optimized out
                            self.telemetry_buffer[layer_idx, 3] = 0 # Optimized out

                    # Apply Adapter
                    if self.adapter_bank:
                        adapted = self.adapter_bank.apply(layer_idx, inp, module_type)
                        if adapted is not inp:
                            inp = adapted

                    # [V8.0] Apply Sentient Affine Modifiers (System 2)
                    if self.current_modifiers is not None:
                        # [V17] Detach modifiers during internal/eval passes to prevent graph leaks
                        mods = self.current_modifiers
                        if self._internal_consolidation_mode or not self.training:
                            mods = mods.detach()
                            
                        if mods.dim() == 1:
                            scale = 1.0 + mods[0]
                            shift = mods[1]
                        else:
                            # Batch of modifiers: [B, 2]
                            b_size = inp.size(0)
                            if mods.size(0) == b_size:
                                s = mods[:, 0]
                                f = mods[:, 1]
                                for _ in range(inp.dim() - 1):
                                    s = s.unsqueeze(-1)
                                    f = f.unsqueeze(-1)
                                scale = 1.0 + s
                                shift = f
                            else:
                                # Fallback to scalar mean
                                m = mods.mean(dim=0)
                                scale = 1.0 + m[0]
                                shift = m[1]
                        inp = inp * scale + shift
                    
                    if inp is not output:
                        return inp
            except Exception:
                pass
            return None
        return hook

    def forward(self, *args, **kwargs):
        """
        Antara Forward Pass (System 1 + System 2 Integration)
        """
        # [V12] Fix: Use get instead of pop to ensure task_id reaches MoE backbone
        task_id = kwargs.get('task_id') or getattr(self, '_current_task_id', None)
        if task_id is not None:
            kwargs['task_id'] = task_id
        self._current_task_id = task_id
        fused_latent = None
        if self.perception and len(args) == 1 and isinstance(args[0], dict):
            # Dictionary input (Multi-Modal)
            fused_latent = self.perception(args[0])
            if fused_latent is not None:
                # Pass fused latent to base model
                output = self.model(fused_latent)
            else:
                output = self.model(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        
        # [V7.1] MoE Handling
        moe_indices = None
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
             if output[1].dtype == torch.long:
                 output, moe_indices = output
        
        log_var = torch.tensor(0.0).to(self.device)
        affine_modifiers = None
        
        try:
            # Aggregate Telemetry
            global_state = self.telemetry_buffer.mean(dim=0)
            global_state = torch.nan_to_num(global_state, nan=0.0)
            
            # Introspection Step
            # [V8.3] Surgical Hardening: Skip modifiers and accumulation in internal maintenance mode
            if self._internal_consolidation_mode:
                log_var, action = torch.tensor(0.0), torch.tensor([0.0, 0.0])
                self.current_modifiers = action.detach() # [V17] Detach
                affine_modifiers = action.detach()
            elif self.training:
                # Standard training flow
                log_var, action, log_prob = self.introspection_engine(global_state)
                # [V15.2 IRON CLAD] Only record log-probs during "Real" training.
                # If we are in internal consolidation (dreaming/replay), we skip 
                # meta-prob collection to prevent graph leakage across steps.
                if log_prob is not None and not self._internal_consolidation_mode:
                    self.meta_log_probs.append(log_prob)
                self.current_modifiers = action.detach().squeeze() # [V17] CRITICAL: Detach to break step-to-step graph link
                affine_modifiers = action.detach()
            else:
                # Standard inference/evaluation flow (Preserves Sentience)
                with torch.no_grad():
                    log_var, action, _ = self.introspection_engine(global_state)
                self.current_modifiers = action.detach().squeeze() # [V17] CRITICAL: Detach to break step-to-step graph link
                affine_modifiers = action.detach()
                
        except Exception:
            self.meta_log_probs.clear()

        # [V8.0] Store fused latent for consciousness
        # [V17] CRITICAL: Detach to prevent cross-step graph leakage
        self._last_fused_latent = fused_latent.detach() if fused_latent is not None else None

        # [V9.0] World Model Foresight - Just Record Inputs for optimization in train_step
        if self.world_model and fused_latent is not None:
            action_context = self.current_modifiers.detach() if self.current_modifiers is not None else None
            if action_context is not None:
                action_context = action_context.unsqueeze(0).expand(fused_latent.size(0), -1)
            
            # Store inputs for next step's World Model training
            # We must DETACH them to avoid cross-step graphs hitting self.model
            self._current_wm_inputs = (fused_latent.detach(), action_context)
            
            # For inference foresight (without gradients)
            with torch.no_grad():
                self._current_z_prediction = self.world_model(fused_latent, action_context)
            
        return output, log_var, affine_modifiers, moe_indices

    def _clear_all_internal_caches(self):
        """[V17] The Nuclear Reset: Purge every member variable that could hold a tensor."""
        self.meta_log_probs.clear()
        self.current_modifiers = None
        self._last_fused_latent = None
        self._current_wm_inputs = None
        self._last_latent = None
        
        # Clear auxiliary losses in MoE
        if hasattr(self.model, 'zero_grad'):
            self.model.zero_grad(set_to_none=True)
        
        # Manually Reach into MoE if possible
        for m in self.modules():
            if hasattr(m, 'aux_loss'):
                # [V17.2] Use 0.0 instead of None to prevent 'NoneType' errors
                # if the next forward pass fails early.
                m.aux_loss = 0.0 
            if hasattr(m, 'expert_usage'):
                # We don't want to lose usage stats, so we just detach it
                if isinstance(m.expert_usage, torch.Tensor):
                    m.expert_usage = m.expert_usage.detach()
        
        if self.consciousness:
            self.consciousness.current_thought_trace.clear()
            
        # [V17] Final detaching of any lingering attributes
        for attr in ['_last_loss_val', 'reward_baseline']:
            if hasattr(self, attr) and isinstance(getattr(self, attr), torch.Tensor):
                setattr(self, attr, getattr(self, attr).detach())
                
    # Removed duplicate inference_step (V8.4) as it is superseded by V9.1 below.

    def clear_cognitive_buffers(self):
        """[V8.3] Explicitly clear all meta-learning and consciousness buffers."""
        self.meta_log_probs.clear()
        self.current_modifiers = None
        if self.consciousness:
            self.consciousness.current_thought_trace.clear()
            # We don't clear the thought_stream as it's a deque for long-term stats,
            # but we could clear it if memory pressure is critical.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_emotional_parameters(self, emotion: str) -> Tuple[float, bool, float]:
        """Map emotional state to learning parameters."""
        # Maps emotion to: (plasticity_gate, apply_memory, learning_rate_multiplier)
        params = {
            "confident": (1.0, True, 1.0),
            "anxious": (0.9, True, 1.2),
            "curious": (1.0, True, 1.1),
            "bored": (0.7, True, 0.8),
            "frustrated": (1.1, True, 1.5), 
            "satisfied": (1.0, True, 1.0),
            "overwhelmed": (0.5, True, 0.6),
        }
        return params.get(emotion, (1.0, True, 1.0))

    def _apply_gradient_centralization(self):
        """[V8.0] Gradient Centralization: GC = grad - mean(grad)."""
        for n, p in self.model.named_parameters():
            if p.grad is None: continue
            if p.dim() > 1: # Only for weights, not biases
                p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, p.dim())), keepdim=True))

    def _lookahead_step(self):
        """[V8.0] Lookahead Optimizer Step — V17 Sacred-Mask-Aware."""
        if not self.config.use_lookahead: return
        
        self.lookahead_step += 1
        if self.lookahead_step % self.lookahead_k == 0:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.slow_weights:
                    dev = p.device
                    fast = p.data
                    
                    # [V26.2] Strict Device Casting for Slow Weights
                    slow = self.slow_weights[n]
                    alpha = float(getattr(self, 'lookahead_alpha', 0.5))
                    
                    new_slow = slow + alpha * (fast - slow)
                    
                    # [V17] Respect Sacred Mask: never overwrite protected coordinates
                    mask = self.memory.sacred_mask.get(n) if getattr(self, 'memory', None) else None
                    if mask is not None:
                        if mask.any():
                            # [V26.5] FIX: Take SLOW (Anchor) instead of FAST (Plastic) for sacred coords
                            new_slow = torch.where(mask.to(dev), slow, new_slow)
                    
                    p.data.copy_(new_slow)
                    self.slow_weights[n] = new_slow.detach()

    def train_step(self, *model_inputs, target_data, task_id: int = 0, enable_dream: bool = True, meta_step: bool = True, record_stats: bool = True):
        """
        Single training step with V8.0 enhancements.
        """
        self.model.train()
        
        # [V9.2 CRITICAL BUGFIX] Zero ALL optimizers
        # Previously only the main optimizer was zeroed, causing gradient 
        # accumulation in adapters and System 2, leading to corruption.
        self.optimizer.zero_grad()
        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer:
            self.adapter_optimizer.zero_grad()
        if hasattr(self, 'meta_optimizer') and self.meta_optimizer:
            self.meta_optimizer.zero_grad()
        if hasattr(self, 'world_model_optimizer') and self.world_model_optimizer:
            self.world_model_optimizer.zero_grad()
            
        # [V17] Hard State Reset: Clear ALL buffers and internal caches
        # to ensure absolute graph isolation across training steps.
        self._clear_all_internal_caches()
        
        # [V9.4] CAS Protocol: Saturation & Expansion Trigger
        
        # [V9.4] CAS Protocol: Saturation & Expansion Trigger
        # If backbone is saturated (>95%), we must expand the mind.
        # [V9.4] BatchNorm Stabilization (NeurIPS Killshot)
        # Ensure that anchored modules stay in .eval() mode to prevent 
        # running_mean/var drift during forward passes of new tasks.
        if self.memory and hasattr(self.memory, 'sacred_mask'):
            for m_name, m in self.named_modules():
                # If any parameter in this module is sacred, we stabilize BN
                is_sacred_module = False
                for p_name, p in m.named_parameters(recurse=False):
                    full_p_name = f"{m_name}.{p_name}" if m_name else p_name
                    if full_p_name in self.memory.sacred_mask and self.memory.sacred_mask[full_p_name].any():
                        is_sacred_module = True
                        break
                
                if is_sacred_module and isinstance(m, (torch.nn.modules.batchnorm._BatchNorm)):
                    m.eval()
                    m.track_running_stats = False # Absolute Freeze

        # [V9.4] Unified Memory Snapshot (Full-Spectrum Coverage)
        # Capture parameters before optimizer.step() for ALL models in the memory system.
        param_before = {}
        if self.memory and self.memory.method != 'none':
            param_before = self.memory.before_step_snapshot()
        
        # 3. Forward Pass & Loss Calculation
        total_loss = None
        try:
            # [V9.2] Use modern torch.amp.autocast
            with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == 'cuda'):
                # [V12] Fix: Pass task_id to forward for expert routing
                output, log_var, modifiers, moe_indices = self.forward(*model_inputs, task_id=task_id)
                
                # Unpack standard model outputs
                if isinstance(output, tuple):
                    logits = output[0]
                    features = output[1] if len(output) > 1 else None
                else:
                    logits = output
                    features = None

                # [V8.0] Consciousness Observation (System 2)
                consciousness_metrics = {}
                if self.consciousness:
                    # Observe and Think (Recursive Global Workspace)
                    cons_features = features.detach() if features is not None else None
                    obs = self.consciousness.observe(
                        y_true=target_data, 
                        y_pred=logits, 
                        features=cons_features,
                        internal_mode=self._internal_consolidation_mode
                    )
                    consciousness_metrics = obs
                
                # 3. Compute Base Loss
                if target_data.dtype in [torch.float16, torch.float32, torch.float64] or logits.shape == target_data.shape:
                    loss = F.mse_loss(logits, target_data)
                else:
                    # [V29] Pure Class-IL: Global cross-entropy over ALL classes.
                    # Task-local slicing was causing 0% accuracy on previous tasks
                    # by zeroing gradient signal to old class neurons.
                    loss = F.cross_entropy(logits, target_data.view(-1))
                
                # 4. Memory Regularization
                reg_loss = torch.tensor(0.0, device=self.device)
                if self.memory:
                    penalty = self.memory.compute_penalty()
                    if 'surprise' in consciousness_metrics:
                        s_val = float(consciousness_metrics['surprise'])
                        s_clamped = max(-3.0, min(3.0, s_val))
                        scaling_factor = math.exp(-s_clamped)
                        penalty *= scaling_factor
                    reg_loss = penalty

                # [V9.0] World Model Latent Prediction
                wm_loss = torch.tensor(0.0, device=self.device)
                if self.world_model and features is not None:
                    # [V26.5] Robust device alignment for temporal forecasting
                    if features.dim() == 3 and features.size(1) > 1:
                        z_t = features[:, :-1, :]
                        z_actual = features[:, 1:, :].detach()
                        z_pred = self.world_model(z_t)
                        _, wm_loss = self.world_model.compute_surprise(z_pred, z_actual)
                    elif hasattr(self, '_last_latent') and self._last_latent is not None:
                        # Ensure temporal context is on the same device as the predictor
                        z_actual = features.detach()
                        z_pred = self.world_model(self._last_latent.to(self.device))
                        _, wm_loss = self.world_model.compute_surprise(z_pred, z_actual)
                    
                    self._last_latent = features.detach()

                # [V9.2] Expert Balancing Loss
                aux_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.model, 'get_aux_loss'):
                    aux_loss = self.model.get_aux_loss() * 0.1

                # Aggregation logic inside autocast
                total_loss = loss + reg_loss + aux_loss + (wm_loss * 0.5)
        
            if total_loss is None:
                raise RuntimeError("Cortex Critical: total_loss was never computed in train_step")
            
            # 5. [V15 TITANIUM] Consolidated Backward Sequence
            # We integrate the Meta-Loss directly into the main gradient flow.
            meta_loss = torch.tensor(0.0, device=self.device)
            has_meta = len(self.meta_log_probs) > 0
            
            if has_meta:
                current_loss_val = loss.item()
                if hasattr(self, '_last_loss_val'):
                    # Advantage-based policy gradient
                    reward = self._last_loss_val - current_loss_val
                    self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
                    advantage = reward - self.reward_baseline
                    
                    # REINFORCE update: Maximize Advantage * LogProb
                    # advantage is cast to float to ensure it acts as a constant in the graph
                    meta_loss = -torch.stack(self.meta_log_probs).mean() * float(advantage)
                
                self._last_loss_val = current_loss_val

            # Final Aggregation
            final_loss = total_loss + meta_loss
            
            # SINGLE BACKWARD PASS (The "Killshot" for graph errors)
            self.scaler.scale(final_loss).backward()
            
            # 6. Unscale & Clip
            self.scaler.unscale_(self.optimizer)
            if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer:
                self.scaler.unscale_(self.adapter_optimizer)
            if hasattr(self, 'meta_optimizer') and self.meta_optimizer:
                self.scaler.unscale_(self.meta_optimizer)
            if hasattr(self, 'world_model_optimizer') and self.world_model_optimizer:
                self.scaler.unscale_(self.world_model_optimizer)
            
            # [V9.2] Memory Handler now sees UNSCALED gradients for OGD projection
            if self.config.use_gradient_centralization:
                self._apply_gradient_centralization()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            if self.world_model:
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.gradient_clip_norm)
            
            # 7. Optimizer Steps via Scaler
            self.scaler.step(self.optimizer)
            if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer:
                self.scaler.step(self.adapter_optimizer)
            if hasattr(self, 'meta_optimizer') and self.meta_optimizer:
                self.scaler.step(self.meta_optimizer)
            if hasattr(self, 'world_model_optimizer') and self.world_model_optimizer:
                self.scaler.step(self.world_model_optimizer)
                
            self.scaler.update()

            # [V17.3] Path Accumulation MUST happen before the finally block clears gradients
            if self.memory and self.memory.method != 'none':
                self.memory.accumulate_path(param_before)

            # [V17] Post-Optimizer Sacred Restoration (combats weight decay drift)
            self._apply_sacred_restoration()
            
            # [V8.0] Lookahead Step
            if self.config.use_lookahead:
                self._lookahead_step()

        finally:
            # MANDATORY CLEANUP (V17.0 ETERNAL MIND - TOTAL AMNESIA)
            # This runs even if forward(), backward() or optimizer.step() fails.
            self._clear_all_internal_caches()
            
            # [V17] Absolute Graph Kill: Zero gradients and release tensor memory
            # We must zero ALL tracked components to prevent graph leaks in meta-policy
            with torch.no_grad():
                self.model.zero_grad(set_to_none=True)
                if self.introspection_engine:
                    self.introspection_engine.zero_grad(set_to_none=True)
                if self.world_model:
                    self.world_model.zero_grad(set_to_none=True)
                if self.perception:
                    self.perception.zero_grad(set_to_none=True)
            
            # [V26.0] Maintenance: Only clear cache after consolidation or periodically
            # Aggressive clearing in every step tanks performance.
            if self.step_count % 100 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # [V8.1] Direct Weight Adaptation via PerformanceMonitor
        if self.performance_monitor and hasattr(self, 'current_modifiers') and self.current_modifiers is not None:
            prev_loss = getattr(self, '_last_loss_val', loss.item())
            self.performance_monitor.adapt_weights(
                current_loss=loss.item(),
                previous_loss=prev_loss,
                activations={
                    'affine_modifiers': self.current_modifiers,
                    'telemetry_buffer': self.telemetry_buffer,
                    'layer_map': getattr(self, 'layer_map', {})
                }
            )
            
        # 9. Meta-Learning & Dreaming (V7.0 Restoration)
        if meta_step and self.meta_controller:
            # [V8.1] Full Meta-Controller Integration - Reptile, LR Scheduling, Curriculum
            prev_loss = getattr(self, '_last_loss_val', loss.item())
            meta_metrics = self.meta_controller.adapt(
                loss=loss.item(),
                performance_metrics={
                    'loss': loss.item(),
                    'loss_improvement': prev_loss - loss.item()
                }
            )
            
        if enable_dream and self.config.enable_dreaming and (self.step_count % self.config.dream_interval == 0):
             self.learn_from_buffer(batch_size=getattr(self.config, 'dream_batch_size', 32))
             
        if enable_dream: self.step_count += 1
        
        # [V8.1] Periodic Memory Consolidation (EWC/SI/OGD)
        if self.memory and self.consolidation_scheduler and self.memory.method != 'none':
            z_score = consciousness_metrics.get('surprise', 0.0)
            should_consolidate, reason = self.consolidation_scheduler.should_consolidate(
                current_step=self.step_count,
                z_score=z_score,
                mode=self.meta_controller.current_mode if self.meta_controller else 'NORMAL',
                criterion=getattr(self.config, 'consolidation_criterion', 'hybrid')
            )
            if should_consolidate:
                self._internal_consolidation_mode = True
                try:
                    self.memory.consolidate(
                        feedback_buffer=self.feedback_buffer,
                        current_step=self.step_count,
                        z_score=z_score,
                        mode=self.meta_controller.current_mode if self.meta_controller else 'NORMAL'
                    )
                    self.consolidation_scheduler.record_consolidation(self.step_count)
                    self.logger.info(f"[MEMORY] Auto-consolidation triggered: {reason}")
                finally:
                    self._internal_consolidation_mode = False
            
        # [V9.0] Periodic Neural Health Check & Autonomic Repair
        if self.health_monitor and self.step_count % self.config.health_check_interval == 0:
            report = self.health_monitor.check_vital_signs()
            projector = getattr(self.memory, 'projector', None) if self.memory else None
            
            # [V9.3] Fetch and Pass Expert Usage for selective repair
            expert_usage = None
            if hasattr(self.model, 'get_expert_usage'):
                expert_usage = self.model.get_expert_usage()
                
            repairs = self.health_monitor.autonomic_repair(report, projector=projector, expert_usage=expert_usage)
            
            # Reset usage for the next health window
            if hasattr(self.model, 'reset_usage'):
                self.model.reset_usage()
                
            if repairs > 0:
                self.logger.info(f"[AUTONOMIC] Neural Health Stabilized ({repairs} repairs).")

        # [V16] Automatically populate feedback buffer for Replay/Dreaming/Consolidation
        if self.feedback_buffer:
            self.feedback_buffer.add(model_inputs, {}, logits, target_data, 0.0, loss.item(), task_id=task_id)
            # [V17] Also populate prioritized_buffer so dreaming can sample
            if self.prioritized_buffer:
                snapshot = self.feedback_buffer.buffer[-1]  # Get the just-added snapshot
                z_score = consciousness_metrics.get('surprise', 0.0)
                self.prioritized_buffer.add(snapshot, z_score=z_score, importance=loss.item())

        # [V8.0] Ensure all metrics for demo are present

        z_score = consciousness_metrics.get('surprise', 0.0)
        mode = self.meta_controller.current_mode if self.meta_controller else 'NORMAL'
        plasticity = consciousness_metrics.get('learning_rate_multiplier', 1.0)

        # [V9.2] Expert Diversity Metrics
        expert_entropy = 0.0
        if moe_indices is not None:
            # Calculate entropy of expert selection in this batch
            # moe_indices: [B, k]
            counts = torch.bincount(moe_indices.view(-1), minlength=getattr(self.model, 'num_experts', 4))
            probs = counts.float() / counts.sum()
            expert_entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

        return {
            'loss': loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item(),
            'z_score': z_score,
            'mode': mode,
            'plasticity': plasticity,
            'expert_entropy': expert_entropy,
            **consciousness_metrics
        }

    def learn_from_buffer(self, batch_size: int = 32, num_epochs: int = 1):
        """
        Active Replay ("Dreaming") for multi-input models.
        """
        if len(self.feedback_buffer.buffer) < 10:
            return
            
        self.model.train()
        self._internal_consolidation_mode = True # [V17] Critical: Detach modifiers to prevent graph leaks
        try:
            for _ in range(num_epochs):
                buffer_size = len(self.feedback_buffer.buffer)
                effective_batch = min(batch_size, buffer_size)

                if effective_batch <= 0:
                    return

                if self.prioritized_buffer:
                    samples = self.prioritized_buffer.sample_batch(
                        effective_batch,
                        use_priorities=True
                    )
                else:
                    samples = random.sample(
                        self.feedback_buffer.buffer,
                        effective_batch
                    )
                    
                if not samples:
                    print("DEBUG: No samples retrieved.")
                    continue
                    
                # --- New Batching Logic for Multi-Input Models ---
                try:
                    # Transpose the list of input_args tuples
                    # Assumes all experiences in the buffer have the same number of input args
                    num_args = len(samples[0].input_args)
                    batch_args = []
                    for i in range(num_args):
                        # For each argument position, concatenate the tensors from all samples
                        # [V26.1] Hardened: Explicitly move to self.device
                        arg_tensors = [s.input_args[i].to(self.device) for s in samples]
                        batch_args.append(torch.cat(arg_tensors, dim=0))
                    
                    batch_targets = torch.cat([s.target.to(self.device) for s in samples], dim=0)

                except Exception as e:
                    print(f"DEBUG: Dream Batch Failed: {e}")
                    self.logger.debug(f"Failed to create replay batch, skipping dream step: {e}")
                    continue
                    
                # Call train_step with unpacked arguments
                # Note: We don't want infinite recursion, so we call a simpler step or just forward/backward manually
                # But for simplicity in V8.0, we'll just do manual forward/backward here to avoid complexity
                self.optimizer.zero_grad()
                
                # [V17] Determine the task_id for this replay batch.
                # All samples in the batch may come from different tasks,
                # so we group by task_id and compute scoped loss for each.
                sample_task_ids = [getattr(s, 'task_id', -1) for s in samples]
                
                if isinstance(batch_args, list):
                    outputs = self.model(*batch_args)
                else:
                    outputs = self.model(batch_args)
                    
                if hasattr(outputs, 'logits'): logits = outputs[0] if getattr(outputs, 'logits', None) is None else outputs.logits
                elif isinstance(outputs, tuple): logits = outputs[0]
                else: logits = outputs


                # [V27] Dynamic Task Mapping for Replay
                num_classes_per_task = getattr(self.config, 'classes_per_task', 10)
                
                # [V17] TASK-SCOPED DREAMING: Compute per-sample task-scoped CE loss
                # to prevent replay from suppressing other tasks' logits.
                if logits.shape != batch_targets.shape and logits.dim() > batch_targets.dim() and batch_targets.dim() == 1:
                    if batch_targets.dtype != torch.long:
                        batch_targets = batch_targets.long()
                    
                    # Check if all samples share the same task
                    unique_tasks = set(sample_task_ids)
                    if len(unique_tasks) == 1 and -1 not in unique_tasks:
                        tid = sample_task_ids[0]
                        if logits.shape[-1] >= (tid + 1) * num_classes_per_task:
                            s, e = tid * num_classes_per_task, (tid + 1) * num_classes_per_task
                            loss = F.cross_entropy(logits[:, s:e], batch_targets % num_classes_per_task)
                        else:
                            loss = F.cross_entropy(logits, batch_targets)
                    elif -1 not in unique_tasks:
                        # Mixed-task batch: compute per-sample scoped loss
                        per_sample_losses = []
                        for i, tid in enumerate(sample_task_ids):
                            # [V27] Dynamic Mapping
                            if logits.shape[-1] >= (tid + 1) * num_classes_per_task:
                                s, e = tid * num_classes_per_task, (tid + 1) * num_classes_per_task
                                per_sample_losses.append(
                                    F.cross_entropy(logits[i:i+1, s:e], (batch_targets[i:i+1] % num_classes_per_task))
                                )
                            else:
                                per_sample_losses.append(
                                    F.cross_entropy(logits[i:i+1], batch_targets[i:i+1])
                                )
                        loss = torch.stack(per_sample_losses).mean()
                    else:
                        # Fallback: no task_id available
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
                elif logits.shape == batch_targets.shape:
                    loss = F.mse_loss(logits.float(), batch_targets.float())
                else:
                    loss = F.mse_loss(logits.float(), batch_targets.float())
                
                # [V9.0] Auxiliary Loss (Load Balancing, etc.)
                if hasattr(self.model, 'get_aux_loss'):
                    aux = self.model.get_aux_loss()
                    loss += aux
                    metrics['aux_loss'] = aux.item() if hasattr(aux, 'item') else 0.0

                # [V9.1] FIX: Add Memory Regularization (EWC/SI) to Dream Loss
                # Dreaming must respect constraints of previous tasks!
                reg_loss = torch.tensor(0.0, device=self.device)
                if self.memory:
                    reg_loss = self.memory.compute_penalty(
                       adaptive_mode='DREAM',
                       step_in_mode=0
                    )
                    metrics['reg_loss'] = reg_loss.item()

                # [V17] Total Dreaming Loss aggregation
                # Experts already handled in 'loss' += aux above, 
                # but we explicitly calculate any remaining components.
                total_loss = loss + reg_loss
                
                # [V17] Hardened Replay: Use scaler if available
                if hasattr(self, 'scaler'):
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    # We don't call scaler.update() here because it's called once at the end of train_step
                else:
                    total_loss.backward()
                    self.optimizer.step()
                
                # [V17] Restore sacred weights after dreaming step
                self._apply_sacred_restoration()
        finally:
            self._internal_consolidation_mode = False

    def learn_from_episodic_memory(self, current_surprise: float, current_loss: float, current_features: Optional[torch.Tensor] = None, k: int = 5):
        """
        Replay specific, relevant episodes from consciousness.
        """
        if not self.consciousness: return

        # 1. Retrieve
        memories = self.consciousness.episodic_memory.retrieve_relevant_memories(
            current_surprise=current_surprise,
            current_error=current_loss,
            current_features=current_features,
            k=k
        )
        
        if not memories: return
        
        self.model.train()
        self._internal_consolidation_mode = True
        try:
            # 2. Construct Batch
            try:
                valid_memories = [m for m in memories if m.y is not None and m.x is not None]
                if not valid_memories: return

                # Stack inputs and targets
                # NOTE: Currently supports single-input models for episodic replay
                batch_x = torch.stack([m.x.to(self.device) for m in valid_memories])
                batch_y = torch.stack([m.y.to(self.device) for m in valid_memories])
                
                # 3. Replay (Manual Step)
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == 'cuda'):
                    outputs = self.model(batch_x)
                    if hasattr(outputs, 'logits'): logits = outputs.logits
                    elif isinstance(outputs, tuple): logits = outputs[0]
                    else: logits = outputs
                    
                    if logits.shape == batch_y.shape:
                        loss = F.mse_loss(logits.float(), batch_y.float())
                    else:
                        if logits.dim() > batch_y.dim() and batch_y.dim() == 1:
                             if batch_y.dtype != torch.long: batch_y = batch_y.long()
                             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                        else:
                             loss = F.mse_loss(logits.float(), batch_y.float())
                
                # [V17] Hardened Episodic Replay: Use scaler
                if hasattr(self, 'scaler'):
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # [V17] Post-Replay Restoration
                self._apply_sacred_restoration()
                
            except Exception as e:
                self.logger.error(f"Replay Batch Error: {e}")
                
        except Exception as e:
            self.logger.error(f"Episodic Replay Outer Error: {e}")
        finally:
            self._internal_consolidation_mode = False

    def consolidate_memory(self, **kwargs):
        """Wrapper for Unified Memory consolidation (Backward Compatibility)."""
        result = self.memory.consolidate(**kwargs)
        # [V26.0] Refresh the surgical restoration cache after importance is recalculated
        self._rebuild_restoration_cache()
        return result

    def save_memory(self, name: Optional[str] = None):
        """Wrapper for saving task memory."""
        return self.memory.save_task_memory(name)

    def load_memory(self, path_or_name: str):
        """Wrapper for loading task memory."""
        return self.memory.load_task_memory(path_or_name)

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': self.config,  # Save the configuration
            'model_state': self.model.state_dict(),
            'introspection': self.introspection_engine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'adapters': None if not self.adapter_bank else self.adapter_bank.state_dict(),
            'memory': self.memory.save_task_memory() # Save active memory state too
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
             self.logger.warning(f"Checkpoint not found: {path}")
             return
             
        # Allow loading complex objects (config, memory) by disabling weights_only restriction
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
             # Fallback for older torch versions
             ckpt = torch.load(path, map_location=self.device)
             
        self.model.load_state_dict(ckpt['model_state'])
        self.introspection_engine.load_state_dict(ckpt['introspection'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        if 'adapters' in ckpt and self.adapter_bank:
            self.adapter_bank.load_state_dict(ckpt['adapters'])
        
        # Load memory if present
        if 'memory' in ckpt and isinstance(ckpt['memory'], str):
             self.memory.load_task_memory(ckpt['memory'])
             # [V26.0] Refresh cache after loading external memory
             self._rebuild_restoration_cache()
            
        self.logger.info(f"Checkpoint loaded from {path}")

    def inference_step(self, *model_inputs, task_id=None, return_diagnostics: bool = False, remember: bool = False):
        """
        Low-overhead forward pass for evaluation.
        """
        self.eval()
        self._current_task_id = task_id
        """
        [V9.1] Production Inference Step.
        Runs the cognitive loop (Perception -> World Model -> Cortex -> Consciousness)
        without updating weights. Thread-safe and optimized for serving.
        
        Args:
            remember (bool): If True, stores the experience in Short-Term Memory and Graph Memory.
                             This enables "One-Shot" retention without weight updates.
        """
        self.model.eval()
        
        diagnostics = {}
        
        with torch.no_grad():
            # 1. Forward Pass
            # This handles Perception, MoE Routing, and Introspection automatically
            # [V17] Pass task_id so MoE routes through the correct expert during eval
            outputs, log_var, affine_modifiers, moe_indices = self.forward(*model_inputs, task_id=task_id)
            
            # Extract main prediction
            if hasattr(outputs, 'logits'):
                prediction = outputs.logits
            elif isinstance(outputs, tuple):
                prediction = outputs[0]
            else:
                prediction = outputs
                
            # 2. [V27] Zero-Leakage Prediction (Global head by default)
            # If task_id is None (Class-IL), we use the full head.
            # If task_id is provided (Task-IL), we slice based on dynamic width.
            if task_id is not None and isinstance(prediction, torch.Tensor) and prediction.dim() == 2:
                # Detect width dynamically
                num_classes_per_task = getattr(self.config, 'classes_per_task', 10)
                if prediction.shape[-1] >= (task_id + 1) * num_classes_per_task:
                    s, e = task_id * num_classes_per_task, (task_id + 1) * num_classes_per_task
                    prediction = prediction[:, s:e]

            # 3. World Model Foresight (Optional)
            if self.world_model and hasattr(self, '_current_z_prediction') and self._current_z_prediction is not None:
                z_pred = self._current_z_prediction
                diagnostics['foresight_vector'] = z_pred.mean(dim=0).cpu().numpy()
            
            # 3. Consciousness State (Optional)
            if self.consciousness:
                obs = self.consciousness.observe(
                    y_true=prediction, 
                    y_pred=prediction, 
                    features=self._last_fused_latent if hasattr(self, '_last_fused_latent') else None,
                    internal_mode=self._internal_consolidation_mode
                )
                diagnostics['consciousness'] = obs
            
            # 4. Expert Usage
            if hasattr(self.model, 'get_expert_usage'):
                expert_usage = self.model.get_expert_usage()
                if isinstance(expert_usage, torch.Tensor):
                    diagnostics['expert_usage'] = expert_usage.detach().cpu().numpy()
                else:
                    diagnostics['expert_usage'] = np.asarray(expert_usage)

            # 5. [V9.2] Live Memory Injection (The "Never Forget" Mechanism)
            if remember and self.memory:
                # Create snapshot
                snapshot = type('Snapshot', (), {})()
                snapshot.input_args = model_inputs
                snapshot.target = prediction # Self-supervised (memories are own experiences)
                snapshot.timestamp = time.time()
                
                # A. Add to Graph Memory (Instant episodic retention)
                features = self._last_fused_latent if hasattr(self, '_last_fused_latent') and self._last_fused_latent is not None else None
                
                # If no latent state, try to use input or prediction based on dimension match
                if features is None and hasattr(self.memory, 'feature_dim'):
                    needed_dim = self.memory.feature_dim
                    # Try Input First (Context Key)
                    if len(model_inputs) > 0 and isinstance(model_inputs[0], torch.Tensor):
                        if model_inputs[0].shape[-1] == needed_dim:
                            features = model_inputs[0]
                    
                    # Try Prediction Second (Result Key)
                    if features is None and prediction.dim() > 1:
                        if prediction.shape[-1] == needed_dim:
                            features = prediction
                            
                # Fallback (Legacy)
                if features is None and prediction.dim() > 1:
                    features = prediction

                
                if hasattr(self.memory, 'graph_memory') and self.memory.graph_memory and features is not None:
                    self.memory.graph_memory.add(snapshot, features.detach())
                    diagnostics['memory_stored'] = True
                else:
                    diagnostics['memory_stored'] = False # Explicit fail tracking
                    
                # B. Add to Feedback Buffer (For future "Dreaming" / Weight Adaptation)
                # We interpret the prediction as the target for reinforcement
                if self.feedback_buffer:
                    # We need to unpack args if tuple
                    kwargs = {} # Empty for now
                    self.feedback_buffer.add(
                        input_args=model_inputs,
                        input_kwargs=kwargs,
                        output=prediction,
                        target=prediction, # Self-consistency
                        reward=0.0,
                        loss=0.0
                    )

        # 6. [Surgical Cleanup] Reset modifiers to prevent cross-batch thought persistence
        self.current_modifiers = None
        
        if return_diagnostics:
            return prediction, diagnostics
        else:
            return prediction

    def cognitive_inference(self, *model_inputs, max_steps: int = 3, threshold: float = 0.5, remember: bool = False):
        """
        [V9.3] Metacognitive Inference ("System 2" Thinking).
        Performs iterative refinement based on internal uncertainty (Entropy).
        
        Algorithm:
        1. Fast System 1 pass.
        2. Check Consciousness Entropy.
        3. If Confused (> threshold):
           a. "Reflect": Use World Model to predict consequence.
           b. "Recall": Query Graph Memory using the Reflection.
           c. Return enriched result.
        """
        # 1. System 1 (Fast)
        pred, diagnostics = self.inference_step(*model_inputs, return_diagnostics=True, remember=remember)
        
        # Determine Query Key: Input (Context) or Prediction (Result)?
        query_key = pred
        if hasattr(self.memory, 'feature_dim'):
            needed_dim = self.memory.feature_dim
            # Prefer Input if it matches memory dim (Context Addressing)
            if len(model_inputs) > 0 and isinstance(model_inputs[0], torch.Tensor):
                x_in = model_inputs[0]
                if x_in.shape[-1] == needed_dim:
                    query_key = x_in
            elif pred.dim() > 1 and pred.shape[-1] == needed_dim:
                query_key = pred
        
        cons = diagnostics.get('consciousness', {})
        entropy = cons.get('entropy', cons.get('uncertainty', 0.0))
        
        if entropy < threshold:
            diagnostics['mode'] = 'System 1 (Intuitive)'
            return pred, diagnostics
            
        # 2. System 2 (Slow / Deliberative)
        diagnostics['mode'] = 'System 2 (Deliberative)'
        diagnostics['initial_uncertainty'] = entropy
        
        # A. Reflection (World Model)
        # What is the consequence of this output?
        reflection_vector = None
        if 'foresight_vector' in diagnostics:
            reflection_vector = diagnostics['foresight_vector']
        elif 'expert_usage' in diagnostics:
            reflection_vector = diagnostics['expert_usage'] # Fallback
            
        # B. Active Recall (RAG)
        retrieved_context = []
        if self.memory and hasattr(self.memory, 'graph_memory') and self.memory.graph_memory:
            # We use the Query Key determined above
            # Search broadly (System 2 scans more)
            results = self.memory.graph_memory.retrieve(
                query_vector=query_key,
                k=max_steps
            )
            
            # Extract content (targets)
            for res in results:
                if hasattr(res, 'target') and res.target is not None:
                    retrieved_context.append(res.target)
                elif hasattr(res, 'output') and res.output is not None:
                     retrieved_context.append(res.output)

        diagnostics['retrieved_memories'] = len(retrieved_context)
        
        # C. Refinement (Ensemble/Consensus)
        # If we found memories, maybe we can average them with our prediction?
        # (Naive "Thinking" - adjusting belief based on past experience)
        if retrieved_context and isinstance(pred, torch.Tensor):
            try:
                # Stack memories: [K, ...]
                ctx_tensor = torch.stack([r.to(pred.device) for r in retrieved_context if isinstance(r, torch.Tensor)])
                
                if ctx_tensor.size(0) > 0:
                    # Average over retrieved items to get a single context vector
                    ctx_mean = ctx_tensor.mean(dim=0)
                    
                    # Consensus = 0.7 * Plan + 0.3 * Memory
                    if ctx_mean.shape == pred.shape: # Exact match
                        refined_pred = 0.7 * pred + 0.3 * ctx_mean
                        return refined_pred, diagnostics
                    elif ctx_mean.numel() == pred.numel(): # Element count match (reshape)
                         refined_pred = 0.8 * pred + 0.2 * ctx_mean.view_as(pred)
                         return refined_pred, diagnostics
            except Exception:
                pass
                
        return pred, diagnostics

    def status(self) -> Dict[str, Any]:
        """
        [V9.4] The 'Mind-Space' Telemetry.
        Returns a detailed report on knowledge density and system health.
        """
        sacred_pct = 0.0
        if self.memory and hasattr(self.memory, 'saturation_level'):
            sacred_pct = self.memory.saturation_level * 100

        adapter_count = 0
        if hasattr(self, 'adapter_bank') and self.adapter_bank:
            adapter_count = len(self.adapter_bank.adapters)

        vault_count = 0
        if self.memory and hasattr(self.memory, 'holographic_vault'):
            vault_count = len(self.memory.holographic_vault.vault)

        return {
            "version": "V9.4 Eternal",
            "regime": self.regime.value.upper(),
            "mind_space": {
                "sacred_knowledge_pct": round(sacred_pct, 2),
                "plastic_capacity_pct": round(100.0 - sacred_pct, 2),
                "expansion_experts": adapter_count,
                "holographic_snapshots": vault_count
            },
            "autonomic": {
                "health_stable": self.health_monitor.is_stable() if self.health_monitor else True,
                "step_count": self.step_count
            },
            "device": str(self.device)
        }

    def _apply_sacred_restoration(self):
        """
        [V26.0] TITAN SOUL: Surgical parameter restoration using cached references.
        Reduces restoration overhead from O(N_total) to O(N_sacred).
        """
        if not self._cached_sacred_params:
            # Fallback if cache is empty or not yet built
            if not self.memory or not hasattr(self.memory, "sacred_mask") or not self.memory.sacred_mask:
                return
            self._rebuild_restoration_cache()
            
        with torch.no_grad():
            for param, mask, anchor in self._cached_sacred_params:
                # Surgical Restoration (In-place indexed assignment)
                param.data[mask] = anchor[mask]

            # [V22] BN Cryostasis (Remains O(N_modules) but modules are fewer than params)
            for name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if getattr(m, '_is_sacred_bn', False):
                        m.eval()
                        m.track_running_stats = False
                        # Running stats are restored via anchor if available in memory
                        # (handled in _rebuild_restoration_cache once per consolidation)

    def _rebuild_restoration_cache(self):
        """Build the list of references to sacred parameters and anchors."""
        self._cached_sacred_params = []
        if not self.memory: return
        
        device = next(self.parameters()).device
        for name, param in self.named_parameters():
            mask = self.memory.sacred_mask.get(name)
            if mask is not None and mask.any():
                anchor = self.memory.anchor.get(name)
                if anchor is not None:
                    # [V26.3] Device Affinity: No transfers
                    self._cached_sacred_params.append((param, mask, anchor))
                    
        # Tag BatchNormalization layers
        for name, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                is_sacred = False
                for p_name in ['weight', 'bias']:
                    full_name = f"{name}.{p_name}" if name else p_name
                    if full_name in self.memory.sacred_mask and self.memory.sacred_mask[full_name].any():
                        is_sacred = True; break
                m._is_sacred_bn = is_sacred
