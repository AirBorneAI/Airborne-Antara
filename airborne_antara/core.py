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
    learning_rate: float = 1e-3
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
    dream_interval: int = 2 # More Frequent (was 10)
    dream_batch_size: int = 32 # Larger (was hardcoded 16)
    
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
    enable_dreaming: bool = True
    enable_tracing: bool = False
    trace_max_records: int = 1000
    
    # SOTA Unified Memory System (V7.0)
    memory_type: str = 'hybrid'  # 'ewc', 'si', or 'hybrid'
    consolidation_criterion: str = 'hybrid'
    consolidation_min_interval: int = 30
    consolidation_max_interval: int = 100
    consolidation_surprise_threshold: float = 2.5
    adaptive_lambda: bool = True
    use_prioritized_replay: bool = True
    replay_priority_temperature: float = 0.6
    
    # --- V7.0: CONSCIOUSNESS LAYER ---
    enable_consciousness: bool = True
    use_attention: bool = True
    use_intrinsic_motivation: bool = True
    consciousness_buffer_size: int = 5000
    novelty_threshold: float = 2.0
    
    # SI Parameters (Restored)
    importance_method: str = 'ewc'  # 'ewc', 'si', or 'hybrid'
    si_lambda: float = 1.0 # For SI
    ewc_lambda: float = 150.0 # [FIX] Lowered from 5000 to allow baseline plasticity
    si_xi: float = 1e-3
    use_graph_memory: bool = False # [V9.0] Graph-Based Episodic Memory
    graph_memory_threshold: float = 0.85
    use_ogd: bool = False 
    ogd_max_basis_size: int = 256
    enable_holographic_compression: bool = True # [V9.2] Future-Proofing

    # [V8.0] Optimization
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    use_gradient_centralization: bool = True

    # --- V7.1: CORTEX ENGINE (MoE) ---
    use_moe: bool = False
    use_hierarchical_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    num_domains: int = 2
    experts_per_domain: int = 2
    input_dim: int = 0 # Required for MoE gating if > 0. Else uses model_dim.

    # Meta-Controller / Reptile Configuration
    use_reptile: bool = True
    reptile_learning_rate: float = 0.1
    reptile_update_interval: int = 5
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    curriculum_start_difficulty: float = 0.1
    curriculum_increase_rate: float = 0.01
    use_learned_optimizer: bool = True
    learned_optimizer_hidden_dim: int = 32

    # --- V8.0: PERCEPTION INTERFACE ---
    enable_perception: bool = False
    vision_dim: int = 3 # Channels
    audio_dim: int = 80 # Mel bins
    text_dim: int = 0   # Optional projection
    perception_layers: int = 2
    perception_heads: int = 4
    
    ewc_lambda: float = 150.0 
    si_xi: float = 1e-3
    use_graph_memory: bool = False 
    graph_memory_threshold: float = 0.85
    use_ogd: bool = False 
    ogd_max_basis_size: int = 256
    enable_holographic_compression: bool = True 
    
    # --- V9.0: SYNTHETIC INTUITION ---
    enable_world_model: bool = False
    world_model_loss_weight: float = 0.1
    world_model_plasticity_gamma: float = 1.0 
    enable_health_monitor: bool = True
    health_check_interval: int = 100 # [STABILIZATION] Increased from 20 to reduce noise
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
        
    def add(self, input_args: tuple, input_kwargs: dict, output: torch.Tensor, target: torch.Tensor, reward: float, loss: float):
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
            episode=self.total_seen
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
                    
                    if param.ndim == 1:
                        param.mul_(1.0 + scale_factor)
                        param.add_(shift_factor)
                    elif param.ndim >= 2:
                        param.mul_(1.0 + scale_factor)

        return abs(raw_scale) + abs(raw_shift)


# ==================== UNIVERSAL FRAMEWORK ====================

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
        
        # [V8.0] Perception Gateway
        self.perception = None
        if self.config.enable_perception:
            self.perception = PerceptionGateway(self.config)
            self.logger.info("Perception Interface Enabled")
        
        # [V7.1] MoE Transformation
        if getattr(config, 'use_moe', False):
            moe_input_dim = config.input_dim if config.input_dim > 0 else config.model_dim
            if getattr(config, 'use_hierarchical_moe', False):
                from .moe import HierarchicalMoE
                self.logger.info("Transforming Cortex into Hierarchical MoE...")
                self.model = HierarchicalMoE(
                    base_model=self.model,
                    input_dim=moe_input_dim,
                    num_domains=config.num_domains,
                    experts_per_domain=config.experts_per_domain,
                    top_k=config.top_k_experts
                ).to(self.device)
            else:
                self.logger.info("Transforming Cortex into Sparse MoE...")
                self.model = SparseMoE(
                    base_model=self.model,
                    input_dim=moe_input_dim,
                    num_experts=config.num_experts,
                    top_k=config.top_k_experts
                ).to(self.device)
            self.logger.info("   [OK] Transformation Complete. The Mind is now distributed.")
        
        # 5. Memory System (Unified Handler V7.0)
        # We explicitly use UnifiedMemoryHandler, removing all EWC legacy code.
        self.memory = UnifiedMemoryHandler(
            self.model,
            method=getattr(config, 'memory_type', 'hybrid'),
            si_lambda=getattr(config, 'si_lambda', 1.0),
            si_xi=getattr(config, 'si_xi', 1e-3),
            ewc_lambda=getattr(config, 'ewc_lambda', 0.4),
            consolidation_criterion=getattr(config, 'consolidation_criterion', 'hybrid'),
            use_graph_memory=getattr(config, 'use_graph_memory', False),
            use_ogd=getattr(config, 'use_ogd', False),
            ogd_max_basis_size=getattr(config, 'ogd_max_basis_size', 256),
            graph_threshold=getattr(config, 'graph_memory_threshold', 0.85),
            feature_dim=config.model_dim
        )
        self.logger.info(
            f"[BRAIN] Unified Memory System Online "
            f"({getattr(config, 'memory_type', 'hybrid')}, Graph={getattr(config, 'use_graph_memory', False)})"
        )
        
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
        
        # [V8.0] Introspection Engine (System 2 Policy)
        self.introspection_engine = IntrospectionEngine(
            input_dim=config.telemetry_dim, 
            hidden_dim=config.model_dim // 4
        ).to(self.device)
        
        # [V9.0] World Model (I-JEPA)
        self.world_model = None
        if self.config.enable_world_model:
            self.world_model = WorldModel(self.config).to(self.device)
            self._last_z_pred = None
            self.logger.info("[SENSORY] World Model (Foresight) Enabled")
        self.current_modifiers = None
        self.meta_log_probs = []
        self.loss_history = []
        self.reward_baseline = 0.0
        self.alpha = 0.1
        self.step_count = 0

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
            self.slow_weights = {n: p.data.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

        self.logger.info("Airborne-Antara Framework Initialized (V8.0 Sentient Edition)")

    def _setup_logging(self):
        logger = logging.getLogger('AdaptiveFramework')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _init_adapters_and_hooks(self):
        """
        Initialize adapters by inspecting layer dimensions upfront.
        This ensures parameters exist before optimizer creation.
        """
        valid_types = (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU, nn.MultiheadAttention)
        self.layer_map = {}
        
        # Count layers first
        idx = 0
        for _ in self.model.named_modules():
            idx += 1
        num_potential = idx
        
        # Initialize Bank
        try:
            self.adapter_bank = AdapterBank(num_layers=num_potential, device=self.device)
        except Exception:
            self.adapter_bank = None
        
        # Attach hooks and pre-allocate adapters
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, valid_types):
                self.layer_map[name] = idx
                
                # Pre-allocate adapter if possible
                if self.adapter_bank:
                    out_dim = None
                    if hasattr(module, 'out_features'): out_dim = module.out_features
                    elif hasattr(module, 'out_channels'): out_dim = module.out_channels
                    elif hasattr(module, 'hidden_size'): out_dim = module.hidden_size
                    
                    if out_dim:
                        self.adapter_bank.ensure_index(idx, out_dim=int(out_dim))
                
                module.register_forward_hook(self._generate_fast_hook(idx, type(module)))
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros(
            (idx, 4), 
            device=self.device, 
            dtype=torch.float32,
            requires_grad=False
        )

    def _generate_fast_hook(self, layer_idx, module_type):
        def hook(module, inputs, output):
            try:
                inp = output
                if isinstance(inp, torch.Tensor):
                    # Fast Telemetry
                    with torch.no_grad():
                        if inp.numel() > 0:
                            # Use simple stats to avoid sync overhead
                            self.telemetry_buffer[layer_idx, 0] = inp.mean()
                            self.telemetry_buffer[layer_idx, 1] = inp.var(unbiased=False)
                            self.telemetry_buffer[layer_idx, 2] = 0 # Optimized out
                            self.telemetry_buffer[layer_idx, 3] = 0 # Optimized out

                    # Apply Adapter
                    if self.adapter_bank:
                        adapted = self.adapter_bank.apply(layer_idx, inp, module_type)
                        if adapted is not inp:
                            inp = adapted

                    # [V8.0] Apply Sentient Affine Modifiers (System 2)
                    if self.current_modifiers is not None:
                        # self.current_modifiers: [B, 2] or [2]
                        mods = self.current_modifiers
                        if mods.dim() == 1:
                            scale = 1.0 + mods[0]
                            shift = mods[1]
                        else:
                            # Batch of modifiers: [B, 2]
                            # Detect if inp is batch-first
                            b_size = inp.size(0)
                            if mods.size(0) == b_size:
                                s = mods[:, 0]
                                f = mods[:, 1]
                                for _ in range(inp.dim() - 1):
                                    s = s.unsqueeze(-1)
                                    f = f.unsqueeze(-1)
                                scale = 1.0 + s
                                shift = f
                        inp = inp * scale + shift
                    
                    if inp is not output:
                        return inp
            except Exception:
                pass
            return None
        return hook

    def forward(self, *args, **kwargs):
        # [V8.0] Perception Gateway Integration
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
                self.current_modifiers = action # [2]
                affine_modifiers = action.detach()
            elif self.training:
                # Standard training flow
                log_var, action, log_prob = self.introspection_engine(global_state)
                self.meta_log_probs.append(log_prob)
                self.current_modifiers = action.squeeze() # [2]
                affine_modifiers = action.detach()
            else:
                # Standard inference/evaluation flow (Preserves Sentience)
                with torch.no_grad():
                    log_var, action, _ = self.introspection_engine(global_state)
                self.current_modifiers = action.squeeze() # [2]
                affine_modifiers = action.detach()
                
        except Exception:
            self.meta_log_probs.clear()

        # [V8.0] Store fused latent for consciousness
        self._last_fused_latent = fused_latent

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

    def inference_step(self, *args, **kwargs):
        """
        [V8.4] Clean inference path for evaluation.
        Ensures cognitive modifiers (System 2) don't compound and corrupt deep architectures
        like ResNet across sequential evaluation batches.
        """
        was_training = self.training
        self.eval()
        
        # Clear modifiers to prevent previous batch's thoughts from polluting this batch
        self.current_modifiers = None
        
        with torch.no_grad():
            output = self.forward(*args, **kwargs)
            
        # Clear again so the NEXT batch doesn't get corrupted
        self.current_modifiers = None
        
        if was_training:
            self.train()
            
        return output

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
        """[V8.0] Lookahead Optimizer Step."""
        if not self.config.use_lookahead: return
        
        self.lookahead_step += 1
        if self.lookahead_step % self.lookahead_k == 0:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.slow_weights:
                    # slow = slow + alpha * (fast - slow)
                    fast = p.data
                    slow = self.slow_weights[n]
                    new_slow = slow + self.lookahead_alpha * (fast - slow)
                    self.slow_weights[n] = new_slow
                    p.data.copy_(new_slow)

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
        
        # [V9.2 BUGFIX] Properly capture param_before for Synaptic Intelligence / Memory Accumulation
        param_before = {}
        if self.memory and self.memory.method != 'none':
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    param_before[n] = p.data.clone().detach()
        
        # 3. Forward Pass & Loss Calculation
        # [V9.2] Use modern torch.amp.autocast
        with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == 'cuda'):
            # Captured MoE routing for diversity metrics
            output, log_var, modifiers, moe_indices = self.forward(*model_inputs)
            
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
                # Use features if available, else fused latent, else None
                cons_features = features.detach() if features is not None else None
                obs = self.consciousness.observe(
                    y_true=target_data, 
                    y_pred=logits, 
                    features=cons_features,
                    internal_mode=self._internal_consolidation_mode
                )
                consciousness_metrics = obs
            
            # 3. Compute Base Loss
            # Universal Loss: Route based on target type (Classification vs Regression)
            if target_data.dtype in [torch.float16, torch.float32, torch.float64] or logits.shape == target_data.shape:
                loss = F.mse_loss(logits, target_data)
            else:
                loss = F.cross_entropy(logits, target_data.view(-1))
            
            # 4. Memory Regularization
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.memory:
                penalty = self.memory.compute_penalty()
                # [V9.2] Dynamic Surprise Scaling
                if 'surprise' in consciousness_metrics:
                    scaling_factor = 1.0 / (1.0 + float(consciousness_metrics['surprise']))
                    penalty *= scaling_factor
                reg_loss = penalty

            # [V9.2] Expert Balancing Loss
            aux_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.model, 'get_aux_loss'):
                aux_loss = self.model.get_aux_loss() * 0.1

            total_loss = loss + reg_loss + aux_loss

        # 5. Backward Pass with AMP Scaling
        self.scaler.scale(total_loss).backward(retain_graph=len(self.meta_log_probs) > 0)
        
        # Unscale for gradient clipping/centralization
        self.scaler.unscale_(self.optimizer)
        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer:
            self.scaler.unscale_(self.adapter_optimizer)
        
        # [V9.2] Memory Handler now sees UNSCALED gradients for OGD projection
        if self.config.use_gradient_centralization:
            self._apply_gradient_centralization()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # 7. Optimizer Step via Scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
# [V8.0] Sentient Meta-Optimization Loop (REINFORCE with Advantage)
        # This module optimizes the IntrospectionEngine by treating its affine modifiers 
        # as a policy that aims to maximize immediate loss reduction.
        # Reward = (Previous Loss - Current Loss)
        if self.meta_log_probs:
            current_loss_val = loss.item()
            if hasattr(self, '_last_loss_val'):
                reward = self._last_loss_val - current_loss_val
                
                # Update reward baseline (moving average)
                self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
                advantage = reward - self.reward_baseline
                
                # REINFORCE update: Maximize Advantage * LogProb
                meta_loss = -torch.stack(self.meta_log_probs).mean() * advantage
                
                self.meta_optimizer.zero_grad()
                meta_loss.backward() 
                self.meta_optimizer.step()
                
            self._last_loss_val = current_loss_val
            self.meta_log_probs.clear()
            self.current_modifiers = None # Reset for next forward

        # [V9.2 BUGFIX] Removed duplicate self.optimizer.step(). 
        # Weights are already updated by self.scaler.step() above.
        
        # [V8.0] Adapter Optimizer Step
        if self.adapter_optimizer:
            self.adapter_optimizer.step()
        
        # [V8.0] Lookahead Step
        if self.config.use_lookahead:
            self._lookahead_step()
            
        if self.memory and self.memory.method != 'none':
            self.memory.accumulate_path(param_before)
        
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
                self.memory.consolidate(
                    feedback_buffer=self.feedback_buffer,
                    current_step=self.step_count,
                    z_score=z_score,
                    mode=self.meta_controller.current_mode if self.meta_controller else 'NORMAL'
                )
                self.consolidation_scheduler.record_consolidation(self.step_count)
                self.logger.info(f"[MEMORY] Auto-consolidation triggered: {reason}")
            
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
            if isinstance(batch_args, list):
                outputs = self.model(*batch_args)
            else:
                outputs = self.model(batch_args)
                
            if hasattr(outputs, 'logits'): logits = outputs[0] if getattr(outputs, 'logits', None) is None else outputs.logits
            elif isinstance(outputs, tuple): logits = outputs[0]
            else: logits = outputs

            metrics = {}
            
            # Loss Calculation
            # Loss Calculation (Universal V2 - Synced with train_step)
            # Supports:
            # - Vision (Images [B, C, H, W])
            # - Audio (Spectrograms [B, C, T, F] or Waveforms [B, 1, T])
            # - Language (Sequences [B, T])
            # - Tabular (Vectors [B, D])
            
            # 1. Regression / Autoencoder / Audio Enhancement (Shapes match)
            if logits.shape == batch_targets.shape:
                loss = F.mse_loss(logits.float(), batch_targets.float())
            
            # 2. Classification / Sequence
            elif logits.dim() > batch_targets.dim():
                # Handle Sequence Classification [Batch, Seq, Vocab] vs [Batch, Seq]
                if logits.dim() == 3 and batch_targets.dim() == 2:
                     # Check if Classification (Long) or Regression (Float)
                     if batch_targets.dtype == torch.long or batch_targets.shape[1] != logits.shape[2]:
                         # Flatten for CrossEntropy: [B*Seq, Vocab] vs [B*Seq]
                         loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
                     else:
                         # Regression: Pool Sequence [B, S, D] -> [B, D] vs [B, D]
                         pooled_logits = logits.mean(dim=1)
                         loss = F.mse_loss(pooled_logits.float(), batch_targets.float())
                
                # Standard Classification [Batch, C] vs [Batch]
                elif batch_targets.dim() == 1:
                     if batch_targets.dtype != torch.long:
                         batch_targets = batch_targets.long()
                     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
                else:
                     loss = F.mse_loss(logits.float(), batch_targets.float())
            
            # 3. Fallback
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

            # [V9.2] Apply Expert Balancing during dreaming
            aux_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.model, 'get_aux_loss'):
                aux_loss = self.model.get_aux_loss() * 0.1

            total_loss = loss + reg_loss + aux_loss
            total_loss.backward()
            
            self.optimizer.step()

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
            
            loss.backward()
            self.optimizer.step()
            
        except Exception as e:
            self.logger.debug(f"Episodic replay failed: {e}")

    def consolidate_memory(self, **kwargs):
        """Wrapper for Unified Memory consolidation (Backward Compatibility)."""
        return self.memory.consolidate(**kwargs)

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
            
        self.logger.info(f"Checkpoint loaded from {path}")

    def inference_step(self, *model_inputs, return_diagnostics: bool = False, remember: bool = False):
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
            outputs, log_var, affine_modifiers, moe_indices = self.forward(*model_inputs)
            
            # Extract main prediction
            if hasattr(outputs, 'logits'):
                prediction = outputs.logits
            elif isinstance(outputs, tuple):
                prediction = outputs[0]
            else:
                prediction = outputs
                
            # 2. World Model Foresight (Optional)
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
