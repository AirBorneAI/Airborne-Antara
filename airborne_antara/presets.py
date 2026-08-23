"""
===== MIRRORMIMD PRESETS SYSTEM =====
One-liner configuration for production-grade meta-learning.

This module provides pre-optimized configurations for different use cases.
Each preset combines years of research into optimal hyperparameter values.

    USAGE:
    ------
    from airborne_antara import AdaptiveFramework, PRESETS
    
    # One-liner: Use production preset
    framework = AdaptiveFramework(model, config=PRESETS.production())
    
    # Or customize: Mix-and-match presets
    config = PRESETS.fast().merge(PRESETS.creativity_boost())
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Any, Optional
from enum import Enum
import torch


class PresetType(Enum):
    """Enum of all available presets."""
    PRODUCTION = "production"
    BALANCED = "balanced"
    FAST = "fast"
    MEMORY_EFFICIENT = "memory_efficient"
    ACCURACY_FOCUS = "accuracy_focus"
    EXPLORATION = "exploration"
    CREATIVITY_BOOST = "creativity_boost"
    STABLE = "stable"
    RESEARCH = "research"
    REAL_TIME = "real_time"


@dataclass
class Preset:
    """
    Base preset: contains all configuration values.
    
    This is a copy of AdaptiveFrameworkConfig optimized for a specific use case.
    Can be merged with other presets or used directly with AdaptiveFramework.
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
    dream_interval: int = 10
    
    # Optimization
    compile_model: bool = True 
    use_amp: bool = False
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_frequency: int = 50
    checkpoint_frequency: int = 500
    gradient_clip_norm: float = 1.0
    adapter_max_norm: float = 2.0
    
    # V6.5: HIERARCHICAL REFLEX
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
    
    # Importance estimation method
    importance_method: str = 'ewc'
    si_lambda: float = 1.0
    si_xi: float = 1e-3
    ewc_lambda: float = 0.4  # EWC regularization strength
    
    # SOTA Unified Memory System (V7.0)
    memory_type: str = 'hybrid'
    consolidation_criterion: str = 'hybrid'
    consolidation_min_interval: int = 30
    consolidation_max_interval: int = 100
    consolidation_surprise_threshold: float = 2.5
    adaptive_lambda: bool = True
    use_prioritized_replay: bool = True
    replay_priority_temperature: float = 0.6
    
    # V7.0: CONSCIOUSNESS LAYER
    enable_consciousness: bool = True
    use_attention: bool = True
    use_intrinsic_motivation: bool = True
    consciousness_buffer_size: int = 5000
    novelty_threshold: float = 2.0
    
    # V1.1.1 "Sentient": Advanced Optimization
    use_lookahead: bool = True       # Lookahead optimizer for better generalization
    use_gradient_centralization: bool = True  # Gradient centralization for stability
    
    # V7.1: CORTEX ENGINE (MoE - Mixture of Experts)
    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    input_dim: int = 0  # Required for MoE gating if > 0. Else uses model_dim.
    
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

    # V8.0: PERCEPTION INTERFACE
    enable_perception: bool = False
    vision_dim: int = 3 # Channels
    audio_dim: int = 80 # Mel bins
    text_dim: int = 0   # Optional projection
    perception_layers: int = 2
    perception_heads: int = 4

    # V9.0: SYNTHETIC INTUITION & HEALTH
    enable_world_model: bool = False
    world_model_loss_weight: float = 0.1
    world_model_plasticity_gamma: float = 1.0
    enable_health_monitor: bool = True
    health_check_interval: int = 20
    enable_performance_monitor: bool = False  # [V8.1] Direct weight editing (experimental)
    
    def merge(self, other: 'Preset') -> 'Preset':
        """Merge another preset into this one (other overwrites self)."""
        self_dict = asdict(self)
        other_dict = asdict(other)
        merged = {**self_dict, **other_dict}
        return Preset(**merged)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for framework initialization."""
        return asdict(self)
    
    def customize(self, **kwargs) -> 'Preset':
        """Create a new preset with custom overrides."""
        return replace(self, **kwargs)
    
    def __repr__(self) -> str:
        """Show key parameters."""
        return (
            f"Preset(lr={self.learning_rate:.1e}, "
            f"model_dim={self.model_dim}, "
            f"buffer_size={self.feedback_buffer_size}, "
            f"memory={self.memory_type}, "
            f"consciousness={self.enable_consciousness})"
        )


class PresetManager:
    """
    Manager class providing access to all presets.
    
    Usage:
        from airborne_antara.presets import PRESETS
        config = PRESETS.production()
        config = PRESETS.fast().customize(learning_rate=5e-4)
    """
    
    @staticmethod
    def production() -> Preset:
        """
        PRODUCTION PRESET (Recommended for real applications)
        
        Goals:
        - Maximum accuracy and stability
        - Handles distribution shifts gracefully
        - Memory-efficient for deployment
        - Robust consolidation strategy
        
        Best for:
        - Live production systems
        - High-stakes decision making
        - Long-running continuous learning
        - Multi-domain scenarios
        
        Key features:
        - Hybrid memory (EWC + SI)
        - Prioritized replay (hard examples matter)
        - Full consciousness layer
        - Moderate learning rate for stability
        - Conservative panic threshold
        """
        return Preset(
            # Large model for expressiveness
            model_dim=512,
            num_layers=8,
            ff_dim=2048,
            num_heads=16,
            dropout=0.1,
            
            # Careful learning: stability over speed
            learning_rate=5e-4,
            meta_learning_rate=5e-5,
            weight_adaptation_lr=5e-6,
            bias_adaptation_lr=5e-6,
            
            # Large buffer for replay
            feedback_buffer_size=20000,
            consciousness_buffer_size=10000,
            
            # SOTA memory: hybrid consolidation
            memory_type='hybrid',
            consolidation_criterion='hybrid',
            importance_method='ewc',
            adaptive_lambda=True,
            use_prioritized_replay=True,
            replay_priority_temperature=0.5,
            
            # Full consciousness: 5D awareness
            enable_consciousness=True,
            use_attention=True,

            use_intrinsic_motivation=True,
            
            # V9.0: Synthetic Intuition (World Model)
            enable_world_model=True,
            enable_health_monitor=True,
            
            # Conservative thresholds
            panic_threshold=0.15,
            active_shield_threshold=0.05,
            novelty_z_threshold=2.5,
            survival_z_threshold=5.0,
            
            # Device optimization
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=torch.cuda.is_available(),  # Use AMP on GPU
            compile_model=True,
            
            # Consolidation: balanced time/surprise
            consolidation_min_interval=50,
            consolidation_max_interval=150,
            consolidation_surprise_threshold=3.0,
            
            # Long warmup for stability
            warmup_steps=100,
        )
    
    @staticmethod
    def balanced() -> Preset:
        """
        BALANCED PRESET (Recommended for most users)
        
        Goals:
        - Good accuracy with reasonable speed
        - Balanced memory usage
        - Easy to tune for specific domains
        
        Best for:
        - General-purpose learning
        - Development and experimentation
        - When you don't know the exact use case
        - Good default starting point
        
        Key features:
        - Medium model size (256 dims)
        - Moderate learning rate
        - Hybrid memory with balanced consolidation
        - Full consciousness but efficient
        """
        return Preset(
            model_dim=256,
            num_layers=6,
            ff_dim=1024,
            num_heads=8,
            dropout=0.1,
            
            learning_rate=1e-3,
            meta_learning_rate=1e-4,
            weight_adaptation_lr=1e-5,
            bias_adaptation_lr=1e-5,
            
            feedback_buffer_size=10000,
            consciousness_buffer_size=5000,
            
            memory_type='hybrid',
            consolidation_criterion='hybrid',
            use_prioritized_replay=True,
            replay_priority_temperature=0.6,
            
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True,
            
            panic_threshold=0.2,
            active_shield_threshold=0.05,
            novelty_z_threshold=2.0,
            survival_z_threshold=4.0,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=False,
            
            consolidation_min_interval=30,
            consolidation_max_interval=100,
            consolidation_surprise_threshold=2.5,
            
            warmup_steps=50,
        )
    
    @staticmethod
    def fast() -> Preset:
        """
        FAST PRESET (Quick iteration, real-time learning)
        
        Goals:
        - Minimize latency
        - Fast adaptation to new domains
        - Real-time responsiveness
        - Smaller memory footprint
        
        Best for:
        - Real-time robotics/RL
        - Fast prototyping
        - Environments with strict time budgets
        - Online learning scenarios
        
        Key features:
        - Small model (128 dims)
        - High learning rates for quick adaptation
        - Reduced replay buffer
        - Consciousness disabled (saves cycles)
        - Aggressive consolidation (frequent resets)
        """
        return Preset(
            # Small model for speed
            model_dim=128,
            num_layers=3,
            ff_dim=512,
            num_heads=4,
            dropout=0.0,
            
            # Aggressive learning
            learning_rate=5e-3,
            meta_learning_rate=5e-4,
            weight_adaptation_lr=1e-4,
            bias_adaptation_lr=1e-4,
            
            # Small buffers for speed
            feedback_buffer_size=2000,
            consciousness_buffer_size=500,
            evaluation_frequency=5,
            dream_interval=5,
            
            # Fast SI: good signal with less computation
            memory_type='si',
            consolidation_criterion='time',
            importance_method='si',
            use_prioritized_replay=False,
            
            # Disable consciousness for speed
            enable_consciousness=False,
            use_attention=False,
            use_intrinsic_motivation=False,
            
            # Aggressive thresholds
            panic_threshold=0.3,
            active_shield_threshold=0.1,
            novelty_z_threshold=1.5,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=False,
            
            # Frequent consolidation
            consolidation_min_interval=10,
            consolidation_max_interval=50,
            consolidation_surprise_threshold=2.0,
            
            # Short warmup
            warmup_steps=20,
            
            # Reduce tracing overhead
            enable_tracing=False,
            log_frequency=100,
        )
    
    @staticmethod
    def memory_efficient() -> Preset:
        """
        MEMORY-EFFICIENT PRESET (Mobile, edge devices, low-RAM)
        
        Goals:
        - Minimal VRAM/RAM usage
        - Long-running with limited memory
        - Efficient consolidation
        
        Best for:
        - Mobile/edge devices
        - Embedded systems
        - Long-running servers with memory constraints
        - Continuous learning with limited budget
        
        Key features:
        - Smallest model (64 dims)
        - Minimal buffering
        - SI-only memory (lightweight)
        - Lightweight consciousness
        - Gradient checkpointing
        """
        return Preset(
            # Minimal model
            model_dim=64,
            num_layers=2,
            ff_dim=256,
            num_heads=2,
            dropout=0.0,
            
            # Moderate learning rates
            learning_rate=1e-3,
            meta_learning_rate=1e-4,
            weight_adaptation_lr=5e-6,
            bias_adaptation_lr=5e-6,
            
            # Minimal buffers
            feedback_buffer_size=1000,
            consciousness_buffer_size=250,
            evaluation_frequency=20,
            dream_interval=20,
            
            # SI only (lighter than EWC)
            memory_type='si',
            consolidation_criterion='time',
            importance_method='si',
            use_prioritized_replay=False,
            
            # Lightweight consciousness
            enable_consciousness=True,
            use_attention=False,
            use_intrinsic_motivation=False,
            
            panic_threshold=0.25,
            active_shield_threshold=0.1,
            novelty_z_threshold=2.0,
            
            device='cpu',  # CPU default for energy efficiency
            use_amp=False,
            compile_model=False,
            
            consolidation_min_interval=50,
            consolidation_max_interval=200,
            
            warmup_steps=30,
            
            # Reduce logging overhead
            log_frequency=200,
            checkpoint_frequency=1000,
            enable_tracing=False,
        )
    
    @staticmethod
    def accuracy_focus() -> Preset:
        """
        ACCURACY-FOCUS PRESET (Maximizing correctness)
        
        Goals:
        - Highest possible accuracy
        - Careful, deliberate learning
        - Excellent generalization
        
        Best for:
        - Medical/healthcare applications
        - Scientific applications
        - High-accuracy prediction
        - When failure cost is high
        
        Key features:
        - Large model (512 dims)
        - Conservative learning rates
        - Large replay buffer with priority sampling
        - Full consciousness + attention
        - Very long consolidation intervals
        """
        return Preset(
            model_dim=512,
            num_layers=10,
            ff_dim=2048,
            num_heads=16,
            dropout=0.15,
            
            # Very careful learning
            learning_rate=1e-4,
            meta_learning_rate=1e-5,
            weight_adaptation_lr=1e-6,
            bias_adaptation_lr=1e-6,
            
            # Large buffer for diverse replay
            feedback_buffer_size=50000,
            consciousness_buffer_size=20000,
            evaluation_frequency=5,
            dream_interval=5,
            
            # Hybrid with emphasis on consolidation
            memory_type='hybrid',
            consolidation_criterion='hybrid',
            importance_method='ewc',
            use_prioritized_replay=True,
            replay_priority_temperature=0.3,  # More greedy to hard examples
            adaptive_lambda=True,
            
            # Full consciousness
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True,
            
            # Very conservative thresholds
            panic_threshold=0.1,
            active_shield_threshold=0.02,
            novelty_z_threshold=3.0,
            survival_z_threshold=6.0,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=torch.cuda.is_available(),
            compile_model=True,
            
            # Very long consolidation intervals
            consolidation_min_interval=100,
            consolidation_max_interval=500,
            consolidation_surprise_threshold=3.5,
            
            warmup_steps=200,
            gradient_clip_norm=0.5,  # Tight clipping
        )
    
    @staticmethod
    def exploration() -> Preset:
        """
        EXPLORATION PRESET (Curiosity-driven learning)
        
        Goals:
        - Maximize exploration and novelty detection
        - Learn diverse behaviors
        - High intrinsic motivation
        
        Best for:
        - Curiosity-driven RL
        - Creative generation
        - Discovery tasks
        - Multi-task learning
        
        Key features:
        - Large model for expressiveness
        - High intrinsic motivation
        - Low consolidation thresholds (frequent resets)
        - Attention enabled
        - Priority on novel/OOD examples
        """
        return Preset(
            model_dim=384,
            num_layers=7,
            ff_dim=1536,
            num_heads=12,
            dropout=0.1,
            
            learning_rate=2e-3,
            meta_learning_rate=2e-4,
            weight_adaptation_lr=2e-5,
            bias_adaptation_lr=2e-5,
            
            feedback_buffer_size=15000,
            consciousness_buffer_size=7500,
            evaluation_frequency=5,
            dream_interval=5,
            
            memory_type='hybrid',
            consolidation_criterion='surprise',  # Consolidate on novelty
            consolidation_surprise_threshold=1.5,  # Lower = more frequent consolidation
            use_prioritized_replay=True,
            replay_priority_temperature=0.8,  # Less greedy, explore more
            
            # FULL consciousness
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True,
            
            # Aggressive novelty detection
            novelty_z_threshold=1.0,  # Very sensitive
            survival_z_threshold=3.0,
            panic_threshold=0.3,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=True,
            
            consolidation_min_interval=20,
            consolidation_max_interval=80,
            
            warmup_steps=30,
        )
    
    @staticmethod
    def creativity_boost() -> Preset:
        """
        CREATIVITY BOOST (Enhanced generation and diversity)
        
        Goals:
        - Maximize output diversity
        - Enhanced creative exploration
        - Better generalization to novel domains
        
        Best for:
        - Generative models
        - Creative applications
        - Diverse multi-task learning
        - Fine-tuning with exploration
        
        Key features:
        - Moderate model size with high dropout
        - Higher exploration/intrinsic motivation
        - Prioritized replay with temperature sampling
        - Attention enabled
        """
        return Preset(
            model_dim=256,
            num_layers=6,
            ff_dim=1024,
            num_heads=8,
            dropout=0.25,  # High dropout for diversity
            
            learning_rate=1.5e-3,
            meta_learning_rate=1.5e-4,
            weight_adaptation_lr=1.5e-5,
            bias_adaptation_lr=1.5e-5,
            
            feedback_buffer_size=12000,
            consciousness_buffer_size=6000,
            
            memory_type='hybrid',
            consolidation_criterion='surprise',
            consolidation_surprise_threshold=2.0,
            use_prioritized_replay=True,
            replay_priority_temperature=1.0,  # Softest sampling = most exploration
            
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True,
            
            novelty_z_threshold=1.5,
            panic_threshold=0.25,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=True,
            
            warmup_steps=40,
        )
    
    @staticmethod
    def stable() -> Preset:
        """
        STABLE PRESET (Maximum robustness and consistency)
        
        Goals:
        - Rock-solid stability
        - Predictable behavior
        - Minimal catastrophic forgetting
        
        Best for:
        - Safety-critical systems
        - Regression (avoiding overfitting)
        - Systems requiring high consistency
        - Continual learning scenarios
        
        Key features:
        - Large model with regularization
        - EWC-only (proven, stable)
        - Very large consolidation intervals
        - Conservative learning
        """
        return Preset(
            model_dim=512,
            num_layers=8,
            ff_dim=2048,
            num_heads=16,
            dropout=0.2,
            
            learning_rate=5e-4,
            meta_learning_rate=5e-5,
            weight_adaptation_lr=5e-6,
            bias_adaptation_lr=5e-6,
            
            feedback_buffer_size=30000,
            consciousness_buffer_size=15000,
            evaluation_frequency=10,
            dream_interval=20,
            
            # EWC-only: proven stable
            memory_type='ewc',
            consolidation_criterion='time',
            importance_method='ewc',
            use_prioritized_replay=False,
            
            enable_consciousness=True,
            use_attention=False,
            use_intrinsic_motivation=False,
            
            panic_threshold=0.15,
            active_shield_threshold=0.03,
            novelty_z_threshold=3.0,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=True,
            
            # Very long consolidation intervals
            consolidation_min_interval=200,
            consolidation_max_interval=1000,
            
            warmup_steps=100,
            gradient_clip_norm=0.5,
        )
    
    @staticmethod
    def research() -> Preset:
        """
        RESEARCH PRESET (Experimentation and ablation)
        
        Goals:
        - Enable all features for research
        - Track all metrics
        - Maximum observability
        
        Best for:
        - Research papers
        - Hyperparameter studies
        - Ablation experiments
        - Understanding framework behavior
        
        Key features:
        - All features enabled
        - Full tracing and logging
        - Balanced hyperparameters
        - Maximum instrumentation
        """
        return Preset(
            model_dim=256,
            num_layers=6,
            ff_dim=1024,
            num_heads=8,
            dropout=0.1,
            
            learning_rate=1e-3,
            meta_learning_rate=1e-4,
            weight_adaptation_lr=1e-5,
            bias_adaptation_lr=1e-5,
            
            feedback_buffer_size=10000,
            consciousness_buffer_size=5000,
            evaluation_frequency=5,
            dream_interval=10,
            
            memory_type='hybrid',
            consolidation_criterion='hybrid',
            use_prioritized_replay=True,
            replay_priority_temperature=0.6,
            
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True,
            
            panic_threshold=0.2,
            novelty_z_threshold=2.0,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=False,
            
            # TRACING ENABLED for research
            enable_tracing=True,
            trace_max_records=5000,
            log_frequency=10,  # Very frequent logging
            checkpoint_frequency=100,  # Frequent checkpoints
            
            consolidation_min_interval=30,
            consolidation_max_interval=100,
            
            warmup_steps=50,
        )
    
    @staticmethod
    def real_time() -> Preset:
        """
        REAL-TIME PRESET (Sub-millisecond inference)
        
        Goals:
        - Minimize inference latency
        - Streaming/online learning
        - Low computational overhead
        
        Best for:
        - Streaming applications
        - Real-time robotics
        - Edge computing
        - Latency-critical applications
        
        Key features:
        - Very small model
        - Minimal consciousness overhead
        - Fast SI consolidation
        - Batch processing optimized
        """
        return Preset(
            model_dim=96,
            num_layers=2,
            ff_dim=384,
            num_heads=3,
            dropout=0.0,
            
            learning_rate=2e-3,
            meta_learning_rate=2e-4,
            weight_adaptation_lr=1e-4,
            bias_adaptation_lr=1e-4,
            
            feedback_buffer_size=1500,
            consciousness_buffer_size=300,
            evaluation_frequency=10,
            dream_interval=10,
            
            memory_type='si',
            consolidation_criterion='time',
            use_prioritized_replay=False,
            
            enable_consciousness=True,
            use_attention=False,
            use_intrinsic_motivation=False,
            
            panic_threshold=0.3,
            novelty_z_threshold=1.5,
            
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=False,
            compile_model=False,
            
            consolidation_min_interval=5,
            consolidation_max_interval=40,
            
            warmup_steps=10,
            
            enable_tracing=False,
            log_frequency=500,
        )


# Global instance for easy access
PRESETS = PresetManager()


# ==================== PRESET UTILITIES ====================

def load_preset(preset_name: str) -> Preset:
    """
    Load a preset by name (string).
    
    Args:
        preset_name: Name of preset (e.g., 'production', 'fast', 'balanced')
    
    Returns:
        Preset object
    
    Example:
        config = load_preset('production')
    """
    name_map = {
        'production': PRESETS.production,
        'balanced': PRESETS.balanced,
        'fast': PRESETS.fast,
        'memory_efficient': PRESETS.memory_efficient,
        'accuracy_focus': PRESETS.accuracy_focus,
        'exploration': PRESETS.exploration,
        'creativity_boost': PRESETS.creativity_boost,
        'stable': PRESETS.stable,
        'research': PRESETS.research,
        'real_time': PRESETS.real_time,
    }
    
    if preset_name.lower() not in name_map:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(name_map.keys())}")
    
    return name_map[preset_name.lower()]()


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.
    
    Returns:
        Dictionary mapping preset name to description
    """
    return {
        'production': 'Production-ready (high accuracy, stable, large model)',
        'balanced': 'Good default (balanced accuracy/speed/memory)',
        'fast': 'Real-time learning (small model, high learning rates)',
        'memory_efficient': 'Minimal footprint (mobile/edge devices)',
        'accuracy_focus': 'Maximum accuracy (careful learning, large buffers)',
        'exploration': 'Curiosity-driven (diversity, novelty detection)',
        'creativity_boost': 'Enhanced generation (high dropout, exploration)',
        'stable': 'Maximum robustness (EWC-only, conservative)',
        'research': 'Full instrumentation (all features, tracing enabled)',
        'real_time': 'Sub-millisecond inference (minimal overhead)',
    }


def compare_presets(*preset_names: str) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple presets side-by-side.
    
    Args:
        preset_names: Names of presets to compare
    
    Returns:
        Dictionary with preset comparison
    
    Example:
        compare_presets('production', 'fast', 'accurate')
    """
    comparison = {}
    for name in preset_names:
        preset = load_preset(name)
        comparison[name] = {
            'model_dim': preset.model_dim,
            'learning_rate': preset.learning_rate,
            'buffer_size': preset.feedback_buffer_size,
            'memory_type': preset.memory_type,
            'consciousness': preset.enable_consciousness,
        }
    return comparison
