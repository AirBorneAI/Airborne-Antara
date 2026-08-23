"""
airborne_antara: Production-ready adaptive meta-learning framework
==============================================================

A lightweight Python package enabling continuous model learning and improvement
in production systems through adaptive optimization cycles and online meta-learning.

Key Components:
    - AdaptiveFramework: Base learner with introspection hooks
    - MetaController: Adaptation layer for online learning
    - ProductionAdapter: Simplified API for inference with online learning
"""

__version__ = "0.1.206"
__license__ = "MIT"
__author__ = "Suryaansh Prithvijit Singh"

# Lazy imports to handle circular dependencies and ensure faster startup
def __getattr__(name):
    # ==================== CORE COMPONENTS ====================
    if name in ['AdaptiveFramework', 'AdaptiveFrameworkConfig', 'IntrospectionEngine', 'PerformanceMonitor', 'PerformanceSnapshot', 'CognitiveRegime']:
        from .core import AdaptiveFramework, AdaptiveFrameworkConfig, IntrospectionEngine, PerformanceMonitor, PerformanceSnapshot, CognitiveRegime
        return locals()[name]
    
    # ==================== MEMORY SYSTEM ====================
    elif name in ['UnifiedMemoryHandler', 'PrioritizedReplayBuffer', 'AdaptiveRegularization', 'DynamicConsolidationScheduler', 'OrthogonalProjector', 'HolographicVault']:
        from .memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, AdaptiveRegularization, DynamicConsolidationScheduler, OrthogonalProjector, HolographicVault
        return locals()[name]
    
    # ==================== META CONTROLLER ====================
    elif name in ['MetaController', 'MetaControllerConfig', 'GradientAnalyzer', 'DynamicLearningRateScheduler', 'CurriculumStrategy', 'ReptileOptimizer']:
        from .meta_controller import MetaController, MetaControllerConfig, GradientAnalyzer, DynamicLearningRateScheduler, CurriculumStrategy, ReptileOptimizer
        return locals()[name]

    # ==================== PRODUCTION ADAPTERS ====================
    elif name in ['ProductionAdapter', 'InferenceMode']:
        from .production import ProductionAdapter, InferenceMode
        return locals()[name]

    # ==================== CONSCIOUSNESS (V2 Backend) ====================
    elif name in ['ConsciousnessCore', 'EnhancedConsciousnessCore', 'EmotionalState', 'EmotionalSystem', 'MetaCognition', 'EpisodicMemory', 'SelfModel', 'Personality', 'Introspection', 'AdaptiveAwareness', 'RecursiveGlobalWorkspace']:
        from .consciousness_v2 import EnhancedConsciousnessCore, EmotionalState, EmotionalSystem, MetaCognition, EpisodicMemory, SelfModel, Personality, AdaptiveAwareness, RecursiveGlobalWorkspace
        if name == 'ConsciousnessCore':
            return EnhancedConsciousnessCore
        if name == 'Introspection':
            return MetaCognition
        return locals()[name]

    # ==================== MOE (Mixture of Experts) ====================
    elif name in ['SparseMoE', 'HierarchicalMoE', 'GatingNetwork', 'ExpertBlock', 'AdaptiveExpertBlock']:
        from .moe import SparseMoE, HierarchicalMoE, GatingNetwork, ExpertBlock, AdaptiveExpertBlock
        return locals()[name]

    # ==================== WORLD MODEL ====================
    elif name in ['WorldModel', 'JEPAPredictor']:
        from .world_model import WorldModel, JEPAPredictor
        return locals()[name]

    # ==================== PERCEPTION ====================
    elif name in ['PerceptionGateway', 'VisionEncoder', 'AudioEncoder', 'ModalityFuser']:
        from .perception import PerceptionGateway, VisionEncoder, AudioEncoder, ModalityFuser
        return locals()[name]

    # ==================== ADAPTERS ====================
    elif name in ['AdapterBank', 'FiLMAdapter', 'BottleneckAdapter']:
        from .adapters import AdapterBank, FiLMAdapter, BottleneckAdapter
        return locals()[name]

    # ==================== HEALTH MONITOR ====================
    elif name == 'NeuralHealthMonitor':
        from .health_monitor import NeuralHealthMonitor
        return NeuralHealthMonitor

    # ==================== CONFIGURATION & PRESETS ====================
    elif name == 'PRESETS':
        from .presets import PRESETS
        return PRESETS
    elif name in ['Preset', 'PresetManager', 'load_preset', 'list_presets', 'compare_presets']:
        from .presets import Preset, PresetManager, load_preset, list_presets, compare_presets
        return locals()[name]
    
    # ==================== VALIDATION ====================
    elif name in ['ConfigValidator', 'validate_config']:
        from .validation import ConfigValidator, validate_config
        return locals()[name]

    # ==================== GOVERNANCE (Iron Mind) ====================
    elif name == 'KnowledgeGovernor':
        from .governance import KnowledgeGovernor
        return KnowledgeGovernor

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # ==================== CORE ====================
    'AdaptiveFramework',
    'AdaptiveFrameworkConfig',
    'IntrospectionEngine',
    'PerformanceMonitor',
    'PerformanceSnapshot',
    'CognitiveRegime',

    # ==================== MEMORY ====================
    'UnifiedMemoryHandler',
    'PrioritizedReplayBuffer',
    'AdaptiveRegularization',
    'DynamicConsolidationScheduler',
    'OrthogonalProjector',
    'HolographicVault',

    # ==================== META CONTROLLER ====================
    'MetaController',
    'MetaControllerConfig',
    'GradientAnalyzer',
    'DynamicLearningRateScheduler',
    'CurriculumStrategy',
    'ReptileOptimizer',

    # ==================== PRODUCTION ====================
    'ProductionAdapter',
    'InferenceMode',

    # ==================== CONSCIOUSNESS (V2) ====================
    'ConsciousnessCore',
    'EnhancedConsciousnessCore',
    'EmotionalState',
    'EmotionalSystem',
    'MetaCognition',
    'EpisodicMemory',
    'SelfModel',
    'Personality',
    'Introspection',
    'AdaptiveAwareness',
    'RecursiveGlobalWorkspace',

    # ==================== MOE ====================
    'SparseMoE',
    'HierarchicalMoE',
    'GatingNetwork',
    'ExpertBlock',
    'AdaptiveExpertBlock',

    # ==================== WORLD MODEL ====================
    'WorldModel',
    'JEPAPredictor',

    # ==================== PERCEPTION ====================
    'PerceptionGateway',
    'VisionEncoder',
    'AudioEncoder',
    'ModalityFuser',

    # ==================== ADAPTERS ====================
    'AdapterBank',
    'FiLMAdapter',
    'BottleneckAdapter',

    # ==================== HEALTH MONITOR ====================
    'NeuralHealthMonitor',

    # ==================== PRESETS ====================
    'PRESETS',
    'Preset',
    'PresetManager',
    'load_preset',
    'list_presets',
    'compare_presets',

    # ==================== VALIDATION ====================
    'ConfigValidator',
    'validate_config',

    # ==================== GOVERNANCE ====================
    'KnowledgeGovernor',
]
