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

__version__ = "0.0.3"
__license__ = "MIT"
__author__ = "Suryaansh Prithvijit Singh"

# Lazy imports to handle circular dependencies and ensure faster startup
def __getattr__(name):
    # ==================== CORE COMPONENTS ====================
    if name in ['AdaptiveFramework', 'AdaptiveFrameworkConfig', 'IntrospectionEngine', 'PerformanceMonitor', 'PerformanceSnapshot']:
        from .core import AdaptiveFramework, AdaptiveFrameworkConfig, IntrospectionEngine, PerformanceMonitor, PerformanceSnapshot
        return locals()[name]
    
    # ==================== MEMORY SYSTEM (Replaces EWC/SI) ====================
    elif name in ['UnifiedMemoryHandler', 'PrioritizedReplayBuffer', 'AdaptiveRegularization', 'DynamicConsolidationScheduler']:
        from .memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, AdaptiveRegularization, DynamicConsolidationScheduler
        return locals()[name]
    
    # ==================== META CONTROLLER ====================
    elif name in ['MetaController', 'MetaControllerConfig', 'GradientAnalyzer', 'DynamicLearningRateScheduler', 'CurriculumStrategy']:
        from .meta_controller import MetaController, MetaControllerConfig, GradientAnalyzer, DynamicLearningRateScheduler, CurriculumStrategy
        return locals()[name]

    # ==================== PRODUCTION ADAPTERS ====================
    elif name in ['ProductionAdapter', 'InferenceMode']:
        from .production import ProductionAdapter, InferenceMode
        return locals()[name]

    # ==================== CONSCIOUSNESS (V2 Backend) ====================
    # Maps 'ConsciousnessCore' to the new V2 implementation automatically
    elif name in ['ConsciousnessCore', 'EnhancedConsciousnessCore', 'EmotionalState', 'EmotionalSystem', 'MetaCognition', 'EpisodicMemory', 'SelfModel', 'Personality', 'Introspection', 'AdaptiveAwareness']:
        from .consciousness_v2 import EnhancedConsciousnessCore, EmotionalState, EmotionalSystem, MetaCognition, EpisodicMemory, SelfModel, Personality, AdaptiveAwareness
        # Alias legacy name to new core
        if name == 'ConsciousnessCore':
            return EnhancedConsciousnessCore
        # Backward-compatible alias to a real exported symbol.
        if name == 'Introspection':
            return MetaCognition
        return locals()[name]

    # ==================== SELF AWARENESS (High-Level Wrapper) ====================
    elif name in ['HumanLikeSelfAwarenessWrapper', 'MetaCognitiveAwarenessEngine', 'MetaCognitiveState', 'MetacognitiveState', 'ConfidenceSignal', 'CompetenceSignal', 'AdaptiveLearningController', 'SelfImprovementPlanner', 'AdaptiveAttentionMechanism', 'OutOfDistributionDetector']:
        from .self_awareness_v2 import HumanLikeSelfAwarenessWrapper, MetaCognitiveAwarenessEngine, MetacognitiveState, ConfidenceSignal, CompetenceSignal, AdaptiveLearningController, SelfImprovementPlanner, AdaptiveAttentionMechanism, OutOfDistributionDetector
        if name == 'MetaCognitiveState':
            return MetacognitiveState
        return locals()[name]
    
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
        
    # ==================== INTEGRATION GUIDES ====================
    elif name in ['MirrorMindWithSelfAwareness', 'MultiTaskSelfAwareLearner']:
        from .integration_guide import MirrorMindWithSelfAwareness, MultiTaskSelfAwareLearner
        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # ==================== CORE ====================
    'AdaptiveFramework',
    'AdaptiveFrameworkConfig',
    'IntrospectionEngine',
    'PerformanceMonitor',
    'PerformanceSnapshot',

    # ==================== MEMORY ====================
    'UnifiedMemoryHandler',
    'PrioritizedReplayBuffer',
    'AdaptiveRegularization',
    'DynamicConsolidationScheduler',

    # ==================== META CONTROLLER ====================
    'MetaController',
    'MetaControllerConfig',
    'GradientAnalyzer',
    'DynamicLearningRateScheduler',
    'CurriculumStrategy',

    # ==================== PRODUCTION ====================
    'ProductionAdapter',
    'InferenceMode',

    # ==================== CONSCIOUSNESS (V2) ====================
    'ConsciousnessCore',              # alias → EnhancedConsciousnessCore
    'EnhancedConsciousnessCore',
    'EmotionalState',
    'EmotionalSystem',
    'MetaCognition',
    'EpisodicMemory',
    'SelfModel',
    'Personality',
    'Introspection',
    'AdaptiveAwareness',

    # ==================== SELF-AWARENESS LAYER ====================
    'HumanLikeSelfAwarenessWrapper',
    'MetaCognitiveAwarenessEngine',
    'MetacognitiveState',
    'MetaCognitiveState',
    'ConfidenceSignal',
    'CompetenceSignal',
    'AdaptiveLearningController',
    'SelfImprovementPlanner',
    'AdaptiveAttentionMechanism',
    'OutOfDistributionDetector',

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

    # ==================== INTEGRATION ====================
    'MirrorMindWithSelfAwareness',
    'MultiTaskSelfAwareLearner',
]
