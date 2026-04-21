
"""
Configuration Validation Module
================================

Validates AdaptiveFrameworkConfig for common mistakes and provides helpful error messages.
This module catches config errors early, before they cause cryptic runtime failures.
"""

import logging
from typing import Tuple, List, Optional
from .core import AdaptiveFrameworkConfig


class ConfigValidator:
    """Validates AdaptiveFrameworkConfig for correctness and sanity."""
    
    def __init__(self):
        self.logger = logging.getLogger('ConfigValidator')
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: AdaptiveFrameworkConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a config and return (is_valid, errors, warnings).
        
        Args:
            config: AdaptiveFrameworkConfig to validate
            
        Returns:
            Tuple of (is_valid, error_list, warning_list)
        """
        # Type validation: ensure config is not None
        if config is None:
            return False, ["AdaptiveFrameworkConfig cannot be None"], []
        
        if not isinstance(config, AdaptiveFrameworkConfig):
            return False, [f"Expected AdaptiveFrameworkConfig, got {type(config)}"], []
        
        self.errors = []
        self.warnings = []
        
        self._validate_learning_rates(config)
        self._validate_network_architecture(config)
        self._validate_memory_settings(config)
        self._validate_consciousness_settings(config)
        self._validate_optimization_settings(config)
        self._validate_replay_settings(config)
        self._validate_consolidation_settings(config)
        self._validate_device_settings(config)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_learning_rates(self, config: AdaptiveFrameworkConfig):
        """Check learning rate settings."""
        if config.learning_rate <= 0:
            self.errors.append(f"learning_rate must be > 0, got {config.learning_rate}")
        elif config.learning_rate > 1.0:
            self.warnings.append(f"learning_rate={config.learning_rate} is very high (usually 1e-3 to 1e-2)")
        
        if config.meta_learning_rate <= 0:
            self.errors.append(f"meta_learning_rate must be > 0, got {config.meta_learning_rate}")
        elif config.meta_learning_rate > config.learning_rate:
            self.warnings.append(f"meta_learning_rate ({config.meta_learning_rate}) > learning_rate ({config.learning_rate}), unusual")
        
        if config.weight_adaptation_lr <= 0:
            self.errors.append(f"weight_adaptation_lr must be > 0, got {config.weight_adaptation_lr}")
    
    def _validate_network_architecture(self, config: AdaptiveFrameworkConfig):
        """Check network architecture settings."""
        if config.model_dim <= 0:
            self.errors.append(f"model_dim must be > 0, got {config.model_dim}")
        
        if config.num_layers <= 0:
            self.errors.append(f"num_layers must be > 0, got {config.num_layers}")
        
        if config.num_heads <= 0:
            self.errors.append(f"num_heads must be > 0, got {config.num_heads}")
        elif config.model_dim % config.num_heads != 0:
            self.errors.append(f"model_dim ({config.model_dim}) must be divisible by num_heads ({config.num_heads})")
        
        if config.dropout < 0 or config.dropout >= 1.0:
            self.errors.append(f"dropout must be in [0, 1), got {config.dropout}")
        elif config.dropout > 0.5:
            self.warnings.append(f"dropout={config.dropout} is very high (typically 0.1-0.3)")
    
    def _validate_memory_settings(self, config: AdaptiveFrameworkConfig):
        """Check memory system settings."""
        valid_memory_types = ['ewc', 'si', 'hybrid']
        if config.memory_type not in valid_memory_types:
            self.errors.append(f"memory_type must be one of {valid_memory_types}, got '{config.memory_type}'")
        
        valid_consolidation = ['time', 'surprise', 'hybrid']
        if config.consolidation_criterion not in valid_consolidation:
            self.errors.append(f"consolidation_criterion must be one of {valid_consolidation}, got '{config.consolidation_criterion}'")
        
        if config.consolidation_min_interval <= 0:
            self.errors.append(f"consolidation_min_interval must be > 0, got {config.consolidation_min_interval}")
        
        if config.consolidation_max_interval <= config.consolidation_min_interval:
            self.errors.append(f"consolidation_max_interval ({config.consolidation_max_interval}) must be > consolidation_min_interval ({config.consolidation_min_interval})")
        
        if config.consolidation_surprise_threshold <= 0:
            self.errors.append(f"consolidation_surprise_threshold must be > 0, got {config.consolidation_surprise_threshold}")
    
    def _validate_consciousness_settings(self, config: AdaptiveFrameworkConfig):
        """Check consciousness layer settings."""
        if config.consciousness_buffer_size <= 0:
            self.errors.append(f"consciousness_buffer_size must be > 0, got {config.consciousness_buffer_size}")
        
        if config.novelty_threshold <= 0:
            self.errors.append(f"novelty_threshold must be > 0, got {config.novelty_threshold}")
        
        if config.enable_consciousness and config.consciousness_buffer_size < 100:
            self.warnings.append(f"consciousness_buffer_size={config.consciousness_buffer_size} is small; recommend >= 100 for meaningful statistics")
    
    def _validate_optimization_settings(self, config: AdaptiveFrameworkConfig):
        """Check optimization settings."""
        if config.gradient_clip_norm <= 0:
            self.errors.append(f"gradient_clip_norm must be > 0, got {config.gradient_clip_norm}")
        
        if config.adapter_max_norm <= 0:
            self.errors.append(f"adapter_max_norm must be > 0, got {config.adapter_max_norm}")
        
        if config.warmup_steps < 0:
            self.errors.append(f"warmup_steps must be >= 0, got {config.warmup_steps}")
        
        if config.evaluation_frequency <= 0:
            self.errors.append(f"evaluation_frequency must be > 0, got {config.evaluation_frequency}")
    
    def _validate_replay_settings(self, config: AdaptiveFrameworkConfig):
        """Check replay settings."""
        if config.feedback_buffer_size <= 0:
            self.errors.append(f"feedback_buffer_size must be > 0, got {config.feedback_buffer_size}")
        
        if config.feedback_buffer_size < 128:
            self.warnings.append(f"feedback_buffer_size={config.feedback_buffer_size} is small; recommend >= 128 for stable replay")
        
        if config.use_prioritized_replay:
            if config.replay_priority_temperature <= 0:
                self.errors.append(f"replay_priority_temperature must be > 0, got {config.replay_priority_temperature}")
            elif config.replay_priority_temperature > 1.0:
                self.warnings.append(f"replay_priority_temperature={config.replay_priority_temperature} > 1; will be more uniform (less prioritization)")
    
    def _validate_consolidation_settings(self, config: AdaptiveFrameworkConfig):
        """Check consolidation settings."""
        if config.active_shield_threshold < 0 or config.active_shield_threshold > 1.0:
            self.errors.append(f"active_shield_threshold must be in [0, 1], got {config.active_shield_threshold}")
        
        if config.active_shield_slope <= 0:
            self.errors.append(f"active_shield_slope must be > 0, got {config.active_shield_slope}")
        
        if config.panic_threshold <= 0:
            self.errors.append(f"panic_threshold must be > 0, got {config.panic_threshold}")
        
        if config.active_shield_threshold >= config.panic_threshold:
            self.warnings.append(f"active_shield_threshold ({config.active_shield_threshold}) >= panic_threshold ({config.panic_threshold}); panic mode may never trigger")
    
    def _validate_device_settings(self, config: AdaptiveFrameworkConfig):
        """Check device settings."""
        valid_devices = ['cpu', 'cuda', 'mps']
        device_lower = config.device.lower().split(':')[0]  # Handle 'cuda:0' format
        if device_lower not in valid_devices:
            self.warnings.append(f"device='{config.device}' may not be supported; typically 'cpu', 'cuda', or 'mps'")
    
    def print_report(self, is_valid: bool, errors: List[str], warnings: List[str]):
        """Print a formatted validation report."""
        if is_valid and len(warnings) == 0:
            self.logger.info("✓ Config validation PASSED (all checks OK)")
            return
        
        if errors:
            self.logger.error("=" * 60)
            self.logger.error("CONFIGURATION VALIDATION FAILED")
            self.logger.error("=" * 60)
            for i, error in enumerate(errors, 1):
                self.logger.error(f"  [{i}] ERROR: {error}")
        
        if warnings:
            self.logger.warning("=" * 60)
            self.logger.warning("WARNINGS (Config may work but is unusual)")
            self.logger.warning("=" * 60)
            for i, warning in enumerate(warnings, 1):
                self.logger.warning(f"  [{i}] WARNING: {warning}")
        
        if is_valid:
            self.logger.info("=" * 60)
            self.logger.info("Config is VALID but has WARNINGS—review above")
            self.logger.info("=" * 60)


def validate_config(config: AdaptiveFrameworkConfig, raise_on_error: bool = True):
    """
    Validate a config and optionally raise an exception on error.
    
    Args:
        config: AdaptiveFrameworkConfig to validate
        raise_on_error: If True, raise ValueError on validation errors
        
    Returns:
        Tuple of (is_valid, error_list, warning_list) or just is_valid if raise_on_error=True
        
    Raises:
        ValueError: If raise_on_error=True and validation fails
    """
    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate(config)
    
    validator.print_report(is_valid, errors, warnings)
    
    if not is_valid and raise_on_error:
        error_msg = "\n".join(errors)
        raise ValueError(f"Config validation failed:\n{error_msg}")
    
    return (is_valid, errors, warnings) if not raise_on_error else is_valid


# Example usage
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    
    # Test valid config
    print("Testing valid config...")
    config = AdaptiveFrameworkConfig()
    validate_config(config)
    
    # Test invalid config
    print("\n\nTesting invalid config...")
    bad_config = AdaptiveFrameworkConfig(
        learning_rate=-0.1,  # Invalid
        model_dim=255,  # Not divisible by num_heads (8)
        feedback_buffer_size=50,  # Too small
    )
    try:
        validate_config(bad_config)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # ==================== INTEGRATION TESTS ====================
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS (V8.1 Fixes)")
    print("=" * 60)
    
    from .core import AdaptiveFramework
    
    def test_feedback_buffer_population():
        """Test 1: Verify feedback_buffer is populated during training."""
        print("\n[TEST 1] Feedback Buffer Population...")
        model = nn.Linear(10, 5)
        cfg = AdaptiveFrameworkConfig(
            model_dim=10, 
            enable_consciousness=False, 
            enable_dreaming=False,
            enable_health_monitor=False
        )
        framework = AdaptiveFramework(model, cfg)
        
        # Run 5 training steps
        for _ in range(5):
            x = torch.randn(2, 10)
            y = torch.randn(2, 5)
            framework.train_step(x, target_data=y)
        
        buffer_len = len(framework.feedback_buffer.buffer)
        assert buffer_len == 5, f"Expected 5 items in buffer, got {buffer_len}"
        print(f"   ✓ Buffer has {buffer_len} items (expected: 5)")
        return True
    
    def test_meta_controller_invoked():
        """Test 2: Verify meta_controller.adapt() is called."""
        print("\n[TEST 2] Meta-Controller Invocation...")
        model = nn.Linear(10, 5)
        cfg = AdaptiveFrameworkConfig(
            model_dim=10, 
            enable_consciousness=False, 
            enable_dreaming=False,
            enable_health_monitor=False,
            use_reptile=True
        )
        framework = AdaptiveFramework(model, cfg)
        
        initial_step = framework.meta_controller.step_count
        
        # Run 3 training steps with meta enabled
        for _ in range(3):
            x = torch.randn(2, 10)
            y = torch.randn(2, 5)
            framework.train_step(x, target_data=y, meta_step=True)
        
        final_step = framework.meta_controller.step_count
        assert final_step == initial_step + 3, f"Meta-controller steps: {final_step}, expected {initial_step + 3}"
        print(f"   ✓ Meta-controller step count: {final_step} (advanced by 3)")
        return True
    
    def test_consolidation_triggers():
        """Test 3: Verify memory consolidation can be triggered."""
        print("\n[TEST 3] Memory Consolidation Trigger...")
        model = nn.Linear(10, 5)
        cfg = AdaptiveFrameworkConfig(
            model_dim=10, 
            enable_consciousness=False, 
            enable_dreaming=False,
            enable_health_monitor=False,
            memory_type='hybrid',
            consolidation_min_interval=5,  # Low for testing
            consolidation_max_interval=10
        )
        framework = AdaptiveFramework(model, cfg)
        
        # Populate buffer first
        for _ in range(10):
            x = torch.randn(2, 10)
            y = torch.randn(2, 5)
            framework.train_step(x, target_data=y)
        
        # Manual consolidation call
        initial_counter = framework.memory.consolidation_counter
        framework.memory.consolidate(
            feedback_buffer=framework.feedback_buffer,
            current_step=100,
            z_score=0.0,
            mode='NORMAL'
        )
        
        final_counter = framework.memory.consolidation_counter
        assert final_counter == initial_counter + 1, f"Consolidation counter didn't increment"
        
        # Check penalty is now potentially non-zero (depends on whether omega accumulated)
        penalty = framework.memory.compute_penalty()
        print(f"   ✓ Consolidation triggered (counter: {initial_counter} -> {final_counter})")
        print(f"   ✓ Memory penalty after consolidation: {penalty.item():.6f}")
        return True
    
    def test_dreaming_works():
        """Test 4: Verify dreaming (learn_from_buffer) works with data."""
        print("\n[TEST 4] Dreaming (Replay) Functionality...")
        model = nn.Linear(10, 5)
        cfg = AdaptiveFrameworkConfig(
            model_dim=10, 
            enable_consciousness=False, 
            enable_dreaming=True,
            dream_interval=1,  # Dream every step
            enable_health_monitor=False
        )
        framework = AdaptiveFramework(model, cfg)
        
        # Populate buffer with 15+ items (minimum 10 required for dreaming)
        for _ in range(15):
            x = torch.randn(2, 10)
            y = torch.randn(2, 5)
            framework.train_step(x, target_data=y, enable_dream=False)
        
        buffer_len = len(framework.feedback_buffer.buffer)
        assert buffer_len >= 10, f"Buffer needs 10+ items for dreaming, got {buffer_len}"
        
        # Now trigger dreaming explicitly
        framework.learn_from_buffer(batch_size=8, num_epochs=1)
        print(f"   ✓ Dreaming executed with {buffer_len} items in buffer")
        return True
    
    # Run all integration tests
    tests = [
        test_feedback_buffer_population,
        test_meta_controller_invoked,
        test_consolidation_triggers,
        test_dreaming_works,
    ]
    
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

