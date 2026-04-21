"""
Integration Guide: Human-Like Self-Awareness in MirrorMind
===========================================================

This guide shows how to integrate the state-of-the-art self-awareness 
framework into your existing MirrorMind wrapper.

The self-awareness system provides:
1. Metacognitive understanding of model knowledge
2. Adaptive learning based on confidence
3. Automatic focus area identification
4. Human-like self-improvement planning
5. Sample importance weighting for better learning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .self_awareness_v2 import (
    HumanLikeSelfAwarenessWrapper,
    MetacognitiveState,
    ConfidenceSignal,
    AdaptiveLearningController,
    SelfImprovementPlanner
)


# ============================================================================
# INTEGRATION PATTERN 1: Wrapper Around Existing Model
# ============================================================================

class MirrorMindWithSelfAwareness:
    """
    Enhanced MirrorMind framework with built-in self-awareness.
    
    Usage:
        model = create_base_model()
        aware_framework = MirrorMindWithSelfAwareness(model)
        
        # Training loop
        for epoch in range(epochs):
            for batch in train_loader:
                output = aware_framework.forward(batch['input'])
                loss = criterion(output, batch['target'])
                
                # Update awareness
                aware_framework.observe(
                    output, batch['target'],
                    domain_id=batch['domain'],
                    input_data=batch['input']
                )
                
                # Get adaptive learning rate
                adaptive_lr = aware_framework.get_adaptive_lr()
                
                loss.backward()
                optimizer.step()
                
                # Periodically get awareness insights
                if step % 100 == 0:
                    insights = aware_framework.get_awareness_insights()
                    print(f"Learning phase: {insights['phase']}")
                    print(f"Focus areas: {insights['focus_areas']}")
    """
    
    def __init__(self, model: nn.Module, buffer_size: int = 10000):
        self.base_model = model
        self.self_awareness = HumanLikeSelfAwarenessWrapper(model, buffer_size)
        self.step_count = 0
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through base model"""
        return self.base_model(x, **kwargs)
    
    def observe(self,
                output: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> ConfidenceSignal:
        """Observe prediction and update awareness"""
        self.step_count += 1
        return self.self_awareness.observe(output, target, **kwargs)
    
    def get_adaptive_lr(self, domain_id: Optional[str] = None, base_lr: float = 1e-3) -> float:
        """Get adaptive learning rate based on confidence"""
        adaptive_multiplier = self.self_awareness.compute_adaptive_lr(domain_id) / self.self_awareness.learning_controller.base_lr
        return base_lr * adaptive_multiplier
    
    def get_awareness_state(self) -> MetacognitiveState:
        """Get current metacognitive state"""
        return self.self_awareness.get_awareness_state()
    
    def get_awareness_insights(self) -> Dict[str, Any]:
        """Get actionable insights from self-awareness"""
        state = self.get_awareness_state()
        recommendations = self.self_awareness.get_learning_recommendations()
        
        return {
            'phase': state.phase.name,
            'confidence': state.global_confidence,
            'competence': state.global_competence,
            'direction': state.learning_direction,
            'focus_areas': state.prioritized_improvements,
            'bottlenecks': state.current_bottlenecks,
            'learning_rate_adjustment': recommendations['learning_rate_multiplier'],
            'exploration_ratio': recommendations['exploration_ratio']
        }
    
    def get_sample_importance(self,
                             output: torch.Tensor,
                             target: torch.Tensor,
                             **kwargs) -> float:
        """Get importance weight for a sample (for priority sampling)"""
        return self.self_awareness.compute_sample_importance(output, target, **kwargs)
    
    def get_improvement_plan(self, horizon: int = 1000) -> Dict[str, Any]:
        """Get learning plan for next N steps"""
        return self.self_awareness.get_learning_plan(horizon)
    
    def print_report(self) -> str:
        """Print detailed self-awareness report"""
        return self.self_awareness.print_awareness_report()


# ============================================================================
# INTEGRATION PATTERN 2: Modify Training Loop
# ============================================================================

def training_loop_with_awareness(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device: Optional[torch.device] = None,
    num_epochs: int = 10,
    enable_adaptive_lr: bool = True,
    enable_priority_sampling: bool = True,
    log_frequency: int = 50
):
    """
    Modified training loop that uses self-awareness for adaptive learning.
    
    Features:
    - Adaptive learning rate based on confidence
    - Priority sampling (hard + OOD examples first)
    - Automatic focus area adaptation
    - Periodic awareness reporting
    """
    
    # Wrap model with self-awareness
    aware_model = MirrorMindWithSelfAwareness(model)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # For priority sampling
    sample_importance_weights = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_data = batch['input'].to(device)
            target = batch['target'].to(device)
            domain_id = batch.get('domain', 'unknown')
            
            # Forward pass
            output = aware_model.forward(input_data)
            loss = criterion(output, target)
            
            # Update self-awareness
            confidence_signal = aware_model.observe(
                output, target,
                domain_id=domain_id,
                input_data=input_data
            )
            
            # Get adaptive learning rate (if enabled)
            if enable_adaptive_lr:
                base_lr = 1e-3
                adaptive_lr = aware_model.get_adaptive_lr(domain_id, base_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = adaptive_lr
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track sample importance (for priority sampling)
            if enable_priority_sampling:
                importance = aware_model.get_sample_importance(
                    output, target,
                    domain_id=domain_id,
                    input_data=input_data
                )
                sample_importance_weights.append(importance)
            
            epoch_loss += loss.item()
            
            # Periodic logging
            if batch_idx % log_frequency == 0:
                insights = aware_model.get_awareness_insights()
                print(f"[Epoch {epoch}, Batch {batch_idx}]")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Phase: {insights['phase']}")
                print(f"  Confidence: {insights['confidence']:.3f}")
                print(f"  Focus: {insights['focus_areas']}")
                print(f"  Learning Rate: {adaptive_lr:.2e}" if enable_adaptive_lr else "")
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Print awareness report every N epochs
        if (epoch + 1) % 5 == 0:
            aware_model.print_report()
            
            # Get improvement plan
            plan = aware_model.get_improvement_plan(horizon=len(train_loader) * 5)
            print(f"\nLearning Plan for next 5 epochs:")
            print(f"  Primary Focus: {plan['primary_focus']}")
            print(f"  Milestones: {plan['estimated_milestones']}")
    
    return aware_model, sample_importance_weights


# ============================================================================
# INTEGRATION PATTERN 3: Custom Forward Hook
# ============================================================================

class SelfAwarenessHook:
    """
    PyTorch hook that injects self-awareness into model forward passes.
    
    Usage:
        model = YourModel()
        hook = SelfAwarenessHook(model)
        
        output = model(input)  # Self-awareness is automatically tracked
        
        awareness = hook.get_awareness_state()
    """
    
    def __init__(self, model: nn.Module):
        self.awareness_wrapper = HumanLikeSelfAwarenessWrapper(model)
        self.last_output = None
        self.last_input = None
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks on model layers"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self._make_hook(name))
    
    def _make_hook(self, layer_name: str):
        """Create a forward hook for a layer"""
        def hook(module, input, output):
            # Track this layer's activation patterns
            if hasattr(self, 'layer_activations'):
                self.layer_activations[layer_name] = output.detach()
        return hook
    
    def observe_batch(self,
                     output: torch.Tensor,
                     target: torch.Tensor,
                     **kwargs):
        """Observe a batch and update awareness"""
        return self.awareness_wrapper.observe(output, target, **kwargs)
    
    def get_awareness(self) -> MetacognitiveState:
        """Get current awareness state"""
        return self.awareness_wrapper.get_awareness_state()


# ============================================================================
# INTEGRATION PATTERN 4: Multi-Task Learning with Self-Awareness
# ============================================================================

class MultiTaskSelfAwareLearner:
    """
    Multi-task learning that uses self-awareness to balance task importance.
    
    Automatically focuses on tasks where confidence is low.
    """
    
    def __init__(self, model: nn.Module, task_names: list):
        self.aware_model = MirrorMindWithSelfAwareness(model)
        self.task_names = task_names
        self.task_losses = {task: [] for task in task_names}
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returns outputs for all tasks"""
        outputs = self.aware_model.forward(x)
        return outputs
    
    def compute_adaptive_task_weights(self) -> Dict[str, float]:
        """
        Compute per-task learning weight based on competence.
        
        Tasks with low competence get higher weight (more learning).
        """
        weights = {}
        state = self.aware_model.get_awareness_state()
        
        for task in self.task_names:
            competence = state.confidence_by_domain.get(task, 0.5)
            # Inverse: low competence → high weight
            weight = 1.0 - competence
            weights[task] = weight
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def backward_with_adaptive_weights(self,
                                      task_losses: Dict[str, torch.Tensor]):
        """Backprop with adaptive task weighting"""
        weights = self.compute_adaptive_task_weights()
        
        total_loss = sum(weights[task] * loss for task, loss in task_losses.items())
        return total_loss


# ============================================================================
# QUICK START EXAMPLE
# ============================================================================

def quick_start_example():
    """
    Minimal example showing self-awareness integration.
    """
    # Step 1: Create or load your model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Step 2: Wrap with self-awareness
    aware_model = MirrorMindWithSelfAwareness(model)
    
    # Step 3: Training loop with awareness
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(1000):
        # Generate dummy batch
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        
        # Forward + backward
        output = aware_model.forward(x)
        loss = criterion(output, y)
        
        # Update awareness
        aware_model.observe(output, y, domain_id='mnist')
        
        # Get adaptive learning rate
        adaptive_lr = aware_model.get_adaptive_lr('mnist', base_lr=1e-3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Periodic insights
        if step % 200 == 0 and step > 0:
            insights = aware_model.get_awareness_insights()
            print(f"\nStep {step}:")
            print(f"  Confidence: {insights['confidence']:.3f}")
            print(f"  Phase: {insights['phase']}")
            print(f"  Focus: {insights['focus_areas']}")
    
    # Final report
    print("\n" + "="*60)
    aware_model.print_report()


if __name__ == "__main__":
    quick_start_example()
