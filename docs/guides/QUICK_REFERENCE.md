# ANTARA Self-Awareness Framework: Quick Reference

## Import All Components

```python
from airbornehrs import (
    # Main wrapper
    HumanLikeSelfAwarenessWrapper,
    ANTARAWithSelfAwareness,
    
    # Core engines
    MetaCognitiveAwarenessEngine,
    AdaptiveLearningController,
    SelfImprovementPlanner,
    AdaptiveAttentionMechanism,
    OutOfDistributionDetector,
    
    # Data structures
    MetaCognitiveState,
    ConfidenceSignal,
    CompetenceSignal,
    
    # Integration
    MultiTaskSelfAwareLearner,
)
```

## Basic Usage (3 Lines)

```python
wrapper = HumanLikeSelfAwarenessWrapper(model)
wrapper.observe(output, target, domain_id='vision')
state = wrapper.get_awareness_state()  # Get full self-awareness
```

## Key Methods

### HumanLikeSelfAwarenessWrapper

| Method | Returns | Purpose |
|--------|---------|---------|
| `observe(output, target, **kwargs)` | `ConfidenceSignal` | Update awareness with prediction |
| `get_awareness_state()` | `MetaCognitiveState` | Get full self-awareness snapshot |
| `get_learning_recommendations()` | `Dict` | Get learning strategy |
| `get_learning_plan(horizon)` | `Dict` | Plan next N steps |
| `compute_adaptive_lr(domain_id)` | `float` | Get adaptive learning rate |
| `compute_sample_importance(**kwargs)` | `float` | Weight sample for priority sampling |
| `print_awareness_report()` | `str` | Print detailed report |
| `get_awareness_metrics()` | `Dict` | Get key metrics |

### MetaCognitiveAwarenessEngine

| Method | Returns | Purpose |
|--------|---------|---------|
| `observe(prediction, target, ...)` | `ConfidenceSignal` | Observe prediction |
| `get_competence(domain_id)` | `CompetenceSignal` | Get domain competence |
| `get_metacognitive_state()` | `MetaCognitiveState` | Full awareness state |

### AdaptiveLearningController

| Method | Returns | Purpose |
|--------|---------|---------|
| `compute_adaptive_lr(domain_id)` | `float` | Learning rate scaled by confidence |
| `compute_exploration_ratio()` | `float` | Exploration ratio (0-1) |
| `get_learning_recommendation()` | `Dict` | Complete learning recommendations |

### SelfImprovementPlanner

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_learning_plan(horizon)` | `Dict` | Learning plan for N steps |

### AdaptiveAttentionMechanism

| Method | Returns | Purpose |
|--------|---------|---------|
| `compute_sample_importance(...)` | `float` | Importance score [0, 1] |
| `get_feature_importance_weights()` | `Dict` | Per-feature importance |
| `update_feature_importance(gradients)` | `None` | Update from gradient info |

## Output Reference

### ConfidenceSignal
```python
signal = wrapper.observe(output, target)

signal.prediction_confidence      # 0-1: How sure about this?
signal.epistemic_uncertainty      # 0-1: What don't I know?
signal.aleatoric_uncertainty      # 0-1: Data noise
signal.estimated_accuracy         # 0-1: Expected accuracy
signal.prediction_entropy         # 0-inf: Focused or spread?
signal.out_of_distribution        # True/False: Novel sample?
signal.surprise_level             # 0-1: How surprising?
signal.reliability                # 0-1: Overall reliability
```

### CompetenceSignal
```python
competence = engine.get_competence('vision')

competence.domain_id                      # 'vision'
competence.accuracy_estimate              # 0.82
competence.task_difficulty_estimate       # 0.18
competence.mastery_level                  # 0.75
competence.learning_velocity              # 0.05 (improving)
competence.convergence_progress           # 0.78
competence.knowledge_stability            # 0.91
competence.recommendation                 # "consolidate"
```

### MetacognitiveState
```python
state = wrapper.get_awareness_state()

state.phase                       # EXPLORATION / CONSOLIDATION / MASTERY / UNCERTAINTY
state.global_confidence           # 0.82
state.global_competence           # 0.75
state.global_uncertainty          # 0.12
state.learning_direction          # "Consolidate knowledge..."
state.prioritized_improvements    # ['audio', 'language', 'fusion']
state.current_bottlenecks         # ['High variance in audio', ...]
state.capability_gaps             # [('audio', 0.5), ('language', 0.3)]
state.estimated_time_to_mastery   # 250.0 (steps)
state.confidence_by_domain        # {'vision': 0.82, 'language': 0.58, ...}
state.performance_trajectory      # [0.50, 0.55, 0.58, ..., 0.82]
state.knowledge_entropy           # 1.45
```

## Common Patterns

### Pattern 1: Adaptive Learning Rate
```python
for batch in loader:
    output = model(batch['x'])
    loss = criterion(output, batch['y'])
    
    # Update awareness
    wrapper.observe(output, batch['y'], domain_id=batch['domain'])
    
    # Get adaptive LR
    lr = wrapper.compute_adaptive_lr(batch['domain'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    loss.backward()
    optimizer.step()
```

### Pattern 2: Priority Sampling
```python
# Compute weights for all samples
weights = []
for x, y in dataset:
    output = model(x)
    importance = wrapper.compute_sample_importance(output, y)
    weights.append(importance)

# Create weighted sampler
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(dataset))
loader = DataLoader(dataset, sampler=sampler)
```

### Pattern 3: Multi-Task Learning
```python
from airbornehrs import MultiTaskSelfAwareLearner

learner = MultiTaskSelfAwareLearner(model, 
                                   task_names=['task1', 'task2', 'task3'])

for batch in loader:
    outputs = learner(batch['input'])
    
    # Get adaptive weights (weak tasks get higher weight)
    loss = learner.backward_with_adaptive_weights({
        'task1': criterion(outputs['task1'], batch['y1']),
        'task2': criterion(outputs['task2'], batch['y2']),
        'task3': criterion(outputs['task3'], batch['y3'])
    })
    
    loss.backward()
    optimizer.step()
```

### Pattern 4: Learning Phase-Based Strategy
```python
state = wrapper.get_awareness_state()

if state.phase.name == 'EXPLORATION':
    lr = 1e-2          # High learning rate
    exploration = 0.3  # High exploration
    regularization = 0.0001
    
elif state.phase.name == 'CONSOLIDATION':
    lr = 1e-3
    exploration = 0.1
    regularization = 0.001
    
elif state.phase.name == 'MASTERY':
    lr = 1e-4
    exploration = 0.01
    regularization = 0.0001
```

### Pattern 5: Focused Learning
```python
plan = wrapper.get_learning_plan(horizon=1000)

# Focus on primary weak area
primary_focus = plan['primary_focus']
print(f"Focus next on: {primary_focus}")

# Get transfer learning opportunities
for source, target in plan['transfer_learning_opportunities']:
    print(f"Knowledge from {source} can help with {target}")

# Consolidate near-mastery areas
for area in plan['consolidation_areas']:
    print(f"Consolidate knowledge in {area}")
```

### Pattern 6: Periodic Reporting
```python
for step in range(num_steps):
    # ... training step ...
    
    if step % 500 == 0:
        # Get insights
        insights = wrapper.get_learning_recommendations()
        print(f"Phase: {insights['phase']}")
        print(f"Focus: {insights['focus_areas']}")
        print(f"Bottlenecks: {insights['bottlenecks']}")
        
        # Full report every 2000 steps
        if step % 2000 == 0:
            wrapper.print_awareness_report()
```

## Configuration

```python
# Default configuration
engine = MetaCognitiveAwarenessEngine(
    model=model,
    buffer_size=10000,           # History size
    evaluation_window=100,       # Moving average window
    domain_count=10              # Expected domains
)

controller = AdaptiveLearningController(
    awareness_engine=engine,
    base_lr=1e-3,                # Base learning rate
    base_exploration=0.1         # Base exploration ratio
)

# With ANTARAWithSelfAwareness
aware = ANTARAWithSelfAwareness(
    model=model,
    buffer_size=10000            # Awareness history size
)
```

## Debugging & Monitoring

### Check Current Learning Phase
```python
state = wrapper.get_awareness_state()
print(f"Current phase: {state.phase.name}")
# Output: EXPLORATION, CONSOLIDATION, MASTERY, or UNCERTAINTY
```

### Find Weak Areas
```python
plan = wrapper.get_learning_plan()
print(f"Primary focus: {plan['primary_focus']}")
print(f"Secondary focuses: {plan['secondary_focuses']}")
```

### Monitor Knowledge Distribution
```python
state = wrapper.get_awareness_state()
entropy = state.knowledge_entropy
# High entropy (>2.0): knowledge spread out → consolidate
# Low entropy (<1.0): knowledge concentrated → explore
```

### Track Learning Progress
```python
state = wrapper.get_awareness_state()
print(f"Global confidence: {state.global_confidence:.1%}")
print(f"Global competence: {state.global_competence:.1%}")

# Check per-domain progress
for domain, confidence in state.confidence_by_domain.items():
    print(f"{domain}: {confidence:.1%}")
```

### Identify Bottlenecks
```python
state = wrapper.get_awareness_state()
for bottleneck in state.current_bottlenecks:
    print(f"Issue: {bottleneck}")
```

## Performance Tips

1. **For speed**: Increase `buffer_size` for better statistics
2. **For memory**: Decrease `buffer_size` or `evaluation_window`
3. **For accuracy**: Use with priority sampling (importance-weighted batches)
4. **For convergence**: Combine with adaptive learning rates
5. **For multi-task**: Use adaptive task weighting per competence

## Common Mistakes to Avoid

❌ **Don't** forget to call `observe()` after predictions
```python
# Wrong
output = model(x)
loss = criterion(output, y)

# Right
output = model(x)
wrapper.observe(output, y)
loss = criterion(output, y)
```

❌ **Don't** use global domain ID for all tasks
```python
# Wrong
wrapper.observe(output, y, domain_id='default')

# Right
wrapper.observe(output, y, domain_id='vision')  # Specific domain
```

❌ **Don't** ignore learning phase recommendations
```python
# Wrong
optimizer.param_groups[0]['lr'] = 1e-3  # Fixed LR

# Right
lr = wrapper.compute_adaptive_lr(domain_id)
optimizer.param_groups[0]['lr'] = lr  # Adaptive LR
```

## When to Use Each Component

| Component | Use Case |
|-----------|----------|
| `HumanLikeSelfAwarenessWrapper` | Simple, drop-in wrapper |
| `MetaCognitiveAwarenessEngine` | Custom integration |
| `AdaptiveLearningController` | Dynamic learning rates |
| `SelfImprovementPlanner` | Learning planning & milestones |
| `AdaptiveAttentionMechanism` | Priority sampling |
| `OutOfDistributionDetector` | Novel sample detection |
| `ANTARAWithSelfAwareness` | Full integration |
| `MultiTaskSelfAwareLearner` | Multi-task learning |

## Theoretical Concepts

**Learning Phase**: `EXPLORATION → CONSOLIDATION → MASTERY`
- Confidence drives phase transitions
- Different strategies per phase

**Confidence Signal**: Prediction certainty (multiple sources)
- Epistemic: what I don't know
- Aleatoric: data noise
- Overall: prediction reliability

**Competence**: Domain-specific mastery
- Per-domain accuracy tracking
- Convergence progress
- Knowledge stability

**Uncertainty**: Inverse of confidence
- Used for exploration decisions
- Indicates learning frontiers

**Transfer Learning**: Knowledge reuse
- Automatically detected
- Based on domain similarity

**Knowledge Entropy**: Spread of knowledge
- High: spread across domains
- Low: concentrated in few domains

---

**Quick Links**:
- Full docs: `SELF_AWARENESS_DOCS.md`
- Implementation: `SELF_AWARENESS_IMPLEMENTATION.md`
- Integration: `airborne_antara/integration_guide.py`
- Example: `examples/self_awareness_demo.py`

**Version**: 2.0 | **Status**: Production-Ready ✅
