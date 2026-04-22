# API Reference (v8.0 "Sentient" Edition)

## Overview

`airborne_antara` provides a unified cognitive wrapper for any PyTorch model. The V8.0 "Sentient" edition is structured around 4 interactive pillars:

1. **AdaptiveFramework**: The primary entry point. A sentient wrapper for your neural substrate.
2. **Consciousness V2**: Recursive Global Workspace for higher-order deliberation.
3. **Unified Memory**: Multi-layered consolidation (SI/EWC/OGD) to prevent forgetting.
4. **Perception Gateway**: High-resolution multi-modal fusion (Vision/Audio/Text).
5. **Autonomic Health**: Real-time neural stability and mixed-precision (AMP) orchestration.

---

## AdaptiveFramework (The Cognitive Wrapper)

The main class that transforms a static model into an adaptive, cognitive agent.

### Initialization

```python
from airborne_antara import AdaptiveFramework, PRESETS
import torch

# 1. Define your base model (any PyTorch model)
my_model = YourPyTorchModel()

# 2. Choose a Sentient Preset
config = PRESETS.production()

# 3. Wrap it
agent = AdaptiveFramework(my_model, config=config, device="cuda")
```

**Parameters:**

* `model` (nn.Module): The base neural substrate to be upgraded.
* `config` (AdaptiveFrameworkConfig): Configuration or Preset.
* `device` (str or torch.device, optional): Device (default: auto-detect).
* `module_configs` (dict, optional): Specific overrides for Memory, Consciousness, etc.

**Parameters:**

* `config` (AdaptiveFrameworkConfig): Configuration
* `device` (str or torch.device, optional): Device to use (default: auto-detect)

### Core Methods

#### `observe(vision=None, audio=None, text=None, **kwargs)`

The primary gateway for sensory input. Automatically fuses multi-modal data.

```python
metrics = agent.observe(vision=img_batch, text=token_batch)
# Returns SentientMetrics with uncertainty, entropy, and thought traces.
```

#### `train_step(data_dict, target_data=None)`

Executes a single SENTIENT training cycle (Perception -> Consciousness -> Memory -> Health).

```python
results = agent.train_step(my_data, targets)
# Automatically handles AMP, gradient clipping, and memory consolidation.
```

#### `evaluate(data_dict, target_data)`

Evaluates cognitive performance without modifying internal weights.

```python
eval_metrics = agent.evaluate(val_data, val_targets)
```

#### `save_sentient_state(path)`

Saves the entire cognitive state, including model weights, memory importance, and consciousness history.

```python
agent.save_sentient_state("sentient_agent_v1.pth")
```

#### `learn_from_buffer(batch_size, num_epochs)`

Learn from experience replay buffer.

```
metrics = framework.learn_from_buffer(batch_size=32, num_epochs=5)
```

#### `save_checkpoint(path)`

Save model checkpoint.

```
framework.save_checkpoint("checkpoint.pt")
```

#### `load_checkpoint(path)`

Load model from checkpoint.

```
framework.load_checkpoint("checkpoint.pt")
```

#### `get_metrics()`

Get summary of collected metrics.

```
summary = framework.get_metrics()
# {'total_steps': 1000, 'avg_recent_loss': 0.05, ...}
```

## IntrospectionEngine

State monitoring component (Recursive State Monitoring).

Provides:

* Internal representation analysis
* Uncertainty estimation
* Performance calibration diagnostics

**Note:** Introspection is an algorithmic monitoring technique, not evidence of consciousness.

## MetaController

Advanced meta-learning orchestrator for the optimization cycle.

### Initialization

```
from airborne_antara import MetaController, MetaControllerConfig

config = MetaControllerConfig(
    base_lr=1e-3,                       # Base learning rate
    inner_loop_steps=5,                 # MAML inner loop steps
    meta_learning_rate=1e-4,            # Meta-learning rate
    curriculum_start_difficulty=0.1,    # Initial curriculum difficulty
)

controller = MetaController(framework, config)
```

### Methods

#### `adapt(loss, gradients, performance_metrics)`

Execute adaptation step in optimization cycle.

```
adaptation_metrics = controller.adapt(
    loss=current_loss,
    performance_metrics={'loss_improvement': 0.01}
)
```

**Parameters:**

* `loss` (float): Current loss value
* `gradients` (dict, optional): Gradient information
* `performance_metrics` (dict, optional): Performance metrics

**Returns:**

* `metrics` (dict): Adaptation metrics

#### `get_summary()`

Get adaptation history summary.

```
summary = controller.get_summary()
# {
#   'step_count': 100,
#   'current_lr': 1e-3,
#   'curriculum_difficulty': 0.35,
#   'gradient_history': [...],
#   'lr_history': [...]
# }
```

## GradientAnalyzer

Analyzes gradient statistics for adaptation decisions.

```
from airborne_antara import GradientAnalyzer

analyzer = GradientAnalyzer(model, config)

# Analyze current gradients
stats = analyzer.analyze()
# {'mean_norm': 0.5, 'max_norm': 2.1, 'variance': 0.3, 'sparsity': 0.2}

# Check if learning rate should be reduced
should_reduce = analyzer.should_reduce_lr()
```

## DynamicLearningRateScheduler

Adapts learning rate based on loss landscape and gradients.

```
from airborne_antara import DynamicLearningRateScheduler

scheduler = DynamicLearningRateScheduler(optimizer, config)

# Adapt learning rate each step
new_lr = scheduler.step(loss=0.05, gradient_stats=grad_stats)

# Get current learning rate
current_lr = scheduler.get_lr()
```

## CurriculumStrategy

Implements curriculum learning: easy-to-hard task progression.

```
from airborne_antara import CurriculumStrategy

curriculum = CurriculumStrategy(config)

# Get current difficulty (0.0 = easy, 1.0 = hard)
difficulty = curriculum.get_difficulty()

# Update difficulty based on learning progress
curriculum.step(loss_improvement=0.02)

# Sample curriculum-adjusted batch
perturbed_batch, targets = curriculum.sample_task_batch(batch, batch_targets)
```

## ProductionAdapter

Simplified API for production inference with optional online learning.

### Initialization

```
from airborne_antara import ProductionAdapter, InferenceMode

# Static inference (no learning)
adapter = ProductionAdapter.load_checkpoint(
    "model.pt",
    inference_mode=InferenceMode.STATIC
)

# Online learning (immediate updates)
adapter = ProductionAdapter.load_checkpoint(
    "model.pt",
    inference_mode=InferenceMode.ONLINE
)

# Buffered learning (batch updates)
adapter = ProductionAdapter.load_checkpoint(
    "model.pt",
    inference_mode=InferenceMode.BUFFERED
)
```

### Methods

#### `predict(input_data, update, target)`

Run inference with optional online learning.

```
# Inference only
output = adapter.predict(new_data)

# Inference with online learning
output = adapter.predict(new_data, update=True, target=new_target)
```

**Parameters:**

* `input_data` (torch.Tensor): Input batch
* `update` (bool): Whether to perform online learning
* `target` (torch.Tensor, optional): Target for online learning

**Returns:**

* `output` (torch.Tensor): Model predictions

#### `get_uncertainty(input_data)`

Get uncertainty estimates for predictions.

```
uncertainty = adapter.get_uncertainty(new_data)
```

**Returns:**

* `uncertainty` (torch.Tensor): Uncertainty values

#### `save_checkpoint(path)`

Save current model state.

```
adapter.save_checkpoint("model_updated.pt")
```

#### `get_metrics()`

Get performance metrics.

```
metrics = adapter.get_metrics()
```

## InferenceMode

Enum for inference modes.

**Values:**

* `InferenceMode.STATIC`: No learning (pure inference)
* `InferenceMode.ONLINE`: Immediate learning from each sample
* `InferenceMode.BUFFERED`: Batched learning from recent samples

## Terminology Mapping (V8.0 Sentient)

We embrace cognitive terminology while maintaining mathematical rigor:

| Sentient Term | Mathematical/Control-Theoretic Implementation |
| :--- | :--- |
| **Consciousness** | Recursive Global Workspace (System 2 Bottleneck) |
| **Synthetic Intuition** | I-JEPA Latent World Model Forecasting |
| **Sentience** | Emergent state of recursive state analysis & health |
| **Neural Shivering** | Autonomic soft-noise jitter for stability |
| **Memory** | Unified importance manifold (SI + EWC + OGD) |
| **Perception** | Multi-modal fusion with Dynamic Positional Interpolation |
| **Thought Trace** | Decoded trajectory of the recursive workspace bottleneck |
| **Foresight** | Minimized World Model Error (I-JEPA prediction) |
| **Neural Health** | AMP scaling + Gradient clipping + Saliency monitoring |

## Examples

### Basic Training

```
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import torch

config = AdaptiveFrameworkConfig(model_dim=128, num_layers=4)
framework = AdaptiveFramework(config)

X = torch.randn(100, 10, 128)
y = torch.randn(100, 10, 128)

for epoch in range(5):
    metrics = framework.train_step(X, y)
    print(f"Loss: {metrics['loss']:.4f}")

framework.save_checkpoint("model.pt")
```

### Production with Online Learning

```
from airborne_antara import ProductionAdapter, InferenceMode

adapter = ProductionAdapter.load_checkpoint(
    "model.pt",
    inference_mode=InferenceMode.ONLINE
)

# In production loop
for data_batch in incoming_data_stream:
    predictions = adapter.predict(data_batch, update=True, target=labels)
    uncertainty = adapter.get_uncertainty(data_batch)
  
    # Use predictions and uncertainty in application...
```

### Advanced Meta-Learning

```
from airborne_antara import AdaptiveFramework, MetaController

framework = AdaptiveFramework(config)
controller = MetaController(framework)

for epoch in range(10):
    metrics = framework.train_step(X_train, y_train)
  
    # Adaptive learning rate, curriculum, etc.
    adaptation = controller.adapt(
        loss=metrics['loss'],
        performance_metrics={'loss_improvement': 0.01}
    )
  
    print(f"LR: {adaptation['learning_rate']:.2e}")
```

## Troubleshooting

### High Memory Usage

* Reduce `model_dim` or `num_layers` in config
* Reduce `feedback_buffer_size`
* Use smaller `batch_size`

### Loss Not Decreasing

* Increase `learning_rate`
* Check input data normalization
* Verify target data quality

### GPU Out of Memory

* Use `device='cpu'` to fall back to CPU
* Reduce batch size
* Reduce model dimensions

## Citation

If you use airborne-antara in your research or applications:

```
@software{airbornehrs2026,
  title = {airborne-antara v1.1.1 Adaptive: Production-Ready Adaptive Meta-Learning Framework},
  author = {Singh, Suryaansh Prithvijit},
  year = {2026},
  version = {1.1.1},
  url = {https://github.com/Ultron09/Mirror_mind}
}
```

For more examples, see `examples/` directory or check the GitHub repository.
