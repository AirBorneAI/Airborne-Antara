# Antara: a feedback-driven PyTorch wrapper

`Antara(model)` adds explicit online learning, bounded experience replay, and continual-learning memory to an existing `torch.nn.Module`.

The model still needs a learning signal: labels, a custom objective, or a loss returned by the model. Inference alone cannot establish whether a prediction was correct.

## One-line wrapping

```python
from airborne_antara import Antara

model = Antara(your_pytorch_model)
output = model(inputs)                   # original forward output and structure
metrics = model.learn(inputs, targets=labels)
prediction = model.predict(inputs)       # eval + no_grad; restores previous modes
```

Complete example:

```python
import torch
from airborne_antara import Antara

model = Antara(torch.nn.Linear(8, 3))
x = torch.randn(16, 8)
y = torch.randint(3, (16,))
model.learn(x, targets=y)
model.on_task_complete(0)                # optional explicit task boundary
model.save_checkpoint("antara.pt")

restored = Antara(torch.nn.Linear(8, 3))
restored.load_checkpoint("antara.pt")
```

Install with `pip install -e .` from this checkout; development tools use `pip install -e '.[dev]'`. Requires Python 3.10+ and PyTorch 2.3+. The root `requirements.txt` includes historical demo dependencies and is not required for the wrapper.

## Inputs, outputs, and objectives

- Positional and keyword inputs are forwarded. Forward preserves the original output structure. Learning supports tensor, tuple, dictionary, and objects exposing `logits` or `loss`.
- Class-index targets use cross entropy; same-shape floating targets use mean squared error. Unambiguous token-last and channels-first classification layouts are supported.
- Other tasks use `loss_fn(output, target)`, returning one finite scalar tensor. Without targets, learning requires that callback or an output containing a scalar loss.
- Replay requires batch-first tensors with consistent batch sizes. It stores individual examples as detached CPU copies without resizing them. Set `feedback_buffer_size=0` for other layouts and supply a suitable objective.
- Ordinary `model(...)` follows PyTorch mode/autograd semantics. Use `predict` for evaluation. Custom model randomness and stateful behavior remain the model's responsibility.

```python
model = Antara(backbone, loss_fn=my_loss, feedback_buffer_size=1024)
model.learn(images, attention_mask=mask, targets=labels, task_id=0)
```

Defaults use AdamW, uniform reservoir replay, and Synaptic Intelligence. Each learning call takes one update on new feedback and, when memory is available, one replay update. Task IDs group replay and consolidation; predict does not require them.

Replay stores actual input examples: choose capacity for their size and sensitivity. No universal memory or speed multiplier is claimed.

## Experimental components

`AdaptiveFramework` and presets remain for research compatibility. Its historical forward returns diagnostics alongside model output; use Antara for the model-preserving interface.

MoE copies, adapters, introspection, world prediction, Reptile, Lookahead, gradient noise, and cognitive telemetry are optional and disabled by default in `Antara(model)`. Presets can enable them and substantially change compute and behavior.

OGD currently projects **Linear weight gradients** using observed input subspaces. Adam's coordinate scaling and momentum prevent this from guaranteeing orthogonal parameter updates or zero forgetting. Protected-coordinate restoration preserves selected parameters, not every prediction.

World prediction requires `(predictions, features)` outputs. Three-dimensional features mean `[batch, time, feature]`. Two-dimensional cross-batch prediction requires `temporal_world_model=True` and aligned trajectories; shuffled batches are not temporal data.

The module named “consciousness” remains experimental telemetry. Its workspace has no validated task-learning objective in this wrapper. It is not evidence of consciousness, reasoning ability, or AGI.

## Checkpoints

Version 2 checkpoints are atomic, self-contained files containing model/auxiliary weights, optimizer/scaler state, replay, importance/anchors/masks, supported controller state, counters, and RNG state. Reconstruct the same model/configuration before loading. Supply custom loss callbacks again with the same semantics.

Tests verify an identical next CPU update after reload, including replay and protected coordinates. Cross-device or nondeterministic GPU execution need not be bitwise identical.

Exact resume currently rejects MoE, cognitive telemetry, perception, graph memory, health/performance editing, learned optimizers, and compiled models because their extra state lacks a complete resume contract. Legacy checkpoints support explicit weights-only migration, not exact training continuation. Only load files you trust.

## Evidence and limits

```bash
python -m examples.continual_learning --output results/continual_smoke.json
```

This experiment uses separate train/test examples, a global four-class head, three fixed seeds, and no task ID at inference. It compares fine-tuning, replay, and replay plus SI. Replay takes additional optimizer steps. This is a synthetic correctness check, not a compute-matched or state-of-the-art benchmark.

Historical outputs do not certify this revision. Earlier headline accuracy and overhead claims have been removed because this checkout does not establish them. Split-CIFAR-100, multiple seeds, matched budgets, ablations, and held-out real data are still required before a competitive claim.

See [repair notes](docs/REPAIR.md) and the [architecture contract](Mandate.md).

## Verification

```bash
pytest -v
black --check airborne_antara tests
ruff check airborne_antara tests
python -m build --sdist --wheel
```

CI enforces tests, formatting, and linting. Security scanning is explicitly advisory. Research submodules are pinned separately and retain historical protocols.
