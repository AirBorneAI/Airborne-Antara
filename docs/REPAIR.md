# Antara repair record — 2026-10-17

The original goal is viable as an interface: wrap a PyTorch model once, then update it from feedback while retaining useful experience. A wrapper cannot infer correctness or create a useful objective for every task. This revision makes that boundary explicit and repairs the core update path.

## What changed

- Added `Antara(model)`: original forward output, explicit `learn`, mode-preserving `predict`, and task consolidation without architecture-specific classifier edits or copying an unused teacher.
- Defaults use a single backbone, AdamW, uniform example-level reservoir replay, and SI. Experimental architecture transformations and optimization heuristics require configuration.
- Inputs/targets are cloned as detached CPU examples. Replay no longer stores entire minibatches as single capacity units or changes image resolution. Keyword inputs and nested tensor structures are preserved.
- A shared loss contract handles classification/regression, user objectives, and model-provided losses. Invalid and detached objectives fail explicitly.
- Replay now measures SI displacement after its optimizer step. Online SI also observes the final protected update.
- Empirical Fisher averages squared per-example gradients. Opposing examples no longer cancel before squaring.
- Linear gradient projection is used during online and replay updates. Basis collection handles token positions and keyword inputs, retains constant-feature directions, and respects rank/capacity.
- Policy actions are detached in the score-function estimator. Inference does not sample policy actions; output logits are not clipped.
- Reptile receives the correct memory reference and prefixed mask keys. Gradient statistics are captured before cleanup. Protected coordinates are restored after auxiliary updates, and task boundaries reset controller state.
- Cross-batch world prediction requires explicit temporal alignment. Detached latent state survives between valid steps; an inactive predictor is not updated merely by incidental weight decay.
- Antara-owned hooks are tracked separately from user hooks.
- Presets convert to the complete configuration schema. Existing root tests were corrected to use actual public APIs, alongside new behavioral regressions.
- Version 2 checkpoints include supported training state and RNGs in one atomic file. Incomplete experimental combinations are rejected explicitly.
- CI no longer hides failed tests or lint checks. Black formatting and unused-import cleanup across existing package files were necessary to make those checks enforceable.
- README and Mandate now distinguish supported behavior, research goals, and unverified historical claims. The previous Mandate is archived unchanged.

## Verification

Environment: Python 3.14.7, PyTorch 2.14.1+cu130, CPU execution with two OpenMP threads. Tests/linters were installed in an isolated temporary dependency directory; no system administrator access was required.

- Root suite: **42 passed**. Includes initial-output equivalence, held-out learning, CNN/GRU/Transformer compatibility, dictionary/namespace outputs, model-provided objectives, deterministic inference, replay isolation, SI, Fisher, policy gradients, OGD application, protected coordinates, world prediction, and exact next-update checkpoint continuation.
- Black: passed for `airborne_antara tests`.
- Ruff: passed for `airborne_antara tests` and the new experiment.
- Source distribution and wheel: built successfully.
- Pinned July component/integration suite: **20 passed, 1 failed**. Its Reptile test manually sets current parameters to 3.5 and slow weights to 1.5, then expects a direct controller call to overwrite protected values with 1.5. The repaired controller leaves protected coordinates alone; the framework's final restoration uses the authoritative memory anchor. The root regression exercises the complete update with replay, Lookahead, Reptile, and optimizer-state sanitation. The historical submodule was left unchanged.

## Learning and retention smoke experiment

Reproduction:

```bash
python -m examples.continual_learning --seeds 17 29 43 --steps 50 \
  --output results/continual_smoke.json
```

Two synthetic tasks introduce four Gaussian classes. Training and testing use separate examples; inference uses the global four-class head and receives no task ID. Each method starts from the same weights and sees the same new-data minibatches. Replay methods retain at most 128 individual examples and take additional updates.

- Fine-tuning: final accuracy **92.48% ± 6.92 percentage points**, BWT **−14.97 ± 13.88 percentage points**.
- Uniform replay: final accuracy **99.93% ± 0.05 percentage points**, BWT **0.00**.
- Antara replay plus SI: final accuracy **99.93% ± 0.05 percentage points**, BWT **0.00**.

Values are mean and population standard deviation across three seeds. Plain replay matches the combined learner here. This verifies that feedback and retention work on a learnable held-out problem; it does not establish an advantage for SI or state-of-the-art performance. The comparison is not compute-matched.

## Remaining research work and limits

The full Split-CIFAR-100 release gate is **not validated**. The historical benchmark launcher was attempted and stopped during imports: seaborn was absent; after installing it in the isolated dependency directory, pandas was also absent. Inspection also shows that `verify_v18.py` monkey-patches forwarding, memory masks, and Reptile, so its results would not directly certify the repaired wrapper. No CIFAR accuracy/BWT result is claimed for this revision.

Before a performance release, use a benchmark that calls the current API without runtime patches, preserves global labels, evaluates held-out data without oracle task IDs, compares strong baselines at matched memory and compute budgets, and reports several seeds. The existing ACC >= 0.50 and BWT >= −0.05 targets remain unmet verification gates, not reported results.

The cognitive workspace still lacks a validated learning objective. MoE, perception, graph memory, learned optimization, and health/performance weight editing remain experimental; exact resume is not supported for them. OGD protects Linear gradients, not all layer types or Adam's final updates. GPU execution, distributed/sharded models, and all possible custom module interfaces were not validated by this CPU repair suite.

There is now a working, tested foundation for the one-line learner. “ULTRON” as a general autonomous intelligence remains a larger research objective requiring measurable capabilities and evidence beyond wrapping a model.
