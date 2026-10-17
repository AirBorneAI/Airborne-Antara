# Antara architecture and verification contract

Updated 2026-10-17. This supersedes the [July description](docs/archive/Mandate-2026-07-14.md), preserved as historical context. Its absolute zero-forgetting, consciousness, and SOTA claims are not established results.

## Intended behavior

`Antara(model)` preserves the supplied PyTorch model's forward interface and adds explicit learning from feedback. Do not invent supervision or update weights simply because inference was requested. README.md defines supported inputs and exceptions.

One-line wrapping for continual adaptation remains the ambition. General intelligence, universal zero forgetting, and state-of-the-art performance are research goals, not API guarantees.

## Responsibilities

- wrapper.py: original forward, feedback-driven learn, mode-preserving predict, explicit task consolidation.
- learning.py: recursive tensor handling, batch-first replay contract, scalar task objectives.
- core.py: research orchestration, optional adapters/policies/world predictor, protected-coordinate restoration.
- memory.py: replay support, SI/EWC importance, Linear input-subspace projection.
- checkpoint.py: versioned, atomic, self-contained supported training checkpoints.
- governance.py, meta_controller.py, moe.py: opt-in research mechanisms.
- consciousness_v2.py: experimental telemetry/workspace; no validated general-reasoning claim.

## Update invariants

1. Validate replay inputs and the objective. Empty batches, missing supervision, non-scalar objectives, and non-finite losses fail explicitly.
2. Clear previous graphs/gradients. Retain temporal latents only as detached state and use them only with the stated temporal contract.
3. Forward user inputs/kwargs without leaking framework control arguments.
4. Backpropagate the task objective plus enabled regularizers and valid auxiliary objectives.
5. Unscale active optimizers, transform/project gradients, reapply coordinate masks, and clip.
6. Perform optimizer and optional auxiliary parameter updates.
7. Restore protected parameters after the last auxiliary writer. Record SI path contributions from actual post-update displacement before clearing gradients.
8. Replay detached individual examples with the same objective and protection mechanisms. Accumulate SI after the replay update.
9. At explicit task completion, consolidate importance, install hooks, rebuild restoration references, restore protected coordinates, sanitize optimizer momentum, and synchronize optional slow weights.

Preserve all three protection mechanisms: gradient masking, post-update sacred restoration, and optimizer-state sanitation. Preserving parameters does not prove that every prediction stays unchanged.

Remove only Antara-owned hooks. Task boundaries must not silently reshape/relabel data or infer classifier architecture.

## Mathematical contracts

- Empirical Fisher: `F_i = (1/N) sum_n (d loss_n / d theta_i)^2`. Squaring the mean minibatch gradient is not equivalent. Use bounded, exact per-example autograd.
- SI path: `w_i += -g_i * (theta_i_after - theta_i_before)`. Consolidation divides by displacement squared plus damping. Existing numerical caps remain and must be disclosed in comparisons.
- OGD: `G_projected = G - (G M) M^T` for Linear weight gradients. Basis columns are orthonormal, rank-bounded directions of uncentered inputs. This constrains gradients, not Adam's entire update.
- Policy score: sampled actions are held constant in log_prob; a reparameterized action in its own score can cancel mean gradients. Evaluation does not sample policy actions.
- World prediction: MSE of predicted versus later observed features. Cross-batch prediction requires explicit aligned temporal streams.
- Replay capacity counts individual examples. Preserve input dtype, shape, and resolution.

The cognitive workspace remains unvalidated and disabled by default. Do not substitute a meaningless auxiliary loss and call it reasoning.

## Persistence

A checkpoint must resume the next supported update with weights, optimizer/scaler state, replay, importance/anchors/masks, controller state, counters, and RNG state. Check compatible configuration and architecture. Do not reference an external file as the only copy of memory.

Unsupported experimental combinations fail explicitly. Legacy partial checkpoints cannot resume exact training. Load Python checkpoint state only from trusted sources.

## Verification gates

Before completing changes:

```bash
pytest -v
black --check airborne_antara tests
ruff check airborne_antara tests
python -m build --sdist --wheel
```

Verify actual learning/preservation, not only execution. Cover initial output equivalence, held-out learning, replay isolation, Fisher cancellation, policy gradients, OGD application, protected updates, and checkpoint continuation.

Run `python -m examples.continual_learning` for reproducible retention smoke evidence. Include plain replay; do not attribute its benefit to an unmeasured mechanism.

The release-level research gate remains full 10-task Split-CIFAR-100, targeting ACC >= 0.50 and BWT >= -0.05. Low BWT at chance accuracy or synthetic smoke tests do not satisfy it. Explicitly report if the gate was not run or failed.

A SOTA claim additionally needs several seeds, held-out evaluation, global-label class-incremental inference without task-ID leakage, strong baselines, matched memory/compute budgets, ablations, and result provenance. Report task-incremental experiments separately.

## Historical suites

July_2026 and NeurIPS remain pinned. Do not rewrite old metrics. Some historical tests use different contracts, such as treating Reptile's slow weights as immutable knowledge anchors. Report such failures rather than weakening protection invariants.
