---
name: antara-mandate
description: >
  Read the Airborne-Antara architectural mandate before modifying any code in the
  airborne_antara package. Use when working on the continual learning framework,
  memory system, MoE routing, governance, or consciousness modules.
---

# Antara Mandate Skill

## When to Trigger

This skill activates whenever an agent is asked to:
- Modify any file in `airborne_antara/`
- Debug training failures, BWT degradation, or forgetting
- Add new components to the framework
- Change formulas, penalties, or loss functions
- Modify the NeurIPS test suite

## Pre-Flight Protocol

1. **Read the Mandate:** Open and read `Mandate.md` at the workspace root:
   ```
   Mandate.md
   ```
   Pay special attention to:
   - §4 Forward Pass (tensor shapes, hook pipeline)
   - §5 Training Step (loss aggregation order)
   - §7 Iron Mind (sacred mask construction)
   - §14 All Formulas (exact equations)
   - §19 Audit Findings (known bugs to avoid)

2. **Identify Impact Zone:** Before making changes, trace the data flow from your target file through the full system:
   - `core.py` changes → check memory, governance, moe impact
   - `memory.py` changes → check governance mask construction
   - `governance.py` changes → check CAS hooks, sacred restoration
   - `moe.py` changes → check expert-class affinity, gate training

3. **Verify Protection Stack:** Any change MUST preserve the 3-layer protection:
   - Layer 1: CAS gradient hooks (grad → 0 for sacred)
   - Layer 2: Sacred restoration (param[mask] = anchor[mask])
   - Layer 3: Optimizer sanitization (zero momentum for sacred)

## Post-Change Protocol

1. **Update Mandate.md:** If you changed any formula, data flow, or module interface, update the corresponding section.

2. **Run Verification:**
   ```bash
   # Quick sanity check (single task)
   python -c "from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig; print('Import OK')"
   ```

3. **Full Gauntlet:** For structural changes, run the Phase I curriculum and verify:
   - BWT ≥ -0.05 (backward transfer / forgetting)
   - ACC ≥ 0.50 (average accuracy)

## Known Critical Bugs (from Audit)

- **A-01:** `dream_batch_size=0` disables all replay. Set to 32 if dreaming is needed.
- **A-03:** `_steps_since_task_start` double-increments. Lines 1326 AND 1577-1581.
- **A-06:** `PrioritizedReplayBuffer` returns `None` entries before buffer is full.
- **A-09:** Teacher distillation passes list instead of unpacked args to SparseMoE.

## File Reference Map

| Module | File | Key Classes |
|--------|------|-------------|
| Orchestrator | `core.py` | `AdaptiveFramework`, `AdaptiveFrameworkConfig` |
| Memory | `memory.py` | `UnifiedMemoryHandler`, `PrioritizedReplayBuffer` |
| Governance | `governance.py` | `KnowledgeGovernor` |
| MoE | `moe.py` | `SparseMoE`, `HierarchicalMoE`, `GatingNetwork` |
| Consciousness | `consciousness_v2.py` | `EnhancedConsciousnessCore`, `RecursiveGlobalWorkspace` |
| Meta-Learning | `meta_controller.py` | `MetaController`, `ReptileOptimizer` |
| World Model | `world_model.py` | `WorldModel` (I-JEPA predictor) |
| Perception | `perception.py` | `PerceptionGateway` |
| Adapters | `adapters.py` | `AdapterBank` |
