# Mirror Mind / Airborne-Antara — Agent Rules & Code of Conduct

## Mandatory Pre-Flight Checklist

Before writing or modifying ANY code in `airborne_antara/` or `NeurIPS/`, every agent MUST:

1. Read [Mandate.md](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Airborne-Antara/Mandate.md) in full.
2. Understand the data flow section (§4–§5) before modifying `core.py`.
3. Understand the protection stack (§7) before touching `governance.py`, `memory.py`, or any mask/anchor logic.
4. Understand the MoE routing (§8) before touching `moe.py`.
5. Verify that any formula changes are reflected back in `Mandate.md` §14.

---

## 🚫 Zero "Random Code" & Technical Integrity Policy

- **No Untested Commits:** Never submit changes without running `pytest -v` locally.
- **No Hallucinated Functions:** Verify exact symbol names and arguments against the active codebase.
- **No Placeholder Logic:** Implement complete, production-grade logic. No dummy `pass` methods.
- **Never bypass the 3-layer protection stack** (CAS hooks, Sacred Restoration, Optimizer Sanitization).
- **Never introduce cross-step tensor references** — all training steps must be graph-isolated.
- **Never modify sacred weights** — any code path that writes to `param.data[mask]` where mask is sacred is forbidden unless it's `_apply_sacred_restoration()`.
- **Always test with Phase I gauntlet** after structural changes. BWT ≥ -0.05 is non-negotiable.

---

## Auto-Update Protocol

If you change any architectural component (formula, data flow, module interface), update the corresponding section of `Mandate.md` in the **same commit**. Mandate.md is the single source of truth.

---

## Code Style & Safety

- Use `torch.no_grad()` for all non-differentiable operations.
- Explicit `.to(device)` for all cross-device tensor operations.
- Guard all optional features with `getattr(self, 'feature', None)`.
- Follow Conventional Commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
