# 🤖 AirBorne-Antara — Autonomous Agent Rules & Code of Conduct

This document establishes the **mandatory operational rules and technical code of conduct** for all AI coding agents, subagents, and automated assistants operating on the `Airborne-Antara` repository.

---

## 🚫 1. ZERO "RANDOM CODE" POLICY

1. **No Untested Commits:**
   - Never write or propose code without executing local verification.
   - Run `pytest -v` and linters before committing any changes.
2. **No Phantom Imports or Hallucinated Methods:**
   - Always verify that classes, functions, and parameters exist in the active codebase before calling them.
   - Check schemas, kwargs, and tensor dimensions explicitly.
3. **No Dummy/Mock Logic in Production Modules:**
   - Do not replace mathematical models, neural modules, or loss calculations with no-op placeholder functions (e.g., `pass` or `return x` where complex logic was intended).
4. **No Dead or Unreachable Code:**
   - Every conditional branch and fallback must serve a verified scenario. Avoid speculative catch-alls.

---

## 🏛️ 2. Core Architectural Laws

- **Respect the Protection Stack:**
  - Never bypass the 3-layer protection stack (CAS hooks, Sacred Restoration, Optimizer Sanitization in `governance.py` / `memory.py`).
  - Never mutate sacred weights directly (`param.data[mask]` where mask is sacred is forbidden outside of `_apply_sacred_restoration()`).
- **Graph Isolation:**
  - Never introduce cross-step tensor references — all training and adaptation steps must be graph-isolated.
- **Device & Precision Safety:**
  - Always ensure explicit `.to(device)` alignment across tensor operations.
  - Wrap non-differentiable operations in `torch.no_grad()`.
- **Submodule Integrity:**
  - Whenever working with submodules (e.g. `NeurIPS`, `July_2026`), ensure `.gitmodules` mappings are synchronized.

---

## 🧪 3. Mandatory Verification Checklist

Before completing any task or pushing a branch:

```bash
# 1. Run all unit tests
pytest -v

# 2. Check code formatting & linting
black --check airborne_antara tests
ruff check airborne_antara tests

# 3. Verify build packaging
python -m build --sdist --wheel
```

---

## 📝 4. Commit & PR Attribution

- Follow **Conventional Commits** (`feat:`, `fix:`, `perf:`, `docs:`, `ci:`, `refactor:`, `test:`).
- When co-authoring or resolving work started by another developer, always include:
  ```text
  Co-authored-by: Name <email>
  ```
