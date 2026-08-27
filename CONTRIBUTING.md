# 🤝 AirBorne Engineering Contribution Guidelines & PR Rules

Welcome to the **AirBorne** and **AirBorneAI** engineering ecosystem! We are committed to building reliable, high-performance autonomous intelligence systems, adaptive neural architectures, and enterprise platforms.

To maintain uncompromising standards of engineering excellence and ensure that **no untested or arbitrary code is introduced**, all contributors (both human engineers and AI agents) must strictly adhere to these rules.

---

## 🏛️ 1. Core Engineering Principles

- **Zero "Vibe-Coding" / Zero Unverified Code:** No code may be committed or merged without local test verification. AI agents and human contributors alike must execute unit tests and linters before submitting PRs.
- **Defensive & Robust Design:** Handle non-stationary distributions, edge cases, dimension mismatches, and malformed inputs explicitly.
- **Architectural Integrity:** Never introduce phantom dependencies, unreachable branches, dummy/mock logic in production paths, or undocumented side effects.
- **Code is Read More than Written:** Maintain modular architecture, explicit type annotations, clear docstrings, and comprehensive test coverage.
- **Zero-Warning Tolerance:** Submissions must pass all linters (`ruff`, `black`, `flake8`), type checkers, and security scans (`bandit`, `codeql`) with zero unaddressed warnings.

---

## 🔒 2. Mandatory Quality Gates for Pull Requests

Every Pull Request must satisfy the following checklist before review and merge:

### Gate 1: Problem Definition & Justification
- Every PR must solve an identified issue, implement a validated feature, or optimize a measured bottleneck.
- The PR description must clearly explain **What** changed, **Why** it was needed, and **How** it was verified.

### Gate 2: Automated Unit Testing
- **100% Behavioral Coverage:** Every bug fix must include a regression test that fails without the fix and passes with it.
- **Feature Tests:** Every new module, class, or method must include corresponding unit tests in `tests/`.
- All tests must pass locally:
  ```bash
  pytest -v
  ```

### Gate 3: Code Formatting & Linting
- Format code using Black and verify with Ruff:
  ```bash
  black --check airborne_antara tests
  ruff check airborne_antara tests
  flake8 airborne_antara tests
  ```

### Gate 4: Submodule & Git Tree Integrity
- Submodule pointers (such as `NeurIPS` and `July_2026`) must have valid mappings in `.gitmodules`.
- No stray temporary files, cache files (`.pyc`, `__pycache__`), or personal environment credentials may be tracked.

---

## 🌿 3. Branching & Commit Conventions

We strictly follow the **Conventional Commits** specification:

### Branch Naming Scheme

| Type | Description | Example |
| :--- | :--- | :--- |
| `feat/` | New features or capabilities | `feat/holographic-saliency-pooling` |
| `fix/` | Bug fixes and runtime repairs | `fix/submodule-mapping-neurips` |
| `perf/` | Latency or memory optimization | `perf/vectorized-moe-routing` |
| `docs/` | Documentation or guide updates | `docs/governance-policy-spec` |
| `ci/` | CI/CD pipelines, workflows, or tooling | `ci/test-matrix-python312` |
| `refactor/` | Structural code refactoring | `refactor/modular-memory-vault` |

### Commit Message Structure

```text
<type>(<optional scope>): <short description in imperative mood>

[optional detailed body explaining rationale and approach]

[optional footer(s), e.g. Fixes #123, Co-authored-by: Name <email>]
```

**Examples:**
- `feat(memory): add universal tensor projection for conv and attention layers`
- `fix(core): guard against None gradients during dynamic consolidation (#42)`
- `docs(presets): document real_time and high_throughput configurations`

---

## 🤖 4. AI Agent Contribution Protocol

AI agents contributing to this codebase are bound by the following mandatory operational protocol:

1. **Pre-Implementation Research:** Read existing module contracts, `Mandate.md`, and relevant test suites before editing files.
2. **No Hallucinated Functions / Attributes:** Verify that all imported functions, methods, and tensor operations exist in the current codebase or specified dependencies.
3. **Mandatory Local Execution:** Run the test suite (`pytest`) and verify that all test assertions pass before proposing changes.
4. **Co-Authorship Attribution:** When finishing or co-authoring work started by another contributor, include the `Co-authored-by: Name <email>` trailer in commit messages.
5. **No Mocking of Production Paths:** Never replace actual mathematical transformations with no-op mocks in production modules.

---

## 🚀 5. Local Development Workflow

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/AirBorneAI/Airborne-Antara.git
cd Airborne-Antara

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install in editable mode with development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]

# 4. Run tests before creating a branch
pytest -v

# 5. Create a feature branch
git checkout -b feat/my-improvement

# 6. Verify changes locally
pytest -v
black --check airborne_antara tests
ruff check airborne_antara tests

# 7. Push and open PR
git push origin feat/my-improvement
```

---

## 🛡️ 6. PR Review & Merge Policy

1. **Automated Status Checks:** All GitHub Actions workflows (`⚡ AirBorne-Antara CI/CD Pipeline`, `🛡️ CodeQL Advanced Security Analysis`, and `📋 PR Semantic Title & Quality Gate`) must be green.
2. **Maintainer Approval:** Every PR requires code review approval from an AirBorne engineering lead.
3. **Squash or Rebase Merge:** Branch history must be clean, linear, and meaningful.
