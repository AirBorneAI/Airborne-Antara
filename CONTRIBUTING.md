# 🤝 AirBorne Engineering Contribution Guidelines

Welcome to the **AirBorne** and **AirBorne-HRS** engineering ecosystem! We are thrilled to have you contribute to our autonomous intelligence systems, machine learning architectures, and cloud platforms.

To maintain our standard of engineering excellence, please review these guidelines before submitting code.

---

## 🏛️ 1. Core Principles

- **Code is Read More than Written:** Strive for extreme clarity, modularity, and concise documentation.
- **Defensive & Robust Design:** Handle edge cases, non-stationary distributions, and invalid inputs explicitly.
- **Zero-Warning Tolerance:** Code submitted must be free of linter errors, deprecation warnings, and type violations.
- **Test-Driven Rigor:** Every bug fix or new feature must be accompanied by comprehensive automated unit tests.

---

## 🚀 2. Getting Started & Development Workflow

### Prerequisites
- **Python:** Version `3.10`, `3.11`, or `3.12`
- **Git:** Version `2.38+`
- **Virtual Environment:** `venv` or `conda`

### Setting up Your Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/AirBorne-HRS/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 3. Upgrade pip and install development dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

---

## 🌿 3. Branching & Commit Conventions

We follow a structured branching strategy based on **Conventional Commits**:

### Branch Naming Convention

| Prefix | Use Case | Example |
| :--- | :--- | :--- |
| `feat/` | New features or algorithmic capabilities | `feat/contrastive-decoding` |
| `fix/` | Bug fixes and runtime repairs | `fix/kv-cache-overflow` |
| `perf/` | Latency, throughput, or memory optimizations | `perf/vectorized-attention` |
| `docs/` | Documentation, guides, or README updates | `docs/access-control-matrix` |
| `ci/` | CI/CD pipelines, workflows, or bots | `ci/codeql-integration` |
| `refactor/` | Code refactoring without behavioral changes | `refactor/modular-evaluator` |

### Commit Message Standard

Commit messages must adhere to the Conventional Commits specification:

```
<type>(<optional scope>): <subject line in imperative mood>

[optional body explaining context and rationale]

[optional footer with issue reference or Co-authored-by trailer]
```

**Examples:**
- `feat(memory): implement universal tensor projection for conv2d layers`
- `fix(attention): resolve off-by-one indexing in rotary embeddings (#42)`
- `docs(security): update disclosure response SLA in SECURITY.md`

---

## 🎨 4. Code Style & Quality Standards

All Python code must strictly pass our automated linting and formatting suite:

```bash
# Run Ruff fast linter
ruff check . --fix

# Format code with Black
black . --check

# Type checking with MyPy
mypy . --ignore-missing-imports
```

### Formatting Guidelines
- Line length limit: **100 characters** (soft limit 120 for mathematical formulas).
- All public functions, classes, and methods must include Google-style docstrings.
- Avoid unnecessary external dependencies; prefer standard library and core framework primitives.

---

## 🧪 5. Testing & Verification

No pull request will be merged without passing unit tests.

```bash
# Run complete test suite with coverage
pytest --cov=. --cov-report=term-missing

# Run quick unit tests
python -m unittest discover -s tests
```

### Test Requirements
- **Unit Tests:** Must test both happy paths and explicit boundary failure cases.
- **Determinism:** Tests must be deterministic and must not rely on unseeded random states or external internet connectivity.

---

## 📋 6. Pull Request (PR) Checklist & Review Lifecycle

Before opening your PR, verify:

- [ ] Branch is rebased onto the latest `main` branch.
- [ ] PR title follows Conventional Commits format (`feat:`, `fix:`, etc.).
- [ ] All automated tests pass locally.
- [ ] Linters (`ruff`, `black`, `mypy`) execute cleanly with 0 errors.
- [ ] Documentation has been updated to reflect the new feature or fix.
- [ ] Sensitive secrets, credentials, or personal keys are not committed.

### Review Lifecycle
1. **Automated CI Validation:** The AirBorne CI/CD gauntlet will validate formatting, security, and test matrices.
2. **Peer Review:** A maintainer will conduct a thorough review within **2 business days**.
3. **Merge:** Merges are performed via Squash & Merge to ensure a clean, linear git history.

---

## 💬 7. Questions & Contact

If you have architectural questions, need guidance, or wish to discuss large RFCs:

- Open a **[GitHub Discussion](https://github.com/AirBorne-HRS/.github/discussions)**
- Email our engineering leads at **[talent@airbornehrs.in](mailto:talent@airbornehrs.in)**

---

```
AIRBORNE PVT. LTD. • BUILT FOR THE FUTURE. ENGINEERED TO LEAD.
```
