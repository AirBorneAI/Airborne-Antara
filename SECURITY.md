# Security Policy

This document outlines the security practices, supported versions, and the
responsible disclosure process for this project.  
We take security seriously and appreciate the efforts of the community in
helping keep the ecosystem safe.

---

## Supported Versions

Security updates are provided **only** for the versions listed below.

| Version | Supported |
|-------|-----------|
| 5.1.x | ✅ Yes |
| 5.0.x | ❌ No |
| 4.0.x | ✅ Yes |
| < 4.0 | ❌ No |

If you are using an unsupported version, please upgrade before reporting
security issues.

---

## Reporting a Vulnerability

We encourage **responsible disclosure** of security vulnerabilities.

### ✅ How to Report

Please report vulnerabilities **privately** via email:

**📧 talent@airbornehrs.in**

**Do NOT** open public GitHub issues, discussions, or pull requests for
security-sensitive findings.

---

### 📄 What to Include

To help us evaluate your report quickly, please include:

- A clear description of the vulnerability
- Affected version(s)
- Reproduction steps or proof-of-concept (PoC)
- Potential impact (data exposure, RCE, privilege escalation, etc.)
- Any suggested mitigation (optional but appreciated)

Incomplete reports may take longer to triage.

---

### ⏱ Response Timeline

We aim to follow this timeline:

- **Acknowledgement:** within **48 hours**
- **Initial assessment:** within **5 business days**
- **Status updates:** every **7–10 days** until resolution

Complex issues may require additional time, but we will keep you informed.

---

### 🧠 Disclosure Process

If a vulnerability is **accepted**:
- We will work on a fix or mitigation
- A coordinated disclosure timeline will be agreed upon
- Credit may be given (optional, at your request)

If a vulnerability is **declined**:
- We will provide a clear technical explanation
- False positives or expected behaviors will be documented

---

## Scope

### ✅ In Scope
- Authentication & authorization flaws
- Data leaks or privacy violations
- Remote code execution
- Dependency vulnerabilities affecting runtime security
- Model misuse leading to security compromise (if applicable)

### ❌ Out of Scope
- Denial-of-service via excessive traffic
- Social engineering or phishing
- Issues in unsupported versions
- Non-security bugs or feature requests
- Vulnerabilities requiring physical access

---

## Safe Harbor

We support **good-faith security research**.

If you:
- Follow this policy
- Avoid data destruction or service disruption
- Do not publicly disclose without coordination

We will **not pursue legal action** against you.

---

## Contributions & Security Fixes

- Security fixes should **only** be submitted **after private disclosure**
- Public PRs for security issues without prior coordination may be closed
- Non-sensitive improvements are always welcome

---

## Final Note

Security is a shared responsibility.  
We value researchers, developers, and users who help strengthen this project
through responsible and ethical practices.

Thank you for helping keep this project secure.
