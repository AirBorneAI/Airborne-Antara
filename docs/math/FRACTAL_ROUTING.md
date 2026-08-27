# 🧮 Mathematical Proof: Fractal Routing (Hierarchical MoE)

**[⬅ Return to Architecture](../technical/SYNTHETIC_INTUITION.md) | [See Implementation (Code)](../../airborne_antara/moe.py)**

---

## 1. The Sparse Objective

Standard neural networks are dense: $y = f(Wx + b)$. Every weight participates in every inference.
Fractal Routing seeks to activate only a sparse subset of weights $\theta_{active} \subset \theta_{total}$ while maintaining the capacity of $\theta_{total}$.

## 2. Gating Dynamics

Let $x \in \mathbb{R}^d$ be the input vector.
Let $\{E_1, ..., E_N\}$ be a set of $N$ expert networks.
Let $G(x): \mathbb{R}^d \rightarrow \mathbb{R}^N$ be the gating network.

The output $y$ is the weighted sum of experts:
$$ y = \sum_{i=1}^N G(x)_i E_i(x) $$

### Top-K Sparsity
To ensure efficiency, we enforce sparsity by keeping only the top $k$ values of $G(x)$:

$$ G(x) = \text{TopK}(\text{Softmax}(W_g x + \text{Noise}), k) $$

Where the noise term (Gaussian) ensures exploration during training:
$$ \text{Noise} \sim \mathcal{N}(0, \frac{1}{softplus(W_{noise} x)}) $$

## 3. Hierarchical Decomposition (The Fractal Step)

A single layer of $N=1000$ experts creates a massive gating bottleneck. AirborneHRS uses a **Bi-Level Hierarchy**.

### Level 1: Domain Routing
Router $R_1$ selects a Domain Cluster $C_j$:
$$ p(C_j | x) = \text{Softmax}(W_{R1} x)_j $$

### Level 2: Expert Routing
Router $R_{2,j}$ selects an Expert $E_{m}$ within Cluster $C_j$:
$$ p(E_m | x, C_j) = \text{Softmax}(W_{R2,j} x)_m $$

The effective probability of selecting Expert $m$ is:
$$ P(E_m) = p(C_j | x) \times p(E_m | x, C_j) $$

This reduces the routing complexity from $O(N)$ to $O(\sqrt{N})$.

## 4. Load Balancing Loss

To prevent "Expert Collapse" (where one expert takes all the load), we add an auxiliary loss $\mathcal{L}_{load}$:

$$ \mathcal{L}_{load} = N \sum_{i=1}^N f_i \cdot P_i $$

Where $f_i$ is the fraction of samples assigned to expert $i$, and $P_i$ is the average gating probability for expert $i$. This minimizes the coefficient of variation of expert load.
