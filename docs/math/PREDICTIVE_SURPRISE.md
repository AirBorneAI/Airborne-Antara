# 🧮 Mathematical Proof: Predictive Surprise (World Model)

**[⬅ Return to Architecture](../technical/SYNTHETIC_INTUITION.md) | [See Implementation (Code)](../../airborne_antara/world_model.py)**

---

## 1. The Objective Function

The core innovation of AirborneHRS V2.0.0 is the **I-JEPA (Joint-Embedding Predictive Architecture)** World Model. It enables self-supervised learning by training a predictor to approximate the *future latent state* of the system.

We define the total loss $\mathcal{L}_{total}$ as:

$$ \mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{surprise} $$

Where $\mathcal{L}_{surprise}$ is the intrinsic motivation signal.

## 2. Derivation of Surprise

Let $x_t \in \mathbb{R}^D$ be the observed state at time $t$.
Let $E_\theta$ be the Encoder network.
Let $P_\phi$ be the Predictor network.

### Step 1: Latent Embedding
We project the high-dimensional observation into a low-dimensional manifold:
$$ z_t = E_\theta(x_t) $$

### Step 2: Latent Foresight
The predictor estimates the next state $\hat{z}_{t+1}$ based on the current state $z_t$ and action $a_t$:
$$ \hat{z}_{t+1} = P_\phi(z_t, a_t) $$

### Step 3: Divergence Calculation
When $t+1$ arrives, we observe $x_{t+1}$ and encode it:
$$ z_{t+1} = E_\theta(x_{t+1}) $$

The **Surprise** is the squared L2 norm between the prediction and reality:
$$ S_t = || \hat{z}_{t+1} - z_{t+1} ||_2^2 $$

## 3. Optimization Dynamics

The gradients with respect to the predictor parameters $\phi$:
$$ \nabla_\phi \mathcal{L}_{surprise} = 2 (\hat{z}_{t+1} - z_{t+1}) \cdot \nabla_\phi \hat{z}_{t+1} $$

The encoder parameters $\theta$ receive gradients from both the task and the world model, forcing feature representations that are both **discriminative** (good for the task) and **predictive** (good for physics).

---

## 4. Exponential Moving Average (EMA) Integration

To prevent collapse (where $E_\theta$ learns to output a constant vector), we maintain an EMA of the target encoder or use a separate target network. In AirborneHRS, we use an EMA baseline for the Surprise metric itself to normalize intrinsic rewards:

$$ \mu_{error} = \alpha \mu_{error} + (1-\alpha) S_t $$
$$ \text{Normalized Surprise} = S_t - \mu_{error} $$
