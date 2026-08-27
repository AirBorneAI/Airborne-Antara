# 🧮 Mathematical Proof: Autonomic Repair (Health Monitor)

**[⬅ Return to Architecture](../technical/SYNTHETIC_INTUITION.md) | [See Implementation (Code)](../../airborne_antara/health_monitor.py)**

---

## 1. The Stability Problem

Deep learning training is non-stationary. Parameters $\theta$ drift into regions of instability.
Two primary failure modes exist:
1.  **Dead Neurons (ReLU Collapse)**: Activations $a_i \le 0$ for all inputs. Gradients become 0. The neuron never learns again.
2.  **Exploding/Vanishing Gradients**: Norm $||\nabla \theta|| \rightarrow \infty$ or $0$.

## 2. Statistical Profiling (The "Probe")

We model the distribution of activations $A$ and gradients $G$ for each layer $l$ as time-series random variables.

At step $t$, we compute moments:
$$ \mu_G(t) = \beta \mu_G(t-1) + (1-\beta) ||G_t|| $$
$$ \sigma_G^2(t) = \beta \sigma_G^2(t-1) + (1-\beta) (||G_t|| - \mu_G(t))^2 $$

Where $\beta \approx 0.99$ (EMA decay).

## 3. Anomaly Detection (The Z-Score)

To detect instability robustly (independent of layer scale), we compute the Z-Score of the current update:

$$ Z_t = \frac{||G_t|| - \mu_G(t)}{\sigma_G(t) + \epsilon} $$

### Trigger Conditions
*   **Explosion**: If $Z_t > 3.0$ (3 standard deviations), the update is an outlier.
*   **Death**: For ReLU activations, we track the sparsity ratio $S$:
    $$ S = \frac{1}{B \times N} \sum_{i=1}^B \sum_{j=1}^N \mathbb{I}(a_{ij} \le 0) $$
    If $S > 0.95$, the layer is "Dead".

## 4. Autonomic Intervention

When a trigger fires, the Monitor executes an interrupt.

### Protocol A: Re-Initialization (Lazarus)
For dead layers, we re-sample weights weights $W_l$ from the original initialization distribution $\mathcal{D}_{init}$ (e.g., Kaiming He Normal):

$$ W_l \leftarrow W_l \sim \mathcal{N}(0, \sqrt{2/n_{in}}) $$

This "resurrects" the neuron, giving it a new random projection to attempt learning.

### Protocol B: Dynamic Damping
For high Z-Scores ($Z > 3$), we damp the Learning Rate $\eta$:

$$ \eta_{new} = \eta_{old} \times \frac{1}{1 + \ln(Z_t)} $$

This is a non-linear decay that aggressively suppresses explosions while allowing minor fluctuations.
