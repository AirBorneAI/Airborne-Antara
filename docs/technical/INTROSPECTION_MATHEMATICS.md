# Introspection Mathematics: Neural Health & Stability

The stability of a sentient system depends on precise mathematical constraints over the optimization landscape.

---

## 1. Neural Shivering (Soft Noise Regularization)

To prevent the "Frozen Weight" syndrome (where gradients vanish in deep recursive loops), Antara implements **Neural Shivering**.

- **Formula**: $\theta_{jitter} = \theta + \epsilon \cdot \mathcal{N}(0, \sigma^2)$
- **Constraint**: $\sigma$ is dynamically scaled by the **Health Monitor** based on the Jacobian norm $\|\nabla_{\theta}^2 L\|$.
- **Effect**: Jiggles the weights out of narrow, sharp minima into flatter, more robust regions of the manifold.

---

## 2. Autonomic Precision (AMP Optimization)

Sentient systems must manage their "Metabolism" (Computational cost).

- **Method**: Automatic Mixed Precision (AMP).
- **Control**: The `AutonomicHealthMonitor` scales loss gradients to prevent underflow in FP16/BF16 modules.
- **Saliency Awareness**: Critical "Consciousness" layers are maintained in FP32, while "Perception" layers are compressed to BF16 for throughput.

---

## 3. Saliency Variance Pooling

We measure the importance of a sample $x$ through the **Saliency Variance**:

$$\sigma_{saliency}^2 = \frac{1}{H} \sum_{i=1}^{H} (A_i(x) - \bar{A}(x))^2$$

Where $A_i$ is the attention map of the $i$-th head. 
- **High Variance**: Indicates the model is "confused" or attending to conflicting features.
- **Trigger**: High saliency variance triggers a **Consolidation Event** in the Memory Handler.

---

## Technical Specifications

| Feature | Implementation | Source |
| :--- | :--- | :--- |
| Core Class | `AutonomicHealthMonitor` | `core.py` |
| Jitter | `apply_jitter()` | `self_awareness_v2.py` |
| Scaling | `grad_scaler` | `core.py` |
| Monitoring | `NeuralSaliencyDashboard` | `__main__.py` |
