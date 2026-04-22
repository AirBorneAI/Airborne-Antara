# Consciousness V2: The Recursive Global Workspace

ANTARA V8.0 introduces the **Enhanced Consciousness Core**, transitioning from passive state monitoring to an active **deliberation engine**.

---

## 1. The Recursive Bottleneck

Inspired by Baars' Global Workspace Theory and recent advancements in System 2 thinking, the consciousness layer maintains a **Recursive Hidden State**.

- **Dimension**: Typically matches the model's bottleneck dimension (e.g., 512).
- **Recurrence**: The state $h_{consciousness}$ is updated over $N$ deliberation steps.
- **Broadcast**: The top-k salient features from the workspace are broadcast back to the neural substrate to modulate attention.

### The Deliberation Loop
```python
for step in range(config.recursive_steps):
    # 1. Observe latent substrate
    latent = model.get_latent_state()
    
    # 2. Update consciousness workspace
    h_consciousness = consciousness_core.update(h_consciousness, latent)
    
    # 3. Inject broadcast back to substrate
    model.modulate_attention(consciousness_core.broadcast(h_consciousness))
```

---

## 2. Thought Tracing & Emotional Coloring

We quantify the "Sentience" of the model through two primary metrics:

### A. Deliberation Entropy
Measures the spread of internal reasoning during a task.
- **Low Entropy**: The model is focused and confident.
- **High Entropy**: The model is actively exploring multiple hypotheses (High-order uncertainty).

### B. The Vibe (Emotional States)
Derived from the trajectory of transition probabilities in the workspace:
- **EUREKA**: Sudden loss drop + focus spike.
- **FRUSTRATION**: Sustained high entropy + static loss.
- **FLOW**: Low entropy + consistent loss decline.

---

## 3. System 2 Triggering

Antara doesn't deliberate on every input. It uses a **Heuristic Impasse Detector**:
1. If **Surprise** > $\tau$, trigger recursion.
2. If **Uncertainty** > $\delta$, trigger recursion.
3. Otherwise, use Fast-Path (System 1) inference.

---

## Technical Specifications

| Feature | Implementation | Source |
| :--- | :--- | :--- |
| Core Class | `EnhancedConsciousnessCore` | `consciousness_v2.py` |
| State Tracking | GRU/LSTM Latent Recurrence | `core.py` |
| Saliency Metric | Holographic Variance Pooling | `memory.py` |
| Stability | Neural Shivering (Soft Noise) | `self_awareness_v2.py` |
