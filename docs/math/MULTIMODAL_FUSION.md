# 🧮 Mathematical Proof: Cross-Modality Attention (XMA)

**[⬅ Return to Architecture](../technical/SYNTHETIC_INTUITION.md) | [See Implementation (Code)](../../airborne_antara/perception.py)**

---

## 1. The Fusion Problem

Classical fusion methods (late fusion via concatenation) fail to capture interaction *between* features (e.g., matching a "barking sound" to the "dog visual").
AirborneHRS V2.0.0 implements **Early Fusion via Shared Attention**, projecting all sensory data into a unified manifold $\mathcal{Z}$.

## 2. Modality Encoding

### Vision (ViT-based)
Input Image $I \in \mathbb{R}^{H \times W \times C}$ is split into $N$ patches of size $P \times P$:
$$ x_{vision} = [x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos} $$
Where $E$ is the linear patch projection and $E_{pos}$ are learned position embeddings.

### Audio (Spectral-Transformer)
Input Spectrogram $S \in \mathbb{R}^{F \times T}$ is treated as a temporal sequence of frequency vectors:
$$ x_{audio} = \text{Transformer}(\text{Linear}(S)) + E_{pos} $$

## 3. The Unified Workspace (XMA)

To fuse disparate topologies (2D Spatial vs 1D Temporal), we introduce **Learnable Modality Tokens** $T_m$:

$$ T_{vision}, T_{audio}, T_{text} \in \mathbb{R}^{D} $$

These tokens act as "anchors" for the attention mechanism.

### The Join Operation
We define the global context sequence $C$ as the concatenation of all active streams, prepended by their anchors:

$$ C = [T_{vis}, x_{vis}^1...x_{vis}^N, T_{aud}, x_{aud}^1...x_{aud}^T] $$

### Cross-Modality Attention
We perform Self-Attention on this unified sequence. This allows a visual patch to attend directly to an audio frame:

$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$
where $Q=K=V=C$.

This results in a fused representation $Z$ where every token contains context from all other modalities.
$$ Z = \text{LayerNorm}(C + \text{Attention}(C)) $$

## 4. Latent Alignment

Because $x_{vision}$ and $x_{audio}$ share the same Transformer weights in the fusion layer, the model learns to align their semantic spaces automatically during the self-supervised pre-training (World Modeling).
The "Surprise" signal propagates back through $Z$, forcing the encoders to extract features that are globally consistent.
