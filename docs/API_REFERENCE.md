# AirborneHRS API Reference (V2.0.0 "Synthetic Intuition")

## 1. Core Framework

### `AdaptiveFramework`
The central cognitive agent wrapper.

**Constructor**:
```python
AdaptiveFramework(
    user_model: nn.Module, 
    config: AdaptiveFrameworkConfig = None, 
    device: str = None
)
```
-   `user_model`: Your standard PyTorch model (CNN, RNN, Transformer, etc.).
-   `config`: Configuration object. Use `AdaptiveFrameworkConfig.production()` for defaults.

**Main Methods**:
-   `train_step(data, target_data) -> metrics`: Performs a cognitive training step (Foresight + Task Learning).
-   `save_checkpoint(path)`: Persists model state + cognitive state (memory, world model).
-   `load_checkpoint(path)`: Restores full state.

### `AdaptiveFrameworkConfig`
Configuration dataclass.

**Factory Method**:
-   `production()`: Returns a config with SOTA defaults enabled (World Model, MoE, Memory, Health Monitor).

**Key Fields**:
-   `enable_world_model` (bool): Active "Intuitive Foresight".
-   `use_hierarchical_moe` (bool): Active Sparse Expert Routing.
-   `use_graph_memory` (bool): Active Relational Memory.
-   `enable_health_monitor` (bool): Active Autonomic Self-Repair.
-   `model_dim` (int): Dimension of the user model's latent representations.

---

## 2. "Synthetic Intuition" Components

### `WorldModel` (airbornehrs/world_model.py)
**[Math Proof ↗](math/PREDICTIVE_SURPRISE.md)**
Implements JEPAPredictor.
-   `forward(z_t, action)`: Predicts next latent state $z_{t+1}$.
-   `compute_surprise(z_pred, z_actual)`: Returns predictive error (MSE) used as intrinsic motivation.

### `HierarchicalMoE` (airbornehrs/moe.py)
**[Math Proof ↗](math/FRACTAL_ROUTING.md)**
Bi-Level Sparse Router.
-   `forward(x)`: Routes input through Domain Router -> Expert Router.
-   `get_active_experts()`: Returns indices of currently active experts.

### `RelationalGraphMemory` (airbornehrs/memory.py)
**[Math Proof ↗](math/SEMANTIC_GRAPH.md)**
Graph-based Associative Memory.
-   `add(snapshot, vector)`: Adds a node to the graph.
-   `retrieve(query_vector, k=3)`: Finds top-K similar past events.
-   `consolidate()`: Prunes weak nodes and merges duplicates.

### `NeuralHealthMonitor` (airbornehrs/health_monitor.py)
**[Math Proof ↗](math/AUTONOMIC_REPAIR.md)**
Autonomic Repair Daemon.
-   `check_health(model)`: Scans for dead neurons (>95% zero activation) and vanishing gradients.
-   `repair(model)`: Re-initializes dead layers or adjusts Learning Rate dynamically.

---

## 3. Usage Examples

**Quick Start**:
```python
config = AdaptiveFrameworkConfig.production()
agent = AdaptiveFramework(model, config)
agent.train_step(x, y)
```

**Custom Tuning**:
```python
config = AdaptiveFrameworkConfig(#we need to change the whole file
    model_dim=512,
    enable_world_model=True,
    world_model_loss_weight=0.5  # Focus 50% on future prediction
)
```
