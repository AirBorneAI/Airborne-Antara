import pytest
import torch
import torch.nn as nn
from airborne_antara.memory import (
    UnifiedMemoryHandler,
    PrioritizedReplayBuffer,
    AdaptiveRegularization,
    DynamicConsolidationScheduler,
    OrthogonalProjector,
    HolographicVault,
)

def test_replay_buffer():
    buffer = PrioritizedReplayBuffer(capacity=50)
    for i in range(10):
        buffer.add(
            state=torch.randn(1, 10),
            target=torch.randn(1, 2),
            loss=0.5 + 0.1 * i,
        )
    assert len(buffer) == 10
    sample = buffer.sample(batch_size=4)
    assert len(sample) == 4

def test_unified_memory_handler():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    handler = UnifiedMemoryHandler(model=model, memory_capacity=100)
    assert handler is not None
    
    # Register an experience and compute consolidation loss
    x = torch.randn(2, 8)
    y = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    handler.update_importance()
    reg_loss = handler.compute_penalty()
    assert isinstance(reg_loss, torch.Tensor)
