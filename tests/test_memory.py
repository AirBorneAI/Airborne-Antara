import torch
import torch.nn as nn
from airborne_antara.memory import (
    UnifiedMemoryHandler,
    PrioritizedReplayBuffer,
)


def test_replay_buffer():
    buffer = PrioritizedReplayBuffer(capacity=50)
    for i in range(10):
        buffer.add(
            snapshot={"input": torch.randn(1, 10), "target": torch.randn(1, 2)},
            importance=0.5 + 0.1 * i,
        )
    assert len(buffer.buffer) == 10
    sample = buffer.sample_batch(batch_size=4)
    assert len(sample) == 4


def test_unified_memory_handler():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    handler = UnifiedMemoryHandler(models=model)
    assert handler is not None

    # Register an experience and compute consolidation loss
    x = torch.randn(2, 8)
    y = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(x), y)
    before = handler.before_step_snapshot()
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    handler.accumulate_path(before)
    assert sum(v.abs().sum() for v in handler.omega_accum.values()) > 0
    reg_loss = handler.compute_penalty()
    assert isinstance(reg_loss, torch.Tensor)
