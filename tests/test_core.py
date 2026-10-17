import torch
import torch.nn as nn
from airborne_antara.core import (
    AdaptiveFramework,
    AdaptiveFrameworkConfig,
    CognitiveRegime,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_adaptive_framework_initialization():
    model = SimpleModel()
    config = AdaptiveFrameworkConfig(
        model_dim=20,
        enable_consciousness=False,
        use_moe=False,
        enable_perception=False,
        enable_world_model=False,
    )
    framework = AdaptiveFramework(model, config=config)
    assert framework is not None
    assert framework.model is model


def test_adaptive_framework_forward():
    model = SimpleModel()
    config = AdaptiveFrameworkConfig(
        model_dim=20,
        enable_consciousness=False,
        use_moe=False,
        enable_perception=False,
        enable_world_model=False,
    )
    framework = AdaptiveFramework(model, config=config)
    x = torch.randn(4, 10)
    out, _, _, _ = framework(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 2)


def test_cognitive_regime_enum():
    assert CognitiveRegime.SCRATCH.value == "scratch"
    assert CognitiveRegime.CONTINUOUS.value == "continuous"
