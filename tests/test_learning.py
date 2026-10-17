"""Behavioral regressions for the feedback-driven wrapper and memory updates."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from airborne_antara import Antara
from airborne_antara.core import IntrospectionEngine
from airborne_antara.learning import task_loss
from airborne_antara.memory import UnifiedMemoryHandler


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)


def test_one_line_wrap_preserves_model_and_prediction():
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(5, 3)
    expected = model(x).detach()
    wrapped = Antara(model)
    assert wrapped.model is model
    torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)
    assert wrapped.adapter_bank is None
    assert wrapped.meta_controller is None


def test_feedback_improves_held_out_accuracy():
    model = Antara(nn.Linear(2, 2), learning_rate=0.05, feedback_buffer_size=64)
    x = torch.randn(128, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    test_x = torch.randn(256, 2)
    test_y = (test_x[:, 0] + test_x[:, 1] > 0).long()
    before = F.cross_entropy(model.predict(test_x), test_y)
    for _ in range(50):
        metrics = model.learn(x, targets=y)
    after = model.predict(test_x)
    assert F.cross_entropy(after, test_y) < before * 0.45
    assert (after.argmax(-1) == test_y).float().mean() > 0.94
    assert metrics["loss"] > 0
    assert model.step_count == 50


def test_inference_is_deterministic_and_preserves_modes_and_weights():
    base = nn.Sequential(nn.Linear(3, 8), nn.Dropout(0.8), nn.Linear(8, 2))
    wrapped = Antara(base)
    base[0].eval()
    modes = [m.training for m in wrapped.modules()]
    state = copy.deepcopy(wrapped.state_dict())
    x = torch.randn(4, 3)
    torch.testing.assert_close(wrapped.predict(x), wrapped.predict(x), rtol=0, atol=0)
    assert modes == [m.training for m in wrapped.modules()]
    for name, value in wrapped.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


class KeywordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x, *, mask, scale=1.0, **kwargs):
        assert not kwargs, "Antara internals leaked into the model interface"
        return {"logits": self.fc(x * mask) * scale, "metadata": "original"}


def test_keyword_inputs_dictionary_output_and_replay():
    wrapped = Antara(KeywordModel())
    x, mask, y = torch.randn(4, 3), torch.ones(4, 3), torch.tensor([0, 1, 0, 1])
    assert wrapped(x, mask=mask)["metadata"] == "original"
    wrapped.learn(x, mask=mask, scale=2.0, targets=y)
    stored = wrapped.feedback_buffer.buffer[0]
    assert stored.input_kwargs["scale"] == 2.0
    assert stored.input_kwargs["mask"].shape == (1, 3)
    wrapped.learn_from_buffer(batch_size=4)


@pytest.mark.parametrize("kind", ["cnn", "rnn", "transformer"])
def test_model_families_forward_and_learning(kind):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = {
                "cnn": lambda: nn.Conv2d(1, 2, 3, padding=1),
                "rnn": lambda: nn.GRU(4, 4, batch_first=True),
                "transformer": lambda: nn.TransformerEncoderLayer(
                    4, 2, 8, dropout=0, batch_first=True
                ),
            }[kind]()

        def forward(self, x):
            return self.layer(x)

    model = Model()
    x = torch.randn(3, 1, 8, 8) if kind == "cnn" else torch.randn(3, 5, 4)
    wrapped = Antara(
        model,
        loss_fn=lambda out, target: (out[0] if isinstance(out, tuple) else out)
        .square()
        .mean(),
    )
    result = wrapped(x)
    assert (
        isinstance(result, tuple) if kind == "rnn" else isinstance(result, torch.Tensor)
    )
    wrapped.learn(x)
    wrapped.learn_from_buffer(batch_size=2)


def test_model_provided_loss():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            out = self.fc(x)
            return {"loss": out.square().mean(), "logits": out}

    wrapped = Antara(Model())
    wrapped.learn(torch.randn(4, 2))
    wrapped.learn_from_buffer()


def test_invalid_objective_does_not_change_parameters():
    wrapped = Antara(nn.Linear(2, 2), loss_fn=lambda out, _: out.mean() * float("nan"))
    before = copy.deepcopy(wrapped.model.state_dict())
    with pytest.raises(ValueError, match="non-finite"):
        wrapped.learn(torch.randn(3, 2))
    for name, value in wrapped.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_missing_learning_signal_fails_explicitly():
    wrapped = Antara(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="requires targets"):
        wrapped.learn(torch.randn(3, 2))


def test_detached_loss_is_not_accepted_as_learning():
    wrapped = Antara(nn.Linear(2, 2), loss_fn=lambda out, _: out.detach().mean())
    with pytest.raises(ValueError, match="differentiable"):
        wrapped.learn(torch.randn(3, 2))


@pytest.mark.parametrize(
    "options",
    [{"feedback_buffer_size": -1}, {"dream_interval": 0}, {"dream_batch_size": 0}],
)
def test_invalid_replay_configuration_fails_early(options):
    with pytest.raises(ValueError):
        Antara(nn.Linear(2, 2), **options)


def test_parameterless_model_can_be_wrapped():
    wrapped = Antara(nn.Identity())
    x = torch.randn(2, 3)
    torch.testing.assert_close(wrapped(x), x)
    with pytest.raises(ValueError, match="trainable"):
        wrapped.learn(x, targets=x)


def test_replay_counts_examples_and_owns_unmodified_copies():
    wrapped = Antara(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 48 * 48, 2)), feedback_buffer_size=7
    )
    x, y = torch.randn(5, 3, 48, 48), torch.zeros(5, dtype=torch.long)
    original = x.clone()
    wrapped.learn(x, targets=y)
    x.zero_()
    entries = wrapped.feedback_buffer.buffer
    assert len(entries) == 5
    assert entries[0].input_args[0].shape == (1, 3, 48, 48)
    torch.testing.assert_close(torch.cat([s.input_args[0] for s in entries]), original)
    wrapped.learn(x, targets=y)
    assert len(entries) == 7
    assert wrapped.feedback_buffer.total_seen == 10


def test_replay_contributes_to_si_path_integral():
    wrapped = Antara(nn.Linear(2, 2))
    wrapped.learn(torch.randn(4, 2), targets=torch.tensor([0, 1, 0, 1]))
    for value in wrapped.memory.omega_accum.values():
        value.zero_()
    wrapped.learn_from_buffer(batch_size=4)
    assert sum(v.abs().sum() for v in wrapped.memory.omega_accum.values()) > 0


def test_policy_mean_has_score_function_gradient():
    engine = IntrospectionEngine()
    _, action, log_prob = engine(torch.ones(4))
    (-log_prob).backward()
    assert not action.requires_grad
    assert engine.policy_net[-1].bias.grad[:2].abs().sum() > 0
    engine.eval()
    _, first, prob = engine(torch.ones(4))
    _, second, _ = engine(torch.ones(4))
    assert prob is None
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_empirical_fisher_does_not_cancel_opposing_examples():
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    memory = UnifiedMemoryHandler(model, method="ewc")
    buffer = SimpleNamespace(
        buffer=[
            SimpleNamespace(
                input_args=(torch.ones(2, 1),),
                input_kwargs={},
                target=torch.tensor([0, 1]),
                task_id=0,
            )
        ]
    )
    memory._consolidate_ewc_fisher_vectorized(buffer)
    torch.testing.assert_close(
        memory.fisher_dict["m0_weight"], torch.full((2, 1), 0.25)
    )
    assert model.training


def test_user_forward_hooks_survive_framework_reinitialization():
    model = nn.Linear(2, 2)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(True))
    wrapped = Antara(model)
    wrapped._clear_all_hooks()
    wrapped(torch.ones(1, 2))
    assert calls == [True]
    handle.remove()


def test_default_regression_and_token_losses():
    out = torch.randn(2, 3, 5, requires_grad=True)
    target = torch.randint(5, (2, 3))
    torch.testing.assert_close(
        task_loss(out, target), F.cross_entropy(out.flatten(0, 1), target.flatten())
    )
    torch.testing.assert_close(
        task_loss(out, torch.zeros_like(out)), out.square().mean()
    )


@pytest.mark.parametrize("advanced", [False, True])
def test_checkpoint_resumes_identical_next_update(tmp_path, advanced):
    options = dict(learning_rate=0.01, feedback_buffer_size=16)
    if advanced:
        options.update(
            enable_meta_controller=True,
            use_reptile=True,
            reptile_update_interval=2,
            use_lookahead=True,
            use_ogd=True,
            use_prioritized_replay=True,
        )
    learner = Antara(nn.Linear(2, 2), **options)
    x, y = torch.randn(8, 2), torch.tensor([0, 1] * 4)
    for _ in range(3):
        learner.learn(x, targets=y)
    learner.memory.sacred_mask["m0_weight"][0, 0] = True
    learner.memory.anchor["m0_weight"] = learner.model.weight.detach().clone()
    learner.apply_cas_protection()
    learner._rebuild_restoration_cache()
    if advanced:
        learner.memory.projector.update_subspace(
            "", torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        )
    checkpoint = tmp_path / "resume.pt"
    learner.save_checkpoint(checkpoint)
    expected_metrics = learner.learn(x, targets=y, task_id=1)
    expected = copy.deepcopy(learner.model.state_dict())
    expected_importance = copy.deepcopy(learner.memory.omega_accum)
    restored = Antara(nn.Linear(2, 2), **options)
    restored.load_checkpoint(checkpoint)
    assert restored.step_count == 3
    assert len(restored.feedback_buffer.buffer) == 16
    actual_metrics = restored.learn(x, targets=y, task_id=1)
    assert actual_metrics["loss"] == expected_metrics["loss"]
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for name, value in restored.memory.omega_accum.items():
        torch.testing.assert_close(value, expected_importance[name], rtol=0, atol=0)


def test_checkpoint_missing_or_mismatched_fails(tmp_path):
    learner = Antara(nn.Linear(2, 2))
    path = tmp_path / "resume.pt"
    with pytest.raises(FileNotFoundError):
        learner.load_checkpoint(path)
    learner.save_checkpoint(path)
    other = Antara(nn.Linear(2, 2), learning_rate=0.1)
    with pytest.raises(ValueError, match="configuration differs"):
        other.load_checkpoint(path)


def test_ogd_runs_in_training_and_handles_sequence_features():
    learner = Antara(nn.Linear(3, 2), use_ogd=True)
    x = torch.zeros(4, 5, 3)
    x[..., 0] = 1
    learner.learn(x, targets=torch.zeros(4, 5, dtype=torch.long))
    learner.memory._consolidate_ogd_subspaces(learner.feedback_buffer)
    basis = learner.memory.projector.subspaces[""]
    assert basis.shape == (3, 1)
    calls = []
    original = learner.memory.projector.project_gradient

    def checked(name, grad):
        result = original(name, grad)
        calls.append(name)
        torch.testing.assert_close(result @ basis, torch.zeros(2, 1), atol=1e-6, rtol=0)
        return result

    learner.memory.projector.project_gradient = checked
    learner.learn(x, targets=torch.ones(4, 5, dtype=torch.long))
    assert len(calls) >= 2  # Online update and replay both project.


def test_protected_coordinates_survive_reptile_lookahead_and_replay():
    learner = Antara(
        nn.Linear(2, 2),
        enable_meta_controller=True,
        use_reptile=True,
        reptile_update_interval=1,
        use_lookahead=True,
        lookahead_k=1,
    )
    x, y = torch.randn(8, 2), torch.tensor([0, 1] * 4)
    learner.learn(x, targets=y)
    anchor = learner.model.weight.detach().clone()
    learner.memory.anchor["m0_weight"] = anchor.clone()
    learner.memory.sacred_mask["m0_weight"][0] = True
    learner.apply_cas_protection()
    learner._rebuild_restoration_cache()
    for _ in range(3):
        learner.learn(x, targets=1 - y, task_id=1)
    torch.testing.assert_close(learner.model.weight[0], anchor[0], rtol=0, atol=0)
    state = learner.optimizer.state[learner.model.weight]
    assert state["exp_avg"][0].count_nonzero() == 0
    assert state["exp_avg_sq"][0].count_nonzero() == 0


@pytest.mark.parametrize("temporal", [False, True])
def test_world_model_needs_explicit_temporal_pairs(temporal):
    class Features(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x), x

    learner = Antara(
        Features(),
        model_dim=4,
        enable_world_model=True,
        temporal_world_model=temporal,
        feedback_buffer_size=0,
    )
    x, y = torch.randn(3, 4), torch.tensor([0, 1, 0])
    before = copy.deepcopy(learner.world_model.state_dict())
    learner.learn(x, targets=y)
    for name, value in learner.world_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    learner.learn(x + 0.1, targets=y)
    changed = any(
        not torch.equal(v, before[n])
        for n, v in learner.world_model.state_dict().items()
    )
    assert changed == temporal
    assert learner._last_latent.grad_fn is None
    assert not learner._last_latent.requires_grad


def test_task_consolidation_preserves_modes_and_importance():
    learner = Antara(nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2)))
    learner.learn(torch.randn(4, 2), targets=torch.tensor([0, 1, 0, 1]))
    modes = [m.training for m in learner.modules()]
    learner.on_task_complete(0)
    assert modes == [m.training for m in learner.modules()]
    assert sum(v.sum() for v in learner.memory.omega.values()) > 0
    assert not hasattr(learner, "teacher_model")


def test_namespace_output_replay_has_no_graph_references():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return SimpleNamespace(logits=self.fc(x))

    learner = Antara(Model())
    learner.learn(torch.randn(3, 2), targets=torch.tensor([0, 1, 0]))
    output = learner.feedback_buffer.buffer[0].output
    assert output.logits.grad_fn is None
    assert output.logits.shape == (1, 2)
    learner.learn_from_buffer()
