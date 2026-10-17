"""Self-contained training checkpoints for the supported online learner.

Checkpoints contain Python state; only load files from a trusted source.
Construct the same model, configuration, and loss callback before loading.
"""

import copy
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

MEMORY_FIELDS = (
    "omega_accum",
    "omega",
    "anchor",
    "sacred_mask",
    "task_omega_snapshots",
    "fisher_dict",
    "opt_param_dict",
    "saturation_level",
    "total_tasks",
    "last_consolidation_step",
    "consolidation_counter",
)
LEARNER_FIELDS = (
    "step_count",
    "reward_baseline",
    "loss_history",
    "regime",
    "_current_task_id",
    "_prev_train_task_id",
    "_steps_since_task_start",
    "_last_task_id_seen",
    "_controller_previous_loss",
    "_last_loss_val",
    "_last_latent",
    "lookahead_step",
    "slow_weights",
    "telemetry_buffer",
)
OPTIMIZERS = (
    "optimizer",
    "adapter_optimizer",
    "meta_optimizer",
    "world_model_optimizer",
)


def _fields(obj, names):
    return {
        name: copy.deepcopy(getattr(obj, name)) for name in names if hasattr(obj, name)
    }


def _restore(obj, state):
    for name, value in state.items():
        setattr(obj, name, value)


def _check_supported(learner):
    # These experimental modules have additional mutable state without a stable
    # resume contract. Reject rather than silently writing a partial checkpoint.
    options = (
        "use_moe",
        "enable_consciousness",
        "enable_perception",
        "enable_health_monitor",
        "enable_performance_monitor",
        "use_graph_memory",
        "use_learned_optimizer",
        "compile_model",
    )
    active = [name for name in options if getattr(learner.config, name, False)]
    if active:
        raise ValueError(
            "Exact training checkpoints do not yet support: " + ", ".join(active)
        )


def _controller_state(controller):
    if controller is None:
        return None
    return {
        "root": _fields(controller, ("current_mode", "step_count")),
        "gradient_analyzer": _fields(
            controller.gradient_analyzer, ("gradient_history",)
        ),
        "lr_scheduler": _fields(
            controller.lr_scheduler, ("current_lr", "loss_history", "grad_history")
        ),
        "curriculum": _fields(controller.curriculum, ("current_difficulty",)),
        "reptile": _fields(controller.reptile, ("anchor_weights", "step_counter")),
    }


def save(learner, path):
    _check_supported(learner)
    state = {
        "format_version": 2,
        "config": asdict(learner.config),
        "custom_loss": learner.loss_fn is not None,
        "modules": learner.state_dict(),
        "modes": {name: module.training for name, module in learner.named_modules()},
        "optimizers": {
            name: opt.state_dict()
            for name in OPTIMIZERS
            if (opt := getattr(learner, name, None)) is not None
        },
        "scaler": learner.scaler.state_dict(),
        "memory": _fields(learner.memory, MEMORY_FIELDS),
        "learner": _fields(learner, LEARNER_FIELDS),
        "feedback": _fields(learner.feedback_buffer, ("buffer", "total_seen")),
        "prioritized": (
            _fields(
                learner.prioritized_buffer,
                ("buffer", "importances", "surprises", "ages", "count", "ptr"),
            )
            if learner.prioritized_buffer is not None
            else None
        ),
        "scheduler": _fields(
            learner.consolidation_scheduler, ("last_step", "consolidation_count")
        ),
        "controller": _controller_state(learner.meta_controller),
        "projector": (
            learner.memory.projector.subspaces if learner.memory.projector else None
        ),
        "governor": learner.governor.task_stats if learner.governor else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + "."
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(learner, path):
    _check_supported(learner)
    state = torch.load(path, map_location=learner.device, weights_only=False)
    if state.get("format_version") != 2:
        raise ValueError(
            "Legacy checkpoints omit training state; load their model_state as weights only."
        )
    current = asdict(learner.config)
    mismatches = [
        key
        for key, value in state["config"].items()
        if key != "device" and current.get(key) != value
    ]
    if mismatches:
        raise ValueError("Checkpoint configuration differs: " + ", ".join(mismatches))
    if state["custom_loss"] != (learner.loss_fn is not None):
        raise ValueError("Recreate the same loss_fn before loading this checkpoint.")
    learner.load_state_dict(state["modules"], strict=True)
    for name, optimizer_state in state["optimizers"].items():
        getattr(learner, name).load_state_dict(optimizer_state)
    learner.scaler.load_state_dict(state["scaler"])
    _restore(learner.memory, state["memory"])
    learner.memory.to(learner.device)
    _restore(learner, state["learner"])
    _restore(learner.feedback_buffer, state["feedback"])
    # Replay stays on CPU regardless of checkpoint map_location.
    for snapshot in learner.feedback_buffer.buffer:
        snapshot.to_device("cpu")
    if state["prioritized"] is not None:
        _restore(learner.prioritized_buffer, state["prioritized"])
        for snapshot in learner.prioritized_buffer.buffer:
            snapshot.to_device("cpu")
    _restore(learner.consolidation_scheduler, state["scheduler"])
    if state["controller"] is not None:
        for name, values in state["controller"].items():
            _restore(
                (
                    learner.meta_controller
                    if name == "root"
                    else getattr(learner.meta_controller, name)
                ),
                values,
            )
    if state["projector"] is not None:
        learner.memory.projector.subspaces = state["projector"]
    if state["governor"] is not None:
        learner.governor.task_stats = state["governor"]
    learner._clear_all_internal_caches()
    learner.apply_cas_protection()
    learner._rebuild_restoration_cache()
    for name, module in learner.named_modules():
        module.training = state["modes"][name]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
