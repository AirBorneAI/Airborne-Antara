"""One-line model wrapping with explicit, feedback-driven online updates."""

import copy

import torch

from .core import AdaptiveFramework, AdaptiveFrameworkConfig


class Antara(AdaptiveFramework):
    """Preserve a model's forward interface; learn from feedback via ``learn``.

    ``Antara(model)`` does not invent targets or train during inference.
    The default objective supports class-index cross entropy and same-shape
    floating-point regression. Other tasks use ``loss_fn(output, target)``
    or a model output containing a scalar ``loss``. Replay uses batch-first
    tensor inputs and stores individual examples on CPU.
    """

    def __init__(self, model, *, loss_fn=None, config=None, **overrides):
        if config is None:
            config = AdaptiveFrameworkConfig(
                memory_type="si",
                si_lambda=1.0,
                ewc_lambda=0.0,
                enable_adapters=False,
                enable_introspection=False,
                use_iron_mind=False,
                use_moe=False,
                use_reptile=False,
                use_lookahead=False,
                use_gradient_centralization=False,
                enable_world_model=False,
                enable_consciousness=False,
                compile_model=False,
                feedback_buffer_size=512,
                enable_meta_controller=False,
                use_prioritized_replay=False,
                dream_interval=1,
                dream_batch_size=32,
            )
            first = next(model.parameters(), None)
            if first is not None:
                config.device = str(first.device)
        else:
            config = copy.deepcopy(
                config.to_config() if hasattr(config, "to_config") else config
            )
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown Antara option: {key}")
            setattr(config, key, value)
        if config.feedback_buffer_size < 0:
            raise ValueError(
                "feedback_buffer_size must be nonnegative (zero disables replay)."
            )
        if config.dream_interval < 1 or config.dream_batch_size < 1:
            raise ValueError("dream_interval and dream_batch_size must be positive.")
        super().__init__(model, config=config, loss_fn=loss_fn)

    def forward(self, *args, **kwargs):
        """Return exactly the wrapped model's output structure."""
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Evaluate without updates, restoring every module's previous mode."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            with torch.no_grad():
                return self(*args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training

    def learn(self, *args, targets=None, task_id=0, **kwargs):
        """Update from labelled feedback or an explicitly supplied objective."""
        modes = [(module, module.training) for module in self.modules()]
        try:
            return self.train_step(
                *args, target_data=targets, task_id=task_id, **kwargs
            )
        finally:
            for module, training in modes:
                module.training = training

    def on_task_complete(self, task_id: int):
        """Consolidate observed importance at an explicit task boundary."""
        self.memory.on_task_complete(
            task_id, feedback_buffer=self.feedback_buffer, current_step=self.step_count
        )
        if self.governor:
            self.governor.update_sacred_mask(
                self.memory,
                task_id,
                self.model,
                classes_per_task=self.config.classes_per_task,
            )
        self.apply_cas_protection()
        self._rebuild_restoration_cache()
        self._apply_sacred_restoration()
        self._sanitize_optimizer_state()
        self.sync_lookahead_weights()
        if self.meta_controller:
            self.meta_controller.reset_task()
        self._last_latent = None
