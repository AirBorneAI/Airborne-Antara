"""Explicit loss and batch contracts shared by online learning and replay."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, Mapping):
        return {key: tree_map(fn, item) for key, item in value.items()}
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):
            return type(value)(*(tree_map(fn, item) for item in value))
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: tree_map(fn, getattr(value, field.name))
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(
            **{key: tree_map(fn, item) for key, item in vars(value).items()}
        )
    return value


def batch_size(args, kwargs, target=None):
    sizes = []

    def collect(tensor):
        if tensor.ndim:
            sizes.append(tensor.shape[0])
        return tensor

    tree_map(collect, (args, kwargs, target))
    if not sizes or any(size != sizes[0] for size in sizes):
        raise ValueError(
            "Replay requires batch-first tensors with the same batch size; "
            "disable replay for other layouts."
        )
    if sizes[0] == 0:
        raise ValueError("Cannot learn from an empty batch.")
    return sizes[0]


def concatenate(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            if not all(torch.equal(first, item) for item in values):
                raise ValueError("Replay cannot merge different scalar arguments.")
            return first
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        if any(item.keys() != first.keys() for item in values):
            raise ValueError("Replay requires the same keyword structure per batch.")
        return {key: concatenate([item[key] for item in values]) for key in first}
    if isinstance(first, (list, tuple)):
        if any(len(item) != len(first) for item in values):
            raise ValueError("Replay input structures differ.")
        return type(first)(
            concatenate([item[i] for item in values]) for i in range(len(first))
        )
    if any(item != first for item in values):
        raise ValueError("Replay cannot merge different non-tensor arguments.")
    return first


def prediction(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return prediction(output[0])
    raise ValueError("Cannot identify predictions; supply loss_fn(output, target).")


def task_loss(output, target=None, loss_fn=None):
    if loss_fn is not None:
        loss = loss_fn(output, target)
    elif target is None:
        loss = (
            output.get("loss")
            if isinstance(output, Mapping)
            else getattr(output, "loss", None)
        )
        if loss is None:
            raise ValueError(
                "Learning requires targets, a model-provided loss, or loss_fn."
            )
    else:
        logits = prediction(output)
        if not isinstance(target, torch.Tensor):
            raise TypeError("Targets must be a tensor, or supply a custom loss_fn.")
        if target.is_floating_point() and logits.shape == target.shape:
            loss = F.mse_loss(logits, target)
        elif target.dtype == torch.long:
            if (
                logits.ndim > 2
                and target.shape == logits.shape[:-1]
                and target.shape == (logits.shape[0], *logits.shape[2:])
            ):
                raise ValueError("Ambiguous classification axis; supply loss_fn.")
            if logits.ndim == 2 and target.shape == logits.shape[:1]:
                loss = F.cross_entropy(logits, target)
            elif logits.ndim > 2 and target.shape == logits.shape[:-1]:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
                )
            elif logits.ndim > 2 and target.shape == (
                logits.shape[0],
                *logits.shape[2:],
            ):
                loss = F.cross_entropy(logits, target)
            else:
                raise ValueError("Ambiguous classification shapes; supply loss_fn.")
        else:
            raise ValueError(
                "Cannot infer the objective; supply loss_fn(output, target)."
            )
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("The learning objective must return a scalar tensor.")
    if not torch.isfinite(loss):
        raise ValueError("The learning objective returned a non-finite loss.")
    if not loss.requires_grad:
        raise ValueError(
            "The learning objective must be differentiable with respect to model parameters."
        )
    return loss
