"""Reproducible two-task learning/retention smoke test; not a SOTA benchmark.

Run from the repository root:
    python -m examples.continual_learning --output results/continual_smoke.json
"""

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from airborne_antara import Antara


def make_tasks(seed):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    tasks = []
    for task in range(2):
        splits = []
        for count in (128, 512):
            y = torch.arange(2).repeat_interleave(count // 2) + task * 2
            x = centers[y] + torch.randn(count, 2, generator=generator) * 0.6
            splits.append((x, y))
        tasks.append(splits)
    return tasks


def run(seed=17, steps=50):
    tasks = make_tasks(seed)
    torch.manual_seed(seed)
    initial = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    results = {}
    for method in ("finetune", "replay", "antara"):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        sampler = torch.Generator().manual_seed(seed + 100)
        model = Antara(
            copy.deepcopy(initial),
            learning_rate=0.01,
            memory_type="si" if method == "antara" else "none",
            feedback_buffer_size=0 if method == "finetune" else 128,
            device="cpu",
        )
        matrix = []
        for task_id, (train, _) in enumerate(tasks):
            x, y = train
            for _ in range(steps):
                indices = torch.randint(len(x), (32,), generator=sampler)
                model.learn(x[indices], targets=y[indices], task_id=task_id)
            model.on_task_complete(task_id)
            row = []
            for _, (test_x, test_y) in tasks:
                predicted = model.predict(test_x).argmax(-1)
                row.append((predicted == test_y).float().mean().item())
            matrix.append(row)
        results[method] = {
            "accuracy_matrix": matrix,
            "final_accuracy": sum(matrix[-1]) / 2,
            "bwt": matrix[1][0] - matrix[0][0],
            "replay_examples": len(model.feedback_buffer.buffer),
        }
    return {"seed": seed, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    logging.disable(logging.CRITICAL)
    runs = [run(seed, args.steps) for seed in args.seeds]
    summary = {}
    for method in runs[0]["results"]:
        summary[method] = {
            metric: {
                "mean": float(np.mean([r["results"][method][metric] for r in runs])),
                "std": float(np.std([r["results"][method][metric] for r in runs])),
            }
            for metric in ("final_accuracy", "bwt")
        }
    report = {
        "protocol": "Synthetic Gaussian classes, disjoint held-out data, 2 tasks, 4-way global head; no task ID at inference.",
        "limitations": "Smoke test only. Replay methods take additional optimizer steps; this is not a compute-matched comparison or a SOTA result.",
        "steps_per_task": args.steps,
        "torch_version": torch.__version__,
        "runs": runs,
        "summary": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
