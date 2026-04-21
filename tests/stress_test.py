"""
ANTARA Stress Test: Cognitive Capacity & Breakpoint Analysis
============================================================
Determines the maximum number of tasks (N_max) the system can learn
before catastrophic collapse occurs.

Methodology:
1. Infinite stream of random tasks (Task 1, Task 2, ...).
2. Train until convergence on current task.
3. Evaluate on ALL previous tasks.
4. Stop when Average Accuracy < 50% (Collapse).

Comparison:
- EWC (Expected to fail early due to Fisher Matrix saturation/rigidity).
- ANTARA-Full (Expected to last 3x longer due to Generative Replay refreshing memory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging
import os
import time

# Suppress logging
logging.disable(logging.CRITICAL)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ INFINITE DATA GENERATOR ============
def generate_random_task_data(task_seed, batch_size=32):
    """Generates data for a unique task based on a seed."""
    # Local random state for this task to ensure consistency
    rng = np.random.RandomState(task_seed)
    
    # Random projection matrix for this task
    proj = rng.randn(10, 10).astype(np.float32)
    proj = torch.from_numpy(proj)
    
    # Task type: 0=Sum, 1=Diff, 2=Product-ish, 3=Alternating
    task_type = rng.randint(0, 4)
    
    x = torch.randn(batch_size, 10)
    
    if task_type == 0:
        # Linear transform + Sum
        transformed = x @ proj
        y = transformed.sum(dim=1, keepdim=True)
    elif task_type == 1:
        # Linear transform + Alternating Diff
        transformed = x @ proj
        y = (transformed[:, ::2] - transformed[:, 1::2]).sum(dim=1, keepdim=True)
    elif task_type == 2:
        # Non-linearish
        transformed = torch.sin(x @ proj)
        y = transformed.sum(dim=1, keepdim=True)
    else:
        # Masked sum
        mask = torch.from_numpy(rng.randint(0, 2, (10,)).astype(np.float32))
        transformed = x * mask
        y = transformed.sum(dim=1, keepdim=True)
        
    return x, y

# ============ MODELS ============
def create_base_model():
    return nn.Sequential(
        nn.Linear(10, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )

class EWCBaseline(nn.Module):
    def __init__(self, ewc_lambda=5000.0):
        super().__init__()
        self.model = create_base_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, task_seed, n_samples=100):
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        for _ in range(n_samples):
            x, y = generate_random_task_data(task_seed, batch_size=1)
            self.model.zero_grad()
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
        for n in self.fisher:
            self.fisher[n] /= n_samples
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.model.train()
    
    def ewc_loss(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self.model(x)
        t_loss = F.mse_loss(pred, y)
        ewc = self.ewc_loss() * self.ewc_lambda if self.fisher else 0.0
        loss = t_loss + ewc
        loss.backward()
        self.optimizer.step()
        return t_loss.item() 

# ============ STRESS TEST LOOP ============
def train_task(agent, task_id_seed, steps=200):
    for _ in range(steps):
        x, y = generate_random_task_data(task_id_seed)
        if isinstance(agent, AdaptiveFramework):
            agent.train_step(x, target_data=y)
        else:
            agent.train_step(x, y)

def evaluate_capacity(agent, task_seeds):
    total_loss = 0.0
    for seed in task_seeds:
        x, y = generate_random_task_data(seed, batch_size=50)
        with torch.no_grad():
            if isinstance(agent, AdaptiveFramework):
                out = agent(x)
                pred = out[0] if isinstance(out, tuple) else out
            else:
                pred = agent(x)
            total_loss += F.mse_loss(pred, y).item()
    return total_loss / len(task_seeds)

def run_stress_test(agent_name, agent, max_tasks=50):
    print(f"\nStressing: {agent_name}...")
    task_seeds = []
    accuracies = [] # We'll use 1/loss as a proxy for "health" or just track loss
    losses = []
    
    start_time = time.time()
    
    for i in range(max_tasks):
        task_seed = i + 1000 # Unique seed per task
        task_seeds.append(task_seed)
        
        # Train on new task
        train_task(agent, task_seed)
        
        # Consolidate
        if isinstance(agent, AdaptiveFramework) and agent.prioritized_buffer:
             agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
        elif isinstance(agent, EWCBaseline):
             agent.compute_fisher(task_seed)
             
        # Evaluate Retention
        avg_mse = evaluate_capacity(agent, task_seeds)
        losses.append(avg_mse)
        
        print(f"  Task {i+1}: Avg MSE = {avg_mse:.4f}")
        
        # Collapse Condition (Loss > Threshold)
        # Assuming MSE > 15.0 means it has lost the plot (random init is usually ~20-30)
        if avg_mse > 15.0:
            print(f"  !!! SYSTEM COLLAPSE DETECTED AT TASK {i+1} !!!")
            break
            
    return losses

# ============ MAIN ============
if __name__ == "__main__":
    print("="*60)
    print("ANTARA STRESS TEST: COGNITIVE CAPACITY")
    print("="*60)
    
    results = {}
    
    # 1. EWC Baseline
    ewc = EWCBaseline(ewc_lambda=5000.0)
    results['EWC'] = run_stress_test("EWC", ewc)
    
    # 2. ANTARA Full
    cfg = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=5000.0,
        enable_dreaming=True,
        dream_interval=10, 
        enable_consciousness=True,
        learning_rate=1e-3,
        model_dim=128
    )
    antara = AdaptiveFramework(create_base_model(), cfg, device='cpu')
    results['ANTARA'] = run_stress_test("ANTARA", antara)
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    for name, loss_curve in results.items():
        plt.plot(range(1, len(loss_curve)+1), loss_curve, marker='o', label=name, linewidth=2)
        
    plt.axhline(y=15.0, color='r', linestyle='--', label='Collapse Threshold')
    plt.xlabel("Number of Tasks Learned")
    plt.ylabel("Average MSE Loss (All Previous Tasks)")
    plt.title("Cognitive Capacity Stress Test")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "stress_test_capacity.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved capacity plot to: {output_path}")
    
    # Analysis
    ewc_len = len(results['EWC'])
    antara_len = len(results['ANTARA'])
    print("\nRESULTS:")
    print(f"EWC Capacity:    {ewc_len} tasks")
    print(f"ANTARA Capacity: {antara_len} tasks")
    
    if antara_len > ewc_len * 1.5:
        print("CONCLUSION: ANTARA demonstrates significantly higher cognitive capacity.")
    else:
        print("CONCLUSION: Capacity difference is marginal.")
