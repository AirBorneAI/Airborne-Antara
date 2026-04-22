"""
AirborneHRS Retention & Ablation Benchmark
==========================================
Generates 3 plots:
1. retention_plot.png: Full Framework on A->B->C->D
2. retention_vs_baseline.png: Framework vs Naive Baseline
3. ablation_plot.png: Full vs No-Memory vs No-Consciousness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging
import copy

logging.disable(logging.CRITICAL)  # Suppress logs for clean output

# --- DATA GENERATION ---
def generate_data(task_id, batch_size=32):
    x = torch.randn(batch_size, 10)
    if task_id == 'A':
        y = x.sum(dim=1, keepdim=True)
    elif task_id == 'B':
        y = -x.sum(dim=1, keepdim=True)
    elif task_id == 'C':
        y = (x[:, ::2]).sum(dim=1, keepdim=True)
    elif task_id == 'D':
        y = (x[:, 1::2]).sum(dim=1, keepdim=True)
    return x, y

# --- TRAINING UTILS ---
def create_base_model():
    return nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

def train_until_convergence(agent, task_id, threshold=1.0, max_steps=300):
    for step in range(max_steps):
        x, y = generate_data(task_id)
        if isinstance(agent, AdaptiveFramework):
            metrics = agent.train_step(x, target_data=y)
            loss = metrics['loss']
        else:
            agent.zero_grad()
            pred = agent(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            agent.optimizer.step()
            loss = loss.item()
        if loss < threshold:
            return step
    return max_steps

def evaluate_all(agent, tasks):
    results = {}
    with torch.no_grad():
        for t in tasks:
            x, y = generate_data(t, batch_size=100)
            if isinstance(agent, AdaptiveFramework):
                output = agent(x)
                pred = output[0] if isinstance(output, tuple) else output
            else:
                pred = agent(x)
            results[t] = F.mse_loss(pred, y).item()
    return results

# --- NAIVE BASELINE (No Memory) ---
class NaiveBaseline(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    def forward(self, x):
        return self.model(x)
    def zero_grad(self):
        self.optimizer.zero_grad()

# --- EXPERIMENT RUNNERS ---
def run_experiment(agent, tasks):
    """Runs ABCD, returns history dict."""
    history = {t: [] for t in tasks}
    for current_task in tasks:
        train_until_convergence(agent, current_task)
        res = evaluate_all(agent, tasks)
        for t in tasks:
            history[t].append(res[t])
        if isinstance(agent, AdaptiveFramework) and agent.prioritized_buffer:
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
    return history

def plot_single(history, tasks, title, filename):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, t in enumerate(tasks):
        plt.plot(tasks, history[t], marker=markers[i], label=f'Task {t}', linewidth=2, markersize=10)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training Phase (After Task X)")
    plt.ylabel("MSE Loss (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"DONE: Saved: {filename}")
    plt.close()

# ============ PLOT 1: Full Framework ============
def generate_plot1():
    print("\n[STEP 1/3] Full AirborneHRS Framework on A->B->C->D...")
    cfg = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=True
    )
    agent = AdaptiveFramework(create_base_model(), cfg, device='cpu')
    history = run_experiment(agent, ['A', 'B', 'C', 'D'])
    plot_single(history, ['A', 'B', 'C', 'D'],
                "AirborneHRS: Multi-Task Retention (A→B→C→D)",
                "tests/retention_plot.png")

# ============ PLOT 2: Framework vs Baseline ============
def generate_plot2():
    print("\n[STEP 2/3] Framework vs Naive Baseline...")
    tasks = ['A', 'B', 'C', 'D']
    
    # Full Framework
    cfg = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=True
    )
    agent_fw = AdaptiveFramework(create_base_model(), cfg, device='cpu')
    history_fw = run_experiment(agent_fw, tasks)
    
    # Baseline (no EWC, no replay)
    baseline = NaiveBaseline(create_base_model())
    history_base = run_experiment(baseline, tasks)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(tasks))
    width = 0.35
    
    # Calculate average retention (mean error across all tasks at end)
    final_fw = [history_fw[t][-1] for t in tasks]
    final_base = [history_base[t][-1] for t in tasks]
    
    bars1 = plt.bar(x_axis - width/2, final_fw, width, label='AirborneHRS', color='#2ecc71')
    bars2 = plt.bar(x_axis + width/2, final_base, width, label='Naive Baseline', color='#e74c3c')
    
    plt.xlabel('Task')
    plt.ylabel('Final MSE Loss (Lower is Better)')
    plt.title('Framework vs Baseline: Final Task Errors After Learning A→B→C→D', fontsize=14, fontweight='bold')
    plt.xticks(x_axis, tasks)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("tests/retention_vs_baseline.png", dpi=150)
    print("DONE: Saved: tests/retention_vs_baseline.png")
    plt.close()

# ============ PLOT 3: Ablation Study ============
def generate_plot3():
    print("\n[STEP 3/3] Ablation Study (Memory & Consciousness)...")
    tasks = ['A', 'B', 'C', 'D']
    results = {}
    
    # Config 1: Full Framework
    print("   - Running: Full Framework")
    cfg_full = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=True
    )
    agent_full = AdaptiveFramework(create_base_model(), cfg_full, device='cpu')
    results['Full Framework'] = run_experiment(agent_full, tasks)
    
    # Config 2: No Memory (disable EWC and dreaming)
    print("   - Running: No Memory")
    cfg_no_mem = AdaptiveFrameworkConfig(
        device='cpu', memory_type='none', ewc_lambda=0.0,
        dream_interval=9999, enable_dreaming=False, enable_consciousness=True
    )
    agent_no_mem = AdaptiveFramework(create_base_model(), cfg_no_mem, device='cpu')
    results['No Memory'] = run_experiment(agent_no_mem, tasks)
    
    # Config 3: No Consciousness
    print("   - Running: No Consciousness")
    cfg_no_con = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=False
    )
    agent_no_con = AdaptiveFramework(create_base_model(), cfg_no_con, device='cpu')
    results['No Consciousness'] = run_experiment(agent_no_con, tasks)
    
    # Plot Ablation
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(tasks))
    width = 0.25
    
    colors = {'Full Framework': '#2ecc71', 'No Memory': '#e74c3c', 'No Consciousness': '#3498db'}
    offsets = {'Full Framework': -width, 'No Memory': 0, 'No Consciousness': width}
    
    for config_name, history in results.items():
        final_errors = [history[t][-1] for t in tasks]
        plt.bar(x_axis + offsets[config_name], final_errors, width, 
                label=config_name, color=colors[config_name])
    
    plt.xlabel('Task')
    plt.ylabel('Final MSE Loss (Lower is Better)')
    plt.title('Ablation Study: Impact of Memory & Consciousness', fontsize=14, fontweight='bold')
    plt.xticks(x_axis, tasks)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("tests/ablation_plot.png", dpi=150)
    print("DONE: Saved: tests/ablation_plot.png")
    plt.close()

# ============ MAIN ============
if __name__ == "__main__":
    print("="*50)
    print("AirborneHRS Retention & Ablation Benchmark")
    print("="*50)
    
    generate_plot1()
    generate_plot2()
    generate_plot3()
    
    print("\n" + "="*50)
    print("FINISH: All 3 plots generated successfully!")
    print("   - tests/retention_plot.png")
    print("   - tests/retention_vs_baseline.png")
    print("   - tests/ablation_plot.png")
    print("="*50)
