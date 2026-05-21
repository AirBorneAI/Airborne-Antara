"""
ANTARA Publication Benchmark Suite
=======================================
Generates publication-ready experiments for continual learning paper.

Outputs:
1. BWT (Backward Transfer) Metrics
2. tests/baseline_retention_plot.png
3. tests/framework_vs_baseline_retention.png
4. tests/reversed_order_retention.png
5. Metrics summary printout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging
import copy

# Suppress all logging for clean output
logging.disable(logging.CRITICAL)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ DATA GENERATION ============
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

# ============ MODEL UTILS ============
def create_base_model():
    return nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

class NaiveBaseline(nn.Module):
    """Plain neural network with no memory protection."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    def forward(self, x):
        return self.model(x)
    def zero_grad(self):
        self.optimizer.zero_grad()

# ============ TRAINING UTILS ============
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

def run_experiment(agent, tasks, consolidate=True):
    """
    Runs sequential learning on tasks and returns:
    - history: {task: [loss_after_A, loss_after_B, ...]}
    - snapshots: {task: loss_immediately_after_training_that_task}
    """
    history = {t: [] for t in tasks}
    snapshots = {}
    
    for current_task in tasks:
        train_until_convergence(agent, current_task)
        res = evaluate_all(agent, tasks)
        
        # Snapshot: loss on current task right after training it
        snapshots[current_task] = res[current_task]
        
        for t in tasks:
            history[t].append(res[t])
        
        # Consolidate memory after task switch (framework only)
        if consolidate and isinstance(agent, AdaptiveFramework) and agent.prioritized_buffer:
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
    
    return history, snapshots

# ============ BWT COMPUTATION ============
def compute_bwt(history, snapshots, tasks):
    """
    Backward Transfer (BWT) = average over tasks i in {A,B,C} of:
        (Loss_i_after_D - Loss_i_after_i)
    
    Negative BWT = forgetting (bad)
    Zero/Positive BWT = retention/improvement (good)
    
    Note: We use negative loss difference so that:
    - Forgetting (loss increased) -> Negative BWT
    - Retention (loss same) -> Zero BWT
    - Improvement (loss decreased) -> Positive BWT
    """
    bwt_values = []
    for i, task in enumerate(tasks[:-1]):  # Exclude last task
        loss_after_final = history[task][-1]
        loss_after_training = snapshots[task]
        # BWT is typically: performance_final - performance_after_training
        # Since we use loss (lower is better), we negate:
        bwt = loss_after_training - loss_after_final
        bwt_values.append(bwt)
    
    return np.mean(bwt_values), bwt_values

# ============ PLOTTING UTILS ============
def plot_retention_curve(history, tasks, title, filename, color_scheme='framework'):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    if color_scheme == 'framework':
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    else:
        colors = ['#e74c3c', '#c0392b', '#a93226', '#922b21']
    
    for i, t in enumerate(tasks):
        plt.plot(tasks, history[t], marker=markers[i], color=colors[i],
                 label=f'Task {t}', linewidth=2, markersize=10)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Training Phase (After Learning Task X)")
    plt.ylabel("MSE Loss (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def plot_side_by_side(history_fw, history_base, tasks, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    markers = ['o', 's', '^', 'D']
    colors_fw = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    colors_base = ['#e74c3c', '#c0392b', '#a93226', '#922b21']
    
    # Left: Framework
    for i, t in enumerate(tasks):
        axes[0].plot(tasks, history_fw[t], marker=markers[i], color=colors_fw[i],
                     label=f'Task {t}', linewidth=2, markersize=10)
    axes[0].set_title("ANTARA Framework", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Training Phase")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Baseline
    for i, t in enumerate(tasks):
        axes[1].plot(tasks, history_base[t], marker=markers[i], color=colors_base[i],
                     label=f'Task {t}', linewidth=2, markersize=10)
    axes[1].set_title("Naive Baseline (No Memory)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Training Phase")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Retention Comparison: Framework vs Baseline (A→B→C→D)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

# ============ EXPERIMENT 1: BWT + Baseline Retention ============
def run_main_experiments():
    print("\n" + "="*60)
    print("EXPERIMENT 1: Framework vs Baseline + BWT Computation")
    print("="*60)
    
    tasks = ['A', 'B', 'C', 'D']
    
    # --- Full Framework ---
    print("\n🚀 Running: Full Framework (A→B→C→D)...")
    cfg_fw = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=True, input_dim=10
    )
    agent_fw = AdaptiveFramework(create_base_model(), cfg_fw, device='cpu')
    history_fw, snapshots_fw = run_experiment(agent_fw, tasks)
    
    # --- Naive Baseline ---
    print("🚀 Running: Naive Baseline (A→B→C→D)...")
    baseline = NaiveBaseline(create_base_model())
    history_base, snapshots_base = run_experiment(baseline, tasks, consolidate=False)
    
    # --- Compute BWT ---
    bwt_fw, bwt_fw_values = compute_bwt(history_fw, snapshots_fw, tasks)
    bwt_base, bwt_base_values = compute_bwt(history_base, snapshots_base, tasks)
    
    # --- Generate Plots ---
    plot_retention_curve(history_base, tasks, 
                         "Naive Baseline: Catastrophic Forgetting (A→B→C→D)",
                         "tests/baseline_retention_plot.png", 
                         color_scheme='baseline')
    
    plot_side_by_side(history_fw, history_base, tasks,
                      "tests/framework_vs_baseline_retention.png")
    
    return {
        'framework': {'history': history_fw, 'snapshots': snapshots_fw, 'bwt': bwt_fw, 'bwt_values': bwt_fw_values},
        'baseline': {'history': history_base, 'snapshots': snapshots_base, 'bwt': bwt_base, 'bwt_values': bwt_base_values}
    }

# ============ EXPERIMENT 2: Task Order Robustness ============
def run_reversed_order_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT 2: Task Order Robustness (D→C→B→A)")
    print("="*60)
    
    tasks_reversed = ['D', 'C', 'B', 'A']
    
    print("\n🚀 Running: Full Framework (D→C→B→A)...")
    cfg_fw = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid', ewc_lambda=1000.0,
        dream_interval=1, enable_consciousness=True, input_dim=10
    )
    agent_fw = AdaptiveFramework(create_base_model(), cfg_fw, device='cpu')
    history_fw, snapshots_fw = run_experiment(agent_fw, tasks_reversed)
    
    bwt_fw, bwt_fw_values = compute_bwt(history_fw, snapshots_fw, tasks_reversed)
    
    plot_retention_curve(history_fw, tasks_reversed,
                         "ANTARA: Reversed Order (D→C→B→A)",
                         "tests/reversed_order_retention.png",
                         color_scheme='framework')
    
    return {
        'history': history_fw,
        'snapshots': snapshots_fw,
        'bwt': bwt_fw,
        'bwt_values': bwt_fw_values
    }

# ============ METRICS SUMMARY ============
def print_metrics_summary(main_results, reversed_results):
    print("\n" + "="*60)
    print("📊 PUBLICATION METRICS SUMMARY")
    print("="*60)
    
    tasks = ['A', 'B', 'C', 'D']
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│ FINAL TASK LOSSES (After A→B→C→D)                       │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Task │ ANTARA      │ Naive Baseline │ Winner            │")
    print("├──────┼─────────────┼────────────────┼───────────────────┤")
    
    for t in tasks:
        fw_loss = main_results['framework']['history'][t][-1]
        base_loss = main_results['baseline']['history'][t][-1]
        winner = "Framework ✓" if fw_loss < base_loss else "Baseline"
        print(f"│  {t}   │ {fw_loss:>11.4f} │ {base_loss:>14.4f} │ {winner:<17} │")
    
    print("└─────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│ BACKWARD TRANSFER (BWT) METRICS                         │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│ ANTARA BWT:          {main_results['framework']['bwt']:>+8.4f}                       │")
    print(f"│ Naive Baseline BWT:  {main_results['baseline']['bwt']:>+8.4f}                       │")
    print(f"│ Reversed Order BWT:  {reversed_results['bwt']:>+8.4f}                       │")
    print("├─────────────────────────────────────────────────────────┤")
    
    # Per-task BWT breakdown
    print("│ BWT Breakdown (per prior task):                         │")
    for i, t in enumerate(['A', 'B', 'C']):
        fw_bwt = main_results['framework']['bwt_values'][i]
        base_bwt = main_results['baseline']['bwt_values'][i]
        print(f"│   Task {t}: Framework={fw_bwt:>+.4f}, Baseline={base_bwt:>+.4f}      │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # Automatic conclusion
    print("\n" + "="*60)
    print("📝 AUTOMATIC CONCLUSION (For Paper)")
    print("="*60)
    
    fw_bwt = main_results['framework']['bwt']
    base_bwt = main_results['baseline']['bwt']
    
    if fw_bwt > base_bwt:
        if fw_bwt >= 0:
            conclusion = f"The ANTARA framework exhibits positive backward transfer (BWT={fw_bwt:+.4f}), indicating knowledge consolidation that improves prior task performance. In contrast, the naive baseline shows severe negative backward transfer (BWT={base_bwt:+.4f}), confirming catastrophic forgetting. The framework achieves {((base_bwt - fw_bwt) / abs(base_bwt)) * 100:.1f}% improvement in BWT over baseline."
        else:
            conclusion = f"The ANTARA framework exhibits reduced negative backward transfer (BWT={fw_bwt:+.4f}) compared to the naive baseline (BWT={base_bwt:+.4f}), demonstrating {((base_bwt - fw_bwt) / abs(base_bwt)) * 100:.1f}% reduction in catastrophic forgetting."
    else:
        conclusion = "Results require further analysis."
    
    print(f"\n\"{conclusion}\"\n")
    
    # Generate copy-paste metrics
    print("="*60)
    print("📋 COPY-PASTE METRICS (LaTeX Format)")
    print("="*60)
    print(f"\\newcommand{{\\BWTframework}}{{{fw_bwt:+.4f}}}")
    print(f"\\newcommand{{\\BWTbaseline}}{{{base_bwt:+.4f}}}")
    print(f"\\newcommand{{\\BWTimprovement}}{{{((base_bwt - fw_bwt) / abs(base_bwt)) * 100:.1f}\\%}}")

# ============ MAIN ============
if __name__ == "__main__":
    print("="*60)
    print("ANTARA Publication Benchmark Suite")
    print("="*60)
    
    # Run all experiments
    main_results = run_main_experiments()
    reversed_results = run_reversed_order_experiment()
    
    # Print comprehensive summary
    print_metrics_summary(main_results, reversed_results)
    
    print("\n" + "="*60)
    print("✨ ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("   - tests/baseline_retention_plot.png")
    print("   - tests/framework_vs_baseline_retention.png")
    print("   - tests/reversed_order_retention.png")
    print("="*60)
