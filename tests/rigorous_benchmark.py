"""
ANTARA Rigorous Baseline Comparison
===================================
Compares ANTARA against established continual learning methods:
1. Naive Baseline (No protection)
2. EWC-Only (Elastic Weight Consolidation)
3. Replay-Only (Experience Replay without EWC)
4. MoE (Mixture of Experts - Sparse routing)
5. ANTARA Full (Hybrid EWC + Replay + Consciousness)

Outputs:
- tests/rigorous_comparison_barchart.png
- tests/rigorous_retention_curves.png
- Comprehensive BWT table
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airborne_antara.moe import SparseMoE
import numpy as np
import logging
import copy
import os

# Suppress logging
logging.disable(logging.CRITICAL)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ DATA GENERATION ============
INPUT_DIM = 50
def generate_data(task_id, batch_size=32):
    # DENSE INPUTS: All dimensions are active random noise
    # This forces interference because the model sees "garbage" in dims relevant to other tasks
    x = torch.randn(batch_size, INPUT_DIM)
    
    if task_id == 'A':
        # Task A: Depends on dims 0-10
        y = x[:, 0:10].sum(dim=1, keepdim=True)
    elif task_id == 'B':
        # Task B: Depends on dims 10-20
        y = x[:, 10:20].sum(dim=1, keepdim=True)
    elif task_id == 'C':
        # Task C: Depends on dims 20-30
        y = x[:, 20:30].sum(dim=1, keepdim=True)
    elif task_id == 'D':
        # Task D: Depends on dims 30-40
        y = x[:, 30:40].sum(dim=1, keepdim=True)
    
    return x, y

# ============ BASE MODEL ============
def create_base_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )

# ============ BASELINE IMPLEMENTATIONS ============

class NaiveBaseline(nn.Module):
    """Vanilla neural network - no continual learning protection."""
    def __init__(self):
        super().__init__()
        self.model = create_base_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class EWCBaseline(nn.Module):
    """EWC-Only: Elastic Weight Consolidation without replay."""
    def __init__(self, ewc_lambda=1000.0):
        super().__init__()
        self.model = create_base_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, data_loader_samples=100):
        """Compute Fisher Information Matrix."""
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for _ in range(data_loader_samples):
            x, y = generate_data('A', batch_size=1)  # Sample from current task
            self.model.zero_grad()
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
        
        for n in self.fisher:
            self.fisher[n] /= data_loader_samples
        
        # Store optimal params
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.model.train()
    
    def ewc_loss(self):
        """Compute EWC penalty."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return loss
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self.model(x)
        task_loss = F.mse_loss(pred, y)
        ewc_penalty = self.ewc_loss() * self.ewc_lambda if self.fisher else 0.0
        total_loss = task_loss + ewc_penalty
        total_loss.backward()
        self.optimizer.step()
        return task_loss.item()
    
    def consolidate(self):
        """Call after each task to update Fisher."""
        self.compute_fisher()


class ReplayBaseline(nn.Module):
    """Replay-Only: Experience Replay without EWC."""
    def __init__(self, buffer_size=500):
        super().__init__()
        self.model = create_base_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = []
        self.buffer_size = buffer_size
    
    def forward(self, x):
        return self.model(x)
    
    def add_to_buffer(self, x, y):
        """Add samples to replay buffer."""
        for i in range(x.size(0)):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((x[i].clone(), y[i].clone()))
            else:
                # Reservoir sampling
                idx = np.random.randint(0, len(self.buffer))
                self.buffer[idx] = (x[i].clone(), y[i].clone())
    
    def replay_step(self, batch_size=16):
        """Replay from buffer."""
        if len(self.buffer) < batch_size:
            return 0.0
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        x_replay = torch.stack([self.buffer[i][0] for i in indices])
        y_replay = torch.stack([self.buffer[i][1] for i in indices])
        
        self.optimizer.zero_grad()
        pred = self.model(x_replay)
        loss = F.mse_loss(pred, y_replay)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_step(self, x, y):
        # Add to buffer
        self.add_to_buffer(x, y)
        
        # Train on current batch
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        self.optimizer.step()
        
        # Interleaved replay
        self.replay_step()
        
        return loss.item()


class MoEBaseline(nn.Module):
    """MoE Continual Learner: Uses expert routing for task specialization."""
    def __init__(self, num_experts=4):
        super().__init__()
        base = create_base_model()
        self.moe = SparseMoE(base, input_dim=INPUT_DIM, num_experts=num_experts, top_k=2)
        self.optimizer = torch.optim.Adam(self.moe.parameters(), lr=1e-3)
    
    def forward(self, x):
        out, _ = self.moe(x)
        return out
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred, _ = self.moe(x)
        task_loss = F.mse_loss(pred, y)
        aux_loss = self.moe.get_aux_loss() * 0.01  # Load balancing
        total_loss = task_loss + aux_loss
        total_loss.backward()
        self.optimizer.step()
        return task_loss.item()


# ============ EXPERIMENT RUNNER ============
def train_until_convergence(agent, task_id, threshold=1.0, max_steps=300):
    for step in range(max_steps):
        x, y = generate_data(task_id)
        
        if isinstance(agent, AdaptiveFramework):
            metrics = agent.train_step(x, target_data=y)
            loss = metrics['loss']
        else:
            loss = agent.train_step(x, y)
        
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

def run_experiment(agent, tasks, name):
    """Run full ABCD experiment and return history + snapshots."""
    print(f"   Running: {name}...")
    history = {t: [] for t in tasks}
    snapshots = {}
    
    for current_task in tasks:
        train_until_convergence(agent, current_task)
        res = evaluate_all(agent, tasks)
        snapshots[current_task] = res[current_task]
        
        for t in tasks:
            history[t].append(res[t])
        
        # Post-task consolidation
        if isinstance(agent, AdaptiveFramework) and agent.prioritized_buffer:
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
        elif isinstance(agent, EWCBaseline):
            agent.consolidate()
    
    return history, snapshots

def compute_bwt(history, snapshots, tasks):
    """Compute Backward Transfer metric."""
    bwt_values = []
    for task in tasks[:-1]:
        loss_after_final = history[task][-1]
        loss_after_training = snapshots[task]
        bwt = loss_after_training - loss_after_final  # Positive = improvement
        bwt_values.append(bwt)
    return np.mean(bwt_values)

def compute_avg_final_loss(history, tasks):
    """Average final loss across all tasks."""
    return np.mean([history[t][-1] for t in tasks])

# ============ MAIN BENCHMARK ============
def run_rigorous_benchmark():
    print("\n" + "="*70)
    print("ANTARA ULTIMATE VERIFICATION SUITE (Academic Rigor)")
    print("="*70)
    
    tasks = ['A', 'B', 'C', 'D']
    results = {}
    
    # 1. Naive Baseline
    print("\n[1/6] Naive Baseline (No CL)...")
    naive = NaiveBaseline()
    results['Naive'] = run_experiment(naive, tasks, "Naive")
    
    # 2. EWC-Only
    print("\n[2/6] EWC-Only...")
    ewc = EWCBaseline(ewc_lambda=1000.0)
    results['EWC'] = run_experiment(ewc, tasks, "EWC")
    
    # 3. Replay-Only
    print("\n[3/6] Replay-Only...")
    replay = ReplayBaseline(buffer_size=500)
    results['Replay'] = run_experiment(replay, tasks, "Replay")
    
    # 4. MoE Baseline
    print("\n[4/6] MoE (4 Experts)...")
    moe = MoEBaseline(num_experts=4)
    results['MoE'] = run_experiment(moe, tasks, "MoE")
    
    # 5. ANTARA-Min (The "Self-Control" Test)
    # Goal: Prove that our base EWC implementation matches the EWC Baseline (sanity check)
    print("\n[5/6] ANTARA-Min (EWC Mode, No Dreaming)...")
    cfg_min = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=1000.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        enable_world_model=False,
        use_moe=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_min = AdaptiveFramework(create_base_model(), cfg_min, device='cpu')
    results['ANTARA-Min'] = run_experiment(antara_min, tasks, "ANTARA-Min")

    # 6. ANTARA-Full (The "Killer App")
    # Goal: Prove Dreaming + Hybrid Memory > EWC
    print("\n[6/6] ANTARA-Full (Dreaming + Consciousness)...")
    cfg_full = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid', # EWC + Replay
        ewc_lambda=1000.0,
        si_lambda=1.0,
        enable_dreaming=True,
        dream_interval=10, # Dream every 10 steps (More frequent replay)
        dream_batch_size=32,
        use_prioritized_replay=True,
        enable_consciousness=True,
        enable_health_monitor=False, # Keep noise low
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_full = AdaptiveFramework(create_base_model(), cfg_full, device='cpu')
    results['ANTARA-Full'] = run_experiment(antara_full, tasks, "ANTARA-Full")

    
    return results, tasks

# ============ VISUALIZATION ============
def plot_comparison_barchart(results, tasks, filename):
    """Bar chart comparing final losses and BWT."""
    methods = list(results.keys())
    
    # Compute metrics
    final_losses = {}
    bwt_scores = {}
    for method, (history, snapshots) in results.items():
        final_losses[method] = compute_avg_final_loss(history, tasks)
        bwt_scores[method] = compute_bwt(history, snapshots, tasks)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors (ANTARA highlighted)
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    
    # Left: Average Final Loss
    x = np.arange(len(methods))
    bars1 = axes[0].bar(x, [final_losses[m] for m in methods], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=12)
    axes[0].set_ylabel("Average Final MSE Loss", fontsize=12)
    axes[0].set_title("Average Final Loss (Lower is Better)", fontsize=14, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, method in zip(bars1, methods):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Right: BWT
    bars2 = axes[1].bar(x, [bwt_scores[m] for m in methods], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=12)
    axes[1].set_ylabel("Backward Transfer (BWT)", fontsize=12)
    axes[1].set_title("Backward Transfer (Higher is Better)", fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, axis='y', alpha=0.3)
    
    for bar, method in zip(bars2, methods):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:+.2f}', ha='center', 
                     va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.suptitle("Rigorous Baseline Comparison (A→B→C→D)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def plot_retention_curves(results, tasks, filename):
    """Multi-panel retention curves for all methods."""
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 5), sharey=True)
    markers = ['o', 's', '^', 'D']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for idx, (method, (history, _)) in enumerate(results.items()):
        ax = axes[idx]
        for i, t in enumerate(tasks):
            ax.plot(tasks, history[t], marker=markers[i], color=colors[i],
                    label=f'Task {t}', linewidth=2, markersize=8)
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_xlabel("Training Phase")
        if idx == 0:
            ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Retention Curves: All Methods (A→B→C→D)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def print_results_table(results, tasks):
    """Print comprehensive results table."""
    print("\n" + "="*70)
    print("== COMPREHENSIVE RESULTS TABLE")
    print("="*70)
    
    # Header
    print(f"\n{'Method':<12} | {'Avg Loss':>10} | {'BWT':>10} | {'Task A':>8} | {'Task B':>8} | {'Task C':>8} | {'Task D':>8}")
    print("-"*85)
    
    for method, (history, snapshots) in results.items():
        avg_loss = compute_avg_final_loss(history, tasks)
        bwt = compute_bwt(history, snapshots, tasks)
        task_losses = [history[t][-1] for t in tasks]
        print(f"{method:<12} | {avg_loss:>10.4f} | {bwt:>+10.4f} | {task_losses[0]:>8.4f} | {task_losses[1]:>8.4f} | {task_losses[2]:>8.4f} | {task_losses[3]:>8.4f}")
    
    print("-"*85)
    
    # Find best method
    bwt_scores = {m: compute_bwt(h, s, tasks) for m, (h, s) in results.items()}
    best_method = max(bwt_scores, key=bwt_scores.get)
    
    print(f"\n> Best BWT: {best_method} ({bwt_scores[best_method]:+.4f})")
    
    # ANTARA vs others
    antara_bwt = bwt_scores.get('ANTARA-Full', -99.9)
    if 'ANTARA-Full' not in bwt_scores: antara_bwt = bwt_scores.get('ANTARA', -99.9)
    print("\n> ANTARA Improvement over Baselines:")
    for method in results:
        if method != 'ANTARA-Full' and method != 'ANTARA':
            other_bwt = bwt_scores[method]
            if other_bwt < 0:
                improvement = ((other_bwt - antara_bwt) / abs(other_bwt)) * 100
                print(f"   vs {method}: {improvement:+.1f}% BWT improvement")
            else:
                diff = antara_bwt - other_bwt
                print(f"   vs {method}: {diff:+.4f} BWT difference")
    
    # LaTeX table
    print("\n" + "="*70)
    print("== LATEX TABLE (Copy-Paste)")
    print("="*70)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("Method & Avg Loss & BWT & Task A & Task B & Task C & Task D \\\\")
    print("\\midrule")
    for method, (history, snapshots) in results.items():
        avg_loss = compute_avg_final_loss(history, tasks)
        bwt = compute_bwt(history, snapshots, tasks)
        task_losses = [history[t][-1] for t in tasks]
        bold = "\\textbf" if 'ANTARA' in method else ""
        if bold:
            print(f"\\textbf{{{method}}} & \\textbf{{{avg_loss:.2f}}} & \\textbf{{{bwt:+.2f}}} & {task_losses[0]:.2f} & {task_losses[1]:.2f} & {task_losses[2]:.2f} & {task_losses[3]:.2f} \\\\")
        else:
            print(f"{method} & {avg_loss:.2f} & {bwt:+.2f} & {task_losses[0]:.2f} & {task_losses[1]:.2f} & {task_losses[2]:.2f} & {task_losses[3]:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Rigorous comparison of continual learning methods on A$\\rightarrow$B$\\rightarrow$C$\\rightarrow$D benchmark.}")
    print("\\label{tab:rigorous_comparison}")
    print("\\end{table}")

# ============ MAIN ============
if __name__ == "__main__":
    results, tasks = run_rigorous_benchmark()
    
    # Generate plots with absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_comparison_barchart(results, tasks, os.path.join(base_dir, "rigorous_comparison_barchart.png"))
    plot_retention_curves(results, tasks, os.path.join(base_dir, "rigorous_retention_curves.png"))
    
    # Print results
    print_results_table(results, tasks)
    
    print("\n" + "="*70)
    print("+ RIGOROUS BENCHMARK COMPLETE")
    print("="*70)
    print("Generated files:")
    print(f"   - {os.path.join(base_dir, 'rigorous_comparison_barchart.png')}")
    print(f"   - {os.path.join(base_dir, 'rigorous_retention_curves.png')}")
    print("="*70)
