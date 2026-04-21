"""
ANTARA Extended Benchmark: 8-Task Challenge
============================================
Demonstrates the value of full ANTARA features (Dreaming, Hybrid Memory, 
Consciousness) on a challenging continual learning benchmark.

Why 8 tasks?
- More opportunities for catastrophic forgetting
- Complex interference patterns
- Memory buffer diversity matters more
- Long-term retention becomes critical

Tasks: A → B → C → D → E → F → G → H
Each task has conflicting objectives to maximize interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging

# Suppress logging
logging.disable(logging.CRITICAL)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ 8 CONFLICTING TASKS ============
# Higher dimensionality (20 inputs) for more complex interference
INPUT_DIM = 20

def generate_data(task_id, batch_size=32):
    """
    8 tasks with carefully designed conflicts:
    - A/B are opposites (sum vs negative sum)
    - C/D target different subspaces
    - E/F are nonlinear conflicts
    - G/H introduce cross-feature interactions
    """
    x = torch.randn(batch_size, INPUT_DIM)
    
    if task_id == 'A':
        y = x.sum(dim=1, keepdim=True)
    elif task_id == 'B':
        y = -x.sum(dim=1, keepdim=True)  # Opposite of A
    elif task_id == 'C':
        y = x[:, :10].sum(dim=1, keepdim=True)  # First half
    elif task_id == 'D':
        y = x[:, 10:].sum(dim=1, keepdim=True)  # Second half
    elif task_id == 'E':
        y = (x ** 2).sum(dim=1, keepdim=True)  # Nonlinear
    elif task_id == 'F':
        y = torch.abs(x).sum(dim=1, keepdim=True)  # Nonlinear, different
    elif task_id == 'G':
        y = (x[:, :10] * x[:, 10:]).sum(dim=1, keepdim=True)  # Interaction
    elif task_id == 'H':
        y = x[:, ::2].sum(dim=1, keepdim=True) - x[:, 1::2].sum(dim=1, keepdim=True)  # Interleaved
    
    return x, y

# ============ MODEL ============
def create_larger_model():
    """Larger network for more complex tasks."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

# ============ BASELINES ============
class NaiveBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_larger_model()
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
    def __init__(self, ewc_lambda=1000.0):
        super().__init__()
        self.model = create_larger_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        self.task_count = 0
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, current_task, samples=100):
        new_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for _ in range(samples):
            x, y = generate_data(current_task, batch_size=1)
            self.model.zero_grad()
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += p.grad.data ** 2
        
        for n in new_fisher:
            new_fisher[n] /= samples
            # Accumulate Fisher across tasks
            if n in self.fisher:
                self.fisher[n] = 0.5 * self.fisher[n] + 0.5 * new_fisher[n]
            else:
                self.fisher[n] = new_fisher[n]
        
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
        task_loss = F.mse_loss(pred, y)
        ewc_penalty = self.ewc_loss() * self.ewc_lambda if self.fisher else 0.0
        total_loss = task_loss + ewc_penalty
        total_loss.backward()
        self.optimizer.step()
        return task_loss.item()
    
    def consolidate(self, current_task):
        self.compute_fisher(current_task)
        self.task_count += 1


class ReplayBaseline(nn.Module):
    def __init__(self, buffer_size=2000):
        super().__init__()
        self.model = create_larger_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = []
        self.buffer_size = buffer_size
    
    def forward(self, x):
        return self.model(x)
    
    def add_to_buffer(self, x, y):
        for i in range(x.size(0)):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((x[i].clone(), y[i].clone()))
            else:
                idx = np.random.randint(0, len(self.buffer))
                self.buffer[idx] = (x[i].clone(), y[i].clone())
    
    def replay_step(self, batch_size=32):
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
        self.add_to_buffer(x, y)
        
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        self.optimizer.step()
        
        self.replay_step()
        return loss.item()


# ============ EXPERIMENT RUNNER ============
def train_until_convergence(agent, task_id, threshold=2.0, max_steps=500):
    """More steps for harder tasks."""
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
            x, y = generate_data(t, batch_size=200)  # More samples for stable eval
            if isinstance(agent, AdaptiveFramework):
                output = agent(x)
                pred = output[0] if isinstance(output, tuple) else output
            else:
                pred = agent(x)
            results[t] = F.mse_loss(pred, y).item()
    return results

def run_experiment(agent, tasks, name):
    print(f"   Running: {name}...")
    history = {t: [] for t in tasks}
    snapshots = {}
    
    for current_task in tasks:
        train_until_convergence(agent, current_task)
        res = evaluate_all(agent, tasks)
        snapshots[current_task] = res[current_task]
        
        for t in tasks:
            history[t].append(res[t])
        
        # Consolidation
        if isinstance(agent, AdaptiveFramework) and agent.prioritized_buffer:
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
        elif isinstance(agent, EWCBaseline):
            agent.consolidate(current_task)
    
    return history, snapshots

def compute_metrics(history, snapshots, tasks):
    """Compute BWT and average final loss."""
    bwt_values = []
    for task in tasks[:-1]:
        loss_after_final = history[task][-1]
        loss_after_training = snapshots[task]
        bwt = loss_after_training - loss_after_final
        bwt_values.append(bwt)
    
    bwt = np.mean(bwt_values)
    avg_loss = np.mean([history[t][-1] for t in tasks])
    return bwt, avg_loss

# ============ MAIN BENCHMARK ============
def run_extended_benchmark():
    print("\n" + "="*70)
    print("ANTARA EXTENDED BENCHMARK: 8-TASK CHALLENGE")
    print("="*70)
    print("Tasks: A → B → C → D → E → F → G → H")
    print("This benchmark demonstrates the value of full ANTARA features.\n")
    
    tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    results = {}
    
    # 1. Naive Baseline
    print("🔬 [1/5] Naive Baseline...")
    naive = NaiveBaseline()
    results['Naive'] = run_experiment(naive, tasks, "Naive")
    
    # 2. EWC-Only
    print("🔬 [2/5] EWC-Only...")
    ewc = EWCBaseline(ewc_lambda=1000.0)
    results['EWC'] = run_experiment(ewc, tasks, "EWC")
    
    # 3. Replay-Only
    print("🔬 [3/5] Replay-Only...")
    replay = ReplayBaseline(buffer_size=2000)
    results['Replay'] = run_experiment(replay, tasks, "Replay")
    
    # 4. ANTARA Minimal (EWC only, no extras)
    print("🔬 [4/5] ANTARA Minimal (EWC only)...")
    cfg_minimal = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=1000.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_min = AdaptiveFramework(create_larger_model(), cfg_minimal, device='cpu')
    results['ANTARA-Min'] = run_experiment(antara_min, tasks, "ANTARA-Min")
    
    # 5. ANTARA Full (Hybrid + Dreaming + All features)
    print("🔬 [5/5] ANTARA Full (Hybrid + Dreaming)...")
    cfg_full = AdaptiveFrameworkConfig(
        device='cpu',
        # HYBRID MEMORY: EWC + SI working together
        memory_type='hybrid',
        ewc_lambda=500.0,
        si_lambda=2.0,
        # DREAMING: Critical for 8-task retention
        enable_dreaming=True,
        dream_interval=25,
        dream_batch_size=64,
        use_prioritized_replay=True,
        feedback_buffer_size=10000,  # Large buffer for 8 tasks
        replay_priority_temperature=0.5,
        # Extra features for complex tasks
        enable_consciousness=True,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_full = AdaptiveFramework(create_larger_model(), cfg_full, device='cpu')
    results['ANTARA-Full'] = run_experiment(antara_full, tasks, "ANTARA-Full")
    
    return results, tasks

# ============ VISUALIZATION ============
def plot_extended_comparison(results, tasks, filename):
    """Bar chart for 8-task benchmark."""
    methods = list(results.keys())
    
    metrics = {}
    for method, (history, snapshots) in results.items():
        bwt, avg_loss = compute_metrics(history, snapshots, tasks)
        metrics[method] = {'bwt': bwt, 'avg_loss': avg_loss}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    
    x = np.arange(len(methods))
    
    # Average Loss
    bars1 = axes[0].bar(x, [metrics[m]['avg_loss'] for m in methods], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, fontsize=10)
    axes[0].set_ylabel("Average Final MSE Loss")
    axes[0].set_title("8-Task Average Loss (Lower is Better)", fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # BWT
    bars2 = axes[1].bar(x, [metrics[m]['bwt'] for m in methods], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, fontsize=10)
    axes[1].set_ylabel("Backward Transfer (BWT)")
    axes[1].set_title("8-Task BWT (Higher is Better)", fontsize=13, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:+.1f}', ha='center',
                     va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.suptitle("Extended Benchmark: A→B→C→D→E→F→G→H", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def plot_extended_retention(results, tasks, filename):
    """Retention curves for all methods on 8 tasks."""
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 5), sharey=True)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for idx, (method, (history, _)) in enumerate(results.items()):
        ax = axes[idx]
        for i, t in enumerate(tasks):
            ax.plot(tasks, history[t], marker=markers[i], color=colors[i],
                    label=f'Task {t}', linewidth=1.5, markersize=6)
        ax.set_title(method, fontsize=11, fontweight='bold')
        ax.set_xlabel("Training Phase")
        if idx == 0:
            ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)
    
    plt.suptitle("8-Task Retention Curves", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def print_extended_results(results, tasks):
    """Print results table."""
    print("\n" + "="*70)
    print("📊 EXTENDED BENCHMARK RESULTS (8 Tasks)")
    print("="*70)
    
    print(f"\n{'Method':<15} | {'Avg Loss':>10} | {'BWT':>10} | Winner")
    print("-"*55)
    
    all_metrics = {}
    for method, (history, snapshots) in results.items():
        bwt, avg_loss = compute_metrics(history, snapshots, tasks)
        all_metrics[method] = {'bwt': bwt, 'avg_loss': avg_loss}
        print(f"{method:<15} | {avg_loss:>10.2f} | {bwt:>+10.2f} |")
    
    print("-"*55)
    
    # Find winners
    best_loss = min(all_metrics.values(), key=lambda x: x['avg_loss'])['avg_loss']
    best_bwt = max(all_metrics.values(), key=lambda x: x['bwt'])['bwt']
    
    loss_winner = [m for m, v in all_metrics.items() if v['avg_loss'] == best_loss][0]
    bwt_winner = [m for m, v in all_metrics.items() if v['bwt'] == best_bwt][0]
    
    print(f"\n🏆 Best Avg Loss: {loss_winner} ({best_loss:.2f})")
    print(f"🏆 Best BWT: {bwt_winner} ({best_bwt:+.2f})")
    
    # ANTARA-Full vs ANTARA-Min comparison
    if 'ANTARA-Full' in all_metrics and 'ANTARA-Min' in all_metrics:
        full = all_metrics['ANTARA-Full']
        mini = all_metrics['ANTARA-Min']
        
        loss_improvement = (mini['avg_loss'] - full['avg_loss']) / mini['avg_loss'] * 100
        bwt_improvement = full['bwt'] - mini['bwt']
        
        print(f"\n📈 ANTARA-Full vs ANTARA-Min:")
        print(f"   Loss improvement: {loss_improvement:+.1f}%")
        print(f"   BWT improvement:  {bwt_improvement:+.2f}")

# ============ MAIN ============
if __name__ == "__main__":
    results, tasks = run_extended_benchmark()
    
    # Generate plots
    plot_extended_comparison(results, tasks, "tests/extended_comparison.png")
    plot_extended_retention(results, tasks, "tests/extended_retention.png")
    
    # Print results
    print_extended_results(results, tasks)
    
    print("\n" + "="*70)
    print("✨ EXTENDED BENCHMARK COMPLETE")
    print("="*70)
    print("Generated files:")
    print("   - tests/extended_comparison.png")
    print("   - tests/extended_retention.png")
    print("="*70)
