"""
ANTARA Complex Benchmark: 20-Task, 100D Challenge
=================================================
This benchmark is designed to STRESS TEST continual learning systems
and demonstrate where full ANTARA features provide genuine value.

Benchmark Properties:
- 100-dimensional inputs (10x larger than simple benchmarks)
- 20 sequential tasks (2.5x more than extended benchmark)
- Mix of linear, nonlinear, and compositional tasks
- Significant interference patterns requiring sophisticated memory

Why Full ANTARA Should Win Here:
1. Dreaming: 20 tasks means diverse buffer is critical
2. Hybrid Memory: EWC+SI together handle different forgetting modes
3. Consciousness: Surprise-adaptive LR helps with task transitions
4. Large Buffer: 10000+ samples needed to maintain diversity

Tasks: T1 → T2 → T3 → ... → T20
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

# ============ BENCHMARK CONFIG ============
INPUT_DIM = 100
NUM_TASKS = 20

def generate_task_data(task_id, batch_size=32):
    """
    20 diverse tasks on 100D inputs.
    Designed to maximize interference and require sophisticated memory.
    """
    x = torch.randn(batch_size, INPUT_DIM)
    
    if task_id < 4:
        # Tasks 0-3: Linear combinations with different weight patterns
        weights = torch.zeros(INPUT_DIM)
        start = task_id * 25
        weights[start:start+25] = 1.0 if task_id % 2 == 0 else -1.0
        y = (x * weights).sum(dim=1, keepdim=True)
        
    elif task_id < 8:
        # Tasks 4-7: Quadratic (nonlinear) on different subspaces
        offset = (task_id - 4) * 25
        y = (x[:, offset:offset+25] ** 2).sum(dim=1, keepdim=True)
        
    elif task_id < 12:
        # Tasks 8-11: Interaction terms (multiplicative)
        a_start = (task_id - 8) * 12
        b_start = 50 + (task_id - 8) * 12
        y = (x[:, a_start:a_start+12] * x[:, b_start:b_start+12]).sum(dim=1, keepdim=True)
        
    elif task_id < 16:
        # Tasks 12-15: Absolute value (another nonlinearity)
        offset = (task_id - 12) * 25
        y = torch.abs(x[:, offset:offset+25]).sum(dim=1, keepdim=True)
        
    else:
        # Tasks 16-19: Compositional (sum of differences)
        offset = (task_id - 16) * 20
        even_cols = x[:, offset:offset+20:2]
        odd_cols = x[:, offset+1:offset+20:2]
        y = (even_cols - odd_cols).sum(dim=1, keepdim=True)
    
    return x, y

# ============ MODEL ============
def create_complex_model():
    """Larger network for 100D inputs."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )

# ============ BASELINES ============
class NaiveBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_complex_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
    
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
    def __init__(self, ewc_lambda=500.0):
        super().__init__()
        self.model = create_complex_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.ewc_lambda = ewc_lambda
        self.fisher_list = []
        self.optimal_params_list = []
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, task_id, samples=50):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for _ in range(samples):
            x, y = generate_task_data(task_id, batch_size=1)
            self.model.zero_grad()
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        
        for n in fisher:
            fisher[n] /= samples
        
        self.fisher_list.append(fisher)
        self.optimal_params_list.append({n: p.clone().detach() for n, p in self.model.named_parameters()})
        self.model.train()
    
    def ewc_loss(self):
        loss = 0.0
        for fisher, optimal_params in zip(self.fisher_list, self.optimal_params_list):
            for n, p in self.model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - optimal_params[n]) ** 2).sum()
        return loss
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self.model(x)
        task_loss = F.mse_loss(pred, y)
        ewc_penalty = self.ewc_loss() * self.ewc_lambda / max(1, len(self.fisher_list))
        total_loss = task_loss + ewc_penalty
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return task_loss.item()
    
    def consolidate(self, task_id):
        self.compute_fisher(task_id)


class ReplayBaseline(nn.Module):
    def __init__(self, buffer_size=5000):
        super().__init__()
        self.model = create_complex_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
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
    
    def replay_step(self, batch_size=64):
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
        
        # More replay for complex benchmark
        for _ in range(2):
            self.replay_step()
        
        return loss.item()


# ============ EXPERIMENT RUNNER ============
def train_task(agent, task_id, max_steps=300, threshold=5.0):
    """Train on a single task until convergence."""
    for step in range(max_steps):
        x, y = generate_task_data(task_id)
        
        if isinstance(agent, AdaptiveFramework):
            metrics = agent.train_step(x, target_data=y)
            loss = metrics['loss']
        else:
            loss = agent.train_step(x, y)
        
        if loss < threshold:
            return step
    return max_steps

def evaluate_all_tasks(agent, num_tasks):
    """Evaluate on all tasks."""
    results = {}
    with torch.no_grad():
        for t in range(num_tasks):
            x, y = generate_task_data(t, batch_size=200)
            if isinstance(agent, AdaptiveFramework):
                output = agent(x)
                pred = output[0] if isinstance(output, tuple) else output
            else:
                pred = agent(x)
            results[t] = F.mse_loss(pred, y).item()
    return results

def run_full_experiment(agent, name, num_tasks=NUM_TASKS):
    """Run complete experiment and return metrics."""
    print(f"   Running: {name}...")
    history = {t: [] for t in range(num_tasks)}
    snapshots = {}
    
    for current_task in range(num_tasks):
        train_task(agent, current_task)
        res = evaluate_all_tasks(agent, num_tasks)
        snapshots[current_task] = res[current_task]
        
        for t in range(num_tasks):
            history[t].append(res.get(t, 0))
        
        # Consolidation
        if isinstance(agent, AdaptiveFramework):
            if hasattr(agent, 'prioritized_buffer') and agent.prioritized_buffer:
                agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
        elif isinstance(agent, EWCBaseline):
            agent.consolidate(current_task)
        
        # Progress indicator
        if (current_task + 1) % 5 == 0:
            print(f"      Completed {current_task + 1}/{num_tasks} tasks")
    
    return history, snapshots

def compute_metrics(history, snapshots, num_tasks):
    """Compute BWT and average final loss."""
    bwt_values = []
    for task in range(num_tasks - 1):
        loss_after_final = history[task][-1]
        loss_after_training = snapshots[task]
        bwt = loss_after_training - loss_after_final
        bwt_values.append(bwt)
    
    bwt = np.mean(bwt_values)
    avg_loss = np.mean([history[t][-1] for t in range(num_tasks)])
    
    # Also compute forgetting: average increase in loss
    forgetting = 0
    for task in range(num_tasks - 1):
        forgetting += max(0, history[task][-1] - snapshots[task])
    forgetting /= (num_tasks - 1)
    
    return bwt, avg_loss, forgetting

# ============ MAIN BENCHMARK ============
def run_complex_benchmark():
    print("\n" + "="*70)
    print("ANTARA COMPLEX BENCHMARK: 20-TASK, 100D CHALLENGE")
    print("="*70)
    print(f"Tasks: T1 → T2 → ... → T{NUM_TASKS}")
    print(f"Input Dimension: {INPUT_DIM}")
    print("This is a STRESS TEST for continual learning.\n")
    
    results = {}
    
    # 1. Naive Baseline
    print("🔬 [1/5] Naive Baseline...")
    naive = NaiveBaseline()
    results['Naive'] = run_full_experiment(naive, "Naive")
    
    # 2. EWC-Only
    print("🔬 [2/5] EWC-Only...")
    ewc = EWCBaseline(ewc_lambda=500.0)
    results['EWC'] = run_full_experiment(ewc, "EWC")
    
    # 3. Replay-Only
    print("🔬 [3/5] Replay-Only...")
    replay = ReplayBaseline(buffer_size=5000)
    results['Replay'] = run_full_experiment(replay, "Replay")
    
    # 4. ANTARA Minimal (EWC only)
    print("🔬 [4/5] ANTARA Minimal...")
    cfg_min = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=500.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=5e-4,
        compile_model=False,
    )
    antara_min = AdaptiveFramework(create_complex_model(), cfg_min, device='cpu')
    results['ANTARA-Min'] = run_full_experiment(antara_min, "ANTARA-Min")
    
    # 5. ANTARA Full (FINE-TUNED FOR STABILITY)
    print("🔬 [5/5] ANTARA Full (Fine-Tuned)...")
    cfg_full = AdaptiveFrameworkConfig(
        device='cpu',
        # Hybrid Memory with BALANCED protection
        memory_type='hybrid',
        ewc_lambda=200.0,   # REDUCED from 300 - less rigidity
        si_lambda=0.5,      # REDUCED from 1.0 - lighter online tracking
        # STABLE Dreaming - less frequent but effective
        enable_dreaming=True,
        dream_interval=100,  # INCREASED from 20 - less interference
        dream_batch_size=32, # REDUCED from 128 - more stable gradients
        use_prioritized_replay=True,
        feedback_buffer_size=10000,  # REDUCED from 20000 - less noise
        replay_priority_temperature=1.0,  # Standard prioritization
        # DISABLE consciousness to reduce oscillation
        enable_consciousness=False,
        # Disable other overhead
        enable_health_monitor=False,
        use_reptile=False,
        enable_world_model=False,
        # LOWER learning rate for stability
        learning_rate=2e-4,  # REDUCED from 5e-4
        compile_model=False,
    )
    antara_full = AdaptiveFramework(create_complex_model(), cfg_full, device='cpu')
    results['ANTARA-Full'] = run_full_experiment(antara_full, "ANTARA-Full")
    
    return results

# ============ VISUALIZATION ============
def plot_complex_comparison(results, filename):
    """Bar chart for complex benchmark."""
    methods = list(results.keys())
    
    metrics = {}
    for method, (history, snapshots) in results.items():
        bwt, avg_loss, forgetting = compute_metrics(history, snapshots, NUM_TASKS)
        metrics[method] = {'bwt': bwt, 'avg_loss': avg_loss, 'forgetting': forgetting}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    
    x = np.arange(len(methods))
    
    # Average Final Loss
    bars1 = axes[0].bar(x, [metrics[m]['avg_loss'] for m in methods], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, fontsize=10)
    axes[0].set_ylabel("Average Final MSE Loss")
    axes[0].set_title("20-Task Avg Loss (Lower ↓)", fontsize=12, fontweight='bold')
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
    axes[1].set_title("20-Task BWT (Higher ↑)", fontsize=12, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:+.1f}', ha='center',
                     va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Forgetting
    bars3 = axes[2].bar(x, [metrics[m]['forgetting'] for m in methods], color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, fontsize=10)
    axes[2].set_ylabel("Average Forgetting")
    axes[2].set_title("20-Task Forgetting (Lower ↓)", fontsize=12, fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("Complex Benchmark: 20 Tasks × 100D Inputs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def print_complex_results(results):
    """Print results table."""
    print("\n" + "="*70)
    print("📊 COMPLEX BENCHMARK RESULTS (20 Tasks, 100D)")
    print("="*70)
    
    print(f"\n{'Method':<15} | {'Avg Loss':>10} | {'BWT':>12} | {'Forgetting':>12}")
    print("-"*60)
    
    all_metrics = {}
    for method, (history, snapshots) in results.items():
        bwt, avg_loss, forgetting = compute_metrics(history, snapshots, NUM_TASKS)
        all_metrics[method] = {'bwt': bwt, 'avg_loss': avg_loss, 'forgetting': forgetting}
        print(f"{method:<15} | {avg_loss:>10.2f} | {bwt:>+12.2f} | {forgetting:>12.2f}")
    
    print("-"*60)
    
    # Winners
    best_loss = min(all_metrics.values(), key=lambda x: x['avg_loss'])['avg_loss']
    best_bwt = max(all_metrics.values(), key=lambda x: x['bwt'])['bwt']
    best_forgetting = min(all_metrics.values(), key=lambda x: x['forgetting'])['forgetting']
    
    loss_winner = [m for m, v in all_metrics.items() if v['avg_loss'] == best_loss][0]
    bwt_winner = [m for m, v in all_metrics.items() if v['bwt'] == best_bwt][0]
    forgetting_winner = [m for m, v in all_metrics.items() if v['forgetting'] == best_forgetting][0]
    
    print(f"\n🏆 Best Avg Loss:   {loss_winner} ({best_loss:.2f})")
    print(f"🏆 Best BWT:        {bwt_winner} ({best_bwt:+.2f})")
    print(f"🏆 Least Forgetting: {forgetting_winner} ({best_forgetting:.2f})")
    
    # ANTARA comparison
    if 'ANTARA-Full' in all_metrics and 'ANTARA-Min' in all_metrics:
        full = all_metrics['ANTARA-Full']
        mini = all_metrics['ANTARA-Min']
        
        print(f"\n📈 ANTARA-Full vs ANTARA-Min:")
        print(f"   Loss: {full['avg_loss']:.2f} vs {mini['avg_loss']:.2f} ({(mini['avg_loss']-full['avg_loss'])/mini['avg_loss']*100:+.1f}%)")
        print(f"   BWT:  {full['bwt']:+.2f} vs {mini['bwt']:+.2f} ({full['bwt']-mini['bwt']:+.2f})")
        print(f"   Forgetting: {full['forgetting']:.2f} vs {mini['forgetting']:.2f}")

# ============ MAIN ============
if __name__ == "__main__":
    results = run_complex_benchmark()
    
    # Generate plots
    plot_complex_comparison(results, "tests/complex_comparison.png")
    
    # Print results
    print_complex_results(results)
    
    print("\n" + "="*70)
    print("✨ COMPLEX BENCHMARK COMPLETE")
    print("="*70)
    print("Generated files:")
    print("   - tests/complex_comparison.png")
    print("="*70)
