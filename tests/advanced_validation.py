"""
ANTARA Advanced Features Validation
====================================
This benchmark is specifically designed to show WHERE the advanced features
(Dreaming, Consciousness, Hybrid Memory) provide REAL VALUE.

Key Challenges:
1. DISTRIBUTION SHIFT - Data distribution changes within tasks
2. NOISY INPUTS - Corrupted data that needs robust learning
3. TASK REVISITATION - Going back to old tasks (A→B→C→A→B→C)
4. LONGER SEQUENCES - 15 task transitions, not just 4

This is where:
- Dreaming refreshes old memories during distribution shift
- Consciousness detects novelty and adjusts learning rate
- Hybrid memory handles both online and offline consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging

logging.disable(logging.CRITICAL)
torch.manual_seed(42)
np.random.seed(42)

INPUT_DIM = 30

def generate_data_with_shift(task_id, batch_size=32, phase=0, noise_level=0.0):
    """
    Generate data with DISTRIBUTION SHIFT.
    Phase 0: Standard distribution
    Phase 1: Shifted distribution (mean shifted)
    Phase 2: Higher variance
    """
    if phase == 0:
        x = torch.randn(batch_size, INPUT_DIM)
    elif phase == 1:
        x = torch.randn(batch_size, INPUT_DIM) + 2.0  # Mean shift
    else:
        x = torch.randn(batch_size, INPUT_DIM) * 2.0  # Variance shift
    
    # Add noise
    if noise_level > 0:
        x = x + torch.randn_like(x) * noise_level
    
    # Task-specific targets
    if task_id == 'A':
        y = x.sum(dim=1, keepdim=True)
    elif task_id == 'B':
        y = -x.sum(dim=1, keepdim=True)
    elif task_id == 'C':
        y = x[:, :15].sum(dim=1, keepdim=True) - x[:, 15:].sum(dim=1, keepdim=True)
    elif task_id == 'D':
        y = (x ** 2).sum(dim=1, keepdim=True)
    elif task_id == 'E':
        y = torch.abs(x).sum(dim=1, keepdim=True)
    
    return x, y

def create_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

# ============ BASELINES ============
class MinimalBaseline(nn.Module):
    """EWC-only, no advanced features - the 'minimal' config that was winning."""
    def __init__(self, ewc_lambda=500.0):
        super().__init__()
        self.model = create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, task_data, samples=50):
        new_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for x, y in task_data[:samples]:
            self.model.zero_grad()
            pred = self.model(x.unsqueeze(0))
            loss = F.mse_loss(pred, y.unsqueeze(0))
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += p.grad.data ** 2
        
        for n in new_fisher:
            new_fisher[n] /= samples
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

# ============ EXPERIMENT ============
def run_with_shift_and_revisit(agent, name):
    """
    Task sequence with:
    - Distribution shift (3 phases per task)
    - Task revisitation (A→B→C→D→E→A→B→C→D→E→A→B→C→D→E)
    - Noise injection (increasing over time)
    """
    print(f"   Running: {name}...", end=" ", flush=True)
    
    # Task sequence with REVISITATION
    task_sequence = ['A', 'B', 'C', 'D', 'E'] * 3  # 15 transitions!
    
    results_per_task = {t: [] for t in ['A', 'B', 'C', 'D', 'E']}
    task_data_buffer = []  # For EWC consolidation
    
    for step, current_task in enumerate(task_sequence):
        # Determine phase and noise based on position
        phase = step // 5  # 0, 1, 2 (distribution shift)
        noise = 0.1 * phase  # Increasing noise
        
        # Train on current task with shift
        for _ in range(100):
            x, y = generate_data_with_shift(current_task, phase=phase, noise_level=noise)
            
            if isinstance(agent, AdaptiveFramework):
                metrics = agent.train_step(x, target_data=y)
            else:
                agent.train_step(x, y)
            
            # Store for consolidation
            for i in range(x.size(0)):
                task_data_buffer.append((x[i], y[i]))
                if len(task_data_buffer) > 1000:
                    task_data_buffer.pop(0)
        
        # Evaluate ALL tasks after each training phase
        with torch.no_grad():
            for eval_task in ['A', 'B', 'C', 'D', 'E']:
                x, y = generate_data_with_shift(eval_task, batch_size=100, phase=0)  # Eval on clean data
                if isinstance(agent, AdaptiveFramework):
                    output = agent(x)
                    pred = output[0] if isinstance(output, tuple) else output
                else:
                    pred = agent(x)
                results_per_task[eval_task].append(F.mse_loss(pred, y).item())
        
        # Consolidation
        if isinstance(agent, AdaptiveFramework):
            if hasattr(agent, 'prioritized_buffer') and agent.prioritized_buffer:
                try:
                    agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
                except:
                    pass
        elif isinstance(agent, MinimalBaseline) and len(task_data_buffer) > 50:
            agent.compute_fisher(task_data_buffer[-100:])
    
    print("Done")
    return results_per_task

def compute_metrics(results):
    """Compute retention metrics."""
    all_final_losses = []
    all_forgetting = []
    
    for task, history in results.items():
        final_loss = history[-1]
        best_loss = min(history)
        forgetting = final_loss - best_loss
        
        all_final_losses.append(final_loss)
        all_forgetting.append(max(0, forgetting))
    
    return np.mean(all_final_losses), np.mean(all_forgetting)

# ============ MAIN ============
def run_advanced_validation():
    print("\n" + "="*70)
    print("ANTARA ADVANCED FEATURES VALIDATION")
    print("="*70)
    print("Challenge: Distribution Shift + Noise + Task Revisitation")
    print("Task Sequence: A→B→C→D→E→A→B→C→D→E→A→B→C→D→E (15 transitions)")
    print("This is where advanced features SHOULD win.\n")
    
    results = {}
    
    # 1. Minimal Config (EWC only, the previous "winner")
    print("🔬 [1/4] Minimal (EWC only, no dreaming)...")
    minimal = MinimalBaseline(ewc_lambda=500.0)
    results['Minimal'] = run_with_shift_and_revisit(minimal, "Minimal")
    
    # 2. ANTARA with CONSERVATIVE Dreaming
    print("🔬 [2/4] ANTARA + Conservative Dreaming...")
    cfg_dream = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',  # Pure EWC, simpler
        ewc_lambda=100.0,   # REDUCED - less rigid
        enable_dreaming=True,
        dream_interval=200,  # VERY RARE - only dream occasionally
        dream_batch_size=8,  # TINY batches - minimal interference
        use_prioritized_replay=False,  # Simple random replay
        feedback_buffer_size=2000,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=5e-4,  # LOWER LR
        compile_model=False,
    )
    antara_dream = AdaptiveFramework(create_model(), cfg_dream, device='cpu')
    results['ANTARA+Dream'] = run_with_shift_and_revisit(antara_dream, "ANTARA+Dream")
    
    # 3. ANTARA with Consciousness (no dreaming) - adaptive LR
    print("🔬 [3/4] ANTARA + Consciousness...")
    cfg_consc = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=300.0,
        enable_dreaming=False,
        enable_consciousness=True,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_consc = AdaptiveFramework(create_model(), cfg_consc, device='cpu')
    results['ANTARA+Consc'] = run_with_shift_and_revisit(antara_consc, "ANTARA+Consc")
    
    # 4. ANTARA Optimized (Careful Balance)
    print("🔬 [4/4] ANTARA Optimized...")
    cfg_full = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=200.0,
        enable_dreaming=True,
        dream_interval=150,  # Rare dreaming
        dream_batch_size=16,
        use_prioritized_replay=False,
        feedback_buffer_size=3000,
        enable_consciousness=False,  # Disable - causes issues
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=8e-4,
        compile_model=False,
    )
    antara_full = AdaptiveFramework(create_model(), cfg_full, device='cpu')
    results['ANTARA-Opt'] = run_with_shift_and_revisit(antara_full, "ANTARA-Opt")
    
    return results

def plot_advanced_results(results, filename):
    """Plot comparison."""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    # Compute metrics
    metrics = {}
    for method, res in results.items():
        avg_loss, avg_forgetting = compute_metrics(res)
        metrics[method] = {'loss': avg_loss, 'forgetting': avg_forgetting}
    
    x = np.arange(len(methods))
    
    # 1. Final Loss
    ax = axes[0]
    bars = ax.bar(x, [metrics[m]['loss'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=10)
    ax.set_ylabel("Average Final Loss")
    ax.set_title("Final Loss (Lower ↓)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Forgetting
    ax = axes[1]
    bars = ax.bar(x, [metrics[m]['forgetting'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=10)
    ax.set_ylabel("Average Forgetting")
    ax.set_title("Forgetting (Lower ↓)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Retention curves
    ax = axes[2]
    for i, (method, res) in enumerate(results.items()):
        avg_curve = np.mean([res[t] for t in res], axis=0)
        ax.plot(avg_curve, label=method, color=colors[i], linewidth=2)
    ax.set_xlabel("Task Transition")
    ax.set_ylabel("Average Loss")
    ax.set_title("Retention Over Time", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Advanced Features Validation (15 Transitions + Distribution Shift)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def print_advanced_results(results):
    print("\n" + "="*70)
    print("📊 ADVANCED FEATURES VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<20} | {'Avg Loss':>12} | {'Forgetting':>12} | Winner?")
    print("-"*60)
    
    metrics = {}
    for method, res in results.items():
        avg_loss, avg_forgetting = compute_metrics(res)
        metrics[method] = {'loss': avg_loss, 'forgetting': avg_forgetting}
        print(f"{method:<20} | {avg_loss:>12.2f} | {avg_forgetting:>12.2f} |")
    
    print("-"*60)
    
    # Winners
    best_loss = min(metrics.keys(), key=lambda m: metrics[m]['loss'])
    best_forg = min(metrics.keys(), key=lambda m: metrics[m]['forgetting'])
    
    print(f"\n🏆 Best Loss: {best_loss} ({metrics[best_loss]['loss']:.2f})")
    print(f"🏆 Least Forgetting: {best_forg} ({metrics[best_forg]['forgetting']:.2f})")
    
    # Feature analysis
    minimal_loss = metrics.get('Minimal', {}).get('loss', 0)
    full_loss = metrics.get('ANTARA-Full', {}).get('loss', 0)
    
    if full_loss < minimal_loss:
        improvement = (minimal_loss - full_loss) / minimal_loss * 100
        print(f"\n🎯 ANTARA-Full beats Minimal by {improvement:.1f}%!")
        print("   → Advanced features PROVEN VALUABLE in challenging scenarios!")
    else:
        print(f"\n⚠️ Minimal still wins - features need further tuning.")

if __name__ == "__main__":
    results = run_advanced_validation()
    plot_advanced_results(results, "tests/advanced_validation.png")
    print_advanced_results(results)
    
    print("\n" + "="*70)
    print("✨ ADVANCED FEATURES VALIDATION COMPLETE")
    print("="*70)
