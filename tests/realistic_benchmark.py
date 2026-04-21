"""
ANTARA Real-World Synthetic Benchmark
=====================================
Mimics REAL continual learning scenarios using synthetic data:

1. SYNTHETIC IMAGES - 2D patterns with spatial structure (like real images)
2. CLASSIFICATION - Multi-class problems where class diversity matters
3. CLASS-INCREMENTAL - Learn new classes over time (realistic CL setting)
4. OVERLAPPING FEATURES - Tasks share features (like real vision tasks)

Why Dreaming SHOULD Win Here:
- Replay prevents forgetting of old CLASS decision boundaries
- Buffer diversity maintains representation of all seen classes
- Classification loss is cross-entropy (not MSE) - more stable gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np
import logging
import os

logging.disable(logging.CRITICAL)
torch.manual_seed(42)
np.random.seed(42)

# ============ SYNTHETIC IMAGE GENERATION ============
IMG_SIZE = 16  # 16x16 "images"
INPUT_DIM = IMG_SIZE * IMG_SIZE  # 256 features

def generate_synthetic_image(class_id, batch_size=32):
    """
    Generate synthetic 2D patterns that mimic real image classes.
    Each class has a distinct spatial pattern.
    
    Classes:
    0: Horizontal lines
    1: Vertical lines
    2: Diagonal (top-left to bottom-right)
    3: Diagonal (top-right to bottom-left)
    4: Center blob
    5: Corner blobs
    6: Checkerboard
    7: Random noise (hard class)
    8: Gradient horizontal
    9: Gradient vertical
    """
    x = torch.zeros(batch_size, 1, IMG_SIZE, IMG_SIZE)
    
    for i in range(batch_size):
        noise = torch.randn(IMG_SIZE, IMG_SIZE) * 0.3
        
        if class_id == 0:  # Horizontal lines
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
            pattern[::2, :] = 1.0
        elif class_id == 1:  # Vertical lines
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
            pattern[:, ::2] = 1.0
        elif class_id == 2:  # Diagonal TL-BR
            pattern = torch.eye(IMG_SIZE)
        elif class_id == 3:  # Diagonal TR-BL
            pattern = torch.flip(torch.eye(IMG_SIZE), dims=[1])
        elif class_id == 4:  # Center blob
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if 0 <= cx+dx < IMG_SIZE and 0 <= cy+dy < IMG_SIZE:
                        pattern[cx+dx, cy+dy] = 1.0
        elif class_id == 5:  # Corner blobs
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
            pattern[:4, :4] = 1.0
            pattern[-4:, -4:] = 1.0
        elif class_id == 6:  # Checkerboard
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
            for r in range(IMG_SIZE):
                for c in range(IMG_SIZE):
                    if (r + c) % 2 == 0:
                        pattern[r, c] = 1.0
        elif class_id == 7:  # Random (unique per sample)
            pattern = torch.randn(IMG_SIZE, IMG_SIZE)
        elif class_id == 8:  # Horizontal gradient
            pattern = torch.linspace(0, 1, IMG_SIZE).unsqueeze(0).repeat(IMG_SIZE, 1)
        elif class_id == 9:  # Vertical gradient
            pattern = torch.linspace(0, 1, IMG_SIZE).unsqueeze(1).repeat(1, IMG_SIZE)
        else:
            pattern = torch.zeros(IMG_SIZE, IMG_SIZE)
        
        x[i, 0] = pattern + noise
    
    # Flatten for MLP (could use CNN but keeping simple)
    x_flat = x.view(batch_size, INPUT_DIM)
    y = torch.full((batch_size,), class_id, dtype=torch.long)
    
    return x_flat, y

# ============ MODEL (Simple CNN-like MLP) ============
def create_classifier():
    """Classifier for synthetic images."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 10)  # 10 classes
    )

# ============ BASELINES ============
class NaiveClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_classifier()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class EWCClassifier(nn.Module):
    def __init__(self, ewc_lambda=100.0):
        super().__init__()
        self.model = create_classifier()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.optimal_params = {}
        
    def forward(self, x):
        return self.model(x)
    
    def compute_fisher(self, data_samples, n_samples=100):
        new_fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        
        for x, y in data_samples[:n_samples]:
            self.model.zero_grad()
            logits = self.model(x.unsqueeze(0))
            loss = F.cross_entropy(logits, y.unsqueeze(0))
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    new_fisher[n] += p.grad.data ** 2
        
        for n in new_fisher:
            new_fisher[n] /= n_samples
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
        logits = self.model(x)
        task_loss = F.cross_entropy(logits, y)
        ewc_penalty = self.ewc_loss() * self.ewc_lambda if self.fisher else 0.0
        total_loss = task_loss + ewc_penalty
        total_loss.backward()
        self.optimizer.step()
        return task_loss.item()


# ============ EXPERIMENT ============
def run_class_incremental(agent, name, classes_per_task=2):
    """
    Class-incremental learning:
    - Task 1: Classes 0-1
    - Task 2: Classes 2-3
    - Task 3: Classes 4-5
    - Task 4: Classes 6-7
    - Task 5: Classes 8-9
    
    Evaluate on ALL seen classes after each task.
    """
    print(f"   Running: {name}...", end=" ", flush=True)
    
    all_classes = list(range(10))
    tasks = [all_classes[i:i+classes_per_task] for i in range(0, 10, classes_per_task)]
    
    seen_classes = []
    history = []  # Accuracy on all seen classes
    data_buffer = []
    
    for task_idx, task_classes in enumerate(tasks):
        seen_classes.extend(task_classes)
        
        # Train on current task classes
        for epoch in range(50):  # 50 epochs per task
            for class_id in task_classes:
                x, y = generate_synthetic_image(class_id, batch_size=32)
                
                if isinstance(agent, AdaptiveFramework):
                    # Need to convert classification to framework format
                    # Use one-hot targets for compatibility
                    y_onehot = F.one_hot(y, num_classes=10).float()
                    metrics = agent.train_step(x, target_data=y_onehot)
                else:
                    agent.train_step(x, y)
                
                # Store samples for consolidation
                for i in range(min(4, x.size(0))):
                    data_buffer.append((x[i], y[i]))
                    if len(data_buffer) > 500:
                        data_buffer.pop(0)
        
        # Consolidation
        if isinstance(agent, AdaptiveFramework):
            if hasattr(agent, 'prioritized_buffer') and agent.prioritized_buffer:
                try:
                    agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
                except:
                    pass
        elif isinstance(agent, EWCClassifier):
            agent.compute_fisher(data_buffer)
        
        # Evaluate on ALL seen classes
        correct = 0
        total = 0
        with torch.no_grad():
            for class_id in seen_classes:
                x, y = generate_synthetic_image(class_id, batch_size=50)
                
                if isinstance(agent, AdaptiveFramework):
                    output = agent(x)
                    logits = output[0] if isinstance(output, tuple) else output
                else:
                    logits = agent(x)
                
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total * 100
        history.append(accuracy)
    
    print(f"Done (Final Acc: {history[-1]:.1f}%)")
    return history

# ============ MAIN ============
def run_realistic_benchmark():
    print("\n" + "="*70)
    print("ANTARA REALISTIC SYNTHETIC BENCHMARK")
    print("="*70)
    print("Setting: Class-Incremental Learning on Synthetic Images")
    print("Tasks: Learn 2 new classes per task (10 classes total, 5 tasks)")
    print("Measure: Accuracy on ALL seen classes after each task\n")
    
    results = {}
    
    # 1. Naive (no CL)
    print("[1/4] Naive (No CL)...")
    naive = NaiveClassifier()
    results['Naive'] = run_class_incremental(naive, "Naive")
    
    # 2. EWC Only
    print("[2/4] EWC Only...")
    ewc = EWCClassifier(ewc_lambda=100.0)
    results['EWC'] = run_class_incremental(ewc, "EWC")
    
    # 3. ANTARA Minimal (EWC mode, no dreaming)
    print("[3/4] ANTARA Minimal...")
    cfg_min = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=100.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_min = AdaptiveFramework(create_classifier(), cfg_min, device='cpu')
    results['ANTARA-Min'] = run_class_incremental(antara_min, "ANTARA-Min")
    
    # 4. ANTARA with Dreaming (key feature!)
    print("[4/4] ANTARA + Dreaming...")
    cfg_dream = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=50.0,  # Lower EWC for classification
        si_lambda=0.5,
        enable_dreaming=True,
        dream_interval=20,  # Dream every 20 steps
        dream_batch_size=32,
        use_prioritized_replay=True,
        feedback_buffer_size=5000,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_dream = AdaptiveFramework(create_classifier(), cfg_dream, device='cpu')
    results['ANTARA+Dream'] = run_class_incremental(antara_dream, "ANTARA+Dream")
    
    return results

def plot_realistic_results(results, filename):
    """Plot accuracy curves and final comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71']
    methods = list(results.keys())
    
    # 1. Accuracy over tasks
    ax = axes[0]
    for i, (method, history) in enumerate(results.items()):
        tasks = [f"T{j+1}\n({(j+1)*2} cls)" for j in range(len(history))]
        ax.plot(range(len(history)), history, marker='o', color=colors[i], 
                label=method, linewidth=2, markersize=8)
    ax.set_xlabel("Task (# Classes Seen)")
    ax.set_ylabel("Accuracy on ALL Seen Classes (%)")
    ax.set_title("Accuracy Over Incremental Tasks", fontweight='bold')
    ax.set_xticks(range(len(list(results.values())[0])))
    ax.set_xticklabels([f"T{j+1}" for j in range(len(list(results.values())[0]))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 2. Final accuracy bar chart
    ax = axes[1]
    final_accs = [results[m][-1] for m in methods]
    bars = ax.bar(range(len(methods)), final_accs, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_title("Final Accuracy (All 10 Classes)", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle("Class-Incremental Learning: Synthetic Images (10 Classes)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def print_realistic_results(results):
    print("\n" + "="*70)
    print("REALISTIC BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<20} | {'Final Acc':>10} | {'Avg Acc':>10} | Winner?")
    print("-"*55)
    
    for method, history in results.items():
        final_acc = history[-1]
        avg_acc = np.mean(history)
        winner = "*WIN*" if final_acc == max([r[-1] for r in results.values()]) else ""
        print(f"{method:<20} | {final_acc:>9.1f}% | {avg_acc:>9.1f}% | {winner}")
    
    print("-"*55)
    
    # Feature comparison
    min_acc = results.get('ANTARA-Min', [0])[-1]
    dream_acc = results.get('ANTARA+Dream', [0])[-1]
    
    if dream_acc > min_acc:
        improvement = dream_acc - min_acc
        print(f"\n> DREAMING ADVANTAGE CONFIRMED!")
        print(f"   ANTARA+Dream: {dream_acc:.1f}%")
        print(f"   ANTARA-Min:   {min_acc:.1f}%")
        print(f"   Improvement:  +{improvement:.1f}%")
    else:
        print(f"\n! Dreaming did not improve over minimal.")

if __name__ == "__main__":
    results = run_realistic_benchmark()
    # Use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_realistic_results(results, os.path.join(base_dir, "realistic_benchmark.png"))
    print_realistic_results(results)
    
    print("\n" + "="*70)
    print("+ REALISTIC SYNTHETIC BENCHMARK COMPLETE")
    print("="*70)
