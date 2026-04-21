"""
ANTARA Comprehensive Preset Benchmark
=====================================
Tests ALL available framework presets and custom configurations.
Identifies optimal settings for different use cases.

Presets Tested:
1. production - Full features, production-ready
2. balanced - Good trade-off speed/accuracy
3. fast - Minimal overhead, maximum speed
4. memory_efficient - Low memory footprint
5. accuracy_focus - Maximum retention at cost of speed
6. exploration - High adaptability
7. creativity_boost - Novel solution finding
8. stable - Conservative, stable training
9. research - Full logging and debugging
10. real_time - Ultra-low latency

Also tests:
- Custom tuned configs
- Baseline comparisons

Outputs:
- Comprehensive comparison table
- Best config recommendations
- Publication-ready plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airborne_antara.presets import PRESETS
import numpy as np
import logging
import time

# Suppress logging
logging.disable(logging.CRITICAL)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ BENCHMARK CONFIG ============
INPUT_DIM = 50
NUM_TASKS = 10  # Moderate for speed

def generate_task_data(task_id, batch_size=32):
    """10 diverse tasks on 50D inputs."""
    x = torch.randn(batch_size, INPUT_DIM)
    
    if task_id == 0:
        y = x.sum(dim=1, keepdim=True)
    elif task_id == 1:
        y = -x.sum(dim=1, keepdim=True)
    elif task_id == 2:
        y = x[:, :25].sum(dim=1, keepdim=True)
    elif task_id == 3:
        y = x[:, 25:].sum(dim=1, keepdim=True)
    elif task_id == 4:
        y = (x ** 2).sum(dim=1, keepdim=True)
    elif task_id == 5:
        y = torch.abs(x).sum(dim=1, keepdim=True)
    elif task_id == 6:
        y = (x[:, :25] * x[:, 25:]).sum(dim=1, keepdim=True)
    elif task_id == 7:
        y = x[:, ::2].sum(dim=1, keepdim=True) - x[:, 1::2].sum(dim=1, keepdim=True)
    elif task_id == 8:
        y = (x[:, :10] + x[:, 10:20] - x[:, 20:30]).sum(dim=1, keepdim=True)
    else:
        y = torch.sin(x).sum(dim=1, keepdim=True)
    
    return x, y

def create_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )

# ============ BASELINE ============
class NaiveBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model()
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

# ============ EXPERIMENT RUNNER ============
def train_task(agent, task_id, max_steps=200, threshold=3.0):
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

def evaluate_all(agent, num_tasks):
    results = {}
    with torch.no_grad():
        for t in range(num_tasks):
            x, y = generate_task_data(t, batch_size=100)
            if isinstance(agent, AdaptiveFramework):
                output = agent(x)
                pred = output[0] if isinstance(output, tuple) else output
            else:
                pred = agent(x)
            results[t] = F.mse_loss(pred, y).item()
    return results

def run_experiment(agent, name, num_tasks=NUM_TASKS):
    print(f"   Testing: {name}...", end=" ", flush=True)
    start_time = time.time()
    
    history = {t: [] for t in range(num_tasks)}
    snapshots = {}
    
    for current_task in range(num_tasks):
        train_task(agent, current_task)
        res = evaluate_all(agent, num_tasks)
        snapshots[current_task] = res[current_task]
        
        for t in range(num_tasks):
            history[t].append(res.get(t, 0))
        
        # Consolidation
        if isinstance(agent, AdaptiveFramework):
            if hasattr(agent, 'prioritized_buffer') and agent.prioritized_buffer:
                try:
                    agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
                except:
                    pass
    
    elapsed = time.time() - start_time
    print(f"Done ({elapsed:.1f}s)")
    
    return history, snapshots, elapsed

def compute_metrics(history, snapshots, num_tasks):
    """Compute BWT, avg loss, and forgetting."""
    bwt_values = []
    for task in range(num_tasks - 1):
        loss_after_final = history[task][-1]
        loss_after_training = snapshots[task]
        bwt = loss_after_training - loss_after_final
        bwt_values.append(bwt)
    
    bwt = np.mean(bwt_values)
    avg_loss = np.mean([history[t][-1] for t in range(num_tasks)])
    
    forgetting = 0
    for task in range(num_tasks - 1):
        forgetting += max(0, history[task][-1] - snapshots[task])
    forgetting /= (num_tasks - 1)
    
    return bwt, avg_loss, forgetting

# ============ PRESET CONFIGS ============
def get_all_preset_configs():
    """Get all preset configurations for testing."""
    configs = {}
    
    # Standard presets
    preset_names = [
        'production', 'balanced', 'fast', 'memory_efficient',
        'accuracy_focus', 'exploration', 'creativity_boost', 
        'stable', 'research', 'real_time'
    ]
    
    for name in preset_names:
        try:
            preset_func = getattr(PRESETS, name, None)
            if preset_func:
                preset = preset_func()
                # Convert preset to AdaptiveFrameworkConfig
                cfg = AdaptiveFrameworkConfig(
                    device='cpu',
                    memory_type=getattr(preset, 'memory_type', 'hybrid'),
                    ewc_lambda=getattr(preset, 'ewc_lambda', 500.0),
                    si_lambda=getattr(preset, 'si_lambda', 1.0),
                    enable_dreaming=getattr(preset, 'enable_dreaming', True),
                    dream_interval=getattr(preset, 'dream_interval', 50),
                    enable_consciousness=False,  # Disable for fair comparison
                    enable_health_monitor=False,
                    use_reptile=False,
                    learning_rate=getattr(preset, 'learning_rate', 1e-3),
                    compile_model=False,
                )
                configs[name] = cfg
        except Exception as e:
            print(f"   Skipping {name}: {e}")
    
    # Add custom tuned configs
    configs['custom_minimal'] = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=500.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    
    configs['custom_balanced'] = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=300.0,
        si_lambda=0.5,
        enable_dreaming=True,
        dream_interval=50,
        dream_batch_size=32,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    
    configs['custom_aggressive'] = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=1000.0,
        si_lambda=2.0,
        enable_dreaming=True,
        dream_interval=20,
        dream_batch_size=64,
        use_prioritized_replay=True,
        enable_consciousness=False,
        enable_health_monitor=False,
        use_reptile=False,
        learning_rate=5e-4,
        compile_model=False,
    )
    
    return configs

# ============ MAIN BENCHMARK ============
def run_preset_benchmark():
    print("\n" + "="*70)
    print("ANTARA COMPREHENSIVE PRESET BENCHMARK")
    print("="*70)
    print(f"Tasks: T1 → T2 → ... → T{NUM_TASKS}")
    print(f"Input Dimension: {INPUT_DIM}")
    print("Testing ALL framework presets + custom configs.\n")
    
    results = {}
    configs_used = {}
    
    # 1. Naive Baseline
    print("🔬 [Baseline] Naive...")
    naive = NaiveBaseline()
    h, s, t = run_experiment(naive, "Naive")
    results['Naive'] = (h, s, t)
    
    # 2. All Presets
    preset_configs = get_all_preset_configs()
    
    for i, (name, cfg) in enumerate(preset_configs.items()):
        print(f"🔬 [{i+1}/{len(preset_configs)}] {name}...")
        try:
            agent = AdaptiveFramework(create_model(), cfg, device='cpu')
            h, s, t = run_experiment(agent, name)
            results[name] = (h, s, t)
            configs_used[name] = cfg
        except Exception as e:
            print(f"   FAILED: {e}")
    
    return results, configs_used

# ============ VISUALIZATION ============
def plot_preset_comparison(results, filename):
    """Comprehensive bar chart for all presets."""
    methods = list(results.keys())
    n = len(methods)
    
    metrics = {}
    for method, (history, snapshots, elapsed) in results.items():
        bwt, avg_loss, forgetting = compute_metrics(history, snapshots, NUM_TASKS)
        metrics[method] = {
            'bwt': bwt, 
            'avg_loss': avg_loss, 
            'forgetting': forgetting,
            'time': elapsed
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    x = np.arange(n)
    
    # 1. Average Loss
    ax = axes[0, 0]
    bars = ax.bar(x, [metrics[m]['avg_loss'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Average Final MSE Loss")
    ax.set_title("Average Loss (Lower ↓)", fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. BWT
    ax = axes[0, 1]
    bars = ax.bar(x, [metrics[m]['bwt'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Backward Transfer (BWT)")
    ax.set_title("BWT (Higher ↑)", fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Forgetting
    ax = axes[1, 0]
    bars = ax.bar(x, [metrics[m]['forgetting'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Average Forgetting")
    ax.set_title("Forgetting (Lower ↓)", fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Time
    ax = axes[1, 1]
    bars = ax.bar(x, [metrics[m]['time'] for m in methods], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Speed (Lower ↓)", fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f"Comprehensive Preset Comparison ({NUM_TASKS} Tasks × {INPUT_DIM}D)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def print_results_table(results, configs_used):
    """Print comprehensive results table."""
    print("\n" + "="*90)
    print("📊 COMPREHENSIVE PRESET BENCHMARK RESULTS")
    print("="*90)
    
    print(f"\n{'Preset':<20} | {'Avg Loss':>10} | {'BWT':>12} | {'Forgetting':>12} | {'Time (s)':>10}")
    print("-"*75)
    
    all_metrics = {}
    for method, (history, snapshots, elapsed) in results.items():
        bwt, avg_loss, forgetting = compute_metrics(history, snapshots, NUM_TASKS)
        all_metrics[method] = {
            'bwt': bwt, 
            'avg_loss': avg_loss, 
            'forgetting': forgetting,
            'time': elapsed
        }
        print(f"{method:<20} | {avg_loss:>10.2f} | {bwt:>+12.2f} | {forgetting:>12.2f} | {elapsed:>10.1f}")
    
    print("-"*75)
    
    # Find winners
    valid_methods = [m for m in all_metrics if m != 'Naive']
    if valid_methods:
        best_loss = min(valid_methods, key=lambda m: all_metrics[m]['avg_loss'])
        best_bwt = max(valid_methods, key=lambda m: all_metrics[m]['bwt'])
        best_forgetting = min(valid_methods, key=lambda m: all_metrics[m]['forgetting'])
        fastest = min(valid_methods, key=lambda m: all_metrics[m]['time'])
        
        print(f"\n🏆 WINNERS:")
        print(f"   Best Avg Loss:     {best_loss} ({all_metrics[best_loss]['avg_loss']:.2f})")
        print(f"   Best BWT:          {best_bwt} ({all_metrics[best_bwt]['bwt']:+.2f})")
        print(f"   Least Forgetting:  {best_forgetting} ({all_metrics[best_forgetting]['forgetting']:.2f})")
        print(f"   Fastest:           {fastest} ({all_metrics[fastest]['time']:.1f}s)")
    
    return all_metrics

def print_best_configs(configs_used, all_metrics):
    """Print best configuration recommendations."""
    print("\n" + "="*90)
    print("🎯 BEST CONFIGURATION RECOMMENDATIONS")
    print("="*90)
    
    # Sort by different criteria
    valid_configs = {k: v for k, v in configs_used.items() if k in all_metrics}
    
    if not valid_configs:
        print("No valid configs to recommend.")
        return
    
    # Best overall (lowest forgetting + reasonable loss)
    best_overall = min(valid_configs.keys(), 
                       key=lambda m: all_metrics[m]['forgetting'] + all_metrics[m]['avg_loss']/100)
    
    # Best for production (balance of metrics)
    best_production = min(valid_configs.keys(),
                          key=lambda m: (all_metrics[m]['forgetting'] * 0.4 + 
                                         all_metrics[m]['avg_loss'] * 0.3 +
                                         all_metrics[m]['time'] * 0.3))
    
    # Best for research (best BWT)
    best_research = max(valid_configs.keys(), key=lambda m: all_metrics[m]['bwt'])
    
    # Best for speed
    best_speed = min(valid_configs.keys(), key=lambda m: all_metrics[m]['time'])
    
    print(f"\n📌 RECOMMENDED CONFIGURATIONS:\n")
    
    print(f"1. BEST OVERALL: {best_overall}")
    if best_overall in configs_used:
        cfg = configs_used[best_overall]
        print(f"   memory_type: '{cfg.memory_type}'")
        print(f"   ewc_lambda: {cfg.ewc_lambda}")
        print(f"   enable_dreaming: {cfg.enable_dreaming}")
        print(f"   dream_interval: {cfg.dream_interval}")
        print(f"   learning_rate: {cfg.learning_rate}")
    
    print(f"\n2. BEST FOR PRODUCTION: {best_production}")
    if best_production in configs_used:
        cfg = configs_used[best_production]
        print(f"   memory_type: '{cfg.memory_type}'")
        print(f"   ewc_lambda: {cfg.ewc_lambda}")
        print(f"   enable_dreaming: {cfg.enable_dreaming}")
        print(f"   learning_rate: {cfg.learning_rate}")
    
    print(f"\n3. BEST FOR RESEARCH (Max BWT): {best_research}")
    if best_research in configs_used:
        cfg = configs_used[best_research]
        print(f"   memory_type: '{cfg.memory_type}'")
        print(f"   ewc_lambda: {cfg.ewc_lambda}")
        print(f"   enable_dreaming: {cfg.enable_dreaming}")
    
    print(f"\n4. FASTEST: {best_speed}")
    if best_speed in configs_used:
        cfg = configs_used[best_speed]
        print(f"   memory_type: '{cfg.memory_type}'")
        print(f"   ewc_lambda: {cfg.ewc_lambda}")
        print(f"   enable_dreaming: {cfg.enable_dreaming}")
    
    # Print optimal config code
    print("\n" + "="*90)
    print("📋 COPY-PASTE OPTIMAL CONFIG:")
    print("="*90)
    
    if best_overall in configs_used:
        cfg = configs_used[best_overall]
        print(f"""
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig

# OPTIMAL CONFIG (based on benchmark results)
optimal_config = AdaptiveFrameworkConfig(
    device='cpu',  # or 'cuda'
    memory_type='{cfg.memory_type}',
    ewc_lambda={cfg.ewc_lambda},
    si_lambda={getattr(cfg, 'si_lambda', 1.0)},
    enable_dreaming={cfg.enable_dreaming},
    dream_interval={cfg.dream_interval},
    dream_batch_size={getattr(cfg, 'dream_batch_size', 32)},
    enable_consciousness=False,
    enable_health_monitor=False,
    use_reptile=False,
    learning_rate={cfg.learning_rate},
    compile_model=False,
)

# Create framework
framework = AdaptiveFramework(your_model, optimal_config, device='cpu')
""")

# ============ MAIN ============
if __name__ == "__main__":
    results, configs_used = run_preset_benchmark()
    
    # Generate plot
    plot_preset_comparison(results, "tests/preset_comparison.png")
    
    # Print results
    all_metrics = print_results_table(results, configs_used)
    
    # Print best configs
    print_best_configs(configs_used, all_metrics)
    
    print("\n" + "="*90)
    print("✨ COMPREHENSIVE PRESET BENCHMARK COMPLETE")
    print("="*90)
    print("Generated files:")
    print("   - tests/preset_comparison.png")
    print("="*90)
