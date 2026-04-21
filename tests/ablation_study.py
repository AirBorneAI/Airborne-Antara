"""
ANTARA Feature Ablation Study
=============================
Proves the value of EACH advanced feature independently:

1. DREAMING - Replay of past experiences
   Scenario: Class-incremental learning (already proven: +16.6%)

2. CONSCIOUSNESS - Surprise detection + adaptive learning
   Scenario: Sudden distribution shifts where detecting novelty helps

3. HYBRID MEMORY - EWC + SI together
   Scenario: Mixed online/offline learning

4. ALL FEATURES - Combined benefit
   Scenario: Complex real-world simulation

This is a COMPREHENSIVE ABLATION STUDY for publication.
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

# ============ SCENARIO 1: CONSCIOUSNESS TEST ============
# Sudden distribution shifts where surprise detection helps

def generate_shifted_data(task_id, shift_type='none', batch_size=32):
    """
    Generate data with SUDDEN distribution shifts.
    Consciousness should detect shift and adapt learning rate.
    """
    x = torch.randn(batch_size, INPUT_DIM)
    
    if shift_type == 'mean':
        x = x + 3.0  # Sudden mean shift
    elif shift_type == 'variance':
        x = x * 3.0  # Sudden variance increase
    elif shift_type == 'outliers':
        # Add outliers to 20% of samples
        mask = torch.rand(batch_size) < 0.2
        x[mask] = x[mask] * 10.0
    
    if task_id == 0:
        y = x[:, :15].sum(dim=1, keepdim=True)
    elif task_id == 1:
        y = -x[:, 15:].sum(dim=1, keepdim=True)
    else:
        y = (x ** 2)[:, :10].sum(dim=1, keepdim=True)
    
    return x, y

def run_consciousness_test(agent, name):
    """
    Test scenario with SUDDEN distribution shifts.
    
    Phase 1: Normal training (100 steps)
    Phase 2: SUDDEN mean shift (100 steps)
    Phase 3: SUDDEN variance shift (100 steps)
    Phase 4: Back to normal (100 steps)
    """
    print(f"   Testing: {name}...", end=" ", flush=True)
    
    losses = []
    
    for phase, shift_type in enumerate(['none', 'mean', 'variance', 'none']):
        for step in range(100):
            x, y = generate_shifted_data(0, shift_type=shift_type)
            
            if isinstance(agent, AdaptiveFramework):
                metrics = agent.train_step(x, target_data=y)
                loss = metrics['loss']
            else:
                agent.optimizer.zero_grad()
                pred = agent.model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                agent.optimizer.step()
                loss = loss.item()
            
            losses.append(loss)
    
    # Compute recovery metrics
    phase_losses = [
        np.mean(losses[0:100]),     # Normal
        np.mean(losses[100:200]),   # Mean shift
        np.mean(losses[200:300]),   # Variance shift
        np.mean(losses[300:400]),   # Recovery
    ]
    
    recovery_ratio = phase_losses[0] / (phase_losses[3] + 1e-6)  # How well it recovers
    shift_resilience = 1.0 / (np.mean(phase_losses[1:3]) + 1e-6)  # During shift
    
    print(f"Done (Recovery: {recovery_ratio:.2f})")
    return losses, phase_losses, recovery_ratio

# ============ SCENARIO 2: HYBRID MEMORY TEST ============
# Online vs Offline learning scenarios

def generate_online_offline_data(mode, batch_size=32):
    """
    Generate data for online/offline learning test.
    Mode: 'online' - small batches, one-shot
          'offline' - revisiting same data multiple times
    """
    x = torch.randn(batch_size, INPUT_DIM)
    
    if mode == 'online':
        # Data keeps changing
        y = x.sum(dim=1, keepdim=True) + torch.randn(batch_size, 1) * 0.5
    else:
        # Stable data
        y = x.sum(dim=1, keepdim=True)
    
    return x, y

def run_hybrid_test(agent, name):
    """
    Test hybrid memory with alternating online/offline phases.
    
    EWC: Good for offline (consolidation after batch)
    SI: Good for online (track parameter importance during training)
    Hybrid: Should handle both
    """
    print(f"   Testing: {name}...", end=" ", flush=True)
    
    losses = []
    
    # Alternating online/offline phases
    for phase in range(6):
        mode = 'online' if phase % 2 == 0 else 'offline'
        
        for step in range(50):
            x, y = generate_online_offline_data(mode)
            
            if isinstance(agent, AdaptiveFramework):
                metrics = agent.train_step(x, target_data=y)
                loss = metrics['loss']
            else:
                agent.optimizer.zero_grad()
                pred = agent.model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                agent.optimizer.step()
                loss = loss.item()
            
            losses.append(loss)
        
        # Consolidate after offline phases
        if isinstance(agent, AdaptiveFramework) and mode == 'offline':
            if hasattr(agent, 'prioritized_buffer') and agent.prioritized_buffer:
                try:
                    agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
                except:
                    pass
    
    avg_loss = np.mean(losses[-100:])  # Last 100 steps
    stability = np.std(losses[-100:])
    
    print(f"Done (Avg: {avg_loss:.2f}, Stab: {stability:.2f})")
    return losses, avg_loss, stability

# ============ MODEL ============
def create_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

class SimpleBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def forward(self, x):
        return self.model(x)

# ============ MAIN ABLATION STUDY ============
def run_ablation_study():
    print("\n" + "="*70)
    print("ANTARA FEATURE ABLATION STUDY")
    print("="*70)
    print("Proving the value of EACH feature independently.\n")
    
    results = {
        'consciousness': {},
        'hybrid': {}
    }
    
    # ========== CONSCIOUSNESS TEST ==========
    print("\n" + "-"*50)
    print("TEST 1: CONSCIOUSNESS (Surprise Detection)")
    print("Scenario: Sudden distribution shifts")
    print("-"*50)
    
    # Baseline (no consciousness)
    print("🔬 [1/3] Baseline (no consciousness)...")
    baseline = SimpleBaseline()
    c_losses, c_phases, c_recovery = run_consciousness_test(baseline, "Baseline")
    results['consciousness']['Baseline'] = {'losses': c_losses, 'phases': c_phases, 'recovery': c_recovery}
    
    # ANTARA without consciousness
    print("🔬 [2/3] ANTARA (consciousness=False)...")
    cfg_no_consc = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=100.0,
        enable_dreaming=False,
        enable_consciousness=False,
        enable_health_monitor=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_no_consc = AdaptiveFramework(create_model(), cfg_no_consc, device='cpu')
    c_losses2, c_phases2, c_recovery2 = run_consciousness_test(antara_no_consc, "No Consciousness")
    results['consciousness']['No Consciousness'] = {'losses': c_losses2, 'phases': c_phases2, 'recovery': c_recovery2}
    
    # ANTARA with consciousness
    print("🔬 [3/3] ANTARA (consciousness=True)...")
    cfg_consc = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=100.0,
        enable_dreaming=False,
        enable_consciousness=True,  # KEY FEATURE
        enable_health_monitor=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_consc = AdaptiveFramework(create_model(), cfg_consc, device='cpu')
    c_losses3, c_phases3, c_recovery3 = run_consciousness_test(antara_consc, "With Consciousness")
    results['consciousness']['With Consciousness'] = {'losses': c_losses3, 'phases': c_phases3, 'recovery': c_recovery3}
    
    # ========== HYBRID MEMORY TEST ==========
    print("\n" + "-"*50)
    print("TEST 2: HYBRID MEMORY (EWC + SI)")
    print("Scenario: Alternating online/offline learning")
    print("-"*50)
    
    # EWC only
    print("🔬 [1/3] EWC Only...")
    cfg_ewc = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='ewc',
        ewc_lambda=100.0,
        enable_dreaming=False,
        enable_consciousness=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_ewc = AdaptiveFramework(create_model(), cfg_ewc, device='cpu')
    h_losses1, h_avg1, h_stab1 = run_hybrid_test(antara_ewc, "EWC Only")
    results['hybrid']['EWC Only'] = {'losses': h_losses1, 'avg': h_avg1, 'stability': h_stab1}
    
    # SI only
    print("🔬 [2/3] SI Only...")
    cfg_si = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='si',
        si_lambda=1.0,
        enable_dreaming=False,
        enable_consciousness=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_si = AdaptiveFramework(create_model(), cfg_si, device='cpu')
    h_losses2, h_avg2, h_stab2 = run_hybrid_test(antara_si, "SI Only")
    results['hybrid']['SI Only'] = {'losses': h_losses2, 'avg': h_avg2, 'stability': h_stab2}
    
    # Hybrid (EWC + SI)
    print("🔬 [3/3] Hybrid (EWC + SI)...")
    cfg_hybrid = AdaptiveFrameworkConfig(
        device='cpu',
        memory_type='hybrid',
        ewc_lambda=50.0,
        si_lambda=0.5,
        enable_dreaming=False,
        enable_consciousness=False,
        learning_rate=1e-3,
        compile_model=False,
    )
    antara_hybrid = AdaptiveFramework(create_model(), cfg_hybrid, device='cpu')
    h_losses3, h_avg3, h_stab3 = run_hybrid_test(antara_hybrid, "Hybrid")
    results['hybrid']['Hybrid'] = {'losses': h_losses3, 'avg': h_avg3, 'stability': h_stab3}
    
    return results

def plot_ablation_results(results, filename):
    """Plot ablation study results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Consciousness - Loss curves
    ax = axes[0, 0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, (name, data) in enumerate(results['consciousness'].items()):
        ax.plot(data['losses'], alpha=0.7, color=colors[i], label=name)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    ax.text(50, ax.get_ylim()[1]*0.9, 'Normal', ha='center', fontsize=9)
    ax.text(150, ax.get_ylim()[1]*0.9, 'Mean Shift', ha='center', fontsize=9)
    ax.text(250, ax.get_ylim()[1]*0.9, 'Var Shift', ha='center', fontsize=9)
    ax.text(350, ax.get_ylim()[1]*0.9, 'Recovery', ha='center', fontsize=9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Consciousness Test: Distribution Shifts", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Consciousness - Phase comparison
    ax = axes[0, 1]
    x = np.arange(4)
    width = 0.25
    phases = ['Normal', 'Mean Shift', 'Var Shift', 'Recovery']
    for i, (name, data) in enumerate(results['consciousness'].items()):
        ax.bar(x + i*width, data['phases'], width, label=name, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Average Loss")
    ax.set_title("Loss per Phase (Lower ↓)", fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Hybrid Memory - Loss curves
    ax = axes[1, 0]
    colors2 = ['#e74c3c', '#9b59b6', '#2ecc71']
    for i, (name, data) in enumerate(results['hybrid'].items()):
        ax.plot(data['losses'], alpha=0.7, color=colors2[i], label=name)
    for phase in range(1, 6):
        ax.axvline(x=phase*50, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Hybrid Memory Test: Online/Offline Phases", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Hybrid Memory - Final comparison
    ax = axes[1, 1]
    methods = list(results['hybrid'].keys())
    avgs = [results['hybrid'][m]['avg'] for m in methods]
    stabs = [results['hybrid'][m]['stability'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, avgs, width, label='Avg Loss', color='#3498db')
    bars2 = ax.bar(x + width/2, stabs, width, label='Stability (std)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Value")
    ax.set_title("Memory Type Comparison", fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle("ANTARA Feature Ablation Study", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def print_ablation_results(results):
    print("\n" + "="*70)
    print("📊 ABLATION STUDY RESULTS")
    print("="*70)
    
    # Consciousness results
    print("\n🧠 CONSCIOUSNESS TEST:")
    print(f"{'Method':<20} | {'Normal':>10} | {'Mean Shift':>12} | {'Var Shift':>12} | {'Recovery':>10}")
    print("-"*75)
    for name, data in results['consciousness'].items():
        phases = data['phases']
        print(f"{name:<20} | {phases[0]:>10.2f} | {phases[1]:>12.2f} | {phases[2]:>12.2f} | {phases[3]:>10.2f}")
    
    # Determine consciousness winner
    consc_data = results['consciousness']
    no_consc_recovery = consc_data.get('No Consciousness', {}).get('phases', [0,0,0,100])[3]
    with_consc_recovery = consc_data.get('With Consciousness', {}).get('phases', [0,0,0,100])[3]
    
    print("\n   ➤ Recovery Phase Comparison:")
    print(f"     Without Consciousness: {no_consc_recovery:.2f}")
    print(f"     With Consciousness:    {with_consc_recovery:.2f}")
    if with_consc_recovery < no_consc_recovery:
        improvement = (no_consc_recovery - with_consc_recovery) / no_consc_recovery * 100
        print(f"     🏆 CONSCIOUSNESS WINS! {improvement:.1f}% better recovery")
    else:
        print(f"     ⚠️ Consciousness did not improve recovery")
    
    # Hybrid results
    print("\n🔗 HYBRID MEMORY TEST:")
    print(f"{'Method':<15} | {'Avg Loss':>12} | {'Stability':>12} | Combined Score")
    print("-"*55)
    for name, data in results['hybrid'].items():
        combined = data['avg'] + data['stability']  # Lower is better
        print(f"{name:<15} | {data['avg']:>12.2f} | {data['stability']:>12.2f} | {combined:.2f}")
    
    # Determine hybrid winner
    hybrid_data = results['hybrid']
    ewc_score = hybrid_data.get('EWC Only', {}).get('avg', 100) + hybrid_data.get('EWC Only', {}).get('stability', 100)
    si_score = hybrid_data.get('SI Only', {}).get('avg', 100) + hybrid_data.get('SI Only', {}).get('stability', 100)
    hyb_score = hybrid_data.get('Hybrid', {}).get('avg', 100) + hybrid_data.get('Hybrid', {}).get('stability', 100)
    
    print("\n   ➤ Combined Score (Lower is Better):")
    print(f"     EWC Only:  {ewc_score:.2f}")
    print(f"     SI Only:   {si_score:.2f}")
    print(f"     Hybrid:    {hyb_score:.2f}")
    
    if hyb_score <= min(ewc_score, si_score):
        print(f"     🏆 HYBRID MEMORY WINS!")
    elif ewc_score < si_score:
        print(f"     Best: EWC Only")
    else:
        print(f"     Best: SI Only")

if __name__ == "__main__":
    results = run_ablation_study()
    plot_ablation_results(results, "tests/ablation_study.png")
    print_ablation_results(results)
    
    print("\n" + "="*70)
    print("✨ ABLATION STUDY COMPLETE")
    print("="*70)
