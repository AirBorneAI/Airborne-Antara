
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import logging
import sys
import os

# Suppress all logging
logging.disable(logging.CRITICAL)

def generate_data(task_id, batch_size=32):
    x = torch.randn(batch_size, 10)
    if task_id == 'A': y = x.sum(dim=1, keepdim=True)
    elif task_id == 'B': y = -x.sum(dim=1, keepdim=True)
    elif task_id == 'C': y = (x[:, ::2]).sum(dim=1, keepdim=True)
    elif task_id == 'D': y = (x[:, 1::2]).sum(dim=1, keepdim=True)
    return x, y

def evaluate_tasks(agent, tasks=['A','B','C','D']):
    losses = {}
    with torch.no_grad():
        for t in tasks:
            x, y = generate_data(t, batch_size=50)
            output = agent(x)
            pred = output[0] if isinstance(output, tuple) else output
            losses[t] = nn.MSELoss()(pred, y).item()
    return losses

def run_single_config(dream_int, ewc_lam):
    # Print mainly to stdout so we can see it even if stderr is piped
    print(f"Testing Config: Dream={dream_int}, EWC={ewc_lam}... ", end="", flush=True)
    
    base_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    cfg = AdaptiveFrameworkConfig(
        device='cpu', memory_type='hybrid',
        ewc_lambda=float(ewc_lam), dream_interval=int(dream_int),
        enable_consciousness=False 
    )
    agent = AdaptiveFramework(base_model, cfg, device='cpu')
    
    tasks = ['A', 'B', 'C', 'D']
    for t in tasks:
        # 100 steps per task - Faster
        for _ in range(100):
            x, y = generate_data(t)
            agent.train_step(x, target_data=y)
        
        # Consolidate
        if agent.prioritized_buffer:
            agent.memory.consolidate(agent.prioritized_buffer, current_step=agent.step_count, mode='NORMAL')
            
    results = evaluate_tasks(agent)
    avg_loss = sum(results.values()) / 4.0
    plasticity = results['D']
    stability = results['A']
    
    print(f"Done. Avg: {avg_loss:.2f} (Stab: {stability:.2f}, Plast: {plasticity:.2f})")
    
    return {
        'Config': f"Dream{dream_int}_EWC{ewc_lam}",
        'avg_loss': avg_loss,
        'Stability (Task A Error)': stability,
        'Plasticity (Task D Error)': plasticity
    }

if __name__ == "__main__":
    print("🚀 Starting Targeted Sweep...")
    
    # 3 Distinct Profiles
    configs = [
        (10, 5000), # Stability Focused (Frequent Dream, High EWC)
        (50, 500),  # Plasticity Focused (Rare Dream, Low EWC)
        (25, 2000)  # Balanced Candidate
    ]
    
    data = []
    for d, e in configs:
        res = run_single_config(d, e)
        data.append(res)
            
    df = pd.DataFrame(data)
    print("\n🏆 Results Summary:")
    print(df.to_string(index=False))
    
    # Textual Graph
    df.set_index('Config')[['Stability (Task A Error)', 'Plasticity (Task D Error)']].plot(kind='bar', figsize=(10, 6))
    plt.title("Trade-off Analysis: Stability vs Plasticity")
    plt.ylabel("MSE Loss (Lower is Better)")
    plt.tight_layout()
    plt.savefig("tests/sweep_results.png")
    print("\n✅ Comparison Graph saved to tests/sweep_results.png")
    
    best = df.loc[df['avg_loss'].idxmin()]
    print(f"\n🌟 GOLDEN CONFIGURATION: {best['Config']} (Avg Loss {best['avg_loss']:.2f})")
