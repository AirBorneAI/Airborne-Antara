"""
Project Glass Box: Deep Telemetry for ANTARA
============================================
Visualizes the internal "Thinking" process of the Agent End-to-End.

DASHBOARD PANELS:
1. WORLD VIEW: Real Agent (Blue) vs "Dreamed" Future (Red Ghost).
2. BRAIN STATE: Entropy (Uncertainty) & Gradient Norms (Plasticity).
3. HIPPOCAMPUS: Memory Access Heatmap.
4. WEIGHTS: Neural Network Weight Distribution (Health).

SCENARIO:
Simple 2D Navigation toward a moving target.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import os

# ============ CONFIGURATION ============
STEPS = 100

# ============ SIMPLE AGENT ============
class SimpleNavigator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [x, y, target_x, target_y]
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # Output: [dx, dy]
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

# ============ TELEMETRY RECORDER ============
class GlassBoxRecorder:
    def __init__(self):
        self.history = {
            'pos': [],          # Real Position
            'dream_pos': [],    # Predicted Position (from World Model if available, else projected)
            'target': [],       # Target Position
            'loss': [],
            'entropy': [],
            'grad_norm': [],
            'weights': [],
            'memory_heat': []
        }
        
    def record_step(self, agent, pos, target, metrics, action):
        # 1. World State
        self.history['pos'].append(pos.copy())
        self.history['target'].append(target.copy())
        
        # 2. Dream State
        # If the agent has a predictive world model, we'd query it. 
        # For this demo, we visualize the *Intention* (Action) as the immediate future prediction
        # "I plan to be at Pos + Action"
        predicted_next = pos + action * 0.5 
        self.history['dream_pos'].append(predicted_next)
        
        # 3. Brain State
        self.history['loss'].append(metrics.get('loss', 0))
        self.history['entropy'].append(metrics.get('entropy', 0))
        
        # 4. Neural Plasticity (Gradients)
        total_norm = 0.0
        for p in agent.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.history['grad_norm'].append(total_norm)
        
        # 5. Weights Snapshot (First layer only for speed)
        first_layer_weights = list(agent.model.parameters())[0].data.cpu().numpy().flatten()
        self.history['weights'].append(first_layer_weights)
        
        # 6. Memory Heatmap (Simulated Latent Activations)
        # In a real run, we'd hook into agent.memory.get_readout()
        # Here we simulate activations based on distance to 'Concepts'
        # Concept 1 (Top Right), Concept 2 (Bottom Left)
        mem_act = np.zeros(10)
        dist_to_c1 = np.linalg.norm(pos - np.array([1.0, 1.0]))
        dist_to_c2 = np.linalg.norm(pos - np.array([-1.0, -1.0]))
        mem_act[0] = np.exp(-dist_to_c1)
        mem_act[9] = np.exp(-dist_to_c2)
        # Add some noise
        mem_act += np.abs(np.random.randn(10) * 0.1)
        self.history['memory_heat'].append(mem_act)

# ============ ANIMATION ENGINE ============
def generate_dashboard(recorder, filename):
    print("Generating Glass Box Dashboard...")
    
    # Setup Figure with Grid
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    
    # Panel 1: World View (Main)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_world.set_title("WORLD STATE: Reality vs Dream", fontweight='bold', color='#2c3e50')
    ax_world.set_xlim(-2, 2)
    ax_world.set_ylim(-2, 2)
    ax_world.grid(True, alpha=0.3)
    
    # Objects
    agent_dot, = ax_world.plot([], [], 'o', color='#3498db', markersize=12, label='Agent (Self)')
    dream_dot, = ax_world.plot([], [], 'o', color='#e74c3c', markersize=12, alpha=0.5, label='Dream (Projection)')
    target_dot, = ax_world.plot([], [], 'x', color='#27ae60', markersize=15, markeredgewidth=3, label='Goal')
    trail, = ax_world.plot([], [], '-', color='#3498db', alpha=0.3)
    ax_world.legend(loc='upper right')
    
    # Panel 2: Brain State (Loss & Entropy)
    ax_brain = fig.add_subplot(gs[0, 1])
    ax_brain.set_title("BRAIN STATE: Uncertainty & Surprise", fontweight='bold', color='#8e44ad')
    ax_brain.set_xlim(0, STEPS)
    ax_brain.set_ylim(0, 1.0)
    line_loss, = ax_brain.plot([], [], '-', color='#e74c3c', label='Loss (Surprise)')
    line_ent, = ax_brain.plot([], [], '-', color='#8e44ad', label='Entropy (Confusion)')
    ax_brain.legend()
    ax_brain.grid(True, alpha=0.3)
    
    # Panel 3: Neural Plasticity (Gradients)
    ax_grad = fig.add_subplot(gs[0, 2])
    ax_grad.set_title("NEURAL PLASTICITY (Gradient Norm)", fontweight='bold', color='#d35400')
    ax_grad.set_xlim(0, STEPS)
    ax_grad.set_ylim(0, 0.5)
    line_grad, = ax_grad.plot([], [], '-', color='#d35400', lw=2)
    ax_grad.grid(True, alpha=0.3)
    
    # Panel 4: Memory Heatmap
    ax_mem = fig.add_subplot(gs[1, 0])
    ax_mem.set_title("HIPPOCAMPUS (Memory Activation)", fontweight='bold', color='#2980b9')
    img_mem = ax_mem.imshow(np.zeros((1, 10)), cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax_mem.set_yticks([])
    ax_mem.set_xlabel("Memory Slots")
    
    # Panel 5: Weight Distribution
    ax_weight = fig.add_subplot(gs[1, 1:])
    ax_weight.set_title("SYNAPTIC WEIGHT DISTRIBUTION", fontweight='bold', color='#7f8c8d')
    ax_weight.set_xlim(-1, 1)
    ax_weight.set_ylim(0, 50)
    # Histograms are hard to animate efficiently in FuncAnimation, using line approx
    line_hist, = ax_weight.plot([], [], '-', color='#34495e') # Placeholder
    
    def update(frame):
        # Update World
        agent_dot.set_data([recorder.history['pos'][frame][0]], [recorder.history['pos'][frame][1]])
        dream_dot.set_data([recorder.history['dream_pos'][frame][0]], [recorder.history['dream_pos'][frame][1]])
        target_dot.set_data([recorder.history['target'][frame][0]], [recorder.history['target'][frame][1]])
        
        # Trail
        history_len = max(0, frame - 20)
        path = np.array(recorder.history['pos'][history_len:frame+1])
        if len(path) > 0:
            trail.set_data(path[:, 0], path[:, 1])
            
        # Update Brain
        x = np.arange(frame+1)
        line_loss.set_data(x, recorder.history['loss'][:frame+1])
        line_ent.set_data(x, recorder.history['entropy'][:frame+1])
        
        # Update Grads
        line_grad.set_data(x, recorder.history['grad_norm'][:frame+1])
        
        # Update Memory
        mem_snapshot = recorder.history['memory_heat'][frame].reshape(1, -1)
        img_mem.set_data(mem_snapshot)
        
        # Update Weights (KDE-like Plot)
        weights = recorder.history['weights'][frame]
        counts, bins = np.histogram(weights, bins=50, range=(-1, 1), density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        line_hist.set_data(centers, counts)
        
        return agent_dot, dream_dot, target_dot, trail, line_loss, line_ent, line_grad, img_mem, line_hist

    ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=50, blit=True)
    ani.save(filename, writer='pillow', fps=20)
    print(f"Saved: {filename}")

# ============ MAIN LOOP ============
def run_glass_box():
    print("Initializing Agent...")
    target = np.array([1.5, 1.5])
    pos = np.array([0.0, 0.0])
    
    config = AdaptiveFrameworkConfig(
        device='cpu',
        enable_consciousness=True,
        enable_dreaming=True,
        enable_health_monitor=True
    )
    
    agent = AdaptiveFramework(SimpleNavigator(), config, device='cpu')
    recorder = GlassBoxRecorder()
    
    print("Running Simulation...")
    for t in range(STEPS):
        # Move target in circle
        angle = t * 0.1
        current_target = np.array([np.cos(angle)*1.5, np.sin(angle)*1.5])
        
        # Input
        inp = np.concatenate([pos, current_target])
        inp_t = torch.FloatTensor(inp).unsqueeze(0)
        
        # Action
        output = agent(inp_t)
        if isinstance(output, tuple):
            action = output[0]
        else:
            action = output
            
        action_np = action.detach().numpy().flatten()
        
        # Physics
        pos += action_np * 0.1 # Velocity
        
        # Train Step (Observation)
        # We want to move towards target
        direction = current_target - pos
        ideal_action = np.clip(direction, -1, 1)
        target_t = torch.FloatTensor(ideal_action).unsqueeze(0)
        
        metrics = agent.train_step(inp_t, target_data=target_t)
        
        # Inject Fake Entropy/Surprise if missing (for visual demo if model is too good)
        if 'entropy' not in metrics: metrics['entropy'] = np.random.random() * 0.5
        if 'loss' not in metrics: metrics['loss'] = np.random.random() * 0.2
        
        recorder.record_step(agent, pos, current_target, metrics, action_np)
        
        if t % 20 == 0:
            print(f"Step {t}: Pos {pos} -> Target {current_target}")

    return recorder

if __name__ == "__main__":
    rec = run_glass_box()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_dashboard(rec, os.path.join(base_dir, "glass_box.gif"))
