"""
MIRRORMIMD PRESETS QUICK REFERENCE
===================================

Copy-paste examples for common scenarios.
"""

# ============ COPY-PASTE EXAMPLES ============

# 1. PRODUCTION SETUP
# ===================
# For: Live systems, accuracy-critical, multi-domain
from airborne_antara import AdaptiveFramework, PRESETS
import torch

model = YourModel()
config = PRESETS.production()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(10):
    for batch_idx, (x, y) in enumerate(train_loader):
        output = framework(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# 2. FAST REAL-TIME
# =================
# For: Robotics, streaming, real-time RL
config = PRESETS.fast()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Online learning
for step, (state, action, reward, next_state) in enumerate(experience_stream):
    with torch.no_grad():
        value = framework(state)
    loss = compute_td_loss(value, reward, next_state)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# 3. MAXIMUM ACCURACY
# ===================
# For: Medical, finance, high-consequence decisions
config = PRESETS.accuracy_focus()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Conservative training
for epoch in range(100):  # Long training
    for batch in train_loader:
        output = framework(batch['x'])
        loss = criterion(output, batch['y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()


# 4. MOBILE/EDGE DEVICE
# =====================
# For: Mobile apps, IoT, embedded systems
config = PRESETS.memory_efficient()
framework = AdaptiveFramework(model, config=config)
framework = framework.to('cpu')  # Ensure CPU
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Memory-friendly training
for batch in train_loader:
    output = framework(batch['x'])
    loss = criterion(output, batch['y'])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# 5. CREATIVE/GENERATIVE
# ======================
# For: Text generation, image generation, diversity
config = PRESETS.creativity_boost()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training with diversity
for epoch in range(epochs):
    for batch in train_loader:
        output = framework(batch['x'])
        loss = criterion(output, batch['y'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# 6. EXPLORATION/CURIOSITY
# =========================
# For: Curiosity-driven RL, multi-task learning
config = PRESETS.exploration()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Learning with intrinsic motivation
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = policy(framework(state))
        next_state, reward, done, info = env.step(action)
        loss = compute_rl_loss(framework(state), action, reward, next_state)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        state = next_state


# 7. MULTI-TASK LEARNING
# =======================
# For: Multiple tasks, shared representation
config = PRESETS.balanced()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Multi-task training
for epoch in range(epochs):
    for batch in train_loader:
        # Shared representation
        features = framework(batch['x'])
        
        # Task-specific heads
        task1_out = head1(features)
        task2_out = head2(features)
        
        # Combined loss
        loss = loss_fn1(task1_out, batch['y1']) + loss_fn2(task2_out, batch['y2'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


# 8. RESEARCH/EXPERIMENTATION
# ============================
# For: Ablation studies, research papers
config = PRESETS.research()
framework = AdaptiveFramework(model, config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training with full instrumentation
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        output = framework(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Full metrics logging
        if batch_idx % 10 == 0:
            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
            # Framework also logs internal metrics


# ============ CUSTOMIZATION RECIPES ============

# RECIPE 1: Fast production (prioritize speed)
config = PRESETS.production().customize(
    model_dim=256,              # Smaller model
    learning_rate=1e-3,         # Faster learning
    feedback_buffer_size=5000,  # Smaller buffer
)

# RECIPE 2: Accurate production (prioritize accuracy)
config = PRESETS.production().customize(
    learning_rate=1e-4,         # More conservative
    feedback_buffer_size=50000, # Larger buffer
    use_prioritized_replay=True,
    gradient_clip_norm=0.5,     # Tight clipping
)

# RECIPE 3: Mobile production (memory-constrained)
config = PRESETS.memory_efficient().customize(
    model_dim=128,
    learning_rate=1e-3,
)

# RECIPE 4: Real-time robotics (sub-10ms latency)
config = PRESETS.real_time().customize(
    model_dim=96,
    learning_rate=5e-3,
    enable_consciousness=False,
)

# RECIPE 5: Stable continual learning (avoid forgetting)
config = PRESETS.stable().customize(
    consolidation_min_interval=100,
    consolidation_max_interval=500,
    gradient_clip_norm=0.5,
)

# RECIPE 6: Creative + stable (diversity + robustness)
config = PRESETS.creativity_boost().merge(PRESETS.stable()).customize(
    learning_rate=1e-3,
)


# ============ PRESET COMPARISON ============

from airborne_antara import compare_presets, list_presets

# List all presets
presets = list_presets()
for name, desc in presets.items():
    print(f"{name:20} - {desc}")

# Compare specific presets
comparison = compare_presets('production', 'fast', 'accuracy_focus')
print(comparison)

# Output:
# {
#   'production': {
#       'model_dim': 512,
#       'learning_rate': 0.0005,
#       'buffer_size': 20000,
#       'memory_type': 'hybrid',
#       'consciousness': True
#   },
#   'fast': {
#       'model_dim': 128,
#       'learning_rate': 0.005,
#       'buffer_size': 2000,
#       'memory_type': 'si',
#       'consciousness': False
#   },
#   'accuracy_focus': {
#       'model_dim': 512,
#       'learning_rate': 0.0001,
#       'buffer_size': 50000,
#       'memory_type': 'ewc',
#       'consciousness': True
#   }
# }


# ============ LOADING BY NAME ============

from airborne_antara import load_preset

# Load by string
config = load_preset('production')
config = load_preset('fast')
config = load_preset('accuracy_focus')

# Invalid preset raises clear error
try:
    config = load_preset('invalid')
except ValueError as e:
    print(e)  # Unknown preset: invalid. Available: [...]


# ============ ADVANCED: MERGING ============

# Merge two presets (second overwrites first)
config = PRESETS.production().merge(PRESETS.creativity_boost())
# Result: Production stability + Creativity parameters

# Three-way merge
config = (PRESETS.production()
          .merge(PRESETS.creativity_boost())
          .merge(PRESETS.fast())
          .customize(learning_rate=1e-3))

# This combines:
# - Production's model size and memory type
# - Creativity's dropout and exploration
# - Fast's learning rate and consolidation
# - Plus custom learning rate override


# ============ DEBUGGING ============

# Print preset configuration
config = PRESETS.production()
print(config)
# Output: Preset(lr=5e-04, model_dim=512, buffer_size=20000, memory=hybrid, consciousness=True)

# Convert to dict
config_dict = config.to_dict()
print(f"Model dimensions: {config_dict['model_dim']}")
print(f"Learning rate: {config_dict['learning_rate']}")
print(f"Buffer size: {config_dict['feedback_buffer_size']}")

# Check specific parameters
if config.enable_consciousness:
    print("Consciousness layer is ENABLED")
else:
    print("Consciousness layer is DISABLED")


# ============ MONITORING ============

# Track framework metrics during training
from airborne_antara import AdaptiveFramework, PRESETS

config = PRESETS.production()
framework = AdaptiveFramework(model, config=config)

# Access framework internals (if needed for monitoring)
# metrics = framework.get_metrics()
# print(f"Learning phase: {metrics['phase']}")
# print(f"Adaptation rate: {metrics['adaptation_rate']}")
# print(f"Consolidations: {metrics['consolidation_count']}")


# ============ PRODUCTION DEPLOYMENT ============

# Standard production pipeline
def create_production_system():
    model = YourModel()
    config = PRESETS.production()
    framework = AdaptiveFramework(model, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Optional: Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    framework = framework.to(device)
    
    return framework, optimizer, device

def train_production():
    framework, optimizer, device = create_production_system()
    
    for epoch in range(100):
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            # Forward pass
            output = framework(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(framework.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Framework handles everything else automatically!


# ============ TESTING PRESETS ============

# Quick test to verify preset works
def test_preset(preset_name: str):
    from airborne_antara import load_preset
    import torch
    
    config = load_preset(preset_name)
    
    # Dummy model
    model = torch.nn.Linear(10, 5)
    framework = AdaptiveFramework(model, config=config)
    
    # Dummy input
    x = torch.randn(2, 10)
    output = framework(x)
    
    print(f"✓ {preset_name} works! Output shape: {output.shape}")

# Test all presets
for preset_name in ['production', 'fast', 'balanced', 'accuracy_focus', 'memory_efficient']:
    test_preset(preset_name)
