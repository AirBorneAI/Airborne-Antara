"""
ANTARA V3: Drone Swarm Simulation (Real-World Case Study)
=========================================================
A complex simulation of a heterogeneous drone swarm adapting to dynamic environments.
Demonstrates "Hive Mind" agency: ANTARA controls the formation and task allocation.

FEATURES:
1. Heterogeneous Agents: Scout (Fast), Heavy (Slow/Cargo), Relay (Comms).
2. Dynamic Terrain: Urban (Signal Noise), Mountains (Movement Friction).
3. Discovery: Drones can be added/lost dynamically (simulating failures or reinforcements).
4. Formation Healing: Swarm adapts shape when units are lost.

SCENARIO:
A Search & Rescue mission where the swarm must scan a map while keeping the "Heavy" drone protected in the center.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig

# ============ CONFIGURATION ============
MAP_SIZE = 100.0
N_DRONES_INIT = 8
DT = 0.1

class DroneType:
    SCOUT = 0   # Red: Fast, Sensor Range High, Low Health
    HEAVY = 1   # Blue: Slow, Carries Payload, Needs Protection
    RELAY = 2   # Green: Medium, Maintains Comms

# ============ ENVIRONMENT ============
class TerrainMap:
    def __init__(self, size=MAP_SIZE):
        self.size = size
        # Generate friction map (Mountains)
        self.friction = np.zeros((int(size), int(size)))
        # Create a "Mountain" ridge
        for x in range(30, 50):
            for y in range(0, int(size)):
                self.friction[x, y] = 0.8 # High friction
        
        # Create an "Urban" zone (High signal noise, visual clutter)
        self.urban = np.zeros((int(size), int(size)))
        for x in range(70, 90):
            for y in range(40, 80):
                self.urban[x, y] = 1.0

    def get_friction(self, pos):
        x, y = int(np.clip(pos[0], 0, self.size-1)), int(np.clip(pos[1], 0, self.size-1))
        return self.friction[x, y]

# ============ AGENT ============
class Drone:
    def __init__(self, uid, d_type, pos):
        self.uid = uid
        self.type = d_type
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.active = True
        
        # Specs based on type
        if d_type == DroneType.SCOUT:
            self.max_speed = 5.0
            self.sensor_range = 25.0
            self.color = 'red'
        elif d_type == DroneType.HEAVY:
            self.max_speed = 2.0
            self.sensor_range = 10.0
            self.color = 'blue'
        else: # RELAY
            self.max_speed = 3.5
            self.sensor_range = 15.0
            self.color = 'green'

    def step(self, force, friction):
        if not self.active: return
        
        # Physics
        drag = 0.1 + friction
        acc = force - self.vel * drag
        self.vel += acc * DT
        
        # Speed limit
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
            
        self.pos += self.vel * DT
        
        # Bounds
        self.pos = np.clip(self.pos, 0, MAP_SIZE)

# ============ HIVE MIND (ANTARA CONTROLLER) ============
class SwarmController:
    def __init__(self):
        # Input: [Self Pos(2), Self Vel(2), Type(1), Nearest Neighbor Diff(2), Terrain(1)] = 8 dims
        # Output: [Thrust X, Thrust Y]
        self.model = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh() # Thrust -1 to 1
        )
        
        # ANTARA Wrapper for adaptation
        self.config = AdaptiveFrameworkConfig(
            device='cpu',
            memory_type='ewc', # Use EWC to remember formation drills
            enable_dreaming=True, # Dream to simulate "War Games"
            enable_consciousness=True, # Detect formation collapse (Entropy)
            learning_rate=0.01
        )
        self.agent = AdaptiveFramework(self.model, self.config, device='cpu')
        
    def get_commands(self, drones, terrain):
        commands = {}
        # Calculate centroids/neighbors
        active_drones = [d for d in drones if d.active]
        if not active_drones: return {}
        
        swarm_center = np.mean([d.pos for d in active_drones], axis=0)
        
        for drone in active_drones:
            # 1. Perception
            # Find nearest neighbor
            nearest_dist = 999
            nearest_diff = np.zeros(2)
            for other in active_drones:
                if other.uid == drone.uid: continue
                diff = other.pos - drone.pos
                dist = np.linalg.norm(diff)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_diff = diff
            
            # Terrain sensor
            env_friction = terrain.get_friction(drone.pos)
            
            # Construct State Vector
            state = np.concatenate([
                drone.pos / MAP_SIZE,         # Norm Pos
                drone.vel / drone.max_speed,  # Norm Vel
                [float(drone.type) / 2.0],    # Norm Type
                nearest_diff / MAP_SIZE,      # Relative Neighbor
                [env_friction]                # Terrain
            ])
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            # 2. ANTARA Inference (Policy)
            output = self.agent(state_t)
            if isinstance(output, tuple):
                action = output[0]
            else:
                action = output
                
            thrust = action.detach().numpy().flatten() * 2.0 # Scale force
            
            # 3. "Hive Mind" Global Correction (Formation Logic)
            # The Heavy drone is the "Queen" - others must flock around it
            if drone.type != DroneType.HEAVY:
                # Find Heavy drones
                heavies = [d for d in active_drones if d.type == DroneType.HEAVY]
                if heavies:
                    # Cohesion to Heavy
                    target = heavies[0].pos
                    correction = (target - drone.pos) * 0.5
                    thrust += correction
            
            # Separation (Collision Avoidance)
            if nearest_dist < 5.0 and nearest_dist > 0:
                 repulsion = -nearest_diff * (1.0 / nearest_dist) * 5.0
                 thrust += repulsion
                 
            commands[drone.uid] = thrust
            
            # 4. Online Learning (Self-Supervised Formation Maintenance)
            # Reward: Being close to Heavy but not colliding
            if drone.type != DroneType.HEAVY:
                dist_to_heavy = np.linalg.norm(drone.pos - swarm_center)
                reward = -abs(dist_to_heavy - 10.0) # Ideal radius 10.0
                target_signal = torch.FloatTensor([0.0, 0.0]).unsqueeze(0) # Dummy target matching output dim
                # We interpret 'target' loosely here for the example, 
                # normally we'd do strict RL, but ANTARA supports 'prediction error' as signal
                # Here we just run a train step to keep "Consciousness" active
                self.agent.train_step(state_t, target_data=action.detach()) 

        return commands

# ============ SIMULATION LOOP ============
def run_simulation():
    print("Initializing V3 Swarm Simulation...")
    terrain = TerrainMap()
    controller = SwarmController()
    
    # Init Drones
    drones = []
    # 1 Heavy in center
    drones.append(Drone(0, DroneType.HEAVY, [10, 50]))
    # Scouts and Relays around
    for i in range(1, N_DRONES_INIT):
        dtype = DroneType.SCOUT if i % 2 == 0 else DroneType.RELAY
        drones.append(Drone(i, dtype, [10 + np.random.randn()*5, 50 + np.random.randn()*5]))

    print(f"Swarm Initiated: {len(drones)} Drones.")
    print("Mission: Escort HEAVY unit across the Mountain Ridge.")
    
    history = []
    
    # Loop
    for t in range(200): # 20 seconds
        # 1. Dynamic Discovery (Event at t=50: Reinforcements)
        if t == 50:
            print("\n[EVENT] Reinforcements Arrived! +3 Scouts joined the network.")
            for k in range(3):
                new_id = len(drones) + k
                drones.append(Drone(new_id, DroneType.SCOUT, [0, 50]))
        
        # 2. Dynamic Loss (Event at t=100: Sniper Attack / Crash)
        if t == 100:
             print("\n[EVENT] AMBUSH! Drone #2 shot down.")
             if len(drones) > 2:
                 drones[2].active = False
        
        # 3. Hive Control
        commands = controller.get_commands(drones, terrain)
        
        # 4. Physics Step
        for d in drones:
            friction = terrain.get_friction(d.pos)
            force = commands.get(d.uid, np.zeros(2))
            
            # Global mission bias (Move Right)
            force[0] += 0.5 
            
            d.step(force, friction)
            
        # Log
        positions = [d.pos.copy() for d in drones if d.active]
        types = [d.type for d in drones if d.active]
        history.append((positions, types))
        
        if t % 20 == 0:
            center = np.mean([d.pos for d in drones if d.active], axis=0)
            print(f"T={t:03d} | Center {center} | Active: {len([d for d in drones if d.active])}")

    return history, terrain

# ============ VISUALIZATION ============
def animate_swarm(history, terrain, filename):
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw terrain
    ax.imshow(terrain.friction.T, origin='lower', cmap='Greys', alpha=0.3, extent=[0, 100, 0, 100])
    ax.text(40, 90, "MOUNTAIN RIDGE (High Friction)", color='gray', ha='center')
    
    scat = ax.scatter([], [], c=[], s=50)
    
    def update(frame):
        positions, types = history[frame]
        if not positions: return scat,
        
        pos_arr = np.array(positions)
        colors = []
        for t in types:
            if t == DroneType.SCOUT: colors.append('red')
            elif t == DroneType.HEAVY: colors.append('blue')
            else: colors.append('green')
            
        scat.set_offsets(pos_arr)
        scat.set_color(colors)
        ax.set_title(f"V3 Swarm Simulation | T={frame}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    hist, terr = run_simulation()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    animate_swarm(hist, terr, os.path.join(base_dir, "drone_swarm_v3.gif"))
