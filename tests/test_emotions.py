
import torch
import torch.nn as nn
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST-EMOTION")

def test_emotions():
    logger.info("--- Testing Machine Consciousness (Emotions) ---")
    
    # 1. Setup
    model = nn.Linear(10, 1) # simple
    cfg = AdaptiveFrameworkConfig(device='cpu', enable_consciousness=True)
    agent = AdaptiveFramework(model, cfg, device='cpu')
    
    # 2. Feed Expectation (Low Surprise)
    # We train on a pattern X -> Y
    x = torch.ones(1, 10)
    y = torch.ones(1, 1)
    
    logger.info("Training Pattern A (Establishing Expectation)...")
    initial_surprise = 0.0
    for _ in range(50):
        metrics = agent.train_step(x, target_data=y)
        initial_surprise = metrics.get('surprise', 0.0)
        
    logger.info(f"State after training: Surprise={initial_surprise:.4f}")
    
    # 3. Feed Violation (High Surprise)
    # Give same X, but target is now -100 (Huge error)
    logger.info("Injecting Anomaly (Violation)...")
    y_anomaly = torch.tensor([[-100.0]])
    metrics = agent.train_step(x, target_data=y_anomaly)
    final_surprise = metrics.get('surprise', 0.0)
    
    logger.info(f"State after anomaly: Surprise={final_surprise:.4f}")
    
    if final_surprise > initial_surprise:
        logger.info(f"✅ Emotion Test PASSED. Surprise increased ({initial_surprise:.4f} -> {final_surprise:.4f})")
        return True
    else:
        logger.error(f"❌ Emotion Test FAILED. Agent was not surprised.")
        return False

if __name__ == "__main__":
    test_emotions()
