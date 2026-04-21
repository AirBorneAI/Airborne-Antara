import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os

# Import Airborne-Antara
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig

# ==================== 1. DEFINE YOUR MODEL ====================
# In a real app, this would be your custom neural network
class CognitiveAgent(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== 2. API CONFIGURATION ====================
app = FastAPI(title="Airborne-Antara Sentiment Node", version="9.1")

# Global State
framework: Optional[AdaptiveFramework] = None

class InferenceRequest(BaseModel):
    vector: List[float]
    context_id: Optional[str] = None
    remember: bool = False # [V9.2] Enable One-Shot Learning
    think: bool = False    # [V9.3] Enable System 2 (Metacognitive) Thinking

class InferenceResponse(BaseModel):
    prediction: List[float]
    consciousness: Dict[str, Any]
    expert_usage: Optional[List[float]] = None
    foresight: Optional[List[float]] = None
    mode: Optional[str] = "System 1"

# ==================== 3. LIFECYCLE MANAGEMENT ====================
@app.on_event("startup")
async def load_brain():
    global framework
    print("[SERVER] Waking up the agent...")
    
    # 1. Define Architecture
    base_model = CognitiveAgent(input_dim=16, output_dim=2)
    
    # 2. Configure Brain (Production Mode)
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = 16 # Match input for this demo
    config.num_heads = 4  # Ensure model_dim is divisible by num_heads
    config.use_moe = True
    config.use_graph_memory = True # Long-term memory
    
    # 3. Initialize
    framework = AdaptiveFramework(base_model, config, device='cpu')
    
    # 4. Load Weights (Optional)
    # framework.load_checkpoint("checkpoints/production_v1.pt")
    
    # 5. Warmup
    dummy = torch.randn(1, 16)
    framework.inference_step(dummy)
    print("[SERVER] Agent is Conscious and Ready.")

# ==================== 4. INFERENCE ENDPOINT ====================
@app.post("/infer", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    if not framework:
        raise HTTPException(status_code=503, detail="Brain not loaded")
    
    try:
        # Preprocess
        tensor_in = torch.tensor([req.vector], dtype=torch.float32)
        
        # CHOICE: Fast (System 1) vs Deep (System 2)
        if req.think:
            # Cognitive Inference (Iterative Refinement)
            prediction_tensor, diagnostics = framework.cognitive_inference(
                tensor_in,
                max_steps=3,
                threshold=0.5
            )
        else:
            # Standard Inference (Memory supported if requested)
            prediction_tensor, diagnostics = framework.inference_step(
                tensor_in, 
                return_diagnostics=True,
                remember=req.remember
            )
        
        return InferenceResponse(
            prediction=prediction_tensor.tolist()[0],
            consciousness=diagnostics.get('consciousness', {}),
            expert_usage=diagnostics.get('expert_usage', np.array([])).tolist(),
            foresight=diagnostics.get('foresight_vector', np.array([])).tolist(),
            mode=diagnostics.get('mode', 'System 1')
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "alive", "memory_usage": "nominal"}

if __name__ == "__main__":
    # Run Scalable Worker
    uvicorn.run(app, host="0.0.0.0", port=8000)
