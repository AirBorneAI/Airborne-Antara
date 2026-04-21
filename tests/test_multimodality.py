
import torch
import torch.nn as nn
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST-MULTIMODAL")

def test_modality(name, input_shape, target_shape, model_type='mlp'):
    logger.info(f"--- Testing {name} ---")
    
    # 1. Create Dummy Model
    if model_type == 'cnn':
        # Assume 2D Input [B, C, H, W]
        if len(input_shape) == 3: # 1D Audio [B, C, L]
             base_model = nn.Sequential(
                nn.Conv1d(input_shape[1], 16, 3, padding=1),
                nn.Flatten(),
                nn.Linear(16 * input_shape[2], 10), # Output 10
             )
        else: # 2D Vision/Spec [B, C, H, W]
             output_dim = target_shape[1] if len(target_shape) > 1 else 100
             base_model = nn.Sequential(
                nn.Conv2d(input_shape[1], 16, 3, padding=1),
                nn.Flatten(),
                nn.Linear(16 * input_shape[2] * input_shape[3], output_dim)
             )

    elif model_type == 'rnn':
        class RNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.Linear(input_shape[2], 100) # Simple projection for test
            def forward(self, x): return self.rnn(x).mean(dim=1) # Pool seq
        base_model = RNN()
    else:
        # Simple MLP for vector data
        base_model = nn.Sequential(
            nn.Linear(input_shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, target_shape[1] if len(target_shape)>1 else 10)
        )

    # 2. Wrap with AirborneHRS
    cfg = AdaptiveFrameworkConfig(device='cpu', enable_consciousness=True) # Run on CPU for safety
    agent = AdaptiveFramework(base_model, cfg, device='cpu')
    
    # 3. Create Synthetic Data
    batch_size = 4
    x = torch.randn(batch_size, *input_shape[1:])
    
    if len(target_shape) == 1: # Classification Targets (Indices)
        y = torch.randint(0, 10, (batch_size,))
    else: # Regression/Reconstruction Targets
        y = torch.randn(batch_size, *target_shape[1:])

    # 4. Train Step
    try:
        metrics = agent.train_step(x, target_data=y)
        loss = metrics['loss']
        logger.info(f"✅ {name} PASSED. Loss: {loss:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ {name} FAILED. Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting Universal Modality Tests...\n")
    
    # Vision: [B, C, H, W] -> Class ID [B]
    test_modality("Vision (CIFAR-Style)", (1, 3, 32, 32), (1,), model_type='cnn')
    
    # Audio: [B, 1, T] -> Generic Reg [B, D]
    test_modality("Audio (Waveform 1D)", (1, 1, 16000), (1, 10), model_type='cnn')
    
    # Audio Spectrogram: [B, 1, F, T] (Image-like)
    test_modality("Audio (Spectrogram 2D)", (1, 1, 64, 64), (1, 10), model_type='cnn')
    
    # NLP: [B, Seq, Embed] -> Class ID [B]
    # Note: Using simple linear model for seq for speed test
    test_modality("NLP (Sequence)", (1, 10, 768), (1,), model_type='rnn')
    
    # Tabular: [B, D] -> Regression [B, D]
    test_modality("Tabular (Finance)", (1, 50), (1, 1), model_type='mlp')

    print("\n✨ All tests completed.")
