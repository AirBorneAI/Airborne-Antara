import torch
import torch.nn as nn
from airborne_antara.moe import HierarchicalMoE, SparseMoE, AdaptiveExpertBlock
from airborne_antara.core import AdaptiveFramework, AdaptiveFrameworkConfig

def test_device_mismatch():#test
    print("--- Testing Device Mismatch on Mask Indexing ---")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping device mismatch test.")
        return
    
    device = torch.device("cuda")
    target_data = torch.randint(0, 10, (10,))
    mask = torch.ones(10, dtype=torch.bool, device=device)
    
    try:
        subset = target_data[mask]
        print("Indexing succeeded (unexpected!). Shape:", subset.shape)
    except Exception as e:
        print("Captured expected exception during CPU indexing with CUDA mask:")
        print(f"  {type(e).__name__}: {e}")

def test_hierarchical_moe_shapes():
    print("\n--- Testing HierarchicalMoE returned domain_indices shapes ---")
    base_model = nn.Sequential(nn.Linear(10, 5))
    moe = HierarchicalMoE(base_model, input_dim=10, num_domains=2, experts_per_domain=2, top_k=2)
    
    x = torch.randn(4, 10)
    moe.train()
    target_data = torch.tensor([0, 1, 10, 11])
    out_train, indices_train = moe(x, task_id=0, target_data=target_data)
    print("Training (with task_id & target_data) returned indices shape:", indices_train.shape)
    
    moe.eval()
    out_eval, indices_eval = moe(x, task_id=None)
    print("Evaluation (no task_id) returned indices shape:", indices_eval.shape)

def test_adaptive_expert_block():
    print("\n--- Testing AdaptiveExpertBlock reshape/view logic ---")
    backbone = nn.Sequential(nn.Linear(10, 5))
    block = AdaptiveExpertBlock(backbone, input_dim=10, adapter_dim=4)
    
    x_flat = torch.randn(4, 10)
    try:
        out = block(x_flat)
        print("AdaptiveExpertBlock forward succeeded with flat input. Shape:", out.shape)
    except Exception as e:
        print("Captured exception with flat input:")
        print(f"  {type(e).__name__}: {e}")
        
    block_conv = AdaptiveExpertBlock(nn.Sequential(nn.Flatten(), nn.Linear(12, 5)), input_dim=12, adapter_dim=4)
    x_img = torch.randn(4, 3, 2, 2)
    try:
        out = block_conv(x_img)
        print("AdaptiveExpertBlock forward succeeded with img input. Shape:", out.shape)
    except Exception as e:
        print("Captured exception with img input:")
        print(f"  {type(e).__name__}: {e}")

def test_adaptive_framework_non_moe():
    print("\n--- Testing AdaptiveFramework with use_moe=False ---")
    base_model = nn.Sequential(nn.Linear(10, 5))
    cfg = AdaptiveFrameworkConfig(
        device='cpu',
        use_moe=False,
        enable_world_model=False,
        enable_consciousness=False
    )
    framework = AdaptiveFramework(base_model, cfg, device='cpu')
    x = torch.randn(4, 10)
    target_data = torch.randn(4, 5)
    
    try:
        metrics = framework.train_step(x, target_data=target_data)
        print("train_step succeeded! Loss:", metrics.get('loss'))
    except Exception as e:
        print("Captured exception during train_step:")
        print(f"  {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_device_mismatch()
    test_hierarchical_moe_shapes()
    test_adaptive_expert_block()
    test_adaptive_framework_non_moe()
