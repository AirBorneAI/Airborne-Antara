import torch
import torch.nn as nn
import logging
from typing import Dict, Any

class KnowledgeGovernor:
    """
    [V15] IRON MIND: The Architectural Knowledge Governor.
    Enforces absolute mathematical boundaries on neural plasticity to achieve zero-forgetting
    across infinite horizons without relying on soft regularization.
    """
    def __init__(self, quota: float = 0.08, device: torch.device = torch.device('cpu')):
        self.quota = quota
        self.device = device
        self.logger = logging.getLogger('KnowledgeGovernor')
        self.task_stats = {} # [V9.5] task_id -> {avg_top, std_top, pct_80, pct_60}
        
    def update_sacred_mask(self, memory_module: Any, task_id: int, backbone_ref: nn.Module):
        """
        Calculates and enforces the V15 Quota.
        1. Snapshots current task's importance (Omega + Fisher).
        2. Rebuilds the global Sacred Mask as the UNION of per-task top-K masks.
        3. Hard-locks FC head classification rows.
        """
        id_to_p = {} 
        id_to_imp = {}

        # 1. Gather all importance metrics across the active network
        with torch.no_grad():
            for m_tracked in memory_module.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad:
                        continue
                    p_id = id(p)
                    id_to_p[p_id] = (name, p)
                    
                    curr = memory_module.omega.get(name, torch.zeros_like(p)).clone()
                    if name in memory_module.fisher_dict:
                        curr = curr + memory_module.fisher_dict[name].to(curr.device)
                    id_to_imp[p_id] = curr.abs()

        if not id_to_imp:
            return

        # 2. Neural Tie-Breaking (V22.2): Ensures we never hit a dead zero-gradient lock
        for pid in id_to_imp:
            id_to_imp[pid] = id_to_imp[pid] + torch.randn_like(id_to_imp[pid]) * 1e-12

        # 3. Snapshot for this task
        if not hasattr(memory_module, 'task_omega_snapshots'):
            memory_module.task_omega_snapshots = {}
            
        memory_module.task_omega_snapshots[task_id] = {
            pid: imp.clone() for pid, imp in id_to_imp.items()
        }

        # 4. Rebuild Global Mask from all snapshots
        cumulative = {}
        for tid, snap in memory_module.task_omega_snapshots.items():
            all_tensors = []
            for pid, imp in snap.items():
                imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                all_tensors.append(imp.view(-1))
            
            flat = torch.cat(all_tensors)
            n = flat.numel()
            k = max(1, min(int(self.quota * n), n))
            
            # Index-based masking ensures EXACT saturation
            top_vals, top_idx = torch.topk(flat, k)
            
            # --- FUTURE ANALYTICS (Current Task Only) ---
            if tid == task_id:
                avg_top = top_vals.mean().item()
                std_top = top_vals.std().item() if k > 1 else 0.0
                
                max_imp = flat.max().item()
                if max_imp > 1e-10:
                    # Percentage of total weights above normalized thresholds
                    pct_80 = (flat >= 0.8 * max_imp).float().mean().item()
                    pct_60 = (flat >= 0.6 * max_imp).float().mean().item()
                else:
                    pct_80 = pct_60 = 0.0
                
                self.task_stats[tid] = {
                    'avg_top': avg_top, 'std_top': std_top,
                    'pct_80': pct_80, 'pct_60': pct_60
                }
                
                self.logger.info(f"  [ANALYTICS] Task {tid} Importance Profile:")
                self.logger.info(f"    - Sacred Quota (Top {self.quota*100}%): Avg={avg_top:.4e}, Std={std_top:.4e}")
                self.logger.info(f"    - High-Value Density: 80%+= {pct_80:.2%}, 60%+= {pct_60:.2%}")
            # ---------------------------------------------
            
            task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            task_mask_flat[top_idx] = True
            
            # Unflatten back to parameters
            curr_pos = 0
            for pid, imp in snap.items():
                p_n = imp.numel()
                m = task_mask_flat[curr_pos : curr_pos + p_n].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                curr_pos += p_n

        # 5. Hard Guarantees (FC Head locking)
        fc = getattr(backbone_ref, 'fc', None)
        if fc is not None:
            fc_w_id = id(fc.weight)
            if fc_w_id not in cumulative:
                cumulative[fc_w_id] = torch.zeros(fc.weight.shape, dtype=torch.bool, device=fc.weight.device)
            
            for tid in memory_module.task_omega_snapshots:
                s, e = tid * 10, min((tid + 1) * 10, fc.weight.shape[0])
                cumulative[fc_w_id][s:e, :] = True
                
            if hasattr(fc, 'bias') and fc.bias is not None:
                fc_b_id = id(fc.bias)
                if fc_b_id not in cumulative:
                    cumulative[fc_b_id] = torch.zeros(fc.bias.shape, dtype=torch.bool, device=fc.bias.device)
                for tid in memory_module.task_omega_snapshots:
                    s, e = tid * 10, min((tid + 1) * 10, fc.bias.shape[0])
                    cumulative[fc_b_id][s:e] = True

        # 5b. MoE Gating Network locking (V9.4 "Best Results" alignment)
        for m_tracked in memory_module.models:
            for name, module in m_tracked.named_modules():
                if "gate" in name.lower() and hasattr(module, 'weight'):
                    g_id = id(module.weight)
                    if g_id not in cumulative:
                        cumulative[g_id] = torch.zeros(module.weight.shape, dtype=torch.bool, device=module.weight.device)
                    
                    for tid in memory_module.task_omega_snapshots:
                        num_experts = module.weight.shape[0]
                        target_expert = tid % num_experts
                        cumulative[g_id][target_expert, :] = True
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        g_b_id = id(module.bias)
                        if g_b_id not in cumulative:
                            cumulative[g_b_id] = torch.zeros(module.bias.shape, dtype=torch.bool, device=module.bias.device)
                        for tid in memory_module.task_omega_snapshots:
                            target_expert = tid % num_experts
                            cumulative[g_b_id][target_expert] = True

        # 6. Apply to Unified Memory
        memory_module.param_id_to_mask = cumulative
        
        # Build global name lookup across all tracked models to sync sacred_mask
        all_names = {}
        for m_tracked in memory_module.models:
            for name, p in m_tracked.named_parameters():
                all_names[id(p)] = name

        for pid, mask in cumulative.items():
            if pid in id_to_p:
                tensor = id_to_p[pid][1]
                # Sync sacred_mask by name
                if pid in all_names:
                    memory_module.sacred_mask[all_names[pid]] = mask.to(tensor.device)
        
        # Calculate Saturation
        total_sacred = sum(m.sum().item() for m in cumulative.values())
        num_total = sum(p.numel() for p in backbone_ref.parameters() if p.requires_grad)
        memory_module.saturation_level = total_sacred / num_total
        
        self.logger.info(f"🛡️ Iron Mind Enforced. Quota: {self.quota*100}%.")
        print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {memory_module.saturation_level:.2%}")
        print(f"  [SENTIENT] Knowledge Anchored. Locked Parameters: {total_sacred:,.0f} / {num_total:,} ({memory_module.saturation_level:.2%})")
        
        if task_id in self.task_stats:
            stats = self.task_stats[task_id]
            print(f"  [SENTIENT] Importance Stats: Avg={stats['avg_top']:.4e}, "
                  f"STD_Div={stats['std_top']:.4e}, "
                  f"High-Value Density (80%+): {stats['pct_80']:.2%}")
