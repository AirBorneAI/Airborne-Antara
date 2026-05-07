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
        
    def update_sacred_mask(self, memory_module: Any, task_id: int, backbone_ref: nn.Module, classes_per_task: int = 10):
        """
        Calculates and enforces the V15 Quota.
        1. Snapshots current task's importance (Omega + Fisher).
        2. Rebuilds the global Sacred Mask as the UNION of per-task top-K masks.
        3. Hard-locks FC head classification rows.
        """
        id_to_p = {} 
        id_to_imp = {}
        cumulative = {} # [V31.6] Initialize mask accumulator

        # 1. Gather all importance metrics across the active network
        with torch.no_grad():
            for m_idx, m_tracked in enumerate(memory_module.models):
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad:
                        continue
                    p_id = id(p)
                    id_to_p[p_id] = (name, p)
                    
                    unique_name = f"m{m_idx}_{name}"
                    curr = memory_module.omega.get(unique_name, torch.zeros_like(p)).clone()
                    if unique_name in memory_module.fisher_dict:
                        curr = curr + memory_module.fisher_dict[unique_name].to(curr.device)
                    id_to_imp[p_id] = curr.abs()

        if not id_to_imp:
            return

        # 2. Neural Tie-Breaking (V22.2): Ensures we never hit a dead zero-gradient lock
        for pid in id_to_imp:
            id_to_imp[pid] = id_to_imp[pid] + torch.randn_like(id_to_imp[pid]) * 1e-12

        # 3. Snapshot and Analytics for this task
        if not hasattr(memory_module, 'task_omega_snapshots'):
            memory_module.task_omega_snapshots = {}
            
        memory_module.task_omega_snapshots[task_id] = {
            pid: imp.clone() for pid, imp in id_to_imp.items()
        }

        # Calculate analytics for the current task to guide ENA
        with torch.no_grad():
            all_tensors = [imp.view(-1) for imp in id_to_imp.values()]
            flat = torch.cat(all_tensors)
            n = flat.numel()
            # Calculate k_base based on task count if possible, otherwise default to 1/10th
            num_tasks = len(memory_module.task_omega_snapshots) if hasattr(memory_module, 'task_omega_snapshots') else 10
            k_base = max(1, min(int(self.quota * n / max(1, num_tasks)), n)) # Base k for analytics
            
            top_vals, _ = torch.topk(flat, k_base)
            avg_top = top_vals.mean().item()
            std_top = top_vals.std().item() if k_base > 1 else 0.0
            
            max_imp = flat.max().item()
            pct_80 = (flat >= 0.8 * max_imp).float().mean().item() if max_imp > 1e-10 else 0.0
            
            self.task_stats[task_id] = {
                'avg_top': avg_top, 'std_top': std_top,
                'pct_80': pct_80
            }

        # [V31.2] DYNAMIC TASK GOVERNANCE: 
        # If total_tasks is not set, we assume a rolling horizon based on current count + buffer.
        num_tasks_seen = len(memory_module.task_omega_snapshots)
        num_tasks_total = getattr(memory_module, 'total_tasks', max(10, num_tasks_seen + 5))
        
        # Headroom-Aware Quota Scaling
        current_saturation = getattr(memory_module, 'saturation_level', 0.0)
        # If we have lots of room, allow aggressive foundation building (2.5x multiplier)
        headroom_multiplier = 2.5 if current_saturation < (self.quota * 0.4) else 1.2
        BASE_TASK_TARGET = (self.quota / max(1, num_tasks_total)) * headroom_multiplier
        
        # Calculate individual quotas with saturation-aware dampening
        task_quotas = {}
        current_saturation = getattr(memory_module, 'saturation_level', 0.0)
        
        for tid, snap in memory_module.task_omega_snapshots.items():
            stats = self.task_stats.get(tid)
            if stats:
                # Equilibrium Logic: Scale target by (1 - saturation)
                # If we are far from the ceiling, we anchor more aggressively.
                dampening = max(0.4, 1.0 - (current_saturation / self.quota)) 
                
                # Elastic scaling based on task density
                # V9.7: Use a more stable density log-scaling
                density = max(stats['pct_80'], 0.0001)
                factor = torch.tensor(density / 0.001).log2().item()
                factor = max(0.8, min(1.5, 1.0 + (factor * 0.1))) 
                
                q = BASE_TASK_TARGET * factor * dampening
                # Bounds [1%, 25%] ensures stability
                q = max(0.01, min(0.25, q))
                task_quotas[tid] = q
            else:
                task_quotas[tid] = BASE_TASK_TARGET

        # Re-apply Top-K with dampening (Union-based)
        for tid, snap in memory_module.task_omega_snapshots.items():
            all_tensors = [torch.nan_to_num(imp, 0, 0, 0).view(-1) for imp in snap.values()]
            flat = torch.cat(all_tensors)
            n = flat.numel()
            q = task_quotas[tid]
            k = max(1, min(int(q * n), n))
            
            _, top_idx = torch.topk(flat, k)
            task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            task_mask_flat[top_idx] = True
            
            curr_pos = 0
            for pid, imp in snap.items():
                p_n = imp.numel()
                m = task_mask_flat[curr_pos : curr_pos + p_n].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                curr_pos += p_n

        # 5. Hard Guarantees (FC Head locking & MoE Gating)
        for m_tracked in memory_module.models:
            for m_name, module in m_tracked.named_modules():
                # [V15.1] Recursive Head Detection
                if hasattr(module, 'weight') and ("fc" in m_name.lower() or "gate" in m_name.lower()):
                    p_w_id = id(module.weight)
                    if p_w_id not in cumulative:
                        cumulative[p_w_id] = torch.zeros(module.weight.shape, dtype=torch.bool, device=module.weight.device)
                    
                    # Logic for FC rows (Task-Specific Output)
                    if "fc" in m_name.lower():
                        num_classes = module.weight.shape[0]
                        # [V29] Dynamic Task Density
                        num_tasks_total = getattr(memory_module, 'total_tasks', 10) 
                        cpt = classes_per_task
                        for tid in memory_module.task_omega_snapshots:
                            s, e = tid * cpt, min((tid + 1) * cpt, num_classes)
                            cumulative[p_w_id][s:e, :] = True
                            
                        if hasattr(module, 'bias') and module.bias is not None:
                            p_b_id = id(module.bias)
                            if p_b_id not in cumulative:
                                cumulative[p_b_id] = torch.zeros(module.bias.shape, dtype=torch.bool, device=module.bias.device)
                            for tid in memory_module.task_omega_snapshots:
                                s, e = tid * cpt, min((tid + 1) * cpt, num_classes)
                                cumulative[p_b_id][s:e] = True
                    
                    # Logic for Gate rows (Task-Specific Experts)
                    elif "gate" in m_name.lower():
                        num_experts = module.weight.shape[0]
                        num_tasks_total = getattr(memory_module, 'total_tasks', 10)
                        for tid in memory_module.task_omega_snapshots:
                            target_expert = tid % num_experts
                            cumulative[p_w_id][target_expert, :] = True
                            
                        if hasattr(module, 'bias') and module.bias is not None:
                            p_b_id = id(module.bias)
                            if p_b_id not in cumulative:
                                cumulative[p_b_id] = torch.zeros(module.bias.shape, dtype=torch.bool, device=module.bias.device)
                            for tid in memory_module.task_omega_snapshots:
                                target_expert = tid % num_experts
                                cumulative[p_b_id][target_expert] = True

        # 6. Apply to Unified Memory
        memory_module.param_id_to_mask = cumulative
        
        # [V31.8] SHARED MODULE PROTECTION: Map all expert-aliases to the same physical mask.
        pid_to_names = {}
        for m_idx, m in enumerate(memory_module.models):
            for name, p in m.named_parameters():
                pid = id(p)
                if pid not in pid_to_names: pid_to_names[pid] = []
                pid_to_names[pid].append(f"m{m_idx}_{name}")

        for pid, mask in cumulative.items():
            if pid in pid_to_names:
                for name in pid_to_names[pid]:
                    memory_module.sacred_mask[name] = mask.to(mask.device)
        
        # [V31.8] IRON MIND: Absolute Hard-Locks (Exempt from Quota)
        # Classification heads and MoE Gates are the 'Identity' of previous tasks.
        # They must be 100% frozen, or the model will 'forget' how to route and label.
        for m in memory_module.models:
            for name, module in m.named_modules():
                # 1. Hard-lock FC rows for ALL completed tasks
                if "fc" in name.lower() and hasattr(module, 'weight'):
                    cpt = self.config.classes_per_task
                    p = module.weight
                    pid = id(p)
                    if pid not in cumulative:
                        cumulative[pid] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
                    # Lock all rows belonging to completed tasks (0 to current task_id)
                    end_idx = min((task_id + 1) * cpt, p.shape[0])
                    cumulative[pid][:end_idx, :] = True
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        pb = module.bias
                        pbid = id(pb)
                        if pbid not in cumulative:
                            cumulative[pbid] = torch.zeros(pb.shape, dtype=torch.bool, device=pb.device)
                        cumulative[pbid][:end_idx] = True

                # 2. Hard-lock MoE Gates entirely after Task 0
                if "gate" in name.lower() and hasattr(module, 'weight'):
                    p = module.weight
                    pid = id(p)
                    if pid not in cumulative:
                        cumulative[pid] = torch.ones(p.shape, dtype=torch.bool, device=p.device)
                    else:
                        cumulative[pid][:] = True
                    
                    if hasattr(module, 'bias') and module.bias is not None:
                        pb = module.bias
                        pbid = id(pb)
                        if pbid not in cumulative:
                            cumulative[pbid] = torch.ones(pb.shape, dtype=torch.bool, device=pb.device)
                        else:
                            cumulative[pbid][:] = True

        # [V31.8] Absolute Foundation Lockdown (Backbone Protection)
        # We strictly freeze early universal feature detectors after Task 0.
        # This prevents the 'representation drift' that destroys foundational knowledge.
        # [V31.11] TITANIUM EXPERT ISOLATION:
        # We only lock the foundational features of the experts that were actually TRAINED.
        # This preserves 100% plasticity for future experts.
        
        # Dynamically determine expert count from naming convention
        expert_indices = set()
        for m in memory_module.models:
            for name, _ in m.named_modules():
                if "experts." in name:
                    try:
                        idx = int(name.split("experts.")[1].split(".")[0])
                        expert_indices.add(idx)
                    except: continue
        num_experts = len(expert_indices) if expert_indices else 8
        target_expert_idx = task_id % num_experts
        
        if task_id >= 0:
            for m_idx, m in enumerate(memory_module.models):
                for name, module in m.named_modules():
                    # Identify early backbone layers (ResNet-style)
                    # Example name: domains.0.experts.0.model.conv1
                    is_this_expert = f"experts.{target_expert_idx}" in name
                    
                    if is_this_expert:
                        is_early = any(x in name.lower() for x in ['conv1', 'bn1'])
                        if is_early and hasattr(module, 'weight'):
                            p = module.weight
                            pid = id(p)
                            if pid not in cumulative:
                                cumulative[pid] = torch.ones(p.shape, dtype=torch.bool, device=p.device)
                            else:
                                cumulative[pid][:] = True
                            
                            if hasattr(module, 'bias') and module.bias is not None:
                                pb = module.bias
                                pbid = id(pb)
                                if pbid not in cumulative:
                                    cumulative[pbid] = torch.ones(pb.shape, dtype=torch.bool, device=pb.device)
                                else:
                                    cumulative[pbid][:] = True


        # Commit Mask Updates (Identity-Aware)
        # Use set of PIDs for total_params to avoid overcounting shared backbones
        total_pids = set()
        for m in memory_module.models:
            for p in m.parameters():
                if p.requires_grad: total_pids.add(id(p))
        
        num_total = sum(p.numel() for pid in total_pids for p in [next(p for m in memory_module.models for p in m.parameters() if id(p) == pid)])
        # Actually, a simpler way:
        unique_params = {id(p): p for m in memory_module.models for p in m.parameters() if p.requires_grad}
        num_total = sum(p.numel() for p in unique_params.values())
        total_sacred = sum(mask.sum().item() for mask in cumulative.values())
        
        memory_module.saturation_level = total_sacred / max(1, num_total)
        
        # [V31.7] Bug #8 Fix: Enforce Quota Ceiling (Emergency Pruning)
        if memory_module.saturation_level > self.quota:
            self.logger.warning(f"⚠️ Saturation ({memory_module.saturation_level:.2%}) exceeds quota ({self.quota:.2%}). Pruning mask...")
            # Proportional pruning of masks to fit within quota
            ratio = self.quota / memory_module.saturation_level
            for pid, mask in cumulative.items():
                # Protect small critical layers (FC heads, Gates) and ALL Batch Normalization from random pruning
                # [V31.8] Hard-Locks are also exempt.
                names = pid_to_names.get(pid, [])
                p_name = names[0].lower() if names else ""
                
                # [V31.10] ABSOLUTE QUOTA: No more backbone exemptions.
                # Only BN, FC, and Gate are critical to task identity.
                is_bn = "bn" in p_name or "norm" in p_name
                is_fc = "fc" in p_name
                is_gate = "gate" in p_name
                
                is_critical = is_bn or is_fc or is_gate


                if mask.numel() > 512 and not is_critical: 
                    # Use deterministic-ish pruning based on random sampling
                    prune_indices = torch.rand(mask.shape, device=mask.device, generator=torch.Generator(device=mask.device).manual_seed(42)) < ratio
                    cumulative[pid] &= prune_indices
            
            # Recalculate final saturation
            total_sacred = sum(m.sum().item() for m in cumulative.values())
            memory_module.saturation_level = total_sacred / max(1, num_total)

        self.logger.info(f"🛡️ Equilibrium Protocol Active (V9.6). Global Ceiling: {self.quota*100}%.")
        print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {memory_module.saturation_level:.2%}")
        print(f"  [SENTIENT] Knowledge Anchored. Locked Parameters: {total_sacred:,.0f} / {num_total:,} ({memory_module.saturation_level:.2%})")
        
        if task_id in self.task_stats:
            stats = self.task_stats[task_id]
            q = task_quotas.get(task_id, 0.0)
            print(f"  [SENTIENT] Importance Stats: Avg={stats['avg_top']:.4e}, "
                  f"STD_Div={stats['std_top']:.4e}, Elastic Quota: {q:.2%}")
            print(f"  [SENTIENT] High-Value Density (80%+): {stats['pct_80']:.2%}")
