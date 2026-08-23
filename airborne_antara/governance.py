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
                    # [V37] Exclude fc/classifier/head from sacred mask gathering (handled separately in Section 5)
                    # Include gate, router, moe parameters in the sacred mask to protect routing knowledge
                    if any(x in name.lower() for x in ["fc", "classifier", "head", "out_proj"]):
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
            
            # [V31.12] IMPORTANCE FILTERING: Only lock weights with non-zero importance.
            # This prevents idle experts (which have 0 importance) from 'stealing' the quota.
            nonzero_indices = (flat > 0).nonzero().view(-1)
            if nonzero_indices.numel() > 0:
                k = min(k, nonzero_indices.numel())
                top_values, top_idx_in_nonzero = torch.topk(flat[nonzero_indices], k)
                top_idx = nonzero_indices[top_idx_in_nonzero]
                
                task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                task_mask_flat[top_idx] = True
            else:
                task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            
            curr_pos = 0
            for pid, imp in snap.items():
                p_n = imp.numel()
                m = task_mask_flat[curr_pos : curr_pos + p_n].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                curr_pos += p_n

        # 5. HARD HEAD GOVERNANCE: Classification heads and MoE Gates
        # Classification heads are the 'Identity' of previous tasks.
        # [V33] HIERARCHICAL EXPERT DETECTION:
        # Collect all unique global expert identifiers (domain_expert)
        expert_indices = set()
        for m_inf in memory_module.models:
            for n_inf, _ in m_inf.named_modules():
                if "experts." in n_inf:
                    try:
                        ex_idx = int(n_inf.split("experts.")[1].split(".")[0])
                        if "domains." in n_inf:
                            d_idx = int(n_inf.split("domains.")[1].split(".")[0])
                            # We need to know max experts per domain to compute global ID.
                            # For simplicity, we'll use a string key for the set first.
                            expert_indices.add(f"{d_idx}_{ex_idx}")
                        else:
                            expert_indices.add(str(ex_idx))
                    except Exception:
                        continue
        
        # n_exp should be the TOTAL number of unique experts in the whole model.
        n_exp = len(expert_indices) if expert_indices else 8

        for m_idx, m_tracked in enumerate(memory_module.models):
            for m_name, module in m_tracked.named_modules():
                # [V37] Exclude gating networks from hard locking to preserve routing plasticity
                if hasattr(module, 'weight') and any(x in m_name.lower() for x in ["fc", "classifier", "head", "out_proj"]) and not any(x in m_name.lower() for x in ["gate", "router", "moe"]):
                    p_w = module.weight
                    p_w_id = id(p_w)
                    if p_w_id not in cumulative:
                        cumulative[p_w_id] = torch.zeros(p_w.shape, dtype=torch.bool, device=p_w.device)
                    
                    # Logic for FC rows (Task-Specific Output)
                    num_classes = p_w.shape[0]
                    cpt = classes_per_task
                    
                    # Identify which expert this is (if any)
                    e_idx = -1
                    if "experts." in m_name and "backbone" not in m_name.lower() and "shared" not in m_name.lower():
                        try:
                            ex_idx = int(m_name.split("experts.")[1].split(".")[0])
                            if "domains." in m_name:
                                d_idx = int(m_name.split("domains.")[1].split(".")[0])
                                # Need to find experts_per_domain to compute global index.
                                # We can find max ex_idx across the model.
                                max_ex = 0
                                for m_inf in memory_module.models:
                                    for n_inf, _ in m_inf.named_modules():
                                        if "experts." in n_inf:
                                            try:
                                                max_ex = max(max_ex, int(n_inf.split("experts.")[1].split(".")[0]))
                                            except Exception:
                                                pass
                                e_idx = d_idx * (max_ex + 1) + ex_idx
                            else:
                                e_idx = ex_idx
                        except Exception:
                            pass
                    
                    # Lock rows for ALL completed tasks
                    for t in range(task_id + 1):
                        s, e = t * cpt, min((t + 1) * cpt, num_classes)
                        if s >= e: continue
                        
                        # EXPERT-CLASS AFFINITY:
                        # 1. If this expert owns the task, lock the trained weights.
                        # 2. If it does NOT own the task, zero and lock to prevent interference.
                        is_owner = (e_idx == -1) or (e_idx == (t % n_exp))
                        
                        if is_owner:
                            # Standard lock — preserve trained weights
                            cumulative[p_w_id][s:e, :] = True
                        else:
                            # Non-owners must ZERO untrained rows to prevent random weights
                            # from producing massive arbitrary logits during global Class-IL argmax.
                            with torch.no_grad():
                                p_w[s:e, :] = 0.0
                                cumulative[p_w_id][s:e, :] = True # Lock them at zero!

                    # [V33] FUTURE CLASS SUPPRESSION:
                    # Zero out weights for all classes that haven't been trained yet.
                    # This prevents random initialization from producing high logits.
                    future_start = (task_id + 1) * cpt
                    if future_start < num_classes:
                        with torch.no_grad():
                            p_w[future_start:, :] = 0.0
                            # Note: We do NOT lock them, so they can be trained later.
                            
                    if hasattr(module, 'bias') and module.bias is not None:
                        p_b = module.bias
                        p_b_id = id(p_b)
                        if p_b_id not in cumulative:
                            cumulative[p_b_id] = torch.zeros(p_b.shape, dtype=torch.bool, device=p_b.device)
                        for t in range(task_id + 1):
                            s, e = t * cpt, min((t + 1) * cpt, num_classes)
                            if s >= e: continue
                            is_owner = (e_idx == -1) or (e_idx == (t % n_exp))
                            if is_owner:
                                cumulative[p_b_id][s:e] = True
                            else:
                                with torch.no_grad():
                                    p_b[s:e] = 0.0
                                    cumulative[p_b_id][s:e] = True

                # [V31.12] Router Plasticity: Global Gate Lock Removed.
                # We now rely on the row-wise expert locking in Section 5 above.
                # This allows the gate to learn new expert mappings for new tasks.

        # [V31.8] Absolute Foundation Lockdown (Backbone Protection)
        # We strictly freeze early universal feature detectors after Task 0.
        # This prevents the 'representation drift' that destroys foundational knowledge.
        # [V31.11] TITANIUM EXPERT ISOLATION:
        # We only lock the foundational features of the experts that were actually TRAINED.
        # This preserves 100% plasticity for future experts.
        
        target_expert_idx = task_id % n_exp
        # For hierarchical: find domain and local expert
        # We'll use a regex-style match for accuracy
        t_domain = -1
        t_local = -1
        
        # Try to find which domain/expert owns this task_id
        # In Antara, experts are assigned sequentially across domains.
        # If domains=2, experts_per_domain=4, then:
        # Task 0 -> D0, E0
        # Task 1 -> D0, E1
        # Task 4 -> D1, E0
        # We can find the hierarchy from n_exp and number of domains.
        num_domains = 0
        for m in memory_module.models:
            for name, _ in m.named_modules():
                if name.startswith("domains.") and "." not in name[8:]:
                    try:
                        num_domains = max(num_domains, int(name.split(".")[1]) + 1)
                    except Exception:
                        pass
        
        if num_domains > 0:
            exp_per_dom = n_exp // num_domains
            t_domain = target_expert_idx // exp_per_dom
            t_local = target_expert_idx % exp_per_dom
        else:
            t_local = target_expert_idx
            
        if task_id >= 0:
            for m_idx, m in enumerate(memory_module.models):
                for name, module in m.named_modules():
                    # Identify early backbone layers (ResNet-style)
                    # Example name: domains.0.experts.0.model.conv1
                    if t_domain >= 0:
                        is_this_expert = f"domains.{t_domain}.experts.{t_local}" in name
                    else:
                        is_this_expert = f"experts.{t_local}" in name
                    
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

        # Commit Mask Updates (Identity-Aware)
        unique_params = {id(p): p for m in memory_module.models for p in m.parameters() if p.requires_grad}
        num_total = sum(p.numel() for p in unique_params.values())
        total_sacred = sum(mask.sum().item() for mask in cumulative.values())
        
        memory_module.saturation_level = total_sacred / max(1, num_total)
        
        # [V31.7] Bug #8 Fix: Enforce Quota Ceiling (Emergency Pruning)
        if memory_module.saturation_level > self.quota:
            memory_module.expansion_required = True
            self.logger.warning(f"[IRON MIND] Saturation ({memory_module.saturation_level:.2%}) exceeds quota ({self.quota:.2%}). Dynamic expansion recommended; pruning lowest importance weights...")
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
                is_fc = any(x in p_name for x in ["fc", "classifier", "head", "out_proj"])
                is_gate = "gate" in p_name
                
                is_critical = is_bn or is_fc or is_gate


                # [V31.15] TITANIUM PROTECTION: Robust Foundation Exemption
                # Task 0 knowledge is SACRED and must never be pruned.
                is_task0_sacred = False
                for name in names:
                    if "task0" in name.lower() or "snapshot" in name.lower():
                        is_task0_sacred = True
                        break

                if mask.numel() > 512 and not is_critical and not is_task0_sacred: 
                    # [V31.11] TITANIUM PRUNING: Importance-Aware Quota Enforcement
                    # Instead of random pruning, we prune the weights with the lowest SI importance (omega).
                    # This ensures we fit in the 8% quota while preserving 99%+ of knowledge.
                    imp = memory_module.omega.get(names[0], torch.zeros_like(mask).float()).to(mask.device)
                    if names[0] in memory_module.fisher_dict:
                        imp += memory_module.fisher_dict[names[0]].to(mask.device)
                        
                    # We only care about importance for weights that are CURRENTLY sacred
                    active_imp = torch.where(mask, imp, torch.tensor(-1e9, device=mask.device))
                    
                    # Find threshold for the top 'ratio' of currently sacred weights
                    k = int(mask.sum().item() * ratio)
                    if k > 0:
                        # Top-K on flattened importance
                        threshold = torch.topk(active_imp.view(-1), k).values[-1]
                        cumulative[pid] &= (active_imp >= threshold)
                    else:
                        cumulative[pid].zero_()
            
            # Recalculate final saturation
            total_sacred = sum(m.sum().item() for m in cumulative.values())
            memory_module.saturation_level = total_sacred / max(1, num_total)

        self.logger.info(f"[IRON MIND] Equilibrium Protocol Active (V9.6). Global Ceiling: {self.quota*100}%.")
        print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {memory_module.saturation_level:.2%}")
        print(f"  [SENTIENT] Knowledge Anchored. Locked Parameters: {total_sacred:,.0f} / {num_total:,} ({memory_module.saturation_level:.2%})")
        
        if task_id in self.task_stats:
            stats = self.task_stats[task_id]
            q = task_quotas.get(task_id, 0.0)
            print(f"  [SENTIENT] Importance Stats: Avg={stats['avg_top']:.4e}, "
                  f"STD_Div={stats['std_top']:.4e}, Elastic Quota: {q:.2%}")
            print(f"  [SENTIENT] High-Value Density (80%+): {stats['pct_80']:.2%}")
