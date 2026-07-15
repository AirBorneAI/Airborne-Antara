# Update Plan: Airborne-Antara Critical Fixes

This document details the planned structural fixes for the critical architectural issues identified in the Airborne-Antara framework audit.

---

## 1. Summary of What Will Happen

We will resolve six critical bugs across `core.py` and `memory.py` that currently disable experience replay, cause double-incrementing steps, break weight adaptation, and crash the self-distillation system during task dreaming:

1. **A-01: Replay Activation:** Change the default `dream_batch_size` in `AdaptiveFrameworkConfig` from `0` to `32`.
2. **A-03: Double-Increment Fix:** Remove the duplicate `_steps_since_task_start` increment from the `finally` block in `train_step`, and restrict resource-intensive garbage collection to run periodically rather than on every training step.
3. **A-04: Direct Weight Adaptation Recovery:** Move `PerformanceMonitor.adapt_weights()` inside the `try` block of `train_step` before the temporary modifiers cache is cleared to `None`.
4. **A-05: Replay Method Alignment:** Replace the call to `self.memory.accumulate_importance` with the correct method `self.memory.accumulate_path` in `learn_from_buffer`.
5. **A-06: PrioritizedReplayBuffer Correction:** Initialize `self.buffer` as an empty list `[]` instead of pre-allocating with `None` values, resolving the sampling of invalid entries.
6. **A-09: Self-Distillation Unpacking:** Correctly unpack `batch_args` as positional parameters (`*batch_args`) when forwarding through the teacher model.

---

## 2. How it Will Happen (Code Modifications)

### Fix 1: Default Dream Batch Size
**File:** [core.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/core.py#L73)
* Change:
```diff
-    dream_batch_size: int = 0 # [V26.5] Zero-Exemplar Protocol
+    dream_batch_size: int = 32 # Re-enabled Experience Replay Batch Size
```

### Fix 2: Steps Counter and GC Performance Fix
**File:** [core.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/core.py#L1570-L1586)
* Change in `finally` block:
```diff
             # [V26.0] Maintenance: Only clear cache after consolidation or periodically
             # Aggressive clearing in every step tanks performance.
             if self.step_count % 100 == 0:
+                import gc
+                gc.collect()
                 if self.device.type == 'cuda':
                     torch.cuda.empty_cache()
 
             # [V31.8] STRATEGIC MODE: Track task-local progression
             if not hasattr(self, '_last_task_id_seen') or self._last_task_id_seen != task_id:
-                self._steps_since_task_start = 0
                 self._last_task_id_seen = task_id
-            else:
-                self._steps_since_task_start = getattr(self, '_steps_since_task_start', 0) + 1
-                import gc
-                gc.collect()
-                if torch.cuda.is_available():
-                    torch.cuda.empty_cache()
```

### Fix 3: Direct Weight Adaptation Placement
**File:** [core.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/core.py#L1553-L1599)
* Change: Move weight adaptation logic from post-finally to the end of the `try` block (right after `_apply_sacred_restoration()`).
```diff
             # [V17] Post-Optimizer Sacred Restoration (combats weight decay drift)
             # [V31.14] Absolute Restoration: Must be the VERY LAST thing before return.
             self._apply_sacred_restoration()
+
+            # [V8.1] Direct Weight Adaptation via PerformanceMonitor
+            if self.performance_monitor and hasattr(self, 'current_modifiers') and self.current_modifiers is not None:
+                prev_loss = getattr(self, '_last_loss_val', loss.item())
+                self.performance_monitor.adapt_weights(
+                    current_loss=loss.item(),
+                    previous_loss=prev_loss,
+                    activations={
+                        'affine_modifiers': self.current_modifiers,
+                        'telemetry_buffer': self.telemetry_buffer,
+                        'layer_map': getattr(self, 'layer_map', {})
+                    }
+                )
 
         finally:
             # MANDATORY CLEANUP (V17.0 ETERNAL MIND - TOTAL AMNESIA)
...
-        # [V8.1] Direct Weight Adaptation via PerformanceMonitor
-        if self.performance_monitor and hasattr(self, 'current_modifiers') and self.current_modifiers is not None:
-            prev_loss = getattr(self, '_last_loss_val', loss.item())
-            self.performance_monitor.adapt_weights(
-                current_loss=loss.item(),
-                previous_loss=prev_loss,
-                activations={
-                    'affine_modifiers': self.current_modifiers,
-                    'telemetry_buffer': self.telemetry_buffer,
-                    'layer_map': getattr(self, 'layer_map', {})
-                }
-            )
```

### Fix 4: Path Accumulation Alignment in Dreaming
**File:** [core.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/core.py#L1883-L1892)
* Change:
```diff
                     # 7. Optimizer Steps via Scaler
                     # [V9.1] SI Accumulation MUST happen after backward but before optimizer clears grads
                     if self.memory and self.memory.method != 'none':
-                        self.memory.accumulate_importance(param_before)
+                        self.memory.accumulate_path(param_before)
                         
                     self.scaler.step(self.optimizer)
                     self.scaler.update() 
                 else:
                     total_loss.backward()
                     
                     # [V9.1] SI Accumulation MUST happen after backward but before optimizer clears grads
                     if self.memory and self.memory.method != 'none':
-                        self.memory.accumulate_importance(param_before)
+                        self.memory.accumulate_path(param_before)
                         
                     self.optimizer.step()
```

### Fix 5: PrioritizedReplayBuffer Optimization
**File:** [memory.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/memory.py#L1181)
* Change `__init__` and `add`:
```diff
     def __init__(self, capacity: int = 10000, temperature: float = 0.6):
         self.capacity = capacity
         self.temperature = max(temperature, 1e-6)  # safety
-        self.buffer = [None] * capacity # Use list for O(1) index access
+        self.buffer = [] # Initialize as empty list to prevent sampling None elements
         # [V26.0] Vectorized meta-buffers
         self.importances = np.zeros(capacity, dtype=np.float32)
         ...
```
```diff
         # Age existing memories (Vectorized)
         if self.count > 0:
             self.ages[:self.count] += 1
 
-        if len(self.buffer) < self.capacity:
+        if len(self.buffer) < self.capacity:
             self.buffer.append(snapshot)
             self.importances[self.count] = float(importance)
             self.surprises[self.count] = float(z_score)
             self.ages[self.count] = 0
             self.count += 1
             self.ptr = (self.ptr + 1) % self.capacity
```

### Fix 6: Distillation Unpacking Fix
**File:** [core.py](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/Mirror_mind/airborne_antara/core.py#L1847)
* Change:
```diff
                 if hasattr(self, 'teacher_model') and self.teacher_model is not None:
                     with torch.no_grad():
                         # [V31.8] Soften the teacher's knowledge
-                        teacher_logits = self.teacher_model(batch_args)
+                        teacher_out = self.teacher_model(*batch_args)
+                        if isinstance(teacher_out, tuple):
+                            teacher_logits = teacher_out[0]
+                        else:
+                            teacher_logits = teacher_out
+                        if hasattr(teacher_logits, 'logits'):
+                            teacher_logits = teacher_logits.logits
```

---

## 3. What Results to Expect

1. **Active Replay & Dreaming:** The dreaming pipeline will now actively execute because `dream_batch_size` is non-zero, allowing the network to replay past task experiences.
2. **Robustness & Stability:** No more `AttributeError` from calling `accumulate_importance` or `TypeError`/`ShapeError` from passing a list directly to `teacher_model`. No more `None` value sampling from the prioritized replay buffer.
3. **Training Speedup:** Removing step-level garbage collection (`gc.collect()`) and CUDA empty cache and limiting them to every 100 steps will significantly boost step throughput.
4. **Correct Metrics and Warmup:** Correctly incrementing `_steps_since_task_start` exactly once per step ensures that schedules (like Gradient Noise Annealing) run for the exact number of steps intended.
5. **Functional Weight Adaptation:** The PerformanceMonitor will now receive the correct `current_modifiers` and properly adapt weights to improve convergence.
