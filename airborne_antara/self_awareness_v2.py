"""
MirrorMind Self-Awareness Framework V2.0 (Production Fixed)
===========================================================
A powerful, state-of-the-art self-awareness system for the wrapper.

FIXES:
1. Reordered OutOfDistributionDetector to prevent NameError.
2. Added tuple unboxing for AdaptiveFramework compatibility.
3. Robust shape matching for loss calculation.

⚠️ EXPERIMENTAL MODULE ⚠️
This module is NOT currently integrated into the core AdaptiveFramework.
Components like AdaptiveLearningController and SelfImprovementPlanner are
defined but never invoked in the main training loop.

To integrate, import and instantiate these classes in your training script:
    from airborne_antara.self_awareness_v2 import HumanLikeSelfAwarenessWrapper
    awareness = HumanLikeSelfAwarenessWrapper(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from datetime import datetime


logger = logging.getLogger('SelfAwareness')


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class AwarenessLevel(Enum):
    """Hierarchy of self-awareness"""
    MICRO = 1      # Individual prediction confidence
    MESO = 2       # Domain/task-level competency
    MACRO = 3      # Overall capability evolution


class LearningPhase(Enum):
    """What learning phase is the model in?"""
    EXPLORATION = 1   # High uncertainty, needs to explore
    CONSOLIDATION = 2 # Moderately confident, consolidating knowledge
    MASTERY = 3       # High confidence, fine-tuning
    UNCERTAINTY = 4   # Encountering new domains, backtracking


@dataclass
class ConfidenceSignal:
    """Encodes confidence about a prediction"""
    prediction_confidence: float  # How sure am I? [0, 1]
    epistemic_uncertainty: float  # How much don't I know? [0, 1]
    aleatoric_uncertainty: float  # How noisy is the data? [0, 1]
    estimated_accuracy: float     # What accuracy do I expect? [0, 1]
    prediction_entropy: float     # Entropy of prediction distribution
    out_of_distribution: bool     # Is this OOD?
    surprise_level: float         # How surprising is this? [0, 1]
    
    @property
    def reliability(self) -> float:
        """Overall reliability score"""
        return self.prediction_confidence * (1 - self.epistemic_uncertainty)


@dataclass
class CompetenceSignal:
    """Encodes competence in a domain"""
    domain_id: str                      
    accuracy_estimate: float            
    task_difficulty_estimate: float     
    mastery_level: float                
    learning_velocity: float            
    convergence_progress: float         
    knowledge_stability: float          
    recommendation: str                 


@dataclass
class MetacognitiveState:
    """Full awareness snapshot"""
    timestamp: datetime
    phase: LearningPhase
    global_confidence: float            
    global_competence: float            
    global_uncertainty: float           
    learning_direction: str             
    prioritized_improvements: List[str] 
    current_bottlenecks: List[str]      
    capability_gaps: List[Tuple[str, float]]  
    estimated_time_to_mastery: float    
    confidence_by_domain: Dict[str, float]
    performance_trajectory: List[float]
    knowledge_entropy: float            


# ============================================================================
# OUT-OF-DISTRIBUTION DETECTION (Moved to top for dependency resolution)
# ============================================================================

class OutOfDistributionDetector:
    """Detects when the model encounters OOD samples"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        self.baseline_mean = 0.5
        self.baseline_std = 0.2
    
    def is_outlier(self,
                   prediction: torch.Tensor,
                   target: torch.Tensor,
                   features: Optional[torch.Tensor] = None) -> bool:
        """
        Determine if this is an out-of-distribution sample.
        """
        # Detach to avoid graph retention
        if prediction.requires_grad: prediction = prediction.detach()
        if target.requires_grad: target = target.detach()

        # Simple MSE for OOD metric
        if prediction.shape != target.shape:
             try:
                 error = F.mse_loss(prediction.view(-1), target.view(-1)).item()
             except:
                 error = 1.0 # Fail safe
        else:
             error = F.mse_loss(prediction, target).item()
        
        # Update buffers
        self.prediction_buffer.append(prediction)
        self.error_buffer.append(error)
        
        # Update baseline statistics
        if len(self.error_buffer) >= 10:
            self.baseline_mean = np.mean(list(self.error_buffer))
            self.baseline_std = np.std(list(self.error_buffer)) + 1e-6
        
        # Z-score based detection
        z_score = (error - self.baseline_mean) / self.baseline_std
        is_ood = abs(z_score) > 2.5
        
        return is_ood


# ============================================================================
# CORE SELF-AWARENESS ENGINE
# ============================================================================

class MetaCognitiveAwarenessEngine:
    """
    The heart of self-awareness. Monitors and understands learning dynamics.
    """
    
    def __init__(self,
                 model: nn.Module,
                 buffer_size: int = 10000,
                 evaluation_window: int = 100,
                 domain_count: int = 10):
        
        self.model = model
        self.buffer_size = buffer_size
        self.eval_window = evaluation_window
        self.domain_count = domain_count
        
        # === MICRO-LEVEL: PREDICTION CONFIDENCE ===
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.confidence_history = deque(maxlen=evaluation_window)
        self.error_history = deque(maxlen=evaluation_window)
        self.uncertainty_estimates = {}
        
        # === MESO-LEVEL: DOMAIN COMPETENCE ===
        self.domain_accuracy = {}  # domain_id -> [accuracies]
        self.domain_mastery = {}   # domain_id -> mastery score
        self.domain_convergence = {}  # domain_id -> convergence progress
        self.task_difficulty_estimates = {}  # task_id -> difficulty score
        
        # === MACRO-LEVEL: LEARNING TRAJECTORY ===
        self.learning_phase = LearningPhase.EXPLORATION
        self.phase_transition_times = []
        self.overall_performance = deque(maxlen=buffer_size)
        self.learning_curve = []  # Track overall improvement
        
        # === KNOWLEDGE STRUCTURE ===
        self.knowledge_map = {}  # Feature/concept -> confidence
        self.knowledge_frontiers = []  # Areas at the edge of knowledge
        self.learning_gaps = {}  # What we're bad at
        self.capability_ceiling = 1.0  # Estimated maximum capability
        
        # === SURPRISE & NOVELTY DETECTION ===
        self.baseline_error_mean = 0.5
        self.baseline_error_std = 0.2
        self.error_z_history = deque(maxlen=evaluation_window)
        self.out_of_distribution_detector = OutOfDistributionDetector(buffer_size=1000)
        
        # === DIAGNOSTICS ===
        self.step_count = 0
        self.episode_count = 0
        self.performance_milestones = []
        
        logger.info(f"Initialized MetaCognitiveAwarenessEngine")
    
    def observe(self,
                prediction: Union[torch.Tensor, Tuple],
                target: torch.Tensor,
                input_data: Optional[torch.Tensor] = None,
                domain_id: Optional[str] = None,
                task_id: Optional[str] = None,
                features: Optional[torch.Tensor] = None) -> ConfidenceSignal:
        """
        Observe a prediction and update awareness.
        """
        self.step_count += 1
        
        # FIX: Handle tuple output from AdaptiveFramework
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        # Ensure detach for safety in metrics
        pred_det = prediction.detach()
        target_det = target.detach()
        
        # FIX: Robust Shape Matching
        if pred_det.shape != target_det.shape:
            # Flatten both to match
            try:
                error = F.mse_loss(pred_det.view(-1), target_det.view(-1), reduction='none').mean()
            except Exception:
                error = torch.tensor(1.0)
        else:
            error = F.mse_loss(pred_det, target_det, reduction='none').mean()
        
        # === CONFIDENCE ESTIMATION ===
        # Method 1: Inverse error
        prediction_confidence = torch.clamp(1.0 - error, 0, 1).item()
        
        # Method 2: Prediction variance (if applicable)
        if hasattr(self.model, 'get_uncertainty'):
            # Only pass input if model expects it
            try:
                epistemic_unc, aleatoric_unc = self.model.get_uncertainty(
                    input_data if input_data is not None else pred_det
                )
            except:
                epistemic_unc, aleatoric_unc = error.item(), 0.1
        else:
            epistemic_unc = error.item()
            aleatoric_unc = 0.1
        
        # Method 3: OOD detection
        is_ood = self.out_of_distribution_detector.is_outlier(
            pred_det, target_det, features
        )
        
        # Method 4: Surprise quantification
        z_score = (error.item() - self.baseline_error_mean) / (self.baseline_error_std + 1e-8)
        surprise_level = torch.sigmoid(torch.tensor(abs(z_score) - 1.0)).item()
        
        # Method 5: Prediction entropy
        if len(pred_det.shape) > 0 and pred_det.shape[-1] > 1:
            probs = F.softmax(pred_det, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        else:
            entropy = 0.0
        
        # === UPDATE RUNNING STATISTICS ===
        self.confidence_history.append(prediction_confidence)
        self.error_history.append(error.item())
        self.error_z_history.append(z_score)
        
        # Update baseline statistics with adaptive EMA
        surprise_magnitude = abs(z_score) if not np.isnan(z_score) and not np.isinf(z_score) else 0.0
        adaptive_ema_weight = min(0.95, max(0.5, 0.95 - 0.1 * np.tanh(surprise_magnitude / 2.0)))
        
        current_error = error.item()
        self.baseline_error_mean = adaptive_ema_weight * self.baseline_error_mean + (1 - adaptive_ema_weight) * current_error
        
        if not hasattr(self, '_error_variance'):
            self._error_variance = 0.0
        variance_increment = (current_error - self.baseline_error_mean) ** 2
        self._error_variance = adaptive_ema_weight * self._error_variance + (1 - adaptive_ema_weight) * variance_increment
        self.baseline_error_std = max(np.sqrt(self._error_variance), 1e-6)
        
        # === DOMAIN-SPECIFIC TRACKING ===
        if domain_id:
            if domain_id not in self.domain_accuracy:
                self.domain_accuracy[domain_id] = []
                self.domain_mastery[domain_id] = 0.0
            
            task_accuracy = 1.0 - error.item()
            self.domain_accuracy[domain_id].append(task_accuracy)
            
            # Update mastery (exponential moving average)
            if self.domain_accuracy[domain_id]:
                avg_acc = np.mean(self.domain_accuracy[domain_id][-self.eval_window:])
                self.domain_mastery[domain_id] = 0.9 * self.domain_mastery[domain_id] + 0.1 * avg_acc
        
        # === LEARNING PHASE DETECTION ===
        self._update_learning_phase()
        
        return ConfidenceSignal(
            prediction_confidence=prediction_confidence,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
            estimated_accuracy=prediction_confidence,
            prediction_entropy=entropy,
            out_of_distribution=is_ood,
            surprise_level=surprise_level
        )
    
    def _update_learning_phase(self):
        """Determine current learning phase based on confidence and uncertainty"""
        if len(self.confidence_history) < 10:
            self.learning_phase = LearningPhase.EXPLORATION
            return
        
        recent_conf = np.mean(list(self.confidence_history)[-self.eval_window:])
        recent_unc = np.std(list(self.error_history)[-self.eval_window:])
        
        if recent_conf < 0.5:
            self.learning_phase = LearningPhase.EXPLORATION
        elif recent_conf < 0.8 and recent_unc > 0.05:
            self.learning_phase = LearningPhase.CONSOLIDATION
        elif recent_conf >= 0.8:
            self.learning_phase = LearningPhase.MASTERY
        else:
            self.learning_phase = LearningPhase.UNCERTAINTY
    
    def get_competence(self, domain_id: str) -> CompetenceSignal:
        """
        Assess competence in a specific domain.
        """
        if domain_id not in self.domain_accuracy:
            return CompetenceSignal(
                domain_id=domain_id,
                accuracy_estimate=0.5,
                task_difficulty_estimate=0.5,
                mastery_level=0.0,
                learning_velocity=0.0,
                convergence_progress=0.0,
                knowledge_stability=0.0,
                recommendation="explore"
            )
        
        accuracies = self.domain_accuracy[domain_id]
        if not accuracies:
            return CompetenceSignal(
                domain_id=domain_id,
                accuracy_estimate=0.5,
                task_difficulty_estimate=0.5,
                mastery_level=0.0,
                learning_velocity=0.0,
                convergence_progress=0.0,
                knowledge_stability=0.0,
                recommendation="explore"
            )
        
        # Compute competence metrics
        recent_accuracy = np.mean(accuracies[-self.eval_window:])
        if len(accuracies) >= 2*self.eval_window:
             accuracy_trend = recent_accuracy - np.mean(accuracies[-2*self.eval_window:-self.eval_window]) 
        else:
             accuracy_trend = 0
             
        stability = 1.0 - np.std(accuracies[-self.eval_window:]) if len(accuracies) >= self.eval_window else 0.5
        task_difficulty = 1.0 - recent_accuracy
        convergence = min(1.0, recent_accuracy / 0.95) if recent_accuracy > 0.5 else 0.0
        
        # Recommendation
        if recent_accuracy < 0.5:
            recommendation = "explore"
        elif recent_accuracy < 0.8:
            recommendation = "consolidate"
        else:
            recommendation = "master"
        
        return CompetenceSignal(
            domain_id=domain_id,
            accuracy_estimate=recent_accuracy,
            task_difficulty_estimate=task_difficulty,
            mastery_level=self.domain_mastery.get(domain_id, 0.0),
            learning_velocity=accuracy_trend,
            convergence_progress=convergence,
            knowledge_stability=stability,
            recommendation=recommendation
        )
    
    def get_metacognitive_state(self) -> MetacognitiveState:
        """
        Get full self-awareness snapshot.
        """
        # Global metrics
        if self.confidence_history:
            global_confidence = np.mean(list(self.confidence_history))
            global_uncertainty = np.std(list(self.error_history))
        else:
            global_confidence = 0.5
            global_uncertainty = 0.5
        
        global_competence = np.mean([self.domain_mastery.get(d, 0.0) 
                                     for d in self.domain_mastery.keys()]) if self.domain_mastery else 0.0
        
        # Identify learning gaps
        capability_gaps = []
        for domain_id, mastery in self.domain_mastery.items():
            if mastery < 0.7:
                importance = 1.0 - mastery
                capability_gaps.append((domain_id, importance))
        
        capability_gaps.sort(key=lambda x: x[1], reverse=True)
        
        # Learning direction
        if global_confidence < 0.6:
            direction = "EXPLORE: Encounter and understand new domains"
        elif global_uncertainty > 0.15:
            direction = "CONSOLIDATE: Stabilize knowledge and reduce variance"
        else:
            direction = "MASTER: Fine-tune and optimize performance"
        
        # Performance trajectory
        perf_trajectory = list(self.learning_curve[-100:])
        
        # Time to mastery estimate
        if self.learning_phase == LearningPhase.MASTERY:
            ttm = 0.0
        elif self.learning_phase == LearningPhase.CONSOLIDATION:
            ttm = 100.0
        elif self.learning_phase == LearningPhase.EXPLORATION:
            ttm = 500.0
        else:
            ttm = 1000.0
        
        # Confidence by domain
        conf_by_domain = {d: self.domain_mastery.get(d, 0.0) for d in self.domain_mastery.keys()}
        
        return MetacognitiveState(
            timestamp=datetime.now(),
            phase=self.learning_phase,
            global_confidence=global_confidence,
            global_competence=global_competence,
            global_uncertainty=global_uncertainty,
            learning_direction=direction,
            prioritized_improvements=[gap[0] for gap in capability_gaps[:3]],
            current_bottlenecks=self._identify_bottlenecks(),
            capability_gaps=capability_gaps,
            estimated_time_to_mastery=ttm,
            confidence_by_domain=conf_by_domain,
            performance_trajectory=perf_trajectory,
            knowledge_entropy=self._compute_knowledge_entropy()
        )
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify what's limiting performance"""
        bottlenecks = []
        
        if self.learning_phase == LearningPhase.EXPLORATION:
            bottlenecks.append("Insufficient domain knowledge")
            bottlenecks.append("High epistemic uncertainty")
        
        error_hist = list(self.error_history)
        if error_hist and np.mean(error_hist[-20:]) > 0.3:
            bottlenecks.append("High error rate limiting progress")
        
        if error_hist and np.std(error_hist[-20:]) > 0.1:
            bottlenecks.append("Unstable performance across samples")
        
        return bottlenecks[:3]
    
    def _compute_knowledge_entropy(self) -> float:
        """How 'spread out' is the model's knowledge?"""
        if not self.domain_mastery:
            return 1.0
        
        probs = np.array(list(self.domain_mastery.values()))
        probs = probs / (probs.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)


# ============================================================================
# ADAPTIVE LEARNING CONTROLLER
# ============================================================================

class AdaptiveLearningController:
    """
    Uses self-awareness to adapt learning strategy dynamically.
    """
    
    def __init__(self,
                 awareness_engine: MetaCognitiveAwarenessEngine,
                 base_lr: float = 1e-3,
                 base_exploration: float = 0.1):
        self.awareness = awareness_engine
        self.base_lr = base_lr
        self.base_exploration = base_exploration
    
    def compute_adaptive_lr(self, domain_id: Optional[str] = None) -> float:
        """
        Compute learning rate based on current confidence.
        """
        if domain_id:
            competence = self.awareness.get_competence(domain_id)
            confidence = competence.accuracy_estimate
        else:
            awareness = self.awareness.get_metacognitive_state()
            confidence = awareness.global_confidence
        
        # Scale LR inversely to confidence
        # Lower confidence = Higher LR (Learn fast)
        lr_multiplier = 1.0 / (1.0 + 2.0 * confidence)
        
        return self.base_lr * lr_multiplier
    
    def compute_exploration_ratio(self) -> float:
        """
        Determine exploration vs exploitation ratio.
        """
        awareness = self.awareness.get_metacognitive_state()
        confidence = awareness.global_confidence
        
        exploration = self.base_exploration * (1.0 - confidence)
        return exploration
    
    def get_learning_recommendation(self) -> Dict[str, Any]:
        """
        Get comprehensive learning recommendations.
        """
        awareness = self.awareness.get_metacognitive_state()
        
        return {
            'phase': awareness.phase.name,
            'learning_rate_multiplier': self.compute_adaptive_lr() / self.base_lr,
            'exploration_ratio': self.compute_exploration_ratio(),
            'focus_areas': awareness.prioritized_improvements,
            'bottlenecks': awareness.current_bottlenecks,
            'direction': awareness.learning_direction,
            'estimated_progress': awareness.global_competence
        }


# ============================================================================
# SELF-IMPROVEMENT PLANNER
# ============================================================================

class SelfImprovementPlanner:
    """
    Uses awareness to plan learning trajectory.
    """
    
    def __init__(self, awareness_engine: MetaCognitiveAwarenessEngine):
        self.awareness = awareness_engine
    
    def get_learning_plan(self, horizon: int = 1000) -> Dict[str, Any]:
        """
        Get a learning plan for the next N steps.
        """
        awareness = self.awareness.get_metacognitive_state()
        
        # Prioritize by importance and current capability gap
        learning_priorities = []
        for gap_name, importance in awareness.capability_gaps:
            priority_score = importance * (1.0 - awareness.confidence_by_domain.get(gap_name, 0.0))
            learning_priorities.append((gap_name, priority_score))
        
        learning_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'phase': awareness.phase.name,
            'primary_focus': learning_priorities[0][0] if learning_priorities else "exploration",
            'secondary_focuses': [p[0] for p in learning_priorities[1:4]],
            'estimated_milestones': self._estimate_milestones(awareness, horizon),
            'transfer_learning_opportunities': self._find_transfer_opportunities(awareness),
            'consolidation_areas': [d for d, m in awareness.confidence_by_domain.items() if 0.6 < m < 0.85],
            'mastered_areas': [d for d, m in awareness.confidence_by_domain.items() if m >= 0.85]
        }
    
    def _estimate_milestones(self, awareness: MetacognitiveState, horizon: int) -> List[str]:
        """Estimate learning milestones"""
        milestones = []
        if awareness.global_confidence < 0.6:
            milestones.append(f"Achieve 60% confidence in {horizon//3} steps")
        if awareness.global_confidence < 0.8:
            milestones.append(f"Reach 80% mastery in {horizon//2} steps")
        if awareness.global_confidence < 0.95:
            milestones.append(f"Approach expert performance in {horizon} steps")
        return milestones
    
    def _find_transfer_opportunities(self, awareness: MetacognitiveState) -> List[Tuple[str, str]]:
        """Find domains that could transfer knowledge"""
        mastered = [d for d, m in awareness.confidence_by_domain.items() if m >= 0.7]
        struggling = [d for d, m in awareness.confidence_by_domain.items() if m < 0.5]
        
        transfers = []
        for m in mastered:
            for s in struggling:
                if self._domains_are_similar(m, s):
                    transfers.append((m, s))
        
        return transfers[:3]
    
    def _domains_are_similar(self, domain1: str, domain2: str) -> bool:
        """Heuristic: are two domains similar?"""
        return len(set(domain1.split('_')) & set(domain2.split('_'))) > 0


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class AdaptiveAttentionMechanism:
    """
    Learns what to pay attention to based on self-awareness.
    """
    
    def __init__(self, awareness_engine: MetaCognitiveAwarenessEngine):
        self.awareness = awareness_engine
        self.feature_importance = {}
        self.task_importance = {}
    
    def compute_sample_importance(self,
                                  prediction: torch.Tensor,
                                  target: torch.Tensor,
                                  features: Optional[torch.Tensor] = None,
                                  domain_id: Optional[str] = None) -> float:
        """
        Compute how important this sample is for learning.
        """
        if prediction.shape != target.shape:
             try:
                 error = F.mse_loss(prediction.view(-1), target.view(-1)).item()
             except:
                 error = 1.0
        else:
             error = F.mse_loss(prediction, target).item()
        
        # Hard examples are important
        error_importance = min(1.0, error / 0.5)
        
        # OOD samples are important (provide new knowledge)
        is_ood = self.awareness.out_of_distribution_detector.is_outlier(prediction, target, features)
        ood_importance = 0.5 if is_ood else 0.0
        
        # Samples in weak domains are important
        if domain_id:
            competence = self.awareness.get_competence(domain_id)
            domain_importance = 1.0 - competence.accuracy_estimate
        else:
            domain_importance = 0.2
        
        # Combine
        total_importance = 0.5 * error_importance + 0.3 * ood_importance + 0.2 * domain_importance
        
        return min(1.0, total_importance)
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get importance weights for each feature"""
        return self.feature_importance
    
    def update_feature_importance(self, gradients: Dict[str, torch.Tensor]):
        """Update which features matter most based on gradients"""
        for feature_name, grad in gradients.items():
            importance = grad.abs().mean().item()
            self.feature_importance[feature_name] = 0.9 * self.feature_importance.get(feature_name, 0.0) + 0.1 * importance


# ============================================================================
# SELF-AWARENESS MONITOR
# ============================================================================

class SelfAwarenessMonitor:
    """Monitors and logs self-awareness metrics"""
    
    def __init__(self, awareness_engine: MetaCognitiveAwarenessEngine):
        self.awareness = awareness_engine
        self.log_history = []
    
    def log_state(self):
        """Log current awareness state"""
        state = self.awareness.get_metacognitive_state()
        
        log_entry = {
            'step': self.awareness.step_count,
            'phase': state.phase.name,
            'confidence': state.global_confidence,
            'competence': state.global_competence,
            'uncertainty': state.global_uncertainty,
            'direction': state.learning_direction,
            'focus': state.prioritized_improvements
        }
        
        self.log_history.append(log_entry)
        
        logger.info(
            f"[Step {log_entry['step']}] "
            f"Phase: {log_entry['phase']} | "
            f"Conf: {log_entry['confidence']:.3f} | "
            f"Comp: {log_entry['competence']:.3f} | "
            f"Focus: {log_entry['focus'][:2]}"
        )
        
        return log_entry
    
    def print_awareness_report(self):
        """Print detailed awareness report"""
        state = self.awareness.get_metacognitive_state()
        
        report = f"""
        
        ╔════════════════════════════════════════════════════════╗
        ║         SELF-AWARENESS REPORT                          ║
        ╚════════════════════════════════════════════════════════╝
        
        LEARNING PHASE: {state.phase.name}
        ─────────────────────────────────────────────────────────
        Global Confidence:      {state.global_confidence:.1%}
        Global Competence:      {state.global_competence:.1%}
        Global Uncertainty:     {state.global_uncertainty:.3f}
        
        LEARNING DIRECTION:
        {state.learning_direction}
        
        CAPABILITY ASSESSMENT:
        ─────────────────────────────────────────────────────────
        {self._format_capabilities(state)}
        
        IMPROVEMENT PRIORITIES:
        ─────────────────────────────────────────────────────────
        {self._format_priorities(state)}
        
        CURRENT BOTTLENECKS:
        ─────────────────────────────────────────────────────────
        {self._format_bottlenecks(state)}
        
        ESTIMATED TIME TO MASTERY: {state.estimated_time_to_mastery:.0f} steps
        
        ╚════════════════════════════════════════════════════════╝
        """
        
        print(report)
        return report
    
    def _format_capabilities(self, state: MetacognitiveState) -> str:
        lines = []
        for domain, conf in state.confidence_by_domain.items():
            bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
            lines.append(f"  {domain:20s} {bar} {conf:.1%}")
        return '\n'.join(lines)
    
    def _format_priorities(self, state: MetacognitiveState) -> str:
        lines = []
        for i, priority in enumerate(state.prioritized_improvements, 1):
            lines.append(f"  {i}. {priority}")
        return '\n'.join(lines)
    
    def _format_bottlenecks(self, state: MetacognitiveState) -> str:
        lines = []
        for bottleneck in state.current_bottlenecks:
            lines.append(f"  • {bottleneck}")
        return '\n'.join(lines)


# ============================================================================
# INTEGRATION WRAPPER
# ============================================================================

class HumanLikeSelfAwarenessWrapper:
    """
    Complete wrapper providing human-like self-awareness to any model.
    """
    
    def __init__(self, model: nn.Module, buffer_size: int = 10000):
        self.model = model
        self.awareness_engine = MetaCognitiveAwarenessEngine(model, buffer_size=buffer_size)
        self.learning_controller = AdaptiveLearningController(self.awareness_engine)
        self.improvement_planner = SelfImprovementPlanner(self.awareness_engine)
        self.attention = AdaptiveAttentionMechanism(self.awareness_engine)
        self.monitor = SelfAwarenessMonitor(self.awareness_engine)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.model(*args, **kwargs)
    
    def observe(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                **kwargs) -> ConfidenceSignal:
        """Observe a prediction and update self-awareness"""
        return self.awareness_engine.observe(prediction, target, **kwargs)
    
    def get_awareness_state(self) -> MetacognitiveState:
        """Get current self-awareness state"""
        return self.awareness_engine.get_metacognitive_state()
    
    def get_learning_recommendations(self) -> Dict[str, Any]:
        """Get learning recommendations"""
        return self.learning_controller.get_learning_recommendation()
    
    def get_learning_plan(self, horizon: int = 1000) -> Dict[str, Any]:
        """Get learning plan for next N steps"""
        return self.improvement_planner.get_learning_plan(horizon)
    
    def compute_adaptive_lr(self, domain_id: Optional[str] = None) -> float:
        """Get adaptive learning rate"""
        return self.learning_controller.compute_adaptive_lr(domain_id)
    
    def compute_sample_importance(self, **kwargs) -> float:
        """Compute importance of a sample"""
        return self.attention.compute_sample_importance(**kwargs)
    
    def print_awareness_report(self) -> str:
        """Print detailed awareness report"""
        return self.monitor.print_awareness_report()
    
    def get_awareness_metrics(self) -> Dict[str, float]:
        """Get key awareness metrics"""
        state = self.get_awareness_state()
        return {
            'confidence': state.global_confidence,
            'competence': state.global_competence,
            'uncertainty': state.global_uncertainty,
            'knowledge_entropy': state.knowledge_entropy,
            'learning_phase': state.phase.name
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Wrap any PyTorch model
    simple_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Create self-aware wrapper
    aware_model = HumanLikeSelfAwarenessWrapper(simple_model)
    
    # Simulate training
    print("Initializing Self-Awareness Engine...")
    for step in range(50):
        # Generate random data
        x = torch.randn(8, 10)
        y = torch.randn(8, 1)
        
        # Forward pass
        output = aware_model.forward(x)
        
        # Observe and update awareness
        confidence = aware_model.observe(
            output, y,
            domain_id='test_domain',
            input_data=x
        )
        
        if step % 10 == 0:
            print(f"[Step {step}] Confidence: {confidence.prediction_confidence:.3f} | Phase: {aware_model.get_awareness_state().phase.name}")

    print("\nSelf-Awareness Integration: SUCCESS")
