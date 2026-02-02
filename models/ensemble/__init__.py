"""
Ensemble Models Module.

Model stacking and signal blending:
- Voting ensemble
- Stacking with meta-learner
- Diversity-weighted ensemble
- Adaptive ensemble
- Signal blending (regime-aware, timeframe-aware)
"""

from .stacking import (
    StackingEnsemble,
    VotingEnsemble,
    DiversityWeightedEnsemble,
    AdaptiveEnsemble,
    ModelPrediction,
    create_stacking_ensemble,
    create_voting_ensemble,
    create_diversity_ensemble,
    create_adaptive_ensemble,
)
from .blending import (
    SignalBlender,
    RegimeAwareBlender,
    TimeframeBlender,
    ConfidenceWeightedBlender,
    HierarchicalBlender,
    TradingSignal,
    BlendedSignal,
    SignalType,
    TimeHorizon,
    create_signal_blender,
    create_regime_aware_blender,
    create_timeframe_blender,
)

__all__ = [
    # Stacking
    "StackingEnsemble",
    "VotingEnsemble",
    "DiversityWeightedEnsemble",
    "AdaptiveEnsemble",
    "ModelPrediction",
    "create_stacking_ensemble",
    "create_voting_ensemble",
    "create_diversity_ensemble",
    "create_adaptive_ensemble",
    # Blending
    "SignalBlender",
    "RegimeAwareBlender",
    "TimeframeBlender",
    "ConfidenceWeightedBlender",
    "HierarchicalBlender",
    "TradingSignal",
    "BlendedSignal",
    "SignalType",
    "TimeHorizon",
    "create_signal_blender",
    "create_regime_aware_blender",
    "create_timeframe_blender",
]
