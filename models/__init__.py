"""
Quantum Alpha Models Module.

Contains ML models for trading:
- LSTM v4: Multi-horizon prediction with uncertainty
- Sentiment: FinBERT-based sentiment analysis
- Reinforcement: PPO agent for trading decisions
- Ensemble: Model stacking and signal blending
"""

from typing import Dict, Any

__all__ = [
    "MultiHorizonLSTM",
    "JumpingWindowGenerator",
    "FinBERTSentimentAnalyzer",
    "PPOAgent",
    "TradingEnvironment",
    "EnsembleModel",
]


def get_model_info() -> Dict[str, Any]:
    """Get information about available models."""
    return {
        "lstm_v4": {
            "description": "Multi-horizon LSTM with uncertainty quantification",
            "horizons": ["1d", "1w", "1m", "6m"],
            "requires": ["tensorflow>=2.10"],
        },
        "sentiment": {
            "description": "FinBERT financial sentiment analyzer",
            "model": "ProsusAI/finbert",
            "requires": ["transformers", "torch"],
        },
        "reinforcement": {
            "description": "PPO agent for trading decisions",
            "algorithm": "Proximal Policy Optimization",
            "requires": ["torch"],
        },
        "ensemble": {
            "description": "Model stacking and signal blending",
            "methods": ["voting", "stacking", "blending"],
            "requires": [],
        },
    }
