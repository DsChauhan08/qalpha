"""
Multi-Horizon LSTM Module.

Implements LSTM v4 architecture with:
- Multi-horizon prediction (1d, 1w, 1m, 6m)
- Heteroscedastic uncertainty quantification
- CAGR-normalized loss functions
- Jumping windows for time series CV
"""

from .architecture import MultiHorizonLSTM, JumpingWindowGenerator
from .trainer import LSTMTrainer

__all__ = ["MultiHorizonLSTM", "JumpingWindowGenerator", "LSTMTrainer"]
