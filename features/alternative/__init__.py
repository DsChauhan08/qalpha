"""
Alternative Data Features Module.

Provides alternative data signals for trading strategies:
- Congressional trading activity
- Insider trading momentum
- Social media sentiment
- Options market sentiment
- Short interest analysis
- Earnings surprise detection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from quantum_alpha.features.alternative.congress_signal import (
    CongressSignalGenerator,
)
from quantum_alpha.features.alternative.insider_momentum import (
    InsiderMomentumGenerator,
)
from quantum_alpha.features.alternative.social_buzz import (
    SocialBuzzAnalyzer,
)
from quantum_alpha.features.alternative.options_sentiment import (
    OptionsSentimentAnalyzer,
)
from quantum_alpha.features.alternative.short_interest import (
    ShortInterestAnalyzer,
)
from quantum_alpha.features.alternative.earnings_surprise import (
    EarningsSurpriseDetector,
)

__all__ = [
    "CongressSignalGenerator",
    "InsiderMomentumGenerator",
    "SocialBuzzAnalyzer",
    "OptionsSentimentAnalyzer",
    "ShortInterestAnalyzer",
    "EarningsSurpriseDetector",
]
