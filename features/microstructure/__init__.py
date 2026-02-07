"""
Market Microstructure Features Module.

Provides microstructure-level features derived from order flow,
bid-ask spread dynamics, trade classification, and volatility signatures.
These features capture information not visible in standard OHLCV data.

Modules:
- Order flow imbalance (buy vs sell pressure)
- Bid-ask spread analysis (liquidity / transaction costs)
- Trade signing (Lee-Ready, tick rule)
- Volatility signature (realised vol as function of sampling frequency)
"""

from __future__ import annotations

from quantum_alpha.features.microstructure.order_flow_imbalance import (
    OrderFlowImbalanceAnalyzer,
)
from quantum_alpha.features.microstructure.bid_ask_spread import (
    BidAskSpreadAnalyzer,
)
from quantum_alpha.features.microstructure.trade_signing import (
    TradeSigningClassifier,
)
from quantum_alpha.features.microstructure.volatility_signature import (
    VolatilitySignatureAnalyzer,
)

__all__ = [
    "OrderFlowImbalanceAnalyzer",
    "BidAskSpreadAnalyzer",
    "TradeSigningClassifier",
    "VolatilitySignatureAnalyzer",
]
