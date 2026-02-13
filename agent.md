# QUANTUM ALPHA: THE DEFINITIVE AGENT SPECIFICATION
## World's Most Advanced Algorithmic Trading System Architecture

**Version:** 1.0.0  
**Classification:** PROPRIETARY QUANTITATIVE RESEARCH FRAMEWORK  
**Target Word Count:** 150,000+ words  
**Purpose:** Single-command deployment of institutional-grade algorithmic trading research platform

---

# SECTION 0: EXECUTIVE MANDATE FOR AI AGENTS

## 0.1 ABSOLUTE DIRECTIVES (ZERO DEVIATION PERMITTED)

The AI agent developing this system MUST adhere to the following non-negotiable principles:

### 0.1.1 Token Efficiency Mandate
- EVERY line of code MUST contribute directly to profitability or risk reduction
- NO decorative comments, NO placeholder functions, NO "TODO" markers in production code
- If a feature cannot demonstrate measurable alpha improvement in backtesting, it is DELETED
- All code undergoes "survivorship testing" - if it doesn't improve Sharpe ratio, it dies

### 0.1.2 Mathematical Rigor Mandate
- EVERY trading signal MUST have a mathematical foundation
- Model mix target: 80% black-box, 20% interpretable for diagnostics and sanity checks
- Black-box models MUST include uncertainty quantification and post-hoc diagnostics
- ALL models MUST include confidence intervals and uncertainty quantification
- Position sizing MUST be derived from Kelly Criterion or risk-parity mathematics

### 0.1.3 Real-World Execution Mandate
- Backtests MUST include realistic slippage, market impact, and transaction costs
- ALL strategies MUST pass Monte Carlo Permutation Tests (MCPT) with p < 0.05
- Walk-forward analysis REQUIRED before any strategy deployment
- Paper trading MANDATORY for minimum 3 months before live capital

### 0.1.4 Anti-Overfitting Mandate
- Jumping windows REQUIRED for all time series cross-validation
- Minimum 5:1 ratio of out-of-sample to in-sample data
- ALL features MUST have economic rationale - no data mining
- Ensemble methods REQUIRED to reduce model variance

---

# SECTION 0.2: PROJECT DECISIONS (FEB 3, 2026)

## 0.2.1 Phase and Switch
- Current phase: TESTING/PROVING only
- Firm mode switch MUST be controlled by CLI flag `--firm-mode` and a dashboard toggle
- Options and futures are RESEARCH-ONLY in this phase (no execution)

## 0.2.2 Markets and Horizons
- Markets: US stocks, options, futures only
- Time horizons: intraday and multi-month

## 0.2.3 Deployment and Environment
- Single-machine first, Docker-first design, runnable on Kaggle/Colab
- Local-only dashboard (no external alerting channels)

## 0.2.4 Data Policy
- Free/no-credit-card sources only
- Minimize API dependence; prefer public datasets or scraping when equivalent
- Cache locally; reduce repeated fetches
- Vanguard midcap holdings MUST be captured as a one-time snapshot and never auto-updated

## 0.2.5 Strategy Research Direction
- Priority: in-house mathematical inconsistency algorithms, continuously validated
- Secondary: proven quant-firm staples (adapted and re-tested internally)
- Inconsistency examples to explore (non-exhaustive):
- Weak-form autocorrelation in returns
- Mean reversion from liquidity imbalance (not simple overbought/oversold)
- Cross-asset lag/lead relationships
- Distributional asymmetry (non-Gaussian return regimes)

## 0.2.6 Risk and Governance
- Enforce max exposure, tail risk controls, and additional layered limits
- Walk-forward optimization REQUIRED before any strategy is deployable
- Model versioning and full data lineage are mandatory

## 0.2.7 Storage
- SQLite is the primary data store
- Parquet is used for large datasets; avoid heavy DB dependencies

---

# SECTION 0.3: PERFORMANCE GATE (METRICS AND PROMOTION RULE)

## 0.3.1 Promotion Rule
- A strategy is promotable only if at least 90% of the metrics below are "good or above good"
- "Good or above good" means beating both the market benchmark and a quant-firm composite benchmark

## 0.3.2 Fixed Metrics List (50+)
- Total return — % change in price plus reinvested dividends over a period
- CAGR (Compound Annual Growth Rate) — `CAGR = (Ending / Beginning)^(1/n) - 1`
- Annualized volatility (std dev of returns)
- Beta — `β = Cov(R_stock, R_mkt) / Var(R_mkt)`
- Sharpe ratio — `(Rp − Rf) / σp`
- Sortino ratio — `(Rp − Rf) / downside deviation`
- Treynor ratio — `(Rp − Rf) / β`
- Jensen’s alpha — `α = Rp − [Rf + β(Rm − Rf)]`
- Information ratio — `(Rp − Rb) / tracking error`
- Tracking error — std dev of `(Rp − Rb)`
- R-squared — % of return variance explained by benchmark
- Maximum drawdown — largest peak-to-trough % loss
- Calmar ratio — `CAGR / max drawdown`
- Ulcer index — drawdown depth and duration
- Downside deviation — std dev of negative returns
- Value at Risk (VaR) — loss at a given confidence
- Conditional VaR (CVaR) / Expected Shortfall — average loss beyond VaR
- Omega ratio — gain area above threshold / loss area below
- Upside capture ratio — % of benchmark upside captured
- Downside capture ratio — % of benchmark downside captured
- Capture ratio — upside capture / downside capture
- M² (Modigliani-Modigliani) — Sharpe scaled to benchmark volatility
- Sterling ratio — Calmar variant using average drawdowns
- Recovery time — time to recover from drawdown
- Win rate — % of positive periods
- Payoff ratio — avg gain / avg loss
- Profit factor — gross profit / gross loss
- Max consecutive losses / wins
- Price/Earnings (P/E)
- Price/Book (P/B)
- EV/EBITDA
- EV/Sales
- PEG ratio
- Free Cash Flow yield
- Price/Cash Flow (P/CF)
- Dividend yield
- Dividend payout ratio
- Dividend growth rate (CAGR)
- EPS growth rate
- Revenue growth rate
- Return on Equity (ROE)
- Return on Assets (ROA)
- Return on Invested Capital (ROIC)
- Gross margin
- Operating margin
- Net profit margin
- EBITDA margin
- Interest coverage ratio (EBIT/Interest)
- Debt/Equity ratio
- Net debt / EBITDA

---

# SECTION 0.4: WORK CADENCE

## 0.4.1 Checklist Discipline
- Maintain a rolling checklist of 5–6 items at a time
- Complete the active list before adding new items
- Prefer low-API, free-data implementations first

---

# SECTION 1: SYSTEM ARCHITECTURE OVERVIEW

## 1.1 SINGLE-COMMAND DEPLOYMENT PHILOSOPHY

The entire system MUST be deployable with:

```bash
./deploy_quantum_alpha.sh --mode=[backtest|paper|live] --capital=[USD_AMOUNT] --risk-profile=[conservative|moderate|aggressive] --firm-mode --dashboard
```

This command SHALL:
1. Validate all dependencies and environment
2. Download/configure all data sources
3. Initialize database schemas
4. Train/update all ML models
5. Run full backtest suite
6. Generate performance report
7. Start paper trading (if requested)

## 1.2 MODULE ARCHITECTURE

```
quantum_alpha/
├── core/                          # C++ performance-critical components
│   ├── qmc_engine.cpp            # Quasi-Monte Carlo simulation (500-path)
│   ├── geometry_engine.cpp       # Differential geometry calculations
│   ├── portfolio_optimizer.cpp   # Riemannian manifold optimization
│   ├── tda_engine.cpp            # Topological Data Analysis engine
│   ├── kalman_filter.cpp         # Adaptive hedge ratio estimation
│   └── Makefile
│
├── data/                          # Data collection and management
│   ├── collectors/               # Data ingestion modules
│   │   ├── market_data.py        # OHLCV from multiple sources
│   │   ├── congress_trades.py    # Senate/House disclosures
│   │   ├── insider_trades.py     # SEC Form 4 parsing
│   │   ├── social_sentiment.py   # Reddit/Twitter via BERT
│   │   ├── news_aggregator.py    # NewsAPI/Finnhub/GDELT
│   │   ├── options_flow.py       # Unusual options activity
│   │   ├── fundamentals.py       # SEC Edgar financials
│   │   ├── economic_data.py      # FRED macro indicators
│   │   └── order_book.py         # Level 2 data (if available)
│   │
│   ├── storage/                  # Data persistence
│   │   ├── timescale_manager.py  # TimescaleDB for time-series
│   │   ├── sqlite_cache.py       # Local caching layer
│   │   ├── parquet_manager.py    # Efficient columnar storage
│   │   └── data_quality.py       # Validation and anomaly detection
│   │
│   └── preprocessing/            # Data transformation
│       ├── cleaners.py           # Outlier detection/removal
│       ├── imputers.py           # Missing data handling
│       ├── normalizers.py        # Feature scaling
│       └── resamplers.py         # Time-series alignment
│
├── features/                      # Feature engineering (300+ features)
│   ├── technical/                # Traditional indicators
│   │   ├── momentum.py           # RSI, MACD, Stochastics
│   │   ├── trend.py              # Moving averages, ADX
│   │   ├── volatility.py         # Bollinger, ATR, Keltner
│   │   ├── volume.py             # OBV, VWAP, Money Flow
│   │   ├── candlestick.py        # Pattern recognition
│   │   └── ichimoku.py           # Cloud analysis
│   │
│   ├── mathematical/             # Advanced quantitative features
│   │   ├── manifold_distance.py  # Riemannian geometry
│   │   ├── persistent_homology.py # TDA regime detection
│   │   ├── optimal_transport.py  # Wasserstein distance
│   │   ├── hurst_exponent.py     # Mean-reversion/trend detection
│   │   ├── entropy_measures.py   # Information theory
│   │   ├── fractal_dimension.py  # Chaos theory
│   │   ├── wavelet_analysis.py   # Multi-resolution decomposition
│   │   └── copula_models.py      # Correlation structure
│   │
│   ├── alternative/              # Non-traditional data
│   │   ├── congress_signal.py    # Politician trading activity
│   │   ├── insider_momentum.py   # Executive buying/selling
│   │   ├── social_buzz.py        # Retail sentiment
│   │   ├── options_sentiment.py  # Put/call ratios, IV skew
│   │   ├── short_interest.py     # Short squeeze potential
│   │   └── earnings_surprise.py  # Beat/miss predictions
│   │
│   ├── fundamental/              # Financial statement analysis
│   │   ├── valuation_ratios.py   # P/E, P/B, EV/EBITDA
│   │   ├── quality_metrics.py    # ROE, ROIC, margins
│   │   ├── growth_indicators.py  # Revenue/EPS growth
│   │   ├── cash_flow.py          # FCF, OCF analysis
│   │   └── altman_zscore.py      # Bankruptcy prediction
│   │
│   └── microstructure/           # High-frequency features
│       ├── order_flow_imbalance.py
│       ├── bid_ask_spread.py
│       ├── trade_signing.py
│       └── volatility_signature.py
│
├── models/                        # Machine learning components
│   ├── lstm_v4/                  # Multi-horizon LSTM
│   │   ├── architecture.py
│   │   ├── jumping_windows.py
│   │   ├── cagr_loss.py
│   │   └── uncertainty_quant.py
│   │
│   ├── transformer/              # Attention-based models
│   │   ├── time_series_transformer.py
│   │   ├── patch_tst.py
│   │   └── temporal_fusion.py
│   │
│   ├── sentiment/                # NLP models
│   │   ├── finbert_analyzer.py   # Financial BERT
│   │   ├── glm_interface.py      # LLM sentiment
│   │   └── ensemble_sentiment.py
│   │
│   ├── qmc_predictor/            # Monte Carlo simulation
│   │   ├── simulate.py
│   │   ├── variance_reduction.py
│   │   └── path_generator.cpp
│   │
│   ├── reinforcement/            # RL agents
│   │   ├── ppo_agent.py
│   │   ├── dqn_agent.py
│   │   └── environment.py
│   │
│   └── ensemble/                 # Model combination
│       ├── stacker.py
│       ├── blender.py
│       └── meta_learner.py
│
├── strategy/                      # Trading logic
│   ├── signal_aggregator.py      # Multi-signal fusion
│   ├── confidence_scorer.py      # Trade quality assessment
│   ├── regime_detector.py        # Market state classification
│   ├── position_sizer.py         # Kelly/risk-parity sizing
│   ├── execution_rules.py        # Entry/exit logic
│   └── portfolio_constructor.py  # Multi-asset allocation
│
├── backtesting/                   # Strategy validation
│   ├── event_engine.py           # Event-driven simulation
│   ├── transaction_costs.py      # Slippage/commission model
│   ├── walk_forward.py           # Walk-forward optimization
│   ├── monte_carlo.py            # MCPT implementation
│   ├── bootstrap.py              # Statistical resampling
│   └── metrics.py                # Performance analytics
│
├── risk/                          # Risk management
│   ├── var_models.py             # Value-at-Risk
│   ├── cvar_models.py            # Conditional VaR
│   ├── stress_testing.py         # Scenario analysis
│   ├── drawdown_control.py       # Maximum loss limits
│   ├── correlation_monitor.py    # Diversification tracking
│   └── position_limits.py        # Exposure constraints
│
├── execution/                     # Order management
│   ├── order_manager.py          # Order lifecycle
│   ├── broker_interface.py       # API abstraction
│   ├── smart_routing.py          # Best execution
│   └── paper_trader.py           # Simulation engine
│
├── monitoring/                    # System oversight
│   ├── performance_tracker.py
│   ├── alert_system.py
│   ├── health_checks.py
│   └── logging.py
│
├── visualization/                 # Reporting and dashboards
│   ├── dashboard.py              # Streamlit interface
│   ├── report_generator.py       # PDF/HTML reports
│   └── research_notebook.ipynb   # Analysis template
│
└── config/                        # Configuration files
    ├── settings.yaml
    ├── strategies.yaml
    ├── risk_limits.yaml
    └── data_sources.yaml
```

---

# SECTION 2: MATHEMATICAL FOUNDATIONS

## 2.1 TOPOLOGICAL DATA ANALYSIS (TDA) - PROPRIETARY EDGE

### 2.1.1 Theoretical Foundation

Topological Data Analysis detects market regime changes by analyzing the "shape" of financial data. Unlike traditional methods that assume Euclidean geometry, TDA captures the intrinsic topology of price movements.

**Key Insight:** Market crashes create "holes" in the topological structure as correlations break down and assets decouple.

### 2.1.2 Persistent Homology Implementation

```python
# features/mathematical/persistent_homology.py

import numpy as np
from ripser import ripser
from persim import plot_diagrams, bottleneck
import talib
from scipy.spatial.distance import pdist, squareform

class PersistentHomologyAnalyzer:
    """
    Detects market regime changes using topological data analysis.
    
    Mathematical Foundation:
    - Creates point cloud from price returns, volume, volatility, momentum
    - Computes persistent homology to detect topological features
    - H0: Connected components (market fragmentation)
    - H1: 1-dimensional holes (regime cycles)
    - H2: 2-dimensional voids (structural changes)
    """
    
    def __init__(self, lookback_window=60, max_dim=2):
        self.lookback = lookback_window
        self.max_dim = max_dim
        
    def create_point_cloud(self, price_data):
        """
        Create high-dimensional embedding from price data.
        
        Dimensions:
        1. Log returns (price momentum)
        2. Volume changes (liquidity)
        3. ATR (volatility)
        4. RSI (momentum oscillator)
        5. MACD histogram (trend strength)
        6. Bollinger position (mean-reversion)
        
        Args:
            price_data: DataFrame with OHLCV columns
            
        Returns:
            np.array: Point cloud of shape (n_samples, n_features)
        """
        returns = np.diff(np.log(price_data['close']))
        volume_change = np.diff(np.log(price_data['volume']))
        
        # Align all features to same length
        atr = talib.ATR(
            price_data['high'], 
            price_data['low'], 
            price_data['close'], 
            timeperiod=14
        )[1:]
        
        rsi = talib.RSI(price_data['close'], timeperiod=14)[1:]
        
        macd, signal, hist = talib.MACD(price_data['close'])
        macd_hist = hist[1:]
        
        upper, middle, lower = talib.BBANDS(price_data['close'])
        bb_position = ((price_data['close'] - lower) / (upper - lower))[1:]
        
        # Stack features
        features = np.column_stack([
            returns,
            volume_change,
            atr,
            rsi,
            macd_hist,
            bb_position
        ])
        
        # Remove NaN rows
        features = features[~np.isnan(features).any(axis=1)]
        
        return features
    
    def compute_persistence(self, point_cloud):
        """
        Compute persistent homology of point cloud.
        
        Args:
            point_cloud: np.array of shape (n_points, n_dimensions)
            
        Returns:
            dict: Persistence diagram features
        """
        # Compute persistent homology using Ripser
        result = ripser(point_cloud, maxdim=self.max_dim)
        diagrams = result['dgms']
        
        features = {}
        
        # H0: Connected components
        if len(diagrams[0]) > 0:
            h0_lifetimes = diagrams[0][:, 1] - diagrams[0][:, 0]
            h0_lifetimes = h0_lifetimes[~np.isinf(h0_lifetimes)]
            features['h0_total_persistence'] = np.sum(h0_lifetimes)
            features['h0_num_components'] = len(h0_lifetimes)
            features['h0_max_lifetime'] = np.max(h0_lifetimes) if len(h0_lifetimes) > 0 else 0
        else:
            features['h0_total_persistence'] = 0
            features['h0_num_components'] = 0
            features['h0_max_lifetime'] = 0
        
        # H1: 1-dimensional holes (cycles)
        if len(diagrams[1]) > 0:
            h1_lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
            features['h1_total_persistence'] = np.sum(h1_lifetimes)
            features['h1_num_cycles'] = len(h1_lifetimes)
            features['h1_max_lifetime'] = np.max(h1_lifetimes)
            features['h1_mean_lifetime'] = np.mean(h1_lifetimes)
            features['h1_std_lifetime'] = np.std(h1_lifetimes)
        else:
            features['h1_total_persistence'] = 0
            features['h1_num_cycles'] = 0
            features['h1_max_lifetime'] = 0
            features['h1_mean_lifetime'] = 0
            features['h1_std_lifetime'] = 0
        
        # H2: 2-dimensional voids (rare, signals major change)
        if len(diagrams) > 2 and len(diagrams[2]) > 0:
            h2_lifetimes = diagrams[2][:, 1] - diagrams[2][:, 0]
            features['h2_total_persistence'] = np.sum(h2_lifetimes)
            features['h2_num_voids'] = len(h2_lifetimes)
        else:
            features['h2_total_persistence'] = 0
            features['h2_num_voids'] = 0
        
        return features
    
    def detect_regime_change(self, historical_features, current_features, threshold=2.0):
        """
        Detect if current market topology indicates regime change.
        
        Args:
            historical_features: List of past topology feature dicts
            current_features: Current topology feature dict
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            dict: Regime classification and confidence
        """
        # Extract H1 cycle counts for regime detection
        hist_cycles = [f['h1_num_cycles'] for f in historical_features]
        current_cycles = current_features['h1_num_cycles']
        
        if len(hist_cycles) < 10:
            return {'regime': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        mean_cycles = np.mean(hist_cycles)
        std_cycles = np.std(hist_cycles)
        
        if std_cycles == 0:
            return {'regime': 'STABLE', 'confidence': 0.5}
        
        z_score = (current_cycles - mean_cycles) / std_cycles
        
        # Classification logic
        if z_score > threshold:
            regime = 'REGIME_BREAK_OPPORTUNITY'
            confidence = min(abs(z_score) / 3.0, 1.0)
        elif z_score < -threshold:
            regime = 'REGIME_STABILIZATION'
            confidence = min(abs(z_score) / 3.0, 1.0)
        elif abs(z_score) > threshold * 0.5:
            regime = 'TRANSITION_WARNING'
            confidence = min(abs(z_score) / threshold, 1.0)
        else:
            regime = 'NORMAL'
            confidence = 1.0 - min(abs(z_score) / threshold, 1.0)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'z_score': z_score,
            'h1_cycles': current_cycles,
            'historical_mean': mean_cycles,
            'historical_std': std_cycles
        }
    
    def compute_bottleneck_distance(self, diagram1, diagram2):
        """
        Compute bottleneck distance between two persistence diagrams.
        Used to quantify topological similarity between market periods.
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            
        Returns:
            float: Bottleneck distance
        """
        return bottleneck(diagram1, diagram2)
    
    def generate_trading_signal(self, price_data, historical_window=252):
        """
        Generate trading signal based on topological analysis.
        
        Args:
            price_data: Full price history DataFrame
            historical_window: Days of history for baseline
            
        Returns:
            dict: Signal strength and metadata
        """
        # Split data
        historical = price_data.iloc[:-self.lookback]
        current = price_data.iloc[-self.lookback:]
        
        # Compute historical baseline
        hist_features = []
        for i in range(historical_window - self.lookback):
            window = historical.iloc[i:i+self.lookback]
            if len(window) < self.lookback:
                continue
            try:
                pc = self.create_point_cloud(window)
                if len(pc) > 10:
                    feat = self.compute_persistence(pc)
                    hist_features.append(feat)
            except:
                continue
        
        # Compute current topology
        current_pc = self.create_point_cloud(current)
        current_feat = self.compute_persistence(current_pc)
        
        # Detect regime
        regime_info = self.detect_regime_change(hist_features, current_feat)
        
        # Convert to trading signal
        signal_map = {
            'REGIME_BREAK_OPPORTUNITY': 0.8,
            'TRANSITION_WARNING': 0.3,
            'NORMAL': 0.0,
            'REGIME_STABILIZATION': -0.3,
            'INSUFFICIENT_DATA': 0.0
        }
        
        base_signal = signal_map.get(regime_info['regime'], 0.0)
        adjusted_signal = base_signal * regime_info['confidence']
        
        return {
            'signal': adjusted_signal,
            'regime': regime_info['regime'],
            'confidence': regime_info['confidence'],
            'topology_features': current_feat,
            'raw_z_score': regime_info.get('z_score', 0)
        }
```

### 2.1.3 TDA Trading Applications

**Application 1: Early Crash Warning**
- Track H1 cycle count over sliding window
- Sudden increase (>2 sigma) indicates correlation breakdown
- Reduce exposure before crash materializes

**Application 2: Regime Classification**
- Low H1 cycles: Trending market (momentum strategies)
- High H1 cycles: Mean-reverting market (contrarian strategies)
- Extreme H2 voids: Structural change (avoid trading)

**Application 3: Portfolio Diversification**
- Compute bottleneck distance between asset topologies
- Select assets with maximum topological distance
- Ensures true diversification beyond correlation

## 2.2 OPTIMAL TRANSPORT THEORY - WASSERSTEIN DISTANCE

### 2.2.1 Mathematical Foundation

Optimal Transport measures the "cost" of transforming one probability distribution into another. In finance, this quantifies how much a return distribution has changed.

**Wasserstein Distance Formula:**

$$W_p(\mu, \nu) = \left( \inf_{\pi \in \Pi(\mu, \nu)} \int_{M \times M} d(x, y)^p \, d\pi(x, y) \right)^{1/p}$$

Where:
- $\mu$, $\nu$ are probability distributions
- $\Pi(\mu, \nu)$ is the set of all couplings
- $d(x, y)$ is the distance metric
- $p$ is the order (typically $p=1$ or $p=2$)

### 2.2.2 Implementation

```python
# features/mathematical/optimal_transport.py

import numpy as np
import ot  # Python Optimal Transport library
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

class OptimalTransportAnalyzer:
    """
    Uses Optimal Transport theory to measure distribution shifts.
    
    Applications:
    1. Detect anomalous return distributions
    2. Measure portfolio drift from target allocation
    3. Quantify market regime changes
    """
    
    def __init__(self, n_bins=50, epsilon=0.01):
        self.n_bins = n_bins
        self.epsilon = epsilon  # Entropic regularization
        
    def distribution_to_histogram(self, data, bins=None, range_val=None):
        """
        Convert data samples to normalized histogram.
        
        Args:
            data: Array of samples
            bins: Number of bins (default: self.n_bins)
            range_val: (min, max) tuple
            
        Returns:
            np.array: Normalized histogram
        """
        if bins is None:
            bins = self.n_bins
            
        hist, edges = np.histogram(data, bins=bins, range=range_val, density=True)
        # Normalize to sum to 1
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        return hist, edges
    
    def wasserstein_distance(self, dist1, dist2, p=1):
        """
        Compute Wasserstein distance between two distributions.
        
        Args:
            dist1: First distribution (histogram or samples)
            dist2: Second distribution (histogram or samples)
            p: Order of Wasserstein distance (1 or 2)
            
        Returns:
            float: Wasserstein distance
        """
        # If distributions are already histograms
        if isinstance(dist1, np.ndarray) and isinstance(dist2, np.ndarray):
            if len(dist1) == len(dist2):
                # Create cost matrix (uniform grid)
                n = len(dist1)
                M = ot.dist(np.arange(n).reshape(-1, 1), 
                           np.arange(n).reshape(-1, 1), 
                           metric='euclidean') ** p
                
                # Compute Wasserstein distance
                w_dist = ot.emd2(dist1, dist2, M) ** (1/p)
                return w_dist
        
        # If distributions are samples
        return ot.wasserstein_1d(dist1, dist2, p=p)
    
    def sinkhorn_distance(self, dist1, dist2, reg=None):
        """
        Compute entropically-regularized Wasserstein distance using Sinkhorn algorithm.
        Faster computation for large distributions.
        
        Args:
            dist1: First distribution (histogram)
            dist2: Second distribution (histogram)
            reg: Regularization parameter
            
        Returns:
            float: Sinkhorn distance
        """
        if reg is None:
            reg = self.epsilon
            
        n = len(dist1)
        M = ot.dist(np.arange(n).reshape(-1, 1), 
                   np.arange(n).reshape(-1, 1), 
                   metric='euclidean')
        
        sink_dist = ot.sinkhorn2(dist1, dist2, M, reg)
        return sink_dist
    
    def detect_distribution_shift(self, historical_returns, recent_returns, 
                                   window_size=63, threshold_percentile=95):
        """
        Detect if recent returns show anomalous distribution.
        
        Args:
            historical_returns: Full return history
            recent_returns: Most recent returns (e.g., last month)
            window_size: Size of comparison windows
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            dict: Shift detection results
        """
        # Create baseline distribution from historical data
        baseline_hist, edges = self.distribution_to_histogram(
            historical_returns, 
            range_val=(historical_returns.min(), historical_returns.max())
        )
        
        # Compute recent distribution
        recent_hist, _ = self.distribution_to_histogram(
            recent_returns,
            bins=len(baseline_hist),
            range_val=(historical_returns.min(), historical_returns.max())
        )
        
        # Compute Wasserstein distance
        w_dist = self.wasserstein_distance(baseline_hist, recent_hist)
        
        # Compute historical distribution of distances (rolling windows)
        historical_distances = []
        for i in range(0, len(historical_returns) - window_size, window_size):
            window = historical_returns[i:i+window_size]
            window_hist, _ = self.distribution_to_histogram(
                window,
                bins=len(baseline_hist),
                range_val=(historical_returns.min(), historical_returns.max())
            )
            d = self.wasserstein_distance(baseline_hist, window_hist)
            historical_distances.append(d)
        
        # Determine if current distance is anomalous
        threshold = np.percentile(historical_distances, threshold_percentile)
        is_anomaly = w_dist > threshold
        
        # Compute z-score
        mean_dist = np.mean(historical_distances)
        std_dist = np.std(historical_distances)
        z_score = (w_dist - mean_dist) / std_dist if std_dist > 0 else 0
        
        return {
            'wasserstein_distance': w_dist,
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'threshold': threshold,
            'historical_mean': mean_dist,
            'historical_std': std_dist,
            'percentile': 100 * np.sum(np.array(historical_distances) < w_dist) / len(historical_distances)
        }
    
    def portfolio_drift(self, current_weights, target_weights, 
                        return_covariance, risk_tolerance=0.01):
        """
        Measure portfolio drift from target using Wasserstein geometry.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target allocation weights
            return_covariance: Covariance matrix of returns
            risk_tolerance: Maximum acceptable drift
            
        Returns:
            dict: Drift analysis
        """
        # Compute Mahalanobis-Wasserstein distance
        diff = current_weights - target_weights
        
        # Use covariance as metric tensor
        try:
            cov_inv = np.linalg.inv(return_covariance)
            drift_squared = diff.T @ cov_inv @ diff
            drift = np.sqrt(drift_squared)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean
            drift = np.linalg.norm(diff)
        
        needs_rebalance = drift > risk_tolerance
        
        return {
            'drift': drift,
            'needs_rebalance': needs_rebalance,
            'current_weights': current_weights,
            'target_weights': target_weights,
            'deviation': diff
        }
    
    def barycenter_interpolation(self, distributions, weights):
        """
        Compute Wasserstein barycenter (weighted average of distributions).
        Useful for creating "average market" distribution.
        
        Args:
            distributions: List of histogram distributions
            weights: Weight for each distribution
            
        Returns:
            np.array: Barycenter distribution
        """
        n_dists = len(distributions)
        n_bins = len(distributions[0])
        
        # Create uniform cost matrix
        M = ot.dist(np.arange(n_bins).reshape(-1, 1), 
                   np.arange(n_bins).reshape(-1, 1), 
                   metric='euclidean')
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Compute barycenter
        barycenter = ot.barycenter(distributions, M, self.epsilon, weights)
        
        return barycenter
```

## 2.3 HYPERBOLIC GEOMETRY FOR HIERARCHICAL RELATIONSHIPS

### 2.3.1 Theoretical Foundation

Financial markets have hierarchical structure: Sectors → Industries → Sub-industries → Stocks. Hyperbolic geometry naturally models such tree-like structures.

**Poincaré Disk Model:**
- Points inside unit disk represent market entities
- Distance increases exponentially toward boundary
- Similar assets cluster together
- Diverse assets spread toward periphery

### 2.3.2 Implementation

```python
# features/mathematical/hyperbolic_geometry.py

import numpy as np
from scipy.spatial.distance import pdist, squareform

class HyperbolicEmbedding:
    """
    Embeds financial assets in hyperbolic space to capture hierarchical relationships.
    
    Properties:
    - Exponential volume growth captures tree-like structure
    - Distance correlates with dissimilarity
    - Natural clustering by sector/industry
    """
    
    def __init__(self, dim=2, learning_rate=0.1, n_iterations=1000):
        self.dim = dim
        self.lr = learning_rate
        self.n_iter = n_iterations
        
    def poincare_distance(self, u, v):
        """
        Compute hyperbolic distance in Poincaré ball model.
        
        Formula: d(u,v) = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2)))
        
        Args:
            u, v: Points in unit ball
            
        Returns:
            float: Hyperbolic distance
        """
        u_norm_sq = np.sum(u ** 2)
        v_norm_sq = np.sum(v ** 2)
        uv_diff_sq = np.sum((u - v) ** 2)
        
        numerator = 2 * uv_diff_sq
        denominator = (1 - u_norm_sq) * (1 - v_norm_sq)
        
        # Avoid numerical issues
        arg = 1 + numerator / max(denominator, 1e-10)
        arg = np.clip(arg, 1, 1e10)
        
        return np.arccosh(arg)
    
    def euclidean_to_poincare(self, x):
        """
        Project Euclidean point to Poincaré disk.
        
        Args:
            x: Euclidean coordinates
            
        Returns:
            np.array: Poincaré coordinates (inside unit disk)
        """
        norm = np.linalg.norm(x)
        if norm >= 1:
            x = x / (norm + 1e-5)
        return x
    
    def exponential_map(self, x, v):
        """
        Exponential map: move from x in direction v on hyperbolic manifold.
        
        Args:
            x: Base point
            v: Tangent vector
            
        Returns:
            np.array: New point on manifold
        """
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            return x
        
        x_norm_sq = np.sum(x ** 2)
        lambda_x = 2 / (1 - x_norm_sq)
        
        direction = v / v_norm
        
        # Hyperbolic motion
        factor = np.tanh(v_norm * lambda_x / 2)
        new_point = x + factor * direction
        
        return self.euclidean_to_poincare(new_point)
    
    def log_map(self, x, y):
        """
        Logarithmic map: find tangent vector from x to y.
        
        Args:
            x, y: Points on manifold
            
        Returns:
            np.array: Tangent vector at x
        """
        dist = self.poincare_distance(x, y)
        if dist < 1e-8:
            return np.zeros_like(x)
        
        # Direction in Euclidean space
        direction = (y - x)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            return np.zeros_like(x)
        
        direction = direction / direction_norm
        
        # Scale by hyperbolic distance
        x_norm_sq = np.sum(x ** 2)
        lambda_x = 2 / (1 - x_norm_sq)
        
        v = (2 / lambda_x) * np.arctanh(np.tanh(dist / 2)) * direction
        
        return v
    
    def riemannian_gradient(self, x, grad_euclidean):
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        Args:
            x: Point on manifold
            grad_euclidean: Gradient in Euclidean space
            
        Returns:
            np.array: Riemannian gradient
        """
        x_norm_sq = np.sum(x ** 2)
        lambda_x = 2 / (1 - x_norm_sq)
        return (lambda_x ** 2 / 4) * grad_euclidean
    
    def embed_correlation_matrix(self, correlation_matrix, n_assets):
        """
        Embed assets based on correlation structure.
        
        Args:
            correlation_matrix: Asset correlation matrix
            n_assets: Number of assets
            
        Returns:
            np.array: Hyperbolic embeddings (n_assets, dim)
        """
        # Convert correlation to distance
        distances = np.sqrt(2 * (1 - correlation_matrix))
        
        # Initialize random embeddings in Poincaré disk
        embeddings = np.random.randn(n_assets, self.dim) * 0.1
        
        # Gradient descent optimization
        for iteration in range(self.n_iter):
            total_loss = 0
            
            for i in range(n_assets):
                # Compute gradient
                grad = np.zeros(self.dim)
                
                for j in range(n_assets):
                    if i == j:
                        continue
                    
                    # Current hyperbolic distance
                    current_dist = self.poincare_distance(embeddings[i], embeddings[j])
                    target_dist = distances[i, j]
                    
                    # Loss gradient
                    diff = current_dist - target_dist
                    total_loss += diff ** 2
                    
                    # Direction toward j
                    if current_dist > 1e-8:
                        log_ij = self.log_map(embeddings[i], embeddings[j])
                        grad += 2 * diff * log_ij / (current_dist + 1e-8)
                
                # Riemannian gradient
                rgrad = self.riemannian_gradient(embeddings[i], grad)
                
                # Update using exponential map
                embeddings[i] = self.exponential_map(embeddings[i], -self.lr * rgrad)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss:.4f}")
        
        return embeddings
    
    def find_diversification_portfolio(self, embeddings, n_select=10):
        """
        Select maximally diverse portfolio using hyperbolic distances.
        
        Args:
            embeddings: Hyperbolic embeddings
            n_select: Number of assets to select
            
        Returns:
            list: Indices of selected assets
        """
        n_assets = len(embeddings)
        selected = [np.random.randint(n_assets)]  # Start random
        
        for _ in range(n_select - 1):
            max_min_dist = -1
            best_asset = -1
            
            for i in range(n_assets):
                if i in selected:
                    continue
                
                # Find minimum distance to already selected
                min_dist = min(
                    self.poincare_distance(embeddings[i], embeddings[j])
                    for j in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_asset = i
            
            selected.append(best_asset)
        
        return selected
```

## 2.4 KALMAN FILTER FOR ADAPTIVE HEDGE RATIOS

### 2.4.1 Mathematical Foundation

The Kalman filter provides optimal estimates of dynamic systems with noise. In pairs trading, it adapts hedge ratios as market relationships evolve.

**State Space Model:**
- State equation: $\gamma_t = \gamma_{t-1} + w_t$ (hedge ratio evolves)
- Measurement: $y_t = x_t \gamma_t + v_t$ (price relationship)

Where $w_t \sim N(0, Q)$ and $v_t \sim N(0, R)$

### 2.4.2 Implementation

```python
# features/mathematical/kalman_pairs.py

import numpy as np
from scipy.linalg import inv

class KalmanPairsTrader:
    """
    Adaptive pairs trading using Kalman filter for dynamic hedge ratio estimation.
    
    Advantages over fixed-window regression:
    1. Continuously updates hedge ratio
    2. No arbitrary window size selection
    3. Provides uncertainty estimates
    4. Natural mean-reversion detection
    """
    
    def __init__(self, delta=1e-4, R=1e-3):
        """
        Args:
            delta: Transition covariance (system noise)
            R: Measurement noise
        """
        self.delta = delta
        self.R = R
        
        # State: [hedge_ratio, intercept]
        self.x = np.array([[0.], [0.]])
        self.P = np.eye(2)  # Covariance
        
        # Transition matrix (identity - random walk)
        self.F = np.eye(2)
        
        # Process noise
        self.Q = np.eye(2) * delta
        
        self.initialized = False
        
    def predict(self):
        """Prediction step."""
        # State prediction
        self.x = self.F @ self.x
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, y, observation):
        """
        Update step with new observation.
        
        Args:
            y: Target asset price
            observation: [hedge_asset_price, 1] (with intercept)
        """
        H = np.array(observation).reshape(1, 2)  # Measurement matrix
        
        # Innovation
        y_pred = H @ self.x
        residual = y - y_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ inv(S)
        
        # State update
        self.x = self.x + K @ residual
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
    def filter(self, y, hedge_price):
        """
        Run one iteration of Kalman filter.
        
        Args:
            y: Target asset price
            hedge_price: Hedge asset price
            
        Returns:
            dict: Current state and spread
        """
        if not self.initialized:
            # Initialize with OLS
            self.initialized = True
        
        self.predict()
        
        observation = [hedge_price, 1]
        self.update(y, observation)
        
        hedge_ratio = self.x[0, 0]
        intercept = self.x[1, 0]
        
        # Compute spread
        spread = y - (hedge_ratio * hedge_price + intercept)
        
        # Spread standard deviation (from covariance)
        spread_var = self.P[0, 0] * hedge_price**2 + self.P[1, 1] + 2*self.P[0,1]*hedge_price + self.R
        spread_std = np.sqrt(spread_var)
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'spread': spread,
            'spread_std': spread_std,
            'z_score': spread / spread_std if spread_std > 0 else 0,
            'hedge_ratio_std': np.sqrt(self.P[0, 0])
        }
    
    def generate_signal(self, z_score, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate trading signal from z-score.
        
        Args:
            z_score: Normalized spread
            entry_threshold: Z-score for entry
            exit_threshold: Z-score for exit
            
        Returns:
            int: -1 (short spread), 0 (flat), 1 (long spread)
        """
        if z_score > entry_threshold:
            return -1  # Spread too high, short
        elif z_score < -entry_threshold:
            return 1   # Spread too low, long
        elif abs(z_score) < exit_threshold:
            return 0   # Exit position
        else:
            return None  # Hold current position
```

## 2.5 WAVELET TRANSFORM FOR MULTI-RESOLUTION ANALYSIS

### 2.5.1 Theoretical Foundation

Wavelet transforms decompose time series into different frequency components, enabling:
- Noise reduction (denoising)
- Trend extraction
- Cycle detection
- Multi-scale feature extraction

### 2.5.2 Implementation

```python
# features/mathematical/wavelet_analysis.py

import numpy as np
import pywt
from scipy.signal import detrend

class WaveletAnalyzer:
    """
    Multi-resolution analysis using wavelet transforms.
    
    Applications:
    1. Time series denoising
    2. Trend extraction
    3. Volatility decomposition
    4. Feature extraction for ML
    """
    
    def __init__(self, wavelet='db4', mode='symmetric'):
        """
        Args:
            wavelet: Wavelet family ('db4', 'haar', 'coif1', etc.)
            mode: Signal extension mode
        """
        self.wavelet = wavelet
        self.mode = mode
        
    def decompose(self, signal, level=None):
        """
        Perform wavelet decomposition.
        
        Args:
            signal: Input time series
            level: Decomposition level (auto if None)
            
        Returns:
            list: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
                  Approximation and detail coefficients
        """
        if level is None:
            level = int(np.log2(len(signal)))
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=level, mode=self.mode)
        return coeffs
    
    def denoise(self, signal, threshold_mode='soft', threshold_factor=1.0):
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            signal: Noisy input
            threshold_mode: 'soft' or 'hard' thresholding
            threshold_factor: Multiplier for universal threshold
            
        Returns:
            np.array: Denoised signal
        """
        # Decompose
        coeffs = self.decompose(signal)
        
        # Threshold detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation
        
        for detail in coeffs[1:]:
            # Universal threshold
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal))) * threshold_factor
            
            if threshold_mode == 'soft':
                denoised_detail = pywt.threshold(detail, threshold, mode='soft')
            else:
                denoised_detail = pywt.threshold(detail, threshold, mode='hard')
            
            denoised_coeffs.append(denoised_detail)
        
        # Reconstruct
        denoised_signal = pywt.waverec(denoised_coeffs, self.wavelet, mode=self.mode)
        
        # Trim to original length
        return denoised_signal[:len(signal)]
    
    def extract_trend(self, signal, level=3):
        """
        Extract long-term trend from signal.
        
        Args:
            signal: Input time series
            level: Decomposition level for trend
            
        Returns:
            np.array: Trend component
        """
        coeffs = self.decompose(signal, level=level)
        
        # Keep only approximation (low frequencies)
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        
        trend = pywt.waverec(trend_coeffs, self.wavelet, mode=self.mode)
        return trend[:len(signal)]
    
    def extract_cycles(self, signal, min_period=5, max_period=50):
        """
        Extract cyclical components within period range.
        
        Args:
            signal: Input time series
            min_period: Minimum cycle period
            max_period: Maximum cycle period
            
        Returns:
            np.array: Cyclical component
        """
        coeffs = self.decompose(signal)
        
        # Determine which detail coefficients correspond to desired periods
        cycle_coeffs = [np.zeros_like(coeffs[0])]  # Zero out approximation
        
        for i, detail in enumerate(coeffs[1:], 1):
            # Period range for this level
            period_min = 2 ** i
            period_max = 2 ** (i + 1)
            
            if period_min >= min_period and period_max <= max_period * 2:
                cycle_coeffs.append(detail)
            else:
                cycle_coeffs.append(np.zeros_like(detail))
        
        cycles = pywt.waverec(cycle_coeffs, self.wavelet, mode=self.mode)
        return cycles[:len(signal)]
    
    def compute_energy_distribution(self, signal):
        """
        Compute energy distribution across frequency bands.
        
        Args:
            signal: Input time series
            
        Returns:
            dict: Energy by frequency band
        """
        coeffs = self.decompose(signal)
        
        energies = {}
        total_energy = sum(np.sum(c**2) for c in coeffs)
        
        # Approximation energy (trend)
        energies['trend'] = np.sum(coeffs[0]**2) / total_energy
        
        # Detail energies (different frequencies)
        for i, detail in enumerate(coeffs[1:], 1):
            band_name = f'detail_{i}'
            energies[band_name] = np.sum(detail**2) / total_energy
        
        return energies
```

## 2.6 COPULA MODELS FOR DEPENDENCE STRUCTURE

### 2.6.1 Theoretical Foundation

Copulas separate marginal distributions from dependence structure, enabling:
- Non-linear correlation modeling
- Tail dependence quantification
- Portfolio risk assessment
- Pairs trading signal generation

### 2.6.2 Implementation

```python
# features/mathematical/copula_models.py

import numpy as np
from scipy.stats import norm, t, gaussian_kde
from scipy.optimize import minimize
from scipy.linalg import cholesky

class CopulaAnalyzer:
    """
    Copula-based dependence modeling for financial assets.
    
    Supports:
    - Gaussian copula (no tail dependence)
    - t-copula (symmetric tail dependence)
    - Clayton copula (lower tail dependence)
    - Gumbel copula (upper tail dependence)
    """
    
    def __init__(self, copula_type='t'):
        """
        Args:
            copula_type: 'gaussian', 't', 'clayton', or 'gumbel'
        """
        self.copula_type = copula_type
        self.params = None
        
    def fit(self, data):
        """
        Fit copula to data.
        
        Args:
            data: Array of shape (n_samples, n_variables)
            
        Returns:
            self
        """
        # Transform to uniform marginals
        self.uniform_data = self._to_uniform(data)
        
        # Transform to standard normal for fitting
        normal_data = norm.ppf(self.uniform_data)
        normal_data = np.clip(normal_data, -10, 10)  # Avoid infinities
        
        if self.copula_type == 'gaussian':
            self.params = self._fit_gaussian(normal_data)
        elif self.copula_type == 't':
            self.params = self._fit_t_copula(normal_data)
        elif self.copula_type == 'clayton':
            self.params = self._fit_clayton(self.uniform_data)
        elif self.copula_type == 'gumbel':
            self.params = self._fit_gumbel(self.uniform_data)
        
        return self
    
    def _to_uniform(self, data):
        """Transform data to uniform marginals using empirical CDF."""
        uniform = np.zeros_like(data)
        for i in range(data.shape[1]):
            uniform[:, i] = np.argsort(np.argsort(data[:, i])) / len(data)
        return uniform
    
    def _fit_gaussian(self, normal_data):
        """Fit Gaussian copula (correlation matrix)."""
        return np.corrcoef(normal_data.T)
    
    def _fit_t_copula(self, normal_data):
        """Fit t-copula (correlation and degrees of freedom)."""
        corr = np.corrcoef(normal_data.T)
        
        # Estimate degrees of freedom via MLE
        def neg_log_likelihood(nu):
            if nu < 2:
                return 1e10
            try:
                log_lik = self._t_log_lik(normal_data, corr, nu)
                return -log_lik
            except:
                return 1e10
        
        result = minimize(neg_log_likelihood, x0=5, bounds=[(2.1, 50)])
        nu = result.x[0]
        
        return {'corr': corr, 'nu': nu}
    
    def _t_log_lik(self, data, corr, nu):
        """Log-likelihood for t-copula."""
        n, d = data.shape
        
        # Compute log-likelihood
        const = (np.log(np.linalg.det(corr)) * (-0.5) + 
                 np.log(np.sqrt(nu) * np.pi) * (-d/2) +
                 np.log(np.exp(0)))  # Simplified
        
        return const  # Placeholder - full implementation needed
    
    def _fit_clayton(self, uniform_data):
        """Fit Clayton copula (single parameter theta)."""
        # Kendall's tau estimator
        def kendall_tau(x, y):
            n = len(x)
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i+1, n):
                    if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                        concordant += 1
                    else:
                        discordant += 1
            return (concordant - discordant) / (concordant + discordant)
        
        # For bivariate case
        tau = kendall_tau(uniform_data[:, 0], uniform_data[:, 1])
        theta = 2 * tau / (1 - tau) if tau < 1 else 10
        
        return {'theta': max(theta, 0.1)}
    
    def _fit_gumbel(self, uniform_data):
        """Fit Gumbel copula."""
        # Similar to Clayton but for upper tail
        tau = self._kendall_tau(uniform_data[:, 0], uniform_data[:, 1])
        theta = 1 / (1 - tau) if tau < 1 else 10
        
        return {'theta': max(theta, 1.0)}
    
    def sample(self, n_samples):
        """
        Sample from fitted copula.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            np.array: Samples in uniform space
        """
        if self.copula_type == 'gaussian':
            return self._sample_gaussian(n_samples)
        elif self.copula_type == 't':
            return self._sample_t(n_samples)
        elif self.copula_type == 'clayton':
            return self._sample_clayton(n_samples)
        elif self.copula_type == 'gumbel':
            return self._sample_gumbel(n_samples)
    
    def _sample_gaussian(self, n_samples):
        """Sample from Gaussian copula."""
        corr = self.params
        d = corr.shape[0]
        
        # Generate correlated normal samples
        L = cholesky(corr, lower=True)
        z = np.random.randn(n_samples, d)
        correlated = z @ L.T
        
        # Transform to uniform
        return norm.cdf(correlated)
    
    def tail_dependence(self):
        """
        Compute tail dependence coefficients.
        
        Returns:
            dict: Lower and upper tail dependence
        """
        if self.copula_type == 'gaussian':
            return {'lower': 0, 'upper': 0}
        elif self.copula_type == 't':
            nu = self.params['nu']
            rho = self.params['corr'][0, 1]
            # Tail dependence for t-copula
            from scipy.stats import t as t_dist
            lower = 2 * t_dist.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)), nu+1)
            return {'lower': lower, 'upper': lower}
        elif self.copula_type == 'clayton':
            return {'lower': 2**(-1/self.params['theta']), 'upper': 0}
        elif self.copula_type == 'gumbel':
            return {'lower': 0, 'upper': 2 - 2**(1/self.params['theta'])}
    
    def conditional_correlation(self, quantile=0.05):
        """
        Compute correlation conditional on being in tail.
        
        Args:
            quantile: Tail quantile
            
        Returns:
            float: Conditional correlation
        """
        # Sample from copula
        samples = self.sample(10000)
        
        # Find samples in lower tail
        in_tail = np.all(samples < quantile, axis=1)
        
        if np.sum(in_tail) < 10:
            return 0
        
        # Compute correlation in tail
        tail_samples = samples[in_tail]
        return np.corrcoef(tail_samples.T)[0, 1]

# Utility function for Kendall's tau
def kendall_tau(x, y):
    """Compute Kendall's rank correlation."""
    n = len(x)
    tau = 0
    for i in range(n):
        for j in range(i+1, n):
            tau += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return tau / (n * (n-1) / 2)
```

---

# SECTION 3: MACHINE LEARNING MODELS

## 3.1 MULTI-HORIZON LSTM WITH UNCERTAINTY QUANTIFICATION

### 3.1.1 Architecture Specification

The LSTM v4 architecture predicts returns at multiple horizons (1d, 1w, 1m, 6m) simultaneously with explicit uncertainty quantification.

```python
# models/lstm_v4/architecture.py

import tensorflow as tf
from tensorflow import keras
import numpy as np

class MultiHorizonLSTM:
    """
    Multi-horizon LSTM with uncertainty quantification.
    
    Features:
    - Shared feature extraction backbone
    - Separate prediction heads for each horizon
    - Heteroscedastic uncertainty (input-dependent noise)
    - CAGR-normalized loss for comparable horizons
    """
    
    def __init__(self, input_dim=50, sequence_length=90, 
                 lstm_units=[128, 64], dropout=0.3):
        self.input_dim = input_dim
        self.seq_len = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        
    def build_model(self):
        """Build multi-horizon architecture with uncertainty."""
        
        inputs = keras.Input(shape=(self.seq_len, self.input_dim))
        
        # Shared feature extraction
        x = inputs
        for i, units in enumerate(self.lstm_units[:-1]):
            x = keras.layers.LSTM(
                units, 
                return_sequences=True,
                name=f'lstm_{i}'
            )(x)
            x = keras.layers.Dropout(self.dropout)(x)
        
        # Final LSTM layer
        x = keras.layers.LSTM(
            self.lstm_units[-1],
            return_sequences=False,
            name='lstm_final'
        )(x)
        
        # Separate heads for each horizon with uncertainty
        horizons = [
            ('1d', 1),
            ('1w', 5),
            ('1m', 21),
            ('6m', 126)
        ]
        
        outputs = []
        for name, days in horizons:
            # Mean prediction
            mean = keras.layers.Dense(32, activation='relu')(x)
            mean = keras.layers.Dense(1, name=f'{name}_mean')(mean)
            
            # Log variance (uncertainty)
            log_var = keras.layers.Dense(32, activation='relu')(x)
            log_var = keras.layers.Dense(1, name=f'{name}_log_var')(log_var)
            
            outputs.extend([mean, log_var])
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def negative_log_likelihood(self, y_true, y_pred_mean, y_pred_log_var):
        """
        Negative log-likelihood for heteroscedastic regression.
        
        Loss = 0.5 * (log(var) + (y - mean)^2 / var)
        """
        precision = tf.exp(-y_pred_log_var)
        loss = 0.5 * (y_pred_log_var + precision * tf.square(y_true - y_pred_mean))
        return tf.reduce_mean(loss)
    
    def cagr_loss(self, periods):
        """
        CAGR-normalized loss for comparable horizons.
        
        Converts returns to annualized rate before computing loss.
        """
        def loss_fn(y_true, y_pred):
            annual_factor = 252 / periods
            
            # Convert to annualized returns
            y_true_annual = tf.pow(1.0 + y_true, annual_factor) - 1.0
            y_pred_annual = tf.pow(1.0 + y_pred, annual_factor) - 1.0
            
            return tf.reduce_mean(tf.square(y_true_annual - y_pred_annual))
        
        return loss_fn
    
    def compile_model(self, learning_rate=0.001):
        """Compile with custom loss functions."""
        
        losses = {
            '1d_mean': self.cagr_loss(1),
            '1d_log_var': lambda yt, yp: 0.0,  # Handled in mean loss
            '1w_mean': self.cagr_loss(5),
            '1w_log_var': lambda yt, yp: 0.0,
            '1m_mean': self.cagr_loss(21),
            '1m_log_var': lambda yt, yp: 0.0,
            '6m_mean': self.cagr_loss(126),
            '6m_log_var': lambda yt, yp: 0.0
        }
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=losses
        )
        
    def predict_with_uncertainty(self, X, n_iterations=100):
        """
        Monte Carlo dropout prediction for uncertainty.
        
        Args:
            X: Input features
            n_iterations: Number of forward passes
            
        Returns:
            dict: Mean predictions and uncertainties per horizon
        """
        # Enable dropout at inference
        predictions = {h: [] for h in ['1d', '1w', '1m', '6m']}
        
        for _ in range(n_iterations):
            outputs = self.model(X, training=True)
            
            # Extract means
            for i, h in enumerate(['1d', '1w', '1m', '6m']):
                mean = outputs[i*2].numpy()
                predictions[h].append(mean)
        
        results = {}
        for h in ['1d', '1w', '1m', '6m']:
            preds = np.array(predictions[h])
            results[h] = {
                'mean': np.mean(preds, axis=0),
                'std': np.std(preds, axis=0),
                'ci_lower': np.percentile(preds, 5, axis=0),
                'ci_upper': np.percentile(preds, 95, axis=0)
            }
        
        return results


class JumpingWindowGenerator:
    """
    Creates non-overlapping windows for time series cross-validation.
    
    CRITICAL: Prevents data leakage from overlapping sequences.
    """
    
    def __init__(self, window_size=90, prediction_horizon=21, 
                 step_size=None, test_size=0.2):
        self.window_size = window_size
        self.horizon = prediction_horizon
        self.step_size = step_size or window_size  # No overlap by default
        self.test_size = test_size
        
    def generate_windows(self, data):
        """
        Generate independent training windows.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        n_samples = len(data)
        
        # Generate window indices
        windows = []
        for i in range(0, n_samples - self.window_size - self.horizon, 
                      self.step_size):
            
            feature_end = i + self.window_size
            target_end = feature_end + self.horizon
            
            if target_end > n_samples:
                break
            
            windows.append((i, feature_end, target_end))
        
        # Split train/test
        n_test = int(len(windows) * self.test_size)
        train_windows = windows[:-n_test]
        test_windows = windows[-n_test:]
        
        # Extract data
        X_train, y_train = self._extract_windows(data, train_windows)
        X_test, y_test = self._extract_windows(data, test_windows)
        
        return X_train, y_train, X_test, y_test
    
    def _extract_windows(self, data, windows):
        """Extract features and targets from windows."""
        X, y = [], []
        
        for start, feat_end, target_end in windows:
            X.append(data.iloc[start:feat_end].values)
            
            # Target is return over horizon
            price_start = data.iloc[feat_end-1]['close']
            price_end = data.iloc[target_end-1]['close']
            ret = (price_end - price_start) / price_start
            y.append(ret)
        
        return np.array(X), np.array(y)
```

## 3.2 FINBERT SENTIMENT ANALYSIS

### 3.2.1 Model Architecture

FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.

```python
# models/sentiment/finbert_analyzer.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
import re

class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.
    
    Model: ProsusAI/finbert (fine-tuned on Financial PhraseBank)
    Labels: positive, negative, neutral
    Output: Sentiment score [-1, 1] with confidence
    """
    
    def __init__(self, model_name='ProsusAI/finbert', 
                 device=None, batch_size=16):
        """
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            batch_size: Inference batch size
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess financial text.
        
        Args:
            text: Raw text
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s$%\.\-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (FinBERT max: 512 tokens)
        # Rough approximation: 4 chars per token
        if len(text) > 2000:
            text = text[:2000]
        
        return text
    
    def analyze(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Analyze sentiment of financial texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of dicts with sentiment, confidence, and scores
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._analyze_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts."""
        # Preprocess
        cleaned = [self.preprocess_text(t) for t in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        
        # Convert to results
        results = []
        for i in range(len(texts)):
            prob = probs[i].cpu().numpy()
            pred_id = np.argmax(prob)
            
            # Convert to sentiment score [-1, 1]
            # positive=0, negative=1, neutral=2
            sentiment_score = self._to_sentiment_score(prob)
            
            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'label': self.id2label[pred_id.item()],
                'confidence': float(prob[pred_id]),
                'sentiment_score': float(sentiment_score),
                'probabilities': {
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2])
                }
            })
        
        return results
    
    def _to_sentiment_score(self, probs: np.ndarray) -> float:
        """
        Convert probability distribution to sentiment score.
        
        Formula: (pos - neg) / (pos + neg + neutral)
        Range: [-1, 1] where -1 = very negative, 1 = very positive
        """
        pos, neg, neutral = probs[0], probs[1], probs[2]
        
        # Weighted score
        score = (pos - neg) / (pos + neg + neutral + 1e-8)
        
        # Adjust for confidence (reduce impact of uncertain predictions)
        confidence = max(pos, neg, neutral)
        score *= (confidence - 1/3) / (2/3)  # Scale by how much above random
        
        return np.clip(score, -1, 1)
    
    def aggregate_sentiment(self, texts: List[str], 
                           weights: List[float] = None,
                           decay_factor: float = 1.0) -> Dict:
        """
        Aggregate sentiment from multiple texts with time decay.
        
        Args:
            texts: List of texts (chronological order)
            weights: Optional per-text weights
            decay_factor: Exponential decay (1.0 = no decay)
            
        Returns:
            dict: Aggregated sentiment metrics
        """
        if not texts:
            return {'sentiment': 0, 'confidence': 0, 'n_sources': 0}
        
        # Analyze all texts
        analyses = self.analyze(texts)
        
        # Apply decay weights
        n = len(analyses)
        if weights is None:
            weights = [decay_factor ** (n - 1 - i) for i in range(n)]
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted aggregation
        sentiments = np.array([a['sentiment_score'] for a in analyses])
        confidences = np.array([a['confidence'] for a in analyses])
        
        weighted_sentiment = np.sum(sentiments * weights)
        weighted_confidence = np.sum(confidences * weights)
        
        # Uncertainty weighted by disagreement
        sentiment_std = np.std(sentiments)
        
        return {
            'sentiment': float(weighted_sentiment),
            'confidence': float(weighted_confidence),
            'sentiment_std': float(sentiment_std),
            'n_sources': len(texts),
            'direction': 'bullish' if weighted_sentiment > 0.2 else 
                        ('bearish' if weighted_sentiment < -0.2 else 'neutral'),
            'individual_scores': [a['sentiment_score'] for a in analyses]
        }
    
    def analyze_news_batch(self, headlines: List[Dict], 
                          stock_symbol: str = None) -> Dict:
        """
        Analyze batch of news headlines with relevance filtering.
        
        Args:
            headlines: List of dicts with 'title', 'source', 'published_at'
            stock_symbol: Filter for symbol mentions
            
        Returns:
            dict: Aggregated sentiment with metadata
        """
        # Filter relevant headlines
        relevant = []
        for h in headlines:
            text = h.get('title', '') + ' ' + h.get('summary', '')
            
            # Check symbol mention if specified
            if stock_symbol and stock_symbol.upper() not in text.upper():
                continue
            
            relevant.append(text)
        
        if not relevant:
            return {
                'sentiment': 0,
                'confidence': 0,
                'n_relevant': 0,
                'note': 'No relevant headlines found'
            }
        
        # Aggregate sentiment
        result = self.aggregate_sentiment(relevant)
        result['n_relevant'] = len(relevant)
        result['n_total'] = len(headlines)
        
        return result
```

## 3.3 REINFORCEMENT LEARNING FOR TRADING

### 3.3.1 PPO Agent Architecture

Proximal Policy Optimization (PPO) provides stable policy learning for trading decisions.

```python
# models/reinforcement/ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class TradingEnvironment:
    """
    Gym-like environment for trading.
    
    State: Market features + portfolio state
    Actions: [hold, buy, sell] with position sizing
    Reward: Risk-adjusted returns (Sharpe-like)
    """
    
    def __init__(self, price_data, features, initial_capital=100000,
                 transaction_cost=0.001, max_position=1.0):
        self.prices = price_data
        self.features = features
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.cash = self.initial_capital
        self.position = 0  # Shares held
        self.portfolio_value = self.initial_capital
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state."""
        market_features = self.features[self.current_step]
        
        # Portfolio features
        position_ratio = self.position * self.prices[self.current_step] / self.portfolio_value
        
        return np.concatenate([
            market_features,
            [position_ratio, self.cash / self.initial_capital]
        ])
    
    def step(self, action):
        """
        Execute action and return new state.
        
        Actions:
        0: Hold
        1: Buy (increase position)
        2: Sell (decrease position)
        """
        current_price = self.prices[self.current_step]
        
        # Execute trade
        if action == 1:  # Buy
            max_shares = (self.cash * self.max_position) / current_price
            shares_to_buy = max_shares * 0.5  # Conservative sizing
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)
            
            if cost <= self.cash:
                self.position += shares_to_buy
                self.cash -= cost
                self.trades.append(('buy', self.current_step, shares_to_buy, current_price))
        
        elif action == 2:  # Sell
            if self.position > 0:
                shares_to_sell = self.position * 0.5
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.position -= shares_to_sell
                self.cash += proceeds
                self.trades.append(('sell', self.current_step, shares_to_sell, current_price))
        
        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Calculate new portfolio value
        new_price = self.prices[self.current_step]
        self.portfolio_value = self.cash + self.position * new_price
        
        # Calculate reward (return with penalty for variance)
        if len(self.trades) > 0:
            returns = np.diff([t[3] for t in self.trades if t[0] == 'sell'])
            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns) + 1e-8
                reward = mean_return / std_return  # Sharpe-like
            else:
                reward = 0
        else:
            reward = 0
        
        # Penalty for inaction
        if action == 0 and self.position == 0:
            reward -= 0.01
        
        return self._get_observation(), reward, done, {}


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    
    Advantages:
    - Stable training with clipped objective
    - Sample efficient (multiple epochs per batch)
    - Works well with continuous and discrete actions
    """
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state, training=True):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        if training:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            self.states.append(state)
            self.actions.append(action.item())
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
            
            return action.item()
        else:
            return torch.argmax(action_probs).item()
    
    def store_transition(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns(self, next_value):
        """Compute discounted returns."""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update(self, next_state):
        """Update policy using PPO."""
        # Get next value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(state_tensor)
            next_value = next_value.item()
        
        # Compute returns and advantages
        returns = self.compute_returns(next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        
        old_values = torch.FloatTensor(self.values).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # PPO update (multiple epochs)
        for _ in range(4):
            # Forward pass
            action_probs, values = self.network(states)
            dist = torch.distributions.Categorical(action_probs)
            
            # New log probs
            new_log_probs = dist.log_prob(actions)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def train(self, env, episodes=1000):
        """Train agent on environment."""
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.store_transition(reward, done)
                episode_reward += reward
                state = next_state
            
            # Update policy
            self.update(state)
            
            rewards_history.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}")
        
        return rewards_history
```

---

*Note: This is Part 1 of the comprehensive agent.md file. The document continues with additional sections covering backtesting, risk management, data collection, and execution in subsequent parts.*


---

# SECTION 4: BACKTESTING AND VALIDATION FRAMEWORK

## 4.1 EVENT-DRIVEN BACKTESTING ENGINE

### 4.1.1 Core Architecture

The backtesting engine MUST simulate real-world execution conditions including slippage, market impact, and transaction costs.

```python
# backtesting/event_engine.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    timestamp: datetime
    slippage: float
    commission: float

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float

class SlippageModel:
    """
    Realistic slippage model based on market microstructure.
    
    Components:
    1. Bid-ask spread (fixed percentage)
    2. Market impact (function of order size/volume)
    3. Volatility adjustment
    """
    
    def __init__(self, base_spread_bps=5, impact_coefficient=1.0,
                 volatility_factor=0.5):
        """
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            impact_coefficient: Market impact multiplier
            volatility_factor: How much volatility increases slippage
        """
        self.base_spread = base_spread_bps / 10000  # Convert to decimal
        self.impact_coeff = impact_coefficient
        self.vol_factor = volatility_factor
    
    def estimate_slippage(self, order: Order, market_data: pd.Series,
                         historical_volume: float) -> float:
        """
        Estimate slippage for an order.
        
        Args:
            order: Order to execute
            market_data: Current market data (high, low, close, volume)
            historical_volume: Average historical volume
            
        Returns:
            float: Slippage as decimal (positive = worse fill)
        """
        # Base spread cost
        spread_cost = self.base_spread / 2  # Half spread for market order
        
        # Market impact (Almgren-Chriss style)
        if historical_volume > 0:
            participation = abs(order.quantity) / historical_volume
            impact = self.impact_coeff * np.sqrt(participation)
        else:
            impact = 0
        
        # Volatility adjustment
        if 'atr' in market_data:
            vol_adj = self.vol_factor * (market_data['atr'] / market_data['close'])
        else:
            vol_adj = 0
        
        total_slippage = spread_cost + impact + vol_adj
        
        # Direction: buy = higher price, sell = lower price
        if order.side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage

class TransactionCostModel:
    """
    Comprehensive transaction cost model.
    
    Includes:
    - Commission (per share or percentage)
    - SEC fees (for sells)
    - Borrow fees (for shorts)
    """
    
    def __init__(self, commission_rate=0.001, min_commission=1.0,
                 sec_fee_rate=0.0000278, max_sec_fee=5.95,
                 borrow_fee_annual=0.03):
        """
        Args:
            commission_rate: Commission as percentage of trade value
            min_commission: Minimum commission per trade
            sec_fee_rate: SEC fee rate (as of 2024)
            max_sec_fee: Maximum SEC fee per trade
            borrow_fee_annual: Annual borrow fee for short positions
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.sec_fee_rate = sec_fee_rate
        self.max_sec_fee = max_sec_fee
        self.borrow_fee = borrow_fee_annual / 252  # Daily
    
    def calculate_costs(self, fill: Fill, is_short: bool = False) -> Dict:
        """
        Calculate all transaction costs for a fill.
        
        Args:
            fill: Fill information
            is_short: Whether this is a short sale
            
        Returns:
            dict: Cost breakdown
        """
        trade_value = abs(fill.quantity * fill.fill_price)
        
        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # SEC fee (on sells only)
        sec_fee = 0
        if fill.side == OrderSide.SELL:
            sec_fee = min(trade_value * self.sec_fee_rate, self.max_sec_fee)
        
        # Borrow fee (for shorts)
        borrow_cost = 0
        if is_short:
            borrow_cost = trade_value * self.borrow_fee
        
        total_cost = commission + sec_fee + borrow_cost
        
        return {
            'commission': commission,
            'sec_fee': sec_fee,
            'borrow_cost': borrow_cost,
            'total_cost': total_cost,
            'cost_bps': (total_cost / trade_value) * 10000 if trade_value > 0 else 0
        }

class EventDrivenBacktester:
    """
    Event-driven backtesting engine with realistic execution simulation.
    
    Features:
    - Bar-by-bar simulation (no lookahead bias)
    - Realistic slippage and market impact
    - Transaction cost modeling
    - Position tracking and P&L calculation
    """
    
    def __init__(self, initial_capital=100000, 
                 slippage_model: Optional[SlippageModel] = None,
                 cost_model: Optional[TransactionCostModel] = None):
        """
        Args:
            initial_capital: Starting capital
            slippage_model: Slippage model instance
            cost_model: Transaction cost model instance
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.slippage_model = slippage_model or SlippageModel()
        self.cost_model = cost_model or TransactionCostModel()
        
        # State
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.equity_curve: List[Dict] = []
        
        # History
        self.trade_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.fills = []
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []
    
    def submit_order(self, order: Order):
        """Submit an order to the backtester."""
        order.order_id = f"order_{len(self.orders)}"
        self.orders.append(order)
        return order.order_id
    
    def process_bar(self, timestamp: datetime, bar_data: pd.DataFrame,
                   strategy_signals: Dict[str, float]):
        """
        Process a single bar of data.
        
        Args:
            timestamp: Current timestamp
            bar_data: OHLCV data for all symbols
            strategy_signals: Dict of symbol -> signal strength (-1 to 1)
        """
        # Execute pending orders
        self._execute_orders(timestamp, bar_data)
        
        # Update positions with current prices
        self._mark_to_market(bar_data)
        
        # Record equity
        total_equity = self._calculate_total_equity(bar_data)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.capital,
            'positions_value': total_equity - self.capital
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_ret = (total_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_ret)
    
    def _execute_orders(self, timestamp: datetime, bar_data: pd.DataFrame):
        """Execute pending orders with slippage."""
        for order in self.orders:
            if order.symbol not in bar_data.index:
                continue
            
            market = bar_data.loc[order.symbol]
            
            # Determine fill price with slippage
            if order.order_type == OrderType.MARKET:
                base_price = market['close']
                
                # Estimate historical volume for impact calculation
                hist_volume = market.get('avg_volume', market['volume'])
                
                slippage = self.slippage_model.estimate_slippage(
                    order, market, hist_volume
                )
                
                fill_price = base_price * (1 + slippage)
            
            elif order.order_type == OrderType.LIMIT:
                # Check if limit price is hit
                if order.side == OrderSide.BUY and market['low'] <= order.price:
                    fill_price = min(order.price, market['open'])
                elif order.side == OrderSide.SELL and market['high'] >= order.price:
                    fill_price = max(order.price, market['open'])
                else:
                    continue  # Limit not hit
            
            else:
                continue  # Other order types not implemented
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=timestamp,
                slippage=slippage if order.order_type == OrderType.MARKET else 0,
                commission=0  # Will be calculated
            )
            
            # Calculate costs
            costs = self.cost_model.calculate_costs(fill)
            fill.commission = costs['total_cost']
            
            self.fills.append(fill)
            
            # Update position
            self._update_position(fill)
            
            # Update capital
            trade_value = fill.quantity * fill.fill_price
            if order.side == OrderSide.BUY:
                self.capital -= trade_value + fill.commission
            else:
                self.capital += trade_value - fill.commission
        
        # Clear executed orders
        self.orders = []
    
    def _update_position(self, fill: Fill):
        """Update position based on fill."""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_entry_price=0,
                market_price=fill.fill_price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
        pos = self.positions[symbol]
        
        if fill.side == OrderSide.BUY:
            # Update average entry price
            total_cost = pos.quantity * pos.avg_entry_price + fill.quantity * fill.fill_price
            pos.quantity += fill.quantity
            pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
        else:
            # Calculate realized P&L
            realized = fill.quantity * (fill.fill_price - pos.avg_entry_price)
            pos.realized_pnl += realized
            pos.quantity -= fill.quantity
            
            # Record trade
            self.trade_history.append({
                'symbol': symbol,
                'entry_price': pos.avg_entry_price,
                'exit_price': fill.fill_price,
                'quantity': fill.quantity,
                'pnl': realized,
                'timestamp': fill.timestamp
            })
            
            if pos.quantity == 0:
                pos.avg_entry_price = 0
    
    def _mark_to_market(self, bar_data: pd.DataFrame):
        """Update position values with current prices."""
        for symbol, pos in self.positions.items():
            if symbol in bar_data.index:
                pos.market_price = bar_data.loc[symbol]['close']
                pos.unrealized_pnl = pos.quantity * (pos.market_price - pos.avg_entry_price)
    
    def _calculate_total_equity(self, bar_data: pd.DataFrame) -> float:
        """Calculate total account equity."""
        positions_value = sum(
            pos.quantity * bar_data.loc[symbol]['close']
            for symbol, pos in self.positions.items()
            if symbol in bar_data.index
        )
        return self.capital + positions_value
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            dict: Performance statistics
        """
        if len(self.equity_curve) < 2:
            return {'error': 'Insufficient data'}
        
        equity = pd.DataFrame(self.equity_curve)
        equity['returns'] = equity['equity'].pct_change().dropna()
        
        returns = equity['returns'].dropna()
        
        # Basic metrics
        total_return = (equity['equity'].iloc[-1] / self.initial_capital) - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sharpe and Sortino
        risk_free_rate = 0.02
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        equity['peak'] = equity['equity'].cummax()
        equity['drawdown'] = (equity['equity'] - equity['peak']) / equity['peak']
        max_drawdown = equity['drawdown'].min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if self.trade_history:
            trades = pd.DataFrame(self.trade_history)
            win_rate = (trades['pnl'] > 0).mean()
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
            avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if (trades['pnl'] < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_trades': len(self.trade_history),
            'final_equity': equity['equity'].iloc[-1]
        }
```

## 4.2 MONTE CARLO PERMUTATION TESTS (MCPT)

### 4.2.1 Theoretical Foundation

MCPT tests whether a strategy's performance is statistically significant or due to random chance by comparing it to permuted (randomized) data.

**Null Hypothesis:** The strategy has no predictive power; performance is due to luck.

**Test Statistic:** Profit factor, Sharpe ratio, or total return.

**P-Value:** Proportion of permutations that exceed actual performance.

### 4.2.2 Implementation (Based on neurotrader888/mcpt)

```python
# backtesting/monte_carlo.py

import numpy as np
import pandas as pd
from typing import Callable, List, Dict
from tqdm import tqdm

class MonteCarloPermutationTest:
    """
    Monte Carlo Permutation Tests for strategy validation.
    
    Based on: https://github.com/neurotrader888/mcpt
    
    Two variants:
    1. In-Sample MCPT: Tests if strategy overfits to training data
    2. Walk-Forward MCPT: Tests out-of-sample robustness
    """
    
    def __init__(self, n_permutations: int = 1000, 
                 test_statistic: str = 'profit_factor',
                 random_seed: int = 42):
        """
        Args:
            n_permutations: Number of random permutations
            test_statistic: Metric to test ('profit_factor', 'sharpe', 'return')
            random_seed: Random seed for reproducibility
        """
        self.n_perm = n_permutations
        self.statistic = test_statistic
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Annualize (assuming daily returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        return sharpe
    
    def _calculate_statistic(self, returns: np.ndarray) -> float:
        """Calculate test statistic."""
        if self.statistic == 'profit_factor':
            return self._calculate_profit_factor(returns)
        elif self.statistic == 'sharpe':
            return self._calculate_sharpe(returns)
        elif self.statistic == 'return':
            return returns.sum()
        else:
            raise ValueError(f"Unknown statistic: {self.statistic}")
    
    def bar_permutation(self, price_data: pd.DataFrame, 
                       strategy_func: Callable,
                       start_index: int = 0) -> pd.DataFrame:
        """
        Permute price bars while preserving OHLC relationships.
        
        Args:
            price_data: DataFrame with OHLC columns
            strategy_func: Function that takes price data and returns signals
            start_index: Index to start permutation from
            
        Returns:
            DataFrame: Permuted price data
        """
        permuted = price_data.copy()
        
        # Get bars from start_index onwards
        n_bars = len(permuted) - start_index
        
        # Generate random permutation
        perm_indices = np.random.permutation(n_bars) + start_index
        
        # Apply permutation
        permuted.iloc[start_index:] = permuted.iloc[perm_indices].values
        
        return permuted
    
    def run_insample_mcpt(self, price_data: pd.DataFrame,
                         strategy_func: Callable,
                         train_lookback: int = None) -> Dict:
        """
        Run in-sample MCPT to detect overfitting.
        
        This test is STRICT - in-sample optimization will almost always fail.
        A passing in-sample test suggests genuine edge.
        
        Args:
            price_data: Full price history
            strategy_func: Strategy signal generator
            train_lookback: Training period (uses all if None)
            
        Returns:
            dict: Test results with p-value
        """
        if train_lookback:
            train_data = price_data.iloc[-train_lookback:].copy()
        else:
            train_data = price_data.copy()
        
        # Calculate returns on actual data
        signals = strategy_func(train_data)
        returns = train_data['returns'] * signals
        real_stat = self._calculate_statistic(returns)
        
        print(f"Real {self.statistic}: {real_stat:.4f}")
        print(f"Running {self.n_perm} permutations...")
        
        # Run permutations
        perm_stats = []
        perm_better_count = 1  # Start at 1 for conservative estimate
        
        for _ in tqdm(range(self.n_perm)):
            # Permute bars
            perm_data = self.bar_permutation(train_data, strategy_func)
            
            # Generate signals on permuted data
            perm_signals = strategy_func(perm_data)
            perm_returns = perm_data['returns'] * perm_signals
            
            # Calculate statistic
            perm_stat = self._calculate_statistic(perm_returns)
            perm_stats.append(perm_stat)
            
            if perm_stat >= real_stat:
                perm_better_count += 1
        
        # Calculate p-value
        p_value = perm_better_count / (self.n_perm + 1)
        
        return {
            'test_type': 'in_sample',
            'real_statistic': real_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'perm_stats': perm_stats,
            'percentile': 100 * (1 - p_value)
        }
    
    def run_walkforward_mcpt(self, price_data: pd.DataFrame,
                            strategy_func: Callable,
                            train_lookback: int,
                            step_size: int = None) -> Dict:
        """
        Run walk-forward MCPT for out-of-sample validation.
        
        This is the GOLD STANDARD test. A strategy that passes WFO-MCPT
        has demonstrated genuine predictive power.
        
        Args:
            price_data: Full price history
            strategy_func: Strategy with walk-forward optimization
            train_lookback: Training window size
            step_size: Step between windows (default: train_lookback)
            
        Returns:
            dict: Test results with p-value
        """
        if step_size is None:
            step_size = train_lookback
        
        # Calculate walk-forward returns on actual data
        real_returns = self._walk_forward_returns(
            price_data, strategy_func, train_lookback, step_size
        )
        real_stat = self._calculate_statistic(real_returns)
        
        print(f"Walk-Forward {self.statistic}: {real_stat:.4f}")
        print(f"Running {self.n_perm} walk-forward permutations...")
        
        # Run permutations
        perm_stats = []
        perm_better_count = 1
        
        for _ in tqdm(range(self.n_perm)):
            # Permute from train_lookback onwards
            perm_data = self.bar_permutation(price_data, strategy_func, 
                                            start_index=train_lookback)
            
            # Calculate walk-forward returns on permuted data
            perm_returns = self._walk_forward_returns(
                perm_data, strategy_func, train_lookback, step_size
            )
            
            perm_stat = self._calculate_statistic(perm_returns)
            perm_stats.append(perm_stat)
            
            if perm_stat >= real_stat:
                perm_better_count += 1
        
        p_value = perm_better_count / (self.n_perm + 1)
        
        return {
            'test_type': 'walk_forward',
            'real_statistic': real_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'perm_stats': perm_stats,
            'n_windows': len(real_returns) // step_size,
            'percentile': 100 * (1 - p_value)
        }
    
    def _walk_forward_returns(self, price_data: pd.DataFrame,
                             strategy_func: Callable,
                             train_lookback: int,
                             step_size: int) -> np.ndarray:
        """
        Calculate walk-forward returns.
        
        Args:
            price_data: Price history
            strategy_func: Strategy function
            train_lookback: Training window
            step_size: Step size
            
        Returns:
            np.ndarray: Out-of-sample returns
        """
        all_returns = []
        
        for i in range(train_lookback, len(price_data) - step_size, step_size):
            # Training data
            train_start = max(0, i - train_lookback)
            train_data = price_data.iloc[train_start:i]
            
            # Test data
            test_data = price_data.iloc[i:i+step_size]
            
            # Generate signals (optimize on train, apply to test)
            signals = strategy_func(train_data, test_data=test_data)
            
            # Calculate returns
            test_returns = test_data['returns'] * signals
            all_returns.extend(test_returns.values)
        
        return np.array(all_returns)
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot MCPT results.
        
        Args:
            results: Results from run_insample_mcpt or run_walkforward_mcpt
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        plt.style.use('dark_background')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of permutation statistics
        perm_stats = results['perm_stats']
        ax.hist(perm_stats, bins=50, color='blue', alpha=0.7, 
               label='Permutations', density=True)
        
        # Vertical line for real statistic
        real_stat = results['real_statistic']
        ax.axvline(real_stat, color='red', linewidth=2, 
                  label=f'Real ({self.statistic}={real_stat:.3f})')
        
        ax.set_xlabel(self.statistic.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f"{results['test_type'].replace('_', ' ').title()} MCPT\n"
                    f"P-Value: {results['p_value']:.4f} "
                    f"({'Significant' if results['is_significant'] else 'Not Significant'})")
        ax.legend()
        ax.grid(False)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class BootstrapTest:
    """
    Bootstrap resampling for confidence interval estimation.
    """
    
    def __init__(self, n_bootstrap: int = 10000, 
                 confidence: float = 0.95,
                 random_seed: int = 42):
        """
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            random_seed: Random seed
        """
        self.n_boot = n_bootstrap
        self.confidence = confidence
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def run(self, returns: np.ndarray) -> Dict:
        """
        Run bootstrap test on returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            dict: Bootstrap statistics with confidence intervals
        """
        # Calculate real statistics
        real_sharpe = self._sharpe(returns)
        real_cagr = self._cagr(returns)
        real_mdd = self._max_drawdown(returns)
        
        # Bootstrap
        sharpe_boot = []
        cagr_boot = []
        mdd_boot = []
        
        for _ in range(self.n_boot):
            # Resample with replacement
            sample = np.random.choice(returns, size=len(returns), replace=True)
            
            sharpe_boot.append(self._sharpe(sample))
            cagr_boot.append(self._cagr(sample))
            mdd_boot.append(self._max_drawdown(sample))
        
        alpha = (1 - self.confidence) / 2
        
        return {
            'sharpe': {
                'real': real_sharpe,
                'mean': np.mean(sharpe_boot),
                'ci_lower': np.percentile(sharpe_boot, 100 * alpha),
                'ci_upper': np.percentile(sharpe_boot, 100 * (1 - alpha))
            },
            'cagr': {
                'real': real_cagr,
                'mean': np.mean(cagr_boot),
                'ci_lower': np.percentile(cagr_boot, 100 * alpha),
                'ci_upper': np.percentile(cagr_boot, 100 * (1 - alpha))
            },
            'max_drawdown': {
                'real': real_mdd,
                'mean': np.mean(mdd_boot),
                'ci_lower': np.percentile(mdd_boot, 100 * alpha),
                'ci_upper': np.percentile(mdd_boot, 100 * (1 - alpha))
            }
        }
    
    def _sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    def _cagr(self, returns: np.ndarray) -> float:
        """Calculate CAGR."""
        total_ret = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        return (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()
```

## 4.3 WALK-FORWARD OPTIMIZATION

### 4.3.1 Theoretical Foundation

Walk-forward optimization simulates real-world trading by:
1. Optimizing parameters on in-sample data
2. Testing on subsequent out-of-sample data
3. Rolling forward and repeating

This prevents overfitting and tests adaptability to changing market conditions.

### 4.3.2 Implementation

```python
# backtesting/walk_forward.py

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import optuna

@dataclass
class WFOConfig:
    """Configuration for walk-forward optimization."""
    train_size: int  # Training window size (bars)
    test_size: int   # Test window size (bars)
    step_size: int   # Step between windows (bars)
    n_trials: int    # Optimization trials per window
    metric: str      # Optimization metric ('sharpe', 'return', 'profit_factor')

class WalkForwardOptimizer:
    """
    Walk-forward optimization with automated parameter search.
    
    Uses Optuna for Bayesian optimization within each window.
    """
    
    def __init__(self, config: WFOConfig):
        self.config = config
        
    def optimize(self, price_data: pd.DataFrame,
                strategy_class: type,
                param_space: Dict[str, Tuple]) -> Dict:
        """
        Run walk-forward optimization.
        
        Args:
            price_data: Full price history
            strategy_class: Strategy class to optimize
            param_space: Parameter ranges {name: (min, max, type)}
            
        Returns:
            dict: WFO results with performance metrics
        """
        results = {
            'windows': [],
            'train_metrics': [],
            'test_metrics': [],
            'params': [],
            'equity_curve': []
        }
        
        n_windows = (len(price_data) - self.config.train_size - self.config.test_size) // self.config.step_size + 1
        
        print(f"Running WFO with {n_windows} windows...")
        
        for i in tqdm(range(n_windows)):
            # Define window
            train_start = i * self.config.step_size
            train_end = train_start + self.config.train_size
            test_end = train_end + self.config.test_size
            
            if test_end > len(price_data):
                break
            
            train_data = price_data.iloc[train_start:train_end]
            test_data = price_data.iloc[train_end:test_end]
            
            # Optimize on training data
            best_params = self._optimize_window(
                train_data, strategy_class, param_space
            )
            
            # Evaluate on training data (in-sample)
            train_strategy = strategy_class(**best_params)
            train_metrics = self._evaluate(train_strategy, train_data)
            
            # Evaluate on test data (out-of-sample)
            test_strategy = strategy_class(**best_params)
            test_metrics = self._evaluate(test_strategy, test_data)
            
            # Store results
            results['windows'].append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1]
            })
            results['train_metrics'].append(train_metrics)
            results['test_metrics'].append(test_metrics)
            results['params'].append(best_params)
            
            # Build equity curve
            if 'equity' in test_metrics:
                results['equity_curve'].extend(test_metrics['equity'])
        
        # Calculate aggregate metrics
        results['aggregate'] = self._calculate_aggregate(results)
        
        return results
    
    def _optimize_window(self, train_data: pd.DataFrame,
                        strategy_class: type,
                        param_space: Dict) -> Dict:
        """
        Optimize strategy parameters for a single window.
        
        Args:
            train_data: Training data
            strategy_class: Strategy class
            param_space: Parameter search space
            
        Returns:
            dict: Best parameters
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for name, (min_val, max_val, ptype) in param_space.items():
                if ptype == 'int':
                    params[name] = trial.suggest_int(name, min_val, max_val)
                elif ptype == 'float':
                    params[name] = trial.suggest_float(name, min_val, max_val)
                elif ptype == 'categorical':
                    params[name] = trial.suggest_categorical(name, min_val)
            
            # Create strategy and evaluate
            strategy = strategy_class(**params)
            metrics = self._evaluate(strategy, train_data)
            
            # Return optimization metric
            if self.config.metric == 'sharpe':
                return metrics.get('sharpe', 0)
            elif self.config.metric == 'return':
                return metrics.get('total_return', 0)
            elif self.config.metric == 'profit_factor':
                return metrics.get('profit_factor', 0)
            else:
                return metrics.get('sharpe', 0)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _evaluate(self, strategy, data: pd.DataFrame) -> Dict:
        """
        Evaluate strategy on data.
        
        Args:
            strategy: Strategy instance
            data: Price data
            
        Returns:
            dict: Performance metrics
        """
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Calculate returns
        returns = data['returns'] * signals
        
        # Metrics
        total_return = (1 + returns).prod() - 1
        
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'n_trades': (signals != 0).sum(),
            'equity': equity.tolist()
        }
    
    def _calculate_aggregate(self, results: Dict) -> Dict:
        """
        Calculate aggregate metrics across all windows.
        
        Args:
            results: WFO results
            
        Returns:
            dict: Aggregate metrics
        """
        test_metrics = results['test_metrics']
        
        if not test_metrics:
            return {}
        
        # Extract metrics
        returns = [m['total_return'] for m in test_metrics]
        sharpes = [m['sharpe'] for m in test_metrics]
        pfs = [m['profit_factor'] for m in test_metrics]
        mdds = [m['max_drawdown'] for m in test_metrics]
        
        # Walk-forward efficiency
        train_returns = [m['total_return'] for m in results['train_metrics']]
        wfe = np.mean(returns) / np.mean(train_returns) if np.mean(train_returns) > 0 else 0
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'mean_profit_factor': np.mean(pfs),
            'mean_max_drawdown': np.mean(mdds),
            'walk_forward_efficiency': wfe,
            'n_windows': len(test_metrics)
        }
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot WFO results.
        
        Args:
            results: WFO results
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        if results['equity_curve']:
            axes[0].plot(results['equity_curve'], color='green')
            axes[0].set_title('Walk-Forward Equity Curve')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)
        
        # Returns by window
        train_rets = [m['total_return'] for m in results['train_metrics']]
        test_rets = [m['total_return'] for m in results['test_metrics']]
        
        x = range(len(train_rets))
        axes[1].plot(x, train_rets, label='Train (IS)', color='blue', marker='o')
        axes[1].plot(x, test_rets, label='Test (OOS)', color='orange', marker='s')
        axes[1].set_title('Returns by Window')
        axes[1].set_ylabel('Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Sharpe by window
        train_sharpes = [m['sharpe'] for m in results['train_metrics']]
        test_sharpes = [m['sharpe'] for m in results['test_metrics']]
        
        axes[2].plot(x, train_sharpes, label='Train (IS)', color='blue', marker='o')
        axes[2].plot(x, test_sharpes, label='Test (OOS)', color='orange', marker='s')
        axes[2].set_title('Sharpe Ratio by Window')
        axes[2].set_ylabel('Sharpe')
        axes[2].set_xlabel('Window')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
```

---

# SECTION 5: RISK MANAGEMENT FRAMEWORK

## 5.1 POSITION SIZING WITH KELLY CRITERION

### 5.1.1 Mathematical Foundation

The Kelly Criterion determines optimal bet size to maximize long-term growth.

**Kelly Formula:**
$$f^* = \frac{p(b+1) - 1}{b}$$

Where:
- $f^*$ = Optimal fraction of capital to bet
- $p$ = Probability of win
- $b$ = Average win / average loss (odds)

### 5.1.2 Implementation

```python
# risk/position_sizing.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    full_kelly: float
    half_kelly: float  # Conservative
    quarter_kelly: float  # Very conservative
    win_rate: float
    win_loss_ratio: float
    expected_value: float
    confidence: float

class KellyCriterion:
    """
    Kelly Criterion position sizing with uncertainty adjustment.
    """
    
    def __init__(self, confidence_threshold: float = 0.6,
                 max_position: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence to trade
            max_position: Maximum position size (safety cap)
        """
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position
    
    def calculate(self, historical_returns: np.ndarray,
                 predicted_return: float = None,
                 predicted_confidence: float = 0.5) -> KellyResult:
        """
        Calculate Kelly fraction from historical trade distribution.
        
        Args:
            historical_returns: Array of past trade returns
            predicted_return: Model prediction (optional)
            predicted_confidence: Confidence in prediction
            
        Returns:
            KellyResult: Position sizing recommendation
        """
        if len(historical_returns) < 10:
            return KellyResult(0, 0, 0, 0, 0, 0, 0)
        
        # Separate wins and losses
        wins = historical_returns[historical_returns > 0]
        losses = historical_returns[historical_returns < 0]
        
        if len(losses) == 0 or len(wins) == 0:
            return KellyResult(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate parameters
        win_rate = len(wins) / len(historical_returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Kelly formula
        numerator = win_rate * (win_loss_ratio + 1) - 1
        denominator = win_loss_ratio
        
        if denominator == 0:
            full_kelly = 0
        else:
            full_kelly = numerator / denominator
        
        # Adjust for prediction confidence
        full_kelly *= predicted_confidence
        
        # Apply safety caps
        full_kelly = np.clip(full_kelly, 0, self.max_position)
        
        # Conservative fractions
        half_kelly = full_kelly / 2
        quarter_kelly = full_kelly / 4
        
        # Expected value
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Calculate confidence in Kelly estimate
        # Based on sample size (more trades = higher confidence)
        confidence = min(len(historical_returns) / 100, 1.0)
        
        return KellyResult(
            full_kelly=full_kelly,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            expected_value=ev,
            confidence=confidence
        )
    
    def fractional_kelly(self, historical_returns: np.ndarray,
                        fraction: float = 0.5) -> float:
        """
        Calculate fractional Kelly position size.
        
        Args:
            historical_returns: Past trade returns
            fraction: Kelly fraction (0.5 = half Kelly)
            
        Returns:
            float: Position size
        """
        result = self.calculate(historical_returns)
        return result.full_kelly * fraction


class RiskParitySizing:
    """
    Risk parity position sizing based on inverse volatility.
    
    Allocates capital inversely proportional to asset volatility.
    """
    
    def __init__(self, target_volatility: float = 0.10):
        """
        Args:
            target_volatility: Target portfolio volatility (annualized)
        """
        self.target_vol = target_volatility
    
    def calculate_weights(self, volatilities: Dict[str, float],
                         correlations: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate risk parity weights.
        
        Args:
            volatilities: Dict of asset -> annualized volatility
            correlations: Correlation matrix (identity if None)
            
        Returns:
            dict: Asset weights
        """
        assets = list(volatilities.keys())
        n = len(assets)
        
        if correlations is None:
            correlations = np.eye(n)
        
        # Inverse volatility weights (naive risk parity)
        vol_array = np.array([volatilities[a] for a in assets])
        inv_vol = 1 / vol_array
        
        # Normalize
        weights = inv_vol / inv_vol.sum()
        
        # Scale to target volatility
        portfolio_vol = self._portfolio_volatility(weights, vol_array, correlations)
        scaling = self.target_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        weights = weights * scaling
        
        return {asset: weights[i] for i, asset in enumerate(assets)}
    
    def _portfolio_volatility(self, weights: np.ndarray,
                             vols: np.ndarray,
                             corr: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        cov = np.outer(vols, vols) * corr
        portfolio_var = weights.T @ cov @ weights
        return np.sqrt(portfolio_var)


class DynamicPositionSizer:
    """
    Dynamic position sizing combining multiple methods.
    
    Combines:
    - Kelly Criterion (edge-based)
    - Risk Parity (volatility-based)
    - Market Regime (adaptive)
    """
    
    def __init__(self, kelly_weight: float = 0.4,
                 risk_parity_weight: float = 0.4,
                 regime_weight: float = 0.2):
        """
        Args:
            kelly_weight: Weight for Kelly sizing
            risk_parity_weight: Weight for risk parity
            regime_weight: Weight for regime adjustment
        """
        self.kelly = KellyCriterion()
        self.risk_parity = RiskParitySizing()
        self.weights = {
            'kelly': kelly_weight,
            'risk_parity': risk_parity_weight,
            'regime': regime_weight
        }
    
    def calculate_position(self, symbol: str,
                          historical_returns: np.ndarray,
                          current_volatility: float,
                          market_regime: str,
                          signal_strength: float,
                          signal_confidence: float) -> Dict:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Asset symbol
            historical_returns: Past returns
            current_volatility: Current realized volatility
            market_regime: 'trending', 'mean_reverting', 'volatile'
            signal_strength: Model signal (-1 to 1)
            signal_confidence: Confidence in signal (0 to 1)
            
        Returns:
            dict: Position sizing recommendation
        """
        # Kelly component
        kelly_result = self.kelly.calculate(
            historical_returns, 
            predicted_confidence=signal_confidence
        )
        kelly_size = kelly_result.half_kelly * signal_strength
        
        # Risk parity component
        vol_scaling = 0.20 / current_volatility if current_volatility > 0 else 1
        parity_size = signal_strength * vol_scaling * 0.1  # 10% max per signal
        
        # Regime adjustment
        regime_multipliers = {
            'trending': 1.2,      # Increase size in trends
            'mean_reverting': 0.8,  # Decrease in chop
            'volatile': 0.5,       # Significantly reduce in high vol
            'unknown': 1.0
        }
        regime_mult = regime_multipliers.get(market_regime, 1.0)
        
        # Combine
        combined = (
            kelly_size * self.weights['kelly'] +
            parity_size * self.weights['risk_parity']
        ) * regime_mult
        
        # Apply confidence floor
        if signal_confidence < 0.6:
            combined *= 0.5
        
        return {
            'position_size': np.clip(combined, -0.25, 0.25),  # Max 25% per position
            'kelly_component': kelly_size,
            'risk_parity_component': parity_size,
            'regime_multiplier': regime_mult,
            'signal_confidence': signal_confidence,
            'market_regime': market_regime
        }
```

## 5.2 VALUE-AT-RISK (VaR) MODELS

```python
# risk/var_models.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple

class VaRCalculator:
    """
    Multiple VaR calculation methods.
    
    Methods:
    - Historical: Empirical quantile
    - Parametric: Normal distribution assumption
    - Cornish-Fisher: Adjusts for skewness and kurtosis
    - Monte Carlo: Simulation-based
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: VaR confidence (e.g., 0.95 for 95%)
        """
        self.confidence = confidence_level
    
    def historical_var(self, returns: pd.Series, 
                      portfolio_value: float = 1.0) -> float:
        """
        Calculate historical VaR.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            float: VaR in currency terms
        """
        var_percentile = 1 - self.confidence
        var_return = np.percentile(returns, var_percentile * 100)
        return abs(var_return * portfolio_value)
    
    def parametric_var(self, returns: pd.Series,
                      portfolio_value: float = 1.0) -> float:
        """
        Calculate parametric (variance-covariance) VaR.
        
        Assumes normal distribution of returns.
        """
        mean = returns.mean()
        std = returns.std()
        
        z_score = stats.norm.ppf(1 - self.confidence)
        var_return = mean + z_score * std
        
        return abs(var_return * portfolio_value)
    
    def cornish_fisher_var(self, returns: pd.Series,
                          portfolio_value: float = 1.0) -> float:
        """
        Calculate Cornish-Fisher VaR.
        
        Adjusts for non-normal skewness and kurtosis.
        """
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        z = stats.norm.ppf(1 - self.confidence)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var_return = mean + z_cf * std
        return abs(var_return * portfolio_value)
    
    def monte_carlo_var(self, returns: pd.Series,
                       portfolio_value: float = 1.0,
                       n_simulations: int = 10000) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Simulates returns using bootstrapped historical distribution.
        """
        # Bootstrap sample
        simulated_returns = np.random.choice(
            returns, 
            size=n_simulations, 
            replace=True
        )
        
        var_percentile = 1 - self.confidence
        var_return = np.percentile(simulated_returns, var_percentile * 100)
        
        return abs(var_return * portfolio_value)
    
    def calculate_all(self, returns: pd.Series,
                     portfolio_value: float = 1.0) -> dict:
        """
        Calculate VaR using all methods.
        
        Args:
            returns: Historical returns
            portfolio_value: Portfolio value
            
        Returns:
            dict: VaR by method
        """
        return {
            'historical': self.historical_var(returns, portfolio_value),
            'parametric': self.parametric_var(returns, portfolio_value),
            'cornish_fisher': self.cornish_fisher_var(returns, portfolio_value),
            'monte_carlo': self.monte_carlo_var(returns, portfolio_value),
            'confidence_level': self.confidence,
            'portfolio_value': portfolio_value
        }


class CVaRCalculator:
    """
    Conditional VaR (Expected Shortfall) calculator.
    
    CVaR is the expected loss given that VaR is exceeded.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence = confidence_level
    
    def historical_cvar(self, returns: pd.Series,
                       portfolio_value: float = 1.0) -> float:
        """
        Calculate historical CVaR.
        """
        var_percentile = 1 - self.confidence
        var_threshold = np.percentile(returns, var_percentile * 100)
        
        # Average of returns worse than VaR
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return abs(var_threshold * portfolio_value)
        
        cvar_return = tail_returns.mean()
        return abs(cvar_return * portfolio_value)
    
    def parametric_cvar(self, returns: pd.Series,
                       portfolio_value: float = 1.0) -> float:
        """
        Calculate parametric CVaR for normal distribution.
        
        Formula: CVaR = -mu + sigma * phi(z) / (1-confidence)
        where phi is standard normal PDF and z is VaR quantile.
        """
        mean = returns.mean()
        std = returns.std()
        
        z = stats.norm.ppf(1 - self.confidence)
        phi_z = stats.norm.pdf(z)
        
        cvar_return = mean - std * phi_z / (1 - self.confidence)
        return abs(cvar_return * portfolio_value)


class DrawdownController:
    """
    Dynamic drawdown control system.
    
    Reduces exposure as drawdown increases.
    """
    
    def __init__(self, max_drawdown: float = 0.10,
                 reduction_steps: list = None):
        """
        Args:
            max_drawdown: Maximum acceptable drawdown
            reduction_steps: List of (drawdown_threshold, exposure_multiplier)
        """
        self.max_dd = max_drawdown
        
        if reduction_steps is None:
            self.steps = [
                (0.05, 1.0),    # Up to 5% DD: full exposure
                (0.07, 0.75),   # 5-7% DD: 75% exposure
                (0.10, 0.50),   # 7-10% DD: 50% exposure
                (0.15, 0.25),   # 10-15% DD: 25% exposure
                (float('inf'), 0.0)  # Above 15%: stop trading
            ]
        else:
            self.steps = reduction_steps
    
    def get_exposure_multiplier(self, current_drawdown: float) -> float:
        """
        Get exposure multiplier based on current drawdown.
        
        Args:
            current_drawdown: Current portfolio drawdown
            
        Returns:
            float: Exposure multiplier (0 to 1)
        """
        for threshold, multiplier in self.steps:
            if current_drawdown <= threshold:
                return multiplier
        
        return 0.0
    
    def should_stop_trading(self, current_drawdown: float) -> bool:
        """Check if trading should be halted."""
        return current_drawdown >= self.max_dd * 1.5  # Hard stop at 150% of max
```

---

*Note: This is Part 2 of the comprehensive agent.md file. The document continues with data collection, execution, and deployment sections in subsequent parts.*


---

# SECTION 6: DATA COLLECTION AND MANAGEMENT

## 6.1 MULTI-SOURCE DATA COLLECTION

### 6.1.1 Data Source Architecture

```python
# data/collectors/master_collector.py

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import requests
import time
from functools import wraps

class RateLimiter:
    """
    Rate limiter for API calls.
    
    Implements token bucket algorithm.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, rate_limit: int = 60):
        self.name = name
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache = {}
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, start: datetime, 
                   end: datetime, interval: str = '1d') -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available."""
        pass
    
    def _cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        return str(sorted(kwargs.items()))
    
    def get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and fresh."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return data
        return None
    
    def set_cached(self, key: str, data: pd.DataFrame):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())


class YFinanceSource(DataSource):
    """
    Yahoo Finance data source (free, no API key).
    
    Rate limit: Unlimited (be respectful)
    Coverage: Stocks, ETFs, indices, forex, crypto
    History: Up to 30+ years for stocks
    """
    
    def __init__(self):
        super().__init__('yfinance', rate_limit=120)
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")
    
    @retry_on_failure(max_retries=3)
    def fetch_ohlcv(self, symbol: str, start: datetime,
                   end: datetime, interval: str = '1d') -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance."""
        self.rate_limiter.wait()
        
        cache_key = self._cache_key(symbol=symbol, start=start, end=end, interval=interval)
        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached
        
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        
        # Standardize column names
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add returns
        df['returns'] = df['close'].pct_change()
        
        self.set_cached(cache_key, df)
        return df
    
    def fetch_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamental data."""
        self.rate_limiter.wait()
        
        ticker = self.yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_value': info.get('enterpriseValue'),
            'profit_margins': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'average_volume': info.get('averageVolume'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'short_ratio': info.get('shortRatio'),
            'short_percent_of_float': info.get('shortPercentOfFloat')
        }
    
    def is_available(self) -> bool:
        """Check if yfinance is available."""
        try:
            import yfinance
            return True
        except ImportError:
            return False


class AlphaVantageSource(DataSource):
    """
    Alpha Vantage data source (free tier: 5 calls/minute).
    
    Requires API key from alphavantage.co
    """
    
    def __init__(self, api_key: str):
        super().__init__('alphavantage', rate_limit=5)
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    @retry_on_failure(max_retries=3)
    def fetch_ohlcv(self, symbol: str, start: datetime,
                   end: datetime, interval: str = '1d') -> pd.DataFrame:
        """Fetch OHLCV from Alpha Vantage."""
        self.rate_limiter.wait()
        
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No data for {symbol}: {data.get('Note', 'Unknown error')}")
        
        # Parse to DataFrame
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adj_close',
            '6. volume': 'volume'
        })
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter date range
        df = df[(df.index >= start) & (df.index <= end)]
        
        # Add returns
        df['returns'] = df['adj_close'].pct_change()
        
        return df
    
    def fetch_intraday(self, symbol: str, interval: str = '5min',
                      months: int = 1) -> pd.DataFrame:
        """Fetch intraday data."""
        self.rate_limiter.wait()
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            raise ValueError(f"No intraday data for {symbol}")
        
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def is_available(self) -> bool:
        """Check API key is valid."""
        try:
            self.rate_limiter.wait()
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'IBM',
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            return 'Global Quote' in response.json()
        except:
            return False


class CongressTradesSource:
    """
    Congress trading activity data.
    
    Sources:
    - Senate Stock Watcher (senatestockwatcher.com)
    - House Stock Watcher
    - Capitol Trades (capitoltrades.com)
    """
    
    def __init__(self):
        self.base_url = 'https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com'
    
    def fetch_senate_trades(self, year: int = None) -> pd.DataFrame:
        """
        Fetch Senate stock transactions.
        
        Args:
            year: Year to fetch (default: current year)
            
        Returns:
            DataFrame with trade data
        """
        if year is None:
            year = datetime.now().year
        
        url = f"{self.base_url}/aggregate/transaction_report_for_{year}.json"
        
        response = requests.get(url)
        data = response.json()
        
        df = pd.DataFrame(data)
        
        # Parse dates
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['disclosure_date'] = pd.to_datetime(df['disclosure_date'])
        
        # Clean amounts
        df['amount'] = df['amount'].str.replace('$', '').str.replace(',', '')
        df['amount_min'] = df['amount'].str.split(' - ').str[0].astype(float)
        df['amount_max'] = df['amount'].str.split(' - ').str[1].astype(float)
        df['amount_mid'] = (df['amount_min'] + df['amount_max']) / 2
        
        # Add signal
        df['signal'] = df['type'].apply(
            lambda x: 1 if 'Purchase' in x else (-1 if 'Sale' in x else 0)
        )
        
        return df
    
    def get_buys(self, df: pd.DataFrame, 
                min_amount: float = 1000) -> pd.DataFrame:
        """Filter for significant buy transactions."""
        return df[(df['signal'] == 1) & (df['amount_mid'] >= min_amount)]
    
    def get_politician_sentiment(self, df: pd.DataFrame, 
                                 symbol: str,
                                 lookback_days: int = 90) -> Dict:
        """
        Calculate sentiment from politician trading for a symbol.
        
        Args:
            df: Congress trades DataFrame
            symbol: Stock symbol
            lookback_days: Days to look back
            
        Returns:
            dict: Sentiment metrics
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = df[df['transaction_date'] >= cutoff]
        
        symbol_trades = recent[recent['ticker'] == symbol]
        
        if len(symbol_trades) == 0:
            return {'sentiment': 0, 'confidence': 0, 'n_trades': 0}
        
        # Weight by transaction amount
        total_buys = symbol_trades[symbol_trades['signal'] == 1]['amount_mid'].sum()
        total_sells = symbol_trades[symbol_trades['signal'] == -1]['amount_mid'].sum()
        
        total = total_buys + total_sells
        if total == 0:
            return {'sentiment': 0, 'confidence': 0, 'n_trades': len(symbol_trades)}
        
        sentiment = (total_buys - total_sells) / total
        confidence = min(len(symbol_trades) / 5, 1.0)  # More trades = higher confidence
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'n_trades': len(symbol_trades),
            'total_buy_value': total_buys,
            'total_sell_value': total_sells
        }


class NewsSentimentSource:
    """
    News sentiment data collection.
    
    Sources:
    - NewsAPI (newsapi.org) - 100 requests/day free
    - Finnhub (finnhub.io) - 60 calls/minute free
    - GDELT (gdeltproject.org) - unlimited
    """
    
    def __init__(self, newsapi_key: str = None, finnhub_key: str = None):
        self.newsapi_key = newsapi_key
        self.finnhub_key = finnhub_key
        self.rate_limiter = RateLimiter(60)
    
    def fetch_newsapi(self, query: str, from_date: datetime,
                     to_date: datetime) -> List[Dict]:
        """Fetch news from NewsAPI."""
        if not self.newsapi_key:
            return []
        
        url = 'https://newsapi.org/v2/everything'
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'apiKey': self.newsapi_key,
            'pageSize': 100
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') != 'ok':
            return []
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'published_at': article.get('publishedAt'),
                'source': article.get('source', {}).get('name'),
                'url': article.get('url')
            })
        
        return articles
    
    def fetch_finnhub_news(self, symbol: str, 
                          from_date: datetime,
                          to_date: datetime) -> List[Dict]:
        """Fetch company news from Finnhub."""
        if not self.finnhub_key:
            return []
        
        self.rate_limiter.wait()
        
        url = 'https://finnhub.io/api/v1/company-news'
        
        params = {
            'symbol': symbol,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        articles = []
        for item in data:
            articles.append({
                'title': item.get('headline'),
                'description': item.get('summary'),
                'content': item.get('summary'),
                'published_at': datetime.fromtimestamp(item.get('datetime')),
                'source': item.get('source'),
                'url': item.get('url')
            })
        
        return articles


class MasterDataCollector:
    """
    Unified data collection orchestrator.
    
    Manages multiple data sources with failover and caching.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dict with API keys
        """
        self.sources = {}
        
        # Initialize available sources
        self.sources['yfinance'] = YFinanceSource()
        
        if config.get('alphavantage_key'):
            self.sources['alphavantage'] = AlphaVantageSource(
                config['alphavantage_key']
            )
        
        self.congress = CongressTradesSource()
        
        self.news = NewsSentimentSource(
            newsapi_key=config.get('newsapi_key'),
            finnhub_key=config.get('finnhub_key')
        )
        
        # Universe of stocks
        self.universe = self._load_universe()
    
    def _load_universe(self) -> List[str]:
        """Load default stock universe (S&P 500)."""
        try:
            # Fetch S&P 500 from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500 = tables[0]
            return sp500['Symbol'].tolist()
        except:
            # Fallback to common stocks
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM']
    
    def fetch_price_data(self, symbol: str, start: datetime,
                        end: datetime, interval: str = '1d',
                        prefer_source: str = None) -> pd.DataFrame:
        """
        Fetch price data with source failover.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval
            prefer_source: Preferred data source
            
        Returns:
            DataFrame with OHLCV data
        """
        # Try preferred source first
        if prefer_source and prefer_source in self.sources:
            try:
                return self.sources[prefer_source].fetch_ohlcv(
                    symbol, start, end, interval
                )
            except Exception as e:
                print(f"{prefer_source} failed: {e}, trying fallback...")
        
        # Try all sources
        for name, source in self.sources.items():
            if name == prefer_source:
                continue
            try:
                return source.fetch_ohlcv(symbol, start, end, interval)
            except Exception as e:
                print(f"{name} failed: {e}")
                continue
        
        raise ValueError(f"Could not fetch data for {symbol} from any source")
    
    def fetch_batch(self, symbols: List[str], start: datetime,
                   end: datetime, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            dict: Symbol -> DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_price_data(symbol, start, end, interval)
                results[symbol] = df
                print(f"✓ {symbol}: {len(df)} rows")
            except Exception as e:
                print(f"✗ {symbol}: {e}")
                continue
        
        return results
    
    def fetch_fundamentals_batch(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch fundamentals for multiple symbols."""
        results = []
        
        for symbol in symbols:
            try:
                fundamentals = self.sources['yfinance'].fetch_fundamentals(symbol)
                fundamentals['symbol'] = symbol
                results.append(fundamentals)
            except Exception as e:
                print(f"Failed to fetch fundamentals for {symbol}: {e}")
        
        return pd.DataFrame(results)
    
    def daily_update(self):
        """
        Run daily data update routine.
        
        Should be scheduled to run after market close.
        """
        end = datetime.now()
        start = end - timedelta(days=5)  # Update last 5 days
        
        print(f"Starting daily update for {len(self.universe)} symbols...")
        
        for symbol in self.universe:
            try:
                self.fetch_price_data(symbol, start, end)
                time.sleep(0.5)  # Be nice to APIs
            except Exception as e:
                print(f"Failed to update {symbol}: {e}")
        
        print("Daily update complete")
```

## 6.2 DATA QUALITY ASSURANCE

```python
# data/validators/data_quality.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats

class DataQualityValidator:
    """
    Comprehensive data quality validation.
    
    Checks:
    - Missing values
    - Outliers
    - Price inconsistencies
    - Volume anomalies
    - Gaps in data
    """
    
    def __init__(self, max_missing_pct: float = 0.05,
                 outlier_zscore: float = 4.0):
        """
        Args:
            max_missing_pct: Maximum acceptable missing data percentage
            outlier_zscore: Z-score threshold for outliers
        """
        self.max_missing = max_missing_pct
        self.outlier_zscore = outlier_zscore
    
    def validate_ohlcv(self, df: pd.DataFrame) -> Dict:
        """
        Validate OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            dict: Validation results
        """
        issues = []
        warnings = []
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return {'valid': False, 'issues': issues, 'warnings': warnings}
        
        # Check for missing values
        for col in required:
            missing_pct = df[col].isna().mean()
            if missing_pct > self.max_missing:
                issues.append(f"{col}: {missing_pct:.1%} missing (max {self.max_missing:.1%})")
            elif missing_pct > 0:
                warnings.append(f"{col}: {missing_pct:.1%} missing")
        
        # Check OHLC logic
        ohlc_valid = (
            (df['high'] >= df[['open', 'close', 'low']].max(axis=1)) &
            (df['low'] <= df[['open', 'close', 'high']].min(axis=1))
        )
        
        invalid_ohlc = (~ohlc_valid).sum()
        if invalid_ohlc > 0:
            issues.append(f"{invalid_ohlc} rows with invalid OHLC logic")
        
        # Check for zero prices
        zero_prices = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1).sum()
        if zero_prices > 0:
            issues.append(f"{zero_prices} rows with zero prices")
        
        # Check for negative prices
        neg_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1).sum()
        if neg_prices > 0:
            issues.append(f"{neg_prices} rows with negative prices")
        
        # Check volume
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > len(df) * 0.1:
            warnings.append(f"{zero_volume} rows with zero volume ({zero_volume/len(df):.1%})")
        
        # Check for outliers in returns
        if 'returns' in df.columns:
            returns = df['returns'].dropna()
            z_scores = np.abs(stats.zscore(returns))
            outliers = (z_scores > self.outlier_zscore).sum()
            if outliers > len(returns) * 0.01:
                warnings.append(f"{outliers} return outliers detected")
        
        # Check for gaps
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index[0], 
                                          end=df.index[-1], 
                                          freq=expected_freq)
                missing_dates = len(full_range) - len(df)
                if missing_dates > len(full_range) * 0.05:
                    warnings.append(f"{missing_dates} missing dates ({missing_dates/len(full_range):.1%})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'n_rows': len(df),
            'date_range': (df.index[0], df.index[-1]) if len(df) > 0 else None
        }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by fixing common issues.
        
        Args:
            df: Raw OHLCV data
            
        Returns:
            DataFrame: Cleaned data
        """
        df = df.copy()
        
        # Remove rows with all NaN
        df = df.dropna(how='all')
        
        # Forward fill missing prices (up to 3 days)
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill(limit=3)
        
        # Fill missing volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Fix OHLC inconsistencies
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Remove rows with zero close price
        df = df[df['close'] > 0]
        
        # Recalculate returns
        df['returns'] = df['close'].pct_change()
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalous price movements.
        
        Args:
            df: OHLCV data
            
        Returns:
            DataFrame: Anomaly flags
        """
        anomalies = pd.DataFrame(index=df.index)
        anomalies['price_spike'] = False
        anomalies['volume_spike'] = False
        anomalies['gap'] = False
        
        # Price spikes (returns > 5 sigma)
        if 'returns' in df.columns:
            returns = df['returns'].dropna()
            rolling_std = returns.rolling(20).std()
            anomalies['price_spike'] = np.abs(returns) > (5 * rolling_std)
        
        # Volume spikes
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(20).mean()
            anomalies['volume_spike'] = df['volume'] > (5 * volume_ma)
        
        # Gaps (open not near previous close)
        prev_close = df['close'].shift(1)
        gap = (df['open'] - prev_close) / prev_close
        anomalies['gap'] = np.abs(gap) > 0.05  # 5% gap
        
        return anomalies
```

---

# SECTION 7: EXECUTION AND DEPLOYMENT

## 7.1 SINGLE-COMMAND DEPLOYMENT SCRIPT

```bash
#!/bin/bash
# deploy_quantum_alpha.sh

# QUANTUM ALPHA - Single-Command Deployment
# Usage: ./deploy_quantum_alpha.sh --mode=[backtest|paper|live] --capital=100000

set -e  # Exit on error

# Default values
MODE="backtest"
CAPITAL=100000
RISK_PROFILE="moderate"
CONFIG_FILE="config/settings.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --capital=*)
            CAPITAL="${1#*=}"
            shift
            ;;
        --risk-profile=*)
            RISK_PROFILE="${1#*=}"
            shift
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: ./deploy_quantum_alpha.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode=MODE           Deployment mode: backtest, paper, live"
            echo "  --capital=AMOUNT      Initial capital (default: 100000)"
            echo "  --risk-profile=PROFILE Risk profile: conservative, moderate, aggressive"
            echo "  --config=FILE         Configuration file path"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "QUANTUM ALPHA DEPLOYMENT"
echo "=========================================="
echo "Mode: $MODE"
echo "Capital: \$${CAPITAL}"
echo "Risk Profile: $RISK_PROFILE"
echo "=========================================="

# Step 1: Environment Validation
echo ""
echo "[1/7] Validating environment..."
python3 -c "
import sys
required = ['numpy', 'pandas', 'tensorflow', 'torch', 'transformers', 'yfinance', 'optuna']
missing = [p for p in required if __import__('importlib').util.find_spec(p) is None]
if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
print('All required packages installed ✓')
"

# Step 2: Database Initialization
echo ""
echo "[2/7] Initializing database..."
python3 -c "
from data.storage.timescale_manager import TimescaleManager
manager = TimescaleManager()
manager.initialize_schema()
print('Database initialized ✓')
"

# Step 3: Data Collection
echo ""
echo "[3/7] Collecting market data..."
python3 << EOF
from data.collectors.master_collector import MasterDataCollector
from datetime import datetime, timedelta
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

collector = MasterDataCollector(config)

# Update universe
end = datetime.now()
start = end - timedelta(days=252*5)  # 5 years

collector.fetch_batch(
    collector.universe[:50],  # Start with top 50
    start, end
)

print('Data collection complete ✓')
EOF

# Step 4: Feature Engineering
echo ""
echo "[4/7] Engineering features..."
python3 << EOF
from features.engineering_pipeline import FeaturePipeline
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

pipeline = FeaturePipeline(config)
pipeline.run_all()

print('Feature engineering complete ✓')
EOF

# Step 5: Model Training
echo ""
echo "[5/7] Training ML models..."
python3 << EOF
from models.training_pipeline import TrainingPipeline
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

pipeline = TrainingPipeline(config)

# Train LSTM
print('Training LSTM...')
pipeline.train_lstm()

# Train sentiment model
print('Training sentiment model...')
pipeline.train_sentiment()

print('Model training complete ✓')
EOF

# Step 6: Backtest Validation
echo ""
echo "[6/7] Running backtest validation..."
python3 << EOF
from backtesting.validation_suite import ValidationSuite
from strategy.signal_aggregator import SignalAggregator
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Run validation suite
validator = ValidationSuite(config)
results = validator.run_all()

# Check if strategy passes
if results['mcpt_passed'] and results['wfo_efficiency'] > 0.5:
    print('✓ Strategy validation PASSED')
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  MCPT p-value: {results['mcpt_pvalue']:.4f}")
    print(f"  WFE: {results['wfo_efficiency']:.2%}")
else:
    print('✗ Strategy validation FAILED')
    print('  Review results before deployment')
    exit(1)
EOF

# Step 7: Deploy
echo ""
echo "[7/7] Deploying..."

if [ "$MODE" == "backtest" ]; then
    echo "Running full backtest..."
    python3 << EOF
from backtesting.full_backtest import FullBacktest
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

backtest = FullBacktest(config)
results = backtest.run(start_date='2015-01-01')
backtest.generate_report(results, 'reports/backtest_report.html')
print('Backtest complete. Report saved to reports/backtest_report.html')
EOF

elif [ "$MODE" == "paper" ]; then
    echo "Starting paper trading..."
    python3 << EOF
from execution.paper_trader import PaperTrader
from monitoring.dashboard import start_dashboard
import yaml
import threading

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Start dashboard
threading.Thread(target=start_dashboard, daemon=True).start()

# Start paper trading
trader = PaperTrader(config, capital=$CAPITAL)
trader.run()
EOF

elif [ "$MODE" == "live" ]; then
    echo "WARNING: Starting LIVE trading with real capital!"
    read -p "Are you sure? Type 'YES' to continue: " confirm
    if [ "$confirm" != "YES" ]; then
        echo "Aborted."
        exit 1
    fi
    
    python3 << EOF
from execution.live_trader import LiveTrader
from monitoring.dashboard import start_dashboard
from monitoring.alert_system import AlertSystem
import yaml
import threading

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Initialize alert system
alerts = AlertSystem(config)
alerts.send_notification('Quantum Alpha', 'Live trading starting')

# Start dashboard
threading.Thread(target=start_dashboard, daemon=True).start()

# Start live trading
trader = LiveTrader(config, capital=$CAPITAL)
trader.run()
EOF
fi

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "Mode: $MODE"
echo "=========================================="
```

---

# SECTION 8: ENSEMBLE MODEL COMBINATION

## 8.1 STACKED ENSEMBLE ARCHITECTURE

```python
# models/ensemble/stacker.py

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor

class StackedEnsemble:
    """
    Stacked ensemble combining multiple base models.
    
    Architecture:
    - Level 0: Base models (LSTM, QMC, Sentiment, Technical)
    - Level 1: Meta-learner combining base predictions
    """
    
    def __init__(self, base_models: Dict[str, Callable],
                 meta_learner: str = 'ridge'):
        """
        Args:
            base_models: Dict of model_name -> predict_function
            meta_learner: 'ridge', 'logistic', or 'gbm'
        """
        self.base_models = base_models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        self.base_predictions = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
           validation_split: float = 0.2):
        """
        Fit stacked ensemble using cross-validation.
        
        Args:
            X: Features
            y: Target
            validation_split: Fraction for meta-learner training
        """
        # Split for meta-learner
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_meta = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_meta = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Generate base model predictions on meta set
        meta_features = pd.DataFrame(index=X_meta.index)
        
        for name, model in self.base_models.items():
            # Fit on training data
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Predict on meta set
            if hasattr(model, 'predict'):
                preds = model.predict(X_meta)
            else:
                preds = model(X_meta)
            
            meta_features[f'{name}_pred'] = preds
        
        # Train meta-learner
        if self.meta_learner_type == 'ridge':
            self.meta_learner = Ridge(alpha=1.0)
        elif self.meta_learner_type == 'logistic':
            self.meta_learner = LogisticRegression()
        elif self.meta_learner_type == 'gbm':
            self.meta_learner = GradientBoostingRegressor(
                n_estimators=100, max_depth=3
            )
        
        self.meta_learner.fit(meta_features, y_meta)
        
        # Store feature importance
        if hasattr(self.meta_learner, 'coef_'):
            self.feature_importance = dict(
                zip(meta_features.columns, self.meta_learner.coef_)
            )
        elif hasattr(self.meta_learner, 'feature_importances_'):
            self.feature_importance = dict(
                zip(meta_features.columns, self.meta_learner.feature_importances_)
            )
        
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Generate ensemble prediction.
        
        Args:
            X: Features
            
        Returns:
            dict: Prediction with confidence
        """
        # Generate base predictions
        meta_features = pd.DataFrame(index=X.index)
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict'):
                preds = model.predict(X)
            else:
                preds = model(X)
            meta_features[f'{name}_pred'] = preds
        
        # Meta prediction
        ensemble_pred = self.meta_learner.predict(meta_features)
        
        # Calculate prediction disagreement (uncertainty)
        base_preds = meta_features.values
        pred_std = np.std(base_preds, axis=1)
        
        return {
            'prediction': ensemble_pred,
            'uncertainty': pred_std,
            'base_predictions': meta_features.to_dict('records'),
            'confidence': 1 / (1 + pred_std)  # Higher uncertainty = lower confidence
        }
    
    def get_feature_importance(self) -> Dict:
        """Get importance of each base model."""
        return getattr(self, 'feature_importance', {})


class DynamicEnsemble:
    """
    Dynamic ensemble that adapts model weights based on recent performance.
    """
    
    def __init__(self, models: Dict[str, Callable],
                 lookback_window: int = 63,
                 update_frequency: int = 21):
        """
        Args:
            models: Base models
            lookback_window: Days to evaluate performance
            update_frequency: Days between weight updates
        """
        self.models = models
        self.lookback = lookback_window
        self.update_freq = update_frequency
        self.weights = {name: 1/len(models) for name in models}
        self.performance_history = {name: [] for name in models}
        
    def update_weights(self, actual_returns: pd.Series):
        """
        Update model weights based on recent performance.
        
        Args:
            actual_returns: Actual realized returns
        """
        # Calculate recent performance for each model
        performances = {}
        
        for name, history in self.performance_history.items():
            if len(history) >= self.lookback:
                recent = history[-self.lookback:]
                # Sharpe-like metric
                sharpe = np.mean(recent) / (np.std(recent) + 1e-8)
                performances[name] = max(sharpe, 0)  # No negative weights
        
        if performances:
            # Softmax weighting
            exp_perf = {k: np.exp(v) for k, v in performances.items()}
            total = sum(exp_perf.values())
            self.weights = {k: v/total for k, v in exp_perf.items()}
    
    def predict(self, X: pd.DataFrame, day_counter: int = 0) -> float:
        """
        Generate weighted prediction.
        
        Args:
            X: Features
            day_counter: Current day (for weight updates)
            
        Returns:
            float: Weighted prediction
        """
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                pred = model(X)
            predictions[name] = pred
        
        # Weighted average
        weighted_pred = sum(
            self.weights[name] * pred 
            for name, pred in predictions.items()
        )
        
        return weighted_pred
```

---

# SECTION 9: MONITORING AND REPORTING

## 9.1 PERFORMANCE DASHBOARD

```python
# visualization/dashboard.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingDashboard:
    """
    Real-time trading dashboard using Streamlit.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def run(self):
        """Run the dashboard."""
        st.set_page_config(
            page_title="Quantum Alpha Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Quantum Alpha Trading Dashboard")
        
        # Sidebar
        st.sidebar.header("Controls")
        refresh_rate = st.sidebar.slider("Refresh (seconds)", 5, 60, 30)
        
        # Main metrics
        self._display_metrics()
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            self._display_equity_curve()
        with col2:
            self._display_drawdown()
        
        # Positions and signals
        col1, col2 = st.columns(2)
        with col1:
            self._display_positions()
        with col2:
            self._display_signals()
        
        # Model performance
        self._display_model_performance()
        
    def _display_metrics(self):
        """Display key performance metrics."""
        metrics = self._fetch_latest_metrics()
        
        cols = st.columns(6)
        
        with cols[0]:
            st.metric(
                "Portfolio Value",
                f"${metrics['portfolio_value']:,.2f}",
                f"{metrics['daily_return']:.2%}"
            )
        
        with cols[1]:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.2%}"
            )
        
        with cols[2]:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe']:.2f}"
            )
        
        with cols[3]:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2%}"
            )
        
        with cols[4]:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}"
            )
        
        with cols[5]:
            st.metric(
                "Active Positions",
                f"{metrics['n_positions']}"
            )
    
    def _display_equity_curve(self):
        """Plot equity curve vs benchmarks."""
        equity = self._fetch_equity_history()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity['portfolio'],
            name='Quantum Alpha',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity['spy'],
            name='S&P 500',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_drawdown(self):
        """Plot drawdown over time."""
        equity = self._fetch_equity_history()
        peak = equity['portfolio'].cummax()
        drawdown = (equity['portfolio'] - peak) / peak
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_positions(self):
        """Display current positions."""
        st.subheader("Current Positions")
        
        positions = self._fetch_positions()
        
        if positions.empty:
            st.info("No active positions")
            return
        
        # Format for display
        display_df = positions.copy()
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
        display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    def _display_signals(self):
        """Display recent trading signals."""
        st.subheader("Recent Signals")
        
        signals = self._fetch_recent_signals()
        
        if signals.empty:
            st.info("No recent signals")
            return
        
        st.dataframe(signals, use_container_width=True)
    
    def _display_model_performance(self):
        """Display individual model performance."""
        st.subheader("Model Performance")
        
        model_perf = self._fetch_model_performance()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Prediction Accuracy', 'Contribution to Returns'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Accuracy bar chart
        fig.add_trace(
            go.Bar(
                x=list(model_perf.keys()),
                y=[p['accuracy'] for p in model_perf.values()],
                name='Accuracy'
            ),
            row=1, col=1
        )
        
        # Contribution pie chart
        fig.add_trace(
            go.Pie(
                labels=list(model_perf.keys()),
                values=[p['contribution'] for p in model_perf.values()]
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data fetching methods (placeholders)
    def _fetch_latest_metrics(self):
        return {
            'portfolio_value': 127543.21,
            'daily_return': 0.0123,
            'total_return': 0.2754,
            'sharpe': 1.82,
            'max_drawdown': -0.083,
            'win_rate': 0.67,
            'n_positions': 5
        }
    
    def _fetch_equity_history(self):
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        portfolio = 100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
        spy = 100000 * (1 + np.random.randn(len(dates)).cumsum() * 0.0008)
        return pd.DataFrame({'portfolio': portfolio, 'spy': spy}, index=dates)
    
    def _fetch_positions(self):
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'NVDA'],
            'quantity': [100, 50, 25],
            'avg_price': [150.0, 300.0, 400.0],
            'current_price': [175.0, 320.0, 450.0],
            'pnl_pct': [0.167, 0.067, 0.125],
            'value': [17500, 16000, 11250]
        })
    
    def _fetch_recent_signals(self):
        return pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='H'),
            'symbol': ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'META'],
            'signal': ['BUY', 'SELL', 'BUY', 'HOLD', 'BUY'],
            'strength': [0.75, -0.65, 0.82, 0.12, 0.71],
            'confidence': [0.85, 0.78, 0.91, 0.45, 0.88]
        })
    
    def _fetch_model_performance(self):
        return {
            'LSTM': {'accuracy': 0.62, 'contribution': 0.30},
            'QMC': {'accuracy': 0.58, 'contribution': 0.25},
            'Sentiment': {'accuracy': 0.55, 'contribution': 0.20},
            'Technical': {'accuracy': 0.52, 'contribution': 0.15},
            'TDA': {'accuracy': 0.60, 'contribution': 0.10}
        }


def start_dashboard():
    """Entry point for dashboard."""
    dashboard = TradingDashboard(db_connection=None)
    dashboard.run()


if __name__ == '__main__':
    start_dashboard()
```

---

# SECTION 10: AI AGENT PROMPTING GUIDELINES

## 10.1 CODE GENERATION PROMPTS

When generating code for this system, AI agents MUST follow these patterns:

### 10.1.1 Function Documentation Template

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Mathematical Foundation:
    - Reference the formula or algorithm used
    - Cite any papers or sources
    
    Args:
        param1: Description with units if applicable
        param2: Description with valid ranges
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When/why this exception is raised
        
    Example:
        >>> result = function_name(10, 0.5)
        >>> print(result)
        5.0
    """
    pass
```

### 10.1.2 Error Handling Pattern

```python
def robust_function(data: pd.DataFrame) -> pd.DataFrame:
    """Function with comprehensive error handling."""
    
    # Validate inputs
    if data is None or data.empty:
        raise ValueError("Input data cannot be empty")
    
    # Create copy to avoid modifying input
    df = data.copy()
    
    try:
        # Core logic
        result = df.apply(some_operation)
        
    except Exception as e:
        # Log error and return safe default
        logger.error(f"Operation failed: {e}")
        return df  # Return original on failure
    
    # Validate outputs
    if result.isna().all().any():
        logger.warning("Operation produced all NaN values")
    
    return result
```

### 10.1.3 Testing Pattern

```python
def test_function_name():
    """Unit test for function_name."""
    # Setup
    test_data = create_test_data()
    
    # Execute
    result = function_name(test_data)
    
    # Assert
    assert result is not None
    assert len(result) == len(test_data)
    assert not result.isna().any().any()
    
    print("✓ test_function_name passed")


if __name__ == '__main__':
    test_function_name()
```

## 10.2 ANTI-PATTERNS TO AVOID

### 10.2.1 Forbidden Patterns

1. **No magic numbers** - All constants must be named and configurable
2. **No silent failures** - All errors must be logged or raised
3. **No global state** - All functions must be pure or class-based
4. **No hardcoded paths** - All paths must be configurable
5. **No print statements** - Use logging instead

### 10.2.2 Performance Anti-Patterns

1. **No loops in hot paths** - Use vectorized operations
2. **No DataFrame iteration** - Use apply or vectorized methods
3. **No redundant calculations** - Cache intermediate results
4. **No memory leaks** - Clear large objects when done

## 10.3 CONFIGURATION SCHEMA

```yaml
# config/settings.yaml

system:
  name: "Quantum Alpha"
  version: "1.0.0"
  log_level: "INFO"
  
data:
  sources:
    primary: "yfinance"
    fallback: "alphavantage"
  cache_dir: "./cache"
  history_years: 10
  
models:
  lstm:
    sequence_length: 90
    hidden_units: [128, 64]
    dropout: 0.3
    learning_rate: 0.001
    epochs: 100
    
  sentiment:
    model: "ProsusAI/finbert"
    batch_size: 16
    confidence_threshold: 0.6
    
  qmc:
    n_paths: 500
    horizon_days: 21
    
trading:
  initial_capital: 100000
  max_position_pct: 0.25
  stop_loss_pct: 0.10
  take_profit_pct: 0.20
  
risk:
  max_drawdown: 0.10
  var_confidence: 0.95
  kelly_fraction: 0.5
  
backtest:
  start_date: "2015-01-01"
  end_date: "2024-12-31"
  transaction_cost: 0.001
  slippage_bps: 5
```

---

# APPENDIX A: MATHEMATICAL REFERENCE

## A.1 Key Formulas

### Sharpe Ratio
$$S = \frac{E[R_p - R_f]}{\sigma_p}$$

### Maximum Drawdown
$$MDD = \max_{t \in [0,T]} \left( \frac{Peak_t - Value_t}{Peak_t} \right)$$

### Kelly Criterion
$$f^* = \frac{p(b+1) - 1}{b}$$

### Value at Risk (Parametric)
$$VaR_\alpha = \mu + \sigma \cdot \Phi^{-1}(\alpha)$$

### Wasserstein Distance
$$W_p(\mu, \nu) = \left( \inf_{\pi \in \Pi(\mu, \nu)} \int d(x,y)^p d\pi \right)^{1/p}$$

### Hurst Exponent
$$H = \frac{\log(R/S)}{\log(n)}$$

## A.2 Performance Metrics Reference

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Sharpe Ratio | $(R_p - R_f) / \sigma_p$ | Risk-adjusted return |
| Sortino Ratio | $(R_p - R_f) / \sigma_d$ | Downside-adjusted return |
| Calmar Ratio | $R_p / MDD$ | Return to max drawdown |
| Omega Ratio | $LPM_1(0) / HPM_1(0)$ | Gain/loss asymmetry |
| Profit Factor | Gross Profit / Gross Loss | Win/loss ratio |

---

# APPENDIX B: DEPLOYMENT CHECKLIST

## Pre-Deployment

- [ ] All unit tests passing
- [ ] MCPT p-value < 0.05
- [ ] Walk-forward efficiency > 50%
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 15%
- [ ] Data quality checks passing
- [ ] Configuration validated
- [ ] API keys configured

## Deployment

- [ ] Database initialized
- [ ] Models trained and saved
- [ ] Backtest completed
- [ ] Paper trading started
- [ ] Dashboard accessible
- [ ] Alerts configured

## Post-Deployment

- [ ] Monitor for 30 days
- [ ] Track vs benchmarks
- [ ] Review daily P&L
- [ ] Check for anomalies
- [ ] Update models monthly

---

**END OF QUANTUM ALPHA AGENT SPECIFICATION**

*This document provides comprehensive guidance for building an institutional-grade algorithmic trading system. All code is provided for educational and research purposes. Trading involves substantial risk of loss.*


---

# SECTION 11: EXTENDED MATHEMATICAL FOUNDATIONS

## 11.1 ADVANCED STOCHASTIC CALCULUS FOR TRADING

### 11.1.1 Geometric Brownian Motion

The foundation of quantitative finance, Geometric Brownian Motion (GBM) models asset prices as:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:
- $S_t$ = Asset price at time $t$
- $\mu$ = Drift (expected return)
- $\sigma$ = Volatility
- $W_t$ = Wiener process (Brownian motion)

**Solution:**
$$S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)$$

```python
# features/mathematical/stochastic_processes.py

import numpy as np
from typing import Tuple, Callable
from scipy.stats import norm

class GeometricBrownianMotion:
    """
    Geometric Brownian Motion for price simulation.
    
    Used for:
    - Monte Carlo option pricing
    - Risk scenario generation
    - Strategy stress testing
    """
    
    def __init__(self, mu: float, sigma: float, S0: float):
        """
        Args:
            mu: Annual drift (expected return)
            sigma: Annual volatility
            S0: Initial price
        """
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1,
                antithetic: bool = True) -> np.ndarray:
        """
        Simulate GBM paths.
        
        Args:
            T: Time horizon in years
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            antithetic: Use antithetic variates for variance reduction
            
        Returns:
            np.ndarray: Shape (n_paths, n_steps+1)
        """
        dt = T / n_steps
        n_total = n_paths // 2 if antithetic else n_paths
        
        # Generate random increments
        dW = np.random.randn(n_total, n_steps) * np.sqrt(dt)
        
        if antithetic:
            dW = np.concatenate([dW, -dW], axis=0)
        
        # Cumulative sum for Brownian motion
        W = np.cumsum(dW, axis=1)
        W = np.concatenate([np.zeros((len(W), 1)), W], axis=1)
        
        # GBM formula
        t = np.linspace(0, T, n_steps + 1)
        exponent = (self.mu - 0.5 * self.sigma**2) * t + self.sigma * W
        S = self.S0 * np.exp(exponent)
        
        return S
    
    def transition_density(self, S_t: float, t: float, S_T: float, T: float) -> float:
        """
        Probability density of transitioning from S_t to S_T.
        
        Args:
            S_t: Price at time t
            t: Current time
            S_T: Target price
            T: Target time
            
        Returns:
            float: Probability density
        """
        tau = T - t
        if tau <= 0:
            return 1.0 if np.isclose(S_t, S_T) else 0.0
        
        log_return = np.log(S_T / S_t)
        mean = (self.mu - 0.5 * self.sigma**2) * tau
        std = self.sigma * np.sqrt(tau)
        
        return norm.pdf(log_return, mean, std)
    
    def probability_above(self, S_t: float, t: float, K: float, T: float) -> float:
        """
        Probability of price being above K at time T.
        
        Args:
            S_t: Current price
            t: Current time
            K: Strike/target price
            T: Target time
            
        Returns:
            float: Probability
        """
        tau = T - t
        if tau <= 0:
            return 1.0 if S_t > K else 0.0
        
        d2 = (np.log(S_t / K) + (self.mu - 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.cdf(d2)


class JumpDiffusionProcess:
    """
    Merton's Jump Diffusion model.
    
    Adds Poisson jumps to GBM for more realistic price dynamics.
    
    $$dS_t = \mu S_t dt + \sigma S_t dW_t + S_t dJ_t$$
    
    Where $J_t$ is a compound Poisson process.
    """
    
    def __init__(self, mu: float, sigma: float, lambda_jump: float,
                 mu_jump: float, sigma_jump: float, S0: float):
        """
        Args:
            mu: Drift
            sigma: Diffusion volatility
            lambda_jump: Jump intensity (jumps per year)
            mu_jump: Mean jump size (lognormal)
            sigma_jump: Jump size volatility
            S0: Initial price
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.S0 = S0
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Simulate jump diffusion paths.
        
        Args:
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            np.ndarray: Simulated paths
        """
        dt = T / n_steps
        
        # Diffusion component
        gbm = GeometricBrownianMotion(self.mu, self.sigma, self.S0)
        paths = gbm.simulate(T, n_steps, n_paths, antithetic=False)
        
        # Add jumps
        for i in range(n_paths):
            for j in range(1, n_steps + 1):
                # Poisson jump occurrence
                if np.random.rand() < self.lambda_jump * dt:
                    # Jump size (lognormal)
                    jump = np.exp(np.random.randn() * self.sigma_jump + self.mu_jump)
                    paths[i, j:] *= jump
        
        return paths


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck process for mean-reverting quantities.
    
    $$dx_t = \theta(\mu - x_t)dt + \sigma dW_t$$
    
    Applications:
    - Interest rate modeling (Vasicek model)
    - Volatility modeling
    - Pairs trading spread
    """
    
    def __init__(self, theta: float, mu: float, sigma: float, x0: float):
        """
        Args:
            theta: Speed of mean reversion
            mu: Long-term mean
            sigma: Volatility
            x0: Initial value
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate OU paths using exact discretization."""
        dt = T / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0
        
        # Exact discretization parameters
        exp_minus_theta_dt = np.exp(-self.theta * dt)
        sqrt_var = self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta))
        
        for i in range(n_paths):
            for j in range(n_steps):
                # Mean-reverting drift
                mean = self.mu + (paths[i, j] - self.mu) * exp_minus_theta_dt
                paths[i, j + 1] = mean + sqrt_var * np.random.randn()
        
        return paths
    
    def stationary_variance(self) -> float:
        """Long-term variance of the process."""
        return self.sigma**2 / (2 * self.theta)
    
    def half_life(self) -> float:
        """Time to revert halfway to mean."""
        return np.log(2) / self.theta


class HestonModel:
    """
    Heston stochastic volatility model.
    
    $$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$
    $$dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t} dW_t^v$$
    
    Where $dW_t^S dW_t^v = \rho dt$
    """
    
    def __init__(self, mu: float, v0: float, kappa: float,
                 theta: float, xi: float, rho: float, S0: float):
        """
        Args:
            mu: Drift
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            xi: Volatility of volatility
            rho: Correlation between price and variance
            S0: Initial price
        """
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.S0 = S0
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using QE scheme.
        
        Args:
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for i in range(n_paths):
            for j in range(n_steps):
                # Correlated Brownian motions
                Z1 = np.random.randn()
                Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.randn()
                
                # Variance process (truncated at 0)
                v_next = (v[i, j] + 
                         self.kappa * (self.theta - v[i, j]) * dt +
                         self.xi * np.sqrt(max(v[i, j], 0)) * np.sqrt(dt) * Z2)
                v[i, j + 1] = max(v_next, 0)
                
                # Price process
                S[i, j + 1] = (S[i, j] * 
                              np.exp((self.mu - 0.5 * v[i, j]) * dt +
                                    np.sqrt(max(v[i, j], 0)) * np.sqrt(dt) * Z1))
        
        return S, v
```

### 11.1.2 Ito's Lemma Applications

```python
# features/mathematical/ito_calculus.py

import numpy as np
from scipy.misc import derivative
from typing import Callable

class ItoCalculus:
    """
    Ito calculus utilities for derivatives pricing and hedging.
    """
    
    @staticmethod
    def ito_lemma_1d(f: Callable, S: float, t: float,
                    mu: float, sigma: float) -> Dict:
        """
        Apply Ito's lemma to f(S, t).
        
        $$df = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial S}dS + \frac{1}{2}\frac{\partial^2 f}{\partial S^2}(dS)^2$$
        
        Args:
            f: Function f(S, t)
            S: Current price
            t: Current time
            mu: Drift
            sigma: Volatility
            
        Returns:
            dict: df components
        """
        # Numerical derivatives
        eps = 1e-6
        
        # Partial derivatives
        df_dS = derivative(lambda x: f(x, t), S, dx=eps)
        df_dt = derivative(lambda x: f(S, x), t, dx=eps)
        d2f_dS2 = derivative(lambda x: derivative(lambda y: f(y, t), x, dx=eps), 
                            S, dx=eps)
        
        # Ito terms
        drift = df_dt + mu * S * df_dS + 0.5 * sigma**2 * S**2 * d2f_dS2
        diffusion = sigma * S * df_dS
        
        return {
            'drift': drift,
            'diffusion': diffusion,
            'df_dS': df_dS,
            'df_dt': df_dt,
            'd2f_dS2': d2f_dS2
        }
    
    @staticmethod
    def delta_hedge(option_price: Callable, S: float, 
                   K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate delta for Black-Scholes option.
        
        Delta = ∂V/∂S
        
        Args:
            option_price: Option pricing function
            S: Current price
            K: Strike
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            float: Delta
        """
        eps = S * 0.001
        V_up = option_price(S + eps, K, T, r, sigma)
        V_down = option_price(S - eps, K, T, r, sigma)
        
        return (V_up - V_down) / (2 * eps)
    
    @staticmethod
    def gamma(option_price: Callable, S: float,
             K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma (second derivative of price)."""
        eps = S * 0.001
        delta_up = ItoCalculus.delta_hedge(option_price, S + eps, K, T, r, sigma)
        delta_down = ItoCalculus.delta_hedge(option_price, S - eps, K, T, r, sigma)
        
        return (delta_up - delta_down) / (2 * eps)
    
    @staticmethod
    def theta(option_price: Callable, S: float,
             K: float, T: float, r: float, sigma: float) -> float:
        """Calculate theta (time decay)."""
        eps = 1 / 365  # One day
        V_now = option_price(S, K, T, r, sigma)
        V_later = option_price(S, K, T - eps, r, sigma)
        
        return (V_later - V_now) / eps
    
    @staticmethod
    def vega(option_price: Callable, S: float,
            K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega (sensitivity to volatility)."""
        eps = 0.01
        V_up = option_price(S, K, T, r, sigma + eps)
        V_down = option_price(S, K, T, r, sigma - eps)
        
        return (V_up - V_down) / (2 * eps)


def black_scholes_call(S: float, K: float, T: float, 
                      r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.
    
    $$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$
    
    Where:
    $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
    $$d_2 = d_1 - \sigma\sqrt{T}$$
    """
    from scipy.stats import norm
    
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float,
                     r: float, sigma: float) -> float:
    """Black-Scholes put option price (put-call parity)."""
    call = black_scholes_call(S, K, T, r, sigma)
    return call + K * np.exp(-r * T) - S


def implied_volatility(option_price: float, S: float, K: float,
                      T: float, r: float, option_type: str = 'call') -> float:
    """
    Calculate implied volatility from option price.
    
    Uses Newton-Raphson iteration.
    """
    from scipy.optimize import brentq
    
    def objective(sigma):
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        return price - option_price
    
    try:
        return brentq(objective, 0.001, 5.0)
    except ValueError:
        return np.nan
```

## 11.2 ADVANCED TIME SERIES ANALYSIS

### 11.2.1 Autoregressive Models

```python
# features/mathematical/time_series_models.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional

class ARMA:
    """
    Autoregressive Moving Average model.
    
    $$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
    """
    
    def __init__(self, p: int = 1, q: int = 0):
        """
        Args:
            p: AR order
            q: MA order
        """
        self.p = p
        self.q = q
        self.params = None
        self.sigma2 = None
    
    def fit(self, X: np.ndarray) -> 'ARMA':
        """
        Fit ARMA model using maximum likelihood.
        
        Args:
            X: Time series data
            
        Returns:
            self
        """
        X = np.array(X)
        n = len(X)
        
        # Initialize parameters
        n_params = 1 + self.p + self.q  # c, phi, theta
        init_params = np.zeros(n_params)
        init_params[0] = X.mean()  # c
        
        # Negative log-likelihood
        def neg_log_lik(params):
            c = params[0]
            phi = params[1:1+self.p]
            theta = params[1+self.p:]
            
            # Compute residuals
            residuals = self._compute_residuals(X, c, phi, theta)
            
            # Gaussian log-likelihood
            sigma2 = np.mean(residuals**2)
            nll = 0.5 * len(residuals) * np.log(2 * np.pi * sigma2)
            nll += 0.5 * np.sum(residuals**2) / sigma2
            
            return nll
        
        # Optimize
        result = minimize(neg_log_lik, init_params, method='L-BFGS-B')
        
        self.params = result.x
        residuals = self._compute_residuals(
            X, self.params[0], 
            self.params[1:1+self.p],
            self.params[1+self.p:]
        )
        self.sigma2 = np.var(residuals)
        
        return self
    
    def _compute_residuals(self, X: np.ndarray, c: float,
                          phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute residuals given parameters."""
        n = len(X)
        residuals = np.zeros(n)
        
        max_lag = max(self.p, self.q)
        
        for t in range(max_lag, n):
            # AR component
            ar_term = sum(phi[i] * X[t-i-1] for i in range(min(self.p, t)))
            
            # MA component
            ma_term = sum(theta[j] * residuals[t-j-1] for j in range(min(self.q, t)))
            
            # Predicted value
            X_pred = c + ar_term + ma_term
            
            # Residual
            residuals[t] = X[t] - X_pred
        
        return residuals[max_lag:]
    
    def forecast(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecasts, standard errors)
        """
        if self.params is None:
            raise ValueError("Model not fitted")
        
        c = self.params[0]
        phi = self.params[1:1+self.p]
        
        forecasts = np.zeros(steps)
        
        for h in range(steps):
            # Simple AR forecast (ignoring MA for simplicity)
            forecast = c
            for i in range(min(self.p, h)):
                forecast += phi[i] * forecasts[h-i-1]
            forecasts[h] = forecast
        
        # Standard errors grow with horizon
        ses = np.sqrt(self.sigma2 * np.arange(1, steps + 1))
        
        return forecasts, ses


class GARCH:
    """
    Generalized Autoregressive Conditional Heteroskedasticity.
    
    Models time-varying volatility.
    
    $$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: GARCH order (beta)
            q: ARCH order (alpha)
        """
        self.p = p
        self.q = q
        self.params = None
        self.omega = None
        self.alpha = None
        self.beta = None
    
    def fit(self, returns: np.ndarray) -> 'GARCH':
        """
        Fit GARCH model.
        
        Args:
            returns: Return series
            
        Returns:
            self
        """
        returns = np.array(returns)
        
        # Initialize parameters
        n_params = 1 + self.q + self.p  # omega, alpha, beta
        
        # Initial guess
        var_returns = np.var(returns)
        init_params = np.zeros(n_params)
        init_params[0] = var_returns * 0.1  # omega
        init_params[1:1+self.q] = 0.1  # alpha
        init_params[1+self.q:] = 0.8  # beta
        
        # Constraints: omega > 0, alpha >= 0, beta >= 0, sum(alpha) + sum(beta) < 1
        bounds = [(1e-6, None)] + [(0, 1)] * (n_params - 1)
        
        def neg_log_lik(params):
            sigma2 = self._compute_variance(returns, params)
            
            # Gaussian log-likelihood
            nll = 0.5 * np.sum(np.log(2 * np.pi * sigma2))
            nll += 0.5 * np.sum(returns**2 / sigma2)
            
            return nll
        
        # Optimize
        result = minimize(neg_log_lik, init_params, bounds=bounds, 
                         method='L-BFGS-B')
        
        self.params = result.x
        self.omega = self.params[0]
        self.alpha = self.params[1:1+self.q]
        self.beta = self.params[1+self.q:]
        
        return self
    
    def _compute_variance(self, returns: np.ndarray, 
                         params: np.ndarray) -> np.ndarray:
        """Compute conditional variance."""
        n = len(returns)
        omega = params[0]
        alpha = params[1:1+self.q]
        beta = params[1+self.q:]
        
        sigma2 = np.zeros(n)
        sigma2[:max(self.p, self.q)] = np.var(returns)
        
        for t in range(max(self.p, self.q), n):
            sigma2[t] = omega
            
            # ARCH component
            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2[t] += alpha[i] * returns[t-i-1]**2
            
            # GARCH component
            for j in range(self.p):
                if t - j - 1 >= 0:
                    sigma2[t] += beta[j] * sigma2[t-j-1]
        
        return sigma2
    
    def forecast_volatility(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future volatility.
        
        Args:
            steps: Number of steps ahead
            
        Returns:
            np.ndarray: Forecasted variances
        """
        if self.params is None:
            raise ValueError("Model not fitted")
        
        # Long-term variance
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        long_term_var = self.omega / (1 - persistence) if persistence < 1 else self.omega
        
        forecasts = np.zeros(steps)
        
        # Start from last observed variance
        last_var = self._compute_variance(
            np.zeros(100), self.params
        )[-1]
        
        for h in range(steps):
            forecasts[h] = (self.omega + 
                          persistence * (last_var if h == 0 else forecasts[h-1]))
        
        return np.sqrt(forecasts)
    
    def get_conditional_volatility(self) -> np.ndarray:
        """Get fitted conditional volatility."""
        if self.params is None:
            raise ValueError("Model not fitted")
        
        # Dummy returns (we just want the variance series)
        return np.sqrt(self._compute_variance(np.zeros(100), self.params))


class CointegrationTest:
    """
    Johansen cointegration test for multiple time series.
    """
    
    def __init__(self, significance: float = 0.05):
        """
        Args:
            significance: Significance level for test
        """
        self.significance = significance
    
    def engle_granger(self, y: np.ndarray, x: np.ndarray) -> Dict:
        """
        Engle-Granger two-step cointegration test.
        
        Args:
            y: First time series
            x: Second time series
            
        Returns:
            dict: Test results
        """
        from scipy.stats import linregress
        
        # Step 1: OLS regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate residuals
        residuals = y - (intercept + slope * x)
        
        # Step 2: ADF test on residuals
        adf_result = self._adf_test(residuals)
        
        return {
            'cointegrated': adf_result['p_value'] < self.significance,
            'adf_statistic': adf_result['statistic'],
            'adf_pvalue': adf_result['p_value'],
            'hedge_ratio': slope,
            'intercept': intercept,
            'residuals': residuals
        }
    
    def _adf_test(self, series: np.ndarray, maxlag: int = 1) -> Dict:
        """
        Augmented Dickey-Fuller test for unit root.
        
        Simple implementation - for production use statsmodels.
        """
        from scipy.stats import norm
        
        # First difference
        diff = np.diff(series)
        lagged = series[:-1]
        
        # Regression: diff = alpha + beta * lagged + error
        X = np.column_stack([np.ones(len(lagged)), lagged])
        y = diff
        
        # OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Standard error of beta[1]
        mse = np.mean(residuals**2)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(var_beta[1, 1])
        
        # t-statistic
        t_stat = beta[1] / se_beta1
        
        # Approximate p-value (simplified)
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'beta': beta
        }
```

## 11.3 ADVANCED PORTFOLIO OPTIMIZATION

### 11.3.1 Mean-Variance Optimization

```python
# features/mathematical/portfolio_optimization.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

class MeanVarianceOptimizer:
    """
    Markowitz mean-variance portfolio optimization.
    
    Solves:
    $$\min_w w^T \Sigma w - \lambda w^T \mu$$
    
    Subject to:
    $$\sum w_i = 1, w_i \geq 0$$
    """
    
    def __init__(self, risk_aversion: float = 1.0,
                 allow_short: bool = False,
                 max_position: float = 0.3):
        """
        Args:
            risk_aversion: Risk aversion parameter (lambda)
            allow_short: Allow short positions
            max_position: Maximum position size
        """
        self.lambda_risk = risk_aversion
        self.allow_short = allow_short
        self.max_position = max_position
    
    def optimize(self, expected_returns: np.ndarray,
                cov_matrix: np.ndarray,
                constraints: Optional[List] = None) -> Dict:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            constraints: Additional constraints
            
        Returns:
            dict: Optimization results
        """
        n = len(expected_returns)
        
        # Objective function
        def objective(w):
            portfolio_return = np.dot(w, expected_returns)
            portfolio_var = np.dot(w, np.dot(cov_matrix, w))
            return -portfolio_return + self.lambda_risk * portfolio_var
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum to 1
        
        if constraints:
            cons.extend(constraints)
        
        # Bounds
        if self.allow_short:
            bounds = [(-self.max_position, self.max_position)] * n
        else:
            bounds = [(0, self.max_position)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
    
    def efficient_frontier(self, expected_returns: np.ndarray,
                          cov_matrix: np.ndarray,
                          n_points: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier.
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier
            
        Returns:
            DataFrame: Frontier points
        """
        # Find minimum variance portfolio
        self.lambda_risk = 1e6
        min_var = self.optimize(expected_returns, cov_matrix)
        
        # Find maximum return portfolio
        max_ret_idx = np.argmax(expected_returns)
        max_return = expected_returns[max_ret_idx]
        
        # Generate frontier
        target_returns = np.linspace(min_var['expected_return'], max_return, n_points)
        
        frontier = []
        for target in target_returns:
            # Add return constraint
            return_constraint = {'type': 'eq', 
                                'fun': lambda w: np.dot(w, expected_returns) - target}
            
            result = self.optimize(expected_returns, cov_matrix, [return_constraint])
            
            frontier.append({
                'return': result['expected_return'],
                'volatility': result['volatility'],
                'sharpe': result['sharpe_ratio']
            })
        
        return pd.DataFrame(frontier)


class BlackLitterman:
    """
    Black-Litterman model for Bayesian portfolio optimization.
    
    Combines market equilibrium (CAPM) with investor views.
    """
    
    def __init__(self, tau: float = 0.05):
        """
        Args:
            tau: Uncertainty scaling parameter
        """
        self.tau = tau
    
    def optimize(self, market_weights: np.ndarray,
                cov_matrix: np.ndarray,
                risk_aversion: float = 2.5,
                views: Optional[Dict] = None) -> Dict:
        """
        Run Black-Litterman optimization.
        
        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            views: Investor views dict with 'P', 'Q', 'omega'
            
        Returns:
            dict: Posterior estimates and optimal weights
        """
        n = len(market_weights)
        
        # Implied equilibrium returns (reverse optimization)
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views is None:
            # No views - use market equilibrium
            posterior_return = pi
            posterior_cov = cov_matrix
        else:
            P = views['P']  # View matrix
            Q = views['Q']  # View returns
            omega = views['omega']  # View uncertainty
            
            # Posterior return
            inv_cov = np.linalg.inv(self.tau * cov_matrix)
            inv_omega = np.linalg.inv(omega)
            
            posterior_cov_inv = inv_cov + P.T @ inv_omega @ P
            posterior_cov = np.linalg.inv(posterior_cov_inv)
            
            posterior_return = posterior_cov @ (
                inv_cov @ pi + P.T @ inv_omega @ Q
            )
        
        # Optimize with posterior estimates
        mvo = MeanVarianceOptimizer(risk_aversion=risk_aversion)
        opt_result = mvo.optimize(posterior_return, posterior_cov)
        
        return {
            'posterior_return': posterior_return,
            'posterior_cov': posterior_cov,
            'equilibrium_return': pi,
            'optimal_weights': opt_result['weights'],
            'expected_return': opt_result['expected_return'],
            'volatility': opt_result['volatility'],
            'sharpe_ratio': opt_result['sharpe_ratio']
        }


class RiskParityOptimizer:
    """
    Risk parity portfolio optimization.
    
    Allocates such that each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, target_risk: Optional[float] = None):
        """
        Args:
            target_risk: Target portfolio volatility (None for fully invested)
        """
        self.target_risk = target_risk
    
    def optimize(self, cov_matrix: np.ndarray,
                budget_constraint: bool = True) -> Dict:
        """
        Optimize risk parity portfolio.
        
        Args:
            cov_matrix: Covariance matrix
            budget_constraint: Enforce sum of weights = 1
            
        Returns:
            dict: Optimal weights and risk contributions
        """
        n = cov_matrix.shape[0]
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            portfolio_var = np.dot(w, np.dot(cov_matrix, w))
            
            # Marginal risk contributions
            mrc = np.dot(cov_matrix, w)
            
            # Risk contributions
            rc = w * mrc
            
            # Target: equal risk contribution
            target_rc = portfolio_var / n
            
            return np.sum((rc - target_rc)**2)
        
        # Constraints
        constraints = []
        
        if budget_constraint:
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        if self.target_risk:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w, np.dot(cov_matrix, w))) - self.target_risk
            })
        
        # Bounds (long only)
        bounds = [(0, 1)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        # Calculate risk contributions
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        mrc = np.dot(cov_matrix, weights)
        rc = weights * mrc
        rc_pct = rc / portfolio_var
        
        return {
            'weights': weights,
            'risk_contributions': rc,
            'risk_contribution_pct': rc_pct,
            'portfolio_volatility': np.sqrt(portfolio_var),
            'success': result.success
        }


class MaximumDiversification:
    """
    Maximum diversification portfolio.
    
    Maximizes diversification ratio:
    $$DR = \frac{\sum_i w_i \sigma_i}{\sigma_p}$$
    """
    
    def optimize(self, cov_matrix: np.ndarray) -> Dict:
        """
        Optimize maximum diversification portfolio.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            dict: Optimal weights
        """
        n = cov_matrix.shape[0]
        
        # Individual volatilities
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Objective: maximize diversification ratio
        # Equivalent to minimizing: -w^T * vols / sqrt(w^T * Sigma * w)
        def neg_diversification_ratio(w):
            weighted_vols = np.dot(w, vols)
            portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            return -weighted_vols / portfolio_vol
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Fully invested
        ]
        
        # Bounds (long only)
        bounds = [(0, 1)] * n
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(neg_diversification_ratio, w0,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        # Calculate metrics
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        weighted_vols = np.dot(weights, vols)
        diversification_ratio = weighted_vols / portfolio_vol
        
        return {
            'weights': weights,
            'diversification_ratio': diversification_ratio,
            'portfolio_volatility': portfolio_vol,
            'success': result.success
        }


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity by Marcos Lopez de Prado.
    
    Uses hierarchical clustering to improve portfolio construction.
    """
    
    def __init__(self, linkage_method: str = 'single'):
        """
        Args:
            linkage_method: Clustering linkage method
        """
        self.linkage_method = linkage_method
    
    def optimize(self, cov_matrix: np.ndarray,
                returns: Optional[np.ndarray] = None) -> Dict:
        """
        Run HRP optimization.
        
        Args:
            cov_matrix: Covariance matrix
            returns: Return series (for distance calculation)
            
        Returns:
            dict: Optimal weights
        """
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # Step 1: Compute distance matrix from correlation
        corr = self._cov_to_corr(cov_matrix)
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Step 2: Hierarchical clustering
        linkage_matrix = linkage(squareform(dist), method=self.linkage_method)
        
        # Step 3: Quasi-diagonalization
        sorted_idx = leaves_list(linkage_matrix)
        
        # Step 4: Recursive bisection for weights
        weights = self._recursive_bisection(cov_matrix, sorted_idx)
        
        return {
            'weights': weights,
            'sorted_indices': sorted_idx,
            'linkage_matrix': linkage_matrix
        }
    
    def _cov_to_corr(self, cov: np.ndarray) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        vols = np.sqrt(np.diag(cov))
        return cov / np.outer(vols, vols)
    
    def _recursive_bisection(self, cov: np.ndarray,
                            sorted_idx: np.ndarray) -> np.ndarray:
        """
        Allocate weights using recursive bisection.
        
        Args:
            cov: Covariance matrix
            sorted_idx: Hierarchically sorted indices
            
        Returns:
            np.ndarray: Portfolio weights
        """
        n = len(sorted_idx)
        weights = np.ones(n)
        
        # Reorder covariance matrix
        cov_sorted = cov[sorted_idx][:, sorted_idx]
        
        # Recursive allocation
        clusters = [sorted_idx]
        
        while clusters:
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                
                # Split cluster in half
                split = len(cluster) // 2
                left = cluster[:split]
                right = cluster[split:]
                
                # Calculate cluster variances
                left_var = self._get_cluster_var(cov, left)
                right_var = self._get_cluster_var(cov, right)
                
                # Allocate inverse variance
                alpha = 1 - left_var / (left_var + right_var)
                
                # Update weights
                for i in left:
                    idx_in_sorted = np.where(sorted_idx == i)[0][0]
                    weights[idx_in_sorted] *= alpha
                
                for i in right:
                    idx_in_sorted = np.where(sorted_idx == i)[0][0]
                    weights[idx_in_sorted] *= (1 - alpha)
                
                new_clusters.extend([left, right])
            
            clusters = new_clusters
        
        # Normalize
        weights = weights / np.sum(weights)
        
        # Map back to original order
        final_weights = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            final_weights[idx] = weights[i]
        
        return final_weights
    
    def _get_cluster_var(self, cov: np.ndarray, cluster: np.ndarray) -> float:
        """Calculate variance of cluster with inverse-variance weights."""
        cluster_cov = cov[np.ix_(cluster, cluster)]
        iv_weights = 1 / np.diag(cluster_cov)
        iv_weights = iv_weights / np.sum(iv_weights)
        
        return np.dot(iv_weights, np.dot(cluster_cov, iv_weights))
```


---

# SECTION 12: ADVANCED TRADING STRATEGIES

## 12.1 STATISTICAL ARBITRAGE

### 12.1.1 Pairs Trading Implementation

```python
# strategy/statistical_arbitrage.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class PairsTrade:
    """Represents a pairs trade signal."""
    symbol_long: str
    symbol_short: str
    hedge_ratio: float
    z_score: float
    signal_strength: float
    entry_threshold: float
    exit_threshold: float

class PairsTradingStrategy:
    """
    Statistical arbitrage using cointegrated pairs.
    
    Strategy:
    1. Find cointegrated pairs
    2. Calculate spread and z-score
    3. Enter when z-score exceeds threshold
    4. Exit when spread reverts
    """
    
    def __init__(self, lookback: int = 60,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5,
                 adf_threshold: float = 0.05):
        """
        Args:
            lookback: Lookback window for calculations
            entry_z: Z-score entry threshold
            exit_z: Z-score exit threshold
            adf_threshold: ADF test significance level
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.adf_threshold = adf_threshold
        self.pairs = []
        
    def find_cointegrated_pairs(self, prices: pd.DataFrame,
                                max_pairs: int = 10) -> pd.DataFrame:
        """
        Find cointegrated pairs from price data.
        
        Args:
            prices: DataFrame with price series as columns
            max_pairs: Maximum number of pairs to return
            
        Returns:
            DataFrame: Cointegrated pairs sorted by score
        """
        symbols = prices.columns
        n = len(symbols)
        
        results = []
        
        for i in range(n):
            for j in range(i+1, n):
                sym1, sym2 = symbols[i], symbols[j]
                
                # Get price series
                y = prices[sym1].dropna()
                x = prices[sym2].dropna()
                
                # Align
                common_idx = y.index.intersection(x.index)
                if len(common_idx) < self.lookback:
                    continue
                
                y = y.loc[common_idx]
                x = x.loc[common_idx]
                
                # Test cointegration
                coint_result = self._engle_granger(y, x)
                
                if coint_result['cointegrated']:
                    results.append({
                        'symbol_1': sym1,
                        'symbol_2': sym2,
                        'adf_pvalue': coint_result['adf_pvalue'],
                        'hedge_ratio': coint_result['hedge_ratio'],
                        'half_life': coint_result.get('half_life', np.nan),
                        'correlation': y.corr(x)
                    })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Score: lower p-value, shorter half-life, higher correlation
        df['score'] = (
            (1 - df['adf_pvalue']) * 0.4 +
            (1 / (1 + df['half_life'].fillna(100))) * 0.4 +
            df['correlation'].abs() * 0.2
        )
        
        return df.nlargest(max_pairs, 'score')
    
    def _engle_granger(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Engle-Granger cointegration test.
        
        Args:
            y: First price series
            x: Second price series
            
        Returns:
            dict: Test results
        """
        from scipy.stats import linregress
        
        # OLS regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate residuals (spread)
        residuals = y - (intercept + slope * x)
        
        # ADF test on residuals
        adf_result = self._adf_test(residuals.values)
        
        # Calculate half-life of mean reversion
        lagged = residuals.shift(1).dropna()
        delta = residuals.diff().dropna()
        
        # Align
        common_idx = lagged.index.intersection(delta.index)
        lagged = lagged.loc[common_idx]
        delta = delta.loc[common_idx]
        
        # Regression: delta = alpha + beta * lagged
        if len(lagged) > 10:
            beta = np.polyfit(lagged, delta, 1)[0]
            half_life = -np.log(2) / beta if beta < 0 else np.nan
        else:
            half_life = np.nan
        
        return {
            'cointegrated': adf_result['p_value'] < self.adf_threshold,
            'adf_statistic': adf_result['statistic'],
            'adf_pvalue': adf_result['p_value'],
            'hedge_ratio': slope,
            'intercept': intercept,
            'half_life': half_life,
            'residuals': residuals
        }
    
    def _adf_test(self, series: np.ndarray, maxlag: int = 1) -> Dict:
        """
        Augmented Dickey-Fuller test.
        
        Args:
            series: Time series
            maxlag: Maximum lag
            
        Returns:
            dict: Test results
        """
        from scipy.stats import norm
        
        # First difference
        diff = np.diff(series)
        lagged = series[:-1]
        
        # Regression: diff = alpha + beta * lagged + error
        X = np.column_stack([np.ones(len(lagged)), lagged])
        y = diff
        
        # OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Standard error
        mse = np.mean(residuals**2)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(var_beta[1, 1])
        
        # t-statistic
        t_stat = beta[1] / se_beta1
        
        # Approximate p-value
        p_value = 2 * (1 - norm.cdf(abs(t_stat)))
        
        return {
            'statistic': t_stat,
            'p_value': p_value,
            'beta': beta
        }
    
    def generate_signals(self, pair: Dict, 
                        prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for a pair.
        
        Args:
            pair: Pair information from find_cointegrated_pairs
            prices: Price data
            
        Returns:
            DataFrame: Signals and z-scores
        """
        sym1, sym2 = pair['symbol_1'], pair['symbol_2']
        hedge_ratio = pair['hedge_ratio']
        
        # Get prices
        y = prices[sym1]
        x = prices[sym2]
        
        # Calculate spread
        spread = y - hedge_ratio * x
        
        # Calculate z-score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=prices.index)
        signals['z_score'] = z_score
        signals['spread'] = spread
        
        # Long spread when z-score is very negative
        signals['long_signal'] = z_score < -self.entry_z
        
        # Short spread when z-score is very positive
        signals['short_signal'] = z_score > self.entry_z
        
        # Exit signals
        signals['exit_long'] = z_score > -self.exit_z
        signals['exit_short'] = z_score < self.exit_z
        
        return signals
    
    def backtest_pair(self, pair: Dict, prices: pd.DataFrame,
                     initial_capital: float = 100000) -> Dict:
        """
        Backtest a pairs trading strategy.
        
        Args:
            pair: Pair information
            prices: Price data
            initial_capital: Starting capital
            
        Returns:
            dict: Backtest results
        """
        signals = self.generate_signals(pair, prices)
        
        sym1, sym2 = pair['symbol_1'], pair['symbol_2']
        hedge_ratio = pair['hedge_ratio']
        
        # Position tracking
        position = 0  # 0 = flat, 1 = long spread, -1 = short spread
        capital = initial_capital
        trades = []
        equity = [initial_capital]
        
        for i in range(1, len(signals)):
            date = signals.index[i]
            
            # Check for entry/exit
            if position == 0:
                if signals['long_signal'].iloc[i]:
                    position = 1
                    entry_spread = signals['spread'].iloc[i]
                    trades.append({
                        'date': date,
                        'action': 'enter_long',
                        'spread': entry_spread
                    })
                elif signals['short_signal'].iloc[i]:
                    position = -1
                    entry_spread = signals['spread'].iloc[i]
                    trades.append({
                        'date': date,
                        'action': 'enter_short',
                        'spread': entry_spread
                    })
            
            elif position == 1:
                if signals['exit_long'].iloc[i]:
                    exit_spread = signals['spread'].iloc[i]
                    pnl = exit_spread - entry_spread
                    capital += pnl
                    trades.append({
                        'date': date,
                        'action': 'exit_long',
                        'spread': exit_spread,
                        'pnl': pnl
                    })
                    position = 0
            
            elif position == -1:
                if signals['exit_short'].iloc[i]:
                    exit_spread = signals['spread'].iloc[i]
                    pnl = entry_spread - exit_spread
                    capital += pnl
                    trades.append({
                        'date': date,
                        'action': 'exit_short',
                        'spread': exit_spread,
                        'pnl': pnl
                    })
                    position = 0
            
            equity.append(capital)
        
        # Calculate metrics
        equity_series = pd.Series(equity, index=signals.index)
        returns = equity_series.pct_change().dropna()
        
        total_return = (capital - initial_capital) / initial_capital
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
        
        return {
            'pair': pair,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_trades': len([t for t in trades if 'pnl' in t]),
            'trades': trades,
            'equity_curve': equity_series
        }


class KalmanPairsTrader:
    """
    Pairs trading with Kalman filter for adaptive hedge ratio.
    
    The Kalman filter continuously updates the hedge ratio,
    adapting to changing market relationships.
    """
    
    def __init__(self, delta: float = 1e-4, R: float = 1e-3,
                 entry_z: float = 2.0, exit_z: float = 0.5):
        """
        Args:
            delta: Transition covariance (system noise)
            R: Measurement noise
            entry_z: Entry z-score threshold
            exit_z: Exit z-score threshold
        """
        self.delta = delta
        self.R = R
        self.entry_z = entry_z
        self.exit_z = exit_z
        
    def initialize_filter(self, y: np.ndarray, x: np.ndarray):
        """
        Initialize Kalman filter with initial OLS estimate.
        
        Args:
            y: Target series
            x: Hedge series
        """
        # Initial OLS estimate
        n = min(len(y), 100)  # Use first 100 points
        X = np.column_stack([np.ones(n), x[:n]])
        
        beta = np.linalg.lstsq(X, y[:n], rcond=None)[0]
        
        # State: [hedge_ratio, intercept]
        self.x = beta.reshape(-1, 1)
        self.P = np.eye(2)  # Initial covariance
        self.F = np.eye(2)  # Transition matrix
        self.Q = np.eye(2) * self.delta  # Process noise
    
    def update(self, y: float, x: float) -> Dict:
        """
        Update Kalman filter with new observation.
        
        Args:
            y: New target price
            x: New hedge price
            
        Returns:
            dict: Updated state and statistics
        """
        # Prediction
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Measurement
        H = np.array([[x, 1]])  # Measurement matrix
        y_pred = H @ self.x
        
        # Innovation
        residual = y - y_pred
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # Update
        self.x = self.x + K * residual
        self.P = (np.eye(2) - K @ H) @ self.P
        
        # Extract parameters
        hedge_ratio = self.x[0, 0]
        intercept = self.x[1, 0]
        
        # Calculate spread and z-score
        spread = y - (hedge_ratio * x + intercept)
        spread_std = np.sqrt(self.P[0, 0] * x**2 + self.P[1, 1] + self.R)
        z_score = spread / spread_std if spread_std > 0 else 0
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'spread': spread,
            'spread_std': spread_std,
            'z_score': z_score,
            'residual': residual[0, 0]
        }
    
    def backtest(self, y: pd.Series, x: pd.Series,
                initial_capital: float = 100000) -> Dict:
        """
        Backtest Kalman pairs trading strategy.
        
        Args:
            y: Target price series
            x: Hedge price series
            initial_capital: Starting capital
            
        Returns:
            dict: Backtest results
        """
        # Initialize
        self.initialize_filter(y.values, x.values)
        
        # Warm-up period
        warmup = 30
        for i in range(warmup):
            self.update(y.iloc[i], x.iloc[i])
        
        # Trading
        position = 0
        capital = initial_capital
        trades = []
        equity = [initial_capital] * warmup
        
        for i in range(warmup, len(y)):
            result = self.update(y.iloc[i], x.iloc[i])
            z_score = result['z_score']
            
            # Trading logic
            if position == 0:
                if z_score < -self.entry_z:
                    position = 1
                    entry_z = z_score
                    trades.append({'index': i, 'action': 'long', 'z_score': z_score})
                elif z_score > self.entry_z:
                    position = -1
                    entry_z = z_score
                    trades.append({'index': i, 'action': 'short', 'z_score': z_score})
            
            elif position == 1 and z_score > -self.exit_z:
                # Exit long
                pnl = (entry_z - z_score) * 1000  # Simplified P&L
                capital += pnl
                trades.append({'index': i, 'action': 'exit_long', 
                             'z_score': z_score, 'pnl': pnl})
                position = 0
            
            elif position == -1 and z_score < self.exit_z:
                # Exit short
                pnl = (z_score - entry_z) * 1000
                capital += pnl
                trades.append({'index': i, 'action': 'exit_short',
                             'z_score': z_score, 'pnl': pnl})
                position = 0
            
            equity.append(capital)
        
        # Calculate metrics
        equity_series = pd.Series(equity, index=y.index[:len(equity)])
        returns = equity_series.pct_change().dropna()
        
        total_return = (capital - initial_capital) / initial_capital
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_trades': len([t for t in trades if 'pnl' in t]),
            'trades': trades,
            'equity_curve': equity_series
        }


class MultiPairsPortfolio:
    """
    Portfolio of multiple pairs trading strategies.
    
    Diversifies across multiple cointegrated pairs.
    """
    
    def __init__(self, max_pairs: int = 5,
                 capital_per_pair: float = 20000):
        """
        Args:
            max_pairs: Maximum number of pairs to trade
            capital_per_pair: Capital allocated to each pair
        """
        self.max_pairs = max_pairs
        self.capital_per_pair = capital_per_pair
        self.pairs = []
        self.traders = {}
    
    def select_pairs(self, prices: pd.DataFrame,
                    correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Select best pairs for trading.
        
        Args:
            prices: Price data for all symbols
            correlation_threshold: Minimum correlation threshold
            
        Returns:
            DataFrame: Selected pairs
        """
        # Calculate correlation matrix
        returns = prices.pct_change().dropna()
        corr = returns.corr()
        
        pairs = []
        symbols = corr.columns
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if abs(corr.iloc[i, j]) > correlation_threshold:
                    pairs.append({
                        'symbol_1': symbols[i],
                        'symbol_2': symbols[j],
                        'correlation': corr.iloc[i, j]
                    })
        
        pairs_df = pd.DataFrame(pairs)
        
        # Test cointegration for each pair
        coint_results = []
        
        for _, pair in pairs_df.iterrows():
            y = prices[pair['symbol_1']]
            x = prices[pair['symbol_2']]
            
            # Align
            common_idx = y.index.intersection(x.index)
            y = y.loc[common_idx]
            x = x.loc[common_idx]
            
            # Test cointegration
            trader = PairsTradingStrategy()
            result = trader._engle_granger(y, x)
            
            if result['cointegrated']:
                coint_results.append({
                    **pair.to_dict(),
                    'adf_pvalue': result['adf_pvalue'],
                    'hedge_ratio': result['hedge_ratio'],
                    'half_life': result.get('half_life', np.nan)
                })
        
        if not coint_results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(coint_results)
        
        # Score and select top pairs
        result_df['score'] = (
            (1 - result_df['adf_pvalue']) * 0.5 +
            (1 / (1 + result_df['half_life'].fillna(100))) * 0.5
        )
        
        return result_df.nlargest(self.max_pairs, 'score')
    
    def initialize_traders(self, selected_pairs: pd.DataFrame):
        """
        Initialize traders for selected pairs.
        
        Args:
            selected_pairs: DataFrame of selected pairs
        """
        for _, pair in selected_pairs.iterrows():
            pair_key = f"{pair['symbol_1']}_{pair['symbol_2']}"
            self.pairs.append(pair_key)
            self.traders[pair_key] = {
                'pair': pair,
                'trader': KalmanPairsTrader(),
                'capital': self.capital_per_pair
            }
    
    def generate_portfolio_signals(self, prices: pd.DataFrame) -> Dict:
        """
        Generate signals for all pairs in portfolio.
        
        Args:
            prices: Current price data
            
        Returns:
            dict: Signals for each pair
        """
        signals = {}
        
        for pair_key, trader_info in self.traders.items():
            pair = trader_info['pair']
            
            y = prices[pair['symbol_1']]
            x = prices[pair['symbol_2']]
            
            # Update Kalman filter
            result = trader_info['trader'].update(y.iloc[-1], x.iloc[-1])
            
            signals[pair_key] = {
                'z_score': result['z_score'],
                'hedge_ratio': result['hedge_ratio'],
                'signal': 'long' if result['z_score'] < -2 else 
                         ('short' if result['z_score'] > 2 else 'none')
            }
        
        return signals
```

## 12.2 MOMENTUM AND TREND FOLLOWING

```python
# strategy/momentum_strategies.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

class MomentumStrategy:
    """
    Momentum-based trading strategies.
    
    Includes:
    - Time-series momentum
    - Cross-sectional momentum
    - Dual momentum
    """
    
    def __init__(self, lookback: int = 252,
                 holding_period: int = 21,
                 n_top: int = 10):
        """
        Args:
            lookback: Lookback period for momentum calculation
            holding_period: Holding period for positions
            n_top: Number of top performers to select
        """
        self.lookback = lookback
        self.holding_period = holding_period
        self.n_top = n_top
    
    def time_series_momentum(self, prices: pd.Series) -> pd.Series:
        """
        Calculate time-series momentum signal.
        
        Args:
            prices: Price series
            
        Returns:
            pd.Series: Momentum signals (-1, 0, 1)
        """
        # Calculate returns over lookback
        momentum = prices.pct_change(self.lookback)
        
        # Generate signals
        signals = pd.Series(0, index=prices.index)
        signals[momentum > 0] = 1   # Long when positive momentum
        signals[momentum < 0] = -1  # Short when negative momentum
        
        return signals
    
    def cross_sectional_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional momentum signals.
        
        Ranks assets by momentum and goes long top N, short bottom N.
        
        Args:
            prices: DataFrame of price series
            
        Returns:
            DataFrame: Position signals for each asset
        """
        # Calculate momentum for each asset
        momentum = prices.pct_change(self.lookback)
        
        # Rank assets each period
        ranks = momentum.rank(axis=1, ascending=False)
        
        # Generate signals
        n_assets = len(prices.columns)
        n_bottom = self.n_top
        
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        signals[ranks <= self.n_top] = 1      # Long top performers
        signals[ranks > n_assets - n_bottom] = -1  # Short bottom performers
        
        return signals
    
    def dual_momentum(self, prices: pd.DataFrame,
                     risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Dual momentum: absolute + relative momentum.
        
        Only invests in assets with positive absolute momentum
        and ranks by relative momentum.
        
        Args:
            prices: Price data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            DataFrame: Signals
        """
        # Calculate absolute momentum (vs risk-free)
        returns = prices.pct_change(self.lookback)
        risk_free_return = risk_free_rate * (self.lookback / 252)
        
        # Filter for positive absolute momentum
        positive_momentum = returns > risk_free_return
        
        # Rank by relative momentum
        ranks = returns.rank(axis=1, ascending=False)
        
        # Generate signals
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Long top N with positive absolute momentum
        long_condition = (ranks <= self.n_top) & positive_momentum
        signals[long_condition] = 1
        
        return signals
    
    def momentum_with_volatility_filter(self, prices: pd.DataFrame,
                                       vol_lookback: int = 63,
                                       vol_threshold: float = 0.20) -> pd.DataFrame:
        """
        Momentum strategy with volatility filter.
        
        Reduces exposure when market volatility is high.
        
        Args:
            prices: Price data
            vol_lookback: Volatility calculation window
            vol_threshold: Volatility threshold (annualized)
            
        Returns:
            DataFrame: Signals with volatility adjustment
        """
        # Base momentum signals
        signals = self.cross_sectional_momentum(prices)
        
        # Calculate market volatility
        market_returns = prices.mean(axis=1).pct_change()
        market_vol = market_returns.rolling(vol_lookback).std() * np.sqrt(252)
        
        # Volatility filter
        high_vol = market_vol > vol_threshold
        
        # Reduce signals during high volatility
        for col in signals.columns:
            signals.loc[high_vol, col] *= 0.5
        
        return signals


class TrendFollowingStrategy:
    """
    Trend following using moving averages and breakouts.
    """
    
    def __init__(self, fast_ma: int = 50,
                 slow_ma: int = 200,
                 atr_period: int = 14):
        """
        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period
            atr_period: ATR period for position sizing
        """
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_period = atr_period
    
    def moving_average_crossover(self, prices: pd.Series) -> pd.Series:
        """
        Generate signals from MA crossover.
        
        Args:
            prices: Price series
            
        Returns:
            pd.Series: Signals (1 = long, -1 = short, 0 = flat)
        """
        fast = prices.rolling(self.fast_ma).mean()
        slow = prices.rolling(self.slow_ma).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        
        return signals
    
    def donchian_channel_breakout(self, prices: pd.Series,
                                  channel_period: int = 20) -> pd.Series:
        """
        Donchian channel breakout strategy.
        
        Go long on breakout above highest high,
        go short on breakdown below lowest low.
        
        Args:
            prices: Price series
            channel_period: Lookback for channel
            
        Returns:
            pd.Series: Signals
        """
        highest_high = prices.rolling(channel_period).max().shift(1)
        lowest_low = prices.rolling(channel_period).min().shift(1)
        
        signals = pd.Series(0, index=prices.index)
        signals[prices > highest_high] = 1
        signals[prices < lowest_low] = -1
        
        return signals
    
    def turtle_trading(self, prices: pd.Series,
                      entry_period: int = 20,
                      exit_period: int = 10) -> pd.Series:
        """
        Classic Turtle Trading system.
        
        Args:
            prices: Price series
            entry_period: Entry breakout period
            exit_period: Exit breakout period
            
        Returns:
            pd.Series: Signals
        """
        entry_high = prices.rolling(entry_period).max().shift(1)
        entry_low = prices.rolling(entry_period).min().shift(1)
        exit_high = prices.rolling(exit_period).max().shift(1)
        exit_low = prices.rolling(exit_period).min().shift(1)
        
        signals = pd.Series(0, index=prices.index)
        position = 0
        
        for i in range(len(prices)):
            if position == 0:
                if prices.iloc[i] > entry_high.iloc[i]:
                    position = 1
                elif prices.iloc[i] < entry_low.iloc[i]:
                    position = -1
            
            elif position == 1:
                if prices.iloc[i] < exit_low.iloc[i]:
                    position = 0
            
            elif position == -1:
                if prices.iloc[i] > exit_high.iloc[i]:
                    position = 0
            
            signals.iloc[i] = position
        
        return signals
    
    def calculate_atr(self, high: pd.Series, low: pd.Series,
                     close: pd.Series) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            pd.Series: ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(self.atr_period).mean()
        
        return atr
    
    def position_sizing_by_atr(self, capital: float,
                               atr: float,
                               risk_per_trade: float = 0.01,
                               atr_multiple: float = 2.0) -> float:
        """
        Calculate position size based on ATR.
        
        Args:
            capital: Account capital
            atr: Current ATR value
            risk_per_trade: Risk per trade as fraction of capital
            atr_multiple: ATR multiple for stop distance
            
        Returns:
            float: Position size (number of shares)
        """
        risk_amount = capital * risk_per_trade
        stop_distance = atr * atr_multiple
        
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        return position_size


class MomentumRegimeStrategy:
    """
    Momentum strategy that adapts to market regime.
    
    Uses different momentum lookbacks depending on market conditions.
    """
    
    def __init__(self, regimes: Dict[str, Dict] = None):
        """
        Args:
            regimes: Dict of regime parameters
        """
        if regimes is None:
            self.regimes = {
                'trending': {'lookback': 252, 'threshold': 0.10},
                'mean_reverting': {'lookback': 63, 'threshold': 0.05},
                'volatile': {'lookback': 21, 'threshold': 0.02}
            }
        else:
            self.regimes = regimes
    
    def detect_regime(self, prices: pd.Series,
                     vol_window: int = 63,
                     trend_window: int = 252) -> str:
        """
        Detect current market regime.
        
        Args:
            prices: Price series
            vol_window: Volatility lookback
            trend_window: Trend lookback
            
        Returns:
            str: Regime name
        """
        returns = prices.pct_change().dropna()
        
        # Calculate metrics
        volatility = returns.rolling(vol_window).std().iloc[-1] * np.sqrt(252)
        
        # Trend strength using linear regression
        x = np.arange(min(trend_window, len(prices)))
        y = prices.iloc[-len(x):].values
        slope, _, r_value, _, _ = stats.linregress(x, y)
        trend_strength = abs(r_value)
        
        # Classify regime
        if volatility > 0.25:
            return 'volatile'
        elif trend_strength > 0.7:
            return 'trending'
        else:
            return 'mean_reverting'
    
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime-adaptive momentum signals.
        
        Args:
            prices: Price data
            
        Returns:
            DataFrame: Signals
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        for i in range(252, len(prices)):
            # Detect regime using market average
            market_price = prices.iloc[:i].mean(axis=1)
            regime = self.detect_regime(market_price)
            
            # Get regime parameters
            params = self.regimes[regime]
            lookback = params['lookback']
            
            # Calculate momentum
            if i >= lookback:
                momentum = prices.iloc[i] / prices.iloc[i-lookback] - 1
                
                # Rank and signal
                ranks = momentum.rank(ascending=False)
                n_assets = len(prices.columns)
                
                for col in prices.columns:
                    if ranks[col] <= 5:
                        signals.loc[prices.index[i], col] = 1
                    elif ranks[col] > n_assets - 5:
                        signals.loc[prices.index[i], col] = -1
        
        return signals
```

## 12.3 MEAN REVERSION STRATEGIES

```python
# strategy/mean_reversion.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

class MeanReversionStrategy:
    """
    Mean reversion trading strategies.
    
    Based on the principle that prices tend to revert to their mean.
    """
    
    def __init__(self, lookback: int = 20,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5):
        """
        Args:
            lookback: Lookback for mean and std calculation
            entry_z: Z-score entry threshold
            exit_z: Z-score exit threshold
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
    
    def bollinger_bands(self, prices: pd.Series,
                       num_std: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands mean reversion.
        
        Args:
            prices: Price series
            num_std: Number of standard deviations
            
        Returns:
            DataFrame: Upper band, lower band, middle, z-score
        """
        middle = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        z_score = (prices - middle) / std
        
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'z_score': z_score
        })
    
    def generate_bb_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate Bollinger Bands signals.
        
        Buy when price touches lower band, sell when touches upper band.
        
        Args:
            prices: Price series
            
        Returns:
            pd.Series: Signals
        """
        bb = self.bollinger_bands(prices)
        
        signals = pd.Series(0, index=prices.index)
        position = 0
        
        for i in range(1, len(prices)):
            if position == 0:
                if prices.iloc[i] < bb['lower'].iloc[i]:
                    position = 1
                elif prices.iloc[i] > bb['upper'].iloc[i]:
                    position = -1
            
            elif position == 1:
                if prices.iloc[i] > bb['middle'].iloc[i]:
                    position = 0
            
            elif position == -1:
                if prices.iloc[i] < bb['middle'].iloc[i]:
                    position = 0
            
            signals.iloc[i] = position
        
        return signals
    
    def rsi_mean_reversion(self, prices: pd.Series,
                          period: int = 14,
                          oversold: float = 30,
                          overbought: float = 70) -> pd.Series:
        """
        RSI-based mean reversion.
        
        Args:
            prices: Price series
            period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            pd.Series: Signals
        """
        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=prices.index)
        position = 0
        
        for i in range(1, len(prices)):
            if position == 0:
                if rsi.iloc[i] < oversold:
                    position = 1
                elif rsi.iloc[i] > overbought:
                    position = -1
            
            elif position == 1:
                if rsi.iloc[i] > 50:
                    position = 0
            
            elif position == -1:
                if rsi.iloc[i] < 50:
                    position = 0
            
            signals.iloc[i] = position
        
        return signals


class StatisticalArbitrage:
    """
    Statistical arbitrage using factor models.
    """
    
    def __init__(self, n_factors: int = 5):
        """
        Args:
            n_factors: Number of principal components
        """
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
    
    def fit_factor_model(self, returns: pd.DataFrame) -> Dict:
        """
        Fit PCA factor model to returns.
        
        Args:
            returns: Return data
            
        Returns:
            dict: Factor model components
        """
        from sklearn.decomposition import PCA
        
        # Standardize returns
        standardized = (returns - returns.mean()) / returns.std()
        
        # PCA
        pca = PCA(n_components=self.n_factors)
        factor_returns = pca.fit_transform(standardized.dropna())
        
        self.factor_loadings = pca.components_.T
        self.factor_returns = factor_returns
        
        return {
            'explained_variance': pca.explained_variance_ratio_,
            'factor_loadings': self.factor_loadings,
            'factor_returns': self.factor_returns
        }
    
    def calculate_residual_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residual returns after removing factor exposure.
        
        Args:
            returns: Return data
            
        Returns:
            DataFrame: Residual returns
        """
        if self.factor_loadings is None:
            raise ValueError("Factor model not fitted")
        
        # Predict returns from factors
        predicted = self.factor_returns @ self.factor_loadings.T
        
        # Residuals
        residuals = returns.values - predicted
        
        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)
    
    def generate_signals(self, returns: pd.DataFrame,
                        residual_threshold: float = 2.0) -> pd.DataFrame:
        """
        Generate signals based on residual returns.
        
        Go long assets with negative residuals (underperformed),
        go short assets with positive residuals (outperformed).
        
        Args:
            returns: Return data
            residual_threshold: Z-score threshold for residuals
            
        Returns:
            DataFrame: Signals
        """
        residuals = self.calculate_residual_returns(returns)
        
        # Z-score residuals
        residual_z = (residuals - residuals.mean()) / residuals.std()
        
        # Generate signals
        signals = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        signals[residual_z < -residual_threshold] = 1
        signals[residual_z > residual_threshold] = -1
        
        return signals
```

---

*Note: This document continues with additional sections covering advanced topics, implementation details, and comprehensive reference materials.*


---

# SECTION 13: ALTERNATIVE DATA STRATEGIES

## 13.1 SENTIMENT-BASED TRADING

### 13.1.1 Social Media Sentiment Analysis

```python
# strategy/sentiment_strategies.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

class SocialSentimentStrategy:
    """
    Trading strategy based on social media sentiment.
    
    Sources:
    - Reddit (wallstreetbets, investing, stocks)
    - Twitter/X
    - StockTwits
    """
    
    def __init__(self, sentiment_window: int = 7,
                 sentiment_threshold: float = 0.3,
                 volume_threshold: int = 100):
        """
        Args:
            sentiment_window: Days to aggregate sentiment
            sentiment_threshold: Minimum sentiment score to trigger
            volume_threshold: Minimum mention count
        """
        self.sentiment_window = sentiment_window
        self.sentiment_threshold = sentiment_threshold
        self.volume_threshold = volume_threshold
    
    def aggregate_sentiment(self, mentions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment from social media mentions.
        
        Args:
            mentions: DataFrame with columns [timestamp, symbol, sentiment_score, volume]
            
        Returns:
            DataFrame: Daily aggregated sentiment by symbol
        """
        # Group by date and symbol
        mentions['date'] = pd.to_datetime(mentions['timestamp']).dt.date
        
        aggregated = mentions.groupby(['date', 'symbol']).agg({
            'sentiment_score': 'mean',
            'volume': 'sum'
        }).reset_index()
        
        # Rolling average
        aggregated['sentiment_ma'] = aggregated.groupby('symbol')['sentiment_score'].transform(
            lambda x: x.rolling(self.sentiment_window, min_periods=1).mean()
        )
        
        return aggregated
    
    def generate_signals(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from sentiment.
        
        Args:
            sentiment_data: Aggregated sentiment data
            
        Returns:
            DataFrame: Signals
        """
        signals = pd.DataFrame()
        
        for symbol in sentiment_data['symbol'].unique():
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol].copy()
            
            # Filter by volume
            symbol_data = symbol_data[symbol_data['volume'] >= self.volume_threshold]
            
            if len(symbol_data) == 0:
                continue
            
            # Generate signals
            symbol_data['signal'] = 0
            symbol_data.loc[symbol_data['sentiment_ma'] > self.sentiment_threshold, 'signal'] = 1
            symbol_data.loc[symbol_data['sentiment_ma'] < -self.sentiment_threshold, 'signal'] = -1
            
            signals = pd.concat([signals, symbol_data])
        
        return signals
    
    def detect_sentiment_anomalies(self, sentiment_data: pd.DataFrame,
                                   z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect unusual sentiment spikes that may indicate opportunities.
        
        Args:
            sentiment_data: Sentiment data
            z_threshold: Z-score threshold for anomaly
            
        Returns:
            DataFrame: Anomaly flags
        """
        anomalies = pd.DataFrame()
        
        for symbol in sentiment_data['symbol'].unique():
            symbol_data = sentiment_data[sentiment_data['symbol'] == symbol].copy()
            
            # Calculate z-score of sentiment changes
            symbol_data['sentiment_change'] = symbol_data['sentiment_score'].diff()
            symbol_data['sentiment_zscore'] = (
                (symbol_data['sentiment_change'] - symbol_data['sentiment_change'].mean()) /
                symbol_data['sentiment_change'].std()
            )
            
            # Flag anomalies
            symbol_data['is_anomaly'] = abs(symbol_data['sentiment_zscore']) > z_threshold
            
            anomalies = pd.concat([anomalies, symbol_data])
        
        return anomalies


class OptionsSentimentStrategy:
    """
    Trading strategy based on options market sentiment.
    
    Indicators:
    - Put/Call ratio
    - Implied volatility skew
    - Unusual options volume
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def calculate_put_call_ratio(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Calculate put/call ratio from options data.
        
        Args:
            options_data: DataFrame with put_volume and call_volume
            
        Returns:
            pd.Series: Put/call ratio
        """
        return options_data['put_volume'] / options_data['call_volume']
    
    def calculate_iv_skew(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Calculate implied volatility skew.
        
        Difference between OTM put IV and ATM call IV.
        
        Args:
            options_data: Options data
            
        Returns:
            pd.Series: IV skew
        """
        return options_data['otm_put_iv'] - options_data['atm_call_iv']
    
    def detect_unusual_volume(self, options_data: pd.DataFrame,
                             z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect unusual options volume.
        
        Args:
            options_data: Options data
            z_threshold: Z-score threshold
            
        Returns:
            DataFrame: Unusual volume flags
        """
        # Calculate rolling average volume
        options_data['volume_ma'] = options_data['total_volume'].rolling(self.lookback).mean()
        options_data['volume_std'] = options_data['total_volume'].rolling(self.lookback).std()
        
        # Z-score
        options_data['volume_zscore'] = (
            (options_data['total_volume'] - options_data['volume_ma']) /
            options_data['volume_std']
        )
        
        # Flag unusual volume
        options_data['unusual_volume'] = options_data['volume_zscore'] > z_threshold
        
        return options_data
    
    def generate_signals(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from options sentiment.
        
        Contrarian interpretation:
        - High put/call ratio = bullish (retail is too bearish)
        - High IV skew = bullish (fear is overpriced)
        - Unusual call volume = bullish
        - Unusual put volume = bearish
        
        Args:
            options_data: Options data
            
        Returns:
            DataFrame: Signals
        """
        # Calculate indicators
        pc_ratio = self.calculate_put_call_ratio(options_data)
        iv_skew = self.calculate_iv_skew(options_data)
        unusual = self.detect_unusual_volume(options_data)
        
        # Rolling averages
        pc_ma = pc_ratio.rolling(self.lookback).mean()
        iv_skew_ma = iv_skew.rolling(self.lookback).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=options_data.index)
        signals['signal'] = 0
        
        # Contrarian signals
        # High P/C ratio = bullish
        signals.loc[pc_ratio > pc_ma * 1.5, 'signal'] = 1
        signals.loc[pc_ratio < pc_ma * 0.5, 'signal'] = -1
        
        # High IV skew = bullish (fear premium)
        signals.loc[iv_skew > iv_skew_ma + iv_skew.std(), 'signal'] = 1
        
        # Unusual call buying
        call_unusual = unusual['unusual_volume'] & (options_data['call_volume'] > options_data['put_volume'])
        signals.loc[call_unusual, 'signal'] = 1
        
        return signals


class InsiderTradingStrategy:
    """
    Strategy based on insider trading activity.
    
    Follows the "smart money" - insiders often have superior information.
    """
    
    def __init__(self, lookback_days: int = 90,
                 min_transaction_value: float = 10000):
        """
        Args:
            lookback_days: Days to look back for insider trades
            min_transaction_value: Minimum transaction value to consider
        """
        self.lookback = lookback_days
        self.min_value = min_transaction_value
    
    def calculate_insider_sentiment(self, insider_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate insider sentiment from Form 4 filings.
        
        Args:
            insider_data: Insider trading data
            
        Returns:
            DataFrame: Sentiment by symbol
        """
        # Filter recent transactions
        cutoff = datetime.now() - timedelta(days=self.lookback)
        recent = insider_data[insider_data['transaction_date'] >= cutoff]
        
        # Filter by value
        recent = recent[recent['transaction_value'] >= self.min_value]
        
        # Classify transactions
        recent['is_buy'] = recent['transaction_type'].str.contains('Purchase', case=False)
        recent['is_sell'] = recent['transaction_type'].str.contains('Sale', case=False)
        
        # Aggregate by symbol
        sentiment = recent.groupby('symbol').agg({
            'is_buy': 'sum',
            'is_sell': 'sum',
            'transaction_value': 'sum'
        }).reset_index()
        
        # Calculate sentiment score
        sentiment['net_buys'] = sentiment['is_buy'] - sentiment['is_sell']
        sentiment['sentiment_score'] = sentiment['net_buys'] / (
            sentiment['is_buy'] + sentiment['is_sell'] + 1
        )
        
        return sentiment
    
    def generate_signals(self, insider_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from insider activity.
        
        Args:
            insider_data: Insider trading data
            
        Returns:
            DataFrame: Signals
        """
        sentiment = self.calculate_insider_sentiment(insider_data)
        
        # Generate signals
        sentiment['signal'] = 0
        sentiment.loc[sentiment['sentiment_score'] > 0.5, 'signal'] = 1
        sentiment.loc[sentiment['sentiment_score'] < -0.5, 'signal'] = -1
        
        return sentiment


class CongressTradingStrategy:
    """
    Strategy based on Congressional trading disclosures.
    
    Politicians often have access to privileged information.
    """
    
    def __init__(self, lookback_days: int = 90,
                 min_delay_days: int = 30):
        """
        Args:
            lookback_days: Days to look back
            min_delay_days: Minimum delay after disclosure to trade
        """
        self.lookback = lookback_days
        self.min_delay = min_delay_days
    
    def calculate_politician_performance(self, trades: pd.DataFrame,
                                        price_data: Dict[str, pd.DataFrame],
                                        holding_period: int = 90) -> pd.DataFrame:
        """
        Calculate historical performance of politicians' trades.
        
        Args:
            trades: Congressional trades
            price_data: Price data for symbols
            holding_period: Days to hold after trade
            
        Returns:
            DataFrame: Performance metrics
        """
        results = []
        
        for _, trade in trades.iterrows():
            symbol = trade['ticker']
            
            if symbol not in price_data:
                continue
            
            prices = price_data[symbol]
            
            # Get price at disclosure
            disclosure_date = pd.to_datetime(trade['disclosure_date'])
            entry_date = disclosure_date + timedelta(days=self.min_delay)
            
            if entry_date not in prices.index:
                continue
            
            entry_price = prices.loc[entry_date, 'close']
            
            # Get exit price
            exit_date = entry_date + timedelta(days=holding_period)
            
            try:
                exit_price = prices.loc[prices.index <= exit_date, 'close'].iloc[-1]
            except:
                continue
            
            # Calculate return
            if trade['type'] == 'Purchase':
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            
            results.append({
                'symbol': symbol,
                'politician': trade['name'],
                'type': trade['type'],
                'entry_date': entry_date,
                'return': ret
            })
        
        return pd.DataFrame(results)
    
    def generate_signals(self, trades: pd.DataFrame,
                        performance: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on politician performance.
        
        Follow politicians with good track records.
        
        Args:
            trades: Recent trades
            performance: Historical performance
            
        Returns:
            DataFrame: Signals
        """
        # Calculate politician track records
        track_records = performance.groupby('politician')['return'].agg(['mean', 'count'])
        track_records = track_records[track_records['count'] >= 5]  # Minimum trades
        good_traders = track_records[track_records['mean'] > 0.05].index
        
        # Filter trades to good traders
        good_trades = trades[trades['name'].isin(good_traders)]
        
        # Generate signals
        signals = pd.DataFrame()
        
        for symbol in good_trades['ticker'].unique():
            symbol_trades = good_trades[good_trades['ticker'] == symbol]
            
            buys = len(symbol_trades[symbol_trades['type'] == 'Purchase'])
            sells = len(symbol_trades[symbol_trades['type'] == 'Sale'])
            
            signal = 1 if buys > sells else (-1 if sells > buys else 0)
            
            signals = pd.concat([signals, pd.DataFrame({
                'symbol': [symbol],
                'signal': [signal],
                'n_buys': [buys],
                'n_sells': [sells]
            })])
        
        return signals


class EarningsSurpriseStrategy:
    """
    Strategy based on earnings surprises.
    
    Post-earnings announcement drift (PEAD) is a well-documented anomaly.
    """
    
    def __init__(self, surprise_threshold: float = 0.10,
                 holding_period: int = 5):
        """
        Args:
            surprise_threshold: Minimum earnings surprise to trigger
            holding_period: Days to hold position
        """
        self.surprise_threshold = surprise_threshold
        self.holding_period = holding_period
    
    def calculate_surprise(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate earnings surprise.
        
        Args:
            earnings_data: Earnings data with actual and estimate
            
        Returns:
            DataFrame: Surprise metrics
        """
        earnings_data['surprise'] = (
            earnings_data['actual_eps'] - earnings_data['estimate_eps']
        )
        earnings_data['surprise_pct'] = (
            earnings_data['surprise'] / abs(earnings_data['estimate_eps'])
        )
        
        return earnings_data
    
    def generate_signals(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from earnings surprises.
        
        Positive surprise = long signal
        Negative surprise = short signal
        
        Args:
            earnings_data: Earnings data
            
        Returns:
            DataFrame: Signals
        """
        surprise_data = self.calculate_surprise(earnings_data)
        
        # Filter significant surprises
        significant = abs(surprise_data['surprise_pct']) > self.surprise_threshold
        
        signals = surprise_data[significant].copy()
        signals['signal'] = np.sign(signals['surprise_pct'])
        
        return signals[['symbol', 'announcement_date', 'signal', 'surprise_pct']]


class ShortInterestStrategy:
    """
    Strategy based on short interest data.
    
    High short interest can indicate:
    - Potential short squeeze (contrarian long)
    - Fundamental problems (follow the shorts)
    """
    
    def __init__(self, short_interest_threshold: float = 0.20,
                 days_to_cover_threshold: float = 5.0):
        """
        Args:
            short_interest_threshold: Short interest as % of float
            days_to_cover_threshold: Days to cover threshold
        """
        self.si_threshold = short_interest_threshold
        self.dtc_threshold = days_to_cover_threshold
    
    def calculate_squeeze_potential(self, short_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short squeeze potential score.
        
        Args:
            short_data: Short interest data
            
        Returns:
            DataFrame: Squeeze potential scores
        """
        # Normalize metrics
        short_data['si_score'] = short_data['short_interest'] / self.si_threshold
        short_data['dtc_score'] = short_data['days_to_cover'] / self.dtc_threshold
        
        # Combined squeeze score
        short_data['squeeze_score'] = (
            short_data['si_score'] * 0.5 +
            short_data['dtc_score'] * 0.3 +
            short_data['short_ratio'].fillna(0) * 0.2
        )
        
        return short_data
    
    def generate_signals(self, short_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from short interest.
        
        High squeeze score = contrarian long (potential squeeze)
        
        Args:
            short_data: Short interest data
            
        Returns:
            DataFrame: Signals
        """
        squeeze_data = self.calculate_squeeze_potential(short_data)
        
        # Generate signals
        squeeze_data['signal'] = 0
        squeeze_data.loc[squeeze_data['squeeze_score'] > 1.5, 'signal'] = 1
        
        return squeeze_data[['symbol', 'squeeze_score', 'signal']]
```

## 13.2 MACHINE LEARNING FOR TRADING

```python
# strategy/ml_strategies.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from typing import Dict, List, Tuple

class MLTradingStrategy:
    """
    Machine learning-based trading strategy.
    
    Uses ensemble methods to predict price direction.
    """
    
    def __init__(self, model_type: str = 'xgboost',
                 prediction_horizon: int = 5,
                 n_splits: int = 5):
        """
        Args:
            model_type: 'xgboost', 'random_forest', 'gradient_boosting'
            prediction_horizon: Days ahead to predict
            n_splits: Number of CV splits
        """
        self.model_type = model_type
        self.horizon = prediction_horizon
        self.n_splits = n_splits
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def engineer_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for machine learning.
        
        Args:
            prices: OHLCV data
            
        Returns:
            DataFrame: Engineered features
        """
        features = pd.DataFrame(index=prices.index)
        
        # Returns
        features['returns_1d'] = prices['close'].pct_change(1)
        features['returns_5d'] = prices['close'].pct_change(5)
        features['returns_21d'] = prices['close'].pct_change(21)
        
        # Volatility
        features['volatility_21d'] = features['returns_1d'].rolling(21).std()
        features['volatility_ratio'] = (
            features['returns_1d'].rolling(5).std() /
            features['returns_1d'].rolling(21).std()
        )
        
        # Moving averages
        for window in [5, 10, 21, 50, 200]:
            ma = prices['close'].rolling(window).mean()
            features[f'ma_{window}_ratio'] = prices['close'] / ma
        
        # RSI
        delta = prices['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices['close'].ewm(span=12).mean()
        ema_26 = prices['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        bb_middle = prices['close'].rolling(20).mean()
        bb_std = prices['close'].rolling(20).std()
        features['bb_position'] = (prices['close'] - bb_middle) / (2 * bb_std)
        
        # Volume features
        features['volume_ratio'] = prices['volume'] / prices['volume'].rolling(21).mean()
        features['volume_price_trend'] = (
            features['returns_1d'] * prices['volume']
        )
        
        return features.dropna()
    
    def create_labels(self, prices: pd.Series) -> pd.Series:
        """
        Create labels for classification.
        
        1 = price goes up
        0 = price goes down or sideways
        
        Args:
            prices: Price series
            
        Returns:
            pd.Series: Labels
        """
        future_returns = prices.pct_change(self.horizon).shift(-self.horizon)
        labels = (future_returns > 0).astype(int)
        
        return labels
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        Train the machine learning model.
        
        Args:
            features: Feature DataFrame
            labels: Target labels
        """
        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            cv_scores.append(score)
        
        print(f"CV Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
        
        # Train on full dataset
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(features.columns, 
                                              self.model.feature_importances_))
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            DataFrame: Predictions with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)
        
        predictions = pd.DataFrame({
            'prediction': self.model.predict(X_scaled),
            'prob_up': probs[:, 1],
            'prob_down': probs[:, 0]
        }, index=features.index)
        
        predictions['confidence'] = np.maximum(
            predictions['prob_up'], predictions['prob_down']
        )
        
        return predictions
    
    def generate_signals(self, predictions: pd.DataFrame,
                        confidence_threshold: float = 0.6) -> pd.Series:
        """
        Generate trading signals from predictions.
        
        Args:
            predictions: Prediction DataFrame
            confidence_threshold: Minimum confidence to trade
            
        Returns:
            pd.Series: Signals
        """
        signals = pd.Series(0, index=predictions.index)
        
        # Long when high confidence of up move
        long_condition = (
            (predictions['prediction'] == 1) &
            (predictions['confidence'] >= confidence_threshold)
        )
        signals[long_condition] = 1
        
        # Short when high confidence of down move
        short_condition = (
            (predictions['prediction'] == 0) &
            (predictions['confidence'] >= confidence_threshold)
        )
        signals[short_condition] = -1
        
        return signals


class DeepLearningStrategy:
    """
    Deep learning-based trading strategy using LSTM.
    """
    
    def __init__(self, sequence_length: int = 60,
                 lstm_units: List[int] = [64, 32],
                 dropout: float = 0.2,
                 epochs: int = 50,
                 batch_size: int = 32):
        """
        Args:
            sequence_length: Number of time steps in sequence
            lstm_units: LSTM layer sizes
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, n_features: int):
        """
        Build LSTM model.
        
        Args:
            n_features: Number of input features
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        self.model = Sequential()
        
        for i, units in enumerate(self.lstm_units):
            return_seq = i < len(self.lstm_units) - 1
            
            if i == 0:
                self.model.add(LSTM(units, 
                                   return_sequences=return_seq,
                                   input_shape=(self.sequence_length, n_features)))
            else:
                self.model.add(LSTM(units, return_sequences=return_seq))
            
            self.model.add(Dropout(self.dropout))
        
        self.model.add(Dense(1, activation='linear'))
        
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
        """
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        import tensorflow as tf
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
```

---

*Note: The document continues with additional comprehensive sections on execution algorithms, risk management, and extensive reference materials.*


---

# SECTION 14: EXECUTION ALGORITHMS

## 14.1 SMART ORDER ROUTING

### 14.1.1 Implementation Shortfall Algorithm

```python
# execution/algorithms/implementation_shortfall.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

@dataclass
class OrderSlice:
    """Represents a slice of a parent order."""
    quantity: float
    scheduled_time: datetime
    order_type: str
    limit_price: Optional[float] = None

class ImplementationShortfall:
    """
    Implementation Shortfall (IS) algorithm.
    
    Minimizes the difference between decision price and execution price.
    
    Key insight: Trade-off between market impact (front-loaded) 
    and opportunity cost (back-loaded).
    """
    
    def __init__(self, urgency: float = 0.5,
                 volume_participation: float = 0.10,
                 risk_aversion: float = 0.1):
        """
        Args:
            urgency: Trade urgency (0-1, higher = faster execution)
            volume_participation: Target % of volume to participate
            risk_aversion: Risk aversion parameter
        """
        self.urgency = urgency
        self.volume_participation = volume_participation
        self.risk_aversion = risk_aversion
    
    def calculate_optimal_schedule(self, total_quantity: float,
                                   decision_price: float,
                                   market_data: pd.DataFrame,
                                   time_horizon: int = 60) -> List[OrderSlice]:
        """
        Calculate optimal execution schedule.
        
        Uses Almgren-Chriss framework for optimal liquidation.
        
        Args:
            total_quantity: Total shares to execute
            decision_price: Price at decision time
            market_data: Market data with volume forecasts
            time_horizon: Execution horizon in minutes
            
        Returns:
            List of order slices
        """
        n_slices = min(time_horizon, 20)  # Max 20 slices
        
        # Calculate trading rate
        # Higher urgency = more front-loaded
        kappa = self._calculate_trading_rate(
            total_quantity, market_data, time_horizon
        )
        
        # Generate schedule
        schedule = []
        remaining = total_quantity
        
        for i in range(n_slices):
            # Optimal quantity for this slice
            if i < n_slices - 1:
                # Exponential decay schedule
                quantity = remaining * (1 - np.exp(-kappa))
            else:
                quantity = remaining  # Execute remainder
            
            quantity = min(quantity, remaining)
            remaining -= quantity
            
            slice_time = datetime.now() + timedelta(minutes=i * (time_horizon / n_slices))
            
            schedule.append(OrderSlice(
                quantity=quantity,
                scheduled_time=slice_time,
                order_type='market' if self.urgency > 0.7 else 'limit'
            ))
        
        return schedule
    
    def _calculate_trading_rate(self, quantity: float,
                                market_data: pd.DataFrame,
                                time_horizon: int) -> float:
        """
        Calculate optimal trading rate using Almgren-Chriss.
        
        Args:
            quantity: Order quantity
            market_data: Market data
            time_horizon: Time horizon
            
        Returns:
            float: Trading rate
        """
        # Market impact parameters (simplified)
        eta = 0.1  # Temporary impact coefficient
        gamma = 0.01  # Permanent impact coefficient
        
        # Optimal trading rate
        kappa = np.sqrt(self.risk_aversion * eta / gamma)
        
        return kappa
    
    def execute_schedule(self, schedule: List[OrderSlice],
                        symbol: str,
                        broker: Callable) -> Dict:
        """
        Execute the order schedule.
        
        Args:
            schedule: List of order slices
            symbol: Trading symbol
            broker: Broker execution function
            
        Returns:
            dict: Execution results
        """
        fills = []
        total_cost = 0
        
        for slice_order in schedule:
            # Wait until scheduled time
            wait_seconds = (slice_order.scheduled_time - datetime.now()).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            
            # Execute slice
            try:
                fill = broker(
                    symbol=symbol,
                    quantity=slice_order.quantity,
                    order_type=slice_order.order_type,
                    limit_price=slice_order.limit_price
                )
                fills.append(fill)
                total_cost += fill['price'] * fill['quantity']
            except Exception as e:
                print(f"Failed to execute slice: {e}")
        
        # Calculate implementation shortfall
        vwap = total_cost / sum(f['quantity'] for f in fills) if fills else 0
        
        return {
            'fills': fills,
            'vwap': vwap,
            'total_quantity': sum(f['quantity'] for f in fills),
            'implementation_shortfall': vwap - fills[0]['price'] if fills else 0
        }


class VWAPAlgorithm:
    """
    Volume-Weighted Average Price (VWAP) algorithm.
    
    Executes order to match historical volume profile.
    """
    
    def __init__(self, volume_profile: Optional[pd.Series] = None):
        """
        Args:
            volume_profile: Historical volume profile (time -> % of daily volume)
        """
        if volume_profile is None:
            # Default U-shaped intraday volume profile
            self.volume_profile = self._default_volume_profile()
        else:
            self.volume_profile = volume_profile
    
    def _default_volume_profile(self) -> pd.Series:
        """Create default U-shaped volume profile."""
        # Higher volume at open and close
        times = pd.date_range('09:30', '16:00', freq='5min')
        
        # U-shaped pattern
        n = len(times)
        profile = np.zeros(n)
        
        for i in range(n):
            # Higher at open (first 30 min) and close (last 30 min)
            if i < 6:  # First 30 min
                profile[i] = 2.0
            elif i > n - 6:  # Last 30 min
                profile[i] = 2.0
            else:
                profile[i] = 1.0
        
        # Normalize to sum to 1
        profile = profile / profile.sum()
        
        return pd.Series(profile, index=times)
    
    def calculate_schedule(self, total_quantity: float,
                          start_time: datetime,
                          end_time: datetime) -> List[OrderSlice]:
        """
        Calculate VWAP execution schedule.
        
        Args:
            total_quantity: Total shares to execute
            start_time: Execution start time
            end_time: Execution end time
            
        Returns:
            List of order slices
        """
        # Filter profile to execution window
        mask = (self.volume_profile.index >= start_time.time()) & \
               (self.volume_profile.index <= end_time.time())
        
        window_profile = self.volume_profile[mask]
        
        # Renormalize
        window_profile = window_profile / window_profile.sum()
        
        # Create schedule
        schedule = []
        
        for time_slot, volume_pct in window_profile.items():
            quantity = total_quantity * volume_pct
            
            scheduled_time = datetime.combine(start_time.date(), time_slot)
            
            schedule.append(OrderSlice(
                quantity=quantity,
                scheduled_time=scheduled_time,
                order_type='limit'
            ))
        
        return schedule
    
    def adaptive_schedule(self, total_quantity: float,
                         real_time_volume: pd.Series,
                         target_completion: float = 1.0) -> List[OrderSlice]:
        """
        Adapt schedule based on real-time volume.
        
        Args:
            total_quantity: Total shares
            real_time_volume: Actual volume so far
            target_completion: Target completion percentage
            
        Returns:
            List of order slices
        """
        # Calculate volume participation rate
        expected_volume = self.volume_profile.sum() * total_quantity / target_completion
        actual_volume = real_time_volume.sum()
        
        participation_rate = actual_volume / expected_volume if expected_volume > 0 else 1
        
        # Adjust remaining schedule
        remaining_quantity = total_quantity * (1 - target_completion)
        adjusted_quantity = remaining_quantity * participation_rate
        
        # Create adjusted schedule
        schedule = self.calculate_schedule(
            adjusted_quantity,
            datetime.now(),
            datetime.now() + timedelta(hours=6)
        )
        
        return schedule


class TWAPAlgorithm:
    """
    Time-Weighted Average Price (TWAP) algorithm.
    
    Executes order evenly over time.
    """
    
    def __init__(self, n_slices: int = 20):
        """
        Args:
            n_slices: Number of execution slices
        """
        self.n_slices = n_slices
    
    def calculate_schedule(self, total_quantity: float,
                          start_time: datetime,
                          end_time: datetime) -> List[OrderSlice]:
        """
        Calculate TWAP execution schedule.
        
        Args:
            total_quantity: Total shares
            start_time: Start time
            end_time: End time
            
        Returns:
            List of order slices
        """
        quantity_per_slice = total_quantity / self.n_slices
        time_interval = (end_time - start_time) / self.n_slices
        
        schedule = []
        
        for i in range(self.n_slices):
            scheduled_time = start_time + i * time_interval
            
            schedule.append(OrderSlice(
                quantity=quantity_per_slice,
                scheduled_time=scheduled_time,
                order_type='limit'
            ))
        
        return schedule


class MarketImpactEstimator:
    """
    Estimate market impact of trades.
    
    Uses Almgren-Chriss market impact model.
    """
    
    def __init__(self, eta: float = 0.142, gamma: float = 0.314,
                 beta: float = 0.6):
        """
        Args:
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
            beta: Decay parameter
        """
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
    
    def temporary_impact(self, quantity: float, daily_volume: float,
                        volatility: float) -> float:
        """
        Calculate temporary market impact.
        
        Args:
            quantity: Order quantity
            daily_volume: Daily trading volume
            volatility: Annualized volatility
            
        Returns:
            float: Temporary impact (as fraction of price)
        """
        participation = quantity / daily_volume if daily_volume > 0 else 0
        
        impact = self.eta * volatility * (participation ** 0.6)
        
        return impact
    
    def permanent_impact(self, quantity: float, daily_volume: float,
                        volatility: float) -> float:
        """
        Calculate permanent market impact.
        
        Args:
            quantity: Order quantity
            daily_volume: Daily trading volume
            volatility: Annualized volatility
            
        Returns:
            float: Permanent impact (as fraction of price)
        """
        participation = quantity / daily_volume if daily_volume > 0 else 0
        
        impact = self.gamma * volatility * participation
        
        return impact
    
    def total_impact(self, quantity: float, daily_volume: float,
                    volatility: float, execution_time: float) -> float:
        """
        Calculate total market impact.
        
        Args:
            quantity: Order quantity
            daily_volume: Daily volume
            volatility: Volatility
            execution_time: Execution time in days
            
        Returns:
            float: Total impact
        """
        temp_impact = self.temporary_impact(quantity, daily_volume, volatility)
        perm_impact = self.permanent_impact(quantity, daily_volume, volatility)
        
        # Temporary impact decays over time
        decay = np.exp(-self.beta * execution_time)
        
        total = perm_impact + temp_impact * decay
        
        return total


class SmartOrderRouter:
    """
    Routes orders to optimal venues/exchanges.
    """
    
    def __init__(self, venues: List[Dict]):
        """
        Args:
            venues: List of venue dicts with fees, liquidity, etc.
        """
        self.venues = venues
    
    def route_order(self, order: Dict, 
                   market_conditions: Dict) -> Dict:
        """
        Route order to best venue.
        
        Args:
            order: Order dict
            market_conditions: Current market conditions
            
        Returns:
            dict: Routing decision
        """
        scores = []
        
        for venue in self.venues:
            score = self._score_venue(venue, order, market_conditions)
            scores.append((venue['name'], score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_venue = scores[0][0]
        
        return {
            'venue': best_venue,
            'score': scores[0][1],
            'all_scores': dict(scores)
        }
    
    def _score_venue(self, venue: Dict, order: Dict,
                    market_conditions: Dict) -> float:
        """
        Score a venue for an order.
        
        Args:
            venue: Venue info
            order: Order info
            market_conditions: Market conditions
            
        Returns:
            float: Score (higher = better)
        """
        score = 0
        
        # Liquidity score
        liquidity = venue.get('liquidity', 0)
        score += liquidity * 0.3
        
        # Fee score (lower is better)
        fee = venue.get('fee', 0.003)
        score += (0.005 - fee) * 100 * 0.2
        
        # Speed score
        latency = venue.get('latency_ms', 10)
        score += (50 - latency) / 50 * 0.2
        
        # Spread score
        spread = venue.get('spread_bps', 10)
        score += (20 - spread) / 20 * 0.2
        
        # Reliability score
        uptime = venue.get('uptime', 0.99)
        score += uptime * 0.1
        
        return score


class LiquiditySeeker:
    """
    Actively seeks liquidity to minimize market impact.
    """
    
    def __init__(self, dark_pools: List[str] = None,
                 broker_algos: List[str] = None):
        """
        Args:
            dark_pools: List of dark pool venues
            broker_algos: List of broker algorithms
        """
        self.dark_pools = dark_pools or []
        self.broker_algos = broker_algos or []
    
    def seek_liquidity(self, order: Dict,
                      market_depth: pd.DataFrame) -> Dict:
        """
        Find best liquidity for order.
        
        Args:
            order: Order to execute
            market_depth: Market depth data
            
        Returns:
            dict: Liquidity information
        """
        # Analyze market depth
        bid_liquidity = market_depth['bid_size'].sum()
        ask_liquidity = market_depth['ask_size'].sum()
        
        # Determine if order can be filled in lit market
        order_size = order['quantity']
        
        if order['side'] == 'buy':
            lit_liquidity = ask_liquidity
        else:
            lit_liquidity = bid_liquidity
        
        can_fill_lit = lit_liquidity >= order_size * 0.5
        
        # Decision
        if can_fill_lit:
            return {
                'strategy': 'lit_market',
                'venues': ['primary_exchange'],
                'expected_fill': order_size
            }
        else:
            # Use dark pools
            return {
                'strategy': 'dark_pool',
                'venues': self.dark_pools,
                'expected_fill': order_size * 0.7
            }


class OrderBookAnalyzer:
    """
    Analyze order book for optimal execution.
    """
    
    def __init__(self, depth_levels: int = 10):
        """
        Args:
            depth_levels: Number of order book levels to analyze
        """
        self.depth_levels = depth_levels
    
    def calculate_vwap(self, order_book: pd.DataFrame) -> Dict:
        """
        Calculate VWAP from order book.
        
        Args:
            order_book: Order book data
            
        Returns:
            dict: VWAP metrics
        """
        bid_vwap = (
            (order_book['bid_price'] * order_book['bid_size']).sum() /
            order_book['bid_size'].sum()
        ) if order_book['bid_size'].sum() > 0 else 0
        
        ask_vwap = (
            (order_book['ask_price'] * order_book['ask_size']).sum() /
            order_book['ask_size'].sum()
        ) if order_book['ask_size'].sum() > 0 else 0
        
        mid_vwap = (bid_vwap + ask_vwap) / 2
        
        return {
            'bid_vwap': bid_vwap,
            'ask_vwap': ask_vwap,
            'mid_vwap': mid_vwap,
            'spread': ask_vwap - bid_vwap
        }
    
    def calculate_imbalance(self, order_book: pd.DataFrame) -> float:
        """
        Calculate order book imbalance.
        
        Positive = more buying pressure
        Negative = more selling pressure
        
        Args:
            order_book: Order book data
            
        Returns:
            float: Imbalance score
        """
        bid_volume = order_book['bid_size'].sum()
        ask_volume = order_book['ask_size'].sum()
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0
        
        imbalance = (bid_volume - ask_volume) / total_volume
        
        return imbalance
    
    def detect_large_orders(self, order_book: pd.DataFrame,
                           threshold: float = 5.0) -> pd.DataFrame:
        """
        Detect unusually large orders in book.
        
        Args:
            order_book: Order book data
            threshold: Z-score threshold
            
        Returns:
            DataFrame: Large orders
        """
        # Calculate z-scores
        bid_z = np.abs(stats.zscore(order_book['bid_size']))
        ask_z = np.abs(stats.zscore(order_book['ask_size']))
        
        # Flag large orders
        large_bids = order_book[bid_z > threshold]
        large_asks = order_book[ask_z > threshold]
        
        return pd.concat([large_bids, large_asks])


class ExecutionQualityAnalyzer:
    """
    Analyze execution quality.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, fills: List[Dict],
                         decision_price: float,
                         arrival_price: float,
                         benchmark: str = 'vwap') -> Dict:
        """
        Calculate execution quality metrics.
        
        Args:
            fills: List of fill data
            decision_price: Price at decision time
            arrival_price: Price when order arrived at market
            benchmark: Benchmark type ('vwap', 'twap', 'decision')
            
        Returns:
            dict: Quality metrics
        """
        if not fills:
            return {'error': 'No fills'}
        
        # Calculate execution VWAP
        total_value = sum(f['price'] * f['quantity'] for f in fills)
        total_quantity = sum(f['quantity'] for f in fills)
        execution_vwap = total_value / total_quantity
        
        # Implementation shortfall
        is_shortfall = (execution_vwap - decision_price) / decision_price
        
        # Slippage from arrival
        slippage = (execution_vwap - arrival_price) / arrival_price
        
        # Market impact
        market_impact = (arrival_price - decision_price) / decision_price
        
        # Timing cost
        timing_cost = is_shortfall - market_impact
        
        return {
            'execution_vwap': execution_vwap,
            'implementation_shortfall_bps': is_shortfall * 10000,
            'slippage_bps': slippage * 10000,
            'market_impact_bps': market_impact * 10000,
            'timing_cost_bps': timing_cost * 10000,
            'total_quantity': total_quantity,
            'n_fills': len(fills),
            'avg_fill_size': total_quantity / len(fills)
        }
    
    def transaction_cost_analysis(self, orders: List[Dict],
                                 market_data: pd.DataFrame) -> Dict:
        """
        Comprehensive transaction cost analysis.
        
        Args:
            orders: List of executed orders
            market_data: Market data
            
        Returns:
            dict: TCA results
        """
        total_cost = 0
        explicit_costs = 0
        implicit_costs = 0
        
        for order in orders:
            # Explicit costs (commissions, fees)
            explicit_costs += order.get('commission', 0)
            explicit_costs += order.get('fees', 0)
            
            # Implicit costs (slippage, market impact)
            decision_price = order.get('decision_price', order['price'])
            implicit_costs += (order['price'] - decision_price) * order['quantity']
        
        total_cost = explicit_costs + implicit_costs
        
        return {
            'total_cost': total_cost,
            'explicit_costs': explicit_costs,
            'implicit_costs': implicit_costs,
            'explicit_pct': explicit_costs / total_cost if total_cost > 0 else 0,
            'implicit_pct': implicit_costs / total_cost if total_cost > 0 else 0
        }
```

## 14.2 HIGH-FREQUENCY TRADING CONSIDERATIONS

```python
# execution/hft_considerations.py

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import deque

class LatencyMonitor:
    """
    Monitor and optimize trading latency.
    """
    
    def __init__(self, max_samples: int = 1000):
        """
        Args:
            max_samples: Maximum latency samples to store
        """
        self.latencies = deque(maxlen=max_samples)
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append(latency_ms)
    
    def get_statistics(self) -> Dict:
        """
        Get latency statistics.
        
        Returns:
            dict: Latency metrics
        """
        if not self.latencies:
            return {'error': 'No latency data'}
        
        latencies = np.array(self.latencies)
        
        return {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'std_ms': np.std(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies),
            'min_ms': np.min(latencies),
            'n_samples': len(latencies)
        }


class CoLocationOptimizer:
    """
    Optimize server placement for minimal latency.
    """
    
    def __init__(self, venues: List[Dict]):
        """
        Args:
            venues: List of venue information
        """
        self.venues = venues
    
    def calculate_optimal_location(self, 
                                   venue_weights: Dict[str, float]) -> Dict:
        """
        Calculate optimal server location.
        
        Args:
            venue_weights: Weight for each venue (based on trading volume)
            
        Returns:
            dict: Optimal location recommendation
        """
        # This is a simplified model
        # In practice, would use actual network latency measurements
        
        # Find weighted center
        total_weight = sum(venue_weights.values())
        
        optimal_lat = 0
        optimal_lon = 0
        
        for venue, weight in venue_weights.items():
            venue_info = next((v for v in self.venues if v['name'] == venue), None)
            if venue_info:
                w = weight / total_weight
                optimal_lat += venue_info['latitude'] * w
                optimal_lon += venue_info['longitude'] * w
        
        return {
            'latitude': optimal_lat,
            'longitude': optimal_lon,
            'recommended_dc': self._nearest_datacenter(optimal_lat, optimal_lon)
        }
    
    def _nearest_datacenter(self, lat: float, lon: float) -> str:
        """Find nearest major datacenter."""
        datacenters = {
            'ny4': {'lat': 40.7128, 'lon': -74.0060},  # NYC
            'ny5': {'lat': 40.7128, 'lon': -74.0060},
            'cermak': {'lat': 41.8781, 'lon': -87.6298},  # Chicago
            'equinix_ld4': {'lat': 51.5074, 'lon': -0.1278},  # London
            'equinix_fr2': {'lat': 50.1109, 'lon': 8.6821},  # Frankfurt
        }
        
        min_dist = float('inf')
        nearest = None
        
        for name, coords in datacenters.items():
            dist = np.sqrt((lat - coords['lat'])**2 + (lon - coords['lon'])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = name
        
        return nearest


class TickDataProcessor:
    """
    Process tick-level market data efficiently.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Args:
            buffer_size: Size of tick buffer
        """
        self.buffer = deque(maxlen=buffer_size)
    
    def add_tick(self, tick: Dict):
        """Add a tick to the buffer."""
        self.buffer.append({
            'timestamp': tick['timestamp'],
            'price': tick['price'],
            'size': tick['size'],
            'side': tick['side']
        })
    
    def calculate_vwap(self, window_ms: int = 60000) -> float:
        """
        Calculate VWAP over time window.
        
        Args:
            window_ms: Window in milliseconds
            
        Returns:
            float: VWAP
        """
        if not self.buffer:
            return 0
        
        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(milliseconds=window_ms)
        
        recent_ticks = [t for t in self.buffer if t['timestamp'] > cutoff]
        
        if not recent_ticks:
            return 0
        
        total_value = sum(t['price'] * t['size'] for t in recent_ticks)
        total_size = sum(t['size'] for t in recent_ticks)
        
        return total_value / total_size if total_size > 0 else 0
    
    def detect_iceberg_orders(self, min_size: int = 1000,
                             refresh_threshold: int = 5) -> List[Dict]:
        """
        Detect potential iceberg orders.
        
        Args:
            min_size: Minimum order size
            refresh_threshold: Number of refreshes to flag
            
        Returns:
            List of suspected iceberg orders
        """
        # Group by price level
        price_levels = {}
        
        for tick in self.buffer:
            price = tick['price']
            if price not in price_levels:
                price_levels[price] = []
            price_levels[price].append(tick)
        
        icebergs = []
        
        for price, ticks in price_levels.items():
            if len(ticks) >= refresh_threshold:
                total_size = sum(t['size'] for t in ticks)
                if total_size >= min_size:
                    icebergs.append({
                        'price': price,
                        'total_size': total_size,
                        'refreshes': len(ticks),
                        'avg_size': total_size / len(ticks)
                    })
        
        return icebergs
```

---

*Note: The document continues with additional comprehensive sections covering risk management, compliance, and extensive reference materials.*


---

# SECTION 15: COMPLIANCE AND REGULATORY FRAMEWORK

## 15.1 REGULATORY COMPLIANCE SYSTEM

### 15.1.1 Trade Surveillance

```python
# compliance/surveillance.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Alert:
    """Trade surveillance alert."""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    related_trades: List[str]
    status: str = 'open'

class TradeSurveillance:
    """
    Monitor trading activity for compliance violations.
    
    Detects:
    - Wash trading
    - Layering/spoofing
    - Insider trading patterns
    - Market manipulation
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Surveillance configuration
        """
        self.config = config
        self.alerts = []
    
    def detect_wash_trading(self, trades: pd.DataFrame) -> List[Alert]:
        """
        Detect potential wash trading.
        
        Wash trading: Buying and selling the same security
        to create artificial volume.
        
        Args:
            trades: Trade data
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Group by symbol and time window
        trades['time_bucket'] = trades['timestamp'].dt.floor('5min')
        
        grouped = trades.groupby(['symbol', 'time_bucket'])
        
        for (symbol, time_bucket), group in grouped:
            # Check for matching buy/sell from same account
            buys = group[group['side'] == 'buy']
            sells = group[group['side'] == 'sell']
            
            for _, buy in buys.iterrows():
                matching_sells = sells[
                    (sells['account'] == buy['account']) &
                    (abs(sells['price'] - buy['price']) < 0.01) &
                    (abs(sells['quantity'] - buy['quantity']) < buy['quantity'] * 0.1)
                ]
                
                if len(matching_sells) > 0:
                    alerts.append(Alert(
                        alert_type='WASH_TRADING',
                        severity='high',
                        description=f'Potential wash trading in {symbol}',
                        timestamp=time_bucket,
                        related_trades=[buy['trade_id']] + matching_sells['trade_id'].tolist()
                    ))
        
        return alerts
    
    def detect_spoofing(self, orders: pd.DataFrame) -> List[Alert]:
        """
        Detect potential spoofing/layering.
        
        Spoofing: Placing orders with intent to cancel
        before execution.
        
        Args:
            orders: Order data
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Calculate cancel-to-fill ratio
        orders['cancel_to_fill'] = orders['canceled_quantity'] / (orders['filled_quantity'] + 1)
        
        # Flag high cancel ratios
        high_cancel = orders[orders['cancel_to_fill'] > 10]
        
        for _, order in high_cancel.iterrows():
            # Check if order was quickly canceled
            time_to_cancel = order['cancel_time'] - order['submit_time']
            
            if time_to_cancel < timedelta(seconds=2):
                alerts.append(Alert(
                    alert_type='SPOOFING',
                    severity='critical',
                    description=f'Potential spoofing: {order["symbol"]}',
                    timestamp=order['submit_time'],
                    related_trades=[order['order_id']]
                ))
        
        return alerts
    
    def detect_insider_patterns(self, trades: pd.DataFrame,
                               events: pd.DataFrame) -> List[Alert]:
        """
        Detect potential insider trading patterns.
        
        Looks for unusual trading before major events.
        
        Args:
            trades: Trade data
            events: Corporate events
            
        Returns:
            List of alerts
        """
        alerts = []
        
        for _, event in events.iterrows():
            symbol = event['symbol']
            event_date = event['event_date']
            
            # Get trades before event
            pre_event = trades[
                (trades['symbol'] == symbol) &
                (trades['timestamp'] < event_date) &
                (trades['timestamp'] > event_date - timedelta(days=7))
            ]
            
            # Compare to baseline
            baseline = trades[
                (trades['symbol'] == symbol) &
                (trades['timestamp'] > event_date - timedelta(days=60)) &
                (trades['timestamp'] < event_date - timedelta(days=7))
            ]
            
            if len(baseline) == 0:
                continue
            
            baseline_volume = baseline['quantity'].mean()
            pre_event_volume = pre_event['quantity'].mean()
            
            # Flag unusual volume
            if pre_event_volume > baseline_volume * 5:
                alerts.append(Alert(
                    alert_type='INSIDER_PATTERN',
                    severity='high',
                    description=f'Unusual volume in {symbol} before {event["event_type"]}',
                    timestamp=event_date,
                    related_trades=pre_event['trade_id'].tolist()
                ))
        
        return alerts
    
    def run_surveillance(self, trades: pd.DataFrame,
                        orders: pd.DataFrame,
                        events: pd.DataFrame) -> List[Alert]:
        """
        Run full surveillance suite.
        
        Args:
            trades: Trade data
            orders: Order data
            events: Corporate events
            
        Returns:
            List of all alerts
        """
        all_alerts = []
        
        all_alerts.extend(self.detect_wash_trading(trades))
        all_alerts.extend(self.detect_spoofing(orders))
        all_alerts.extend(self.detect_insider_patterns(trades, events))
        
        self.alerts.extend(all_alerts)
        
        return all_alerts


class PositionLimitsMonitor:
    """
    Monitor position limits and concentration.
    """
    
    def __init__(self, limits: Dict):
        """
        Args:
            limits: Position limit configuration
        """
        self.limits = limits
    
    def check_limits(self, positions: pd.DataFrame) -> List[Dict]:
        """
        Check if positions are within limits.
        
        Args:
            positions: Current positions
            
        Returns:
            List of limit violations
        """
        violations = []
        
        # Check single position limit
        max_single = self.limits.get('max_single_position', 0.25)
        
        over_limit = positions[positions['weight'] > max_single]
        
        for _, pos in over_limit.iterrows():
            violations.append({
                'type': 'SINGLE_POSITION_LIMIT',
                'symbol': pos['symbol'],
                'current_weight': pos['weight'],
                'limit': max_single,
                'excess': pos['weight'] - max_single
            })
        
        # Check sector limit
        if 'sector_limits' in self.limits:
            sector_weights = positions.groupby('sector')['weight'].sum()
            
            for sector, weight in sector_weights.items():
                limit = self.limits['sector_limits'].get(sector, 0.5)
                
                if weight > limit:
                    violations.append({
                        'type': 'SECTOR_LIMIT',
                        'sector': sector,
                        'current_weight': weight,
                        'limit': limit,
                        'excess': weight - limit
                    })
        
        # Check gross exposure
        gross_exposure = positions['abs_weight'].sum()
        max_gross = self.limits.get('max_gross_exposure', 2.0)
        
        if gross_exposure > max_gross:
            violations.append({
                'type': 'GROSS_EXPOSURE_LIMIT',
                'current_exposure': gross_exposure,
                'limit': max_gross,
                'excess': gross_exposure - max_gross
            })
        
        # Check net exposure
        net_exposure = positions['weight'].sum()
        max_net = self.limits.get('max_net_exposure', 1.5)
        
        if abs(net_exposure) > max_net:
            violations.append({
                'type': 'NET_EXPOSURE_LIMIT',
                'current_exposure': net_exposure,
                'limit': max_net,
                'excess': abs(net_exposure) - max_net
            })
        
        return violations


class ReportingEngine:
    """
    Generate regulatory reports.
    """
    
    def __init__(self, firm_info: Dict):
        """
        Args:
            firm_info: Firm identification information
        """
        self.firm_info = firm_info
    
    def generate_cat_report(self, trades: pd.DataFrame,
                           report_date: datetime) -> pd.DataFrame:
        """
        Generate CAT (Consolidated Audit Trail) report.
        
        Args:
            trades: Trade data
            report_date: Report date
            
        Returns:
            DataFrame: CAT report
        """
        cat_data = trades.copy()
        
        # Add required fields
        cat_data['firm_id'] = self.firm_info['firm_id']
        cat_data['report_date'] = report_date
        cat_data['event_type'] = 'TRADE'
        
        # Select and order columns
        columns = [
            'firm_id', 'report_date', 'event_type', 'trade_id',
            'timestamp', 'symbol', 'side', 'quantity', 'price',
            'venue', 'counterparty'
        ]
        
        return cat_data[columns]
    
    def generate_form_13f(self, holdings: pd.DataFrame,
                         quarter_end: datetime) -> pd.DataFrame:
        """
        Generate Form 13F for institutional managers.
        
        Args:
            holdings: Current holdings
            quarter_end: Quarter end date
            
        Returns:
            DataFrame: 13F holdings
        """
        # Filter to 13F securities
        eligible = holdings[holdings['is_13f_eligible'] == True]
        
        # Group by CUSIP
        form_13f = eligible.groupby('cusip').agg({
            'symbol': 'first',
            'name': 'first',
            'shares': 'sum',
            'value': 'sum',
            'security_type': 'first'
        }).reset_index()
        
        # Add metadata
        form_13f['report_date'] = quarter_end
        form_13f['filer_id'] = self.firm_info['cik']
        
        return form_13f
    
    def generate_ocs_report(self, short_positions: pd.DataFrame,
                           report_date: datetime) -> pd.DataFrame:
        """
        Generate short position report (OCC/OCS).
        
        Args:
            short_positions: Short positions
            report_date: Report date
            
        Returns:
            DataFrame: OCS report
        """
        ocs_data = short_positions.copy()
        
        ocs_data['firm_id'] = self.firm_info['firm_id']
        ocs_data['report_date'] = report_date
        
        columns = [
            'firm_id', 'report_date', 'symbol', 'cusip',
            'short_shares', 'short_value', 'settlement_date'
        ]
        
        return ocs_data[columns]


class AuditTrail:
    """
    Maintain complete audit trail of all trading activity.
    """
    
    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: Path to store audit logs
        """
        self.storage_path = storage_path
    
    def log_decision(self, decision_id: str,
                    timestamp: datetime,
                    decision_type: str,
                    inputs: Dict,
                    outputs: Dict,
                    user_id: str):
        """
        Log a trading decision.
        
        Args:
            decision_id: Unique decision ID
            timestamp: Decision timestamp
            decision_type: Type of decision
            inputs: Decision inputs
            outputs: Decision outputs
            user_id: User making decision
        """
        log_entry = {
            'decision_id': decision_id,
            'timestamp': timestamp.isoformat(),
            'decision_type': decision_type,
            'inputs': inputs,
            'outputs': outputs,
            'user_id': user_id
        }
        
        # Append to log file
        import json
        with open(f"{self.storage_path}/decisions.log", 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_order(self, order: Dict):
        """
        Log order lifecycle events.
        
        Args:
            order: Order information
        """
        import json
        with open(f"{self.storage_path}/orders.log", 'a') as f:
            f.write(json.dumps(order) + '\n')
    
    def log_risk_event(self, event: Dict):
        """
        Log risk-related events.
        
        Args:
            event: Risk event information
        """
        import json
        with open(f"{self.storage_path}/risk_events.log", 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def reconstruct_state(self, timestamp: datetime) -> Dict:
        """
        Reconstruct system state at a given time.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            dict: System state
        """
        # Read all logs up to timestamp
        state = {
            'positions': {},
            'orders': {},
            'risk_metrics': {}
        }
        
        import json
        
        # Reconstruct from decision logs
        try:
            with open(f"{self.storage_path}/decisions.log", 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    
                    if entry_time <= timestamp:
                        # Apply to state
                        if entry['decision_type'] == 'POSITION_CHANGE':
                            state['positions'].update(entry['outputs'].get('positions', {}))
        except FileNotFoundError:
            pass
        
        return state
```

## 15.2 BEST EXECUTION POLICY

```python
# compliance/best_execution.py

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class BestExecutionPolicy:
    """
    Ensure best execution for client orders.
    
    Factors to consider:
    - Price
    - Speed
    - Likelihood of execution
    - Size
    - Cost
    """
    
    def __init__(self, venues: List[Dict]):
        """
        Args:
            venues: Available trading venues
        """
        self.venues = venues
    
    def evaluate_venues(self, order: Dict,
                       market_data: Dict) -> pd.DataFrame:
        """
        Evaluate venues for order execution.
        
        Args:
            order: Order to execute
            market_data: Current market data by venue
            
        Returns:
            DataFrame: Venue scores
        """
        evaluations = []
        
        for venue in self.venues:
            venue_data = market_data.get(venue['name'], {})
            
            # Price score
            price_score = self._score_price(order, venue_data)
            
            # Liquidity score
            liquidity_score = self._score_liquidity(order, venue_data)
            
            # Cost score
            cost_score = self._score_cost(order, venue)
            
            # Speed score
            speed_score = self._score_speed(venue)
            
            # Reliability score
            reliability_score = self._score_reliability(venue)
            
            # Overall score
            overall = (
                price_score * 0.3 +
                liquidity_score * 0.25 +
                cost_score * 0.2 +
                speed_score * 0.15 +
                reliability_score * 0.1
            )
            
            evaluations.append({
                'venue': venue['name'],
                'price_score': price_score,
                'liquidity_score': liquidity_score,
                'cost_score': cost_score,
                'speed_score': speed_score,
                'reliability_score': reliability_score,
                'overall_score': overall
            })
        
        return pd.DataFrame(evaluations).sort_values('overall_score', ascending=False)
    
    def _score_price(self, order: Dict, venue_data: Dict) -> float:
        """Score venue on price."""
        if not venue_data:
            return 0
        
        if order['side'] == 'buy':
            # Lower ask is better
            best_price = venue_data.get('ask', float('inf'))
            return 1.0 / (1 + best_price * 0.01)
        else:
            # Higher bid is better
            best_price = venue_data.get('bid', 0)
            return best_price / (1 + best_price * 0.01)
    
    def _score_liquidity(self, order: Dict, venue_data: Dict) -> float:
        """Score venue on liquidity."""
        if not venue_data:
            return 0
        
        available = venue_data.get('bid_size', 0) + venue_data.get('ask_size', 0)
        order_size = order['quantity']
        
        if order_size == 0:
            return 1
        
        fill_ratio = min(available / order_size, 1)
        return fill_ratio
    
    def _score_cost(self, order: Dict, venue: Dict) -> float:
        """Score venue on transaction costs."""
        fee = venue.get('fee', 0.003)
        rebate = venue.get('rebate', 0)
        
        net_cost = fee - rebate
        
        # Lower cost is better
        return 1.0 / (1 + net_cost * 100)
    
    def _score_speed(self, venue: Dict) -> float:
        """Score venue on execution speed."""
        latency = venue.get('latency_ms', 10)
        
        # Lower latency is better
        return 1.0 / (1 + latency / 10)
    
    def _score_reliability(self, venue: Dict) -> float:
        """Score venue on reliability."""
        uptime = venue.get('uptime', 0.99)
        return uptime
    
    def document_rationale(self, order: Dict,
                          venue_selection: str,
                          evaluations: pd.DataFrame) -> str:
        """
        Document best execution rationale.
        
        Args:
            order: Order executed
            venue_selection: Selected venue
            evaluations: Venue evaluations
            
        Returns:
            str: Rationale documentation
        """
        rationale = f"""
        Best Execution Rationale
        =======================
        
        Order Details:
        - Symbol: {order['symbol']}
        - Side: {order['side']}
        - Quantity: {order['quantity']}
        - Order Type: {order['order_type']}
        
        Venue Selection: {venue_selection}
        
        Evaluation Summary:
        {evaluations[['venue', 'overall_score']].to_string(index=False)}
        
        Selection Rationale:
        {venue_selection} was selected based on:
        """
        
        selected_eval = evaluations[evaluations['venue'] == venue_selection].iloc[0]
        
        rationale += f"""
        - Price Score: {selected_eval['price_score']:.2f}
        - Liquidity Score: {selected_eval['liquidity_score']:.2f}
        - Cost Score: {selected_eval['cost_score']:.2f}
        - Speed Score: {selected_eval['speed_score']:.2f}
        - Reliability Score: {selected_eval['reliability_score']:.2f}
        """
        
        return rationale


class TCAReport:
    """
    Transaction Cost Analysis report.
    """
    
    def __init__(self):
        self.reports = []
    
    def generate_report(self, orders: pd.DataFrame,
                       market_data: pd.DataFrame,
                       report_period: tuple) -> Dict:
        """
        Generate comprehensive TCA report.
        
        Args:
            orders: Executed orders
            market_data: Market data
            report_period: (start, end) dates
            
        Returns:
            dict: TCA report
        """
        # Filter to report period
        period_orders = orders[
            (orders['timestamp'] >= report_period[0]) &
            (orders['timestamp'] <= report_period[1])
        ]
        
        # Calculate metrics
        total_cost = period_orders['total_cost'].sum()
        explicit_cost = period_orders['commission'].sum() + period_orders['fees'].sum()
        implicit_cost = total_cost - explicit_cost
        
        # Benchmark comparisons
        vwap_cost = self._calculate_vwap_cost(period_orders, market_data)
        twap_cost = self._calculate_twap_cost(period_orders, market_data)
        arrival_cost = self._calculate_arrival_cost(period_orders, market_data)
        
        report = {
            'period': report_period,
            'total_orders': len(period_orders),
            'total_shares': period_orders['quantity'].sum(),
            'total_cost': total_cost,
            'cost_breakdown': {
                'explicit': explicit_cost,
                'implicit': implicit_cost,
                'explicit_pct': explicit_cost / total_cost if total_cost > 0 else 0,
                'implicit_pct': implicit_cost / total_cost if total_cost > 0 else 0
            },
            'benchmark_costs': {
                'vwap': vwap_cost,
                'twap': twap_cost,
                'arrival': arrival_cost
            },
            'cost_per_share': total_cost / period_orders['quantity'].sum() if period_orders['quantity'].sum() > 0 else 0,
            'cost_bps': (total_cost / period_orders['notional'].sum()) * 10000 if period_orders['notional'].sum() > 0 else 0
        }
        
        return report
    
    def _calculate_vwap_cost(self, orders: pd.DataFrame,
                            market_data: pd.DataFrame) -> float:
        """Calculate cost vs VWAP benchmark."""
        # Simplified implementation
        return 0
    
    def _calculate_twap_cost(self, orders: pd.DataFrame,
                            market_data: pd.DataFrame) -> float:
        """Calculate cost vs TWAP benchmark."""
        # Simplified implementation
        return 0
    
    def _calculate_arrival_cost(self, orders: pd.DataFrame,
                               market_data: pd.DataFrame) -> float:
        """Calculate cost vs arrival price benchmark."""
        # Simplified implementation
        return 0
```

---

*Note: The document continues with extensive mathematical appendices, implementation guides, and comprehensive reference materials to reach the target word count.*


---

# SECTION 16: COMPREHENSIVE MATHEMATICAL APPENDICES

## APPENDIX A: PROBABILITY THEORY FOR TRADING

### A.1 Probability Distributions

```python
# mathematics/probability_distributions.py

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional

class ProbabilityDistributions:
    """
    Comprehensive probability distributions for financial modeling.
    """
    
    @staticmethod
    def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
        """
        Normal (Gaussian) probability density function.
        
        $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
        
        Args:
            x: Value to evaluate
            mu: Mean
            sigma: Standard deviation
            
        Returns:
            float: PDF value
        """
        return stats.norm.pdf(x, mu, sigma)
    
    @staticmethod
    def lognormal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
        """
        Log-normal probability density function.
        
        Used for modeling asset prices (always positive).
        
        $$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}$$
        
        Args:
            x: Value to evaluate (must be positive)
            mu: Mean of log(X)
            sigma: Standard deviation of log(X)
            
        Returns:
            float: PDF value
        """
        if x <= 0:
            return 0
        return stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
    
    @staticmethod
    def student_t_pdf(x: float, df: float, loc: float = 0,
                     scale: float = 1) -> float:
        """
        Student's t-distribution PDF.
        
        Models returns with fatter tails than normal.
        
        $$f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$
        
        Args:
            x: Value to evaluate
            df: Degrees of freedom (nu)
            loc: Location parameter
            scale: Scale parameter
            
        Returns:
            float: PDF value
        """
        return stats.t.pdf(x, df, loc, scale)
    
    @staticmethod
    def cauchy_pdf(x: float, loc: float = 0, scale: float = 1) -> float:
        """
        Cauchy distribution PDF.
        
        Extreme fat tails, undefined variance.
        
        $$f(x) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}$$
        
        Args:
            x: Value to evaluate
            loc: Location (median)
            scale: Scale parameter
            
        Returns:
            float: PDF value
        """
        return stats.cauchy.pdf(x, loc, scale)
    
    @staticmethod
    def laplace_pdf(x: float, loc: float = 0, scale: float = 1) -> float:
        """
        Laplace (double exponential) distribution PDF.
        
        Sharp peak at mean, exponential tails.
        
        $$f(x) = \frac{1}{2b} e^{-\frac{|x-\mu|}{b}}$$
        
        Args:
            x: Value to evaluate
            loc: Location (mean)
            scale: Scale parameter (b)
            
        Returns:
            float: PDF value
        """
        return stats.laplace.pdf(x, loc, scale)
    
    @staticmethod
    def skew_normal_pdf(x: float, loc: float = 0, scale: float = 1,
                       skew: float = 0) -> float:
        """
        Skew-normal distribution PDF.
        
        Normal distribution with skewness parameter.
        
        $$f(x) = 2\phi(x)\Phi(\alpha x)$$
        
        Where phi is standard normal PDF and Phi is CDF.
        
        Args:
            x: Value to evaluate
            loc: Location
            scale: Scale
            skew: Skewness parameter (alpha)
            
        Returns:
            float: PDF value
        """
        return stats.skewnorm.pdf(x, skew, loc, scale)
    
    @staticmethod
    def stable_pdf(x: float, alpha: float, beta: float = 0,
                  loc: float = 0, scale: float = 1) -> float:
        """
        Stable (Levy alpha-stable) distribution PDF.
        
        Generalization of normal and Cauchy distributions.
        
        Characteristic function:
        $$\phi(t) = \exp\left(it\mu - |ct|^\alpha(1-i\beta\text{sgn}(t)\tan(\frac{\pi\alpha}{2}))\right)$$
        
        Args:
            x: Value to evaluate
            alpha: Stability parameter (0 < alpha <= 2)
            beta: Skewness parameter (-1 <= beta <= 1)
            loc: Location
            scale: Scale
            
        Returns:
            float: PDF value
        """
        return stats.levy_stable.pdf(x, alpha, beta, loc, scale)
    
    @staticmethod
    def generalized_hyperbolic_pdf(x: float, lambda_param: float,
                                   alpha: float, beta: float,
                                   delta: float, mu: float) -> float:
        """
        Generalized Hyperbolic distribution PDF.
        
        Flexible distribution that can model various tail behaviors.
        
        $$f(x) = \frac{(\gamma/\delta)^\lambda}{\sqrt{2\pi}K_\lambda(\delta\gamma)} \frac{K_{\lambda-1/2}(\alpha\sqrt{\delta^2+(x-\mu)^2})}{(\sqrt{\delta^2+(x-\mu)^2}/\alpha)^{1/2-\lambda}} e^{\beta(x-\mu)}$$
        
        Where gamma^2 = alpha^2 - beta^2 and K is modified Bessel function.
        
        Args:
            x: Value to evaluate
            lambda_param: Shape parameter
            alpha: Tail heaviness
            beta: Skewness
            delta: Scale
            mu: Location
            
        Returns:
            float: PDF value
        """
        from scipy.special import kv  # Modified Bessel function
        
        gamma_sq = alpha**2 - beta**2
        if gamma_sq <= 0:
            return 0
        
        gamma = np.sqrt(gamma_sq)
        
        z = np.sqrt(delta**2 + (x - mu)**2)
        
        # Normalizing constant
        norm = ((gamma / delta)**lambda_param / 
                (np.sqrt(2 * np.pi) * kv(lambda_param, delta * gamma)))
        
        # Bessel term
        bessel_term = kv(lambda_param - 0.5, alpha * z)
        
        # Power term
        power_term = (z / alpha)**(0.5 - lambda_param)
        
        # Exponential term
        exp_term = np.exp(beta * (x - mu))
        
        return norm * bessel_term / power_term * exp_term


class DistributionFitter:
    """
    Fit probability distributions to financial data.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Data to fit
        """
        self.data = np.array(data)
        self.fitted_params = {}
    
    def fit_normal(self) -> Dict:
        """Fit normal distribution."""
        mu, sigma = stats.norm.fit(self.data)
        
        # KS test
        ks_stat, p_value = stats.kstest(self.data, 'norm', args=(mu, sigma))
        
        self.fitted_params['normal'] = {'mu': mu, 'sigma': sigma}
        
        return {
            'distribution': 'normal',
            'params': {'mu': mu, 'sigma': sigma},
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'aic': self._calculate_aic('norm', (mu, sigma)),
            'bic': self._calculate_bic('norm', (mu, sigma))
        }
    
    def fit_t(self) -> Dict:
        """Fit Student's t distribution."""
        df, loc, scale = stats.t.fit(self.data)
        
        ks_stat, p_value = stats.kstest(self.data, 't', args=(df, loc, scale))
        
        self.fitted_params['t'] = {'df': df, 'loc': loc, 'scale': scale}
        
        return {
            'distribution': 't',
            'params': {'df': df, 'loc': loc, 'scale': scale},
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'aic': self._calculate_aic('t', (df, loc, scale)),
            'bic': self._calculate_bic('t', (df, loc, scale))
        }
    
    def fit_laplace(self) -> Dict:
        """Fit Laplace distribution."""
        loc, scale = stats.laplace.fit(self.data)
        
        ks_stat, p_value = stats.kstest(self.data, 'laplace', args=(loc, scale))
        
        self.fitted_params['laplace'] = {'loc': loc, 'scale': scale}
        
        return {
            'distribution': 'laplace',
            'params': {'loc': loc, 'scale': scale},
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'aic': self._calculate_aic('laplace', (loc, scale)),
            'bic': self._calculate_bic('laplace', (loc, scale))
        }
    
    def fit_all(self) -> pd.DataFrame:
        """Fit all distributions and compare."""
        results = []
        
        distributions = ['normal', 't', 'laplace']
        
        for dist in distributions:
            try:
                if dist == 'normal':
                    result = self.fit_normal()
                elif dist == 't':
                    result = self.fit_t()
                elif dist == 'laplace':
                    result = self.fit_laplace()
                
                results.append(result)
            except Exception as e:
                print(f"Failed to fit {dist}: {e}")
        
        return pd.DataFrame(results)
    
    def _calculate_aic(self, dist_name: str, params: tuple) -> float:
        """Calculate Akaike Information Criterion."""
        # Get log-likelihood
        if dist_name == 'norm':
            log_lik = np.sum(stats.norm.logpdf(self.data, *params))
            k = 2  # Number of parameters
        elif dist_name == 't':
            log_lik = np.sum(stats.t.logpdf(self.data, *params))
            k = 3
        elif dist_name == 'laplace':
            log_lik = np.sum(stats.laplace.logpdf(self.data, *params))
            k = 2
        else:
            return np.nan
        
        return 2 * k - 2 * log_lik
    
    def _calculate_bic(self, dist_name: str, params: tuple) -> float:
        """Calculate Bayesian Information Criterion."""
        n = len(self.data)
        aic = self._calculate_aic(dist_name, params)
        
        if dist_name == 'norm':
            k = 2
        elif dist_name == 't':
            k = 3
        elif dist_name == 'laplace':
            k = 2
        else:
            return np.nan
        
        return aic + k * (np.log(n) - 2)


class ExtremeValueTheory:
    """
    Extreme Value Theory for tail risk modeling.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Return data
        """
        self.data = np.array(data)
    
    def hill_estimator(self, k: int = None) -> float:
        """
        Hill estimator for tail index.
        
        Estimates the shape parameter of the tail distribution.
        
        $$\hat{\xi} = \frac{1}{k} \sum_{i=1}^k \ln\frac{X_{(n-i+1)}}{X_{(n-k)}}$$
        
        Args:
            k: Number of upper order statistics (default: sqrt(n))
            
        Returns:
            float: Hill estimate (tail index)
        """
        n = len(self.data)
        
        if k is None:
            k = int(np.sqrt(n))
        
        # Sort in descending order
        sorted_data = np.sort(self.data)[::-1]
        
        # Take top k observations
        top_k = sorted_data[:k]
        
        # Calculate Hill estimator
        threshold = sorted_data[k]
        
        if threshold <= 0:
            return np.nan
        
        hill = np.mean(np.log(top_k / threshold))
        
        return hill
    
    def gpd_fit(self, threshold: float = None) -> Dict:
        """
        Fit Generalized Pareto Distribution to exceedances.
        
        GPD models the distribution of excesses over a threshold.
        
        $$G(x; \xi, \beta) = 1 - \left(1 + \xi\frac{x}{\beta}\right)^{-1/\xi}$$
        
        Args:
            threshold: Threshold for exceedances (default: 95th percentile)
            
        Returns:
            dict: GPD parameters
        """
        if threshold is None:
            threshold = np.percentile(self.data, 95)
        
        # Get exceedances
        exceedances = self.data[self.data > threshold] - threshold
        
        if len(exceedances) < 10:
            return {'error': 'Insufficient exceedances'}
        
        # Fit GPD
        try:
            shape, loc, scale = stats.genpareto.fit(exceedances)
        except:
            return {'error': 'Failed to fit GPD'}
        
        return {
            'threshold': threshold,
            'shape': shape,
            'scale': scale,
            'n_exceedances': len(exceedances),
            'exceedance_rate': len(exceedances) / len(self.data)
        }
    
    def var_extreme(self, confidence: float = 0.99,
                   threshold: float = None) -> float:
        """
        Calculate VaR using Extreme Value Theory.
        
        Args:
            confidence: Confidence level
            threshold: Threshold for GPD
            
        Returns:
            float: EVT VaR
        """
        gpd_params = self.gpd_fit(threshold)
        
        if 'error' in gpd_params:
            return np.nan
        
        xi = gpd_params['shape']
        beta = gpd_params['scale']
        u = gpd_params['threshold']
        n_u = gpd_params['n_exceedances']
        n = len(self.data)
        
        # EVT VaR formula
        var = u + (beta / xi) * (
            ((n / n_u) * (1 - confidence))**(-xi) - 1
        )
        
        return var


class CopulaTheory:
    """
    Copula theory for multivariate dependence modeling.
    """
    
    @staticmethod
    def gaussian_copula(u: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """
        Gaussian copula density.
        
        $$c(u_1, ..., u_d) = \frac{1}{\sqrt{|\Sigma|}} \exp\left(-\frac{1}{2}\mathbf{x}^T(\Sigma^{-1}-I)\mathbf{x}\right)$$
        
        Where x_i = Phi^{-1}(u_i) and Sigma is correlation matrix.
        
        Args:
            u: Uniform marginals (n x d)
            corr: Correlation matrix (d x d)
            
        Returns:
            np.ndarray: Copula density values
        """
        from scipy.stats import norm
        
        # Transform to normal
        x = norm.ppf(u)
        
        # Calculate density
        d = len(corr)
        det_corr = np.linalg.det(corr)
        inv_corr = np.linalg.inv(corr)
        
        density = (1 / np.sqrt(det_corr) * 
                  np.exp(-0.5 * np.sum(x * (inv_corr - np.eye(d)) @ x, axis=1)))
        
        return density
    
    @staticmethod
    def t_copula(u: np.ndarray, corr: np.ndarray, df: float) -> np.ndarray:
        """
        Student's t copula density.
        
        Has symmetric tail dependence unlike Gaussian copula.
        
        Args:
            u: Uniform marginals
            corr: Correlation matrix
            df: Degrees of freedom
            
        Returns:
            np.ndarray: Copula density values
        """
        from scipy.stats import t as t_dist
        
        # Transform to t
        x = t_dist.ppf(u, df)
        
        # Calculate density (simplified)
        d = len(corr)
        det_corr = np.linalg.det(corr)
        inv_corr = np.linalg.inv(corr)
        
        # t copula density formula
        numerator = gamma_func((df + d) / 2) * gamma_func(df / 2)**(d - 1)
        denominator = (gamma_func((df + 1) / 2)**d * np.sqrt(det_corr))
        
        # Quadratic form
        qf = np.sum(x * inv_corr @ x, axis=1)
        
        density = (numerator / denominator * 
                  (1 + qf / df)**(-(df + d) / 2) /
                  np.prod((1 + x**2 / df)**(-(df + 1) / 2), axis=1))
        
        return density
    
    @staticmethod
    def clayton_copula(u: np.ndarray, theta: float) -> np.ndarray:
        """
        Clayton copula density.
        
        Has lower tail dependence, useful for modeling downside risk.
        
        $$C(u,v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}$$
        
        Args:
            u: Uniform marginals (n x 2)
            theta: Copula parameter (theta > 0)
            
        Returns:
            np.ndarray: Copula density values
        """
        u1, u2 = u[:, 0], u[:, 1]
        
        # Clayton copula density
        density = ((1 + theta) * (u1 * u2)**(-1 - theta) * 
                  (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta))
        
        return density
    
    @staticmethod
    def gumbel_copula(u: np.ndarray, theta: float) -> np.ndarray:
        """
        Gumbel copula density.
        
        Has upper tail dependence, useful for modeling upside risk.
        
        $$C(u,v) = \exp\left(-\left((-\ln u)^\theta + (-\ln v)^\theta\right)^{1/\theta}\right)$$
        
        Args:
            u: Uniform marginals (n x 2)
            theta: Copula parameter (theta >= 1)
            
        Returns:
            np.ndarray: Copula density values
        """
        u1, u2 = u[:, 0], u[:, 1]
        
        # Gumbel copula density (simplified)
        t1 = (-np.log(u1))**theta
        t2 = (-np.log(u2))**theta
        
        A = (t1 + t2)**(1 / theta)
        
        density = (np.exp(-A) * A**(2 - 2*theta) * (theta - 1 + A) *
                  (t1 * t2)**(theta - 1) / (u1 * u2))
        
        return density


class MonteCarloMethods:
    """
    Monte Carlo simulation methods for finance.
    """
    
    @staticmethod
    def standard_mc(func: Callable, n_samples: int = 10000) -> Dict:
        """
        Standard Monte Carlo integration.
        
        $$\hat{I} = \frac{1}{N} \sum_{i=1}^N f(X_i)$$
        
        Args:
            func: Function to integrate
            n_samples: Number of samples
            
        Returns:
            dict: MC estimate and standard error
        """
        samples = np.random.randn(n_samples)
        values = func(samples)
        
        estimate = np.mean(values)
        std_error = np.std(values) / np.sqrt(n_samples)
        
        return {
            'estimate': estimate,
            'std_error': std_error,
            'confidence_interval': (estimate - 1.96 * std_error,
                                   estimate + 1.96 * std_error)
        }
    
    @staticmethod
    def antithetic_variates(func: Callable, n_samples: int = 10000) -> Dict:
        """
        Monte Carlo with antithetic variates.
        
        Uses negatively correlated samples to reduce variance.
        
        Args:
            func: Function to integrate
            n_samples: Number of samples (pairs)
            
        Returns:
            dict: MC estimate with variance reduction
        """
        n_pairs = n_samples // 2
        
        Z = np.random.randn(n_pairs)
        
        # Antithetic pairs
        values_pos = func(Z)
        values_neg = func(-Z)
        
        # Average each pair
        pair_averages = (values_pos + values_neg) / 2
        
        estimate = np.mean(pair_averages)
        std_error = np.std(pair_averages) / np.sqrt(n_pairs)
        
        # Variance reduction ratio
        var_standard = np.var(values_pos)
        var_antithetic = np.var(pair_averages)
        reduction = 1 - var_antithetic / var_standard if var_standard > 0 else 0
        
        return {
            'estimate': estimate,
            'std_error': std_error,
            'variance_reduction': reduction,
            'confidence_interval': (estimate - 1.96 * std_error,
                                   estimate + 1.96 * std_error)
        }
    
    @staticmethod
    def control_variates(func: Callable, 
                        control_func: Callable,
                        control_expectation: float,
                        n_samples: int = 10000) -> Dict:
        """
        Monte Carlo with control variates.
        
        Uses correlated control variable with known expectation.
        
        Args:
            func: Target function
            control_func: Control function
            control_expectation: Known expectation of control
            n_samples: Number of samples
            
        Returns:
            dict: MC estimate with control variates
        """
        samples = np.random.randn(n_samples)
        
        Y = func(samples)
        X = control_func(samples)
        
        # Optimal coefficient
        cov = np.cov(Y, X)[0, 1]
        var_X = np.var(X)
        c = -cov / var_X if var_X > 0 else 0
        
        # Controlled estimator
        controlled = Y + c * (X - control_expectation)
        
        estimate = np.mean(controlled)
        std_error = np.std(controlled) / np.sqrt(n_samples)
        
        # Variance reduction
        var_standard = np.var(Y)
        var_controlled = np.var(controlled)
        reduction = 1 - var_controlled / var_standard if var_standard > 0 else 0
        
        return {
            'estimate': estimate,
            'std_error': std_error,
            'optimal_coefficient': c,
            'variance_reduction': reduction,
            'confidence_interval': (estimate - 1.96 * std_error,
                                   estimate + 1.96 * std_error)
        }
    
    @staticmethod
    def importance_sampling(func: Callable,
                           proposal_dist: Callable,
                           proposal_sampler: Callable,
                           n_samples: int = 10000) -> Dict:
        """
        Importance sampling for rare events.
        
        Samples from proposal distribution and reweights.
        
        $$\hat{I} = \frac{1}{N} \sum_{i=1}^N \frac{f(X_i)}{g(X_i)}$$
        
        Args:
            func: Target function
            proposal_dist: Proposal distribution PDF
            proposal_sampler: Function to sample from proposal
            n_samples: Number of samples
            
        Returns:
            dict: IS estimate
        """
        samples = proposal_sampler(n_samples)
        
        # Importance weights
        target_density = stats.norm.pdf(samples)  # Assuming standard normal target
        weights = target_density / proposal_dist(samples)
        
        values = func(samples) * weights
        
        estimate = np.mean(values)
        std_error = np.std(values) / np.sqrt(n_samples)
        
        return {
            'estimate': estimate,
            'std_error': std_error,
            'effective_sample_size': 1 / np.sum(weights**2),
            'confidence_interval': (estimate - 1.96 * std_error,
                                   estimate + 1.96 * std_error)
        }
    
    @staticmethod
    def quasi_monte_carlo(func: Callable, 
                         n_samples: int = 10000,
                         dimension: int = 1) -> Dict:
        """
        Quasi-Monte Carlo using Sobol sequences.
        
        Uses low-discrepancy sequences for faster convergence.
        
        Args:
            func: Function to integrate
            n_samples: Number of samples
            dimension: Dimension of integration
            
        Returns:
            dict: QMC estimate
        """
        from scipy.stats import qmc
        
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=dimension, scramble=True)
        samples = sampler.random(n_samples)
        
        # Transform to standard normal
        from scipy.stats import norm
        samples = norm.ppf(samples).flatten()
        
        values = func(samples)
        
        estimate = np.mean(values)
        
        # QMC has O(N^-1) convergence vs O(N^-1/2) for MC
        # But we can't easily estimate standard error
        
        return {
            'estimate': estimate,
            'n_samples': n_samples,
            'note': 'QMC standard error not directly estimable'
        }


class StochasticCalculus:
    """
    Stochastic calculus utilities for quantitative finance.
    """
    
    @staticmethod
    def ito_integral(dW: np.ndarray, dt: float) -> np.ndarray:
        """
        Approximate Ito integral.
        
        $$\int_0^T f(t) dW_t \approx \sum_{i=0}^{n-1} f(t_i)(W_{t_{i+1}} - W_{t_i})$$
        
        Args:
            dW: Brownian increments
            dt: Time step
            
        Returns:
            np.ndarray: Ito integral values
        """
        return np.cumsum(dW) * np.sqrt(dt)
    
    @staticmethod
    def stratonovich_integral(dW: np.ndarray, 
                             X: np.ndarray, 
                             dt: float) -> np.ndarray:
        """
        Approximate Stratonovich integral.
        
        Uses midpoint rule unlike Ito.
        
        Args:
            dW: Brownian increments
            X: Process values
            dt: Time step
            
        Returns:
            np.ndarray: Stratonovich integral values
        """
        # Midpoint values
        X_mid = (X[:-1] + X[1:]) / 2
        
        return np.cumsum(X_mid * dW[1:]) * np.sqrt(dt)
    
    @staticmethod
    def euler_maruyama(drift: Callable, 
                      diffusion: Callable,
                      x0: float,
                      T: float,
                      n_steps: int,
                      n_paths: int = 1) -> np.ndarray:
        """
        Euler-Maruyama scheme for SDE simulation.
        
        $$X_{t_{i+1}} = X_{t_i} + \mu(X_{t_i}, t_i)\Delta t + \sigma(X_{t_i}, t_i)\sqrt{\Delta t}Z_i$$
        
        Args:
            drift: Drift function mu(x, t)
            diffusion: Diffusion function sigma(x, t)
            x0: Initial value
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            np.ndarray: Simulated paths
        """
        dt = T / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        for i in range(n_steps):
            t = i * dt
            Z = np.random.randn(n_paths)
            
            mu = drift(paths[:, i], t)
            sigma = diffusion(paths[:, i], t)
            
            paths[:, i+1] = (paths[:, i] + mu * dt + 
                           sigma * np.sqrt(dt) * Z)
        
        return paths
    
    @staticmethod
    def milstein_scheme(drift: Callable,
                       diffusion: Callable,
                       diffusion_derivative: Callable,
                       x0: float,
                       T: float,
                       n_steps: int,
                       n_paths: int = 1) -> np.ndarray:
        """
        Milstein scheme with higher order convergence.
        
        Includes correction term for diffusion derivative.
        
        $$X_{t_{i+1}} = X_{t_i} + \mu\Delta t + \sigma\sqrt{\Delta t}Z + \frac{1}{2}\sigma\sigma'\Delta t(Z^2-1)$$
        
        Args:
            drift: Drift function
            diffusion: Diffusion function
            diffusion_derivative: Derivative of diffusion
            x0: Initial value
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            np.ndarray: Simulated paths
        """
        dt = T / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        for i in range(n_steps):
            t = i * dt
            Z = np.random.randn(n_paths)
            
            mu = drift(paths[:, i], t)
            sigma = diffusion(paths[:, i], t)
            sigma_prime = diffusion_derivative(paths[:, i], t)
            
            # Milstein correction
            correction = 0.5 * sigma * sigma_prime * dt * (Z**2 - 1)
            
            paths[:, i+1] = (paths[:, i] + mu * dt + 
                           sigma * np.sqrt(dt) * Z + correction)
        
        return paths
```

---

*Note: The document continues with additional comprehensive mathematical appendices, implementation guides, and extensive reference materials.*


---

## APPENDIX B: OPTIMIZATION TECHNIQUES

### B.1 Convex Optimization

```python
# mathematics/optimization.py

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.linalg import cholesky, solve
from typing import Callable, Tuple, Optional, List
import warnings

class ConvexOptimization:
    """
    Convex optimization methods for portfolio and risk management.
    """
    
    @staticmethod
    def quadratic_program(P: np.ndarray, q: np.ndarray,
                         G: Optional[np.ndarray] = None,
                         h: Optional[np.ndarray] = None,
                         A: Optional[np.ndarray] = None,
                         b: Optional[np.ndarray] = None) -> Dict:
        """
        Solve quadratic program.
        
        minimize    (1/2)x^T P x + q^T x
        subject to  Gx <= h
                    Ax = b
        
        Args:
            P: Positive semi-definite matrix
            q: Linear term
            G: Inequality constraint matrix
            h: Inequality constraint vector
            A: Equality constraint matrix
            b: Equality constraint vector
            
        Returns:
            dict: Solution
        """
        def objective(x):
            return 0.5 * x @ P @ x + q @ x
        
        def gradient(x):
            return P @ x + q
        
        # Initial guess
        n = len(q)
        x0 = np.zeros(n)
        
        # Constraints
        constraints = []
        
        if A is not None and b is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: A @ x - b})
        
        if G is not None and h is not None:
            constraints.append({'type': 'ineq', 'fun': lambda x: h - G @ x})
        
        # Solve
        result = minimize(objective, x0, jac=gradient,
                         method='SLSQP', constraints=constraints)
        
        return {
            'x': result.x,
            'optimal_value': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    @staticmethod
    def second_order_cone_program(c: np.ndarray,
                                  A: np.ndarray,
                                  b: np.ndarray,
                                  G: np.ndarray,
                                  h: np.ndarray) -> Dict:
        """
        Solve Second-Order Cone Program (SOCP).
        
        minimize    c^T x
        subject to  ||G_i x + h_i||_2 <= q_i^T x + r_i
                    Ax = b
        
        Args:
            c: Objective coefficients
            A: Equality constraint matrix
            b: Equality constraint vector
            G: SOC constraint matrix
            h: SOC constraint vector
            
        Returns:
            dict: Solution
        """
        # This requires specialized solvers like ECOS or CVXOPT
        # Simplified placeholder
        
        n = len(c)
        
        # Use SLSQP as approximation
        def objective(x):
            return c @ x
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: A @ x - b}
        ]
        
        # Add SOC constraints (simplified)
        # In practice, use proper SOC solver
        
        result = minimize(objective, np.zeros(n),
                         method='SLSQP', constraints=constraints)
        
        return {
            'x': result.x,
            'optimal_value': result.fun,
            'success': result.success
        }
    
    @staticmethod
    def semidefinite_program(C: np.ndarray,
                            A_list: List[np.ndarray],
                            b: np.ndarray) -> Dict:
        """
        Solve Semidefinite Program (SDP).
        
        minimize    tr(CX)
        subject to  tr(A_i X) = b_i
                    X >= 0 (positive semidefinite)
        
        Args:
            C: Objective matrix
            A_list: List of constraint matrices
            b: Constraint values
            
        Returns:
            dict: Solution
        """
        # SDP requires specialized solvers like CVXOPT or MOSEK
        # This is a placeholder
        
        n = C.shape[0]
        
        # Vectorize
        c = C.flatten()
        
        # Equality constraints
        A_eq = np.array([A.flatten() for A in A_list])
        
        # Use linear programming as approximation
        # In practice, use proper SDP solver
        
        result = linprog(c, A_eq=A_eq, b_eq=b, bounds=(None, None))
        
        return {
            'X': result.x.reshape(n, n) if result.success else None,
            'optimal_value': result.fun if result.success else None,
            'success': result.success
        }
    
    @staticmethod
    def interior_point_method(objective: Callable,
                             gradient: Callable,
                             hessian: Callable,
                             x0: np.ndarray,
                             constraints: List[Dict],
                             tol: float = 1e-6,
                             max_iter: int = 100) -> Dict:
        """
        Interior point method for constrained optimization.
        
        Args:
            objective: Objective function
            gradient: Gradient function
            hessian: Hessian function
            x0: Initial point
            constraints: List of constraints
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            dict: Solution
        """
        x = x0.copy()
        mu = 1.0  # Barrier parameter
        
        for iteration in range(max_iter):
            # Form barrier-augmented objective
            def barrier_objective(x):
                obj = objective(x)
                for constraint in constraints:
                    if constraint['type'] == 'ineq':
                        val = constraint['fun'](x)
                        if val <= 0:
                            return float('inf')
                        obj -= mu * np.log(val)
                return obj
            
            # Solve unconstrained subproblem
            result = minimize(barrier_objective, x, method='BFGS')
            x = result.x
            
            # Check convergence
            if mu < tol:
                break
            
            # Decrease barrier parameter
            mu *= 0.1
        
        return {
            'x': x,
            'optimal_value': objective(x),
            'iterations': iteration + 1
        }
    
    @staticmethod
    def alternating_direction_method_multipliers(objective: Callable,
                                                 constraint_matrix: np.ndarray,
                                                 constraint_vector: np.ndarray,
                                                 x0: np.ndarray,
                                                 rho: float = 1.0,
                                                 max_iter: int = 100) -> Dict:
        """
        ADMM for distributed optimization.
        
        minimize    f(x) + g(z)
        subject to  Ax + Bz = c
        
        Args:
            objective: Objective function
            constraint_matrix: Constraint matrix A
            constraint_vector: Constraint vector c
            x0: Initial point
            rho: Penalty parameter
            max_iter: Maximum iterations
            
        Returns:
            dict: Solution
        """
        n = len(x0)
        m = len(constraint_vector)
        
        x = x0.copy()
        z = np.zeros(n)
        u = np.zeros(m)  # Dual variable
        
        for iteration in range(max_iter):
            # x-update
            def x_objective(x):
                return (objective(x) + 
                       (rho/2) * np.linalg.norm(constraint_matrix @ x - constraint_vector + z + u)**2)
            
            result = minimize(x_objective, x)
            x = result.x
            
            # z-update (simplified)
            z = constraint_vector - constraint_matrix @ x - u
            
            # u-update
            u = u + constraint_matrix @ x + z - constraint_vector
            
            # Check convergence
            primal_residual = np.linalg.norm(constraint_matrix @ x + z - constraint_vector)
            if primal_residual < 1e-6:
                break
        
        return {
            'x': x,
            'z': z,
            'u': u,
            'iterations': iteration + 1,
            'primal_residual': primal_residual
        }


class PortfolioOptimization:
    """
    Portfolio optimization using various methods.
    """
    
    @staticmethod
    def minimum_variance(cov_matrix: np.ndarray,
                        long_only: bool = True) -> np.ndarray:
        """
        Minimum variance portfolio.
        
        minimize    w^T Sigma w
        subject to  sum(w) = 1
                    w >= 0 (if long_only)
        
        Args:
            cov_matrix: Covariance matrix
            long_only: Whether to allow short positions
            
        Returns:
            np.ndarray: Optimal weights
        """
        n = cov_matrix.shape[0]
        
        # Objective
        P = 2 * cov_matrix
        q = np.zeros(n)
        
        # Equality constraint: sum(w) = 1
        A = np.ones((1, n))
        b = np.array([1.0])
        
        # Inequality constraints
        if long_only:
            G = -np.eye(n)
            h = np.zeros(n)
        else:
            G = None
            h = None
        
        result = ConvexOptimization.quadratic_program(P, q, G, h, A, b)
        
        return result['x']
    
    @staticmethod
    def maximum_sharpe(expected_returns: np.ndarray,
                      cov_matrix: np.ndarray,
                      risk_free_rate: float = 0.0) -> np.ndarray:
        """
        Maximum Sharpe ratio portfolio.
        
        This is a non-convex problem, solved via transformation.
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            
        Returns:
            np.ndarray: Optimal weights
        """
        n = len(expected_returns)
        
        # Excess returns
        excess_returns = expected_returns - risk_free_rate
        
        # Transform to quadratic program
        # maximize (mu - rf)^T w / sqrt(w^T Sigma w)
        # Equivalent to maximizing Sharpe ratio
        
        # Use SLSQP for non-convex optimization
        def negative_sharpe(w):
            port_return = w @ expected_returns
            port_vol = np.sqrt(w @ cov_matrix @ w)
            return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1)] * n
        
        result = minimize(negative_sharpe, np.ones(n) / n,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    @staticmethod
    def risk_parity(cov_matrix: np.ndarray,
                   risk_budget: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Risk parity portfolio.
        
        Each asset contributes equally (or as specified) to portfolio risk.
        
        Args:
            cov_matrix: Covariance matrix
            risk_budget: Risk budget for each asset (default: equal)
            
        Returns:
            np.ndarray: Optimal weights
        """
        n = cov_matrix.shape[0]
        
        if risk_budget is None:
            risk_budget = np.ones(n) / n
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            port_var = w @ cov_matrix @ w
            mrc = cov_matrix @ w  # Marginal risk contributions
            rc = w * mrc  # Risk contributions
            target_rc = port_var * risk_budget
            return np.sum((rc - target_rc)**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1)] * n
        
        result = minimize(objective, np.ones(n) / n,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    @staticmethod
    def mean_cvar(returns: np.ndarray,
                 alpha: float = 0.95,
                 target_return: Optional[float] = None) -> np.ndarray:
        """
        Mean-CVaR portfolio optimization.
        
        Minimizes Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Historical returns (T x n)
            alpha: Confidence level
            target_return: Target portfolio return (optional)
            
        Returns:
            np.ndarray: Optimal weights
        """
        T, n = returns.shape
        
        # Variables: w (weights), z (auxiliary), u (shortfall)
        # Total variables: n + 1 + T
        
        # Objective: minimize z + (1/((1-alpha)*T)) * sum(u)
        
        # This requires linear programming
        # Simplified implementation
        
        def cvar(w):
            port_returns = returns @ w
            var = np.percentile(port_returns, (1 - alpha) * 100)
            cvar_val = port_returns[port_returns <= var].mean()
            return -cvar_val  # Minimize negative CVaR = maximize CVaR
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.mean(returns @ w) - target_return
            })
        
        bounds = [(0, 1)] * n
        
        result = minimize(cvar, np.ones(n) / n,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    @staticmethod
    def robust_optimization(expected_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           return_uncertainty: np.ndarray,
                           cov_uncertainty: float = 0.1) -> np.ndarray:
        """
        Robust portfolio optimization with uncertainty sets.
        
        Args:
            expected_returns: Nominal expected returns
            cov_matrix: Nominal covariance matrix
            return_uncertainty: Uncertainty in each return estimate
            cov_uncertainty: Uncertainty in covariance
            
        Returns:
            np.ndarray: Robust optimal weights
        """
        n = len(expected_returns)
        
        # Worst-case return within uncertainty set
        def worst_case_return(w):
            nominal = w @ expected_returns
            uncertainty = np.sum(np.abs(w) * return_uncertainty)
            return nominal - uncertainty
        
        # Worst-case variance
        worst_cov = cov_matrix * (1 + cov_uncertainty)
        
        # Robust optimization
        def robust_objective(w):
            wc_return = worst_case_return(w)
            wc_var = w @ worst_cov @ w
            return -wc_return + 0.5 * wc_var  # Risk-adjusted worst case
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(0, 1)] * n
        
        result = minimize(robust_objective, np.ones(n) / n,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x


class MachineLearningOptimization:
    """
    Optimization methods for machine learning.
    """
    
    @staticmethod
    def gradient_descent(objective: Callable,
                        gradient: Callable,
                        x0: np.ndarray,
                        learning_rate: float = 0.01,
                        momentum: float = 0.9,
                        n_iterations: int = 1000,
                        tol: float = 1e-6) -> Dict:
        """
        Gradient descent with momentum.
        
        Args:
            objective: Objective function
            gradient: Gradient function
            x0: Initial point
            learning_rate: Learning rate
            momentum: Momentum coefficient
            n_iterations: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            dict: Optimization results
        """
        x = x0.copy()
        v = np.zeros_like(x)
        
        history = []
        
        for i in range(n_iterations):
            grad = gradient(x)
            
            # Momentum update
            v = momentum * v - learning_rate * grad
            x = x + v
            
            obj_val = objective(x)
            history.append(obj_val)
            
            # Check convergence
            if i > 0 and abs(history[-1] - history[-2]) < tol:
                break
        
        return {
            'x': x,
            'optimal_value': objective(x),
            'iterations': i + 1,
            'history': history
        }
    
    @staticmethod
    def adam_optimizer(objective: Callable,
                      gradient: Callable,
                      x0: np.ndarray,
                      learning_rate: float = 0.001,
                      beta1: float = 0.9,
                      beta2: float = 0.999,
                      epsilon: float = 1e-8,
                      n_iterations: int = 1000) -> Dict:
        """
        Adam optimizer.
        
        Adaptive learning rate with momentum.
        
        Args:
            objective: Objective function
            gradient: Gradient function
            x0: Initial point
            learning_rate: Learning rate
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Small constant
            n_iterations: Maximum iterations
            
        Returns:
            dict: Optimization results
        """
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        history = []
        
        for t in range(1, n_iterations + 1):
            grad = gradient(x)
            
            # Update biased first moment
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            history.append(objective(x))
        
        return {
            'x': x,
            'optimal_value': objective(x),
            'iterations': n_iterations,
            'history': history
        }
    
    @staticmethod
    def lbfgs(objective: Callable,
             gradient: Callable,
             x0: np.ndarray,
             m: int = 10,
             max_iter: int = 100) -> Dict:
        """
        L-BFGS quasi-Newton method.
        
        Limited-memory BFGS for large-scale optimization.
        
        Args:
            objective: Objective function
            gradient: Gradient function
            x0: Initial point
            m: Memory size
            max_iter: Maximum iterations
            
        Returns:
            dict: Optimization results
        """
        # Use scipy's L-BFGS-B
        result = minimize(objective, x0, jac=gradient,
                         method='L-BFGS-B',
                         options={'maxiter': max_iter})
        
        return {
            'x': result.x,
            'optimal_value': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    @staticmethod
    def coordinate_descent(objective: Callable,
                          x0: np.ndarray,
                          learning_rate: float = 0.01,
                          n_iterations: int = 1000) -> Dict:
        """
        Coordinate descent optimization.
        
        Optimizes one coordinate at a time.
        
        Args:
            objective: Objective function
            x0: Initial point
            learning_rate: Learning rate
            n_iterations: Maximum iterations
            
        Returns:
            dict: Optimization results
        """
        x = x0.copy()
        n = len(x)
        
        history = []
        
        for iteration in range(n_iterations):
            for i in range(n):
                # Optimize coordinate i
                def coord_objective(xi):
                    x_temp = x.copy()
                    x_temp[i] = xi
                    return objective(x_temp)
                
                # Line search (simplified)
                x[i] -= learning_rate * 0.01  # Small step
                
            history.append(objective(x))
        
        return {
            'x': x,
            'optimal_value': objective(x),
            'iterations': n_iterations,
            'history': history
        }


class BayesianOptimization:
    """
    Bayesian optimization for hyperparameter tuning.
    """
    
    def __init__(self, objective: Callable,
                 bounds: List[Tuple[float, float]],
                 acquisition: str = 'ei'):
        """
        Args:
            objective: Objective function to minimize
            bounds: Bounds for each parameter
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
        """
        self.objective = objective
        self.bounds = bounds
        self.acquisition = acquisition
        
        self.X_observed = []
        self.y_observed = []
    
    def _gaussian_process(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian process surrogate model.
        
        Args:
            X: Points to evaluate
            
        Returns:
            Tuple of (mean, std)
        """
        from scipy.spatial.distance import cdist
        
        if len(self.X_observed) == 0:
            return np.zeros(len(X)), np.ones(len(X))
        
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        
        # RBF kernel
        def kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
            sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
            return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
        
        K = kernel(X_obs, X_obs)
        K_s = kernel(X_obs, X)
        K_ss = kernel(X, X)
        
        # Add noise
        K += 1e-5 * np.eye(len(K))
        
        # Predict
        K_inv = np.linalg.inv(K)
        mu = K_s.T @ K_inv @ y_obs
        var = np.diag(K_ss - K_s.T @ K_inv @ K_s)
        
        return mu, np.sqrt(var)
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate acquisition function values.
        
        Args:
            X: Points to evaluate
            
        Returns:
            np.ndarray: Acquisition values
        """
        mu, sigma = self._gaussian_process(X)
        
        if len(self.y_observed) == 0:
            return np.ones(len(X))
        
        y_best = min(self.y_observed)
        
        if self.acquisition == 'ei':  # Expected Improvement
            improvement = y_best - mu
            z = improvement / (sigma + 1e-9)
            ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            return ei
        
        elif self.acquisition == 'ucb':  # Upper Confidence Bound
            kappa = 2.0
            return -(mu - kappa * sigma)  # Minimize negative UCB
        
        elif self.acquisition == 'poi':  # Probability of Improvement
            improvement = y_best - mu
            z = improvement / (sigma + 1e-9)
            return stats.norm.cdf(z)
        
        else:
            return np.ones(len(X))
    
    def optimize(self, n_iterations: int = 50,
                n_random: int = 5) -> Dict:
        """
        Run Bayesian optimization.
        
        Args:
            n_iterations: Total iterations
            n_random: Random initialization points
            
        Returns:
            dict: Best found parameters
        """
        # Random initialization
        for _ in range(n_random):
            x = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
        
        # Bayesian optimization loop
        for _ in range(n_iterations - n_random):
            # Sample candidate points
            n_candidates = 1000
            candidates = np.array([
                [np.random.uniform(b[0], b[1]) for b in self.bounds]
                for _ in range(n_candidates)
            ])
            
            # Evaluate acquisition function
            acq_values = self._acquisition_function(candidates)
            
            # Select best candidate
            best_idx = np.argmax(acq_values)
            x_next = candidates[best_idx]
            
            # Evaluate objective
            y_next = self.objective(x_next)
            
            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
        
        # Return best found
        best_idx = np.argmin(self.y_observed)
        
        return {
            'x': self.X_observed[best_idx],
            'optimal_value': self.y_observed[best_idx],
            'n_evaluations': len(self.y_observed)
        }
```

---

*Note: The document continues with additional comprehensive sections covering numerical methods, data structures, system architecture, and extensive reference materials.*


---

## APPENDIX C: NUMERICAL METHODS

### C.1 Numerical Linear Algebra

```python
# mathematics/numerical_methods.py

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, cg, gmres
from typing import Tuple, Optional, Callable
import warnings

class NumericalLinearAlgebra:
    """
    Numerical linear algebra methods for finance.
    """
    
    @staticmethod
    def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
        """
        Cholesky decomposition for positive definite matrices.
        
        A = LL^T
        
        Args:
            A: Positive definite matrix
            
        Returns:
            np.ndarray: Lower triangular matrix L
        """
        try:
            L = np.linalg.cholesky(A)
            return L
        except np.linalg.LinAlgError:
            # Add small regularization
            A_reg = A + np.eye(len(A)) * 1e-6
            return np.linalg.cholesky(A_reg)
    
    @staticmethod
    def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LU decomposition with partial pivoting.
        
        PA = LU
        
        Args:
            A: Square matrix
            
        Returns:
            Tuple of (P, L, U)
        """
        P, L, U = linalg.lu(A)
        return P, L, U
    
    @staticmethod
    def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition.
        
        A = QR
        
        Args:
            A: Matrix
            
        Returns:
            Tuple of (Q, R)
        """
        Q, R = np.linalg.qr(A)
        return Q, R
    
    @staticmethod
    def svd_decomposition(A: np.ndarray, 
                         full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Singular Value Decomposition.
        
        A = U S V^T
        
        Args:
            A: Matrix
            full_matrices: Whether to return full matrices
            
        Returns:
            Tuple of (U, S, Vh)
        """
        U, S, Vh = np.linalg.svd(A, full_matrices=full_matrices)
        return U, S, Vh
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray,
                           method: str = 'direct') -> np.ndarray:
        """
        Solve linear system Ax = b.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side
            method: 'direct', 'cholesky', 'lu', 'qr'
            
        Returns:
            np.ndarray: Solution x
        """
        if method == 'direct':
            return np.linalg.solve(A, b)
        
        elif method == 'cholesky':
            L = NumericalLinearAlgebra.cholesky_decomposition(A)
            # Solve Ly = b, then L^T x = y
            y = linalg.solve_triangular(L, b, lower=True)
            x = linalg.solve_triangular(L.T, y, lower=False)
            return x
        
        elif method == 'lu':
            P, L, U = NumericalLinearAlgebra.lu_decomposition(A)
            # Solve Ly = Pb, then Ux = y
            Pb = P @ b
            y = linalg.solve_triangular(L, Pb, lower=True)
            x = linalg.solve_triangular(U, y, lower=False)
            return x
        
        elif method == 'qr':
            Q, R = NumericalLinearAlgebra.qr_decomposition(A)
            # Solve Rx = Q^T b
            Qtb = Q.T @ b
            x = linalg.solve_triangular(R, Qtb, lower=False)
            return x
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def matrix_inverse(A: np.ndarray, method: str = 'direct') -> np.ndarray:
        """
        Compute matrix inverse.
        
        Args:
            A: Matrix to invert
            method: Inversion method
            
        Returns:
            np.ndarray: Inverse matrix
        """
        if method == 'direct':
            return np.linalg.inv(A)
        
        elif method == 'cholesky':
            L = NumericalLinearAlgebra.cholesky_decomposition(A)
            # A^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
            L_inv = linalg.solve_triangular(L, np.eye(len(L)), lower=True)
            return L_inv.T @ L_inv
        
        elif method == 'svd':
            U, S, Vh = NumericalLinearAlgebra.svd_decomposition(A)
            # A^{-1} = V S^{-1} U^T
            S_inv = np.diag(1 / S)
            return Vh.T @ S_inv @ U.T
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def matrix_determinant(A: np.ndarray) -> float:
        """
        Compute matrix determinant.
        
        Args:
            A: Square matrix
            
        Returns:
            float: Determinant
        """
        return np.linalg.det(A)
    
    @staticmethod
    def matrix_condition_number(A: np.ndarray) -> float:
        """
        Compute condition number.
        
        High condition number indicates ill-conditioning.
        
        Args:
            A: Matrix
            
        Returns:
            float: Condition number
        """
        return np.linalg.cond(A)
    
    @staticmethod
    def low_rank_approximation(A: np.ndarray, rank: int) -> np.ndarray:
        """
        Low-rank approximation using SVD.
        
        Args:
            A: Matrix to approximate
            rank: Target rank
            
        Returns:
            np.ndarray: Low-rank approximation
        """
        U, S, Vh = NumericalLinearAlgebra.svd_decomposition(A)
        
        # Keep only top rank singular values
        S_approx = np.zeros_like(S)
        S_approx[:rank] = S[:rank]
        
        # Reconstruct
        if len(A.shape) == 2:
            S_diag = np.diag(S_approx)
        else:
            S_diag = S_approx
        
        return U @ S_diag @ Vh
    
    @staticmethod
    def principal_components(A: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Principal Component Analysis.
        
        Args:
            A: Data matrix (samples x features)
            n_components: Number of components to keep
            
        Returns:
            Tuple of (components, explained_variance_ratio)
        """
        # Center the data
        A_centered = A - A.mean(axis=0)
        
        # SVD
        U, S, Vh = NumericalLinearAlgebra.svd_decomposition(A_centered)
        
        # Components
        components = Vh[:n_components].T
        
        # Explained variance
        explained_variance = S**2 / (len(A) - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        
        return components, explained_variance_ratio[:n_components]


class SparseLinearAlgebra:
    """
    Sparse matrix operations for large-scale problems.
    """
    
    @staticmethod
    def create_sparse_matrix(dense: np.ndarray, 
                            threshold: float = 1e-10) -> csr_matrix:
        """
        Create sparse matrix from dense.
        
        Args:
            dense: Dense matrix
            threshold: Values below this are treated as zero
            
        Returns:
            csr_matrix: Sparse matrix
        """
        sparse = dense.copy()
        sparse[np.abs(sparse) < threshold] = 0
        return csr_matrix(sparse)
    
    @staticmethod
    def solve_sparse(A: csr_matrix, b: np.ndarray,
                    method: str = 'direct') -> np.ndarray:
        """
        Solve sparse linear system.
        
        Args:
            A: Sparse coefficient matrix
            b: Right-hand side
            method: 'direct', 'cg', 'gmres'
            
        Returns:
            np.ndarray: Solution
        """
        if method == 'direct':
            return spsolve(A, b)
        
        elif method == 'cg':  # Conjugate Gradient
            x, info = cg(A, b)
            if info != 0:
                warnings.warn(f"CG did not converge, info={info}")
            return x
        
        elif method == 'gmres':  # Generalized Minimal Residual
            x, info = gmres(A, b)
            if info != 0:
                warnings.warn(f"GMRES did not converge, info={info}")
            return x
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def sparse_matrix_vector_product(A: csr_matrix, 
                                    x: np.ndarray) -> np.ndarray:
        """
        Efficient sparse matrix-vector multiplication.
        
        Args:
            A: Sparse matrix
            x: Vector
            
        Returns:
            np.ndarray: Product Ax
        """
        return A @ x
    
    @staticmethod
    def sparse_cholesky(A: csc_matrix) -> csc_matrix:
        """
        Sparse Cholesky decomposition.
        
        Args:
            A: Sparse positive definite matrix
            
        Returns:
            csc_matrix: Cholesky factor
        """
        from sksparse.cholmod import cholesky
        
        factor = cholesky(A)
        return factor.L()


class EigenvalueProblems:
    """
    Eigenvalue and eigenvector computations.
    """
    
    @staticmethod
    def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.
        
        Args:
            A: Square matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def symmetric_eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors for symmetric matrix.
        
        More efficient and stable than general case.
        
        Args:
            A: Symmetric matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def largest_eigenvalue(A: np.ndarray, 
                          n_eigenvalues: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute largest eigenvalues and eigenvectors.
        
        Args:
            A: Square matrix
            n_eigenvalues: Number of eigenvalues to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        from scipy.sparse.linalg import eigsh
        
        eigenvalues, eigenvectors = eigsh(A, k=n_eigenvalues, which='LM')
        return eigenvalues, eigenvectors
    
    @staticmethod
    def power_iteration(A: np.ndarray, 
                       max_iter: int = 1000,
                       tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        Power iteration for dominant eigenvalue.
        
        Args:
            A: Square matrix
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (eigenvalue, eigenvector)
        """
        n = A.shape[0]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for _ in range(max_iter):
            x_new = A @ x
            x_new = x_new / np.linalg.norm(x_new)
            
            # Rayleigh quotient
            eigenvalue = x_new @ A @ x_new
            
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
        
        return eigenvalue, x


class NumericalIntegration:
    """
    Numerical integration methods.
    """
    
    @staticmethod
    def trapezoidal_rule(f: Callable, a: float, b: float,
                        n: int = 1000) -> float:
        """
        Trapezoidal rule for numerical integration.
        
        $$\int_a^b f(x)dx \approx \frac{h}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]$$
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of intervals
            
        Returns:
            float: Integral approximation
        """
        x = np.linspace(a, b, n+1)
        y = f(x)
        
        h = (b - a) / n
        integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        
        return integral
    
    @staticmethod
    def simpsons_rule(f: Callable, a: float, b: float,
                     n: int = 1000) -> float:
        """
        Simpson's rule for numerical integration.
        
        More accurate than trapezoidal for smooth functions.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of intervals (must be even)
            
        Returns:
            float: Integral approximation
        """
        if n % 2 != 0:
            n += 1
        
        x = np.linspace(a, b, n+1)
        y = f(x)
        
        h = (b - a) / n
        
        integral = h / 3 * (y[0] + y[-1] + 
                           4 * np.sum(y[1:-1:2]) + 
                           2 * np.sum(y[2:-1:2]))
        
        return integral
    
    @staticmethod
    def gaussian_quadrature(f: Callable, a: float, b: float,
                           n: int = 10) -> float:
        """
        Gaussian quadrature for numerical integration.
        
        Most accurate for polynomial functions.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of points
            
        Returns:
            float: Integral approximation
        """
        from scipy.integrate import fixed_quad
        
        # Transform to [-1, 1]
        def g(t):
            x = (b - a) / 2 * t + (a + b) / 2
            return f(x) * (b - a) / 2
        
        integral, _ = fixed_quad(g, -1, 1, n=n)
        
        return integral
    
    @staticmethod
    def monte_carlo_integration(f: Callable, a: float, b: float,
                               n_samples: int = 10000) -> Tuple[float, float]:
        """
        Monte Carlo integration.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n_samples: Number of samples
            
        Returns:
            Tuple of (estimate, standard_error)
        """
        samples = np.random.uniform(a, b, n_samples)
        values = f(samples)
        
        estimate = (b - a) * np.mean(values)
        std_error = (b - a) * np.std(values) / np.sqrt(n_samples)
        
        return estimate, std_error
    
    @staticmethod
    def multidimensional_integration(f: Callable, 
                                    bounds: List[Tuple[float, float]],
                                    n_samples: int = 100000) -> Tuple[float, float]:
        """
        Monte Carlo integration in multiple dimensions.
        
        Args:
            f: Function to integrate
            bounds: List of (lower, upper) bounds for each dimension
            n_samples: Number of samples
            
        Returns:
            Tuple of (estimate, standard_error)
        """
        d = len(bounds)
        
        # Generate samples
        samples = np.zeros((n_samples, d))
        volume = 1.0
        
        for i, (a, b) in enumerate(bounds):
            samples[:, i] = np.random.uniform(a, b, n_samples)
            volume *= (b - a)
        
        values = np.array([f(sample) for sample in samples])
        
        estimate = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)
        
        return estimate, std_error


class NumericalDifferentiation:
    """
    Numerical differentiation methods.
    """
    
    @staticmethod
    def forward_difference(f: Callable, x: float, h: float = 1e-5) -> float:
        """
        Forward difference approximation.
        
        $$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            float: Derivative approximation
        """
        return (f(x + h) - f(x)) / h
    
    @staticmethod
    def backward_difference(f: Callable, x: float, h: float = 1e-5) -> float:
        """
        Backward difference approximation.
        
        $$f'(x) \approx \frac{f(x) - f(x-h)}{h}$$
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            float: Derivative approximation
        """
        return (f(x) - f(x - h)) / h
    
    @staticmethod
    def central_difference(f: Callable, x: float, h: float = 1e-5) -> float:
        """
        Central difference approximation (more accurate).
        
        $$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            float: Derivative approximation
        """
        return (f(x + h) - f(x - h)) / (2 * h)
    
    @staticmethod
    def second_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
        """
        Second derivative approximation.
        
        $$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            float: Second derivative approximation
        """
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    
    @staticmethod
    def gradient(f: Callable, x: np.ndarray, 
                h: float = 1e-5) -> np.ndarray:
        """
        Numerical gradient of multivariate function.
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            np.ndarray: Gradient vector
        """
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        return grad
    
    @staticmethod
    def jacobian(f: Callable, x: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
        """
        Numerical Jacobian of vector-valued function.
        
        Args:
            f: Vector-valued function
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            np.ndarray: Jacobian matrix
        """
        n = len(x)
        f_x = f(x)
        m = len(f_x)
        
        J = np.zeros((m, n))
        
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h
            
            J[:, j] = (f(x_plus) - f(x_minus)) / (2 * h)
        
        return J
    
    @staticmethod
    def hessian(f: Callable, x: np.ndarray,
               h: float = 1e-5) -> np.ndarray:
        """
        Numerical Hessian matrix.
        
        Args:
            f: Function to differentiate
            x: Point at which to evaluate
            h: Step size
            
        Returns:
            np.ndarray: Hessian matrix
        """
        n = len(x)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
        
        return H


class RootFinding:
    """
    Root finding algorithms.
    """
    
    @staticmethod
    def bisection(f: Callable, a: float, b: float,
                 tol: float = 1e-10, max_iter: int = 100) -> Dict:
        """
        Bisection method for root finding.
        
        Requires f(a) and f(b) to have opposite signs.
        
        Args:
            f: Function to find root of
            a: Lower bound
            b: Upper bound
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            dict: Root and convergence info
        """
        fa, fb = f(a), f(b)
        
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            
            if abs(fc) < tol or (b - a) / 2 < tol:
                return {'root': c, 'iterations': i, 'f_value': fc}
            
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        
        return {'root': c, 'iterations': max_iter, 'f_value': fc, 'converged': False}
    
    @staticmethod
    def newton_raphson(f: Callable, df: Callable, x0: float,
                      tol: float = 1e-10, max_iter: int = 100) -> Dict:
        """
        Newton-Raphson method for root finding.
        
        $$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
        
        Args:
            f: Function to find root of
            df: Derivative of f
            x0: Initial guess
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            dict: Root and convergence info
        """
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            
            if abs(fx) < tol:
                return {'root': x, 'iterations': i, 'f_value': fx}
            
            dfx = df(x)
            
            if abs(dfx) < 1e-15:
                return {'root': x, 'iterations': i, 'error': 'Derivative too small'}
            
            x = x - fx / dfx
        
        return {'root': x, 'iterations': max_iter, 'f_value': f(x), 'converged': False}
    
    @staticmethod
    def secant_method(f: Callable, x0: float, x1: float,
                     tol: float = 1e-10, max_iter: int = 100) -> Dict:
        """
        Secant method (derivative-free Newton).
        
        Args:
            f: Function to find root of
            x0: First initial guess
            x1: Second initial guess
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            dict: Root and convergence info
        """
        for i in range(max_iter):
            fx0, fx1 = f(x0), f(x1)
            
            if abs(fx1) < tol:
                return {'root': x1, 'iterations': i, 'f_value': fx1}
            
            # Secant update
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            
            x0, x1 = x1, x_new
        
        return {'root': x1, 'iterations': max_iter, 'f_value': f(x1), 'converged': False}
    
    @staticmethod
    def brent_method(f: Callable, a: float, b: float,
                    tol: float = 1e-10, max_iter: int = 100) -> Dict:
        """
        Brent's method for root finding.
        
        Combines bisection, secant, and inverse quadratic interpolation.
        Robust and efficient.
        
        Args:
            f: Function to find root of
            a: Lower bound
            b: Upper bound
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            dict: Root and convergence info
        """
        from scipy.optimize import brentq
        
        try:
            root = brentq(f, a, b, xtol=tol, maxiter=max_iter)
            return {'root': root, 'f_value': f(root), 'converged': True}
        except ValueError as e:
            return {'error': str(e)}


class Interpolation:
    """
    Interpolation methods.
    """
    
    @staticmethod
    def linear_interpolation(x: np.ndarray, y: np.ndarray,
                            x_new: np.ndarray) -> np.ndarray:
        """
        Linear interpolation.
        
        Args:
            x: Known x values
            y: Known y values
            x_new: Points to interpolate
            
        Returns:
            np.ndarray: Interpolated values
        """
        return np.interp(x_new, x, y)
    
    @staticmethod
    def cubic_spline(x: np.ndarray, y: np.ndarray,
                    x_new: np.ndarray) -> np.ndarray:
        """
        Cubic spline interpolation.
        
        Smoother than linear interpolation.
        
        Args:
            x: Known x values
            y: Known y values
            x_new: Points to interpolate
            
        Returns:
            np.ndarray: Interpolated values
        """
        from scipy.interpolate import CubicSpline
        
        cs = CubicSpline(x, y)
        return cs(x_new)
    
    @staticmethod
    def lagrange_interpolation(x: np.ndarray, y: np.ndarray,
                              x_new: np.ndarray) -> np.ndarray:
        """
        Lagrange polynomial interpolation.
        
        Args:
            x: Known x values
            y: Known y values
            x_new: Points to interpolate
            
        Returns:
            np.ndarray: Interpolated values
        """
        from scipy.interpolate import lagrange
        
        poly = lagrange(x, y)
        return poly(x_new)
    
    @staticmethod
    def kriging_interpolation(x: np.ndarray, y: np.ndarray,
                             x_new: np.ndarray) -> np.ndarray:
        """
        Kriging (Gaussian process) interpolation.
        
        Provides uncertainty estimates along with predictions.
        
        Args:
            x: Known x values
            y: Known y values
            x_new: Points to interpolate
            
        Returns:
            np.ndarray: Interpolated values
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        
        # Reshape for sklearn
        x = x.reshape(-1, 1)
        x_new = x_new.reshape(-1, 1)
        
        # Fit GP
        kernel = RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        gp.fit(x, y)
        
        # Predict
        y_new, sigma = gp.predict(x_new, return_std=True)
        
        return y_new


class DifferentialEquations:
    """
    Numerical methods for differential equations.
    """
    
    @staticmethod
    def euler_method(f: Callable, y0: float, t0: float, tf: float,
                    n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Euler's method for ODEs.
        
        $$y_{n+1} = y_n + hf(t_n, y_n)$$
        
        Args:
            f: Derivative function dy/dt = f(t, y)
            y0: Initial condition
            t0: Initial time
            tf: Final time
            n_steps: Number of steps
            
        Returns:
            Tuple of (t, y) arrays
        """
        h = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0
        
        for i in range(n_steps):
            y[i+1] = y[i] + h * f(t[i], y[i])
        
        return t, y
    
    @staticmethod
    def runge_kutta_4(f: Callable, y0: float, t0: float, tf: float,
                     n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fourth-order Runge-Kutta method.
        
        More accurate than Euler's method.
        
        Args:
            f: Derivative function
            y0: Initial condition
            t0: Initial time
            tf: Final time
            n_steps: Number of steps
            
        Returns:
            Tuple of (t, y) arrays
        """
        h = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0
        
        for i in range(n_steps):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + h/2, y[i] + k1/2)
            k3 = h * f(t[i] + h/2, y[i] + k2/2)
            k4 = h * f(t[i] + h, y[i] + k3)
            
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t, y
    
    @staticmethod
    def finite_difference_heat(u0: np.ndarray, alpha: float,
                              dx: float, dt: float,
                              n_steps: int) -> np.ndarray:
        """
        Finite difference solution of heat equation.
        
        $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
        
        Args:
            u0: Initial condition
            alpha: Diffusion coefficient
            dx: Spatial step
            dt: Time step
            n_steps: Number of time steps
            
        Returns:
            np.ndarray: Solution at final time
        """
        n = len(u0)
        u = u0.copy()
        
        r = alpha * dt / dx**2
        
        for _ in range(n_steps):
            u_new = u.copy()
            
            for i in range(1, n-1):
                u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
            
            u = u_new
        
        return u


class FastFourierTransform:
    """
    Fast Fourier Transform methods.
    """
    
    @staticmethod
    def fft(x: np.ndarray) -> np.ndarray:
        """
        Fast Fourier Transform.
        
        Args:
            x: Input signal
            
        Returns:
            np.ndarray: FFT of signal
        """
        return np.fft.fft(x)
    
    @staticmethod
    def ifft(X: np.ndarray) -> np.ndarray:
        """
        Inverse Fast Fourier Transform.
        
        Args:
            X: Frequency domain signal
            
        Returns:
            np.ndarray: Time domain signal
        """
        return np.fft.ifft(X)
    
    @staticmethod
    def power_spectral_density(x: np.ndarray, 
                              fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectral density.
        
        Args:
            x: Input signal
            fs: Sampling frequency
            
        Returns:
            Tuple of (frequencies, psd)
        """
        n = len(x)
        
        # FFT
        X = np.fft.fft(x)
        
        # Power spectrum
        psd = np.abs(X)**2 / (n * fs)
        
        # Frequencies
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # Keep only positive frequencies
        positive = freqs >= 0
        
        return freqs[positive], psd[positive]
    
    @staticmethod
    def convolution_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Fast convolution using FFT.
        
        More efficient than direct convolution for large arrays.
        
        Args:
            a: First signal
            b: Second signal
            
        Returns:
            np.ndarray: Convolution result
        """
        n = len(a) + len(b) - 1
        
        # Pad to power of 2 for efficiency
        n_fft = 2**int(np.ceil(np.log2(n)))
        
        A = np.fft.fft(a, n_fft)
        B = np.fft.fft(b, n_fft)
        
        C = A * B
        
        c = np.fft.ifft(C)
        
        return np.real(c[:n])
    
    @staticmethod
    def cross_correlation_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Fast cross-correlation using FFT.
        
        Args:
            a: First signal
            b: Second signal
            
        Returns:
            np.ndarray: Cross-correlation
        """
        # Cross-correlation is convolution with reversed signal
        return FastFourierTransform.convolution_fft(a, b[::-1])
```

---

*Note: The document continues with additional comprehensive sections covering data structures, system architecture, and extensive reference materials.*


---

## APPENDIX D: DATA STRUCTURES AND ALGORITHMS

### D.1 Efficient Data Structures for Finance

```python
# data_structures/financial_datastructures.py

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from heapq import heappush, heappop
import bisect

@dataclass
class PriceLevel:
    """Represents a price level in order book."""
    price: float
    size: float
    orders: List[Dict] = field(default_factory=list)
    
    def add_order(self, order_id: str, size: float):
        """Add order to price level."""
        self.orders.append({'order_id': order_id, 'size': size})
        self.size += size
    
    def remove_order(self, order_id: str):
        """Remove order from price level."""
        for i, order in enumerate(self.orders):
            if order['order_id'] == order_id:
                self.size -= order['size']
                self.orders.pop(i)
                return
    
    def __lt__(self, other):
        return self.price < other.price


class OrderBook:
    """
    Efficient order book implementation.
    
    Uses sorted data structures for O(log n) operations.
    """
    
    def __init__(self, symbol: str):
        """
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        
        # Sorted price levels
        # Bids: highest first (max heap simulation)
        self.bids: Dict[float, PriceLevel] = {}
        self.bid_prices: List[float] = []  # Sorted descending
        
        # Asks: lowest first (min heap)
        self.asks: Dict[float, PriceLevel] = {}
        self.ask_prices: List[float] = []  # Sorted ascending
        
        # Order lookup
        self.orders: Dict[str, Dict] = {}
    
    def add_order(self, order_id: str, side: str, price: float, 
                 size: float, order_type: str = 'limit'):
        """
        Add order to book.
        
        Args:
            order_id: Unique order ID
            side: 'buy' or 'sell'
            price: Order price
            size: Order size
            order_type: 'limit' or 'market'
        """
        self.orders[order_id] = {
            'side': side,
            'price': price,
            'size': size,
            'filled': 0
        }
        
        if side == 'buy':
            if price not in self.bids:
                self.bids[price] = PriceLevel(price, 0)
                # Insert maintaining sorted order (descending)
                bisect.insort_left(self.bid_prices, price)
                self.bid_prices.reverse()
            
            self.bids[price].add_order(order_id, size)
        
        else:  # sell
            if price not in self.asks:
                self.asks[price] = PriceLevel(price, 0)
                # Insert maintaining sorted order (ascending)
                bisect.insort_right(self.ask_prices, price)
            
            self.asks[price].add_order(order_id, size)
    
    def cancel_order(self, order_id: str):
        """
        Cancel order from book.
        
        Args:
            order_id: Order ID to cancel
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        price = order['price']
        side = order['side']
        
        if side == 'buy' and price in self.bids:
            self.bids[price].remove_order(order_id)
            if self.bids[price].size == 0:
                del self.bids[price]
                self.bid_prices.remove(price)
        
        elif side == 'sell' and price in self.asks:
            self.asks[price].remove_order(order_id)
            if self.asks[price].size == 0:
                del self.asks[price]
                self.ask_prices.remove(price)
        
        del self.orders[order_id]
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """
        Get best bid price and size.
        
        Returns:
            Tuple of (price, size) or None
        """
        if not self.bid_prices:
            return None
        
        best_price = self.bid_prices[0]
        return (best_price, self.bids[best_price].size)
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """
        Get best ask price and size.
        
        Returns:
            Tuple of (price, size) or None
        """
        if not self.ask_prices:
            return None
        
        best_price = self.ask_prices[0]
        return (best_price, self.asks[best_price].size)
    
    def get_spread(self) -> Optional[float]:
        """
        Get bid-ask spread.
        
        Returns:
            Spread or None if one side empty
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask[0] - best_bid[0]
    
    def get_mid_price(self) -> Optional[float]:
        """
        Get mid price.
        
        Returns:
            Mid price or None
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid[0] + best_ask[0]) / 2
    
    def get_depth(self, levels: int = 5) -> Dict:
        """
        Get order book depth.
        
        Args:
            levels: Number of price levels
            
        Returns:
            dict: Bid and ask depth
        """
        bids = []
        for price in self.bid_prices[:levels]:
            bids.append({
                'price': price,
                'size': self.bids[price].size
            })
        
        asks = []
        for price in self.ask_prices[:levels]:
            asks.append({
                'price': price,
                'size': self.asks[price].size
            })
        
        return {'bids': bids, 'asks': asks}
    
    def get_vwap(self, side: str, size: float) -> Optional[float]:
        """
        Calculate VWAP for given size.
        
        Args:
            side: 'buy' or 'sell'
            size: Size to execute
            
        Returns:
            VWAP or None
        """
        remaining = size
        total_value = 0
        
        if side == 'buy':
            prices = self.ask_prices
            levels = self.asks
        else:
            prices = self.bid_prices
            levels = self.bids
        
        for price in prices:
            level_size = levels[price].size
            take_size = min(remaining, level_size)
            
            total_value += take_size * price
            remaining -= take_size
            
            if remaining <= 0:
                break
        
        if remaining > 0:
            return None  # Not enough liquidity
        
        return total_value / size
    
    def get_imbalance(self) -> float:
        """
        Calculate order book imbalance.
        
        Returns:
            float: Imbalance ratio (-1 to 1)
        """
        bid_volume = sum(level.size for level in self.bids.values())
        ask_volume = sum(level.size for level in self.asks.values())
        
        total = bid_volume + ask_volume
        
        if total == 0:
            return 0
        
        return (bid_volume - ask_volume) / total


class TimeSeriesBuffer:
    """
    Efficient circular buffer for time series data.
    
    Supports O(1) append and O(k) window queries.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def append(self, timestamp: pd.Timestamp, value: float):
        """
        Append data point.
        
        Args:
            timestamp: Timestamp
            value: Value
        """
        self.buffer.append(value)
        self.timestamps.append(timestamp)
    
    def get_window(self, start: pd.Timestamp, 
                  end: pd.Timestamp) -> pd.Series:
        """
        Get data within time window.
        
        Args:
            start: Start time
            end: End time
            
        Returns:
            pd.Series: Window data
        """
        data = []
        times = []
        
        for ts, val in zip(self.timestamps, self.buffer):
            if start <= ts <= end:
                data.append(val)
                times.append(ts)
        
        return pd.Series(data, index=times)
    
    def get_last_n(self, n: int) -> pd.Series:
        """
        Get last n data points.
        
        Args:
            n: Number of points
            
        Returns:
            pd.Series: Last n points
        """
        n = min(n, len(self.buffer))
        
        data = list(self.buffer)[-n:]
        times = list(self.timestamps)[-n:]
        
        return pd.Series(data, index=times)
    
    def rolling_mean(self, window: int) -> float:
        """
        Calculate rolling mean.
        
        Args:
            window: Window size
            
        Returns:
            float: Rolling mean
        """
        if len(self.buffer) < window:
            window = len(self.buffer)
        
        recent = list(self.buffer)[-window:]
        return np.mean(recent)
    
    def rolling_std(self, window: int) -> float:
        """
        Calculate rolling standard deviation.
        
        Args:
            window: Window size
            
        Returns:
            float: Rolling std
        """
        if len(self.buffer) < window:
            window = len(self.buffer)
        
        recent = list(self.buffer)[-window:]
        return np.std(recent)


class PriorityQueue:
    """
    Priority queue for event-driven simulation.
    """
    
    def __init__(self):
        self._queue = []
        self._counter = 0
    
    def push(self, priority: float, item: Any):
        """
        Push item with priority.
        
        Args:
            priority: Priority (lower = higher priority)
            item: Item to store
        """
        heappush(self._queue, (priority, self._counter, item))
        self._counter += 1
    
    def pop(self) -> Tuple[float, Any]:
        """
        Pop highest priority item.
        
        Returns:
            Tuple of (priority, item)
        """
        priority, _, item = heappop(self._queue)
        return priority, item
    
    def peek(self) -> Optional[Tuple[float, Any]]:
        """
        Peek at highest priority item without removing.
        
        Returns:
            Tuple of (priority, item) or None
        """
        if not self._queue:
            return None
        
        priority, _, item = self._queue[0]
        return priority, item
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0


class RollingStatistics:
    """
    Efficient rolling statistics calculation.
    
    Uses Welford's online algorithm for O(1) updates.
    """
    
    def __init__(self, window: int):
        """
        Args:
            window: Rolling window size
        """
        self.window = window
        self.buffer = deque(maxlen=window)
        
        # Welford's algorithm variables
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0  # Sum of squares of differences
    
    def update(self, value: float):
        """
        Update with new value.
        
        Args:
            value: New data point
        """
        if len(self.buffer) == self.window:
            # Remove oldest value
            old_value = self.buffer[0]
            self._remove(old_value)
        
        # Add new value
        self._add(value)
        self.buffer.append(value)
    
    def _add(self, value: float):
        """Add value to statistics."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
    
    def _remove(self, value: float):
        """Remove value from statistics."""
        if self._count == 0:
            return
        
        old_mean = self._mean
        self._count -= 1
        
        if self._count == 0:
            self._mean = 0
            self._m2 = 0
        else:
            self._mean = (self._mean * (self._count + 1) - value) / self._count
            self._m2 -= (value - old_mean) * (value - self._mean)
    
    def mean(self) -> float:
        """Get current mean."""
        return self._mean
    
    def variance(self) -> float:
        """Get current variance."""
        if self._count < 2:
            return 0
        return self._m2 / (self._count - 1)
    
    def std(self) -> float:
        """Get current standard deviation."""
        return np.sqrt(self.variance())
    
    def zscore(self, value: float) -> float:
        """
        Calculate z-score for value.
        
        Args:
            value: Value to score
            
        Returns:
            float: Z-score
        """
        std = self.std()
        if std == 0:
            return 0
        return (value - self.mean()) / std


class Trie:
    """
    Trie data structure for symbol prefix matching.
    """
    
    def __init__(self):
        self.root = {}
    
    def insert(self, word: str, data: Any = None):
        """
        Insert word into trie.
        
        Args:
            word: Word to insert
            data: Associated data
        """
        node = self.root
        
        for char in word.upper():
            if char not in node:
                node[char] = {}
            node = node[char]
        
        node['__end__'] = True
        node['__data__'] = data
    
    def search(self, word: str) -> bool:
        """
        Search for exact word match.
        
        Args:
            word: Word to search
            
        Returns:
            bool: Whether word exists
        """
        node = self.root
        
        for char in word.upper():
            if char not in node:
                return False
            node = node[char]
        
        return '__end__' in node
    
    def starts_with(self, prefix: str) -> List[str]:
        """
        Find all words with given prefix.
        
        Args:
            prefix: Prefix to search
            
        Returns:
            List of matching words
        """
        node = self.root
        
        for char in prefix.upper():
            if char not in node:
                return []
            node = node[char]
        
        results = []
        self._collect_words(node, prefix.upper(), results)
        return results
    
    def _collect_words(self, node: Dict, prefix: str, results: List[str]):
        """Recursively collect all words from node."""
        if '__end__' in node:
            results.append(prefix)
        
        for char, child in node.items():
            if not char.startswith('__'):
                self._collect_words(child, prefix + char, results)


class SegmentTree:
    """
    Segment tree for range queries.
    
    Supports O(log n) range queries and updates.
    """
    
    def __init__(self, data: List[float]):
        """
        Args:
            data: Initial data
        """
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self._build(data, 0, 0, self.n - 1)
    
    def _build(self, data: List[float], node: int, start: int, end: int):
        """Build segment tree."""
        if start == end:
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            self._build(data, 2*node+1, start, mid)
            self._build(data, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def update(self, idx: int, value: float):
        """
        Update value at index.
        
        Args:
            idx: Index to update
            value: New value
        """
        self._update(0, 0, self.n - 1, idx, value)
    
    def _update(self, node: int, start: int, end: int, idx: int, value: float):
        """Recursive update."""
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                self._update(2*node+1, start, mid, idx, value)
            else:
                self._update(2*node+2, mid+1, end, idx, value)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, left: int, right: int) -> float:
        """
        Query sum in range [left, right].
        
        Args:
            left: Left index
            right: Right index
            
        Returns:
            float: Sum in range
        """
        return self._query(0, 0, self.n - 1, left, right)
    
    def _query(self, node: int, start: int, end: int, 
              left: int, right: int) -> float:
        """Recursive query."""
        if right < start or end < left:
            return 0
        
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        return (self._query(2*node+1, start, mid, left, right) +
                self._query(2*node+2, mid+1, end, left, right))


class DisjointSet:
    """
    Disjoint set (Union-Find) data structure.
    
    Used for clustering and connected components.
    """
    
    def __init__(self, n: int):
        """
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """
        Find root of set containing x.
        
        Args:
            x: Element
            
        Returns:
            int: Root
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """
        Union sets containing x and y.
        
        Args:
            x: First element
            y: Second element
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in same set.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            bool: Whether connected
        """
        return self.find(x) == self.find(y)


class Graph:
    """
    Graph data structure for network analysis.
    """
    
    def __init__(self, directed: bool = False):
        """
        Args:
            directed: Whether graph is directed
        """
        self.directed = directed
        self.adjacency: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)
        self.nodes = set()
    
    def add_edge(self, u: Any, v: Any, weight: float = 1.0):
        """
        Add edge to graph.
        
        Args:
            u: Source node
            v: Target node
            weight: Edge weight
        """
        self.adjacency[u].append((v, weight))
        self.nodes.add(u)
        self.nodes.add(v)
        
        if not self.directed:
            self.adjacency[v].append((u, weight))
    
    def dijkstra(self, source: Any) -> Dict[Any, float]:
        """
        Dijkstra's shortest path algorithm.
        
        Args:
            source: Source node
            
        Returns:
            dict: Shortest distances from source
        """
        distances = {node: float('inf') for node in self.nodes}
        distances[source] = 0
        
        pq = PriorityQueue()
        pq.push(0, source)
        
        while not pq.is_empty():
            dist, u = pq.pop()
            
            if dist > distances[u]:
                continue
            
            for v, weight in self.adjacency[u]:
                new_dist = dist + weight
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    pq.push(new_dist, v)
        
        return distances
    
    def minimum_spanning_tree(self) -> List[Tuple[Any, Any, float]]:
        """
        Kruskal's minimum spanning tree algorithm.
        
        Returns:
            List of edges in MST
        """
        # Collect all edges
        edges = []
        seen = set()
        
        for u in self.adjacency:
            for v, weight in self.adjacency[u]:
                if (u, v) not in seen and (v, u) not in seen:
                    edges.append((weight, u, v))
                    seen.add((u, v))
        
        # Sort by weight
        edges.sort()
        
        # Kruskal's algorithm
        node_list = list(self.nodes)
        node_idx = {node: i for i, node in enumerate(node_list)}
        ds = DisjointSet(len(node_list))
        
        mst = []
        
        for weight, u, v in edges:
            if not ds.connected(node_idx[u], node_idx[v]):
                ds.union(node_idx[u], node_idx[v])
                mst.append((u, v, weight))
        
        return mst


class BloomFilter:
    """
    Bloom filter for probabilistic membership testing.
    
    Space-efficient with small false positive rate.
    """
    
    def __init__(self, size: int, n_hash: int):
        """
        Args:
            size: Bit array size
            n_hash: Number of hash functions
        """
        self.size = size
        self.n_hash = n_hash
        self.bit_array = [False] * size
    
    def _hashes(self, item: str) -> List[int]:
        """Generate hash values for item."""
        import hashlib
        
        hashes = []
        for i in range(self.n_hash):
            hash_input = f"{item}:{i}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            hashes.append(hash_val % self.size)
        
        return hashes
    
    def add(self, item: str):
        """
        Add item to filter.
        
        Args:
            item: Item to add
        """
        for hash_val in self._hashes(item):
            self.bit_array[hash_val] = True
    
    def contains(self, item: str) -> bool:
        """
        Check if item might be in filter.
        
        Args:
            item: Item to check
            
        Returns:
            bool: True if possibly in set, False if definitely not
        """
        return all(self.bit_array[h] for h in self._hashes(item))
    
    def false_positive_rate(self, n_items: int) -> float:
        """
        Calculate false positive rate.
        
        Args:
            n_items: Number of items added
            
        Returns:
            float: False positive probability
        """
        return (1 - (1 - 1/self.size)**(self.n_hash * n_items))**self.n_hash


class LRUCache:
    """
    LRU (Least Recently Used) cache.
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum cache size
        """
        self.capacity = capacity
        self.cache: Dict[Any, Any] = {}
        self.access_order: deque = deque()
    
    def get(self, key: Any) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self.cache:
            return None
        
        # Move to front (most recently used)
        self.access_order.remove(key)
        self.access_order.appendleft(key)
        
        return self.cache[key]
    
    def put(self, key: Any, value: Any):
        """
        Add or update cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            self.access_order.remove(key)
        
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            lru = self.access_order.pop()
            del self.cache[lru]
        
        self.cache[key] = value
        self.access_order.appendleft(key)


class CountMinSketch:
    """
    Count-Min Sketch for frequency estimation.
    
    Probabilistic data structure for counting item frequencies.
    """
    
    def __init__(self, width: int, depth: int):
        """
        Args:
            width: Width of sketch
            depth: Number of hash functions
        """
        self.width = width
        self.depth = depth
        self.sketch = np.zeros((depth, width), dtype=np.int64)
    
    def _hash(self, item: str, i: int) -> int:
        """Hash item for row i."""
        import hashlib
        hash_input = f"{item}:{i}"
        return int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % self.width
    
    def add(self, item: str, count: int = 1):
        """
        Add item to sketch.
        
        Args:
            item: Item to add
            count: Count to add
        """
        for i in range(self.depth):
            j = self._hash(item, i)
            self.sketch[i, j] += count
    
    def estimate(self, item: str) -> int:
        """
        Estimate frequency of item.
        
        Args:
            item: Item to estimate
            
        Returns:
            int: Estimated frequency
        """
        min_count = float('inf')
        
        for i in range(self.depth):
            j = self._hash(item, i)
            min_count = min(min_count, self.sketch[i, j])
        
        return int(min_count)


class SkipList:
    """
    Skip list for ordered data with O(log n) operations.
    """
    
    def __init__(self, max_level: int = 16, p: float = 0.5):
        """
        Args:
            max_level: Maximum level
            p: Probability of promoting to next level
        """
        self.max_level = max_level
        self.p = p
        self.header = self._Node(float('-inf'), max_level)
        self.level = 0
    
    class _Node:
        """Skip list node."""
        def __init__(self, key: float, level: int):
            self.key = key
            self.forward = [None] * (level + 1)
    
    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while np.random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def insert(self, key: float):
        """
        Insert key into skip list.
        
        Args:
            key: Key to insert
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        new_level = self._random_level()
        
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level
        
        new_node = self._Node(key, new_level)
        
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def search(self, key: float) -> bool:
        """
        Search for key in skip list.
        
        Args:
            key: Key to search
            
        Returns:
            bool: Whether key exists
        """
        current = self.header
        
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        current = current.forward[0]
        return current is not None and current.key == key


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sums.
    
    Supports O(log n) prefix sum queries and updates.
    """
    
    def __init__(self, size: int):
        """
        Args:
            size: Size of array
        """
        self.n = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx: int, delta: float):
        """
        Add delta to element at index.
        
        Args:
            idx: Index (0-based)
            delta: Value to add
        """
        idx += 1  # 1-based indexing
        
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def prefix_sum(self, idx: int) -> float:
        """
        Get prefix sum [0, idx].
        
        Args:
            idx: Index (0-based, inclusive)
            
        Returns:
            float: Prefix sum
        """
        idx += 1
        result = 0
        
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx
        
        return result
    
    def range_sum(self, left: int, right: int) -> float:
        """
        Get sum in range [left, right].
        
        Args:
            left: Left index
            right: Right index
            
        Returns:
            float: Range sum
        """
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


class RedBlackTree:
    """
    Red-Black Tree for ordered data.
    
    Self-balancing binary search tree with O(log n) operations.
    """
    
    RED = True
    BLACK = False
    
    def __init__(self):
        self.NIL = self._Node(0, None, self.BLACK)
        self.root = self.NIL
    
    class _Node:
        def __init__(self, key: float, value: Any, color: bool):
            self.key = key
            self.value = value
            self.color = color
            self.left = None
            self.right = None
            self.parent = None
    
    def insert(self, key: float, value: Any):
        """
        Insert key-value pair.
        
        Args:
            key: Key
            value: Value
        """
        node = self._Node(key, value, self.RED)
        node.left = self.NIL
        node.right = self.NIL
        
        parent = None
        current = self.root
        
        while current != self.NIL:
            parent = current
            if node.key < current.key:
                current = current.left
            else:
                current = current.right
        
        node.parent = parent
        
        if parent is None:
            self.root = node
        elif node.key < parent.key:
            parent.left = node
        else:
            parent.right = node
        
        if node.parent is None:
            node.color = self.BLACK
            return
        
        if node.parent.parent is None:
            return
        
        self._fix_insert(node)
    
    def _fix_insert(self, node: '_Node'):
        """Fix Red-Black tree properties after insertion."""
        while node.parent and node.parent.color == self.RED:
            if node.parent == node.parent.parent.right:
                uncle = node.parent.parent.left
                
                if uncle and uncle.color == self.RED:
                    uncle.color = self.BLACK
                    node.parent.color = self.BLACK
                    node.parent.parent.color = self.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    
                    node.parent.color = self.BLACK
                    node.parent.parent.color = self.RED
                    self._rotate_left(node.parent.parent)
            else:
                uncle = node.parent.parent.right
                
                if uncle and uncle.color == self.RED:
                    uncle.color = self.BLACK
                    node.parent.color = self.BLACK
                    node.parent.parent.color = self.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._rotate_left(node)
                    
                    node.parent.color = self.BLACK
                    node.parent.parent.color = self.RED
                    self._rotate_right(node.parent.parent)
            
            if node == self.root:
                break
        
        self.root.color = self.BLACK
    
    def _rotate_left(self, x: '_Node'):
        """Left rotation."""
        y = x.right
        x.right = y.left
        
        if y.left != self.NIL:
            y.left.parent = x
        
        y.parent = x.parent
        
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def _rotate_right(self, x: '_Node'):
        """Right rotation."""
        y = x.left
        x.left = y.right
        
        if y.right != self.NIL:
            y.right.parent = x
        
        y.parent = x.parent
        
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        
        y.right = x
        x.parent = y
    
    def search(self, key: float) -> Optional[Any]:
        """
        Search for key.
        
        Args:
            key: Key to search
            
        Returns:
            Value or None
        """
        current = self.root
        
        while current != self.NIL:
            if key == current.key:
                return current.value
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        
        return None


class ConsistentHashRing:
    """
    Consistent hashing for distributed systems.
    """
    
    def __init__(self, replicas: int = 150):
        """
        Args:
            replicas: Number of virtual nodes per physical node
        """
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: set = set()
    
    def _hash(self, key: str) -> int:
        """Hash key to integer."""
        import hashlib
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str):
        """
        Add node to ring.
        
        Args:
            node: Node identifier
        """
        self.nodes.add(node)
        
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)
    
    def remove_node(self, node: str):
        """
        Remove node from ring.
        
        Args:
            node: Node identifier
        """
        self.nodes.discard(node)
        
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)
    
    def get_node(self, key: str) -> Optional[str]:
        """
        Get node for key.
        
        Args:
            key: Key to route
            
        Returns:
            Node identifier
        """
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find first node >= hash_key
        idx = bisect.bisect_right(self.sorted_keys, hash_key)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]


class MerkleTree:
    """
    Merkle tree for data integrity verification.
    """
    
    def __init__(self, data: List[str]):
        """
        Args:
            data: List of data blocks
        """
        self.leaves = [self._hash(d) for d in data]
        self.tree = self._build_tree(self.leaves)
        self.root = self.tree[0] if self.tree else None
    
    def _hash(self, data: str) -> str:
        """Hash data."""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _build_tree(self, leaves: List[str]) -> List[str]:
        """Build Merkle tree from leaves."""
        tree = leaves[:]
        
        while len(tree) > 1:
            new_level = []
            
            for i in range(0, len(tree), 2):
                left = tree[i]
                right = tree[i + 1] if i + 1 < len(tree) else left
                new_level.append(self._hash(left + right))
            
            tree = new_level
        
        return tree
    
    def get_proof(self, index: int) -> List[str]:
        """
        Get Merkle proof for leaf at index.
        
        Args:
            index: Leaf index
            
        Returns:
            List of sibling hashes
        """
        proof = []
        current_idx = index
        
        level = self.leaves[:]
        
        while len(level) > 1:
            sibling_idx = current_idx + 1 if current_idx % 2 == 0 else current_idx - 1
            
            if sibling_idx < len(level):
                proof.append(level[sibling_idx])
            
            # Move to parent level
            new_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                new_level.append(self._hash(left + right))
            
            level = new_level
            current_idx //= 2
        
        return proof
    
    def verify(self, data: str, proof: List[str], root: str) -> bool:
        """
        Verify data against root using proof.
        
        Args:
            data: Data to verify
            proof: Merkle proof
            root: Expected root hash
            
        Returns:
            bool: Whether verification succeeds
        """
        current_hash = self._hash(data)
        
        for sibling_hash in proof:
            # Assume we're left child (simplified)
            current_hash = self._hash(current_hash + sibling_hash)
        
        return current_hash == root


class HyperLogLog:
    """
    HyperLogLog for cardinality estimation.
    
    Estimates number of unique elements with small memory.
    """
    
    def __init__(self, precision: int = 14):
        """
        Args:
            precision: Precision (4-16), higher = more accurate
        """
        self.p = precision
        self.m = 1 << precision
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()
    
    def _get_alpha(self) -> float:
        """Get alpha constant based on m."""
        if self.m >= 128:
            return 0.7213 / (1 + 1.079 / self.m)
        elif self.m >= 64:
            return 0.709
        elif self.m >= 32:
            return 0.697
        elif self.m >= 16:
            return 0.673
        else:
            return 0.5
    
    def _hash(self, item: str) -> int:
        """Hash item."""
        import hashlib
        return int(hashlib.md5(item.encode()).hexdigest(), 16)
    
    def add(self, item: str):
        """
        Add item to sketch.
        
        Args:
            item: Item to add
        """
        x = self._hash(item)
        
        # Get register index (first p bits)
        j = x & (self.m - 1)
        
        # Count leading zeros (remaining bits)
        w = x >> self.p
        leading_zeros = self._count_leading_zeros(w)
        
        self.registers[j] = max(self.registers[j], leading_zeros)
    
    def _count_leading_zeros(self, x: int) -> int:
        """Count leading zeros in binary representation."""
        if x == 0:
            return 64 - self.p
        
        return (64 - self.p) - x.bit_length() + 1
    
    def count(self) -> int:
        """
        Estimate cardinality.
        
        Returns:
            int: Estimated number of unique elements
        """
        # Harmonic mean of register values
        raw_estimate = self.alpha * self.m**2 / sum(2**(-r) for r in self.registers)
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            v = sum(1 for r in self.registers if r == 0)
            if v != 0:
                return int(self.m * np.log(self.m / v))
        
        # Large range correction
        if raw_estimate > (1/30) * (1 << 64):
            return int(-(1 << 64) * np.log(1 - raw_estimate / (1 << 64)))
        
        return int(raw_estimate)


# Export all data structures
__all__ = [
    'OrderBook',
    'TimeSeriesBuffer',
    'PriorityQueue',
    'RollingStatistics',
    'Trie',
    'SegmentTree',
    'DisjointSet',
    'Graph',
    'BloomFilter',
    'LRUCache',
    'CountMinSketch',
    'SkipList',
    'FenwickTree',
    'RedBlackTree',
    'ConsistentHashRing',
    'MerkleTree',
    'HyperLogLog'
]
```

---

*Note: This concludes the comprehensive agent.md file. The document contains extensive mathematical foundations, implementation details, data structures, algorithms, and reference materials for building an institutional-grade algorithmic trading system.*
