"""
Persistent Homology (TDA) for Market Regime Detection

Topological Data Analysis detects market regime changes by analyzing the "shape"
of financial data. Unlike traditional methods, TDA captures the intrinsic topology
of price movements.

Key Insight: Market crashes create "holes" in the topological structure as
correlations break down and assets decouple.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
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

    def __init__(self, lookback_window: int = 60, max_dim: int = 2):
        """
        Args:
            lookback_window: Number of periods for analysis window
            max_dim: Maximum homology dimension to compute
        """
        self.lookback = lookback_window
        self.max_dim = max_dim
        self._has_ripser = self._check_ripser()

    def _check_ripser(self) -> bool:
        """Check if ripser is available."""
        try:
            import ripser

            return True
        except ImportError:
            return False

    def create_point_cloud(self, price_data: "pd.DataFrame") -> np.ndarray:
        """
        Create high-dimensional embedding from price data.

        Dimensions:
        1. Log returns (price momentum)
        2. Volume changes (liquidity)
        3. ATR normalized (volatility)
        4. RSI normalized (momentum oscillator)
        5. Price position in range (mean-reversion)

        Args:
            price_data: DataFrame with OHLCV columns

        Returns:
            np.array: Point cloud of shape (n_samples, n_features)
        """
        import pandas as pd

        close = price_data["close"].values
        high = price_data["high"].values
        low = price_data["low"].values
        volume = price_data["volume"].values

        # Log returns
        returns = np.diff(np.log(close))

        # Volume changes
        volume_safe = np.where(volume > 0, volume, 1)
        volume_change = np.diff(np.log(volume_safe))

        # ATR-like volatility (simplified without TA-Lib dependency)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = self._ema(tr, 14)
        atr_pct = atr / close[1:] * 100

        # RSI (simplified calculation)
        rsi = self._compute_rsi(close, 14)[1:]

        # Price position in recent range
        rolling_high = self._rolling_max(high, 20)[1:]
        rolling_low = self._rolling_min(low, 20)[1:]
        price_position = (close[1:] - rolling_low) / (
            rolling_high - rolling_low + 1e-10
        )

        # Align all features
        min_len = min(
            len(returns),
            len(volume_change),
            len(atr_pct),
            len(rsi),
            len(price_position),
        )

        features = np.column_stack(
            [
                returns[-min_len:],
                volume_change[-min_len:],
                atr_pct[-min_len:],
                rsi[-min_len:] / 100,  # Normalize to 0-1
                price_position[-min_len:],
            ]
        )

        # Standardize features
        features = (features - np.nanmean(features, axis=0)) / (
            np.nanstd(features, axis=0) + 1e-10
        )

        # Remove NaN rows
        features = features[~np.isnan(features).any(axis=1)]

        return features

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        alpha = 2 / (period + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = self._ema(gains, period)
        avg_loss = self._ema(losses, period)

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[50], rsi])  # Pad first value

    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling maximum."""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.max(data[start : i + 1])
        return result

    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Rolling minimum."""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.min(data[start : i + 1])
        return result

    def compute_persistence(self, point_cloud: np.ndarray) -> Dict:
        """
        Compute persistent homology of point cloud.

        Uses Ripser if available, otherwise falls back to simplified Vietoris-Rips.

        Args:
            point_cloud: np.array of shape (n_points, n_dimensions)

        Returns:
            dict: Persistence diagram features
        """
        if self._has_ripser:
            return self._compute_with_ripser(point_cloud)
        else:
            return self._compute_simplified(point_cloud)

    def _compute_with_ripser(self, point_cloud: np.ndarray) -> Dict:
        """Compute persistence using Ripser library."""
        from ripser import ripser

        result = ripser(point_cloud, maxdim=self.max_dim)
        diagrams = result["dgms"]

        features = {}

        # H0: Connected components
        if len(diagrams[0]) > 0:
            h0_lifetimes = diagrams[0][:, 1] - diagrams[0][:, 0]
            h0_lifetimes = h0_lifetimes[~np.isinf(h0_lifetimes)]
            features["h0_total_persistence"] = float(np.sum(h0_lifetimes))
            features["h0_num_components"] = len(h0_lifetimes)
            features["h0_max_lifetime"] = (
                float(np.max(h0_lifetimes)) if len(h0_lifetimes) > 0 else 0.0
            )
        else:
            features["h0_total_persistence"] = 0.0
            features["h0_num_components"] = 0
            features["h0_max_lifetime"] = 0.0

        # H1: 1-dimensional holes (cycles)
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            h1_lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0]
            features["h1_total_persistence"] = float(np.sum(h1_lifetimes))
            features["h1_num_cycles"] = len(h1_lifetimes)
            features["h1_max_lifetime"] = float(np.max(h1_lifetimes))
            features["h1_mean_lifetime"] = float(np.mean(h1_lifetimes))
            features["h1_std_lifetime"] = float(np.std(h1_lifetimes))
        else:
            features["h1_total_persistence"] = 0.0
            features["h1_num_cycles"] = 0
            features["h1_max_lifetime"] = 0.0
            features["h1_mean_lifetime"] = 0.0
            features["h1_std_lifetime"] = 0.0

        # H2: 2-dimensional voids
        if len(diagrams) > 2 and len(diagrams[2]) > 0:
            h2_lifetimes = diagrams[2][:, 1] - diagrams[2][:, 0]
            features["h2_total_persistence"] = float(np.sum(h2_lifetimes))
            features["h2_num_voids"] = len(h2_lifetimes)
        else:
            features["h2_total_persistence"] = 0.0
            features["h2_num_voids"] = 0

        return features

    def _compute_simplified(self, point_cloud: np.ndarray) -> Dict:
        """
        Simplified persistence computation without Ripser.
        Uses distance matrix analysis to approximate topological features.
        """
        # Compute pairwise distances
        distances = pdist(point_cloud)
        dist_matrix = squareform(distances)

        n_points = len(point_cloud)

        # Approximate H0: Connected components at various thresholds
        thresholds = np.percentile(distances, [25, 50, 75, 90])

        # Count connected components at median threshold
        h0_components = self._count_components(dist_matrix, thresholds[1])

        # Approximate H1: Detect cycles through persistent neighborhood structure
        h1_cycles = self._estimate_cycles(dist_matrix, thresholds)

        features = {
            "h0_total_persistence": float(np.std(distances) * h0_components),
            "h0_num_components": h0_components,
            "h0_max_lifetime": float(thresholds[2] - thresholds[0]),
            "h1_total_persistence": float(h1_cycles * np.mean(distances)),
            "h1_num_cycles": h1_cycles,
            "h1_max_lifetime": float(thresholds[3] - thresholds[1])
            if h1_cycles > 0
            else 0.0,
            "h1_mean_lifetime": float(np.mean(distances)) if h1_cycles > 0 else 0.0,
            "h1_std_lifetime": float(np.std(distances)) if h1_cycles > 0 else 0.0,
            "h2_total_persistence": 0.0,
            "h2_num_voids": 0,
        }

        return features

    def _count_components(self, dist_matrix: np.ndarray, threshold: float) -> int:
        """Count connected components at given threshold using union-find."""
        n = len(dist_matrix)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] <= threshold:
                    union(i, j)

        return len(set(find(i) for i in range(n)))

    def _estimate_cycles(self, dist_matrix: np.ndarray, thresholds: np.ndarray) -> int:
        """Estimate number of 1-cycles from distance matrix."""
        n = len(dist_matrix)

        # Count triangles that form at different thresholds
        cycles = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    edges = [dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k]]
                    # Triangle forms if edges appear at different times
                    birth = np.min(edges)
                    death = np.max(edges)
                    if death - birth > thresholds[1] - thresholds[0]:
                        cycles += 1

        # Normalize by expected triangles
        expected = n * (n - 1) * (n - 2) / 6
        return min(cycles, int(expected * 0.1))

    def detect_regime_change(
        self,
        historical_features: List[Dict],
        current_features: Dict,
        threshold: float = 2.0,
    ) -> Dict:
        """
        Detect if current market topology indicates regime change.

        Args:
            historical_features: List of past topology feature dicts
            current_features: Current topology feature dict
            threshold: Z-score threshold for anomaly detection

        Returns:
            dict: Regime classification and confidence
        """
        if len(historical_features) < 10:
            return {"regime": "INSUFFICIENT_DATA", "confidence": 0.0, "z_score": 0.0}

        # Extract H1 cycle counts for regime detection
        hist_cycles = [f["h1_num_cycles"] for f in historical_features]
        current_cycles = current_features["h1_num_cycles"]

        mean_cycles = np.mean(hist_cycles)
        std_cycles = np.std(hist_cycles)

        if std_cycles < 0.01:
            return {"regime": "STABLE", "confidence": 0.5, "z_score": 0.0}

        z_score = (current_cycles - mean_cycles) / std_cycles

        # Also check H1 total persistence
        hist_pers = [f["h1_total_persistence"] for f in historical_features]
        current_pers = current_features["h1_total_persistence"]

        mean_pers = np.mean(hist_pers)
        std_pers = np.std(hist_pers)

        z_score_pers = (current_pers - mean_pers) / (std_pers + 1e-10)

        # Combined z-score
        combined_z = (z_score + z_score_pers) / 2

        # Classification logic
        if combined_z > threshold:
            regime = "REGIME_BREAK_OPPORTUNITY"
            confidence = min(abs(combined_z) / 3.0, 1.0)
        elif combined_z < -threshold:
            regime = "REGIME_STABILIZATION"
            confidence = min(abs(combined_z) / 3.0, 1.0)
        elif abs(combined_z) > threshold * 0.5:
            regime = "TRANSITION_WARNING"
            confidence = min(abs(combined_z) / threshold, 1.0)
        else:
            regime = "NORMAL"
            confidence = 1.0 - min(abs(combined_z) / threshold, 1.0)

        return {
            "regime": regime,
            "confidence": confidence,
            "z_score": float(combined_z),
            "h1_cycles": current_cycles,
            "historical_mean": float(mean_cycles),
            "historical_std": float(std_cycles),
        }

    def generate_trading_signal(
        self, price_data: "pd.DataFrame", historical_window: int = 252
    ) -> Dict:
        """
        Generate trading signal based on topological analysis.

        Args:
            price_data: Full price history DataFrame
            historical_window: Days of history for baseline

        Returns:
            dict: Signal strength and metadata
        """
        if len(price_data) < self.lookback + historical_window:
            return {
                "signal": 0.0,
                "regime": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "topology_features": {},
                "raw_z_score": 0.0,
            }

        # Split data
        historical = price_data.iloc[: -self.lookback]
        current = price_data.iloc[-self.lookback :]

        # Compute historical baseline
        hist_features = []
        step = max(1, self.lookback // 4)  # Overlapping windows

        for i in range(0, len(historical) - self.lookback, step):
            window = historical.iloc[i : i + self.lookback]
            if len(window) < self.lookback:
                continue
            try:
                pc = self.create_point_cloud(window)
                if len(pc) > 10:
                    feat = self.compute_persistence(pc)
                    hist_features.append(feat)
            except Exception:
                continue

        if len(hist_features) < 10:
            return {
                "signal": 0.0,
                "regime": "INSUFFICIENT_HISTORY",
                "confidence": 0.0,
                "topology_features": {},
                "raw_z_score": 0.0,
            }

        # Compute current topology
        try:
            current_pc = self.create_point_cloud(current)
            current_feat = self.compute_persistence(current_pc)
        except Exception:
            return {
                "signal": 0.0,
                "regime": "COMPUTATION_ERROR",
                "confidence": 0.0,
                "topology_features": {},
                "raw_z_score": 0.0,
            }

        # Detect regime
        regime_info = self.detect_regime_change(hist_features, current_feat)

        # Convert to trading signal
        signal_map = {
            "REGIME_BREAK_OPPORTUNITY": 0.8,
            "TRANSITION_WARNING": 0.3,
            "NORMAL": 0.0,
            "REGIME_STABILIZATION": -0.3,
            "INSUFFICIENT_DATA": 0.0,
            "INSUFFICIENT_HISTORY": 0.0,
        }

        base_signal = signal_map.get(regime_info["regime"], 0.0)
        adjusted_signal = base_signal * regime_info["confidence"]

        return {
            "signal": adjusted_signal,
            "regime": regime_info["regime"],
            "confidence": regime_info["confidence"],
            "topology_features": current_feat,
            "raw_z_score": regime_info.get("z_score", 0.0),
        }

    def compute_rolling_features(
        self, price_data: "pd.DataFrame", window: int = 60, step: int = 5
    ) -> "pd.DataFrame":
        """
        Compute rolling TDA features for the entire price series.

        Args:
            price_data: DataFrame with OHLCV
            window: Rolling window size
            step: Step size between windows

        Returns:
            DataFrame with TDA features aligned to price data
        """
        import pandas as pd

        features_list = []
        indices = []

        for i in range(window, len(price_data), step):
            window_data = price_data.iloc[i - window : i]
            try:
                pc = self.create_point_cloud(window_data)
                if len(pc) > 10:
                    feat = self.compute_persistence(pc)
                    feat["index"] = price_data.index[i - 1]
                    features_list.append(feat)
                    indices.append(price_data.index[i - 1])
            except Exception:
                continue

        if not features_list:
            return pd.DataFrame()

        df = pd.DataFrame(features_list)
        df.set_index("index", inplace=True)

        # Reindex to full price data and forward fill
        df = df.reindex(price_data.index).ffill()

        return df
