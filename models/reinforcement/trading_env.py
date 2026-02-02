"""
Trading Environment for Reinforcement Learning.

Gym-like environment for training RL agents on trading tasks.

Based on agent.md Section 3.3:
- State: Market features + portfolio state
- Actions: [hold, buy, sell] with position sizing
- Reward: Risk-adjusted returns (Sharpe-like)

Supports:
- Realistic transaction costs and slippage
- Multiple position sizing modes
- Continuous and discrete action spaces
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Discrete actions for trading."""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class EnvironmentConfig:
    """Configuration for trading environment."""

    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position: float = 1.0  # Max position as fraction of capital
    min_trade_size: float = 0.01  # Min trade as fraction of capital
    reward_scaling: float = 1000.0  # Scale rewards for training stability
    lookback_window: int = 20  # Bars of history in state
    include_position_state: bool = True
    include_pnl_state: bool = True


class TradingEnvironment:
    """
    Gym-like environment for trading.

    State Space:
    - Market features (OHLCV, indicators)
    - Portfolio state (position, cash, unrealized PnL)

    Action Space (Discrete):
    - 0: Hold (do nothing)
    - 1: Buy (increase position)
    - 2: Sell (decrease position)

    Rewards:
    - Risk-adjusted returns (Sharpe-like)
    - Penalty for excessive trading
    - Bonus for profitable trades
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        config: Optional[EnvironmentConfig] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            prices: Price array of shape (n_steps,) - close prices
            features: Feature array of shape (n_steps, n_features)
            config: Environment configuration
        """
        self.prices = np.asarray(prices).flatten()
        self.features = np.asarray(features)
        self.config = config or EnvironmentConfig()

        if len(self.prices) != len(self.features):
            raise ValueError(
                f"Price length ({len(self.prices)}) must match features ({len(self.features)})"
            )

        self.n_steps = len(self.prices)
        self.n_features = self.features.shape[1]

        # Calculate state dimension
        self.state_dim = self.n_features
        if self.config.include_position_state:
            self.state_dim += 1  # Position ratio
        if self.config.include_pnl_state:
            self.state_dim += 2  # Cash ratio, unrealized PnL ratio

        self.action_dim = 3  # Hold, Buy, Sell

        # Initialize state
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = self.config.lookback_window
        self.cash = self.config.initial_capital
        self.position = 0.0  # Shares held
        self.position_value = 0.0
        self.portfolio_value = self.config.initial_capital

        # Tracking
        self.trades: List[Dict] = []
        self.returns: List[float] = []
        self.portfolio_values: List[float] = [self.config.initial_capital]
        self.entry_price = 0.0

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Market features
        market_state = self.features[self.current_step].copy()

        state_components = [market_state]

        # Portfolio state
        if self.config.include_position_state:
            position_ratio = (self.position * self.prices[self.current_step]) / max(
                self.portfolio_value, 1
            )
            state_components.append(np.array([position_ratio]))

        if self.config.include_pnl_state:
            cash_ratio = self.cash / self.config.initial_capital

            # Unrealized PnL ratio
            if self.position > 0:
                unrealized_pnl = self.position * (
                    self.prices[self.current_step] - self.entry_price
                )
                pnl_ratio = unrealized_pnl / self.config.initial_capital
            else:
                pnl_ratio = 0.0

            state_components.append(np.array([cash_ratio, pnl_ratio]))

        return np.concatenate(state_components)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)

        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self.portfolio_value

        # Execute trade
        trade_info = self._execute_action(action, current_price)

        # Advance time
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Update portfolio value
        new_price = self.prices[self.current_step] if not done else current_price
        self.position_value = self.position * new_price
        self.portfolio_value = self.cash + self.position_value

        # Calculate return
        step_return = (
            self.portfolio_value - prev_portfolio_value
        ) / prev_portfolio_value
        self.returns.append(step_return)
        self.portfolio_values.append(self.portfolio_value)

        # Calculate reward
        reward = self._calculate_reward(step_return, trade_info)

        # Get new observation
        observation = self._get_observation() if not done else np.zeros(self.state_dim)

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "cash": self.cash,
            "step_return": step_return,
            "trade": trade_info,
            "price": new_price,
        }

        return observation, reward, done, info

    def _execute_action(self, action: int, price: float) -> Dict:
        """Execute trading action."""
        trade_info = {"action": action, "executed": False, "cost": 0.0}

        if action == Action.HOLD:
            return trade_info

        # Calculate effective price with slippage
        if action == Action.BUY:
            effective_price = price * (1 + self.config.slippage)
        else:  # SELL
            effective_price = price * (1 - self.config.slippage)

        if action == Action.BUY:
            # Calculate position to buy (50% of available cash)
            available = self.cash * 0.5

            if available < self.config.min_trade_size * self.config.initial_capital:
                return trade_info

            # Check max position constraint
            max_position_value = self.config.max_position * self.portfolio_value
            current_position_value = self.position * price
            allowed_buy = max_position_value - current_position_value

            trade_value = min(available, allowed_buy)
            if trade_value <= 0:
                return trade_info

            # Calculate shares and costs
            shares = trade_value / effective_price
            cost = trade_value * self.config.transaction_cost
            total_cost = trade_value + cost

            if total_cost > self.cash:
                return trade_info

            # Execute
            self.position += shares
            self.cash -= total_cost

            # Update entry price (weighted average)
            if self.entry_price == 0:
                self.entry_price = effective_price
            else:
                total_shares = self.position
                self.entry_price = (
                    self.entry_price * (total_shares - shares)
                    + effective_price * shares
                ) / total_shares

            trade_info = {
                "action": action,
                "executed": True,
                "shares": shares,
                "price": effective_price,
                "cost": cost,
                "type": "buy",
            }

            self.trades.append(trade_info)

        elif action == Action.SELL:
            if self.position <= 0:
                return trade_info

            # Sell 50% of position
            shares_to_sell = self.position * 0.5

            if (
                shares_to_sell * price
                < self.config.min_trade_size * self.config.initial_capital
            ):
                shares_to_sell = self.position  # Sell all if remainder too small

            # Calculate proceeds
            proceeds = shares_to_sell * effective_price
            cost = proceeds * self.config.transaction_cost
            net_proceeds = proceeds - cost

            # Calculate realized PnL
            realized_pnl = shares_to_sell * (effective_price - self.entry_price)

            # Execute
            self.position -= shares_to_sell
            self.cash += net_proceeds

            if self.position < 1e-8:  # Effectively zero
                self.position = 0.0
                self.entry_price = 0.0

            trade_info = {
                "action": action,
                "executed": True,
                "shares": shares_to_sell,
                "price": effective_price,
                "cost": cost,
                "realized_pnl": realized_pnl,
                "type": "sell",
            }

            self.trades.append(trade_info)

        return trade_info

    def _calculate_reward(self, step_return: float, trade_info: Dict) -> float:
        """
        Calculate reward for the step.

        Reward components:
        1. Portfolio return (scaled)
        2. Risk-adjusted component (Sharpe-like)
        3. Trading penalty (discourage overtrading)
        4. Profit bonus for winning trades
        """
        reward = 0.0

        # 1. Base return reward
        reward += step_return * self.config.reward_scaling

        # 2. Risk-adjusted component (rolling Sharpe)
        if len(self.returns) >= 10:
            recent_returns = np.array(self.returns[-10:])
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns) + 1e-8
            sharpe_component = (mean_ret / std_ret) * 0.1
            reward += sharpe_component * self.config.reward_scaling

        # 3. Trading penalty
        if trade_info.get("executed", False):
            reward -= (
                trade_info.get("cost", 0)
                / self.config.initial_capital
                * self.config.reward_scaling
            )

        # 4. Profit bonus for winning sells
        if trade_info.get("type") == "sell" and trade_info.get("realized_pnl", 0) > 0:
            bonus = trade_info["realized_pnl"] / self.config.initial_capital * 0.5
            reward += bonus * self.config.reward_scaling

        # 5. Holding penalty (encourage action)
        if trade_info.get("action") == Action.HOLD and self.position == 0:
            reward -= 0.001 * self.config.reward_scaling

        return reward

    def get_metrics(self) -> Dict:
        """Get performance metrics for the episode."""
        if len(self.portfolio_values) < 2:
            return {"error": "Insufficient data"}

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = [t for t in self.trades if t.get("realized_pnl", 0) > 0]
        total_trades = len([t for t in self.trades if t.get("type") == "sell"])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_trades": len(self.trades),
            "final_value": float(values[-1]),
            "total_cost": sum(t.get("cost", 0) for t in self.trades),
        }

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Get observation space shape."""
        return (self.state_dim,)

    @property
    def action_space_n(self) -> int:
        """Get number of actions."""
        return self.action_dim
