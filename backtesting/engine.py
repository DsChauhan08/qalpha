"""
Event-Driven Backtesting Engine - V1
Realistic backtesting with slippage, transaction costs, and proper P&L calculation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable
from datetime import datetime


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    order_id: str = ""
    timestamp: Optional[datetime] = None


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
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    market_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.avg_entry_price)

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price


class SlippageModel:
    """
    Realistic slippage based on spread and market impact.

    Components:
    - Base spread cost (bid-ask spread)
    - Market impact (Almgren-Chriss style)
    - Volatility adjustment
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        volatility_factor: float = 0.5,
    ):
        self.base_spread = base_spread_bps / 10000
        self.impact_coeff = impact_coefficient
        self.vol_factor = volatility_factor

    def estimate(self, order: Order, bar: pd.Series, avg_volume: float) -> float:
        """
        Estimate slippage for an order.

        Args:
            order: Order to execute
            bar: Current bar data (open, high, low, close, volume, atr)
            avg_volume: Historical average volume

        Returns:
            Slippage as decimal (positive = worse fill for buyer)
        """
        # Base spread
        spread_cost = self.base_spread / 2

        # Market impact
        if avg_volume > 0:
            participation = abs(order.quantity * bar["close"]) / (
                avg_volume * bar["close"]
            )
            impact = self.impact_coeff * np.sqrt(max(participation, 0))
        else:
            impact = 0

        # Volatility adjustment
        if "atr" in bar.index and bar["atr"] > 0:
            vol_adj = self.vol_factor * (bar["atr"] / bar["close"])
        else:
            vol_adj = 0

        total_slippage = spread_cost + impact + vol_adj

        return total_slippage if order.side == OrderSide.BUY else -total_slippage


class TransactionCostModel:
    """
    Transaction cost model including commissions and fees.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        min_commission: float = 1.0,
        sec_fee_rate: float = 0.0000278,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.sec_fee_rate = sec_fee_rate

    def calculate(self, fill: Fill) -> float:
        """Calculate total transaction costs."""
        trade_value = abs(fill.quantity * fill.fill_price)

        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # SEC fee (sells only)
        sec_fee = 0
        if fill.side == OrderSide.SELL:
            sec_fee = min(trade_value * self.sec_fee_rate, 5.95)

        return commission + sec_fee


class Backtester:
    """
    Event-driven backtester with realistic execution simulation.

    Features:
    - Bar-by-bar simulation (no lookahead)
    - Slippage and transaction costs
    - Position tracking
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        slippage_model: Optional[SlippageModel] = None,
        cost_model: Optional[TransactionCostModel] = None,
    ):
        self.initial_capital = initial_capital
        self.slippage = slippage_model or SlippageModel()
        self.costs = cost_model or TransactionCostModel()
        self.reset()

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.fills: List[Fill] = []
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self._order_counter = 0

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> str:
        """Submit an order."""
        self._order_counter += 1
        order_id = f"ORD_{self._order_counter:06d}"

        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            order_id=order_id,
        )
        self.pending_orders.append(order)
        return order_id

    def _execute_orders(self, timestamp: datetime, bar_data: Dict[str, pd.Series]):
        """Execute pending orders against current bar."""
        executed = []

        for order in self.pending_orders:
            if order.symbol not in bar_data:
                continue

            bar = bar_data[order.symbol]
            avg_volume = bar.get("avg_volume", bar["volume"])

            # Determine execution price
            if order.order_type == OrderType.MARKET:
                base_price = bar["open"]  # Execute at open
                slippage = self.slippage.estimate(order, bar, avg_volume)
                fill_price = base_price * (1 + slippage)

            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and bar["low"] <= order.price:
                    fill_price = min(order.price, bar["open"])
                elif order.side == OrderSide.SELL and bar["high"] >= order.price:
                    fill_price = max(order.price, bar["open"])
                else:
                    continue  # Limit not hit
                slippage = 0
            else:
                continue

            # Create fill
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=timestamp,
                slippage=slippage,
                commission=0,
            )

            # Calculate costs
            fill.commission = self.costs.calculate(fill)

            # Update position
            self._update_position(fill)

            # Update capital
            trade_value = fill.quantity * fill.fill_price
            if order.side == OrderSide.BUY:
                self.capital -= trade_value + fill.commission
            else:
                self.capital += trade_value - fill.commission

            self.fills.append(fill)
            executed.append(order)

        # Remove executed orders
        for order in executed:
            self.pending_orders.remove(order)

    def _update_position(self, fill: Fill):
        """Update position based on fill."""
        symbol = fill.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        # Same-direction add or open
        if pos.quantity == 0 or np.sign(pos.quantity) == np.sign(signed_qty):
            new_qty = pos.quantity + signed_qty
            total_cost = (
                abs(pos.quantity) * pos.avg_entry_price + fill.quantity * fill.fill_price
            )
            pos.quantity = new_qty
            pos.avg_entry_price = (
                total_cost / abs(new_qty) if new_qty != 0 else 0.0
            )
            return

        # Opposite-direction trade: reduce, close, or flip
        closed_qty = min(abs(pos.quantity), fill.quantity)
        if pos.quantity > 0:  # Closing long with sell
            realized = closed_qty * (fill.fill_price - pos.avg_entry_price)
        else:  # Closing short with buy
            realized = closed_qty * (pos.avg_entry_price - fill.fill_price)

        pos.realized_pnl += realized
        commission_alloc = (
            fill.commission * (closed_qty / fill.quantity) if fill.quantity else 0.0
        )

        self.trades.append(
            {
                "symbol": symbol,
                "entry_price": pos.avg_entry_price,
                "exit_price": fill.fill_price,
                "quantity": closed_qty,
                "pnl": realized - commission_alloc,
                "timestamp": fill.timestamp,
            }
        )

        new_qty = pos.quantity + signed_qty
        if new_qty == 0:
            pos.quantity = 0.0
            pos.avg_entry_price = 0.0
        elif np.sign(new_qty) == np.sign(pos.quantity):
            # Reduced but not flipped
            pos.quantity = new_qty
        else:
            # Flipped direction: remaining position opened at fill price
            pos.quantity = new_qty
            pos.avg_entry_price = fill.fill_price

    def _mark_to_market(self, bar_data: Dict[str, pd.Series]):
        """Update position values."""
        for symbol, pos in self.positions.items():
            if symbol in bar_data:
                pos.market_price = bar_data[symbol]["close"]

    def _total_equity(self) -> float:
        """Calculate total equity."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.capital + positions_value

    def process_bar(
        self,
        timestamp: datetime,
        bar_data: Dict[str, pd.Series],
        signals: Dict[str, float] = None,
    ):
        """
        Process a single bar.

        Args:
            timestamp: Bar timestamp
            bar_data: Dict of symbol -> bar Series
            signals: Optional signals to generate orders
        """
        # Execute pending orders
        self._execute_orders(timestamp, bar_data)

        # Mark to market
        self._mark_to_market(bar_data)

        # Record equity
        equity = self._total_equity()
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.capital,
                "positions_value": equity - self.capital,
            }
        )

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Callable[[datetime, Dict[str, pd.Series], "Backtester"], None],
    ):
        """
        Run backtest.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV
            strategy: Strategy function(timestamp, bars, backtester)
        """
        self.reset()

        # Get all timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index.tolist())
        timestamps = sorted(all_timestamps)

        for ts in timestamps:
            # Get current bars
            bar_data = {}
            for symbol, df in data.items():
                if ts in df.index:
                    bar_data[symbol] = df.loc[ts]

            if not bar_data:
                continue

            # Execute strategy
            strategy(ts, bar_data, self)

            # Process bar
            self.process_bar(ts, bar_data)

    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dict with performance statistics
        """
        if len(self.equity_curve) < 2:
            return {"error": "Insufficient data"}

        equity = pd.DataFrame(self.equity_curve)
        returns = equity["equity"].pct_change().dropna()

        # Basic metrics
        total_return = (equity["equity"].iloc[-1] / self.initial_capital) - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        # Ratios
        risk_free = 0.02
        sharpe = (annual_return - risk_free) / volatility if volatility > 0 else 0
        sortino = (annual_return - risk_free) / downside_vol if downside_vol > 0 else 0

        # Drawdown
        equity["peak"] = equity["equity"].cummax()
        equity["drawdown"] = (equity["equity"] - equity["peak"]) / equity["peak"]
        max_drawdown = equity["drawdown"].min()

        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade stats
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df["pnl"] > 0).mean()
            avg_win = (
                trades_df[trades_df["pnl"] > 0]["pnl"].mean()
                if (trades_df["pnl"] > 0).any()
                else 0
            )
            avg_loss = (
                trades_df[trades_df["pnl"] < 0]["pnl"].mean()
                if (trades_df["pnl"] < 0).any()
                else 0
            )
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": len(self.trades),
            "final_equity": equity["equity"].iloc[-1],
        }

    def get_equity_series(self) -> pd.Series:
        """Get equity curve as Series."""
        equity = pd.DataFrame(self.equity_curve)
        return equity.set_index("timestamp")["equity"]
