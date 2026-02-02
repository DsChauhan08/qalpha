"""
Order Manager Module - Quantum Alpha V1
Order lifecycle management per agent.md Section 6.

Implements:
- Order creation and validation
- Order state machine
- Order book management
- Fill tracking and reconciliation
- Order modification and cancellation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types supported."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MOC = "market_on_close"
    MOO = "market_on_open"


class OrderStatus(Enum):
    """Order lifecycle states."""

    PENDING = "pending"  # Created, not yet submitted
    SUBMITTED = "submitted"  # Sent to broker
    ACCEPTED = "accepted"  # Acknowledged by broker
    PARTIAL = "partial"  # Partially filled
    FILLED = "filled"  # Completely filled
    CANCELLED = "cancelled"  # Cancelled by user
    REJECTED = "rejected"  # Rejected by broker
    EXPIRED = "expired"  # Time-in-force expired
    FAILED = "failed"  # System error


class TimeInForce(Enum):
    """Order time-in-force options."""

    DAY = "day"  # Valid for trading day
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close


@dataclass
class OrderFill:
    """Record of an order fill."""

    fill_id: str
    order_id: str
    timestamp: datetime
    quantity: float
    price: float
    commission: float = 0.0
    exchange: str = ""


@dataclass
class Order:
    """Trading order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    # Price fields (depending on order type)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None

    # Time in force
    time_in_force: TimeInForce = TimeInForce.DAY

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Fill tracking
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)

    # Fees
    total_commission: float = 0.0

    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For bracket orders
    notes: str = ""

    # Broker response
    broker_order_id: Optional[str] = None
    reject_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
        ]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        ]

    @property
    def fill_rate(self) -> float:
        """Get fill rate (0-1)."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

    def add_fill(self, fill: OrderFill):
        """Add a fill to this order."""
        self.fills.append(fill)

        # Update fill tracking
        total_cost = self.average_fill_price * self.filled_quantity
        total_cost += fill.price * fill.quantity
        self.filled_quantity += fill.quantity

        if self.filled_quantity > 0:
            self.average_fill_price = total_cost / self.filled_quantity

        self.total_commission += fill.commission

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill.timestamp
        else:
            self.status = OrderStatus.PARTIAL


@dataclass
class OrderRequest:
    """Request to create an order."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None


class OrderValidator:
    """
    Validate orders before submission.
    """

    def __init__(
        self,
        min_order_value: float = 1.0,
        max_order_value: float = 1000000.0,
        min_quantity: float = 1.0,
        max_quantity: float = 100000.0,
        allowed_symbols: Optional[List[str]] = None,
    ):
        self.min_order_value = min_order_value
        self.max_order_value = max_order_value
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.allowed_symbols = allowed_symbols

    def validate(
        self, order: Order, current_price: float, buying_power: float = float("inf")
    ) -> Tuple[bool, str]:
        """
        Validate an order.

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check symbol
        if self.allowed_symbols and order.symbol not in self.allowed_symbols:
            return False, f"Symbol {order.symbol} not in allowed list"

        # Check quantity
        if order.quantity < self.min_quantity:
            return False, f"Quantity {order.quantity} below minimum {self.min_quantity}"

        if order.quantity > self.max_quantity:
            return (
                False,
                f"Quantity {order.quantity} exceeds maximum {self.max_quantity}",
            )

        # Check order value
        order_value = order.quantity * current_price
        if order_value < self.min_order_value:
            return (
                False,
                f"Order value ${order_value:.2f} below minimum ${self.min_order_value:.2f}",
            )

        if order_value > self.max_order_value:
            return (
                False,
                f"Order value ${order_value:.2f} exceeds maximum ${self.max_order_value:.2f}",
            )

        # Check buying power for buys
        if order.side == OrderSide.BUY and order_value > buying_power:
            return (
                False,
                f"Insufficient buying power: need ${order_value:.2f}, have ${buying_power:.2f}",
            )

        # Check limit price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.limit_price is None:
                return False, "Limit price required for limit order"
            if order.limit_price <= 0:
                return False, "Limit price must be positive"

        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                return False, "Stop price required for stop order"
            if order.stop_price <= 0:
                return False, "Stop price must be positive"

        # Validate stop price logic
        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and order.stop_price < current_price:
                return False, "Buy stop price must be above current price"
            if order.side == OrderSide.SELL and order.stop_price > current_price:
                return False, "Sell stop price must be below current price"

        return True, "Valid"


class OrderBook:
    """
    Manages collection of orders.
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)

    def add(self, order: Order):
        """Add order to book."""
        self.orders[order.order_id] = order
        self.orders_by_symbol[order.symbol].append(order.order_id)
        self.orders_by_status[order.status].append(order.order_id)

        if order.strategy_id:
            self.orders_by_strategy[order.strategy_id].append(order.order_id)

    def get(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def update_status(self, order_id: str, new_status: OrderStatus):
        """Update order status."""
        order = self.orders.get(order_id)
        if order:
            # Remove from old status list
            if order_id in self.orders_by_status[order.status]:
                self.orders_by_status[order.status].remove(order_id)

            # Update status
            order.status = new_status

            # Add to new status list
            self.orders_by_status[new_status].append(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        active_statuses = [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
        ]

        active_ids = []
        for status in active_statuses:
            active_ids.extend(self.orders_by_status[status])

        orders = [self.orders[oid] for oid in active_ids if oid in self.orders]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy."""
        order_ids = self.orders_by_strategy.get(strategy_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_recent_fills(self, hours: int = 24) -> List[OrderFill]:
        """Get recent fills."""
        cutoff = datetime.now() - timedelta(hours=hours)
        fills = []

        for order in self.orders.values():
            for fill in order.fills:
                if fill.timestamp >= cutoff:
                    fills.append(fill)

        return sorted(fills, key=lambda f: f.timestamp, reverse=True)


class OrderManager:
    """
    Main order management system.
    Handles order lifecycle from creation to completion.
    """

    def __init__(
        self,
        validator: Optional[OrderValidator] = None,
        on_order_update: Optional[Callable[[Order], None]] = None,
        on_fill: Optional[Callable[[Order, OrderFill], None]] = None,
    ):
        self.order_book = OrderBook()
        self.validator = validator or OrderValidator()
        self.on_order_update = on_order_update
        self.on_fill = on_fill

        # Broker interface (to be set)
        self.broker = None

        # Statistics
        self.stats = {
            "orders_created": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_commission": 0.0,
            "total_volume": 0.0,
        }

    def set_broker(self, broker):
        """Set the broker interface."""
        self.broker = broker

    def create_order(
        self,
        request: OrderRequest,
        current_price: float,
        buying_power: float = float("inf"),
    ) -> Tuple[Optional[Order], str]:
        """
        Create a new order from request.

        Returns:
            Tuple of (order, message)
        """
        # Generate order ID
        order_id = str(uuid.uuid4())[:8]

        # Create order object
        order = Order(
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            strategy_id=request.strategy_id,
            signal_id=request.signal_id,
        )

        # Validate
        is_valid, reason = self.validator.validate(order, current_price, buying_power)

        if not is_valid:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            logger.warning(f"Order {order_id} rejected: {reason}")
            return order, reason

        # Add to book
        self.order_book.add(order)
        self.stats["orders_created"] += 1

        logger.info(
            f"Order created: {order_id} {order.side.value} {order.quantity} {order.symbol}"
        )

        return order, "Order created"

    def submit_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Submit order to broker.

        Returns:
            Tuple of (success, message)
        """
        order = self.order_book.get(order_id)
        if not order:
            return False, "Order not found"

        if order.status != OrderStatus.PENDING:
            return False, f"Cannot submit order in {order.status.value} status"

        # Submit to broker
        if self.broker:
            try:
                broker_id = self.broker.submit_order(order)
                order.broker_order_id = broker_id
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now()

                self.order_book.update_status(order_id, OrderStatus.SUBMITTED)

                logger.info(f"Order {order_id} submitted to broker: {broker_id}")

                if self.on_order_update:
                    self.on_order_update(order)

                return True, f"Submitted with broker ID: {broker_id}"

            except Exception as e:
                order.status = OrderStatus.FAILED
                order.reject_reason = str(e)
                self.order_book.update_status(order_id, OrderStatus.FAILED)
                logger.error(f"Order {order_id} submission failed: {e}")
                return False, str(e)
        else:
            # No broker - simulate immediate acceptance
            order.status = OrderStatus.ACCEPTED
            order.submitted_at = datetime.now()
            self.order_book.update_status(order_id, OrderStatus.ACCEPTED)
            return True, "Accepted (no broker)"

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Cancel an order.

        Returns:
            Tuple of (success, message)
        """
        order = self.order_book.get(order_id)
        if not order:
            return False, "Order not found"

        if not order.is_active:
            return False, f"Cannot cancel order in {order.status.value} status"

        # Cancel with broker
        if self.broker and order.broker_order_id:
            try:
                self.broker.cancel_order(order.broker_order_id)
            except Exception as e:
                logger.warning(f"Broker cancel failed: {e}")

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        self.order_book.update_status(order_id, OrderStatus.CANCELLED)
        self.stats["orders_cancelled"] += 1

        logger.info(f"Order {order_id} cancelled")

        if self.on_order_update:
            self.on_order_update(order)

        return True, "Order cancelled"

    def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_limit_price: Optional[float] = None,
        new_stop_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Modify an existing order.

        Returns:
            Tuple of (success, message)
        """
        order = self.order_book.get(order_id)
        if not order:
            return False, "Order not found"

        if not order.is_active:
            return False, f"Cannot modify order in {order.status.value} status"

        # Modify with broker
        if self.broker and order.broker_order_id:
            try:
                self.broker.modify_order(
                    order.broker_order_id,
                    quantity=new_quantity,
                    limit_price=new_limit_price,
                    stop_price=new_stop_price,
                )
            except Exception as e:
                logger.warning(f"Broker modify failed: {e}")
                return False, str(e)

        # Update local order
        if new_quantity is not None:
            order.quantity = new_quantity
        if new_limit_price is not None:
            order.limit_price = new_limit_price
        if new_stop_price is not None:
            order.stop_price = new_stop_price

        logger.info(f"Order {order_id} modified")

        if self.on_order_update:
            self.on_order_update(order)

        return True, "Order modified"

    def process_fill(
        self,
        order_id: str,
        fill_quantity: float,
        fill_price: float,
        commission: float = 0.0,
        exchange: str = "",
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Process a fill for an order.

        Returns:
            Tuple of (success, message)
        """
        order = self.order_book.get(order_id)
        if not order:
            return False, "Order not found"

        # Create fill record
        fill = OrderFill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order_id,
            timestamp=timestamp or datetime.now(),
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            exchange=exchange,
        )

        # Add fill to order
        old_status = order.status
        order.add_fill(fill)

        # Update order book if status changed
        if order.status != old_status:
            self.order_book.update_status(order_id, order.status)

        # Update statistics
        self.stats["total_commission"] += commission
        self.stats["total_volume"] += fill_quantity * fill_price

        if order.status == OrderStatus.FILLED:
            self.stats["orders_filled"] += 1

        logger.info(
            f"Fill: {order_id} {fill_quantity}@{fill_price:.2f} "
            f"({order.fill_rate:.1%} filled)"
        )

        # Callbacks
        if self.on_fill:
            self.on_fill(order, fill)
        if self.on_order_update:
            self.on_order_update(order)

        return True, f"Fill processed: {fill_quantity}@{fill_price:.2f}"

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all active orders.

        Returns:
            Number of orders cancelled
        """
        active = self.order_book.get_active_orders(symbol)
        cancelled = 0

        for order in active:
            success, _ = self.cancel_order(order.order_id)
            if success:
                cancelled += 1

        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    def get_position_from_fills(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate position from fills for a symbol.

        Returns:
            Dict with position details
        """
        orders = self.order_book.get_orders_by_symbol(symbol)

        net_quantity = 0.0
        total_cost = 0.0
        total_commission = 0.0

        for order in orders:
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                sign = 1 if order.side == OrderSide.BUY else -1
                net_quantity += sign * order.filled_quantity
                total_cost += sign * order.filled_quantity * order.average_fill_price
                total_commission += order.total_commission

        avg_price = abs(total_cost / net_quantity) if net_quantity != 0 else 0

        return {
            "symbol": symbol,
            "quantity": net_quantity,
            "average_price": avg_price,
            "cost_basis": total_cost,
            "commission": total_commission,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        return {
            **self.stats,
            "active_orders": len(self.order_book.get_active_orders()),
            "total_orders": len(self.order_book.orders),
        }


class BracketOrderManager:
    """
    Manages bracket orders (entry + stop loss + take profit).
    """

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.brackets: Dict[str, Dict[str, str]] = {}  # bracket_id -> order_ids

    def create_bracket(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        current_price: float,
        entry_type: OrderType = OrderType.LIMIT,
    ) -> Tuple[Optional[str], str]:
        """
        Create a bracket order.

        Returns:
            Tuple of (bracket_id, message)
        """
        bracket_id = str(uuid.uuid4())[:8]

        # Create entry order
        entry_request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=entry_type,
            limit_price=entry_price if entry_type == OrderType.LIMIT else None,
        )

        entry_order, msg = self.order_manager.create_order(entry_request, current_price)
        if not entry_order or entry_order.status == OrderStatus.REJECTED:
            return None, f"Entry order failed: {msg}"

        # Create stop loss order (opposite side)
        stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        stop_request = OrderRequest(
            symbol=symbol,
            side=stop_side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
        )

        stop_order, _ = self.order_manager.create_order(stop_request, current_price)

        # Create take profit order (opposite side)
        tp_request = OrderRequest(
            symbol=symbol,
            side=stop_side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit_price,
        )

        tp_order, _ = self.order_manager.create_order(tp_request, current_price)

        # Store bracket
        self.brackets[bracket_id] = {
            "entry": entry_order.order_id,
            "stop_loss": stop_order.order_id if stop_order else None,
            "take_profit": tp_order.order_id if tp_order else None,
        }

        logger.info(f"Bracket order created: {bracket_id}")

        return bracket_id, "Bracket order created"

    def cancel_bracket(self, bracket_id: str) -> bool:
        """Cancel all orders in a bracket."""
        if bracket_id not in self.brackets:
            return False

        for order_id in self.brackets[bracket_id].values():
            if order_id:
                self.order_manager.cancel_order(order_id)

        del self.brackets[bracket_id]
        return True


def create_market_order(
    symbol: str, quantity: float, side: str = "buy"
) -> OrderRequest:
    """Convenience function to create a market order request."""
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.MARKET,
    )


def create_limit_order(
    symbol: str, quantity: float, price: float, side: str = "buy"
) -> OrderRequest:
    """Convenience function to create a limit order request."""
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=price,
    )


def create_stop_order(
    symbol: str, quantity: float, stop_price: float, side: str = "sell"
) -> OrderRequest:
    """Convenience function to create a stop order request."""
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.STOP,
        stop_price=stop_price,
    )
