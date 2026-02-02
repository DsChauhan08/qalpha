"""
Broker Interface Module - Quantum Alpha V1
Abstract broker API and implementations per agent.md Section 6.

Implements:
- Abstract broker interface
- Alpaca paper trading integration
- IBKR interface (stub)
- Broker-agnostic order handling
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""

    ALPACA = "alpaca"
    IBKR = "ibkr"
    PAPER = "paper"  # Internal paper trading


@dataclass
class BrokerPosition:
    """Standardized position from broker."""

    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    current_price: float
    side: str  # "long" or "short"


@dataclass
class BrokerAccount:
    """Standardized account info from broker."""

    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    margin_used: float = 0.0
    day_trades_remaining: int = 3
    pattern_day_trader: bool = False


@dataclass
class BrokerOrder:
    """Standardized order from broker."""

    broker_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    average_fill_price: Optional[float]


class BrokerInterface(ABC):
    """
    Abstract broker interface.
    All broker implementations must inherit from this.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker. Returns success status."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """Get account information."""
        pass

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Get all positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> str:
        """Submit order, return broker order ID."""
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order by broker order ID."""
        pass

    @abstractmethod
    def get_order(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get order status."""
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get all open orders."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote (bid, ask, last)."""
        pass

    @abstractmethod
    def get_bars(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars."""
        pass


class AlpacaBroker(BrokerInterface):
    """
    Alpaca broker implementation.
    Uses alpaca-trade-api for paper and live trading.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if paper:
            self.base_url = base_url or os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            )
        else:
            self.base_url = base_url or "https://api.alpaca.markets"

        self.api = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            # Try to import alpaca
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient

                self._use_new_sdk = True
            except ImportError:
                try:
                    import alpaca_trade_api as tradeapi

                    self._use_new_sdk = False
                except ImportError:
                    logger.error("Neither alpaca-py nor alpaca-trade-api installed")
                    return False

            if not self.api_key or not self.secret_key:
                logger.error("Alpaca API credentials not provided")
                return False

            if self._use_new_sdk:
                self.api = TradingClient(self.api_key, self.secret_key, paper=True)
                self.data_client = StockHistoricalDataClient(
                    self.api_key, self.secret_key
                )
            else:
                self.api = tradeapi.REST(
                    self.api_key, self.secret_key, self.base_url, api_version="v2"
                )

            # Test connection
            account = self.get_account()
            if account:
                self._connected = True
                logger.info(f"Connected to Alpaca: {account.account_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self):
        """Disconnect from Alpaca."""
        self.api = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self.api is not None

    def get_account(self) -> Optional[BrokerAccount]:
        """Get Alpaca account info."""
        if not self.api:
            return None

        try:
            if self._use_new_sdk:
                account = self.api.get_account()
                return BrokerAccount(
                    account_id=str(account.id),
                    buying_power=float(account.buying_power),
                    cash=float(account.cash),
                    portfolio_value=float(account.portfolio_value),
                    equity=float(account.equity),
                    day_trades_remaining=account.daytrade_count
                    if hasattr(account, "daytrade_count")
                    else 3,
                    pattern_day_trader=account.pattern_day_trader,
                )
            else:
                account = self.api.get_account()
                return BrokerAccount(
                    account_id=account.id,
                    buying_power=float(account.buying_power),
                    cash=float(account.cash),
                    portfolio_value=float(account.portfolio_value),
                    equity=float(account.equity),
                    day_trades_remaining=int(account.daytrade_count)
                    if hasattr(account, "daytrade_count")
                    else 3,
                    pattern_day_trader=account.pattern_day_trader,
                )
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_positions(self) -> List[BrokerPosition]:
        """Get all Alpaca positions."""
        if not self.api:
            return []

        try:
            if self._use_new_sdk:
                positions = self.api.get_all_positions()
            else:
                positions = self.api.list_positions()

            return [
                BrokerPosition(
                    symbol=p.symbol,
                    quantity=float(p.qty),
                    market_value=float(p.market_value),
                    cost_basis=float(p.cost_basis),
                    unrealized_pnl=float(p.unrealized_pl),
                    current_price=float(p.current_price),
                    side="long" if float(p.qty) > 0 else "short",
                )
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for symbol."""
        if not self.api:
            return None

        try:
            if self._use_new_sdk:
                p = self.api.get_open_position(symbol)
            else:
                p = self.api.get_position(symbol)

            return BrokerPosition(
                symbol=p.symbol,
                quantity=float(p.qty),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pnl=float(p.unrealized_pl),
                current_price=float(p.current_price),
                side="long" if float(p.qty) > 0 else "short",
            )
        except Exception:
            return None

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> str:
        """Submit order to Alpaca."""
        if not self.api:
            raise Exception("Not connected to broker")

        try:
            if self._use_new_sdk:
                from alpaca.trading.requests import (
                    MarketOrderRequest,
                    LimitOrderRequest,
                    StopOrderRequest,
                    StopLimitOrderRequest,
                )
                from alpaca.trading.enums import OrderSide, TimeInForce as TIF

                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                tif = TIF.DAY if time_in_force.lower() == "day" else TIF.GTC

                if order_type.lower() == "market":
                    request = MarketOrderRequest(
                        symbol=symbol, qty=quantity, side=order_side, time_in_force=tif
                    )
                elif order_type.lower() == "limit":
                    request = LimitOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=order_side,
                        time_in_force=tif,
                        limit_price=limit_price,
                    )
                elif order_type.lower() == "stop":
                    request = StopOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=order_side,
                        time_in_force=tif,
                        stop_price=stop_price,
                    )
                else:
                    request = StopLimitOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=order_side,
                        time_in_force=tif,
                        limit_price=limit_price,
                        stop_price=stop_price,
                    )

                order = self.api.submit_order(request)
                return str(order.id)
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_price=stop_price,
                )
                return order.id

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel Alpaca order."""
        if not self.api:
            return False

        try:
            if self._use_new_sdk:
                self.api.cancel_order_by_id(broker_order_id)
            else:
                self.api.cancel_order(broker_order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_order(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get Alpaca order status."""
        if not self.api:
            return None

        try:
            if self._use_new_sdk:
                o = self.api.get_order_by_id(broker_order_id)
            else:
                o = self.api.get_order(broker_order_id)

            return BrokerOrder(
                broker_order_id=str(o.id),
                symbol=o.symbol,
                side=str(o.side),
                order_type=str(o.type),
                quantity=float(o.qty),
                filled_quantity=float(o.filled_qty) if o.filled_qty else 0,
                limit_price=float(o.limit_price) if o.limit_price else None,
                stop_price=float(o.stop_price) if o.stop_price else None,
                status=str(o.status),
                submitted_at=o.submitted_at,
                filled_at=o.filled_at,
                average_fill_price=float(o.filled_avg_price)
                if o.filled_avg_price
                else None,
            )
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open Alpaca orders."""
        if not self.api:
            return []

        try:
            if self._use_new_sdk:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus

                request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                orders = self.api.get_orders(request)
            else:
                orders = self.api.list_orders(status="open")

            result = [
                BrokerOrder(
                    broker_order_id=str(o.id),
                    symbol=o.symbol,
                    side=str(o.side),
                    order_type=str(o.type),
                    quantity=float(o.qty),
                    filled_quantity=float(o.filled_qty) if o.filled_qty else 0,
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    stop_price=float(o.stop_price) if o.stop_price else None,
                    status=str(o.status),
                    submitted_at=o.submitted_at,
                    filled_at=o.filled_at,
                    average_fill_price=float(o.filled_avg_price)
                    if o.filled_avg_price
                    else None,
                )
                for o in orders
            ]

            if symbol:
                result = [o for o in result if o.symbol == symbol]

            return result
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote from Alpaca."""
        if not self.api:
            return {}

        try:
            if self._use_new_sdk:
                from alpaca.data.requests import StockLatestQuoteRequest

                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self.data_client.get_stock_latest_quote(request)[symbol]
                return {
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                }
            else:
                quote = self.api.get_latest_quote(symbol)
                return {
                    "bid": float(quote.bp),
                    "ask": float(quote.ap),
                    "last": (float(quote.bp) + float(quote.ap)) / 2,
                }
        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return {}

    def get_bars(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars from Alpaca."""
        if not self.api:
            return []

        try:
            if self._use_new_sdk:
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                from datetime import datetime, timedelta

                tf_map = {
                    "1min": TimeFrame.Minute,
                    "5min": TimeFrame.Minute,
                    "15min": TimeFrame.Minute,
                    "1hour": TimeFrame.Hour,
                    "1day": TimeFrame.Day,
                }

                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf_map.get(timeframe, TimeFrame.Day),
                    start=datetime.now() - timedelta(days=limit),
                    limit=limit,
                )
                bars = self.data_client.get_stock_bars(request)[symbol]

                return [
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }
                    for bar in bars
                ]
            else:
                bars = self.api.get_bars(symbol, timeframe, limit=limit)
                return [
                    {
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                    }
                    for bar in bars
                ]
        except Exception as e:
            logger.error(f"Failed to get bars: {e}")
            return []


class IBKRBroker(BrokerInterface):
    """
    Interactive Brokers interface stub.
    Requires IB Gateway or TWS running.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to IB Gateway/TWS."""
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = self.ib.isConnected()

            if self._connected:
                logger.info("Connected to Interactive Brokers")

            return self._connected

        except ImportError:
            logger.error("ib_insync not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib:
            self.ib.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self.ib and self.ib.isConnected()

    def get_account(self) -> Optional[BrokerAccount]:
        """Get IBKR account info."""
        if not self.is_connected():
            return None

        try:
            account_values = self.ib.accountValues()

            values = {v.tag: v.value for v in account_values}

            return BrokerAccount(
                account_id=account_values[0].account if account_values else "unknown",
                buying_power=float(values.get("BuyingPower", 0)),
                cash=float(values.get("TotalCashValue", 0)),
                portfolio_value=float(values.get("NetLiquidation", 0)),
                equity=float(values.get("EquityWithLoanValue", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to get IBKR account: {e}")
            return None

    def get_positions(self) -> List[BrokerPosition]:
        """Get IBKR positions."""
        if not self.is_connected():
            return []

        try:
            positions = self.ib.positions()

            return [
                BrokerPosition(
                    symbol=p.contract.symbol,
                    quantity=float(p.position),
                    market_value=float(p.marketValue)
                    if hasattr(p, "marketValue")
                    else 0,
                    cost_basis=float(p.avgCost) * float(p.position),
                    unrealized_pnl=float(p.unrealizedPNL)
                    if hasattr(p, "unrealizedPNL")
                    else 0,
                    current_price=float(p.marketPrice)
                    if hasattr(p, "marketPrice")
                    else 0,
                    side="long" if float(p.position) > 0 else "short",
                )
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get IBKR positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get IBKR position for symbol."""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> str:
        """Submit order to IBKR."""
        if not self.is_connected():
            raise Exception("Not connected to IBKR")

        try:
            from ib_insync import Stock, Order

            contract = Stock(symbol, "SMART", "USD")

            action = "BUY" if side.lower() == "buy" else "SELL"

            if order_type.lower() == "market":
                order = Order(action=action, totalQuantity=quantity, orderType="MKT")
            elif order_type.lower() == "limit":
                order = Order(
                    action=action,
                    totalQuantity=quantity,
                    orderType="LMT",
                    lmtPrice=limit_price,
                )
            elif order_type.lower() == "stop":
                order = Order(
                    action=action,
                    totalQuantity=quantity,
                    orderType="STP",
                    auxPrice=stop_price,
                )
            else:
                order = Order(
                    action=action,
                    totalQuantity=quantity,
                    orderType="STP LMT",
                    lmtPrice=limit_price,
                    auxPrice=stop_price,
                )

            trade = self.ib.placeOrder(contract, order)

            return str(trade.order.orderId)

        except Exception as e:
            logger.error(f"Failed to submit IBKR order: {e}")
            raise

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel IBKR order."""
        if not self.is_connected():
            return False

        try:
            # Find the trade by order ID
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == broker_order_id:
                    self.ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel IBKR order: {e}")
            return False

    def get_order(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get IBKR order status."""
        # Stub - would need proper implementation
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open IBKR orders."""
        # Stub - would need proper implementation
        return []

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote from IBKR."""
        if not self.is_connected():
            return {}

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            ticker = self.ib.reqMktData(contract)
            time.sleep(1)  # Wait for data

            return {
                "bid": float(ticker.bid) if ticker.bid else 0,
                "ask": float(ticker.ask) if ticker.ask else 0,
                "last": float(ticker.last) if ticker.last else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get IBKR quote: {e}")
            return {}

    def get_bars(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars from IBKR."""
        # Stub - would need proper implementation
        return []


def get_broker(broker_type: str = "alpaca", **kwargs) -> BrokerInterface:
    """
    Factory function to get broker instance.

    Args:
        broker_type: Type of broker (alpaca, ibkr, paper)
        **kwargs: Broker-specific arguments

    Returns:
        BrokerInterface instance
    """
    if broker_type.lower() == "alpaca":
        return AlpacaBroker(**kwargs)
    elif broker_type.lower() == "ibkr":
        return IBKRBroker(**kwargs)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
