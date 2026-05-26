"""
Abstract base class for all exchange integrations.
Defines the common interface every exchange wrapper must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OHLCV:
    timestamp: float        # Unix timestamp (seconds)
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float


@dataclass
class Ticker:
    symbol:        str
    bid:           float
    ask:           float
    last:          float
    mark_price:    Optional[float] = None
    spot_price:    Optional[float] = None
    funding_rate:  Optional[float] = None
    open_interest: Optional[float] = None


@dataclass
class OrderResult:
    order_id:  str
    symbol:    str
    side:      str          # 'buy' or 'sell'
    type:      str          # 'market', 'limit', etc.
    amount:    float
    price:     float
    status:    str          # 'open', 'closed', 'canceled', etc.
    timestamp: float        # Unix timestamp (seconds)


# ---------------------------------------------------------------------------
# Abstract base exchange
# ---------------------------------------------------------------------------

class BaseExchange(ABC):
    """
    Abstract base class for all exchange wrappers.
    Subclasses must implement every abstract method.
    """

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> List[OHLCV]:
        """
        Fetch OHLCV candlestick data.

        Parameters
        ----------
        symbol    : trading pair, e.g. 'BTCUSD'
        timeframe : candle size, e.g. '1m', '5m', '15m', '1h', '4h', '1d'
        limit     : number of candles to return

        Returns
        -------
        List of OHLCV dataclasses, oldest first.
        """
        ...

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch the latest ticker snapshot for a symbol.

        Returns
        -------
        Ticker dataclass.
        """
        ...

    @abstractmethod
    def fetch_orderbook(self, symbol: str) -> dict:
        """
        Fetch the current order book.

        Returns
        -------
        dict with keys 'bids' and 'asks', each a list of [price, amount] pairs.
        """
        ...

    @abstractmethod
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch the current funding rate for a perpetual contract.

        Returns
        -------
        Funding rate as a decimal fraction (e.g. 0.0001 = 0.01%).
        """
        ...

    @abstractmethod
    def fetch_open_interest(self, symbol: str) -> float:
        """
        Fetch the current open interest for a perpetual contract.

        Returns
        -------
        Open interest in USD notional.
        """
        ...

    # ------------------------------------------------------------------
    # Account / trading
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_balance(self) -> dict:
        """
        Fetch the account balance.

        Returns
        -------
        dict mapping asset symbol to available balance float,
        e.g. {'USDT': 1234.56, 'BTC': 0.05}.
        """
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a new order.

        Parameters
        ----------
        symbol     : trading pair
        side       : 'buy' or 'sell'
        order_type : 'market' or 'limit'
        amount     : order size in base currency
        price      : limit price (required for 'limit' orders)

        Returns
        -------
        OrderResult dataclass.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Parameters
        ----------
        order_id : exchange-assigned order ID
        symbol   : trading pair

        Returns
        -------
        True if the cancellation was accepted, False otherwise.
        """
        ...

    @abstractmethod
    def fetch_open_orders(self, symbol: str) -> List[OrderResult]:
        """
        Fetch all open orders for a symbol.

        Returns
        -------
        List of OrderResult dataclasses.
        """
        ...
