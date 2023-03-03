# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:50:46 2023

@author: Sjoerd
"""
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pandas import DataFrame

class Order:
    def __init__(self, price: float, quantity: float, direction: str, order_type: str) -> None:
        """
        Initializes an Order instance.

        Args:
            price (float): The price of the order.
            quantity (float): The quantity of the asset to be bought or sold.
            direction (str): The direction of the order, either "bid" or "ask".
            order_type (str): The type of the order, either "limit" or "market".
        """
        self.price = price
        self.quantity = quantity
        self.direction = direction
        self.order_type = order_type
        self.timestamp = None
        self.filled_quantity = 0
        self.action = None

    def __str__(self) -> str:
        return f"Order(price={self.price}, quantity={self.quantity}, direction='{self.direction}', order_type='{self.order_type}')"

class OrderBook:
    def __init__(self) -> None:
        """
        Initializes an OrderBook instance.
        """
        self.bids = deque()
        self.asks = deque()
        self.orders = 0

    def add_order(self, order: Order) -> Tuple[float, float]:
        """
        Adds an order to the book, either a limit or a market order.

        Args:
            order (Order): The order to be added.

        Returns:
            If the order is a market order, returns the total price and quantity of the filled orders.
        """
        if order.order_type == "limit":
            order.timestamp = self.orders
            self.orders += 1
            if order.direction == "bid":
                self.add_bid_order(order)
            elif order.direction == "ask":
                self.add_ask_order(order)
            self.match_limit_orders()
        elif order.order_type == "market":
            return self.execute_market_order(order)

    def add_bid_order(self, order: Order) -> None:
        """
        Adds a bid order to the book.

        Args:
            order (Order): The bid order to be added.
        """
        self.bids.append(order)
        self.bids = deque(sorted(self.bids, key=lambda x: (-x.price, x.timestamp)))

    def add_ask_order(self, order: Order) -> None:
        """
        Adds an ask order to the book.

        Args:
            order (Order): The ask order to be added.
        """
        self.asks.append(order)
        self.asks = deque(sorted(self.asks, key=lambda x: (x.price, x.timestamp)))

    def match_limit_orders(self) -> None:
        """
        Matches the best bid and ask orders in the book and fills them.
        """
        while self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            trade_quantity = min(best_bid.quantity, best_ask.quantity)

            best_bid.quantity -= trade_quantity
            best_ask.quantity -= trade_quantity

            best_bid.filled_quantity += trade_quantity
            best_ask.filled_quantity += trade_quantity

            if best_bid.quantity == 0:
                self.bids.popleft()
            if best_ask.quantity == 0:
                self.asks.popleft()

    def execute_market_order(self, order: Order) -> Tuple[float, float]:
        """
        Executes a market order and returns the total price and quantity of the filled orders.

        Args:
            order (Order): The market order to be executed.

        Returns:
            A tuple containing the total price and quantity of the filled orders.
        """
        total_price = 0
        total_quantity = 0
        book = [self.asks, self.bids][order.direction == "ask"]
        while order.quantity > 0 and book:
            best_order = book[0]
            if order.quantity >= best_order.quantity:
                trade_quantity = best_order.quantity
                order.quantity -= trade_quantity
                total_price += best_order.price * trade_quantity
                total_quantity += trade_quantity
                best_order.quantity = 0
                best_order.filled_quantity += trade_quantity
                book.popleft()
            else:
                trade_quantity = order.quantity
                order.quantity = 0
                total_price += best_order.price * trade_quantity
                total_quantity += trade_quantity
                best_order.quantity -= trade_quantity
                best_order.filled_quantity += trade_quantity
                break

        return total_price, total_quantity

    def change_order(self, order: Order, price: float, quantity: float) -> None:
        """
        Changes the price and/or quantity of an existing order in the book.

        Args:
            order (Order): The order to be changed.
            price (float): The new price of the order.
            quantity (float): The new quantity of the order.

        Raises:
            ValueError: If the order is not found in the book.
        """
        if order.direction == "bid":
            orders = self.bids
        else:
            orders = self.asks

        # find the order with the given timestamp and update its attributes
        for o in orders:
            if o.timestamp == order.timestamp:
                o.price = price
                o.quantity = quantity
                break
        else:
            raise ValueError("Order not found in the order book")

        # re-sort the deque according to the order book sorting logic
        orders = deque(sorted(orders, key=lambda x: (-x.price if x.direction == "bid" else x.price, x.timestamp)))
        if order.direction == "bid":
            self.bids = orders
        else:
            self.asks = orders

    def delete_order(self, order: Order) -> None:
        """
        Deletes an existing order from the book.

        Args:
            order (Order): The order to be deleted.

        Raises:
            ValueError: If the order is not found in the book.
        """
        if order.direction == "bid":
            orders = self.bids
        else:
            orders = self.asks

        for o in orders:
            if o.timestamp == order.timestamp:
                orders.remove(o)
                if order.direction == "bid":
                    self.bids = deque(sorted(orders, key=lambda x: (-x.price, x.timestamp)))
                else:
                    self.asks = deque(sorted(orders, key=lambda x: (x.price, x.timestamp)))
                return

        raise ValueError("Order not found in the order book")

    def get_best_bid(self) -> Optional[float]:
        """
        Returns the price of the best bid order in the book.

        Returns:
            The price of the best bid order in the book, or None if there are no bids.
        """
        return self.bids[0].price if self.bids else 1000

    def get_best_ask(self) -> Optional[float]:
        """
        Returns the price of the best ask order in the book.

        Returns:
            The price of the best ask order in the book, or None if there are no asks.
        """
        return self.asks[0].price if self.asks else 1

    def clean(self) -> None:
        """
        Removes orders that have been in the book for more than 500 timesteps.

        Args:
            current_timestamp (int): The current timestamp.
        """
        for i, orders in enumerate([self.bids, self.asks]):
            orders_re = deque(sorted(orders, key=lambda x: x.timestamp))

            while orders_re and self.get_time() - orders_re[0].timestamp > 2500:
                orders_re.popleft()
            if orders:
                orders_re = deque(sorted(orders_re, key=lambda x: (-x.price if x.direction == "bid" else x.price, x.timestamp)))

            if i == 0:
                self.bids = orders_re
            else:
                self.asks = orders_re

    def get_time(self):
        return self.orders

    def get_mid_price(self):
        """
        Returns midprice of the orderbook.
        """

        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()
        return (self.get_best_ask() + self.get_best_bid())/2

    def get_total_quantity(self, side):
        """
        Returns total quantity in the orderbook given a side.
        """

        if side == 'bid':
            return sum([order.quantity for order in self.bids])
        elif side == 'ask':
            return sum([order.quantity for order in self.asks])
        else:
            raise ValueError('Invalid side. Must be "bids" or "asks".')

    def get_order_book_imbalance(self):
        """
        Returns orderbook imbalance based on quantity in bids and asks.
        """
        total_bid_qty = self.get_total_quantity('bid')
        total_ask_qty = self.get_total_quantity('ask')
        if total_bid_qty + total_ask_qty == 0:
            return 0.0
        else:
            return (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

    def plot(self) -> None:
        """
        Plots a bar chart of the bid and ask orders in the book.
        """
        bids = [(order.price, order.quantity) for order in self.bids]
        asks = [(order.price, order.quantity) for order in self.asks]
        bids = DataFrame(bids).groupby(0).sum()
        asks = DataFrame(asks).groupby(0).sum()
        bids_prices, bids_quantities = bids.index.to_numpy(), bids.loc[:, 1].values
        asks_prices, asks_quantities = asks.index.to_numpy(), asks.loc[:, 1].values
        max_quantity = max(max(bids_quantities, default=0), max(asks_quantities, default=0))
        fig, ax = plt.subplots()
        ax.barh(bids_prices, bids_quantities, height=0.4, color='b', alpha=0.5, label='Bids')
        ax.barh(asks_prices, asks_quantities, height=0.4, color='y', alpha=0.5, label='Asks')
        ax.set_ylim(min(bids_prices)-1, max(asks_prices, default=0) + 1)
        ax.set_xlim(0, max_quantity + 1)
        ax.invert_yaxis()
        ax.legend()
        plt.show()

    def get_volume_weighted_price(self, side: str) -> float:
            """
            Returns the volume-weighted price (VWP) of the specified side of the order book,
            either at a specific price level or for the entire book.

            Args:
                side (str): The side of the order book to calculate VWP for, either "bids" or "asks".

            Returns:
                The volume-weighted price of the specified side of the order book.
            """
            if side == "bid":
                orders = self.bids
            elif side == "ask":
                orders = self.asks
            else:
                raise ValueError('Invalid side. Must be "bid" or "ask".')

             # calculate vwp for the entire side of the book
            total_qty = 0
            vwp = 0
            for order in orders:
                total_qty += order.quantity
                vwp += order.quantity * order.price
            if total_qty == 0:
                return 0.0
            else:
                return vwp / total_qty
