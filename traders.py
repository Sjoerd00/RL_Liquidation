# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:01:23 2023

@author: Sjoerd
"""

from abc import ABC, abstractmethod
import random
from orderbook import Order
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from pandas import DataFrame
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class Trader(ABC):
    """
    Abstract base class for a trading strategy that interacts with an order book.

    Args:
        orderbook (OrderBook): The order book that the trader interacts with.
        inventory (int): The initial inventory of the trader.

    Attributes:
        orderbook (OrderBook): The order book that the trader interacts with.
        inventory (int): The current inventory of the trader.
        orders (list of Order): The list of orders submitted by the trader.
        unrealized (float): The unrealized P&L of the trader.
        realized (float): The realized P&L of the trader.
    """

    @abstractmethod
    def submit_orders(self):
        """
        Submits orders to the order book based on the trader's trading strategy.

        This method must be implemented by derived classes.
        """
        pass

    def get_pnl(self):
        """
        Calculates the total P&L of the trader.

        Returns:
            float: The total P&L of the trader.
        """
        # Calculate realized P&L and inventory position
        partial_trades = [order for order in self.orders if (order.filled_quantity > 0)&(order.quantity>0)]
        full_trades = [order for order in self.orders if order.quantity==0]

        # remove completed orders
        self.executed_trades(len([order for order in self.orders if order.quantity==0]))
        self.orders = [order for order in self.orders if order.quantity>0]

        self.inventory += sum([(-order.filled_quantity, order.filled_quantity)[order.direction=='bid']\
                               for order in full_trades])
        self.realized += sum([(order.price, -order.price)[order.direction=='bid'] * order.filled_quantity\
                              for order in full_trades])

        self.limit_partial_realized = sum([(order.price, -order.price)[order.direction=='bid'] *\
                                      order.filled_quantity for order in partial_trades])
        self.limit_partial_inventory = sum([(-order.filled_quantity, order.filled_quantity)[order.direction=='bid']\
                               for order in partial_trades])

        # Calculate unrealized P&L
        best_price = self.ob.get_best_bid() if self.inventory > 0 else self.ob.get_best_ask()
        self.unrealized = (self.inventory + self.limit_partial_inventory) * best_price

        # Calculate total P&L
        pnl = self.realized + self.limit_partial_realized + self.unrealized

        return pnl

    def executed_trades(self, t):
        self.executed += t

    def plot_outstanding(self):
        bids = [(order.price, order.quantity) for order in self.orders if order.direction=='bid']
        asks = [(order.price, order.quantity) for order in self.orders if order.direction=='ask']
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

class ZeroIntelligenceTrader(Trader):
    """
    A subclass of Trader that represents a simple trading algorithm that submits
    limit and market orders based on a random decision process.

    Attributes:
    -----------
    orderbook : OrderBook
        The order book that the trader will interact with.
    inventory : int
        The inventory position of the trader.
    orders : List[Order]
        The list of orders submitted by the trader.
    unrealized : float
        The unrealized P&L of the trader.
    realized : float
        The realized P&L of the trader.
    epsilon : float
        The probability of submitting a market order, between 0 and 1.

    Methods:
    --------
    submit_orders():
        Submit a limit or market order based on a random decision process.
    """
    def __init__(self, orderbook, inventory=0, epsilon=0.0):
        """
        Initializes a new instance of the ZeroIntelligenceTrader class.

        Parameters:
        -----------
        orderbook : OrderBook
            The order book that the trader will interact with.
        inventory : int, optional
            The inventory position of the trader, default is 0.
        epsilon : float, optional
            The probability of submitting a market order, between 0 and 1, default is 0.0.
        """
        self.ob = orderbook
        self.inventory = inventory
        self.orders = []
        self.unrealized = 0
        self.realized = 0
        self.epsilon = epsilon
        self.executed = 0

    def submit_orders(self):
        """
        Submit a limit or market order based on a random decision process.
        """
        ## important to pay attention to intialization prices
        # quantity = random.randint(1, 10)
        quantity = 1
        if random.random() > self.epsilon:
            if random.random() < 0.5:
                bid_price = self.ob.get_best_bid() if self.ob.bids else 50
                order_type = "limit"
                direction = "bid"
                price = max(1, bid_price - random.randint(1, 5) / 10)
            else:
                ask_price = self.ob.get_best_ask() if self.ob.asks else 50
                order_type = "limit"
                direction = "ask"
                price = max(1, ask_price + random.randint(1, 5) / 10)
            order = Order(price, quantity, direction, order_type)
            self.ob.add_order(order)
            self.orders.append(order)
        else:
            order_type = "market"
            direction = "bid" if random.random() < 0.5 else "ask"
            price = None
            order = Order(price, quantity, direction, order_type)
            P, Q = self.ob.add_order(order)

            if direction == "bid":
                self.inventory += Q
                self.realized -= Q * P
            else:
                self.inventory -= Q
                self.realized += Q * P

class OptimalLiquidatorSL(Trader):
    """
    An implementation of a simple liquidation strategy for a large inventory over a fixed time horizon,
    where orders are placed at the best bid or ask price and are not modified or cancelled.
    Inherits from the Trader class.

    Parameters:
    -----------
    orderbook : OrderBook
        An instance of the OrderBook class representing the current state of the market.
    inventory : float
        The current inventory to be liquidated.
    time_to_execute : int, default=10
        The maximum time horizon for the liquidation process.

    Attributes:
    -----------
    ob : OrderBook
        An instance of the OrderBook class representing the current state of the market.
    inventory : float
        The current inventory to be liquidated.
    side : str
        The side of the market on which to execute orders (either 'ask' or 'bid').
    orders : list
        A list of Order objects representing the agent's current outstanding orders.
    revenue : float
        The total revenue generated by the agent during the liquidation process.
    quantity : float
        The total quantity of shares sold by the agent during the liquidation process.
    previous_best_price : float
        The best bid or ask price in the orderbook at the time of the agent's previous action.
    score : float
        The agent's total score, defined as the sum of the rewards obtained during the liquidation process.
    max_time : int
        The maximum time horizon for the liquidation process.

    Methods:
    --------
    submit_orders():
        Submits orders to the orderbook at the best bid or ask price, depending on the agent's side and inventory.
    update(P=None, Q=None):
        Updates the agent's state, outstanding orders, revenue, and score based on the current state of the market.
    """

    def __init__(self, orderbook, inventory, time_to_execute=10):
        self.ob = orderbook
        self.inventory = inventory
        self.side = 'ask' if inventory > 0 else 'bid'
        self.orders, self.revenue, self.quantity, self.previous_best_price,\
            self.score, self.max_time = [], 0, 0, 0, 0, time_to_execute

    def submit_orders(self):
        self.update()
        if self.inventory > 0 and self.max_time > 1:
            self.previous_best_price = self.ob.get_best_ask() if self.side == 'ask' else self.ob.get_best_bid()
            order = Order(self.previous_best_price, self.inventory, self.side, 'limit')
            self.ob.add_order(order)
            self.orders.append(order)

        elif self.inventory > 0 and self.max_time == 1:
            self.previous_best_price = self.ob.get_best_ask() if self.side == 'ask' \
                else self.ob.get_best_bid()
            Revenue, Q = self.ob.add_order(Order(None, self.inventory, self.side, 'market'))
            self.max_time -= 1
            if Q>0:
                self.update(Revenue/Q, Q)

        else:
            self.max_time = 0
            return

        self.max_time -= 1

    def update(self, P=None, Q=None):
        if self.max_time>=0 and self.inventory>0:
            for o in self.orders:
                self.revenue += o.filled_quantity * o.price
                self.quantity += o.filled_quantity
                self.inventory -= o.filled_quantity
                self.score += o.filled_quantity * o.price - \
                    (o.filled_quantity + o.quantity) * self.previous_best_price
                self.orders.remove(o)
                if o.quantity > 0:
                    self.ob.delete_order(o)

            if P is not None:
                self.revenue += P * Q
                self.quantity += Q
                self.inventory -= Q
                self.score += P * Q - max(self.inventory, Q)  * self.previous_best_price

            # print(f'Score is {round(self.score,2)}, with {self.inventory} inventory and {self.max_time} time left.')

class OptimalLiquidatorQLNN(Trader):
    """
    An implementation of a deep Q-learning algorithm for optimal liquidation
    of a large inventory over a fixed time horizon, using a neural network
    to approximate the optimal action-value function. Inherits from the Trader class.

    Parameters:
    -----------
    orderbook : OrderBook
        An instance of the OrderBook class representing the current state of the market.
    inventory : float
        The current inventory to be liquidated.
    time_to_execute : int, default=10
        The maximum time horizon for the liquidation process.
    alpha : float, default=0.001
        The learning rate for the optimizer.
    gamma : float, default=0.1
        The discount factor for future rewards.
    epsilon : float, default=0.1
        The exploration rate for the epsilon-greedy algorithm.
    q_network : keras.Sequential, default=None
        A neural network used to approximate the optimal action-value function.
    counter : int, default=0
        A counter used to decrease the exploration rate over time.

    Attributes:
    -----------
    ob : OrderBook
        An instance of the OrderBook class representing the current state of the market.
    inventory : float
        The current inventory to be liquidated.
    side : str
        The side of the market on which to execute orders (either 'ask' or 'bid').
    max_time : int
        The maximum time horizon for the liquidation process.
    gamma : float
        The discount factor for future rewards.
    epsilon : float
        The exploration rate for the epsilon-greedy algorithm.
    alpha : float
        The learning rate for the optimizer.
    q_network : keras.Sequential
        A neural network used to approximate the optimal action-value function.
    replay_buffer : list
        A list of tuples representing the agent's experience replay buffer.
    revenue : float
        The total revenue generated by the agent during the liquidation process.
    quantity : float
        The total quantity of shares sold by the agent during the liquidation process.
    previous_best_price : float
        The best bid or ask price in the orderbook at the time of the agent's previous action.
    score : float
        The agent's total score, defined as the sum of the rewards obtained during the liquidation process.
    state : list
        A list of features representing the current state of the agent.
    action : int
        The action chosen by the agent.
    orders : list
        A list of Order objects representing the agent's current outstanding orders.
    possible_action : list
        A list of possible actions the agent can take at each timestep.
    optimizer : keras.optimizers.Adam
        An optimizer used to update the parameters of the neural network.
    trained : bool
        A flag indicating whether the neural network has been trained.
    starting_inv : float
        The initial inventory to be liquidated.
    starting_time : int
        The initial maximum time horizon for the liquidation process.
    counter : int
        A counter used to decrease the exploration rate over time.

    Methods:
    --------
    initialize_q_network():
        Initializes the neural network used to approximate the optimal action-value function.
    choose_action():
        Chooses an action for the agent based on the current state and exploration rate.
    submit_orders():
        Submits orders to the orderbook based on the current state and chosen action.
    update(P=None, Q=None):
        Updates the agent's state, outstanding orders, revenue, and score based on the current state of the market.
    get_normalized_states(vector):
        Normalizes the input vector to a range of [0, 1] based on the initial inventory, maximum time horizon,
        and possible action values.
    get_stats():
        Computes statistics on the current state of the orderbook, including the orderbook imbalance,
        volume-weighted prices, and inventory-to-depth ratio.
    update_q_network(market=False, market_order=None):
        Updates the parameters of the neural network based on the agent's experience replay buffer and
        the Bellman equation. If `market=True`, also updates the network based on a market order.
    """
    def __init__(self, orderbook, inventory, time_to_execute=10, alpha=0.001, gamma=0.1, epsilon=0.1, q_network=None,
                 counter=0):
        self.ob = orderbook
        self.inventory = inventory
        self.side = 'ask' if inventory > 0 else 'bid'
        self.max_time = time_to_execute
        self.gamma = gamma
        self.epsilon = epsilon * np.exp(-counter/10)
        self.alpha = alpha
        self.q_network = q_network
        self.replay_buffer = []
        self.revenue, self.quantity, self.previous_best_price, self.score = 0, 0, 0, 0
        self.state = None
        self.action = None
        self.orders = []
        self.possible_action = [i for i in range(0, 20, 5)]
        self.optimizer = Adam(learning_rate=alpha)
        self.trained = False
        self.starting_inv = inventory
        self.starting_time = time_to_execute
        self.counter = counter

        if q_network is None:
            self.initialize_q_network()

    def initialize_q_network(self):
        input_dim = 7  # inventory, max_time, relative price, imbalance, vwp_a/best_a, vwp-b/best_b, inv to depth
        output_dim = 1  # reward
        self.q_network = Sequential()
        self.q_network.add(Dense(32, input_dim=input_dim, activation='relu'))
        self.q_network.add(Dense(16, activation='relu'))
        self.q_network.add(Dense(output_dim, activation='linear'))
        self.q_network.compile(loss='mse', optimizer=self.optimizer)

    def choose_action(self):
        imb, vwpa_besta, vwpb_bestb, inv_to_depth = self.get_stats()
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.possible_action)
        else:
            states = np.array([(self.inventory, self.max_time, a, imb, \
                                vwpa_besta, vwpb_bestb, inv_to_depth)\
                               for a in self.possible_action], dtype=np.float32)
            q_values = self.q_network.predict(self.get_normalized_states(states))
            action = self.possible_action[np.argmax(q_values)]
        return action, imb, vwpa_besta, vwpb_bestb, inv_to_depth

    def submit_orders(self):
        self.update()
        if self.inventory == 0 and not self.trained:
            self.update_q_network()

        if self.inventory > 0 and self.max_time > 1:
            self.previous_best_price = self.ob.get_best_ask() if self.side == 'ask' else self.ob.get_best_bid()
            self.action, im, a, b, itd = self.choose_action()
            order = Order(self.previous_best_price+self.action/100, self.inventory, self.side, 'limit')
            self.ob.add_order(order)
            self.orders.append(order)
            self.replay_buffer.append(([self.inventory, self.max_time, self.action, im, a, b, itd], None, None, None))

        elif self.inventory > 0 and self.max_time == 1:
            self.previous_best_price = self.ob.get_best_ask() if self.side == 'ask' else self.ob.get_best_bid()
            Revenue, Q = self.ob.add_order(Order(None, self.inventory, self.side, 'market'))
            if Q>0:
                self.update(Revenue/Q, Q)
            else:
                self.update(0, 0)
            self.update_q_network(market=True, market_order=(Revenue, Q))
            self.max_time -= 1

        else:
            self.max_time = 0
            return

        self.max_time -= 1

    def update(self, P=None, Q=None):
        if self.max_time >= 0 and self.inventory > 0:
            for o in self.orders:
                self.revenue += o.filled_quantity * o.price
                self.quantity += o.filled_quantity
                self.inventory -= o.filled_quantity
                reward =  (o.filled_quantity * o.price)  - \
                          (o.filled_quantity+o.quantity) * self.previous_best_price
                self.score += reward
                self.orders.remove(o)
                if o.quantity > 0:
                    self.ob.delete_order(o)

                next_state = [self.inventory, self.max_time, None] + list(self.get_stats())

                self.replay_buffer[-1] = (self.replay_buffer[-1][0], reward, next_state, self.max_time)

            if P is not None:
                self.revenue += P * Q
                self.quantity += Q
                self.inventory -= Q
                self.score += P * Q - max(self.inventory, Q)  * self.previous_best_price
                # self.update_q_network()

            # print(f'Score is {round(self.score, 2)}, with {self.inventory} inventory and {self.max_time} time left.')

    def get_normalized_states(self, vector):

        if len(vector.shape) == 1:
            vector = vector.reshape((1, -1))

        vector = vector / np.array([self.starting_inv, self.starting_time,
                                    max(self.possible_action),
                                   1,1,1,1])

        return vector

    def get_stats(self):
        imb = self.ob.get_order_book_imbalance()
        vwpa_besta = self.ob.get_volume_weighted_price('ask') / self.ob.get_best_ask()
        vwpb_bestb = self.ob.get_volume_weighted_price('bid') / self.ob.get_best_bid()
        inv_to_depth = self.inventory / self.ob.get_total_quantity(self.side)
        return imb, vwpa_besta, vwpb_bestb, inv_to_depth

    def update_q_network(self, market=False, market_order=None):


        # minibatch = np.array(self.replay_buffer)[np.random.choice(len(self.replay_buffer)+1, size=10, replace=False)]
        minibatch = self.replay_buffer

        if len(minibatch)==0:
            return
        if len(minibatch) == 1:
            rewards = np.array([minibatch[0][1]], dtype=np.float32)
            states = np.array([minibatch[0][0]], dtype=np.float32)
            next_states = np.array([minibatch[0][2]], dtype=np.float32)
        else:
            rewards = np.array([exp[1] for exp in minibatch], dtype=np.float32)
            states = np.array([exp[0] for exp in minibatch], dtype=np.float32)
            next_states = np.array([exp[2] for exp in minibatch], dtype=np.float32)

        if market:
            rewards = rewards - (market_order[1] * self.previous_best_price - market_order[0])

        next_q_values = np.zeros((len(minibatch, )))
        for j, s in enumerate(next_states):
            next_states_a = np.array([(s[0], s[1], a, s[3], s[4], s[5], s[6]) for a in self.possible_action], dtype=np.float32)
            next_q_values[j] = self.q_network.predict(self.get_normalized_states(next_states_a)).max()

        targets = rewards + self.gamma * next_q_values

        self.q_network.train_on_batch(self.get_normalized_states(states), targets)
        # K.set_value(self.q_network.optimizer.lr, self.alpha * np.exp(-self.counter/10)
        self.trained = True
