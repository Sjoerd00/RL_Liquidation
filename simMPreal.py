# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:57:36 2023

@author: Sjoerd
"""
import numpy as np
from orderbook import OrderBook, Order
from traders import OptimalLiquidatorSL, OptimalLiquidatorQLNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def run_simulation(j, settings, QLNN=None, data=None, sim_steps=1000, r=0):
    # if QLNN is not None:
    #     ob = QLNN.ob
    # else:
    ob = OrderBook()
    trader3 = OptimalLiquidatorSL(ob, settings.inventory, time_to_execute=settings.step)
    if QLNN is None:
        trader4 = OptimalLiquidatorQLNN(ob, settings.inventory, time_to_execute=settings.step)
    else:
        trader4 = OptimalLiquidatorQLNN(ob, settings.inventory, time_to_execute=settings.step, q_network=QLNN.q_network, counter=QLNN.counter)

    rng = np.random.default_rng()
    start = np.random.randint(0, 100000)

    for i in range(start, data.shape[0]):

        # price process
        for _ in range(5):

            if len(ob.asks)+len(ob.asks)>5000:
                ob.clean()
            order_data = data.iloc[i, :].values
            direction = 'ask' if order_data[2]==2 else 'bid'
            quantity = order_data[-1]
            order_price = order_data[-2]

            order =  Order(order_price, quantity, direction, 'limit')
            ob.add_order(order)

        order_type = rng.choice(["limit", "market"], p=[0.5, .5])
        if order_type == "market" and i>200:
            quantity = rng.integers(1, 4)
            ob.add_order(Order(None, quantity, direction, order_type))

        if (i % settings.time_step == 0) and (i > start+200):
            trader4.submit_orders()
            trader3.submit_orders()

        if trader4.max_time==0:
            trader4.counter += 1
            break

    return (trader4, trader3, i)

class settings:
    def __init__(self, H, t, I):
        self.horizon = H
        self.time_step = t
        self.inventory = I
        self.step = H//t

    def __str__(self) -> str:
        return f"settings(horizon={self.horizon}, time_step={self.time_step},steps='{self.step}', inventory='{self.inventory}')"


np.random.seed(123)
runs = 100
sim_steps = 350
window_size = 10
data = pd.read_csv('real_data.csv')


H = [10, 50, 100]
T = [2, 5, 10]
I = [100, 1000]

settings_list = [settings(h, t, i) for h in H for t in T for i in I]

# Create a list of all combinations of H, T, and I
index = pd.MultiIndex.from_product([I, H, T], names=['I', 'H', 'T'])

# Create an empty DataFrame with the specified index
df = pd.DataFrame(np.zeros((len(index), 4)), index=index, columns=['mean', 'std', 'nobs', 't-test'])
t = 0
for sett in settings_list[:]:
    scores1 = []  # to store trader 1 scores from each run
    scores2 = []  # to store trader 2 scores from each run
    for j in tqdm(range(runs)):
        if j == 0:
            trader1, trader2, r = run_simulation(j, sett, sim_steps=sim_steps, QLNN=None, data=data)
        else:
            trader1, trader2, r = run_simulation(j, sett, sim_steps=sim_steps, QLNN=trader1,data=data, r=r)


        scores1.append(trader1.score)  # save trader 1 score from each run
        scores2.append(trader2.score)  # save trader 2 score from each run

    t += 1
    test_obs = (np.array(scores1 )- np.array(scores2))[int(len(scores1)/2):]
    t_test = np.mean(test_obs) / (np.std(test_obs) / np.sqrt(len(test_obs)))

    df.loc[(sett.inventory, sett.horizon, sett.time_step), :] = (np.mean(test_obs), np.std(test_obs),
                                                               len(test_obs), t_test)

