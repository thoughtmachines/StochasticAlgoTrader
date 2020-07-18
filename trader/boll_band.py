import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import  matplotlib.pyplot as plt
import numpy as np
from data.bb_data_loader import daily_data

DEVICE = torch.device("cpu")


def boll_band(coin, amount):

    print("\n",coin.upper(),":")
    dataloader = daily_data(coin, DEVICE=DEVICE)
    start = amount
    cash = amount
    no_of_coins = 0

    mean = []
    ub = []
    lb = []
    x = []

    for i,(sma_20,sma_30,sma_5,ubb,lbb,price) in enumerate(dataloader):
        # Wait for the Moving Average
        if(i < 29):
            continue
        

        trend = sma_5-sma_30
        x.append(i)
        ub.append(ubb)
        lb.append(lbb)
        mean.append(sma_20)

        if trend > 0:
            # If Upward Trend and price touches Upper Bollinger Band then Buy
            if price-ubb >= 0 and no_of_coins == 0:
                # print("Buy\n")
                no_of_coins = cash/price
                cash = 0
        
        else:
            # If Downward Trend and price touches Lower Bollinger Band then Sell
            if price-lbb <= 0 and no_of_coins != 0:
                # print("Sell\n")
                cash = no_of_coins*price
                no_of_coins = 0

    if no_of_coins != 0:
        cash = no_of_coins*price
        no_of_coins = 0

    print("\nStart: ", start, " End: ", cash)

    # Plotting the Moving Average and Bollinger Bands
    # x = np.asarray(x)
    # ub = np.asarray(ub)
    # lb = np.asarray(lb)
    # mean = np.asarray(mean)

    # plt.plot(x,ub,'g',x,mean,'b',x,lb,'r')
    # plt.show()

coins = ['btc','eth','ltc']
cash = 10000

for coin in coins:
    boll_band(coin,cash)
