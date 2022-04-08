import sys, os
from numpy import False_
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from data.loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cpu")

def rsi(coin, amount, lower_threshold, n):
    
    upper_threshold = 100 - lower_threshold
    print("\n",coin.upper(),":")
    dataloader = cryptoData(coin, DEVICE=DEVICE)
    start = amount
    cash = amount
    no_of_coins = 0


    buyAndHold = 10000
    time = 0
    for i,(x_input,target) in enumerate(dataloader): # TODO: Discuss standard time slot across all algorithms
        # Wait for the Moving Average
        if(i < n + 1):
            continue
        time+=1
        rsi_val, price = dataloader.getRSIData(i, n)
        if rsi_val > upper_threshold and no_of_coins > 0:
            ## sell
            cash += no_of_coins * price
            no_of_coins = 0

        if rsi_val < lower_threshold and cash > 0:
            ## buy
            no_of_coins = cash/price
            cash = 0
        # print(rsi_val, cash, no_of_coins)

    cash += no_of_coins * price

    print("\nStart: ", start, " End: ", cash)
    print(buyAndHold * target, time)
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
    rsi(coin,cash, 35, 14)
