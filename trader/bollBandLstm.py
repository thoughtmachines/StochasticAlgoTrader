import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import  matplotlib.pyplot as plt
import numpy as np
from data.loader import cryptoData
from models.model import  SeqRegressor

DEVICE = torch.device("cpu")
MODEL = "norm"

def boll_band(coin, amount):
    
    model = SeqRegressor(coin=coin,model= MODEL)
    model.to(DEVICE)

    print("\n",coin.upper(),":")
    dataloader = cryptoData(coin, DEVICE=DEVICE)
    model.eval(dataloader[28][0].unsqueeze(1))
    start = amount
    cash = amount
    no_of_coins = 0

    mean = []
    ub = []
    lb = []
    x = []

    for i,(x_input,target) in enumerate(dataloader): # TODO: Discuss standard time slot across all algorithms
        # Wait for the Moving Average
        if(i < 34):
            continue
        (sma_20,sma_30,sma_5,ubb,lbb,price) = dataloader.getBollBandData(i)

        trend = sma_5-sma_30
        x.append(i)
        ub.append(ubb)
        lb.append(lbb)
        mean.append(sma_20)

        predictedPrice = model(x_input.unsqueeze(1)) * dataloader.pmax
        predictedPrice = predictedPrice.squeeze()

        if trend > 0:
            # If Upward Trend and price touches Upper Bollinger Band then Buy
            if predictedPrice-ubb >= 0 and no_of_coins == 0:
                # print("Buy\n")
                no_of_coins = cash/price
                cash = 0
        
        else:
            # If Downward Trend and price touches Lower Bollinger Band then Sell
            if predictedPrice-lbb <= 0 and no_of_coins != 0:
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
