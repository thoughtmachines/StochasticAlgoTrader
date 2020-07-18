import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import  matplotlib.pyplot as plt
import numpy as np

from data.loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cpu")
MODEL = "norm"

# Function for MACD Strategy

def macd(coin_name, amount):

    model = MLPRegressor(coin=coin_name.lower(),model= MODEL)
    model.to(DEVICE)

    dataloaderX = cryptoData(coin_name.lower(),DEVICE=DEVICE)
    DAYS = len(dataloaderX)
    model.eval(dataloaderX[33][0])

    macd_line = []
    signal_line = []
    x = []
    
    start_amount = amount
    no_of_coins = 0
    print("\n",coin_name.upper(),":")
    
    for i,(x_input,target) in enumerate(dataloaderX): # TODO: Discuss standard time slot across all algorithms
        if i < 34: # let dataloader catchup with macd range
            continue
        if i == DAYS:
            break
    
        predictedPrice = model(x_input) * dataloaderX.pmax

        (macd, macd_9, ema_12, ema_26, price) = dataloaderX.getMacDData(i)

        ema_12 = ema_12 + (predictedPrice - ema_12)*dataloaderX.multiplier_12
        ema_26 = ema_26 + (predictedPrice - ema_26)*dataloaderX.multiplier_26
        macd = ema_12 - ema_26
        macd_9 = macd_9 + (macd - macd_9)*dataloaderX.multiplier_9


        x.append(i)
        macd_line.append(macd)
        signal_line.append(macd_9)

        # If MACD crosses over Signal Line then bullish market
        if macd-macd_9 > 0 and no_of_coins == 0:
            no_of_coins = amount/price
            amount = 0
        
        # If MACD crosses below Signal Line then bearish market
        if macd-macd_9 <= 0 and no_of_coins != 0:
            amount = no_of_coins*price
            no_of_coins = 0

    if amount == 0:
        amount = no_of_coins*price

    print("\nStart: ",start_amount," End: ",amount)

    macd_line = np.asarray(macd_line)
    signal_line = np.asarray(signal_line)
    x = np.asarray(x)

    # Plotting the MACD and Signal Line
    # plt.plot(x,macd_line,'b',x,signal_line,'r')
    # plt.show()




cash = 10000

coins = ['btc','eth','ltc']

for coin in coins:
    macd(coin, cash)