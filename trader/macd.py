import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import  matplotlib.pyplot as plt
import numpy as np
from data.data_loader import daily_data

DEVICE = torch.device("cpu")

# Function for MACD Strategy

def macd(coin_name, amount):
    dataloader = daily_data(coin_name.lower(), DEVICE=DEVICE)
    
    macd_line = []
    signal_line = []
    x = []
    
    start_amount = amount
    no_of_coins = 0
    print("\n",coin_name.upper(),":")
    
    for i,(macd, macd_9, ema_12, ema_26, price) in enumerate(dataloader):

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