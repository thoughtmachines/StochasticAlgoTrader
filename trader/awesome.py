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

    sma5_line = []
    sma34_line = []
    ao_line = []
    
    start_amount = amount
    no_of_coins = 0
    print("\n",coin_name.upper(),":")
    
    for i,(x_input,target) in enumerate(dataloaderX): # TODO: Discuss standard time slot across all algorithms
        if i < 34: # let dataloader catchup with macd range
            continue
        if i == DAYS:
            break
    
        predictedPrice = model(x_input) * dataloaderX.pmax

        sma_5,sma_34,AO,price,(new_sma5_lower,new_sma34_lower) = dataloaderX.getAOData(i)
        sma5_line.append(sma_5)
        sma34_line.append(sma_34)
        ao_line.append(AO)

        predicted_sma5 = (new_sma5_lower + predictedPrice)/2
        predicted_sma34 = (new_sma34_lower + predictedPrice)/2

        # Averaging out AO and predicted AO
        AO = (predicted_sma5 -predicted_sma34 + AO)/2 

        if AO > 0 and no_of_coins == 0:
            no_of_coins = amount/price
            amount = 0
        
        if AO <= 0 and no_of_coins != 0:
            amount = no_of_coins*price
            no_of_coins = 0

    if amount == 0:
        amount = no_of_coins*price

    print("\nStart: ",start_amount," End: ",amount)

    sma5_line = np.asarray(sma5_line)
    sma34_line = np.asarray(sma34_line)
    ao_line = np.asarray(ao_line)

    # Plotting the MACD and Signal Line
    # plt.plot(sma5_line)
    # plt.plot(sma34_line)
    # plt.plot(ao_line)
    # plt.show()




cash = 10000

coins = ['btc','eth','ltc']

for coin in coins:
    macd(coin, cash)