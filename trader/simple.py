import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
import  matplotlib.pyplot as plt

from data.trader import hourly
from models.model import SeqRegressor, MLPRegressor

DEVICE = torch.device("cpu")
COIN = "eth"
MODEL = "norm"

if __name__ == "__main__":
    

    model = MLPRegressor(coin=COIN,model= MODEL)
    model.to(DEVICE)

    dataloader = hourly(COIN,DEVICE=DEVICE,model=MODEL)
    DAYS = len(dataloader)
    model.eval(dataloader[0][0])

    cash = 1000
    coin = 0
    transactions = 0
    state = "CASH"
    for i,(x,hourly,target) in enumerate(dataloader):
        if i == DAYS:
            break

        out = model(x)
        out = out.item()*dataloader.pmax.item()
        # hourly = 24 x Open,High,Low,Close,Volume
                
        for hour in range(24):
            _open = hourly[hour][0].item()
            _close = hourly[hour][3].item()
            
            if state == "CASH" :
                if  out > _close :
                    coin = cash/_close
                    cash = 0
                    state = "CRYPTO"
                    transactions+=1
                    buy = _close
            if state == "CRYPTO" :
                if  out < _close:
                    cash = _close*coin
                    coin = 0
                    state = "CASH"
                    transactions+=1

    print("Cash:",cash,"\nCoin:",coin*_close,"\nTransactions:",transactions,"\nPrice:",_close)

    
