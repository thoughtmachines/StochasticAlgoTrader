import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  SeqRegressor

DEVICE = torch.device("cpu")
MODE = "test"
MODE = "test"
COIN = "btc"
MODEL = "unorm"
STOCHASTIC = False

if __name__ == "__main__":

    model = SeqRegressor(stochastic=STOCHASTIC,model=MODEL,coin=COIN)

    if MODE == "train":
        test = False

    else:
        test = True

    
    dataloader = cryptoData(COIN,test=test,DEVICE=DEVICE,model=MODEL)
    model.to(DEVICE)

    model.eval(dataloader[0][0].unsqueeze(1))
    breaker = len(dataloader)
    t,h = [],[]
    z = 0 
    m = 0
    r=0
    for i,(x,target) in enumerate(dataloader):
        if i == breaker:
            break

        x.unsqueeze_(1)


        out = model(x)
        out = out.squeeze()

        t.append(target.item()*dataloader.pmax.item())
        h.append(out.item()*dataloader.pmax.item())
        # print(out.item()*dataloader.pmax.item())
        z+= abs((t[-1] - h[-1])/t[-1])
        m+= abs((t[-1] - h[-1]))**2
        r+= abs((t[-1] - h[-1])/t[-1])**2

    plt.plot(t)
    plt.plot(h)
    plt.show()

    print("MAPE",z/breaker * 100)
    print("MAE",z/breaker)
    print("RMSE",(r/breaker)**0.5)
    print("MSE",m/breaker)
    
    # print(z/breaker * 100)
    # print(z/breaker)
    # print((r/breaker)**0.5)
    # print(m/breaker)

        
