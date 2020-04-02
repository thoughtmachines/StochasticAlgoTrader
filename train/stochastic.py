import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  MLPRegressor, SeqRegressor

DEVICE = torch.device("cpu")
MODE = "train"
TYPE = "lstm"
COIN = "btc"
MODEL = "norm"
STOCHASTIC = True

if __name__ == "__main__":

    if MODE == "train":
        test = False
        
    else:
        test = True

    if TYPE == "mlp":
        model = MLPRegressor(coin=COIN,model=MODEL,stochastic=STOCHASTIC)
    else:
        model = SeqRegressor(stochastic=STOCHASTIC,model=MODEL,coin=COIN)
    model.to(DEVICE)
    
    dataloader = cryptoData(COIN,test=test,DEVICE=DEVICE,model=MODEL)
    breaker = len(dataloader)

    if TYPE == "lstm":
        model.eval(dataloader[0][0].unsqueeze(1))
    else:
        model.eval(dataloader[0][0])

    optimizer = Adam([model.SM.gamma], lr=0.001, weight_decay=0.000001)
    lossfn = nn.MSELoss(reduction='mean')

    
    for j in range(60):

        if TYPE == "lstm":
            hidden_state = torch.ones(1,1, 23).to(DEVICE)
            cell_state = torch.ones(1,1, 23).to(DEVICE)
            model.lstm.hidden = (hidden_state,cell_state)

        z = 0 
        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break 
            
            if TYPE == "lstm":
                x.unsqueeze_(1)

            target = target.view(1,1)
            optimizer.zero_grad()
            
            out = model(x)
            
            loss = lossfn(out,target)
            
            loss.backward(retain_graph=True)

            optimizer.step()

            t = target.item()
            h = out.item()
            z+= abs((t - h)/t)
            
        torch.save(model.SM.gamma,model.SM.name)
        print("Epoch: ",j,z*100/breaker,model.SM.gamma)