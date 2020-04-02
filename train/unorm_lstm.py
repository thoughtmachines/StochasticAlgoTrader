import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.unorm_loader import cryptoData
from models.model import  SeqRegressor

DEVICE = torch.device("cpu")
MODE = "train"

if __name__ == "__main__":

    model = SeqRegressor()

    optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    lossfn = nn.MSELoss(reduction='mean')

    if MODE == "train":
        test = False
    else:
        test = True

    
    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE,window=7)
    model.to(DEVICE)
    breaker = len(dataloader)

    
    for j in range(700):
        tots = 0
        hidden_state = torch.ones(1,1, 23).to(DEVICE)
        cell_state = torch.ones(1,1, 23).to(DEVICE)
        model.lstm.hidden = (hidden_state,cell_state)

        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break 
            
            x.unsqueeze_(1)
             
            target = target.view(1,1)
            optimizer.zero_grad()
            
            out = model(x)
            
            loss = lossfn(out,target)
            tots+=loss.item()
            loss.backward()

            optimizer.step()
        torch.save(model.state_dict(),"weights/unorm_btc_lstm.pth")
        print("Epoch: " + str(j) + " " + str(tots/693)+"\t" + str(out.item()) + "\t" + str(target.item()))