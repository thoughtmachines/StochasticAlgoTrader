# train

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.unorm_loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cuda:0")
MODE = "train"
# MODE = "test"

if __name__ == "__main__":

    if MODE == "train":
        test = False
        
    else:
        test = True

    model = MLPRegressor()
    # model.load_state_dict(torch.load("weights/mlpreg_final.pth"))
    
    optimizer = Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    lossfn = nn.MSELoss(reduction='mean')

    dataloader = cryptoData("btc",test=test,DEVICE=DEVICE)

    model.to(DEVICE)

    breaker = len(dataloader)
    
    for j in range(700):
        tots = 0
        for i,(x,target) in enumerate(dataloader):
            if i == breaker:
                break

            optimizer.zero_grad()

            out = model(x)
            
            loss = lossfn(out,target)
            tots += loss.item()
            loss.backward()

            optimizer.step()
        print("Epoch: " + str(j) + " " + str(tots/693)+"\t" + str(out.item()) + "\t" + str(target.item()))
        torch.save(model.state_dict(),"weights/unorm_btc_mlp.pth")