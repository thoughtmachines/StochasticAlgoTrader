import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.init import xavier_normal as xavier
import  matplotlib.pyplot as plt

from data.loader import cryptoData
from models.model import  MLPRegressor

DEVICE = torch.device("cpu")
COIN1 = "btc"
COIN2 = "eth"
MODEL = "norm"

class Regressor(nn.Module):

    def __init__(self):
        super(Regressor,self).__init__()
        self.layer = nn.Linear(1,1)

    def forward(self,x):
        return self.layer(x)

class RegressorTrainer(object):

    def __init__(self,dataloader_coin1,dataloader_coin2):
        self.dataloader_coin1 = dataloader_coin1
        self.dataloader_coin2 = dataloader_coin2
        self.model = Regressor()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.lossfn = nn.MSELoss(reduction='mean')

    def train(self,index):
        iters = 200
        y,yMean,yStd = self.dataloader_coin1.getDataFrame(index)
        x,xMean,xStd = self.dataloader_coin2.getDataFrame(index)
        x = (x - xMean)/xStd
        y = (y - yMean)/yStd
        self.xMean = xMean
        self.yMean = yMean
        self.xStd = xStd
        self.yStd = yStd
        for i in range(iters):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.lossfn(out,y)
            loss.backward()
            self.optimizer.step()
            # if i == 0 or i == iters-1:
            #     print("\tIter: ",i," Loss: ",loss.item())

    def residues(self,value,x):
        return (x-self.xMean)/self.xStd - self.model((value-self.yMean)/self.yStd)


if __name__ == "__main__":
    
    model_coin1 = MLPRegressor(coin=COIN1,model= MODEL)
    model_coin1.to(DEVICE)
    model_coin2 = MLPRegressor(coin=COIN2,model= MODEL)
    model_coin2.to(DEVICE)

    dataloader_coin1 = cryptoData(COIN1,DEVICE=DEVICE,model=MODEL)
    DAYS_coin1 = len(dataloader_coin1)
    dataloader_coin2 = cryptoData(COIN2,DEVICE=DEVICE,model=MODEL)
    DAYS_coin2 = len(dataloader_coin2)

    model_coin1.eval(dataloader_coin1[0][0])
    model_coin2.eval(dataloader_coin2[0][0])

    residualModel = RegressorTrainer(dataloader_coin1,dataloader_coin2)

    counter = 0
    runningAverageHolder = 0
    shorts = longs = holds = 0
    for i in range(200,min(DAYS_coin1,DAYS_coin2)):
        counter+=1

        x_coin1,target_coin1 = dataloader_coin1[i]
        x_coin2,target_coin2 = dataloader_coin2[i]

        out_coin1 = model_coin1(x_coin1) 
        out_coin2 = model_coin2(x_coin2) 
        
        residualModel.train(i)
        residuals = residualModel.residues(out_coin2,out_coin1)
        runningAverageHolder += residuals

        adjustedZScore = (residuals.item()*counter)/runningAverageHolder.item()

        if adjustedZScore > 1.25:
            shorts+=1
        elif adjustedZScore < 0.75:
            longs+=1
        else:
            holds+=1
        print(counter,"\t",adjustedZScore,shorts,longs,holds)

        out_coin1 = out_coin1.item()*dataloader_coin1.pmax.item()
        out_coin2 = out_coin2.item()*dataloader_coin2.pmax.item()

