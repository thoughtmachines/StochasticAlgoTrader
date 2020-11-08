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
COIN1 = "eth"
COIN2 = "btc"
MODEL = "norm"

class Residual(object):

    def __init__(self,dataloader_coin1,dataloader_coin2):
        self.dataloader_coin1 = dataloader_coin1
        self.dataloader_coin2 = dataloader_coin2

    def zScore(self,upperIndex,out_coin1,out_coin2):
        coin1_30 = self.dataloader_coin1.getDataFrame(upperIndex,20)
        coin2_30 = self.dataloader_coin2.getDataFrame(upperIndex,20)
        coin1_30 = torch.cat((coin1_30,out_coin1))
        coin2_30 = torch.cat((coin2_30,out_coin2))
        
        meanDiffernce30 = torch.mean(coin1_30-coin2_30)
        standardDev30 = torch.std(coin1_30-coin2_30)

        coin1_5 = self.dataloader_coin1.getDataFrame(upperIndex,5)
        coin2_5 = self.dataloader_coin2.getDataFrame(upperIndex,5)
        coin1_5 = torch.cat((coin1_5,out_coin1))
        coin2_5 = torch.cat((coin2_5,out_coin2))

        meanDiffernce5 = torch.mean(coin1_5-coin2_5)

        if standardDev30 > 0:
            return (meanDiffernce5 - meanDiffernce30)/standardDev30, self.riskModel(coin1_30,coin2_30)
        else:
            return 0, self.riskModel(coin1_30,coin2_30)

    def riskModel(self,coin1_30,coin2_30):
        c1 = coin1_30 - coin1_30.mean()
        c2 = coin2_30 - coin2_30.mean()

        corr = torch.sum(c1*c2) / (torch.sqrt(torch.sum(c1 ** 2)) * torch.sqrt(torch.sum(c2 ** 2)))
        if corr > 0.9:
            risk = False
        else:
            risk = True
        return risk

def getGeneralTrends(dataloader,upperIndex):
    upper = dataloader.getDataFrame(upperIndex,10).mean()
    lower = dataloader.getDataFrame(upperIndex,30).mean()
    return upper/lower

def main(COIN1,COIN2):
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

    residualModel = Residual(dataloader_coin1,dataloader_coin2)

    coin1_amt = 0
    coin2_amt = 0
    cash = 0

    startDay = 34
    trendThreshold = 1
    shorts = longs = holds = 0
    time = 0
    for i in range(startDay,min(DAYS_coin1,DAYS_coin2)):
        time+=1
        x_coin1,target_coin1 = dataloader_coin1[i]
        x_coin2,target_coin2 = dataloader_coin2[i]
        price_coin1 = dataloader_coin1.getDataFrame(i,1)
        price_coin2 = dataloader_coin2.getDataFrame(i,1)

        if i == startDay:
            coin1_amt = 5000/ price_coin1
            coin2_amt = 5000/ price_coin2

        out_coin1 = model_coin1(x_coin1) 
        out_coin2 = model_coin2(x_coin2) 
        
        zScore, risk = residualModel.zScore(i,out_coin1,out_coin2)
        trend_coin1 = getGeneralTrends(dataloader_coin1,i)
        trend_coin2 = getGeneralTrends(dataloader_coin2,i)

        if not risk:
            if zScore > 1:
                shorts+=1
                if trend_coin2 > trendThreshold:
                    temp = coin1_amt* price_coin1
                    coin1_amt = 0
                    coin2_amt += (temp / price_coin2)
                # print("\t",i,"Transaction: short at ",price_coin1.item(),price_coin2.item())
            elif zScore < -1:
                longs+=1
                if trend_coin1 > trendThreshold:
                    temp = coin2_amt* price_coin2
                    coin2_amt = 0
                    coin1_amt += (temp / price_coin1)
                # print("\t",i,"Transaction: long at ",price_coin1.item(),price_coin2.item())
            else:
                holds+=1


        
        

        out_coin1 = out_coin1.item()*dataloader_coin1.pmax.item()
        out_coin2 = out_coin2.item()*dataloader_coin2.pmax.item()
    print(COIN1,COIN2,"\n\t",(coin1_amt * price_coin1) + (coin2_amt * price_coin2) + cash)
    print(time,'\n')
    
if __name__ == "__main__":
    
    main("eth","btc")    
    main("eth","ltc")    
    main("ltc","btc")