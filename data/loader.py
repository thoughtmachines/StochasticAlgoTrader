import pandas as pd
import numpy as np
import torch
from copy import deepcopy


class cryptoData(object):

    def __init__(self,currency,test = False,DEVICE = torch.device("cpu"),window=7,model=None):
        data = pd.read_csv("data/data/"+currency+"Final.csv")
        window = 7
        data = data.drop(["Unnamed: 0",
                        "Unnamed: 0_x",
                        "timestamp",
                        "datetime",
                        "index",
                        "top100cap-"+currency,
                        "mediantransactionvalue-"+currency,
                        "Unnamed: 0_y"
                        ], axis=1)
        self.test = test

        if currency == "btc":
            var = "bitcoin-price"
        elif currency == "eth":
            var = "ethereum-price"
        else:
            var = "litecoin-price"

        data = data.iloc[:851]
        train_data = data.iloc[:650]
        test_data = data.iloc[650:]
        self.raw = data

        price = np.asarray(data[var])
        mean = train_data.mean()
        self.raw_prices = deepcopy(price)

        train_data = train_data/mean
        test_data = test_data/mean
        self.raw = self.raw/mean
        
        train_data = torch.Tensor(train_data.to_numpy())
        test_data = torch.Tensor(test_data.to_numpy())
        self.raw = torch.Tensor(self.raw.to_numpy())

        train_data[:,5] = torch.Tensor(price[:650])
        test_data[:,5] = torch.Tensor(price[650:])
        self.raw[:,5] = torch.Tensor(self.raw_prices[:])
        self.raw_prices = torch.Tensor(self.raw_prices)
        self.emaPreparation(deepcopy(self.raw_prices))

        xtrain = []
        ytrain = []
        self.window = window
        for i in range(len(train_data)-window):
            xtrain.append(np.asarray(train_data[i:i+window,:]))
            ytrain.append(np.asarray(train_data[i+window][5]))

        self.xtrain = torch.Tensor(np.asarray(xtrain)).to(DEVICE)
        self.ytrain = torch.Tensor(np.asarray(ytrain)).to(DEVICE)

        if model == None or model== "norm":
            self.pmax = torch.mean(self.ytrain)
        elif model == "unorm":
            self.pmax = torch.Tensor([1])

        xtest = []
        ytest = []
        for i in range(len(test_data)-window):
            xtest.append(np.asarray(test_data[i:i+window,:]))
            ytest.append(np.asarray(test_data[i+window][5]))

        self.xtest = torch.Tensor(np.asarray(xtest)).to(DEVICE)
        self.ytest = torch.Tensor(np.asarray(ytest)).to(DEVICE)

        self.xtrain[:,:,5] /= self.pmax
        self.ytrain = self.ytrain/self.pmax
        self.xtest[:,:,5] = self.xtest[:,:,5]/self.pmax
        self.ytest = self.ytest/self.pmax
        self.raw /= self.pmax

    def emaPreparation(self,price):
        sma_12 = 0
        sma_26 = 0

        for i in range(12):
            sma_12 += price[i]

        sma_12 = sma_12/12

        for i in range(26):
            sma_26 += price[i]

        sma_26 = sma_26/26        

        self.ema_12 = sma_12
        self.ema_26 = sma_26
        self.multiplier_12 = 2/13
        self.multiplier_26 = 2/27
        self.multiplier_9 = 2/10

        for i in range(12,25):
            self.ema_12 = self.ema_12 + (price[i] - self.ema_12)*self.multiplier_12

        self.macd = 0
        self.macd_9 = 0

        for i in range(25,34):
            self.ema_12 = self.ema_12 + (price[i] - self.ema_12)*self.multiplier_12
            if i == 25:
                self.ema_26 = sma_26
            else:
                 self.ema_26 = self.ema_26 + (price[i] - self.ema_26)*self.multiplier_26

            self.macd = self.ema_12 - self.ema_26
            self.macd_9 += self.macd
        
        self.macd_9 = self.macd_9/9

        self.final_price = price[34:]

    # 11 - sma_12
    # 25 - sma_26/macd
    # 33 - macd_9
    # Start returning from 34 day (macd and macd_9)

    def getMacDData(self, key):
        self.ema_12 = self.ema_12 + (self.final_price[key] - self.ema_12)*self.multiplier_12
        self.ema_26 = self.ema_26 + (self.final_price[key] - self.ema_26)*self.multiplier_26
        self.macd = self.ema_12 - self.ema_26
        self.macd_9 = self.macd_9 + (self.macd - self.macd_9)*self.multiplier_9

        return self.macd, self.macd_9, self.ema_12, self.ema_26, self.final_price[key]

    def getAOData(self,key):
        
        sma_5 = (self.raw_prices[key-5] + self.raw_prices[key])/2
        sma_34 = (self.raw_prices[key-34] + self.raw_prices[key])/2
        AO= sma_5 - sma_34
        return sma_5,sma_34,AO, self.raw_prices[key], (self.raw_prices[key-4],self.raw_prices[key-33])

    def getRSIData(self, key, n):

        diff =  self.raw_prices[key-n:key] - self.raw_prices[key-n-1:key-1]
        up = diff[diff > 0]
        down = (- diff[diff < 0])
        return (100 - (100/(1 + (up.mean()/down.mean())))).item(), self.final_price[key]
        

    def getBollBandData(self,key):
        if key < 29:
            return 0,0,0,0,0,0

        sma_20 = self.raw_prices[key-19:key+1].sum()/20
        sma_30 = self.raw_prices[key-29:key+1].sum()/30
        sma_5 = self.raw_prices[key-4:key+1].sum()/5

        dataRange = self.raw_prices[key-19:key+1]
        upper_boll_band = sma_20 + 2*abs(torch.std(dataRange))
        lower_boll_band = sma_20 - 2*abs(torch.std(dataRange))

        return sma_20, sma_30, sma_5, upper_boll_band, lower_boll_band, self.raw_prices[key]
    
    def __getitem__(self,key):
        if not self.test:
            seven_day_data = self.xtrain[key]
            target = self.ytrain[key].view(1,1)
        else:
            seven_day_data = self.xtest[key]
            target  = self.ytest[key].view(1,1)
        return seven_day_data, target

    
    def __len__(self):
        if self.test:
            return 230-self.window
        return 650-self.window

    def getDataFrame(self,maxRange,period):
        prices = self.raw_prices[maxRange-period:maxRange]
        return prices.view(-1,1)