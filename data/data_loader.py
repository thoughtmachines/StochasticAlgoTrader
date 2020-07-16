import pandas as pd
import numpy as np
import torch


class daily_data(object):

    def __init__(self,currency,DEVICE = torch.device("cpu")):
        data = pd.read_csv("data/data/"+currency+"Final.csv")
        data = data.drop(["Unnamed: 0",
                        "Unnamed: 0_x",
                        "timestamp",
                        "datetime",
                        "index",
                        "top100cap-"+currency,
                        "mediantransactionvalue-"+currency,
                        "Unnamed: 0_y"
                        ], axis=1)

        if currency == "btc":
            var = "bitcoin-price"
        elif currency == "eth":
            var = "ethereum-price"
        else:
            var = "litecoin-price"

        data = data.iloc[:851]
        
        price = np.asarray(data[var])

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
    
    def __getitem__(self, key):
        self.ema_12 = self.ema_12 + (self.final_price[key] - self.ema_12)*self.multiplier_12
        self.ema_26 = self.ema_26 + (self.final_price[key] - self.ema_26)*self.multiplier_26
        self.macd = self.ema_12 - self.ema_26
        self.macd_9 = self.macd_9 + (self.macd - self.macd_9)*self.multiplier_9

        return self.macd, self.macd_9, self.ema_12, self.ema_26, self.final_price[key]