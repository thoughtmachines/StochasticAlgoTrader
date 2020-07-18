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
        
        self.price = np.asarray(data[var])
    
    def __getitem__(self, key):

        if key < 29:
            return 0,0,0,0,0,0

        self.sma_20 = 0
        for i in range(key-19,key+1):
            self.sma_20 += self.price[i]

        self.sma_20 = self.sma_20/20

        self.sma_30 = 0
        for i in range(key-29,key+1):
            self.sma_30 += self.price[i]

        self.sma_30 = self.sma_30/30

        self.sma_5 = 0
        for i in range(key-4,key+1):
            self.sma_5 += self.price[i]
        
        self.sma_5 = self.sma_5/5

        self.range = self.price[key-19:key+1]
        self.upper_boll_band = self.sma_20 + 2*abs(np.std(self.range))
        self.lower_boll_band = self.sma_20 - 2*abs(np.std(self.range))

        return self.sma_20, self.sma_30, self.sma_5, self.upper_boll_band, self.lower_boll_band, self.price[key]