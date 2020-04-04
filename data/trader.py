import pandas as pd
import numpy as np
import torch

class hourly(object):

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

        price = np.asarray(data[var])
        mean = train_data.mean()

        data = data/mean
        
        data = torch.Tensor(data.to_numpy())

        data[:,5] = torch.Tensor(price[:])

        x = []
        y = []
        self.window = window
        for i in range(len(data)-window):
            x.append(np.asarray(data[i:i+window,:]))
            y.append(np.asarray(data[i+window][5]))

        self.x = torch.Tensor(np.asarray(x)).to(DEVICE)
        self.y = torch.Tensor(np.asarray(y)).to(DEVICE)

        if model == None or model== "norm":
            self.pmax = torch.mean(self.y)
        else:
            self.pmax = torch.Tensor([1])


        self.x[:,:,5] /= self.pmax
        self.y = self.y/self.pmax


        raw = pd.read_csv("data/data/Coinbase_"+currency+"USD_1h.csv")
        raw = raw.drop(["Date"],axis=1).to_numpy()

        hourly = []
        for i in range(844):
            hourly.append(raw[i*24:i*24+24])

        self.hourly = torch.Tensor(np.asarray(hourly)).to(DEVICE)

    def __len__(self):
        return 844

    def __getitem__(self,key):
        seven_day_data = self.x[key]
        target = self.y[key].view(1,1)
        hourly = self.hourly[key]
        return seven_day_data,hourly,target


        