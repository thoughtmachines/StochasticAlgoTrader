import pandas as pd
import numpy as np
import torch

from data.loader import cryptoData


class hourly(cryptoData):

    def __init__(self,currency,test = False,DEVICE = torch.device("cpu"),window=7,model=None):
        super(hourly,self).__init__(currency,test,DEVICE,window,model)

        self.hourly_raw = pd.read_csv("data/data/Coinbase_"+currency+"USD_1h.csv")

        