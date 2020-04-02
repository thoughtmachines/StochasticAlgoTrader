import torch
from torch import nn
import random
from torch.autograd import Variable

class SeqRegressor(nn.Module):

    def __init__(self,hidden_size=23,stochastic=False,coin=None,model=None):
        super(SeqRegressor,self).__init__()
        self.stochastic = stochastic
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(23,hidden_size,1)

        self.layer1 = nn.Linear(hidden_size,20)
        self.layer2 = nn.Linear(20,15)
        self.layer3 = nn.Linear(15,7)
        self.layer4 = nn.Linear(7,1)
        self.relu = nn.ReLU(inplace=False)

        hidden_state = torch.ones(1,1, hidden_size)
        cell_state = torch.ones(1,1, hidden_size)
        self.hidden = (hidden_state,cell_state)

        if coin is not None and model is not None:
            name = "weights/mape_"+model+"_"+coin+"_lstm.pth"
            self.load_state_dict(torch.load(name))
        
        self.SM = stochastic_module(_type="lstm",coin=coin,model=model)
        self.sigmoid = nn.Sigmoid()


    def eval(self,x):
        memory = []

        out,_ = self.lstm(x,self.hidden)

        x = self.relu(out[:,-1,:][-1])
        memory.append(x)
        x = self.relu(self.layer1(x))
        memory.append(x)
        x = self.relu(self.layer2(x))
        memory.append(x)
        x = self.relu(self.layer3(x))
        memory.append(x)
        x = self.relu(self.layer4(x))

        self.memory = memory


    def forward(self,x):

        """
        hidden_state = torch.randn(no_stack, batch_size, hidden_dim)
        cell_state = torch.randn(no_stack, batch_size, hidden_dim)
        """
        
        out,_ = self.lstm(x,self.hidden)
        out = out.detach()
        x = out[:,-1,:][-1]
        x = self.relu(x)
        if self.stochastic:
            x= x+ (x- self.memory[0]).detach()* torch.rand(x.shape) * self.sigmoid(self.SM.gamma[0])
            self.memory[0] = x.clone().detach()

        x = self.relu(self.layer1(x))
        if self.stochastic:
            x= x+ (x- self.memory[1]).detach()* torch.rand(x.shape) * self.sigmoid(self.SM.gamma[1])
            self.memory[1] = x.clone().detach()

        x = self.relu(self.layer2(x))
        if self.stochastic:
            x= x+  (x- self.memory[2]).detach()* torch.rand(x.shape) * self.sigmoid(self.SM.gamma[2])
            self.memory[2] = x.clone().detach()

        x = self.relu(self.layer3(x))
        if self.stochastic:
            x= x+ (x- self.memory[3]).detach()* torch.rand(x.shape) * self.sigmoid(self.SM.gamma[3])
            self.memory[3] = x.clone().detach()

        x = self.relu(self.layer4(x))
        return x


class MLPRegressor(nn.Module):

    def __init__(self,stochastic=False,coin=None,model=None):
        super(MLPRegressor,self).__init__()
        self.stochastic = stochastic
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.SM = stochastic_module(_type="mlp",coin=coin,model=model)

        self.layer1 = nn.Linear(161,130)
        self.layer2 = nn.Linear(130,100)
        self.layer3 = nn.Linear(100,50)
        self.layer4 = nn.Linear(50,25)
        self.layer5 = nn.Linear(25,10)
        self.layer6 = nn.Linear(10,1)

        if coin is not None and model is not None:
            name = "weights/_"+model+"_"+coin+"_mlp.pth"
            self.load_state_dict(torch.load(name))

    

    def eval(self,x):
        x = x.flatten().view(1,161)
        memory = []

        x = self.relu(self.layer1(x))
        memory.append(x)

        x = self.relu(self.layer2(x))
        memory.append(x)

        x = self.relu(self.layer3(x))
        memory.append(x)

        x = self.relu(self.layer4(x))
        memory.append(x)

        x = self.relu(self.layer5(x))
        memory.append(x)

        x = self.relu(self.layer6(x))
        memory.append(x)

        self.memory = memory
    
    def forward(self,x):

        x = x.flatten().view(1,161)

        x = self.relu(self.layer1(x))
        if self.stochastic:
            x+= (x- self.memory[0])* torch.rand(x.shape) * self.sigmoid(self.SM.gamma[0])
            self.memory[0] = x.clone().detach()

        x = self.relu(self.layer2(x))
        if self.stochastic:
            x+= (x- self.memory[1])* torch.rand(x.shape)* self.sigmoid(self.SM.gamma[1])
            self.memory[1] = x.clone().detach()

        x = self.relu(self.layer3(x))
        if self.stochastic:
            x+= (x- self.memory[2])* torch.rand(x.shape)* self.sigmoid(self.SM.gamma[2])
            self.memory[2] = x.clone().detach()

        x = self.relu(self.layer4(x))
        if self.stochastic:
            x+= (x- self.memory[3])* torch.rand(x.shape)* self.sigmoid(self.SM.gamma[3])
            self.memory[3] = x.clone().detach()

        x = self.relu(self.layer5(x))
        if self.stochastic:
            x+= (x- self.memory[4])* torch.rand(x.shape)* self.sigmoid(self.SM.gamma[4])
            self.memory[4] = x.clone().detach()

        x = self.relu(self.layer6(x))
        
        return x

class stochastic_module(nn.Module):

    def __init__(self,_type="mlp",coin=None,model=None):
        """type: mlp/lstm"""
        super(stochastic_module,self).__init__()
      
        self.name = "weights/stochastic/"+model+"_"+coin+"_"+_type+".pt"
        tensor = torch.load(self.name)
        self.gamma = Variable(tensor,requires_grad=True)

