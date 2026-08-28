import torch
import torch.nn as nn
import torch.nn.functional as F


class firstNN(nn.Module):
    """First NN I made, predicts 2 outputs,
        i.e. it used to predict u/v.\n
        It is a MLP with Relu, n_inputs->128->64
        linear 64->2"""
    def __init__(self, n_inputs):
        """First NN I made, predicts 2 outputs,
        i.e. it used to predict u/v.\n
        It is a MLP with Relu, n_inputs->128->64
        linear 64->2"""
        super().__init__()
        self.fc1=nn.Linear(n_inputs,128)
        self.fc2=nn.Linear(128,64)
        # self.fc3=nn.Linear(64,32)
        # self.fc4=nn.Linear(32,16)
        # self.fc5=nn.Linear(16,8)
        # self.fc6=nn.Linear(8,4)
        self.final=nn.Linear(64,2)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        # x=F.sigmoid(self.fc3(x))
        # x=F.sigmoid(self.fc4(x))
        # x=F.sigmoid(self.fc5(x))
        # x=F.sigmoid(self.fc6(x))
        x=self.final(x)
        return x

class uv256NN(nn.Module):
    """To calculate u/v\n
        MLP: 256->128->64 with ReLu\n
        linear 64->2"""
    def __init__(self, n_inputs):
        """To calculate u/v\n
        MLP: 256->128->64 with ReLu\n
        linear 64->2"""
        super().__init__()
        self.fc1=nn.Linear(n_inputs,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        # self.fc4=nn.Linear(32,16)
        # self.fc5=nn.Linear(16,8)
        # self.fc6=nn.Linear(8,4)
        self.final=nn.Linear(64,2)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        # x=F.sigmoid(self.fc4(x))
        # x=F.sigmoid(self.fc5(x))
        # x=F.sigmoid(self.fc6(x))
        x=self.final(x)
        return x

class staticNN(nn.Module):
    """To calculate Staticity\n 
        MLP with Relu, n_inputs->256->128->64
        linear 64->1"""
    def __init__(self, n_inputs):
        """To calculate Staticity\n 
        MLP with Relu, n_inputs->256->128->64
        linear 64->1"""
        super().__init__()
        self.fc1=nn.Linear(n_inputs,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        # self.fc4=nn.Linear(32,16)
        # self.fc5=nn.Linear(16,8)
        # self.fc6=nn.Linear(8,4)
        self.final=nn.Linear(64,1)
        #self.fourier=nn.Linear(n_inputs,n_inputs)

    def forward(self,x):
        #x=torch.cos(self.fourier(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        # x=F.sigmoid(self.fc4(x))
        # x=F.sigmoid(self.fc5(x))
        # x=F.sigmoid(self.fc6(x))
        x=self.final(x)
        return x

class NNforNorms(nn.Module):
    """THE FINAL LAYER IS NOT LINEAR FOR THE LOG1Ped Norms
        MLP with Relu, 256->128->64\n
        SOFTPLUS 64->2"""
    def __init__(self, n_inputs):
        """THE FINAL LAYER IS NOT LINEAR FOR THE LOG1Ped Norms
        MLP with Relu, 256->128->64\n
        SOFTPLUS 64->2"""
        super().__init__()
        self.fc1=nn.Linear(n_inputs,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        # self.fc4=nn.Linear(32,16)
        # self.fc5=nn.Linear(16,8)
        # self.fc6=nn.Linear(8,4)
        self.final=nn.Linear(64,1)
        #self.fourier=nn.Linear(n_inputs,n_inputs)

    def forward(self,x):
        #x=torch.cos(self.fourier(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        # x=F.sigmoid(self.fc4(x))
        # x=F.sigmoid(self.fc5(x))
        # x=F.sigmoid(self.fc6(x))
        x=self.final(x)
        return F.softplus(x)
