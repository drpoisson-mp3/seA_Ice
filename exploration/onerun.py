import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 

from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F


print(os.listdir('../data'))

trainpd=pd.read_csv('../data/DRIFT_DATA_TRAIN.csv')

bs = 64 # batch size

trainpd=trainpd.drop(['year','month','day','x_EASE','y_EASE','id_buoy'],axis=1) #unwanted quantities



# NORMALIZATION STEP
#for day of year, simply divide by 364 to get smth between 0 and 1 (order 1)
trainpd['doy']=trainpd['doy']/364

# for the speeds, we substract the mean and divide by the standard deviation to get some cleaner distribution
speeds=['u_buoy', 'v_buoy', 'u_ERA5', 'v_ERA5']

means={}
stds={}
for label in speeds:
    means[label]=trainpd[label].mean()
    stds[label]=trainpd[label].std()
    trainpd[label]=(trainpd[label]-trainpd[label].mean())/trainpd[label].std()


print(means)
print(stds)
# SIC is already [0,1] and sea ice thickness is typically between 0 and idk 5 nothing is needed



target = pd.DataFrame([trainpd['u_buoy'],trainpd['v_buoy']]).T# had to reshape from u,v x rows to rows x u,v

trainpd = trainpd.drop(['u_buoy', 'v_buoy'],axis=1) # Take away the time of the year, wrong dtype and seems to be useless anyway
labels = [target.columns, trainpd.columns]
# we keep the day of year tho

traintensor = torch.tensor(trainpd.values, dtype= torch.float32)
targetensor = torch.tensor(target.values, dtype= torch.float32) 
print(target.shape)
print(traintensor.shape)
baseline_mse = torch.mean(targetensor**2).item()  # predicting all-zeros
print("Baseline MSE (predict mean):", baseline_mse)
trainset=TensorDataset(targetensor,traintensor)
trainloader=DataLoader(trainset,batch_size=bs,shuffle=True)

# creating my neural network

class firstNN(nn.Module):
    """Hummmm"""
    def __init__(self, n_inputs):
        """GANG"""
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
device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')

print(device)

babygpu= firstNN(n_inputs=traintensor.shape[1]).to(device)
print(babygpu.parameters)

# ALL RIGHT LETS TRY TO TRAIN THAT THING
import torch.optim as optim

# Hyperparameters
lr=0.001
epoch=20


optimizer = optim.Adam(babygpu.parameters(),lr=lr) # optimizers
criterion = nn.MSELoss()

def RMSE(target,outputs):
    '''Calculates the RMSE between, the target and the model outputs'''
    return (torch.mean((target-outputs)**2))**.5

rlosses=[]
losses=[]
rloss=0
rcount=0


for epoch in range(epoch):
    print('Epoch: ',epoch)
    curpercent=0
    for i,data in enumerate(trainloader):

        # get the inputs and the target
        truth,inputs= data[0].to(device),data[1].to(device)

        # zero the gradients,
        optimizer.zero_grad()

        # Forward,
        out=babygpu(inputs)


        loss = criterion(out,truth)
        # Backward
        loss.backward()
        losses.append(loss.item())
        # Optimize

        optimizer.step()

        rloss+= loss.item()
        rcount +=1
        #print where we at plus the loss
        percent= round(i/len(trainloader)*100)
        if percent != curpercent:
            print(percent, 'loss: ', rloss/rcount)
            curpercent=percent
            rlosses.append(rloss/rcount)
            rloss=0
            rcount=0

plt.plot(rlosses,'.')

plt.show()