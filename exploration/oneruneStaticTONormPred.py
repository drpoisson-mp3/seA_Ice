import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 

from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
# for checking time 
from datetime import datetime 
with open('timestamps.txt', 'a') as f:
    f.write(f"{datetime.now()}\n")


#########################################################################################################################################33

trainpd=pd.read_csv('../data/DRIFT_DATA_TRAIN.csv')

bs = 2**13 # batch size

trainpd=trainpd.drop(['year','month','day','id_buoy'],axis=1) #unwanted quantities

#ADDING THE COS AND SIN OF THE YEAR
trainpd['sin']=np.sin(trainpd['doy']*(2*np.pi/364))
trainpd['cos']=np.cos(trainpd['doy']*(2*np.pi/364))

#dropping the day of year
trainpd=trainpd.drop(['doy'],axis =1 )

# adding the norms

trainpd['windnorm']=np.sqrt(trainpd['u_ERA5']**2+trainpd['v_ERA5']**2)
trainpd['buoynorm']=np.sqrt(trainpd['u_buoy']**2+trainpd['v_buoy']**2)


# plt.hist(trainpd['windnorm'])
# plt.savefig('testhistwindnorm.png')
# plt.close('all')

# Adding the bathymetry
bath= pd.read_csv('../data/bathymetry_EASE.csv').values
trainpd['bath']= bath[trainpd['y_EASE'].astype(int),trainpd['x_EASE'].astype(int)]

# NORMALIZATION STEP
# For the norms, I log1p

trainpd['buoynorm']=np.log1p(trainpd['buoynorm'])
# trainpd['windnorm']=np.log1p(trainpd['windnorm'])

# for the speeds, we substract the mean and divide by the standard deviation to get some cleaner distribution
speeds=['u_buoy', 'v_buoy', 'u_ERA5', 'v_ERA5','windnorm']

means={}
stds={}
for label in speeds:
    means[label]=trainpd[label].mean()
    stds[label]=trainpd[label].std()
    trainpd[label]=(trainpd[label]-trainpd[label].mean())/trainpd[label].std()


# print(means)
# print(stds)
# SIC is already [0,1] and sea ice thickness is typically between 0 and idk 5 nothing is needed

# For the x,y, we simply divide by the max?
max_x=np.max(np.abs(trainpd['x_EASE']))
trainpd['x_EASE']=trainpd['x_EASE']/max_x

max_y =np.max(np.abs(trainpd['y_EASE']))
trainpd['y_EASE']=trainpd['y_EASE']/max_y

#For the Bathymetry we divide by the max as well (is actually a min)

max_bathy=np.max(np.abs(trainpd['bath']))
trainpd['bath']=trainpd['bath']/max_bathy


staticpd = trainpd.drop(['u_buoy', 'v_buoy','buoynorm'],axis=1) # Take away the time of the year, wrong dtype and seems to be useless anyway


class firstNN(nn.Module):
    """Hummmm"""
    def __init__(self, n_inputs):
        """MLP with Relu, 256->128->64
        linear 64->2"""
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


# Bring it to GPU
device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')


staticgpu= firstNN(n_inputs=staticpd.shape[1]).to(device)

# print(staticgpu)
staticgpu.load_state_dict(torch.load('.//weights//babygpu_static_10input_999E_weight147.pt', weights_only=True))
staticgpu.eval()  # if you're doing inference, not continuing training

staticity= torch.sigmoid(staticgpu(torch.tensor(staticpd.values, dtype= torch.float32).to(device)).to('cpu').detach()).numpy()


# print(staticity)

trainpd['staticity']=staticity # Adding the staticity to the dataframe

mask = trainpd['staticity'] >= 0.950 # mask for 95% sure of staticity
staticbuoy = trainpd[mask].copy() # the static cases
trainpd = trainpd[~mask].copy() # the moving case

print(len(staticbuoy),len(trainpd))

# plt.hist(trainpd['buoynorm'],100)
# plt.show()

# plt.hist(staticbuoy['buoynorm'],100)
# plt.show()

target = pd.DataFrame(trainpd['buoynorm'])# redfine target with trainpd with b
trainpd = trainpd.drop(['u_buoy', 'v_buoy','buoynorm'],axis=1)# dropping for velpredictions

labels = [target.columns, trainpd.columns]
print('Labels: ', labels)


traintensor = torch.tensor(trainpd.values, dtype= torch.float32)
targetensor = torch.tensor(target.values, dtype= torch.float32) 
print(trainpd)

baseline_mse = torch.mean(targetensor**2).item()  # predicting all-zeros
print("Baseline MSE (predict mean):", baseline_mse)

# Building the dataset and loader
trainset=TensorDataset(targetensor,traintensor)
trainloader=DataLoader(trainset,batch_size=bs,shuffle=True)


class NNforNorms(nn.Module):
    """Hummmm"""
    def __init__(self, n_inputs):
        """MLP with Relu, 256->128->64
        linear 64->2"""
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


combinedgpu= NNforNorms(n_inputs=traintensor.shape[1]).to(device)
print(combinedgpu.parameters)


# ###########################################################################################################################################################3
# # ALL RIGHT LETS TRY TO TRAIN THAT THING
import torch.optim as optim

# Hyperparameters
# lr=0.001

epoch=1000
lr_start=1e-3
lr_end=1e-5
lr= np.linspace(lr_start, lr_end, epoch)

# optimizer=optim.Adam(combinedgpu.parameters())
# torch.optim.lr_scheduler.LinearLR(optim.Adam(combinedgpu.parameters()),)
# l=np.full_like(lr,0.00001)
# lr=np.hstack((lr,l))

print('Total Epoch: ', epoch)
print(f'Linear decrease in learning rate from {lr_start} to {lr_end}')


criterion = nn.MSELoss()

def RMSE(target,outputs):
    '''Calculates the RMSE between, the target and the model outputs'''
    return (torch.mean((target-outputs)**2))**.5

rlosses={}
losses=[]

rloss=0
rcount=0
#learningrates=[]

for epoch in range(epoch):
    print('Starting Epoch ',epoch)
    print('Learning Rate: ', lr[epoch])
    curpercent=0
    rl=[]

    optimizer = optim.Adam(combinedgpu.parameters(),lr=lr[epoch]) # optimizers# optimizers

    for i,data in enumerate(trainloader):

        # get the inputs and the target
        truth,inputs= data[0].to(device),data[1].to(device)

        # zero the gradients,
        optimizer.zero_grad()

        # Forward,
        out=combinedgpu(inputs)


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
        #    print(percent, 'loss: ', rloss/rcount)
            curpercent=percent
            rl.append(rloss/rcount)
            rloss=0
            rcount=0
    rlosses[epoch]=rl
    print('Epoch avg loss: ', np.mean(rl))
    # learningrates.append(lr)
    # lr*=.75


epoch+=1
torch.save(combinedgpu.state_dict(), f'.//weights//combinedgpu_{labels[0].item()}_{traintensor.shape[1]}input_{epoch}E_{lr_start:.1e}lr{lr_end:.1e}.pt')

fig,ax =plt.subplots()
epochavg=[]

for i in (rlosses):
    x=np.arange(i*len(rlosses[0]),len(rlosses[0])+i*len(rlosses[0]))
    ax.plot(x,rlosses[i],label=f'Epoch {i+1}, lr={lr[i]:.1e}')
    avg=np.mean(rlosses[i])
    epochavg.append(avg)

xx=[i*len(rlosses[0]) for i in rlosses]
ax.plot(xx,epochavg,'.-k')
#fig.legend()
#plt.xlim([15000,20000])
plt.savefig(f'.//outputs//combinedgpu_{labels[0].item()}_{traintensor.shape[1]}input_{epoch}E_{lr_start:.1e}lr{lr_end:.1e}.png')

with open('timestamps.txt', 'a') as f: 
    f.write(f"End-{datetime.now()}\n")