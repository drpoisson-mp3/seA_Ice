import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from time import strftime
from AIce.functions import trainloader,testloader,redim,torchRMSE
from AIce.models import NNforNorms

device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
inputlist=['u_ERA5','v_ERA5','h_piomas','sic_CDR','x_EASE','y_EASE','bath','sin','cos', 'windnorm']

# Hyperparameters
lr=1e-3
n_epoch=200
print('Total Epoch: ', n_epoch)
print('Learning Rate: ', lr)

data,_,dataloader,means,stds,maxes,labels=trainloader('../data/DRIFT_DATA_TRAIN.csv',256,inputlist,target='buoynorm')

print(labels)
print('aaa')
print(labels[1])

mlp256=NNforNorms(len(labels[1])).to(device)
print('Total Epoch: ', n_epoch)
print('Learning Rate: ', lr)
print(labels)

rlosses={}
losses=[]
rloss=0
rcount=0

optimizer = optim.Adam(mlp256.parameters(),lr=lr) # optimizers

for epoch in range(n_epoch):
    print('Starting Epoch ',epoch)
    curpercent=0
    rl=[]
    for i,data in enumerate(dataloader):

        # get the inputs and the target
        truth,inputs= data[0].to(device),data[1].to(device)

        # zero the gradients,
        optimizer.zero_grad()

        # Forward,
        out=mlp256(inputs)


        loss = torchRMSE(target=truth, outputs=out)
        # Backward
        loss.backward()
        losses.append(loss.item())
        # Optimize

        optimizer.step()

        rloss+= loss.item()
        rcount +=1
        #print where we at plus the loss
        percent= round(i/len(dataloader)*100)
        if percent != curpercent:
        #    print(percent, 'loss: ', rloss/rcount)
            curpercent=percent
            rl.append(rloss/rcount)
            rloss=0
            rcount=0
    rlosses[epoch]=rl
    print('Epoch avg loss: ', np.mean(rl))

saving_name=f'Norms_{len(labels[1])}inputs_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}'

torch.save(mlp256.state_dict(), f'.//weights//{saving_name}.pt')

with open(f'.//weights//{saving_name}.txt','w') as f:
    f.write(f'Model: NNforNorms (on gpu)\n')
    f.write(f'Weights: {saving_name}.pt\n')
    f.write(f'Associated inputs: {inputlist}')

fig,ax =plt.subplots()
epochavg=[]

for i in (rlosses):
    x=np.arange(i*len(rlosses[0]),len(rlosses[0])+i*len(rlosses[0]))
    ax.plot(x,rlosses[i])#,label=f'Epoch {i+1}, lr={lr[i]:.1e}')
    avg=np.mean(rlosses[i])
    epochavg.append(avg)
xx=[i*len(rlosses[0]) for i in rlosses]
ax.plot(xx,epochavg,'.-k')
plt.savefig(f'.//outputs//loss_for_{saving_name}.png')
plt.close('all')
