import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tqdm.auto import tqdm
from time import strftime

from AIce.functions import torchRMSE,trainloader,redim,getRMSE,testloader
from  AIce.models import NNforNormsandAngles


filename='../data/TrainsetD2C.csv'
bs=256
inputlist=['u_ERA5','v_ERA5','h_piomas','res_uw','res_vw''d2cwind']
target='norm_and_angle'

datapd,dataset,dataloader,means,stds,maxes,labels=trainloader(filename, bs, inputlist, target, log1p=True, static_threshold=1e-3, 
                log1pwind=False, shuffle=True, dotcross=False, no_static=True,
                model_norm_angle=True
                )
print(means,stds)
# Training loop for base inputs

device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')

# Hyperparameters
lr=1e-3
n_epoch=200
print('Total Epoch: ', n_epoch)
print('Learning Rate: ', lr)
print(labels)


# saving name
print('DID YOU CHECK FOR THE RIGHT INPUT_TYPE')
input_type='_D2C_and_res_currents'
print(input_type, "<- is that what we want?")

saving_name=f'Norms_and_Angles_Inputs_{input_type}_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}'
month='3. November'

mlp256=NNforNormsandAngles(len(labels[1])).to(device) # Model
optimizer = optim.Adam(mlp256.parameters(),lr=lr) # optimizers


# Setting up losses for plotting 
losses={
    'total':{},
    'norm':{},
    'cos':{},
    'sin':{}
    } 
epoch_loss={
    'total':[],
    'norm':[],
    'cos':[],
    'sin':[],
    } 
avg_losses={
    'total':{},
    'norm':{},
    'cos':{},
    'sin':{}
    } 


# Setting up the overaching progress bar
epoch_bar = tqdm(range(n_epoch), 
                 desc='Training',            # NEW (outer bar)
                 position=0, 
                 unit='epoch')

for epoch in (epoch_bar):
    # Using the bar print out so no print('Starting Epoch ',epoch)
    
    # resetting loss trackers
    curpercent=0
    epoch_loss={
        'total':[],
        'norm':[],
        'cos':[],
        'sin':[],
        } 
    batch_bar = tqdm(dataloader, 
                     desc=f'Epoch {epoch+1}/{n_epoch}',   # NEW (inner bar)
                     position=1, 
                     leave=False, 
                     unit='batch')
    
    for i,data in enumerate(batch_bar): # using bars

        # get the inputs and the target
        truth,inputs= data[0].to(device),data[1].to(device)
        # truth is tensor shape [256, 3]
        # inputs is tensor shape [256, n_inputs]

        target_norm=truth[:,0] # Tensor shape (N,) 
        target_cos=truth[:,1]  # Tensor shape (N,)
        target_sin=truth[:,2]  # Tensor shape (N,)

        # zero the gradients,
        optimizer.zero_grad()

        # Forward,
        pred_norm,pred_cos,pred_sin=mlp256(inputs) # Tensor shape (N,), Tensor shape (N,), Tensor shape (N,) 


        normloss = torchRMSE(target=target_norm, outputs=pred_norm)
        cosloss = torchRMSE(target=target_cos, outputs=pred_cos)
        sinloss = torchRMSE(target=target_sin, outputs=pred_sin)
        loss= normloss+cosloss+sinloss

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Saving Losses
        epoch_loss['total'].append(loss.item())
        epoch_loss['norm'].append(normloss.item())
        epoch_loss['cos'].append(cosloss.item())
        epoch_loss['sin'].append(sinloss.item())

        # NEW: running average of each loss over the batches seen so far this epoch
        batch_bar.set_postfix({k: f'{np.mean(v):.4f}' for k, v in epoch_loss.items()})
    batch_bar.close()


    losses['total'][epoch]=epoch_loss['total']
    losses['norm'][epoch]=epoch_loss['norm']
    losses['cos'][epoch]=epoch_loss['cos']
    losses['sin'][epoch]=epoch_loss['sin']

    avg_losses['total'][epoch]=np.average(epoch_loss['total'])
    avg_losses['norm'][epoch]=np.average(epoch_loss['norm'])
    avg_losses['cos'][epoch]=np.average(epoch_loss['cos'])
    avg_losses['sin'][epoch]=np.average(epoch_loss['sin'])

    # NEW: outer bar shows the average losses of the epoch that just finished
    epoch_bar.set_postfix({k: f'{v[epoch]:.4f}' for k, v in avg_losses.items()})
    # NEW (optional): permanent per-epoch record that doesn't break the bars
    epoch_bar.write(f'Epoch {epoch+1}: ' + ' | '.join(f'{k}={v[epoch]:.4f}' for k, v in avg_losses.items()))

# Making a loss plot
fig,ax = plt.subplots(

)

per_epoch_len=len(losses['total'][0])
max_plot=losses['total'][0][0]

x_line= np.arange(0,n_epoch)*per_epoch_len
x_avg=x_line+(per_epoch_len//2)
ax.plot(x_avg, avg_losses['total'].values(),'ko', label='Total Epoch-Average Loss ')
ax.plot(x_avg, avg_losses['norm'].values(),'ro' ,label='Norm Epoch-Average Loss')
ax.plot(x_avg, avg_losses['cos'].values(),'bo' ,label='Cos Epoch-Average Loss')
ax.plot(x_avg, avg_losses['sin'].values(),'go' ,label='Sin Epoch-Average Loss')

x_per_epoch=np.arange(0,per_epoch_len)
for i in range(n_epoch):
    xepoch=x_per_epoch+(i*per_epoch_len)
    ax.plot(xepoch,losses['total'][i],'k-', alpha=.3)
    ax.plot(xepoch,losses['norm'][i],'r-', alpha=.3)
    ax.plot(xepoch,losses['cos'][i],'b-', alpha=.3)
    ax.plot(xepoch,losses['sin'][i],'g-', alpha=.3)

ax.vlines(x_line,ymin=0,ymax = max_plot, linestyles= 'dashed', colors='cornflowerblue',alpha=.5, label='Epoch Starts')

ax.legend()

ax.set_ylim([0,max_plot])

ax.set_ylabel('Loss, non-dimensional')

ax.set_xlim([-5, n_epoch*per_epoch_len])
ax.set_xlabel('Epoch')

plt.savefig(f'..//{month}//outputs//training//loss_for_{saving_name}.png')
plt.show()


# Saving the model and it's info
torch.save(mlp256.state_dict(), f'..//{month}//weights//{saving_name}.pt')

with open(f'..//{month}//weights//{saving_name}.txt','w') as f:
    f.write(f'Model: NNforNormsandAngles (on gpu)\n')
    f.write(f'Weights: {saving_name}.pt\n')
    f.write(f'Associated inputs: {inputlist}\n')

# Store Losses (serialize)
with open(f'..//{month}//outputs//training//loss_{input_type}_{strftime("%d-%Hh_%Mm_%Ss")}.pickle', 'wb') as handle:
    pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)