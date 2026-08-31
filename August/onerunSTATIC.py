import torch
import matplotlib.pyplot as plt
from time import strftime
import numpy as np

from AIce.functions import trainloader
from AIce.models import staticNN

device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
inputlist=['u_ERA5','v_ERA5','h_piomas','sic_CDR','x_EASE','y_EASE','bath','sin','cos', 'windnorm']
datapd,dataset,dataloader,means,stds,maxes,labels=trainloader('../data/DRIFT_DATA_TRAIN.csv',256,
                                                              inputlist=inputlist,target='static',static_threshold =1e-3 )

static256 = staticNN(len(labels[1])).to(device)

# print(static256)

# weight for the static case
total_cases=len(datapd)

total_static= (datapd['static']).sum()
n_moving = total_cases -total_static
pos_weight = torch.tensor([n_moving / total_static]).to(device) # pos_weight multiplies the positive 1 case, i.e. the is static case 

pos_weight*=2

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Hyperparameters
lr=1e-3
n_epoch=200
print('Total Epoch: ', n_epoch)
print('Learning Rate: ', lr)

epochavg=[]
rlosses={}
optimizer = torch.optim.Adam(static256.parameters(),lr=lr) # optimizers
for epoch in range(n_epoch):
    print('Starting Epoch ',epoch)
    curpercent=0
    rl=[]
    losses=[]
    rloss=0
    rcount=0

    for i,data in enumerate(dataloader):

        # get the inputs and the target
        truth,inputs= data[0].to(device),data[1].to(device)

        # zero the gradients,
        optimizer.zero_grad()

        # Forward,
        out=static256(inputs)


        loss = criterion(out,truth)
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
    e_avg=np.mean(rl)
    epochavg.append(e_avg)
    print('Epoch avg loss: ', e_avg)

torch.save(static256.state_dict(), f'.//weights//staticNN_{len(labels[1])}inputs_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}.pt')

with open(f'.//weights//staticNN_{len(labels[1])}inputs_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}.txt','w') as f:
    f.write(f'Model: static256 (on gpu)\n')
    f.write(f'Weights: stat256_{len(labels[1])}inputs_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}.pt\n')
    f.write(f'Associated inputs: {inputlist}')
    f.write(f'Weight favouring true positive: {pos_weight}')



fig,ax =plt.subplots()

for i in (rlosses.keys()):

    x=np.arange(i*len(rlosses[0]),len(rlosses[0])+i*len(rlosses[0]))
    ax.plot(x,rlosses[i])#,label=f'Epoch {i+1}, lr={lr[i]:.1e}')

xx=[i*len(rlosses[0]) for i in rlosses]
ax.plot(xx,epochavg,'.-k')
plt.savefig(f'.//outputs//loss_for_staticNN_{len(labels[1])}inputs_{n_epoch}E_{lr}lr_{strftime("%d-%Hh_%Mm_%Ss")}.png')
plt.close('all')