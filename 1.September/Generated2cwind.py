import os
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from AIce.functions import torchRMSE,trainloader,redim,getRMSE,testloader
from  AIce.models import uv256NN
from AIce.plotting import ResidualOnTheMap,PredAgainstTarget
from time import strftime

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    All inputs in degrees, can be arrays
    Returns distance in m
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371000 * c  # m



###########################################################################################################################

device='cuda'
allinput=['u_ERA5','v_ERA5','h_piomas','sic_CDR','x_EASE','y_EASE','bath','sin','cos', 'windnorm']
#lst=['u_ERA5', 'v_ERA5', 'h_piomas', 'sic_CDR', 'bath']
#weightpath='./weights/UV_5inputs_20E_0.001lr_18-13h_37m_54s.pt'data/DRIFT_DATA_TRAIN.csv
path='/aos/home/drpoisson/Documents/CODE/recherchemsc/seA_Ice/data/DRIFT_DATA_TRAIN.csv'
testdata,_,_,means,stds,maxes,labels=testloader(path,inputlist=allinput ,target='u/v',
                                trainingset_loaded=False, training_file=path)# Get the means,stds,maxes using the longest list possible

testdata=redim(testdata,
               means=means,
               stds=stds,
               maxes=maxes)

testdata['anglewind']=np.degrees(np.arctan2(testdata['u_ERA5'],testdata['v_ERA5']))

print(len(testdata))

#########################################################################################3
bath=pd.read_csv('/aos/home/drpoisson/Documents/CODE/recherchemsc/seA_Ice/data/bathymetry_EASE.csv', header= None)
lat_grid = pd.read_csv('/aos/home/drpoisson/Documents/CODE/recherchemsc/seA_Ice/data/latitude_EASE.csv', header=None).values
lon_grid = pd.read_csv('/aos/home/drpoisson/Documents/CODE/recherchemsc/seA_Ice/data/longitude_EASE.csv', header=None).values

# Convert your x,y grid indices to lat/lon for each sample
# # (assumes data['x'], data['y'] are integer row/col indices into the grid)
lats = lat_grid[testdata['y_EASE'].astype(int), testdata['x_EASE'].astype(int)]
lons = lon_grid[testdata['y_EASE'].astype(int), testdata['x_EASE'].astype(int)]

#datamask= (1-bath.isna()) # make sure to check where we have data
land=(bath.where(bath==3000,2999)-2999).to_numpy().astype(np.float32) # getting a land numpy array

testdata['d2cwind']= np.zeros_like(testdata['anglewind'], dtype=np.float32)


a=testdata['u_ERA5']/testdata['v_ERA5'] 
b=testdata['x_EASE']-a*testdata['y_EASE']





inwind=np.array([y,x]).squeeze().T

roundd=np.round(inwind).astype(int)
inside = ((roundd >= 0) & (roundd <= 360)).all(axis=1) # selects the inside
roundd = roundd[inside]

latland = lonland = np.nan

for i in roundd:
    y_idx=i[0]
    x_idx=i[1]

    
    if land[y_idx,x_idx]==1:
        latland=lat_grid[y_idx,x_idx]
        lonland=lon_grid[y_idx,x_idx]
        break


latpoint=lats[buoy]
lonpoint=lons[buoy]

testdata.loc[testdata.index[buoy], 'd2cwind']=(np.round(haversine_vectorized(latland,lonland,latpoint,lonpoint)*1e-3))
percent=(buoy/testdata.shape[0])*100
if percent %1==0:
    print(percent)

testdata.to_csv('TrainsetD2C.csv', mode='w')