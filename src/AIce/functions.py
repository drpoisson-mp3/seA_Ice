import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset,DataLoader


def getRMSE(Target,Outputs):
    """To get the RMSE over the entire testset"""
    return np.sqrt(np.mean((Target-Outputs)**2))

def testloader(filename, inputlist, target,bs=0, means={}, stds={}, maxes={}, trainingset_loaded=True, training_file='', log1p=True, static_threshold=1e-5, log1pwind=False, shuffle=True):
    """For easy outputs: datapd,dataset,dataloader,means,stds,maxes,labels \n
    Function returning a test data (pd.dframe), dataset, dataloader (using bs or if bs==0-> bs=len(testset)
    based on the filename containing
    the data and uses all available data,\n
    IMPORTANTLY IT USES THE MEAN, STDS and MAXES from the training set\n
    if NO MEANS/STDS/MAXES provided, it opens training_file and calculates the above\n
    an inputlist which contains the same inputs models trained on\n
    x/y_ease, \n
    bath (bathymetry)\n
    sin/cos (of the year), \n
    u/v_ERA5,\n
    windnorm,\n
    sic_CDR',\n
    h_piomas,\n

    Target can be (for now) again case-sensitive: \n
    u/v \n
    buoynorm (precise log1p for proper normalization)\n
    static\n

    The year/month/day and id_buoy are consistently removed from the datasets.
    """
    if trainingset_loaded==False:
        #########################################################################################################################################33
        # to get the means and stds
        trainpd=pd.read_csv('../data/DRIFT_DATA_TRAIN.csv')
        # adding the norms

        trainpd['windnorm']=np.sqrt(trainpd['u_ERA5']**2+trainpd['v_ERA5']**2)
        trainpd['buoynorm']=np.sqrt(trainpd['u_buoy']**2+trainpd['v_buoy']**2)

        # for the speeds, we substract the mean and divide by the standard deviation to get some cleaner distribution
        speeds=['u_buoy', 'v_buoy', 'u_ERA5', 'v_ERA5','windnorm']

        means={}
        stds={}
        for label in speeds:
            means[label]=trainpd[label].mean()
            stds[label]=trainpd[label].std()
            trainpd[label]=(trainpd[label]-trainpd[label].mean())/trainpd[label].std()

        #For the Bathymetry we divide by the max as well (is actually a min)
        bath= pd.read_csv('../data/bathymetry_EASE.csv').values
        trainpd['bath']= bath[trainpd['y_EASE'].astype(int),trainpd['x_EASE'].astype(int)]
        max_bathy=np.max(np.abs(trainpd['bath']))

        # For the x,y, we simply divide by the max?
        max_x=np.max(np.abs(trainpd['x_EASE']))
        max_y =np.max(np.abs(trainpd['y_EASE']))



        maxes={
            'bath':max_bathy,
            'x_EASE': max_x,
            'y_EASE': max_y
        }

    # opening the csv
    ds=pd.read_csv(filename)


    if bs==0:
        bs=len(ds)
        print('Bath size is the full test set')

    # STARTING WITH TARGET

    # BUOYNORMS
    if 'buoynorm'==target:
        ds['buoynorm']=np.sqrt(ds['u_buoy']**2+ds['v_buoy']**2)
        if log1p==True:
            #log1p Normalization
            ds['buoynorm']=np.log1p(ds['buoynorm'])
            targeted=ds[['buoynorm']]
            print('Target is buoynorm normalized by log1p')
        
        elif log1p==False:
            #z-score normalization
            ds['buoynorm']=(ds['buoynorm']-means['buoynorm'])/stds['buoynorm']
            targeted=ds[['buoynorm']]
            print('Target is buoynorm normalized by z-score')
        
        else:
            print('wdym boolean is not True or False')
        
    #STATICITY
    elif 'static'==target:
        u0=np.abs(ds['u_buoy'])<=static_threshold
        v0=np.abs(ds['v_buoy'])<=static_threshold
        nospeed=u0*v0
        targeted=nospeed.to_frame(name='static')
        print(f'Target is Staticity with threshold {static_threshold}')

    
    # u/v_buoy
    elif 'u/v'==target:
        uv=['u_buoy','v_buoy']

        for label in uv:
            ds[label]=(ds[label]-means[label])/stds[label]
        targeted = pd.DataFrame([ds['u_buoy'],ds['v_buoy']]).T# had to reshape from u,v x rows to rows x u,v
        print('Target is u/v components of the wind normalized by z-score')

    else:
        print('Please specify a valid target ( buoynorm, u/v, static)')
        print(targettensor)

    
    #time of the year
    if 'sin' in inputlist:
        #ADDING THE COS AND SIN OF THE YEAR
        ds['sin']=np.sin(ds['doy']*(2*np.pi/364))
        ds['cos']=np.cos(ds['doy']*(2*np.pi/364))
        print('Sin and Cos added')

    # POSITIONS
  
    # bathymetry
    if 'bath' in inputlist:
        bath= pd.read_csv('../data/bathymetry_EASE.csv').values
        ds['bath']= bath[ds['y_EASE'].astype(int),ds['x_EASE'].astype(int)]

        ds['bath']=ds['bath']/maxes['bath']
        print('Bathymetry (bath) normalized by maximum')

    #x/y_ease
    if 'x_EASE' in inputlist:
        # simply divide by max normalization
        max_x=np.max(np.abs(ds['x_EASE']))
        max_y =np.max(np.abs(ds['y_EASE']))

        ds['x_EASE']=ds['x_EASE']/maxes['x_EASE']
        ds['y_EASE']=ds['y_EASE']/maxes['y_EASE']
        print('x/y (x_EASE, y_EASE) normalized by maximum')

    # WIND
    #norm
    if 'windnorm' in inputlist:
        ds['windnorm']=np.sqrt(ds['u_ERA5']**2+ds['v_ERA5']**2)
        if log1pwind==True:
            #log1p Normalization
            ds['windnorm']=np.log1p(ds['windnorm'])

            print('Windnorm normalized by log1p')
        
        elif log1pwind==False:
            #z-score normalization
            ds['windnorm']=(ds['windnorm']-means['windnorm'])/stds['windnorm']
  
            print('Windnorm normalized by z-score')
        else:
            print('wdym boolean is not True or False')

    #u/v
    if 'u_ERA5' in inputlist:
        uv=['u_ERA5','v_ERA5']

        for label in uv:

            ds[label]=(ds[label]-means[label])/stds[label]
            print('Wind components (u/v_ERA5) normalized by z-score')

    #Nothing to do with 'sic_CDR', 'h_piomas'


    for i in ds.columns:
        if i not in inputlist:
            ds=ds.drop(i,axis=1)

    labels=[targeted.columns,ds.columns]



    datapd=pd.concat((targeted,ds), axis=1)
    inputtensor=torch.tensor(ds.values, dtype=torch.float32)
    targettensor=torch.tensor(targeted.values, dtype=torch.float32)
    dataset=TensorDataset(targettensor,inputtensor)
    dataloader=DataLoader(dataset,batch_size=bs,shuffle=shuffle)

    return datapd,dataset,dataloader,means,stds,maxes,labels

def trainloader(filename, bs, inputlist, target, log1p=True, static_threshold=1e-5, log1pwind=False, shuffle=True):
    """For easy outputs: datapd,dataset,dataloader,means,stds,maxes,labels \n
    Function returning a training data (pd.dframe), dataset, dataloader, the means and stds and maxes, based on the filename containing
    the data, a selected batchsize = bs, an inputlist which could contain (case-sensitive, w/o what's in between the brackets):\n
     x/y_ease, \n
    bath (bathymetry)\n
    sin/cos (of the year), \n
    u/v_ERA5,\n
    windnorm,\n
    sic_CDR',\n
    h_piomas,\n
    Target can be (for now) again case-sensitive: \n
    u/v \n
    buoynorm (precise log1p for proper normalization)\n
    static\n

    The year/month/day and id_buoy are consistently removed from the datasets.
    """

    # opening the csv
    ds=pd.read_csv(filename)

    means={}
    stds={}
    maxes={}


    # STARTING WITH TARGET

    # BUOYNORMS
    if 'buoynorm'==target:
        ds['buoynorm']=np.sqrt(ds['u_buoy']**2+ds['v_buoy']**2)
        if log1p==True:
            #log1p Normalization
            ds['buoynorm']=np.log1p(ds['buoynorm'])
            targeted=ds[['buoynorm']]
            print('Target is buoynorm normalized by log1p')
        
        elif log1p==False:
            #z-score normalization
            means['buoynorm']=ds['buoynorm'].mean()
            stds['buoynorm']=ds['buoynorm'].std()
            ds['buoynorm']=(ds['buoynorm']-ds['buoynorm'].mean())/ds['buoynorm'].std()
            print('Target is buoynorm normalized by z-score')
            targeted=ds[['buynorm']]
        
        else:
            print('wdym boolean is not True or False')
        
    #STATICITY
    elif 'static'==target:
        u0=np.abs(ds['u_buoy'])<=static_threshold
        v0=np.abs(ds['v_buoy'])<=static_threshold
        nospeed=u0*v0
        targeted=nospeed.to_frame(name='static')
        print(f'Target is Staticity with threshold {static_threshold}')
    
    
    # u/v_buoy
    elif 'u/v'==target:
        uv=['u_buoy','v_buoy']

        for label in uv:
            means[label]=ds[label].mean()
            stds[label]=ds[label].std()
            ds[label]=(ds[label]-ds[label].mean())/ds[label].std()

        targeted = pd.DataFrame([ds['u_buoy'],ds['v_buoy']]).T# had to reshape from u,v x rows to rows x u,v
        print('Target is u/v components of the wind normalized by z-score')

    else:
        print('Please specify a valid target ( buoynorm, u/v, static)')
        print(targettensor)

    
    #time of the year
    if 'sin' in inputlist:
        #ADDING THE COS AND SIN OF THE YEAR
        ds['sin']=np.sin(ds['doy']*(2*np.pi/364))
        ds['cos']=np.cos(ds['doy']*(2*np.pi/364))
        print('Sin and Cos added')
    # POSITIONS
  
    # bathymetry
    if 'bath' in inputlist:
        bath= pd.read_csv('../data/bathymetry_EASE.csv').values
        ds['bath']= bath[ds['y_EASE'].astype(int),ds['x_EASE'].astype(int)]

        max_bath=np.max(np.abs(ds['bath']))
        ds['bath']=ds['bath']/max_bath
        maxes['bath']=max_bath
        print('Bathymetry (bath) normalized by maximum')
    
    #x/y_ease
    if 'x_EASE' in inputlist:
        # simply divide by max normalization
        max_x=np.max(np.abs(ds['x_EASE']))
        max_y =np.max(np.abs(ds['y_EASE']))
        maxes['x_EASE']= max_x
        maxes['y_EASE']= max_y

        ds['x_EASE']=ds['x_EASE']/max_x
        ds['y_EASE']=ds['y_EASE']/max_y
        print('x/y (x_EASE, y_EASE) normalized by maximum')


    # WIND
    #norm
    if 'windnorm' in inputlist:
        ds['windnorm']=np.sqrt(ds['u_ERA5']**2+ds['v_ERA5']**2)
        if log1pwind==True:
            #log1p Normalization
            ds['windnorm']=np.log1p(ds['windnorm'])

            print('Windnorm normalized by log1p')
        
        elif log1pwind==False:
            #z-score normalization
            means['windnorm']=ds['windnorm'].mean()
            stds['windnorm']=ds['windnorm'].std()
            ds['windnorm']=(ds['windnorm']-ds['windnorm'].mean())/ds['windnorm'].std()
  
            print('Windnorm normalized by z-score')
        else:
            print('wdym boolean is not True or False')

    #u/v
    if 'u_ERA5' in inputlist:
        uv=['u_ERA5','v_ERA5']

        for label in uv:
            means[label]=ds[label].mean()
            stds[label]=ds[label].std()
            ds[label]=(ds[label]-ds[label].mean())/ds[label].std()
        print('Wind components (u/v_ERA5) normalized by z-score')
    #Nothing to do with 'sic_CDR', 'h_piomas'


    for i in ds.columns:
        if i not in inputlist:
            ds=ds.drop(i,axis=1)

    labels=[targeted.columns,ds.columns]


    
    
    datapd=pd.concat((targeted,ds), axis=1)
    inputtensor=torch.tensor(ds.values, dtype=torch.float32)
    targettensor=torch.tensor(targeted.values, dtype=torch.float32)
    dataset=TensorDataset(targettensor,inputtensor)
    dataloader=DataLoader(dataset,batch_size=bs,shuffle=shuffle)

    return datapd,dataset,dataloader,means,stds,maxes,labels


def redim(datapd,means,stds,maxes, log1p=True):
    """Redimensionalize the data, there could still be some mistakes
    DOES NOT INCLUDE THE WINDNORM BY LOG1P"""
    for i in datapd.columns:
        if i in means.keys():
            datapd[i]=datapd[i]*stds[i]+means[i]
        if i in maxes.keys():
            datapd[i]*=maxes[i]
        if i =='buoynorm':
            datapd[i]=np.expm1(datapd[i])

    return datapd