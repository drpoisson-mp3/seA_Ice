#one time running to get the linearly interpolated maps of speed from the buoys
import pandas as pd
import xarray as xr
import numpy as np
# import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import griddata

# Opening the dataset
ds = pd.read_csv('../data/DRIFT_DATA_TRAIN.csv')

ds=ds.drop(['year','month','day'], axis=1) # removing all other time info
ds=ds.set_index(['doy'])
ds=ds.drop_duplicates()# just in case

doyly={doy: dailydata for doy,dailydata in ds.groupby(ds.index)} # group by daily data and turn to dict


# load the bathymetry as a proxy for the land
bath=pd.read_csv('../data/bathymetry_EASE.csv', header= None)

datamask= (1-bath.isna()) # make sure to check where we have data
land=(bath.where(bath==3000,np.nan)-3000).to_numpy().astype(np.float32) # getting a land numpy array
landmask=bath.where(bath!=3000,np.nan)
oceanmask=(datamask*landmask)
oceanmask= oceanmask/oceanmask

lat_grid = pd.read_csv('../data/latitude_EASE.csv', header=None).values
lon_grid = pd.read_csv('../data/longitude_EASE.csv', header=None).values
y_idx = np.arange(land.shape[0])
x_idx = np.arange(land.shape[0])

u_map = land.copy()

# v_slice = np.empty_like(land)
# interU = np.tile(land_arr, (ntime, 1, 1))
# interV = interU.copy()


grid_y, grid_x = np.meshgrid(y_idx, x_idx, indexing='ij')  # once, outside the loop

# making a map for day1
interU = np.tile(land, (len(doyly), 1, 1))

for i in doyly:
    rows = doyly[i]['y_EASE'].astype(int).to_numpy()
    cols = doyly[i]['x_EASE'].astype(int).to_numpy()

    u_map[rows,cols]=doyly[i]['u_buoy']
    u_mapPOINTS=~np.isnan(u_map)
    interp1=griddata((grid_y[u_mapPOINTS],grid_x[u_mapPOINTS]), u_map[u_mapPOINTS], (grid_y,grid_x), method='cubic')
    interp1*=oceanmask
    interU[i-1,:,:]=interp1
ds = xr.Dataset(
    {
        'u_inter':(['doy', 'y', 'x'], interU),

    },
    coords={
        'doy': list(doyly.keys()),
        'y': y_idx,
        'x': x_idx,
        'lat': (['y', 'x'], lat_grid),
        'lon': (['y', 'x'], lon_grid),
    }
)

ds.to_netcdf('./interpolated_velocities.nc','w')