#one time running to get the linearly interpolated maps of speed from the buoys
import pandas as pd
import xarray as xr
import numpy as np
# import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import griddata


# Opening the dataset
ds = pd.read_csv('../data/DRIFT_DATA_TRAIN.csv')

ds['time']=pd.to_datetime(ds[['year',	'month',	'day']]) # Creating datetime index
ds=ds.drop(['year','month','day','doy'], axis=1) # removing all other time info
ds=ds.set_index(['time'])
ds=ds.drop_duplicates()# just in case

daily={time: dailydata for time,dailydata in ds.groupby(ds.index.date)} # group by daily data and turn to dict
ntime=len(daily) # Useful for later

# load the bathymetry as a proxy for the land
bath=pd.read_csv('../data/bathymetry_EASE.csv', header= None)

datamask= (1-bath.isna()) # make sure to check where we have data
land=(bath.where(bath==3000,np.nan)-3000)


lat_grid = pd.read_csv('../data/latitude_EASE.csv', header=None).values
lon_grid = pd.read_csv('../data/longitude_EASE.csv', header=None).values






land_arr = land.to_numpy().astype(np.float32)  # compute once

u_slice = np.empty_like(land_arr)
v_slice = np.empty_like(land_arr)
interU = np.tile(land_arr, (ntime, 1, 1))
interV = interU.copy()

y_idx = np.arange(interU.shape[1])
x_idx = np.arange(interU.shape[2])

grid_y, grid_x = np.meshgrid(y_idx, x_idx, indexing='ij')  # once, outside the loop

for i, day in enumerate(daily.keys()):
    u_slice[:] = land_arr  # cheap in-place copy of an already-numpy array
    v_slice[:] = land_arr
    rows = daily[day]['y_EASE'].astype(int).to_numpy()
    cols = daily[day]['x_EASE'].astype(int).to_numpy()
    u_slice[rows, cols] = daily[day]['u_buoy']
    v_slice[rows, cols] = daily[day]['v_buoy']

    valid_u = ~np.isnan(u_slice)
    valid_v = ~np.isnan(v_slice)
    interU[i] = griddata((grid_y[valid_u], grid_x[valid_u]), u_slice[valid_u], (grid_y, grid_x), method='linear')
    interV[i] = griddata((grid_y[valid_v], grid_x[valid_v]), v_slice[valid_v], (grid_y, grid_x), method='linear')



ds = xr.Dataset(
    {
        'u_inter':(['time', 'y', 'x'], interU),
        'v_inter':(['time', 'y', 'x'], interV)
    },
    coords={
        'time': list(daily.keys()),
        'y': y_idx,
        'x': x_idx,
        'lat': (['y', 'x'], lat_grid),
        'lon': (['y', 'x'], lon_grid),
    }
)

ds.to_netcdf('./interpolated_velocities.nc','w')

