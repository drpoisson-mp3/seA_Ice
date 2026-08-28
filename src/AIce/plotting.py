import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as cm
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def PredAgainstTarget(target,pred,colorer,cmap='viridis',title='Predictions in Terms of Observations',
                      save=False, savepath=''):
    """Add a description later #flemme"""
    minn=np.min([np.min(pred),np.min(target)])
    maxx=np.max([np.max(pred),np.max(target)])
    norm=Normalize(np.min(colorer),np.max(colorer))

    plt.title(title)
    plt.ylabel(pred.name)
    plt.xlabel(target.name)
    plt.xlim([0,maxx])
    plt.ylim([0,maxx])
    plt.scatter(target,pred,c=colorer,cmap=cmap, norm=norm, alpha=1)
    plt.plot(np.arange(maxx),'k')
    plt.colorbar(label=f'{colorer.name}')
    if save==True:
        plt.savefig(savepath)
    plt.show()


def ResidualOnTheMap(data,pred='',target='buoynorm', vmin=-10,vmax=10):
    """Make sur to have ../data/latitude_EASE.csv same for lon, and write the target as a STRING"""
    # Load the EASE grid lat/lon (only needs to happen once)
    lat_grid = pd.read_csv('../data/latitude_EASE.csv', header=None).values
    lon_grid = pd.read_csv('../data/longitude_EASE.csv', header=None).values

    # Convert your x,y grid indices to lat/lon for each sample
    # (assumes data['x'], data['y'] are integer row/col indices into the grid)
    lats = lat_grid[data['y_EASE'].astype(int), data['x_EASE'].astype(int)]
    lons = lon_grid[data['y_EASE'].astype(int), data['x_EASE'].astype(int)]

    residual = data[target] - data[pred]
    norm = Normalize(vmin, vmax)
    cmap = 'coolwarm'

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )

    # Set a sensible Arctic extent — adjust bounds to your buoy region
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

    # Add map context
    ax.add_feature(cfeature.LAND, facecolor='black')#, zorder=0)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.gridlines(alpha=.5)#draw_labels=True, linewidth=0.3, alpha=0.5)

    sc = ax.scatter(
        lons, lats,
        c=residual, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),  # tells cartopy the data is in lon/lat, not the plot's projection
        s=15, alpha=0.75
    )

    plt.colorbar(sc, ax=ax, label='Residual (Target - Prediction) [cm/s]', shrink=0.7)
    #plt.title('Spatial distribution of velocity-norm residuals')
    plt.show()
