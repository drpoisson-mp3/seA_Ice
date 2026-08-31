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


def RecallPrecisionStatic(truth,pred,res=100,xmin=0,xmax=1):
    scores =pred
    y_true= truth
    thresholds = np.linspace(0, 1, res+1) # Looking at predictions from 0,100%
    precisions = []
    recalls = []
    
    for t in thresholds:
        
        pred_static = scores >= t # retains the number of prediction with certainty above the t threshold

        tp = (pred_static & y_true).sum() # True positive (pred=true, y=true)
        fp = (pred_static & ~y_true).sum() # false positive (pred= true, y=false)
        fn = (~pred_static & y_true).sum() # false negative (pred = false, y=true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan # "of the things I called static, how many actually were?"
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan # "how many of the real statics did I find?"

        precisions.append(precision)
        recalls.append(recall)

    plt.plot(thresholds, precisions, label='precision')
    plt.plot(thresholds, recalls, label='recall')
    plt.xlabel('threshold')
    plt.xticks([x/10 for x in range(10) ])
    plt.xlim([xmin,xmax])
    plt.grid()
    plt.legend()
    plt.show()