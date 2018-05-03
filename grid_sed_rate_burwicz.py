# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:02:30 2017

@author: rickdberg

Create Holocene sed rate grid based on water depth
Burwicz and Wallmann, 2011


"""
import numpy as np
import pandas as pd
import rasterio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from user_parameters import (std_grids_path, ml_inputs_path)


# Get coordinates of porosity grid, which all others will be matched to
f = rasterio.open(ml_inputs_path + "Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd")
newaff = f.transform
top_left = f.transform * (0,0)
bottom_right = f.transform * (f.width, f.height)
lat_interval = (bottom_right[1]-top_left[1])/f.height
lon_interval = (bottom_right[0] - top_left[0])/f.width
lat = f.xy(0,0)[1] + np.arange(f.height)*lat_interval
lon = f.xy(0,0)[0] + np.arange(f.width)*lon_interval
lon[lon > 180] -= 360
f.close()

grid = pd.read_csv(std_grids_path + "etopo1_depth_std.txt", sep='\t', header=None)
grid = grid.as_matrix() * -1
grid[grid[:,:] <= 0] = np.nan
sed_rate = ((0.117/(1+(grid/200)**3))+(0.006/(1+(grid/4000)**10)))/100

np.savetxt(std_grids_path + 'sed_rate_burwicz_std.txt', sed_rate, delimiter='\t')



# Plot grid
# Load 'sed_rate', grid
fluxes = sed_rate

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Sedimentation\ rate\ Burwicz\ (m/y)$'

# Read image into ndarray
im = src.read()
# transpose the array from (band, row, col) to (row, col, band)
im = np.transpose(im, [1,2,0])
im = im[:,:,0]
xmin = src.transform[2]
xmax = src.transform[2] + src.transform[0]*src.width
ymin = src.transform[5] + src.transform[4]*src.height
ymax = src.transform[5]
#ax.set_global()

# define cartopy crs for the raster, based on rasterio metadata
crs = ccrs.PlateCarree()

# create figure
ax = plt.axes(projection=crs)
plt.title(title, fontsize=20)
ax.set_xmargin(0.05)
ax.set_ymargin(0.10)
# ax.stock_img()

# plot raster
plt.imshow(im, origin='upper', extent=[xmin, xmax, ymin, ymax], transform=crs, cmap="Greys")
# ax.coastlines(resolution='10m', color='k', linewidth=0.2)
# plt.colorbar(shrink=0.5)
ax.add_feature(cartopy.feature.LAND)
# ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)
# ax.add_feature(cartopy.feature.LAND, zorder=50, edgecolor='k')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  color='gray', alpha=0.1, linestyle='--', )
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()

# eof
