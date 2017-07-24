# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:26 2017

@author: rickdberg

Create maps


"""

import numpy as np
import matplotlib.pyplot as plt

import rasterio
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Get template
f = rasterio.open(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)
newaff = f.transform
top_left = f.transform * (0,0)
bottom_right = f.transform * (f.width, f.height)
lat_interval = (bottom_right[1]-top_left[1])/f.height
lon_interval = (bottom_right[0] - top_left[0])/f.width
lat = f.xy(0,0)[1] + np.arange(f.height)*lat_interval
lon = f.xy(0,0)[0] + np.arange(f.width)*lon_interval
lon[lon > 180] -= 360

# Load gridded data

# Load WOA bw temp grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\woa_temp_std.txt"
, delimiter='\t')

woat = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woat.write(fluxes, 1)
src = woat
woat.close()
title = '$Bottom\ water\ temperature\ (^\circ C)$'


# Load WOA bw salinity grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\woa_salinity_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Bottom\ water\ salinity\ (psu)$'


# Load etopo1_depth grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\etopo1_depth_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Water\ depth\ (mbsl)$'


# Load 'surface_productivity', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\surface_productivity_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Surface\ productivity$'


# Load 'toc_wood' grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\toc_wood_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Total\ organic\ carbon$'


# Load 'woa_o2' grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\woa_o2_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Bottom\ water\ oxygen$'


# Load 'surface_porosity' grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\surface_porosity_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Surface\ porosity$'


# Load 'coast_distance' grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\coast_distance_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Distance\ to\ coast$'


# Load 'ridge_distance' grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\ridge_distance_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Distance\ to\ ridge$'


# Load 'seamount', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\seamount_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Nearby\ seamounts$'


# Load 'opal', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\opal_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Opal\ concentration$'


# Load 'caco3', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\caco3_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$CaCO3\ concentration$'


# Load 'crustal_age', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\crustal_age_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Crustal\ age$'


# Load 'sed_thickness', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\sed_thickness_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Sediment\ thickness$'

# Load 'acc_rate_archer', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\acc_rate_archer_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$CaCO3\ accumulation\ rate$'

# Load 'caco3_archer', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\caco3_archer_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$CaCO3$'

# Load 'sed_rate_combined', grid
fluxes = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\sed_rate_combined_std.txt"
, delimiter='\t')

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Sedimentation\ Rate$'


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
plt.imshow(im, origin='upper', extent=[xmin, xmax, ymin, ymax], transform=crs)
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
