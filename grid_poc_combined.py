# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:26:37 2017

@author: rickdberg

POC grid, using Seiter, then Jahnke, then Marquardt et al 2010 relationship

"""

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.basemap import maskoceans


# Function for opening grid files and retrieving datasets into arrays
def rast(f_path):
    src = rasterio.open(f_path)
    nodata = src.nodata
    arr = src.read(1).astype(float)
    src.close()
    arr[arr == nodata] = np.nan
    return arr

newarr_shape = (2160,4320)

toc_jahnke = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Jahnke - TOC burial\OrganicCDistribution.csv"
, delimiter=',')
def jahnke_arrange(toc_jahnke):
    st_left = toc_jahnke[:,:30]
    st_right = toc_jahnke[:,30:]
    cut = np.concatenate((st_right, st_left), axis=1)
    cut[cut > 100] = np.nan
    cut[cut < 0] = np.nan
    top_filler = np.empty((14,cut.shape[1])) * np.nan  # Only takes it to 89 degrees
    bottom_filler = np.empty((14,cut.shape[1])) * np.nan
    return np.concatenate((top_filler, cut, bottom_filler))
toc_jahnke = jahnke_arrange(toc_jahnke)

toc = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Seiter - TOC\TOC_Seiteretal2004.asc"
)
def toc_fill(toc):
    top_filler = np.empty((4,toc.shape[1])) * np.nan
    toc[toc > 100] = np.nan
    toc[toc < 0] = np.nan
    return np.concatenate((top_filler, toc))
toc_seiter = toc_fill(toc)

sed_rate = np.loadtxt('sed_rate_combined.txt', delimiter='\t')
toc_marquardt = 3 - 2.8*np.exp(-44.5*sed_rate*100)

# Get coordinates of porosity grid, which all others will be matched to
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
f.close()


# Mask
grid_lons, grid_lats = np.meshgrid(lon, lat)
lons = grid_lons.flatten()
lats = grid_lats.flatten()
locations = np.c_[lons, lats]
fakedata = np.empty(len(locations))
ocmask = maskoceans(locations[:,0],locations[:,1], fakedata,
                    inlands=False, resolution='l')
mask = np.ma.reshape(ocmask, (len(lat),len(lon)))
mask = ~mask.mask

# Resample gridded data
reg_datasets = [toc_seiter, toc_jahnke]
# Function for resampling datasets to be consistent with 5" pixel-registered porosity dataset
def resamp(dataset, resampler, newarr_shape, newaff):
    arr = dataset
    height = arr.shape[0]
    width = arr.shape[1]
    newarr = np.empty(shape=newarr_shape)
    if arr.shape[1] % 2 == 0:
        aff = Affine(360/(width), 0.0, -180.0,
                      0.0, -180/(height), 90)
    else:
        lat_interval = 180/(height-1)
        lon_interval = 360/(width-1)
        aff = Affine(360/(width-1), 0.0, -180.0-lon_interval,
                      0.0, -180/(height-1), 90+lat_interval)
    reproject(
        arr, newarr,
        src_transform = aff,
        dst_transform = newaff,
        src_crs = {},
        dst_crs = {},
        resample = resampler)
    return newarr

regs = np.arange(len(reg_datasets))

gridded_data = np.empty((newarr_shape[0], newarr_shape[1], len(regs)))  #+len(categories)))
for n in regs:
    newarr = resamp(reg_datasets[n], Resampling.bilinear, newarr_shape, newaff)
    gridded_data[:,:,n] = newarr

# Combine Seiter, Jahnke, and Marquardt grids
toc = gridded_data[:,:,0]
#toc[toc < 0] = np.nan
toc[np.isnan(toc)] = gridded_data[:,:,1][np.isnan(toc)]
#toc[toc < 0] = np.nan
toc[np.isnan(toc)] = toc_marquardt[np.isnan(toc)]
#toc[toc < 0] = np.nan
toc = np.ma.masked_array(toc,
                         mask=mask,
                         fill_value=np.nan)
toc.data[mask] = np.nan

np.savetxt('toc_combined.txt', toc.data, delimiter='\t')

# Plot grid
# Load 'sed_rate', grid
fluxes = np.log(toc.data)

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=newarr_shape[0], width=newarr_shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=newaff)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$POC\ (wt\%)$'

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
plt.imshow(im, origin='upper', extent=[xmin, xmax, ymin, ymax],transform=crs, cmap="Greys")
#plt.contourf(lons, lats, im, transform=crs, cmap="Greys")

plt.colorbar(shrink=0.5)
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='gray',
                  alpha=0.1, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()

# eof
