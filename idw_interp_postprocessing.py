# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:28 2017

@author: rickdberg
"""

import numpy as np
import matplotlib.pyplot as plt

import rasterio
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from site_metadata_compiler_completed import comp


database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load hole data
site_metadata = comp(database, metadata, site_info, hole_info)
site_metadata = site_metadata[site_metadata['advection'].astype(float) >= 0]
fluxes = site_metadata['interface_flux'].astype(float).as_matrix()
site_coords = np.array(site_metadata[['lat', 'lon']])

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
print('coordinates loaded')

# Load interpolated fluxes
fluxes = np.empty((len(lat), len(lon)))
ranges = [0,1000,2000,3000,4320]
# Load 'idw', grids
for n in np.arange(4):
    chunk = np.loadtxt(
        r"C:\Users\rickdberg\Downloads\idw_grid{}.txt".format(n+1)
        , delimiter='\t')
    fluxes[:,ranges[n]:ranges[n+1]] = chunk[:,ranges[n]:ranges[n+1]]
    del chunk
    print('chunk ', n, 'loaded')

# Remove continent areas
mask = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\continent_mask.txt"
, delimiter='\t')
mask = mask.astype('bool')
fluxes[mask] = np.nan
print('mask loaded')

np.savetxt('idw_fluxes_std.txt',fluxes,delimiter='\t')

# Calculate grid areas
grid_areas = np.empty((len(lat), len(lon)))
for n in np.arange(len(lat)):
    grid_areas[n,:] = 9.265*9.277*np.cos(lat[n]*2*np.pi/360)

# Calculate total flux
idw_flux = np.sum(fluxes[~mask]*grid_areas[~mask]*1000000)

# Write to nc file
woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
f.close()
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Inverse-distance\ weighted\ interpolation\ fluxes$'

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
