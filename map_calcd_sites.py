# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:26 2017

@author: rickdberg

Create maps



"""

import numpy as np
import scipy as sp
from sqlalchemy import create_engine
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from site_metadata_compiler_completed import comp
import pandas as pd
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#Datasets to pull from
database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)

mask = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\grid_mask.txt"
, delimiter='\t')
mask = mask.astype('bool')


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

"""
# Load random forest grid into template
fluxes = np.loadtxt('fluxes_rf_noridge.txt', delimiter='\t')
rf = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
rf.write(fluxes, 1)
src = rf
rf.close()
"""
title = '$Sites\ with\ quantified\ Mg\ fluxes$'

"""
# Plot random forest grid
# read image into ndarray
im = src.read()

# transpose the array from (band, row, col) to (row, col, band)
im = np.transpose(im, [1,2,0])
im = im[:,:,0]


xmin = src.transform[2]
xmax = src.transform[2] + src.transform[0]*src.width
ymin = src.transform[5] + src.transform[4]*src.height
ymax = src.transform[5]
"""
# define cartopy crs for the raster, based on rasterio metadata
crs = ccrs.PlateCarree()

# create figure
ax = plt.axes(projection=crs)
plt.title(title, fontsize=20)
ax.set_xmargin(0.05)
ax.set_ymargin(0.10)
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
# ax.stock_img()


# plot coastlines
#ax.add_feature(cartopy.feature.LAND)
#ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)
#ax.set_global()
ax.stock_img()

# To add points
fname = site_metadata[['lon','lat']].as_matrix()

# points = list(cartopy.io.shapereader.Reader(fname).geometries())
ax.scatter(fname[:,0], fname[:,1],
           transform=ccrs.Geodetic(), c='y')


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  color='gray', alpha=0.2, linestyle='--', )
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()












