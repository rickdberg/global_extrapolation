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
engine = create_engine(database)
# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)

sql = """select site, avg(lat) as alat, avg(lon) as alon
    from summary_all
    where site in ('807', '1171', '1082', '1086', '1012', '925', '1219', 'U1414', 'C0002', 'U1378')
    group by site
    ;"""

site_coords_in  = pd.read_sql(sql, engine)
site_coords_in = site_coords_in[['alat', 'alon']].as_matrix()
site_18 = np.array((19.15242, 85.77293))
site_coords_in = np.vstack((site_coords_in,site_18))


sql = """select site, avg(lat) as alat, avg(lon) as alon
    from summary_all
    where site in ('984', '1039')
    group by site
    ;"""
site_coords_out  = pd.read_sql(sql, engine)
site_coords_out = site_coords_out[['alat', 'alon']].as_matrix()

# Get template
f = rasterio.open(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)

# Load random forest grid into template
fluxes = np.loadtxt('fluxes_rf_noridge.txt', delimiter='\t')
rf = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=f.transform)
rf.write(fluxes, 1)
src = rf
rf.close()
title = '$Scientific\ ocean\ drilling\ sites$\n $with\ Mg\ isotope\ data$'

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
ax.scatter(site_coords_out[:,1], site_coords_out[:,0],
           transform=ccrs.Geodetic(), c='b', s=70, label= 'positive fractionation')

ax.scatter(site_coords_in[:,1], site_coords_in[:,0],
           transform=ccrs.Geodetic(), c='y', s=70, label = 'negative fractionation')


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  color='gray', alpha=0.2, linestyle='--', )
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.legend(loc='lower left')
plt.show()












