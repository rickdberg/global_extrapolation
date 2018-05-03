# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:26 2017

@author: rickdberg

Create maps



"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from site_metadata_compiler_completed import comp
import pandas as pd
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from user_parameters import (engine, metadata_table,
                             site_info, hole_info)

# Define fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Verdana'
mpl.rc('font',family='sans-serif')
mpl.rcParams['font.sans-serif'] = 'Verdana'
mpl.rcParams['font.cursive'] = 'Verdana'

# Load site data
site_metadata = comp(engine, metadata_table, site_info, hole_info)

site_coords  = pd.read_csv('sites_w_mg_coords.csv', sep=',')
site_coords = site_coords[['alat', 'alon']].as_matrix()

# define cartopy crs for the raster, based on rasterio metadata
crs = ccrs.PlateCarree()

# create figure
title = '$Scientific\ ocean\ drilling\ sites$\n $with\ Mg\ data\ (1966-2015)$'

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
ax.scatter(site_coords[:,1], site_coords[:,0],
           transform=ccrs.Geodetic(), c='y', s=5)


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  color='gray', alpha=0.2, linestyle='--', )
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()

# eof
