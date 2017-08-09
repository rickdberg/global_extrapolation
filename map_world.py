# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:26 2017

@author: rickdberg

Create maps



"""

import numpy as np
import scipy as sp
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

# define cartopy crs for the raster, based on rasterio metadata
crs = ccrs.PlateCarree()

# create figure
ax = plt.axes(projection=crs)
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
"""
land_50m = cartopy.feature.NaturalEarthFeature('physical', 'NE1', '50m',
                                               facecolor='none')

"""
# plot coastlines
#ax.add_feature(land_50m,edgecolor='gray')
#ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)
#ax.set_global()
ax.stock_img()
ax.add_feature(cartopy.feature.OCEAN)



plt.show()












