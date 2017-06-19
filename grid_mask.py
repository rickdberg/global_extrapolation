# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:56:55 2017

@author: rickdberg

Pair plots for gridded data

"""
import numpy as np

from mpl_toolkits.basemap import maskoceans
import rasterio


# Load gridded data
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
mask = mask.astype('bool')
np.savetxt('continent_mask.txt', mask, delimiter='\t')

# eof
