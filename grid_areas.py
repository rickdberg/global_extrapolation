# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg
Make global gridded dataset of areas, 5'x5', grid-centered

"""

import numpy as np
import rasterio


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

###############################################################################
# Calculate grid areas
grid_areas = np.empty((len(lat), len(lon)))
for n in np.arange(len(lat)):
    #grid_areas[n,:] = 9.265*9.277*np.cos(lat[n]*2*np.pi/360)
    grid_areas[n,:] = (np.sin((lat[n]+0.5/12)*2*np.pi/360) - np.sin((lat[n]-0.5/12)*2*np.pi/360)) * (1/12*2*np.pi/360) * 6371**2
np.savetxt('grid_areas.txt', grid_areas, delimiter='\t')

# eof
