# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:56:55 2017

@author: rickdberg

Pair plots for gridded data

"""
import numpy as np

import rasterio
from user_parameters import (std_grids_path, ml_inputs_path)


# Load gridded data
f = rasterio.open(ml_inputs_path + "Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
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

grid_lons, grid_lats = np.meshgrid(lon, lat)
lons = grid_lons.flatten()
lats = grid_lats.flatten()
np.savetxt(std_grids_path + "lats_std.txt"
,grid_lats, delimiter='\t')
np.savetxt(std_grids_path + "lons_std.txt"
,grid_lons, delimiter='\t')

# eof
