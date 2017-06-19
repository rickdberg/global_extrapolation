# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:22:19 2017

@author: rickdberg

Unpack 4D NETCDF file variables for 2D bottom-water value grids
"""
import numpy as np
import netCDF4 as ncdf

directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs"

nc_files = [
"{}\WOA - water temp, salinity\woa13_decav_t00_04v2.nc".format(directory),
"{}\WOA - water temp, salinity\woa13_decav_s00_04v2.nc".format(directory),
"{}\WOA - water temp, salinity\woa13_all_o00_01.nc".format(directory)
]

nc_write = [
'bottom_temp_original.csv',
'bottom_salintity_original.csv',
'bottom_o2_original.csv'
]

nc_vars = [
't_an',
's_an',
'o_an'
]

# Temperature data

def bottom_value(nc_file, nc_var_name, nc_write):
    f = ncdf.Dataset(nc_file, "r")
    var_3d = f.variables[nc_var_name][0,:,:,:]
    var_3d = np.flipud(var_3d)
    var_3d = var_3d.data
    fill_val = f.variables[nc_var_name]._FillValue
    f.close()

    bottom_values = np.empty(var_3d.shape[1:3])
    lat_row = np.empty(var_3d.shape[1])
    for lon in np.arange(var_3d.shape[2]):
        for lat in np.arange(var_3d.shape[1]):
            bottom_value = next((x for x in var_3d[:,lat, lon] if x != fill_val), np.nan)
            lat_row[lat] = bottom_value
        bottom_values[:,lon] = lat_row
    np.savetxt(nc_write, bottom_values, delimiter=",")

for n in np.arange(len(nc_files)):
    bottom_value(nc_files[n], nc_vars[n], nc_write[n])

# eof
