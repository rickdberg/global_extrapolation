# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:22:19 2017

@author: rickdberg

Assign World Ocean Atlas bottom-water values to every drilling hole

"""
import numpy as np
import netCDF4 as ncdf
import pandas as pd
import MySQLdb
from matplotlib import pyplot
import scipy as sp


# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
hole_table = 'summary_all'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cursor = con.cursor()

# Pore water chemistry data
sql = """SELECT *
FROM {}
; """.format(hole_table)
hole_data = pd.read_sql(sql, con)
site_lat = hole_data['lat']
site_lon = hole_data['lon']


directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs"

nc_files = [
"{}\WOA - water temp, salinity\woa13_decav_t00_04v2.nc".format(directory),
"{}\WOA - water temp, salinity\woa13_decav_s00_04v2.nc".format(directory),
"{}\WOA - water temp, salinity\woa13_all_o00_01.nc".format(directory)
]

nc_vars = [
'lat',
'lon'
]

woa_bottom = [
"{}\WOA - water temp, salinity\\bottom_temp_original.csv".format(directory),
"{}\WOA - water temp, salinity\\bottom_salintity_original.csv".format(directory),
"{}\WOA - water temp, salinity\\bottom_o2_original.csv".format(directory)
]

woa_names = [
'woa_bottom_temp',
'woa_bottom_salinity',
'woa_bottom_o2'
]

def feature_pull(site_lat, site_lon, lat, lon, z):
    lat_idx = []
    for n in site_lat:
        lat_idx.append((np.abs(lat-n)).argmin())
    lon_idx = []
    for n in site_lon:
        lon_idx.append((np.abs(lon-n)).argmin())

    # If closest grid values are null, assign nearest value
    full_mask = np.isnan(z)
    known_points = np.nonzero(~full_mask)
    known_coords = np.vstack((lat[known_points[0]],
                              lon[known_points[1]])) # convert to coords
    known_values = z[~full_mask]
    coords = np.vstack((site_lat, site_lon))
    z[lat_idx, lon_idx] = sp.interpolate.griddata(known_coords.T, known_values, coords.T, method='nearest', rescale=False)
    return z[lat_idx, lon_idx]

woa_holes = np.empty((len(hole_data), len(woa_bottom)))
for n in np.arange(len(woa_bottom)):
    f = ncdf.Dataset(nc_files[n], "r")
    lat = f.variables[nc_vars[0]][:]
    lon = f.variables[nc_vars[1]][:]
    f.close()
    hole_values = feature_pull(site_lat, site_lon, lat, lon, np.array(pd.read_csv(woa_bottom[n], header=None)))
    hole_data = pd.concat((hole_data, pd.DataFrame(hole_values, columns=[woa_names[n]])), axis=1)
hole_data.to_csv('hole_data.csv', index=False, na_rep='NULL')




# eof
