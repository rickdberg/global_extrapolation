# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:02:30 2017

@author: rickdberg

Apply gridded datasets to individual hole locations

['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
              'crustal_age','coast_distance', 'ridge_distance', 'seamount',
              'surface_productivity','toc_seiter', 'opal', 'caco3',
              'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
              'caco3_archer','acc_rate_archer','toc_combined', 'toc_wood',
              'sed_rate_combined','lithology','lith1','lith2','lith3','lith4',
              'lith5','lith6','lith7','lith8','lith9','lith10','lith11',
              'lith12','lith13']
"""
import numpy as np
import pandas as pd
import MySQLdb
import scipy as sp
import rasterio

# Connect to database
user = 'root'
passwd = 'neogene227'
host = '127.0.0.1'
db = 'iodp_compiled'
hole_table = 'summary_all_bare'
con = MySQLdb.connect(user=user, passwd=passwd, host=host, db=db)
cursor = con.cursor()

# Hole data
sql = """SELECT *
FROM {}
; """.format(hole_table)
hole_data = pd.read_sql(sql, con)
site_lat = hole_data['lat']
site_lon = hole_data['lon']

directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files"

grid_names = ['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
              'crustal_age','coast_distance', 'ridge_distance', 'seamount',
              'surface_productivity','toc_seiter', 'opal', 'caco3',
              'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
              'caco3_archer','acc_rate_archer','toc_combined', 'toc_wood',
              'sed_rate_combined','lithology','lith1','lith2','lith3','lith4',
              'lith5','lith6','lith7','lith8','lith9','lith10','lith11',
              'lith12','lith13']


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

# Assign grid values to each hole
for n in np.arange(len(grid_names)):
    grid = pd.read_csv("{}//{}_std.txt".format(directory, grid_names[n]), sep='\t', header=None)
    hole_values = feature_pull(site_lat, site_lon, lat, lon, np.array(grid))
    hole_data = pd.concat((hole_data, pd.DataFrame(hole_values, columns=[grid_names[n]])), axis=1)
    print(n,'out of', len(grid_names))
hole_data.to_csv('hole_grid_data_nghp18.csv', index=False, na_rep='NULL')

# eof
