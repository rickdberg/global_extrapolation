# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 08:56:55 2017

@author: rickdberg

Pair plots for gridded data

['etopo1_depth', 'surface_porosity', 'sed_thickness', 'crustal_age',
                'coast_distance', 'ridge_distance', 'seamount', 'surface_productivity',
                'toc', 'opal', 'caco3', 'woa_temp', 'woa_salinity', 'woa_o2',
                'caco3_archer','acc_rate_archer','lith1','lith2','lith3',
                'lith4','lith5','lith6','lith7','lith8',
                'lith9','lith10','lith11','lith12','lith13']

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import pairplot

from site_metadata_compiler_completed import comp


database = "mysql://root:neogene227@localhost/iodp_compiled"
directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\\"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"
lat_name = 'lat'
lon_name = 'lon'
grid_mask_fp = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\grid_mask.txt"

grids = [
    'etopo1_depth',
    'coast_distance',
    'sed_rate', 'woa_temp', 'woa_salinity'
            ]

# Load hole data
site_metadata = comp(database, metadata, site_info, hole_info)
training_inputs = pd.read_csv(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\training_inputs.csv"
,sep=',')
ml_train = pd.concat((site_metadata, training_inputs), axis=1)
ml_train = ml_train[ml_train['advection'].astype(float) >= 0]

training_data = ml_train[['lat','lon',
    'etopo1_depth',
    'coast_distance',
    'sed_rate', 'woa_temp', 'woa_salinity'
 ]]

training_labels = ['training' for x in training_data.index]
tr_labels = pd.DataFrame({'type':training_labels}, index=training_data.index)
training_data = pd.concat((training_data, tr_labels), axis=1)

# Load gridded data

grid_mask = np.loadtxt(grid_mask_fp, delimiter='\t').astype('int')
grid_mask = grid_mask.astype('bool')
flat_data = pd.DataFrame(columns=grids)
for n in np.arange(len(grids)):
    grid = np.loadtxt(directory + grids[n]+'_std.txt', delimiter='\t')
    grid_valid = grid[~grid_mask]
    flat_data.iloc[:,n] = grid_valid
del grid, grid_mask

# Add lat and lon
lat = np.linspace(-90, 90, num=len(grid_valid))
lon = np.linspace(-180, 180, num=len(grid_valid))
grid_data =  pd.concat((flat_data, pd.DataFrame({'lat':lat, 'lon':lon})), axis=1)
del flat_data, lat, lon, grid_valid

grid_labels = ['gridded' for x in grid_data.index]
gd_labels = pd.DataFrame({'type':grid_labels}, index=grid_data.index)
grid_data = pd.concat((grid_data, gd_labels), axis=1)


grid_data.to_csv('flat_grid_data.txt.gz', sep='\t', compression='gzip', chunksize=100000)
grid_data = pd.read_csv(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\flat_grid_data.txt.gz"
, sep='\t')

all_data = pd.concat((grid_data, training_data))
all_data = all_data.reset_index(drop=True)

plt.close('all')
plt.ioff()
fig = pairplot(all_data.loc[-1000:,['caco3', 'coast_distance','type']], hue= 'type', palette="husl")
fig.show()
# eof
