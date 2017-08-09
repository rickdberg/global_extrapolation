# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:28 2017

@author: rickdberg

Module for applying machine learning method to flux data

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import linear_model

from site_metadata_compiler_completed import comp

#Load variables
database = "mysql://root:neogene227@localhost/iodp_compiled"
directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\\"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

grids = ['etopo1_depth', 'surface_porosity',
         'surface_productivity','woa_temp', 'woa_salinity', 'woa_o2',
         'acc_rate_archer','toc_wood','sed_rate_combined'
         ]

# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)
ml_train = site_metadata
ml_train = ml_train[ml_train['advection'].astype(float) >= 0]

oc_burial = ml_train['sed_rate_combined'].astype(float)*ml_train['toc_combined'].astype(float)
"""
X = pd.concat((ml_train[['etopo1_depth', 'surface_porosity',
  'surface_productivity',
  'woa_temp', 'woa_salinity', 'woa_o2',
  'acc_rate_archer'
                         ]], oc_burial), axis=1)
"""
X = ml_train[['etopo1_depth', 'surface_porosity',
              'surface_productivity','woa_temp', 'woa_salinity', 'woa_o2',
              'acc_rate_archer','toc_wood','sed_rate_combined'
                         ]]

X = np.array(X)

y = ml_train['interface_flux'].astype(float)
y = np.array(y)

###############################################################################
# Apply RF to grid

# Load mask from Basemap's maskocean
mask = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\continent_mask.txt"
, delimiter='\t')
mask = mask.astype('bool')

# Load gridded data
flat_grid_data = pd.DataFrame(columns=grids)
for n in np.arange(len(grids)):
    grid = np.loadtxt(directory + grids[n]+'_std.txt', delimiter='\t')
    grid_valid = grid[~mask]
    flat_grid_data.iloc[:,n] = grid_valid
del grid, grid_valid

# Random Forest Regression
flat_grid_data = np.array(flat_grid_data)
flat_nonnan_data = flat_grid_data[~np.isnan(flat_grid_data).any(axis=1)]

grid_areas = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\standardized files\grid_areas_std.txt"
, delimiter='\t')
grid_areas_masked = grid_areas[~mask]
grid_nonnan_areas = grid_areas_masked[~np.isnan(flat_grid_data).any(axis=1)]


# Random Forest Regression
"""
regressor = RandomForestRegressor(n_estimators=40,
                                  n_jobs=-1,
                                  min_samples_leaf=2)

"""
"""
# Multiple Linear Regression
regressor = linear_model.LinearRegression(n_jobs=-1)

# Adaboost Regressor
regressor = AdaBoostRegressor(n_estimators=100,
                              learning_rate=0.75)
"""
"""
regressor = linear_model.LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)

"""


# Gradient Boosting Regressor
regressor = GradientBoostingRegressor(loss='ls',n_estimators=120,

                                          learning_rate=0.1,
                                          min_samples_leaf=9,
                                          criterion='friedman_mse')

regressor.fit(X, y)

fluxes = regressor.predict(flat_nonnan_data)

total_flux = np.sum(fluxes*grid_nonnan_areas*1000000)

fluxes_masked = np.empty(len(grid_areas_masked)) * np.nan
fluxes_masked[~np.isnan(flat_grid_data).any(axis=1)] = fluxes

fluxes = np.empty(grid_areas.shape) * np.nan
fluxes[~mask] = fluxes_masked
#np.savetxt('mg_flux_gbr.txt', fluxes, delimiter='\t')

plt.close('all')
plt.imshow(fluxes, cmap='plasma')
plt.show = lambda : None  # prevents showing during doctests
plt.colorbar()
plt.show()

# eof
