# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:28 2017

@author: rickdberg

Module for applying machine learning method to flux data

"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import linear_model
from pylab import savefig

from site_metadata_compiler_completed import comp
from user_parameters import (engine, metadata_table,
                             site_info, hole_info, ml_outputs_path,
                             std_grids_path)


#Define regression method (gbr, rf, mlr, adr)
meth = 'gbr'
features = ['etopo1_depth', 'surface_porosity','surface_productivity',
            'woa_temp', 'sed_rate_combined']

# Define fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Verdana'
mpl.rc('font',family='sans-serif')
mpl.rcParams['font.sans-serif'] = 'Verdana'
mpl.rcParams['font.cursive'] = 'Verdana'

###############################################################################

# Load site data
site_metadata = comp(engine, metadata_table, site_info, hole_info)
ml_train = site_metadata
ml_train['interface_flux'] = ml_train['interface_flux'].astype(float)

#advection = -0.0001
#ml_train.ix[ml_train['sed_thickness_combined'].astype(float) < 200, 'interface_flux'] = ml_train['interface_flux']+advection*ml_train['bottom_conc'].astype(float)
#oc_burial = ml_train['sed_rate_combined'].astype(float)*ml_train['toc_combined'].astype(float)

X = ml_train[features]
y = ml_train['interface_flux'].astype(float)
X = np.array(X)
y = np.array(y)

###############################################################################
# Apply regression to grid

# Load mask from Basemap's maskocean
mask = np.loadtxt(std_grids_path + "continent_mask.txt"
, delimiter='\t')
mask = mask.astype('bool')

# Load gridded data
flat_grid_data = pd.DataFrame(columns=features)
for n in np.arange(len(features)):
    grid = np.loadtxt(std_grids_path + features[n]+'_std.txt', delimiter='\t')
    grid_valid = grid[~mask]
    flat_grid_data.iloc[:,n] = grid_valid
del grid, grid_valid

flat_grid_data = np.array(flat_grid_data)
flat_nonnan_data = flat_grid_data[~np.isnan(flat_grid_data).any(axis=1)]

# Find grid areas (square km)
grid_areas = np.loadtxt(std_grids_path + "grid_areas_std.txt"
, delimiter='\t')
grid_areas_masked = grid_areas[~mask]
grid_nonnan_areas = grid_areas_masked[~np.isnan(flat_grid_data).any(axis=1)]

# Define regressor object
if meth == 'gbr':
    regressor = GradientBoostingRegressor(loss='ls',n_estimators=200,
                                          learning_rate=0.1,
                                          min_samples_leaf=9,
                                          criterion='friedman_mse')
elif meth == 'rf':
    regressor = RandomForestRegressor(n_estimators=200,
                                      n_jobs=-1,
                                      min_samples_leaf=3,
                                      criterion = 'friedman_mse')
elif meth == 'mlr':
    regressor = linear_model.LinearRegression(fit_intercept=True,
                                              normalize=True,
                                              n_jobs=-1)
elif meth == 'abr':
    regressor = AdaBoostRegressor(loss='linear',
                                  n_estimators=200,
                                  learning_rate=0.1)
else:
    print('Choose an available modeling method')
    quit()

regressor.fit(X, y)

fluxes = regressor.predict(flat_nonnan_data)
flux_err = 0.5*fluxes  # Calculated from plot of error vs. flux

total_flux = np.sum(fluxes*grid_nonnan_areas*1000000)
total_error = np.sum(flux_err*grid_nonnan_areas*1000000)

# Find fluxes of margins vs. abyssal
dist = np.loadtxt(std_grids_path + 'coast_distance_std.txt', delimiter='\t')
dist_valid = dist[~mask]
flat_dist_data = dist_valid
flat_dist_data = np.array(flat_dist_data)
flat_nonnan_dist = flat_dist_data[~np.isnan(flat_grid_data).any(axis=1)]

rates =  fluxes*grid_nonnan_areas*1000000
cutoff = 100
margin_fluxes = fluxes[flat_nonnan_dist <= cutoff]
abyssal_fluxes = fluxes[flat_nonnan_dist > cutoff]
np.savetxt(ml_outputs_path + 'margin_fluxes_{}.csv'.format(meth), margin_fluxes, delimiter='\t')
np.savetxt(ml_outputs_path + 'abyssal_fluxes_{}.csv'.format(meth), abyssal_fluxes, delimiter='\t')

# np.loadtxt(ml_outputs_path + 'margin_fluxes_{}.csv'.format(meth), delimiter='\t')
# np.loadtxt(ml_outputs_path + 'abyssal_fluxes_{}.csv'.format(meth), delimiter='\t')
mva_fig = plt.figure(figsize=(45/25.4, 35/25.4))
ax1 = mva_fig.add_axes([0.23,0.2,0.7,0.7])
plt.hist(abyssal_fluxes*1000,
         color='#006f00',
         alpha=0.8,
         normed=True,
         bins=100,
         label='>100 km',
         histtype='bar',
         edgecolor='black',
         linewidth=0.1)
plt.hist(margin_fluxes*1000,
         color='#a4ff48',
         alpha=0.65,
         normed=True,
         bins=100,
         label='<100 km',
         histtype='bar',
         edgecolor='black',
         linewidth=0.1)
plt.legend(fontsize=6, frameon=False)
plt.ylabel('$Probability\ Density$', fontsize=6, linespacing=0.5, labelpad=1)
plt.xlabel('$Mg^{2+}\ Flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=6, linespacing=0.5, labelpad=1)
plt.tick_params(axis='both', which='major', labelsize=6)
savefig(ml_outputs_path + 'plot_histogram_{}.pdf'.format(meth), dpi=1000)
savefig(ml_outputs_path + 'plot_histogram_{}.eps'.format(meth), dpi=1000)

margin_rate = np.sum(rates[flat_nonnan_dist <= 150])
abyssal_rate = np.sum(rates[flat_nonnan_dist > 150])


fluxes_masked = np.empty(len(grid_areas_masked)) * np.nan
fluxes_masked[~np.isnan(flat_grid_data).any(axis=1)] = fluxes

fluxes_shaped = np.empty(grid_areas.shape) * np.nan
fluxes_shaped[~mask] = fluxes_masked
np.savetxt(ml_outputs_path + 'fluxes_{}.txt'.format(meth),
fluxes_shaped, delimiter='\t')

plt.close('all')
plt.imshow(fluxes_shaped, cmap='plasma')
plt.show = lambda : None  # prevents showing during doctests
plt.colorbar(shrink=0.5)
plt.show()

# eof
