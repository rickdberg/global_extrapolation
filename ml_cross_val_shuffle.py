# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:47:32 2017

@author: rickdberg

['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood'
  'sed_rate_combined','lithology','lith1','lith2','lith3','lith4','lith5',
  'lith6','lith7','lith8','lith9','lith10','lith11','lith12',
  'lith13']


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from site_metadata_compiler_completed import comp

from user_parameters import (engine, metadata_table,
                             site_info, hole_info)

meth = 'gbr'

#Datasets to pull from
lat_name = 'lat'
lon_name = 'lon'

# Load site data
site_metadata = comp(engine, metadata_table, site_info, hole_info)
ml_train = site_metadata
ml_train = ml_train[ml_train['advection'].astype(float) >= 0]
ml_train = ml_train[ml_train['site'] != '796']
ml_train = ml_train[ml_train['interface_flux'].astype(float) <= 0.04]
ml_train = ml_train[ml_train['interface_flux'].astype(float) >= -0.01]


y = ml_train['interface_flux'].astype(float)
y = np.array(y)  # .reshape(-1,1)
X = ml_train[['etopo1_depth', 'lat', 'lon',
    'coast_distance','sed_rate','toc_combined',
    'woa_temp', 'woa_salinity', 'acc_rate_archer'
 ]]
X = np.array(X)

# Create a random dataset

cycles = 10
testsize = int(np.ceil(len(ml_train)/10))

train_score = np.empty(cycles)
test_score = np.empty(cycles)
y_train_rf = np.empty((len(y)-testsize, cycles))
y_test_rf = np.empty((testsize, cycles))
y_train_orig = np.empty((len(y)-testsize, cycles))
y_test_orig = np.empty((testsize, cycles))
for n in np.arange(cycles):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=testsize
                                                        )
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

    regressor.fit(X_train, y_train)

    y_train_orig[:,n] = y_train
    y_test_orig[:,n] = y_test
    y_test_rf[:,n] = regressor.predict(X_test)
    y_train_rf[:,n] = regressor.predict(X_train)
    train_score[n] = regressor.score(X_train, y_train)
    test_score[n] = regressor.score(X_test, y_test)

# Plot
plt.close('all')
plt.scatter(y_train_orig, y_train_rf,
            c="orange", s=20, marker="s", alpha=0.5,
            label="Training Data, RF score=%.2f" % np.average(train_score))
plt.scatter(y_test_orig, y_test_rf,
            c="b", s=20, marker="^", alpha=1,
            label="Test Data, RF score=%.2f" % np.average(test_score))
plt.xlabel('Measured flux', fontsize=20)
plt.ylabel('Estimated flux', fontsize=20)
plt.xlim((-0.01, 0.04))
plt.ylim((-0.01, 0.04))
plt.legend(loc='upper left')

# eof
