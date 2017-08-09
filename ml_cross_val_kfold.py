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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import importlib

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import KFold

import site_metadata_compiler_completed as comp

importlib.reload(comp)

#Datasets to pull from
database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
site_metadata = comp.comp(database, metadata, site_info, hole_info)
ml_train = site_metadata
#ml_train = ml_train[ml_train['interface_flux'].astype(float) <= 0.04]
#ml_train = ml_train[ml_train['interface_flux'].astype(float) >= -0.01]
oc_burial = ml_train['sed_rate_combined'].astype(float)*ml_train['toc_wood'].astype(float)

X = pd.concat((ml_train[['etopo1_depth', 'surface_porosity',
                         'surface_productivity',
                         'woa_temp', 'woa_salinity', 'woa_o2',
                         'acc_rate_archer','toc_wood',
                         'sed_rate_combined']], oc_burial), axis=1)

"""
X = ml_train[['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity',
  'woa_temp', 'woa_salinity', 'woa_o2',
  'acc_rate_archer','toc_wood',
  'sed_rate_combined','lithology']]
"""
X = np.array(X)

y = ml_train['interface_flux'].astype(float)
y = np.array(y)  # .reshape(-1,1)

# Create a random dataset

n_splits = len(ml_train)
y_predicted = np.empty(len(ml_train))
cv = KFold(n_splits=n_splits)

# Define regressor object
"""
regressor = GradientBoostingRegressor(loss='ls',n_estimators=120,
                                      learning_rate=0.1,
                                      min_samples_leaf=9,
                                      criterion='friedman_mse')
"""
"""
regressor = RandomForestRegressor(n_estimators=40,
                                n_jobs=-1,
                                min_samples_leaf=2,
                                criterion = 'friedman_mse')
"""

regressor = linear_model.LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)


for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    regressor.fit(X_train, y_train)

    y_predicted[test_idx] = regressor.predict(X_test)

slope, intercept, r_value, p_value, std_err = linregress(y, y_predicted)
print('r_squared:', r_value**2)
r_squared = r_value**2

regressor.fit(X, y)
y_predicted_all = regressor.predict(X)
slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(y, y_predicted_all)
r_squared_all = r_value_all**2

# feature_imp = regressor.feature_importances_

# Plot
plt.close('all')
plt.scatter(y, y_predicted_all,
            c="orange", s=20, marker="s", alpha=0.7,
            label="Training score, r-squared=%.3f" % np.average(r_squared_all))
plt.scatter(y, y_predicted,
            c="b", s=20, marker="^", alpha=1,
            label="Cross-validation, r-squared=%.3f" % r_squared)
plt.xlabel('Measured flux', fontsize=20)
plt.ylabel('Estimated flux', fontsize=20)
plt.xlim((-0.01, 0.035))
plt.ylim((-0.01, 0.035))
plt.legend(loc='upper left', framealpha=0.1)
plt.show()

plt.savefig('mlr_cv.png', facecolor='none')
# eof
