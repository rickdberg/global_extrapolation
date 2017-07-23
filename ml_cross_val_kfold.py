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
y = np.array(y)  # .reshape(-1,1)

# Create a random dataset

n_splits = len(ml_train)
y_predicted = np.empty(n_splits)
cv = KFold(n_splits=n_splits)

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Random Forest Regression
    """
    regressor = GradientBoostingRegressor(n_estimators=110,
                                        min_samples_leaf=8,
                                        criterion='friedman_mse')

    """

    regressor = RandomForestRegressor(n_estimators=40,
                                    n_jobs=-1,
                                    min_samples_leaf=2,
                                    criterion = 'friedman_mse')


    """
    regressor = linear_model.LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
    """
    regressor.fit(X_train, y_train)

    y_predicted[test_idx] = regressor.predict(X_test)

slope, intercept, r_value, p_value, std_err = linregress(y, y_predicted)
print('r_squared:', r_value**2)
r_squared = r_value**2

regressor.feature_importances_

# Plot
plt.close('all')
plt.scatter(y, y_predicted,
            c="b", s=20, marker="^", alpha=1,
            label="Test Data, R-squared=%.4f" % r_squared)
plt.xlabel('Measured flux', fontsize=20)
plt.ylabel('Estimated flux', fontsize=20)
plt.xlim((-0.01, 0.035))
plt.ylim((-0.01, 0.035))
plt.legend(loc='upper left')
plt.show()
# eof
