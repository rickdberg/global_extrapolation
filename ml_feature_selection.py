# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:20:50 2017

@author: rickdberg

Random forest feature selection

['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood',
  'sed_rate_combined','lithology','lith1','lith2','lith3','lith4','lith5',
  'lith6','lith7','lith8','lith9','lith10','lith11','lith12',
  'lith13']

"""

# Feature selection

from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from site_metadata_compiler_completed import comp
from sklearn.model_selection import ShuffleSplit, KFold, LeaveOneOut

from user_parameters import (engine, metadata_table,
                             site_info, hole_info)

meth = 'mlr'
features = ['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood',
  'sed_rate_combined','lith1','lith2','lith3','lith4','lith5',
  'lith6','lith7','lith8','lith9','lith10','lith11','lith12',
  'lith13']

# Load site data
site_metadata = comp(engine, metadata_table, site_info, hole_info)
ml_train = site_metadata

# oc_burial = ml_train['sed_rate_combined'].astype(float)*ml_train['toc_wood'].astype(float)
X = ml_train[features]

X = np.array(X)


y = ml_train['interface_flux'].astype(float)
y = np.array(y)  # .reshape(-1,1)

# Define method estimator

# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 10% data randomly selected as a validation set.
n_splits = len(ml_train)
cv = LeaveOneOut()
cv = ShuffleSplit(n_splits=50, test_size=0.15)
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

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=regressor, step=1, cv=cv, n_jobs=-1)  # , scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Feature ranking:")
print(rfecv.support_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# eof
