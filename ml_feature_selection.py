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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from site_metadata_compiler_completed import comp
from sklearn.model_selection import ShuffleSplit


#Datasets to pull from
database = "mysql://root:neogene227@localhost/iodp_compiled"
directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)
ml_train = site_metadata

oc_burial = ml_train['sed_rate_combined'].astype(float)*ml_train['toc_wood'].astype(float)

X = pd.concat((ml_train[['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood',
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

# Define method estimator

# Cross validation with 50 iterations to get smoother mean test and train
# score curves, each time with 10% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=20, test_size=0.10)
"""
estimator = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1, min_samples_leaf=7
                                , criterion = 'friedman_mse')
"""
estimator = GradientBoostingRegressor(loss='ls',n_estimators=120,
                                      learning_rate=0.1,
                                      min_samples_leaf=9,
                                      criterion='friedman_mse')

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=estimator, step=1, cv=cv)  # , scoring='accuracy')
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
