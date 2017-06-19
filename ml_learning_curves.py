# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:28 2017

@author: rickdberg

Module for running Random Forest Classifier on flux data

['etopo1_depth', 'surface_porosity', 'sed_thickness', 'crustal_age',
                'coast_distance', 'ridge_distance', 'seamount', 'surface_productivity',
                'toc', 'opal', 'caco3', 'woa_temp', 'woa_salinity', 'woa_o2',
                'caco3_archer','acc_rate_archer','lith1','lith2','lith3',
                'lith4','lith5','lith6','lith7','lith8',
                'lith9','lith10','lith11','lith12','lith13']

Best starting set:
[
 'lat','lon', 'etopo1_depth', 'surface_porosity',
 'sed_thickness', 'crustal_age','coast_distance',
 'ridge_distance', 'seamount', 'surface_productivity'
 ]


Objectively pared down:
 ['lat','lon', 'etopo1_depth',
  'coast_distance',
 'ridge_distance', 'woa_temp', 'woa_salinity',
 ]

"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from site_metadata_compiler_completed import comp
from sklearn.model_selection import ShuffleSplit
from learning_curve import plot_learning_curve


#Datasets to pull from
database = "mysql://root:neogene227@localhost/iodp_compiled"
directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)
ml_train = site_metadata
ml_train = ml_train[ml_train['advection'].astype(float) >= 0]

y = ml_train['interface_flux'].astype(float)
y = np.array(y)  # .reshape(-1,1)
X = ml_train[[
 'lat','lon', 'etopo1_depth','coast_distance'
 ]]
X = np.array(X)

# Plot Learning Curves
"""
title = "Learning Curves (Gradient Boosting)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=50, test_size=0.15, random_state=0)
estimator = GradientBoostingRegressor(random_state=1, min_samples_leaf=3)
plot_learning_curve(estimator, title, X, y, (0.0, 1.01), cv=cv, n_jobs=-1)
"""


title = "Learning Curves (Random Forest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 15% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.15)
estimator = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1, min_samples_leaf=3
                                , criterion = 'mse')
plot_learning_curve(estimator, title, X, y, (-0.2, 1.01), cv=cv, n_jobs=-1)
"""

title = "Learning Curves (AdaBoost)"
cv = ShuffleSplit(n_splits=50, test_size=0.15)
estimator = AdaBoostRegressor(n_estimators=100, learning_rate=0.75, loss='square')
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=-1)

"""
"""
title = "Learning Curves (Multiple linear regression)"
cv = ShuffleSplit(n_splits=50, test_size=0.1, random_state=0)
estimator = linear_model.LinearRegression(n_jobs=-1)
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=-1)
"""


# eof
