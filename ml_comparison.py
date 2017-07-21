# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:15:19 2017

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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from site_metadata_compiler_completed import comp
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import matplotlib.gridspec as gridspec
from learning_curve import plot_learning_curve
# import xgboost as xgb


#Datasets to pull from
database = "mysql://root:neogene227@localhost/iodp_compiled"
directory = r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"
lat_name = 'lat'
lon_name = 'lon'

# Load site data
site_metadata = comp(database, metadata, site_info, hole_info)
ml_train = site_metadata
# training_inputs = pd.read_csv(r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\training_inputs.csv",sep=',')
# idw = pd.read_csv('idw_interp.csv', sep=',')
# ml_train = pd.concat((site_metadata, training_inputs), axis=1)
# ml_train = pd.concat((ml_train, idw), axis=1)
oc_burial = ml_train['sed_rate'].astype(float)*ml_train['toc_combined'].astype(float)
X = pd.concat((ml_train[['etopo1_depth', 'surface_porosity',
                         'acc_rate_archer',
                         'sed_rate_combined','toc_combined'
              ]], oc_burial), axis=1)

X = np.array(X)


y = ml_train['interface_flux'].astype(float)
y = np.array(y)  # .reshape(-1,1)

# Create a random dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=3)

# Random Forest Regression
regr_rf = RandomForestRegressor(n_estimators=100,
                                n_jobs=-1, min_samples_leaf=7)
regr_rf.fit(X_train, y_train)
y_test_rf = regr_rf.predict(X_test)
y_train_rf = regr_rf.predict(X_train)

# Multiple Linear Regression
mlr = linear_model.LinearRegression(n_jobs=-1)
mlr.fit(X_train, y_train)
y_test_mlr = mlr.predict(X_test)
y_train_mlr = mlr.predict(X_train)

# Adaboost Regressor
regr_ab = AdaBoostRegressor(n_estimators=100, learning_rate=0.5)
regr_ab.fit(X_train, y_train)
y_test_ab = regr_ab.predict(X_test)
y_train_ab = regr_ab.predict(X_train)

# Gradient Boosting Regressor
regr_gb = GradientBoostingRegressor(n_estimators=28,
                                        min_samples_leaf=7,
                                        criterion='friedman_mse')
regr_gb.fit(X_train, y_train)
y_test_gb = regr_gb.predict(X_test)
y_train_gb = regr_gb.predict(X_train)

###############################################################################
# Plotting

# Set up axes and subplot grid
figure_1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(19, 7))
grid = gridspec.GridSpec(3, 8, wspace=0.7)
ax1 = plt.subplot(grid[0:3, :2])
ax1.grid()
ax2 = plt.subplot(grid[0:3, 2:4])
ax2.grid()
ax3 = plt.subplot(grid[0:3, 4:6])
ax3.grid()
ax4 = plt.subplot(grid[0:3, 6:8])
ax4.grid()


# Figure title
figure_1.suptitle("$Cross-validation\ analysis$", fontsize=26)
s = 50
a = 0.4


# Plot input data
ax1.scatter(y_train, y_train_rf,
            c="navy", s=s, marker="s", alpha=a,
            label="Training Data, RF score=%.2f" % regr_rf.score(X_train, y_train))
ax1.scatter(y_test, y_test_rf,
            c="c", s=s, marker="^", alpha=a,
            label="Test Data, RF score=%.2f" % regr_rf.score(X_test, y_test))

ax2.scatter(y_train, y_train_mlr,
            c="navy", s=s, marker="s", alpha=a,
            label="Training Data, RF score=%.2f" % mlr.score(X_train, y_train))
ax2.scatter(y_test, y_test_mlr,
            c="c", s=s, marker="^", alpha=a,
            label="Test Data, RF score=%.2f" % mlr.score(X_test, y_test))

ax3.scatter(y_train, y_train_ab,
            c="navy", s=s, marker="s", alpha=a,
            label="Training Data, RF score=%.2f" % regr_ab.score(X_train, y_train))
ax3.scatter(y_test, y_test_ab,
            c="c", s=s, marker="^", alpha=a,
            label="Test Data, RF score=%.2f" % regr_ab.score(X_test, y_test))

ax4.scatter(y_train, y_train_gb,
            c="navy", s=s, marker="s", alpha=a,
            label="Training Data, RF score=%.2f" % regr_gb.score(X_train, y_train))
ax4.scatter(y_test, y_test_gb,
            c="c", s=s, marker="^", alpha=a,
            label="Test Data, RF score=%.2f" % regr_gb.score(X_test, y_test))



# Additional formatting
ax1.legend(loc='best', fontsize='small')
ax2.legend(loc='best', fontsize='small')
ax3.legend(loc='best', fontsize='small')
ax4.legend(loc='best', fontsize='small')

ax1.set_ylabel('Predicted flux')
ax1.set_xlabel('Measured flux')
ax2.set_xlabel('Measured flux')
ax3.set_xlabel('Measured flux')
ax4.set_xlabel('Measured flux')

figure_1.show()

# eof
