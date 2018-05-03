# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:47:32 2017
@author: rickdberg

Module for K-FOld Cross-Validation for selection of various machine
learning algorithms

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from scipy import stats
from matplotlib import mlab
import matplotlib as mpl
from pylab import savefig
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor)

import site_metadata_compiler_completed as comp
from user_parameters import (engine, metadata_table,
                             site_info, hole_info, ml_ouputs_path)
importlib.reload(comp)


# Define regression method and features
solute = 'Si'
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

# Load training and feature data
site_metadata = comp.comp(engine, metadata_table, site_info, hole_info)
ml_train = site_metadata
ml_train['interface_flux'] = ml_train['interface_flux'].astype(float)

X = ml_train[features]
y = ml_train['interface_flux'].astype(float)
X = np.array(X)
y = np.array(y)

# Define number of splits and kfold cross-validation method
n_splits = len(ml_train)
cv = KFold(n_splits=n_splits)

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

# Compile predicted values from all runs for LOO Analysis
y_predicted = np.empty(len(ml_train))
errors = np.empty(len(ml_train))

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    regressor.fit(X_train, y_train)
    y_predicted[test_idx] = regressor.predict(X_test)

# Get feature importances (or coefficients for MLR)
if meth in ('gbr', 'rf', 'abr'):
    feature_imp = regressor.feature_importances_
elif meth == 'mlr':
    feature_imp = regressor.coef_

# Cross-validation statistics
def rsq(modeled, measured):
    """
    Calculate the coefficient of determination (R-squared) value
    of a regression.

    modeled:  model predictions of dependent variable
    measured: measured values of dependent variable
    """
    yresid = measured - modeled
    sse = sum(yresid**2)
    sstotal = (len(measured) - 1) * np.var(measured, ddof=1)
    return 1 - sse / sstotal

def rmse(model_values, measured_values):
    return np.sqrt(((model_values-measured_values)**2).mean())

flux_rmse = rmse(y_predicted, y)
flux_res = y - y_predicted
r_squared = rsq(y_predicted, y)
print('r_squared:', r_squared)
print(np.std(abs(flux_res)))


"""
# Option to view the fit to all training data

regressor.fit(X, y)
y_predicted_all = regressor.predict(X)
r_squared_all = rsq(y_predicted_all, y)

# Plot fit to all data

plt.close('all')
fig1 = plt.figure(figsize=(8,8))
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
ax1.scatter(y, y_predicted_all,
            c="orange", s=20, marker="s", alpha=0.7,
            label="Training score, r-squared=%.3f" % np.average(r_squared_all))
ax1.scatter(y, y_predicted,
            c="b", s=20, marker="^", alpha=1,
            label="Cross-validation, r-squared=%.3f" % r_squared)
ax1.set_xlabel('Measured flux', fontsize=18)
ax1.set_ylabel('Estimated flux', fontsize=18)
ax1.set_xlim((-0.005, 0.04))
ax1.set_ylim((-0.005, 0.04))
ax1.legend(loc='upper left', framealpha=0.1)
fig1.show()
"""

# Cross-validation plot
plt.close('all')
fig1 = plt.figure(figsize=(3.1,3.1))
ax1 = fig1.add_axes([0.20,0.20,0.7,0.7])
ax1.scatter(y*1000, y_predicted*1000,
            c="k", s=3,
            label="Cross-validation, r-squared={}".format(np.round(r_squared,3)))
ax1.set_xlabel('$Measured\ flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=8)
ax1.set_ylabel('$Modeled\ flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=8)
ax1.set_xlim((-5, 40))
ax1.set_ylim((-5, 30))
ax1.tick_params(labelsize=8)
[left_raw, right_raw] = ax1.get_xlim()
[bottom_raw, top_raw] = ax1.get_ylim()
ax1.text((left_raw+(right_raw-left_raw)/10),
         (top_raw-(top_raw-bottom_raw)/10),
         '$R^2\ =\ {:.2f}$'.format(np.round(r_squared, 2)), fontsize=8)
fig1.show()
savefig(ml_ouputs_path + 'cross_validation_{}.pdf'.format(meth), dpi=1000)
savefig(ml_ouputs_path + 'cross_validation_{}.tif'.format(meth), dpi=1000)



# Residuals plot
residuals = flux_res*1000
residuals_standardized = (residuals - np.mean(residuals))/np.std(residuals)
test_stat, p_value = stats.kstest(residuals_standardized, 'norm')
fig2 = plt.figure(figsize=(3.1,2))
ax2 = fig2.add_axes([0.20,0.20,0.7,0.7])
n_1, bins_1, patches_1 = ax2.hist(residuals_standardized, bins='fd', normed=True, facecolor='white', linewidth=0.7)
ax2.set_ylabel('$Frequency\ density$', fontsize=8)
ax2.set_xlabel('$Standardized\ residuals$', fontsize=8)
ax2.set_xlim((-4, 4))
#ax2.set_ylim((0, 0.8))
ax2.tick_params(labelsize=8)
# Best fit normal distribution line to results
bf_line_1 = mlab.normpdf(bins_1, np.mean(residuals_standardized), np.std(residuals_standardized))
ax2.plot(bins_1, bf_line_1, 'k--', linewidth=1)
fig2.show()
savefig(ml_ouputs_path + 'residuals_{}.pdf'.format(meth), dpi=1000)
savefig(ml_ouputs_path + 'residuals_{}.tif'.format(meth), dpi=1000)

"""
# Heteroskedasticity plot
fig3 = plt.figure(figsize=(8,8))
ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
res_plot = plt.scatter(y_predicted*1000, residuals_standardized)
ax3.set_ylabel('$Standardized\ residuals$', fontsize=18)
ax3.set_xlabel('$Modeled\ flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=18)
fig3.show()

# Prediction error vs. flux
fig3 = plt.figure(figsize=(8,8))
ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
res_plot = plt.scatter(y_predicted*1000, abs(residuals))
m,b,rvalue, pvalue, stderr = sp.stats.linregress(y_predicted*1000, abs(residuals))
indy = np.arange(0, 30, 0.1)
ax3.plot(indy, (m*indy+b))
ax3.set_ylabel('$Residual\ magnitudes$', fontsize=18)
ax3.set_xlabel('$Modeled\ flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=18)
fig3.show()
"""

feature_importance_list = pd.concat((pd.Series(features),
                                     pd.Series(feature_imp)), axis=1)
feature_importance_list.to_csv(ml_ouputs_path + '{}_importances_{}.txt'.format(solute,meth), sep=',')

# eof
