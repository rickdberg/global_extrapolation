# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:28 2017

@author: rickdberg

Script for nearest-neighbors inverse-distance weighted linear interpolation of
flux values to be run on AWS EC2 instances
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress

from geopy.distance import great_circle

from site_metadata_compiler_completed import comp


database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load hole data
site_metadata = comp(database, metadata, site_info, hole_info)
site_metadata = site_metadata[site_metadata['advection'].astype(float) >= 0]

fluxes = site_metadata['interface_flux'].astype(float).as_matrix()
site_coords = np.array(site_metadata[['lat', 'lon']])
coords_train, coords_test, flux_train, flux_test = train_test_split(site_coords, fluxes, test_size=0.1)

cycles = 20
idw = np.empty((len(coords_test)*cycles, 2))
for c in np.arange(cycles):
    coords_train, coords_test, flux_train, flux_test = train_test_split(site_coords, fluxes, test_size=0.1)
    for n in np.arange(len(coords_test)):
        distances = np.empty([len(coords_train)])
        for i in np.arange(len(coords_train)):
            distances[i] = great_circle(coords_test[n],coords_train[i,:]).meters
        sorted_cut_idx = np.argsort(distances)[:6]
        weights = 1/distances[sorted_cut_idx]
        weights /= weights.sum(axis=0)
        idw[n+len(flux_test)*c,1] = np.dot(weights, flux_train[sorted_cut_idx])
        idw[n+len(flux_test)*c,0] = flux_test[n]
        print(n)

np.savetxt('idw_grid_cv.txt', idw, delimiter='\t')

slope, intercept, r_value, p_value, std_err = linregress(idw[:,0], idw[:,1])
print('r_squared:', r_value**2)
r_squared = r_value**2

# Plot
plt.close('all')
plt.scatter(idw[:,0], idw[:,1],
            c="b", s=20, marker="o", alpha=1)
plt.xlabel('Measured flux', fontsize=20)
plt.ylabel('Estimated flux', fontsize=20)
plt.xlim((-0.01, 0.04))
plt.ylim((-0.01, 0.04))
#plt.legend()
plt.show()
# eof
