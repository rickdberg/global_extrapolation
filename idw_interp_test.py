# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:28 2017

@author: rickdberg
"""

import numpy as np
import matplotlib.pyplot as plt

from geopy.distance import great_circle

from site_metadata_compiler_completed import comp


database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load hole data
site_metadata = comp(database, metadata, site_info, hole_info)
site_metadata = site_metadata[site_metadata['advection'].astype(float) >= 0]

# Get coordinates of porosity grid, which all others will be matched to
lat = np.flipud(np.arange(-89.5, 90.5, 1))
lon = np.arange(-179.5, 180.5, 1)

fluxes = site_metadata['interface_flux'].astype(float).as_matrix()
site_coords = np.array(site_metadata[['lat', 'lon']])
idw = np.empty((len(lat), len(lon)))
for n in np.arange(len(lon)):
    for m in np.arange(len(lat)):
        grid_coords = (lat[m], lon[n])
        distances = np.empty([len(site_coords)])
        for i in np.arange(len(site_coords)):
            distances[i] = great_circle(grid_coords,site_coords[i,:]).meters
        sorted_cut_idx = np.argsort(distances)[:6]
        weights = 1/distances[sorted_cut_idx]
        weights /= weights.sum(axis=0)
        idw[m,n] = np.dot(weights, fluxes[sorted_cut_idx])
    print(n)

np.savetxt('idw_grid_test.txt', idw, delimiter='\t')

# View datasets
plt.close('all')
plt.imshow(idw, cmap='plasma')
plt.scatter(site_coords[:,1]+180, abs(site_coords[:,0]-90), c=fluxes)
plt.show = lambda : None  # prevents showing during doctests
plt.ion()
plt.show()


# eof
