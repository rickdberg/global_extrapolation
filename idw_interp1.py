# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:28 2017

@author: rickdberg

Script for nearest-neighbors inverse-distance weighted linear interpolation of
flux values to be run on AWS EC2 instances
"""

import numpy as np

from geopy.distance import great_circle

from site_metadata_compiler_completed import comp

"""
from sshtunnel import SSHTunnelForwarder
server =  SSHTunnelForwarder(
     ('108.179.132.174', 3306),
     ssh_password="neogene227",
     ssh_username="root",
     remote_bind_address=('127.0.0.1', 3306))
server.start()
database = 'mysql+mysqldb://root:neogene227@127.0.0.1:%s/iodp_compiled' % server.local_bind_port
"""

database = "mysql+mysqlconnector://root:neogene227@108.179.132.174/iodp_compiled"
#database = "mysql://root:neogene227@localhost/iodp_compiled"
metadata = "metadata_mg_flux"
site_info = "site_info"
hole_info = "summary_all"

# Load hole data
site_metadata = comp(database, metadata, site_info, hole_info)

lat = np.loadtxt(
"/home/ubuntu/std_lat.txt"
, delimiter='\t')

lon = np.loadtxt(
"/home/ubuntu/std_lon.txt"
, delimiter='\t')



fluxes = site_metadata['interface_flux'].astype(float).as_matrix()
site_coords = np.array(site_metadata[['lat', 'lon']])
idw = np.empty((len(lat), len(lon)))

"""
Attempting to make it faster, worth it?
mask = np.loadtxt(
"/home/ubuntu/continent_mask.txt"
,delimiter='\t')
mask = mask.astype('bool')
idw_masked = idw[~mask]
"""


for n in np.arange(0,1000):
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

np.savetxt('idw_grid.txt', idw, delimiter='\t')


# eof
