# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:23:26 2017

@author: rickdberg

Create maps

"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from site_metadata_compiler_completed import comp
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from pylab import savefig
import cmocean

from user_parameters import (engine, metadata_table,
                             site_info, hole_info, std_grids_path,
                             ml_inputs_path, ml_outputs_path)

plt.close('all')

# Load site data
site_metadata = comp(engine, metadata_table, site_info, hole_info)
isodata = site_metadata[site_metadata['alpha'].notnull()]
site_lat = isodata['lat']
site_lon = isodata['lon']
epsilon = (isodata['alpha'].astype(float) - 1) * 1000

# Load coordinates
grid_lats = np.loadtxt(std_grids_path + "lats_std.txt"
, delimiter='\t')
grid_lons = np.loadtxt(std_grids_path + "lons_std.txt"
, delimiter='\t')

# Load mask
mask = np.loadtxt(std_grids_path + "continent_mask.txt"
, delimiter='\t')
mask = mask.astype('bool')

# Get template
f = rasterio.open(ml_inputs_path + "Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)
newaff = f.transform
top_left = f.transform * (0,0)
bottom_right = f.transform * (f.width, f.height)
lat_interval = (bottom_right[1]-top_left[1])/f.height
lon_interval = (bottom_right[0] - top_left[0])/f.width
lat = f.xy(0,0)[1] + np.arange(f.height)*lat_interval
lon = f.xy(0,0)[0] + np.arange(f.width)*lon_interval
lon[lon > 180] -= 360

# Load fractionation grid into template
fract = np.loadtxt(ml_outputs_path + 'mg_fract.txt'
, delimiter='\t')

rf = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=f.shape[0], width=f.shape[1],
                             count=1, dtype=fract.dtype,
                             crs='+proj=latlong', transform=f.transform)
f.close()
rf.write(fract, 1)
src = rf
rf.close()

# Read image into ndarray
im = src.read()

# transpose the array from (band, row, col) to (row, col, band)
im = np.transpose(im, [1,2,0])
im = im[:,:,0]

# Define fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Verdana'
mpl.rc('font',family='sans-serif')
mpl.rcParams['font.sans-serif'] = 'Verdana'
mpl.rcParams['font.cursive'] = 'Verdana'

# Create Figure
plt.close('all')
fig = plt.figure(figsize=(130/25.4,80/25.4))
ax = fig.add_axes([0.05,0.15,0.9,0.8])
m = Basemap(projection='moll',lon_0=0,resolution='c', ax=ax)
shp_info = m.readshapefile(r'C:\Users\rickdberg\Downloads\ne_10m_land_scale_rank',
                           'scalerank',
                           drawbounds=True, linewidth=0, color='0.6')

paths = []
for line in shp_info[4]._paths:
    paths.append(Path(line.vertices, codes=line.codes))

coll = PathCollection(paths,
                      linewidth=0,
                      facecolors='0.6',
                      edgecolors='0.6',
                      zorder=2)
m.fillcontinents(color='0.6', lake_color='0.6')

m.drawparallels(np.arange(-90.,90.,30.),
                color='0.5',
                linewidth=0.5,
                textcolor='k',
                labels=[0,1,0,0],
                dashes=(None,None),
                ax=ax,
                fontsize=10)
m.drawmeridians(np.arange(0.,360.,60.),
                color='0.5',
                linewidth=0.5,
                dashes=(None,None),
                ax=ax)
m.drawmapboundary(color='k', fill_color='white', linewidth=0.5)
ax.add_collection(coll)

coasts = m.drawcoastlines(zorder=1,color='0.6',linewidth=0)
coasts_paths = coasts.get_paths()

ipolygons = np.append(np.arange(78), np.arange(1)+147)
for ipoly in ipolygons:
    r = coasts_paths[ipoly]
    # Convert into lon/lat vertices
    polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                        r.iter_segments(simplify=False)]
    px = [polygon_vertices[i][0] for i in range(len(polygon_vertices))]
    py = [polygon_vertices[i][1] for i in range(len(polygon_vertices))]
    m.plot(px,py,linewidth=0.25,zorder=3,color='black')

g_lons, g_lats = m(grid_lons, grid_lats)
data1 = m.contourf(g_lons,
           g_lats,
           im,
           cmap=cmocean.cm.delta,
           levels=[-3,-2,-1,0,1,2],
           vmin=-3,
           vmax=3,
           ax=ax)
cbar = fig.colorbar(mappable=data1,
                orientation='horizontal',
                fraction=0.05,
                pad=0.03,
                ax=ax,
                shrink=0.7)

cbar.set_label(r'$\epsilon\ ^{^{\frac{26}{24}}}Mg\ (‰)$', fontsize=10)
cbar.ax.tick_params(labelsize=10)
# cbar.ax.xaxis.set_label_position('top')

# To add points
fname = isodata[['lon','lat']].as_matrix()
point_lon = fname[:,0]
point_lat = fname[:,1]

lons, lats = m(point_lon, point_lat)
data2 = m.scatter(lons,
          lats,
          c='#ffff00',
          s=50,
          ax=ax)

# plt.show()

savefig(ml_outputs_path + 'fract_map.pdf', dpi=1000)
savefig(ml_outputs_path + 'fract_map.eps', dpi=1000)

# eof
