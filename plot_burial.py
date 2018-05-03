# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Data explorer

Feature options:
['etopo1_depth', 'surface_porosity', 'sed_thickness_combined',
  'crustal_age','coast_distance', 'ridge_distance', 'seamount',
  'surface_productivity','toc_seiter', 'opal', 'caco3',
  'sed_rate_burwicz', 'woa_temp', 'woa_salinity', 'woa_o2',
  'caco3_archer','acc_rate_archer','toc_combined','toc_wood',
  'sed_rate_combined','lithology','lith1','lith2','lith3','lith4','lith5',
  'lith6','lith7','lith8','lith9','lith10','lith11','lith12',
  'lith13']

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from site_metadata_compiler_completed import comp
from pylab import savefig

from user_parameters import (engine, metadata_table,
                             site_info, hole_info, ml_outputs_path)

# Define fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Verdana'
mpl.rc('font',family='sans-serif')
mpl.rcParams['font.sans-serif'] = 'Verdana'
mpl.rcParams['font.cursive'] = 'Verdana'


# Load site data
data = comp(engine, metadata_table, site_info, hole_info)

###############################################################################
# Burial flux figure
y = data['burial_flux'].astype(float)*1000
x = data['interface_flux'].astype(float)*1000

size = 5
color = ''
burial_fig = plt.figure(figsize=(45/25.4,35/25.4))
ax1 = burial_fig.add_axes([0.2,0.2,0.7,0.7])
plt.scatter(x, y, s= size, c=color, linewidth=0.3)
oto = np.arange(-1,40,0.1)
plt.plot(oto, oto, '--', c='k')
plt.xlabel('$Total\ Flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=6, linespacing=0.5, labelpad=1)
plt.ylabel('$Burial\ Flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=6, linespacing=0.5, labelpad=1)
plt.text(x=26.5, y=27, s='1:1', fontsize=6, ha='right')
plt.xlim([-1,35])
plt.ylim([-1,35])
plt.tick_params(axis='both', which='major', labelsize=6)
plt.show()

savefig(ml_outputs_path + 'plot_burial.pdf', dpi=1000)
savefig(ml_outputs_path + 'plot_burial.eps', dpi=1000)


# eof
