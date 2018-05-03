# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:44:52 2018

@author: rickdberg
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:42:04 2017

@author: rickdberg

Fractionation figure and total fractionation calculation


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import savefig

from site_metadata_compiler_completed import comp
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
isodata = data[data['alpha'].notnull()]
epsilon_error = (isodata['alpha_stdev'].astype(float)) * 1000
epsilon = (isodata['alpha'].astype(float) - 1) * 1000
margin = [1,0,1,0,1,1,1,0,1,0,1,1]
margin_type = ['rise','abyssal','slope','abyssal','slope','slope','rise',
               'abyssal','slope','abyssal','slope','slope']
toc = [0.01,0.3,3.1,1.5,3.5,0.9,0.5,0.1,2.,1.9,0.5,1.2]
lith = ['calcareous ooze','lithogenic','lithogenic','siliceous ooze','lithogenic',
        'calcareous ooze','calcareous ooze','siliceous ooze','lithogenic','lithogenic',
        'lithogenic','lithogenic']
fit_label = ['calcareous ooze','sil','sil','sil','sil',
        'calcareous ooze','calcareous ooze','sil','sil','sil',
        'sil','sil']

###############################################################################

# Fractionation figure by toc and lithology
plt.close('all')
fract_fig, ax1 = plt.subplots(1, 1,
                             figsize=(50/25.4, 50/25.4),
                             facecolor='none', gridspec_kw={'wspace':0.2,
                                                            'top':0.95,
                                                            'bottom':0.15,
                                                            'left':0.20,
                                                            'right':0.95})

mpl.rc('text', usetex=False)
# mpl.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')
x = pd.concat((epsilon, epsilon_error), axis=1).reset_index(drop=True)
y = pd.Series(toc, name='y') # * isodata['sed_rate'].astype(float)*1000
# y = isodata['interface_flux'].astype(float)*100
size = 5

#df = pd.DataFrame(dict(x=x, y=y, label=lith))
df1 = pd.concat((pd.Series(lith, name='label'), x), axis=1)
df = pd.concat((df1,y), axis=1)
df_fit = pd.DataFrame(dict(x=x['alpha'], y=y, label=fit_label))
groups = df.groupby('label')
groups_fit = df_fit.groupby('label')
colors = ['#930093','#ff6cff','#ff6cff']
shapes = ['o','s','v']
colors_fit = ['#930093', '#ff6cff']
n = 0
for name, group in groups:
    xx  = group.alpha.reset_index(drop=True)
    yy = group.y.reset_index(drop=True)
    xx_err = group.alpha_stdev.reset_index(drop=True)
    err_plt = ax1.errorbar(x=xx, y=yy, xerr=xx_err, ecolor=colors[n], mfc=colors[n], mec='k', fmt=shapes[n], ms=size, label=name)
    n = n+1
ax1.set_clip_on(False)
n = 0
m = np.empty(len(colors_fit))
b = np.empty(len(colors_fit))
for name, group in groups_fit:
    m[n],b[n] = np.polyfit(group.x, group.y, 1)
    indy = np.arange(0, 4.1, 0.1)
    ax1.plot((indy-b[n])/m[n], indy, colors_fit[n])
    n = n + 1
# ax1.legend(loc=1, numpoints = 1, fontsize=6)
ax1.set_ylabel(r'$Total\ Organic\ Carbon\ (wt\%)$', fontsize=6, linespacing=0.5, labelpad=1)
ax1.set_xlabel(r'$\epsilon\ ^{^{\frac{26}{24}}}Mg\ (‰)$', fontsize=6, linespacing=0.5, labelpad=1)
ax1.set_ylim([-0.1,4])
ax1.set_xlim([-3,1.5])
ax1.locator_params(axis='x', nbins=5)
ax1.locator_params(axis='y', nbins=5)
ax1.tick_params(axis='both', which='major', labelsize=6)

fract_fig.show()
savefig(ml_outputs_path + 'fract_regression.pdf', dpi=1000)
savefig(ml_outputs_path + 'fract_regression.eps', dpi=1000)

# eof
