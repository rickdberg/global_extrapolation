# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:28 2017

@author: rickdberg

Module for applying machine learning method to flux data

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import savefig

from user_parameters import (ml_outputs_path)

meth = 'gbr'

# Define fonts
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Verdana'
mpl.rcParams['mathtext.it'] = 'Verdana'
mpl.rc('font',family='sans-serif')
mpl.rcParams['font.sans-serif'] = 'Verdana'
mpl.rcParams['font.cursive'] = 'Verdana'

margin_fluxes = np.loadtxt(ml_outputs_path + 'margin_fluxes_{}.csv'.format(meth), delimiter='\t')
abyssal_fluxes = np.loadtxt(ml_outputs_path + 'abyssal_fluxes_{}.csv'.format(meth), delimiter='\t')
mva_fig = plt.figure(figsize=(45/25.4, 35/25.4))
ax1 = mva_fig.add_axes([0.23,0.2,0.7,0.7])
plt.hist(abyssal_fluxes*1000,
         color='#006f00',
         alpha=0.8,
         normed=True,
         bins=100,
         label='>100 km',
         histtype='bar',
         edgecolor='black',
         linewidth=0.1)
plt.hist(margin_fluxes*1000,
         color='#a4ff48',
         alpha=0.65,
         normed=True,
         bins=100,
         label='<100 km',
         histtype='bar',
         edgecolor='black',
         linewidth=0.1)
plt.legend(fontsize=6, frameon=False)
plt.ylabel('$Probability\ Density$', fontsize=6, linespacing=0.5, labelpad=1)
plt.xlabel('$Mg^{2+}\ Flux\ (mmol\ m^{-2}\ y^{-1})$', fontsize=6, linespacing=0.5, labelpad=1)
plt.tick_params(axis='both', which='major', labelsize=6)
savefig(ml_outputs_path + 'plot_histogram_{}.pdf'.format(meth), dpi=1000)


# eof
