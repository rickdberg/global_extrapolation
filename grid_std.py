# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg
Standardize global gridded datasets to match grid size, extent, and orientation

"""

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


# Function for opening grid files and retrieving datasets into arrays
def rast(f_path):
    src = rasterio.open(f_path)
    nodata = src.nodata
    arr = src.read(1).astype(float)
    src.close()
    arr[arr == nodata] = np.nan
    return arr

newarr_shape = (2160,4320)

# Open all single layer gridded datasets
sed_thickness_laske = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Laske - sed thickness\sedmap.grd"
)
def sed_arrange(sed_thickness_laske):
    st_left = sed_thickness_laske[:,1:(sed_thickness_laske.shape[1]+1)/2+1]
    st_right = sed_thickness_laske[:,(sed_thickness_laske.shape[1])/2+1:]
    return np.concatenate((st_right, st_left), axis=1)
sed_thickness_laske = sed_arrange(sed_thickness_laske)*1000



toc_jahnke = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Jahnke - TOC burial\OrganicCDistribution.csv"
, delimiter=',')
def jahnke_arrange(toc_jahnke):
    st_left = toc_jahnke[:,:30]
    st_right = toc_jahnke[:,30:]
    cut = np.concatenate((st_right, st_left), axis=1)
    top_filler = np.empty((14,cut.shape[1])) * np.nan  # Only takes it to 89 degrees
    bottom_filler = np.empty((14,cut.shape[1])) * np.nan
    return np.concatenate((top_filler, cut, bottom_filler))
toc_jahnke = jahnke_arrange(toc_jahnke)

caco3_archer = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Archer - caco3\pct_CaCO3.coretop.cdf"
)
def caco3_arrange(caco3_archer):
    caco3_archer[caco3_archer > 100] = np.nan
    caco3_archer[caco3_archer < 0] = np.nan
    st_left = caco3_archer[:,:180]
    st_right = caco3_archer[:,180:]
    return np.concatenate((st_right, st_left), axis=1)
caco3_archer = caco3_arrange(caco3_archer)

acc_rate_archer = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Archer - accumulation rate\accum.coretop.cdf"
)

etopo1_depth = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Amante - etopo1\ETOPO1_Bed_g_gmt4.grd"
)

surface_porosity = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)

sed_thickness_whittaker = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Whittaker - sed thickness\sedthick_world_v2.grd"
)
def sed_arrange(sed_thickness_whittaker):
    top_filler = np.empty((112,sed_thickness_whittaker.shape[1])) * np.nan
    bottom_filler = np.empty((224,sed_thickness_whittaker.shape[1])) * np.nan
    sed_filled = np.concatenate((top_filler, sed_thickness_whittaker, bottom_filler))
    st_left = sed_filled[:,1:(sed_filled.shape[1]+1)/2+1]
    st_right = sed_filled[:,(sed_filled.shape[1])/2+1:]
    return np.concatenate((st_right, st_left), axis=1)
sed_thickness_whittaker = sed_arrange(sed_thickness_whittaker)

crustal_age = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Muller - crustal age\age.3.2.nc"
)

coast_distance = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0007-supinfo.grd"
)
def coast_fill(coast_distance):
    top_filler = np.empty((1,coast_distance.shape[1])) * np.nan
    bottom_filler = np.empty((1,coast_distance.shape[1])) * np.nan
    return np.concatenate((top_filler, coast_distance, bottom_filler))
coast_distance = coast_fill(coast_distance)


ridge_distance = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0006-supinfo.grd"
)
def ridge_fill(ridge_distance):
    top_filler = np.empty((12,ridge_distance.shape[1])) * np.nan
    bottom_filler = np.empty((12,ridge_distance.shape[1])) * np.nan
    return np.concatenate((top_filler, ridge_distance, bottom_filler))
ridge_distance = ridge_fill(ridge_distance)

seamount = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0005-supinfo.grd"
)

surface_productivity = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0008-supinfo.grd"
)

toc = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Seiter - TOC\TOC_Seiteretal2004.asc"
)
def toc_fill(toc):
    top_filler = np.empty((4,toc.shape[1])) * np.nan
    toc[toc > 100] = np.nan
    return np.concatenate((top_filler, toc))
toc = toc_fill(toc)

caco3 = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Seiter - TOC\calcite_seiteretal.2004.asc"
)

opal = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Seiter - TOC\opal_seiteretal.2004.asc"
)

lithology = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Dutkiewitcz - lithology\seabed_lithology_v1.nc"
)
categories = np.arange(13)+1

woa_temp = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\WOA - water temp, salinity\bottom_temp_original.csv"
, delimiter=',')
woa_temp = np.flipud(woa_temp)

woa_salinity = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\WOA - water temp, salinity\bottom_salintity_original.csv"
, delimiter=',')
woa_salinity = np.flipud(woa_salinity)

woa_o2 = np.loadtxt(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\WOA - water temp, salinity\bottom_o2_original.csv"
, delimiter=',')
woa_o2 = np.flipud(woa_o2)

# Get coordinates of porosity grid, which all others will be matched to
f = rasterio.open(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Martin - porosity productivity distances\grl53425-sup-0002-supinfo.grd"
)
newaff = f.transform
top_left = f.transform * (0,0)
bottom_right = f.transform * (f.width, f.height)
lat_interval = (bottom_right[1]-top_left[1])/f.height
lon_interval = (bottom_right[0] - top_left[0])/f.width
lat = f.xy(0,0)[1] + np.arange(f.height)*lat_interval
lon = f.xy(0,0)[0] + np.arange(f.width)*lon_interval
lon[lon > 180] -= 360
f.close()

###############################################################################
"""
# Save individual files
np.savetxt('etopo1_depth.txt.gz', etopo1_depth, delimiter='\t')
np.savetxt('surface_porosity.txt.gz', surface_porosity, delimiter='\t')
np.savetxt('sed_thickness_whittaker.txt.gz', sed_thickness_whittaker, delimiter='\t')
np.savetxt('crustal_age.txt.gz', crustal_age, delimiter='\t')
np.savetxt('coast_distance.txt.gz', coast_distance, delimiter='\t')
np.savetxt('ridge_distance.txt.gz', ridge_distance, delimiter='\t')
np.savetxt('seamount.txt.gz', seamount, delimiter='\t')
np.savetxt('surface_productivity.txt.gz', surface_productivity, delimiter='\t')
np.savetxt('toc.txt.gz', toc, delimiter='\t')
np.savetxt('caco3.txt.gz', caco3, delimiter='\t')
np.savetxt('opal.txt.gz', opal, delimiter='\t')
np.savetxt('lithology.txt.gz', lithology, delimiter='\t')
np.savetxt('caco3_archer.txt.gz', caco3_archer, delimiter='\t')
np.savetxt('acc_rate_archer.txt.gz', acc_rate_archer, delimiter='\t')
np.savetxt('sed_thickness_laske.txt.gz', sed_thickness_laske, delimiter='\t')

"""


# Resample all grids to match 5" pixel-registered porosity grid
reg_datasets = [etopo1_depth, surface_porosity, sed_thickness_whittaker, crustal_age,
                coast_distance, ridge_distance, seamount, surface_productivity,
                toc, opal, caco3, woa_temp, woa_salinity, woa_o2,
                caco3_archer, acc_rate_archer,sed_thickness_laske]
reg_datasets = [lithology]
# Function for resampling datasets to be consistent with 5" pixel-registered porosity dataset
def resamp(dataset, resampler, newarr_shape):
    arr = dataset
    height = arr.shape[0]
    width = arr.shape[1]
    newarr = np.empty(shape=newarr_shape)
    if arr.shape[1] % 2 == 0:
        aff = Affine(360/(width), 0.0, -180.0,
                      0.0, -180/(height), 90)
    else:
        lat_interval = 180/(height-1)
        lon_interval = 360/(width-1)
        aff = Affine(360/(width-1), 0.0, -180.0-lon_interval,
                      0.0, -180/(height-1), 90+lat_interval)
    reproject(
        arr, newarr,
        src_transform = aff,
        dst_transform = newaff,
        src_crs = {},
        dst_crs = {},
        resample = resampler)
    return newarr

regs = np.arange(len(reg_datasets))

# Resample continuous gridded datasets with bilinear fit
gridded_data = np.empty((newarr_shape[0], newarr_shape[1], len(regs)))  #+len(categories)))
for n in regs:
    newarr = resamp(reg_datasets[n], Resampling.bilinear, newarr_shape)
    gridded_data[:,:,n] = newarr


# Free up memory
for dataset in reg_datasets:
    del dataset

# Resample categorical datasets with nearest values
for num in categories:
    lith_resamp = resamp(lithology, Resampling.nearest, newarr_shape).astype(int)
    lith_resamp[lith_resamp != num] = 0
    lith_resamp[lith_resamp == num] = 1
    gridded_data[:,:,len(regs)+num-1] = lith_resamp

###############################################################################
"""
# Save standardized datasets
filenames = ['etopo1_depth', 'surface_porosity', 'sed_thickness_whittaker', 'crustal_age',
                'coast_distance', 'ridge_distance', 'seamount', 'surface_productivity',
                'toc', 'opal', 'caco3', 'woa_temp', 'woa_salinity', 'woa_o2',
                'caco3_archer', 'acc_rate_archer','sed_thickness_laske',
                'lith1','lith2','lith3',
                'lith4','lith5','lith6','lith7','lith8',
                'lith9','lith10','lith11','lith12','lith13']

filenames = ['lithology']
for n in np.arange(len(filenames)):
    np.savetxt(filenames[n]+'_std.txt', gridded_data[:,:,n], delimiter='\t')
"""



"""
# View datasets
plt.close('all')
plt.imshow(gridded_data[:,:,n], cmap='plasma')
plt.show = lambda : None  # prevents showing during doctests
plt.show()
"""

# eof
