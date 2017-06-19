# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg
Caclulate sed rate from sed thickness/crustal age
Units: m/y

"""

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from geopy.distance import great_circle
from mpl_toolkits.basemap import maskoceans

# Function for opening grid files and retrieving datasets into arrays
def rast(f_path):
    src = rasterio.open(f_path)
    nodata = src.nodata
    arr = src.read(1).astype(float)
    src.close()
    if nodata is None:
        nodata = np.nan
    arr[arr == nodata] = np.nan
    return arr

fine_shape = (2160,4320)
fine_aff = Affine(360/(4320), 0.0, -180.0,
                      0.0, -180/(2160), 90)
coarse_shape = (360,720)
coarse_aff = Affine(360/(720), 0.0, -180.0,
                      0.0, -180/(360), 90)

# Open all single layer gridded datasets
sed_thickness_laske = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Laske - sed thickness\sedmap.grd"
)
def sed_arrange(sed_thickness_laske):
    st_left = sed_thickness_laske[:,1:int((sed_thickness_laske.shape[1]+1)/2+1)]
    st_right = sed_thickness_laske[:,int((sed_thickness_laske.shape[1])/2+1):]
    return np.concatenate((st_right, st_left), axis=1)
sed_thickness_laske = sed_arrange(sed_thickness_laske)*1000

sed_thickness_whittaker = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Whittaker - sed thickness\sedthick_world_v2.grd"
)
def sed_arrange(sed_thickness_whittaker):
    top_filler = np.empty((112,sed_thickness_whittaker.shape[1])) * np.nan
    bottom_filler = np.empty((224,sed_thickness_whittaker.shape[1])) * np.nan
    sed_filled = np.concatenate((top_filler, sed_thickness_whittaker, bottom_filler))
    st_left = sed_filled[:,1:int((sed_filled.shape[1]+1)/2+1)]
    st_right = sed_filled[:,int((sed_filled.shape[1])/2+1):]
    return np.concatenate((st_right, st_left), axis=1)
sed_thickness_whittaker = sed_arrange(sed_thickness_whittaker)

sed_thickness_divins = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Divins - sed thickness\sedthick_world.grd"
)

crustal_age = rast(
r"C:\Users\rickdberg\Documents\UW Projects\Magnesium uptake\Data\ML Inputs\Muller - crustal age\age.3.2.nc"
)

# Resample all grids to match 5" pixel-registered porosity grid
reg_datasets = [sed_thickness_divins,sed_thickness_laske, crustal_age]

# Function for resampling datasets to be consistent with 5" pixel-registered porosity dataset
def resamp(dataset, resampler, newarr_shape, newaff):
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
gridded_data = np.empty((fine_shape[0], fine_shape[1], len(regs))) * np.nan  #+len(categories)))
for n in regs:
    newarr = resamp(reg_datasets[n], Resampling.bilinear, fine_shape, fine_aff)
    gridded_data[:,:,n] = newarr

# Make new full coverage sed thickness grid and calculate
# sed rate using crustal age, leaving out crust younger than 5My to reduce artifacts
sed_thickness = gridded_data[:,:,0]
sed_thickness[np.isnan(sed_thickness)] = gridded_data[:,:,1][np.isnan(sed_thickness)]
sed_thickness[sed_thickness <= 0] = np.nan
age = gridded_data[:,:,2]

sed_rate_calcd = np.empty(fine_shape) * np.nan
# sed_rate_calcd[~np.isnan(gridded_data[:,:,2])] = sed_thickness[~np.isnan(gridded_data[:,:,2])]/gridded_data[:,:,2][~np.isnan(gridded_data[:,:,2])]
sed_rate_calcd[age > 500] = sed_thickness[age > 500]/(age[age > 500]*10000)


# Resample to 0.5 degrees
sed_rate_calcd = resamp(sed_rate_calcd, Resampling.bilinear, coarse_shape, coarse_aff)

# Free up memory
for dataset in reg_datasets:
    del dataset

# Get coordinates of coarse grid
lat = np.arange(89.75, -90.25, -0.5)
lon = np.arange(-179.75, 180.25, 0.5)
coarse_lons, coarse_lats = np.meshgrid(lon, lat)
lons = coarse_lons.flatten()
lats = coarse_lats.flatten()

# Mask
locations = np.c_[lons, lats]
fakedata = np.empty(len(locations))
ocmask = maskoceans(locations[:,0],locations[:,1], fakedata,
                    inlands=False, resolution='l')
coarse_mask = np.ma.reshape(ocmask, (len(lat),len(lon)))
coarse_mask = ~coarse_mask.mask

# Mask coordinate and sed rate arrays
coarse_lats_masked = np.ma.masked_array(coarse_lats,
                                     mask=coarse_mask,
                                     fill_value=np.nan)
coarse_lons_masked = np.ma.masked_array(coarse_lons,
                                     mask=coarse_mask,
                                     fill_value=np.nan)
sed_rate_masked = np.ma.masked_array(sed_rate_calcd,
                                     mask=coarse_mask,
                                     fill_value=np.nan)
sed_rate_masked.data[coarse_mask] = np.nan

sed_rate_masked_nans = sed_rate_masked[np.isnan(sed_rate_masked)]
sed_rate_masked_nonnans = sed_rate_masked[~np.isnan(sed_rate_masked)]

coarse_lats_masked_nans = coarse_lats_masked[np.isnan(sed_rate_masked)]
coarse_lons_masked_nans = coarse_lons_masked[np.isnan(sed_rate_masked)]
coarse_lats_masked_nonnans = coarse_lats_masked[~np.isnan(sed_rate_masked)]
coarse_lons_masked_nonnans = coarse_lons_masked[~np.isnan(sed_rate_masked)]

grid_coords = np.ma.vstack((coarse_lats_masked_nans,
                         coarse_lons_masked_nans)).T
site_coords = np.ma.vstack((coarse_lats_masked_nonnans,
                         coarse_lons_masked_nonnans)).T
sites = np.ma.empty((len(site_coords),3))
sites[:,0] = site_coords[:,0]
sites[:,1] = site_coords[:,1]
sites[:,2] = sed_rate_masked_nonnans


for n in np.arange(len(grid_coords)):
    try:
        near_sites = sites[sites[:,0] < grid_coords[n,0]+20]
        near_sites = near_sites[near_sites[:,0] > grid_coords[n,0]-20]
        near_sites = near_sites[near_sites[:,1] < grid_coords[n,1]+15]
        near_sites = near_sites[near_sites[:,1] > grid_coords[n,1]-15]
        distances = np.empty([len(near_sites)])
        for i in np.arange(len(near_sites)):
            distances[i] = great_circle(grid_coords[n],near_sites[i,:2]).meters
            try:
                nearest = np.argsort(distances)[0]
                sed_rate_masked_nans[n] = near_sites[nearest,2]
            except IndexError:
                pass
    except IndexError:
        pass
    print(n)

sed_rate_masked[np.isnan(sed_rate_masked)] = sed_rate_masked_nans


np.savetxt('sed_rate_masked.txt', sed_rate_masked.data, delimiter='\t')

# Plot grid

# Load 'sed_rate', grid
fluxes = np.log(sed_rate)

woas = rasterio.open('rf.nc', 'w', driver='GMT',
                             height=coarse_shape[0], width=coarse_shape[1],
                             count=1, dtype=fluxes.dtype,
                             crs='+proj=latlong', transform=coarse_aff)
woas.write(fluxes, 1)
src = woas
woas.close()
title = '$Sedimentation\ rate\ long-term\ (m/y)$'

# Read image into ndarray
im = src.read()
# transpose the array from (band, row, col) to (row, col, band)
im = np.transpose(im, [1,2,0])
im = im[:,:,0]
xmin = src.transform[2]
xmax = src.transform[2] + src.transform[0]*src.width
ymin = src.transform[5] + src.transform[4]*src.height
ymax = src.transform[5]
#ax.set_global()

# define cartopy crs for the raster, based on rasterio metadata
crs = ccrs.PlateCarree()

# create figure
ax = plt.axes(projection=crs)
plt.title(title, fontsize=20)
ax.set_xmargin(0.05)
ax.set_ymargin(0.10)
# ax.stock_img()

# plot raster
plt.imshow(im, origin='upper', extent=[xmin, xmax, ymin, ymax],
           transform=crs, cmap="Greys")
#plt.contourf(coarse_lons, coarse_lats, im, transform=crs, cmap="Greys")

plt.colorbar(shrink=0.5)
#ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
#ax.add_feature(cartopy.feature.RIVERS)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='gray',
                  alpha=0.1, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.show()

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

sed_rate = resamp(sed_rate_masked, Resampling.bilinear, fine_shape, fine_aff)

np.savetxt('sed_rate_combined.txt', sed_rate, delimiter='\t')


# eof
