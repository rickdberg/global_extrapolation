# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:39:14 2017

@author: rickdberg

Combine calculated metadata, site_info, and summary_all tables for sites with
calculated fluxes
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def comp(database, metadata, site_info, hole_info):
    engine = create_engine(database)

    # Load metadata
    sql = "SELECT * FROM {};".format(metadata)
    metadata = pd.read_sql(sql, engine)

    # Load site data
    sql = "SELECT * FROM {};".format(site_info)
    sitedata = pd.read_sql(sql, engine)

    # Load hole data
    sql = "SELECT * FROM {};".format(hole_info)
    holedata = pd.read_sql(sql, engine)
    # Group and average hole data for sites
    hole_grouped = holedata.loc[:,('site_key', 'lat','lon','water_depth',
                                   'total_penetration','etopo1_depth',
                                   'surface_porosity', 'sed_thickness_combined',
                                   'crustal_age','coast_distance',
                                   'ridge_distance', 'seamount',
                                   'surface_productivity','toc_seiter', 'opal',
                                   'caco3','sed_rate_burwicz', 'woa_temp',
                                   'woa_salinity', 'woa_o2','caco3_archer',
                                   'acc_rate_archer','toc_combined','toc_wood',
                                   'sed_rate_combined','lithology','lith1',
                                   'lith2','lith3','lith4','lith5','lith6',
                                   'lith7','lith8','lith9','lith10','lith11',
                                   'lith12','lith13'
                                   )].groupby("site_key").mean().reset_index()

    # Combine all tables
    site_meta_data = pd.merge(metadata, sitedata, how='outer', on=('site_key', 'leg', 'site'))
    data = pd.merge(site_meta_data, hole_grouped, how='outer', on=('site_key')).fillna(np.nan)
    site_metadata = data.dropna(subset = ['interface_flux']).reset_index(drop=True)
    site_metadata = site_metadata[site_metadata['advection'].astype(float) >= 0]
    #site_metadata = site_metadata[site_metadata['top_por'].astype(float) >= 0.35]
    site_metadata = site_metadata[site_metadata['datapoints'].astype(float) >= 3]
    site_metadata = site_metadata[site_metadata['site'] != '796']
    site_metadata = site_metadata[site_metadata['site'] != '791'] # Database Mg data different than in report
    site_metadata = site_metadata[site_metadata['site'] != '1130'] # Brine diffusion
    site_metadata = site_metadata[site_metadata['site'] != '813'] # COncentration curve has structure not reflected in data
    site_metadata = site_metadata[abs(site_metadata['stdev_flux'].astype(float)
                                      /site_metadata['interface_flux'].astype(float)) <= 10]
    return site_metadata

# eof
