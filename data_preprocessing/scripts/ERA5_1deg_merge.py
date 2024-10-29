'''
This script creates 1 deg ERA5 training data
for CREDIT stage 2

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import interp_utils as iu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# get year from input
year = int(args['year'])

base_dir_mlevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_1deg/'
base_dir_plevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_plevel_1deg/'
base_dir_output = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_1deg_stage1/'

fn_fmt_mlevel = base_dir_mlevel + 'upper_air/ERA5_mlevel_1deg_6h_{}_conserve.zarr'
fn_fmt_plevel = base_dir_plevel + 'all_in_one/ERA5_plevel_1deg_6h_{}_conserve.zarr'
fn_fmt_static = base_dir_plevel + 'static/ERA5_plevel_1deg_6h_conserve_static.zarr'
fn_mean_std = '/glade/derecho/scratch/ksha/CREDIT_data/mean_6h_1979_2018_16lev_0.25deg.nc'

mlevel_inds = [ 10,  30,  40,  50,  60,  70,  80,  90,  95, 100, 105, 110, 120, 130, 136, 137]
varnames_mlevel = ['specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'SP', 'VAR_2T']

var_mlevel = {
    'specific_humidity': mlevel_inds,
    'temperature': mlevel_inds,
    'u_component_of_wind': mlevel_inds,
    'v_component_of_wind': mlevel_inds
}

var_plevel = {
    'SP': None,
    'VAR_2T': None,
    'toa_incident_solar_radiation': None,
    'U': [500,], 
    'V': [500,], 
    'T': [500,], 
    'Q': [500,],
    'Z': [500,]
}

varnames_500 = ['U', 'V', 'T', 'Q', 'Z']

var_rename = {
    'U': 'U500',
    'V': 'V500',
    'T': 'T500',
    'Q': 'Q500',
    'Z': 'Z500',
    'specific_humidity': 'Q',
    'temperature': 'T',
    'u_component_of_wind': 'U',
    'v_component_of_wind': 'V',
    'VAR_2T': 't2m',
    'toa_incident_solar_radiation': 'tsi'}

chunk_size_3d = {
    'time': 10,
    'latitude': 181,
    'longitude': 360
}

chunk_size_4d = {
    'time': 10,
    'level': 16,
    'latitude': 181,
    'longitude': 360
}

encode_size_3d = dict(
    chunks=(
        chunk_size_3d['time'],
        chunk_size_3d['latitude'],
        chunk_size_3d['longitude']
    )
)

encode_size_4d = dict(
    chunks=(
        chunk_size_4d['time'],
        chunk_size_4d['level'],
        chunk_size_4d['latitude'],
        chunk_size_4d['longitude']
    )
)

ds_mean_std = xr.open_dataset(fn_mean_std)

ds_mlevel = xr.open_zarr(fn_fmt_mlevel.format(year))
ds_plevel = xr.open_zarr(fn_fmt_plevel.format(year))

ds_mlevel_sub = vu.ds_subset_everything(ds_mlevel, var_mlevel)
ds_plevel_sub = vu.ds_subset_everything(ds_plevel, var_plevel)
for var in varnames_500:
    ds_plevel_sub[var] = ds_plevel_sub[var].squeeze(dim="level")

ds_plevel_sub = ds_plevel_sub.drop_vars(['level',])
ds_merge = xr.merge([ds_mlevel_sub, ds_plevel_sub])
ds_merge = ds_merge.rename(var_rename)

ds_merge['level'] = ds_mean_std['level']

varnames = list(ds_merge.keys())
varname_4D = ['Q', 'T', 'U', 'V']

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_merge[var] = ds_merge[var].chunk(chunk_size_4d)
    else:
        ds_merge[var] = ds_merge[var].chunk(chunk_size_3d)

dict_encoding = {}

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **encode_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **encode_size_3d}

save_name = base_dir_output + 'all_in_one/ERA5_mlevel_1deg_6h_lev16_{}.zarr'.format(year)
ds_merge.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

