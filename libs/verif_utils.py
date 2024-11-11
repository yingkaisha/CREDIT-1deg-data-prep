'''
A collection of functions used for the verification of CREDIT project
-----------------------------------------------------------------------
Content:
    - create_dir
    - get_nc_files
    - ds_subset_everything
    
Yingkai Sha
ksha@ucar.edu
'''

import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

def create_dir(path):
    """
    Create dir if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def lead_to_index(leads_exist, leads_verif):
    '''
    Check if a list of verified lead time exists in a list of all available lead times
    return the index of the verified lead time.
    '''
    ind_lead = []
    for lead_verif in leads_verif:
        if lead_verif in leads_exist:
            ind_lead.append(leads_exist.index(lead_verif))
        else:
            print('lead time {}h is not covered'.format(lead_verif))
            raise
    return ind_lead

def get_forward_data_netCDF4(filename) -> xr.DataArray:
    """Lazily opens netCDF4 files.
    """
    dataset = xr.open_dataset(filename)
    return dataset

def get_forward_data(filename) -> xr.DataArray:
    '''
    Lazily opens the Zarr store on gladefilesystem.
    '''
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset
    
def ds_subset_everything(ds, variables_levels, time_intervals=None):
    """
    Subset a given xarray.Dataset, preserve specific variable/level/time

    args:
        ds: xarray.Dataset
        variables_levels: a dictionary that looks like this
            variables_levels = {
                                'V500': None,  # Keep all levels
                                'SP': None,
                                't2m': None,
                                'U': [14, 10, 5],  
                                'V': [14, 10, 5], 
                            }
            Leave level as None if (1) keeping all levels or (2) the variable does not have level dim
        time_intervals: a time slice that applies to each variable (optional) 
    """
    # allocate the output xarray.Dataset
    ds_selected = xr.Dataset()

    # loop through the subset info
    for var, levels in variables_levels.items():
        if var in ds:
            if levels is None:
                # keep all level
                ds_selected[var] = ds[var]
            else:
                # subset levels
                ds_selected[var] = ds[var].sel(level=levels)
        else:
            print('variable {} does not exist in the given xarray.Dataset'.format(var))

    # optional time subset
    if time_intervals is not None:
        ds_selected = ds_selected.isel(time=time_intervals)
        
    return ds_selected

