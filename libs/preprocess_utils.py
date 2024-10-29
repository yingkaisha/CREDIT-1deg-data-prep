'''
preprocess_utils.py
-------------------------------------------------------
Content:
    - get_forward_data
    - zscore_var
    - residual_zscore_var

Yingkai Sha
ksha@ucar.edu
'''

import os
import yaml
import numpy as np
import xarray as xr
from glob import glob
from dask.distributed import Client
from dask_jobqueue import PBSCluster


def get_forward_data(filename) -> xr.DataArray:
    '''
    Check nc vs. zarr files
    open file as xr.Dataset
    '''
    if filename[-3:] == '.nc' or filename[-4:] == '.nc4':
        dataset = xr.open_dataset(filename)
    else:
        dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


def compute_mean_std_dask(
    base_path,
    project_num,
    output_path,
    input_files_pattern,
    output_mean_file_name,
    output_std_file_name):
    """
    Compute the mean and standard deviation of datasets using Dask for parallel computation.

    Parameters
    ----------
    base_path : str
        The common base path for input files (e.g., '/glade/derecho/scratch/wchapman').
    project_num : str
        The project number to be used for PBS cluster accounting.
    output_path : str
        The output directory path to save the computed results.
    input_files_pattern : str
        The pattern to match input files (e.g., 'SixHourly_y_ONEdeg_*').
    output_mean_file_name : str
        The filename for saving the computed mean result (e.g., 'V2_mean_time_6h_1979_2022_16lev_1deg.nc').
    output_std_file_name : str
        The filename for saving the computed standard deviation result (e.g., 'V2_std_time_6h_1979_2022_16lev_1deg.nc').

    Returns
    -------
    None
        The function saves the results as NetCDF files in the specified output path.
    """
    print('Starting mean and standard deviation computation...')
    print('Warning: BE PATIENT! This is a big operation, likely it won’t work on 1/4 degree data.')

    # Initialize Dask cluster
    cluster = PBSCluster(
        account=project_num,
        walltime='12:00:00',
        cores=1,
        memory='70GB',
        shared_temp_directory=os.path.join(base_path, 'tmp'),
        queue='casper'
    )
    cluster.scale(jobs=40)
    client = Client(cluster)
    # Print the URL for the Dask dashboard
    print(f"Dask dashboard available at: {client.dashboard_link}")


    # Define file paths
    input_files = os.path.join(base_path, input_files_pattern)

    # Load datasets
    FNS = sorted(glob.glob(input_files))
    combined_ds = xr.open_mfdataset(
        FNS,
        engine='zarr',  # Specify the Zarr engine
        concat_dim='time',  # Concatenate along the time dimension
        combine='nested',  # Combine along a specific dimension
        parallel=True  # Enable parallel loading with Dask
    )

    # Compute mean and standard deviation across time, latitude, and longitude
    mean_ds = combined_ds.mean(['time', 'lat', 'lon']).persist()
    std_ds = combined_ds.std(['time', 'lat', 'lon']).persist()

    # Trigger computation
    mean_ds.compute()
    std_ds.compute()

    # Save results to NetCDF
    output_mean_file_path = os.path.join(output_path, output_mean_file_name)
    output_std_file_path = os.path.join(output_path, output_std_file_name)
    
    mean_ds.to_netcdf(output_mean_file_path)
    std_ds.to_netcdf(output_std_file_path)

    print('Mean and standard deviation computation complete. Results saved.')


def resid_norm_dask(
    base_path, 
    project_num, 
    output_path, 
    input_files_pattern, 
    mean_file_name, 
    std_file_name, 
    output_mean_file_name_noceof, 
    output_var_file_name_noceof,
    output_var_file_name,
):
    """
    Compute the residual coefficient norm using Dask for parallel computation.

    Parameters
    ----------
    base_path : str
        The common base path for input files (e.g., '/glade/derecho/scratch/wchapman').
    project_num : str
        The project number to be used for PBS cluster accounting.
    output_path : str
        The output directory path to save the computed results.
    input_files_pattern : str
        The pattern to match input files (e.g., 'y_ONEdeg_*').
    mean_file_name : str
        The filename for the mean dataset (e.g., 'V2_mean_time_1h_1979_2022_16lev_1deg.nc').
    std_file_name : str
        The filename for the standard deviation dataset (e.g., 'V2_std_time_1h_1979_2022_16lev_1deg.nc').
    output_mean_file_name_noceof : str
        The filename for saving the computed mean result (e.g., 'resid_nocoef_mean_time_1h_1979_2022_16lev_1deg.nc').
    output_var_file_name_noceof : str
        The filename for saving the computed variance result (e.g., 'resid_nocoef_var_time_1h_1979_2022_16lev_1deg.nc').

    output_var_file_name : str
        The filename for saving the computed variance result (e.g., 'resid_var_time_1h_1979_2022_16lev_1deg.nc').


    Returns
    -------
    None
        The function saves the results as NetCDF files in the specified output path.
    """
    print('Warning: This is a big operation, likely it won’t work on 1/4 degree data.')

    # Initialize Dask cluster
    cluster = PBSCluster(
        account=project_num,
        walltime='12:00:00',
        cores=1,
        memory='70GB',
        shared_temp_directory=os.path.join(base_path, 'tmp'),
        queue='casper'
    )
    cluster.scale(jobs=40)
    client = Client(cluster)

    # Define file paths
    input_files = os.path.join(base_path, input_files_pattern)
    mean_file = os.path.join(base_path, mean_file_name)
    std_file = os.path.join(base_path, std_file_name)

    # Load dataset
    FNS = sorted(glob.glob(input_files))
    DS = xr.open_mfdataset(
        FNS,
        engine='zarr',  # Specify the Zarr engine
        concat_dim='time',  # Concatenate along the time dimension
        combine='nested',  # Combine along a specific dimension
        parallel=True  # Enable parallel loading with Dask
    )

    # Load mean and standard deviation datasets
    ds_mean = xr.open_dataset(mean_file)
    ds_std = xr.open_dataset(std_file)

    # Standardize the dataset
    DS = (DS - ds_mean) / ds_std
    DS = DS.diff("time")

    # Compute mean and variance across time, latitude, and longitude
    DS_mean_tot = DS.mean(['time', 'lat', 'lon']).persist()
    DS_var_tot = DS.var(['time', 'lat', 'lon']).persist()

    # Trigger computation
    DS_mean_tot.compute()
    DS_var_tot.compute()

    # Save results to NetCDF
    output_mean_file_path = os.path.join(output_path, output_mean_file_name_nocoef)
    output_var_file_path = os.path.join(output_path, output_var_file_name_nocoef)
    
    DS_mean_tot.to_netcdf(output_mean_file_path)
    DS_var_tot.to_netcdf(output_var_file_path)

    ds = xr.open_dataset(output_var_file_path)
    # Convert all variables to a single DataArray
    data_array = ds.to_array()
    # Flatten the DataArray to a 1D array
    flattened_array = data_array.values.flatten()
    #############
    # gmeands = gmean(flattened_array)
    std_g = gmean(np.sqrt(flattened_array))
    ds = np.sqrt(ds) / std_g
    ds.to_netcdf(os.path.join(output_path, output_var_file_name))


def zscore_var(conf, varname, ind_level=None, flag_float64=True, skipna=False):
    '''
    Compute mean and variance (can be converted to std) from yearly zarr or nc files
    It combines two yearly files iteratively using the pooling equations:
    https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    The function relies on one of the credit-data/data_preprocessing/data_config.yml

    This function works for all variables
    Input coordinates should be (time, level, longitude, latitude)
    '''
    # ------------------------------------------------------------------------------------ #
    # lists yearly files and open as xr.Dataset
    filenames = sorted(glob(conf['zscore'][varname]))
    
    year_range = conf['zscore']['years_range']
    train_years = [str(year) for year in range(year_range[0], year_range[1])]
    train_files = [file for file in filenames if any(year in file for year in train_years)]
    
    list_ds_train = []
    
    for fn in train_files:
        list_ds_train.append(get_forward_data(fn))
        
    # ------------------------------------------------------------------------------------ #
    # determine if the var has levels
    ds_example = list_ds_train[0][varname]
    var_shape = ds_example.shape
    N_grids = var_shape[-1] * var_shape[-2]
    mean_std_save = np.empty((2,))
    mean_std_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # loop thorugh files and compute mean and std 
    for i_fn, ds in enumerate(list_ds_train):
        
        # get the xr.Dataset per var per level
        if ind_level is not None:
            ds_subset = ds[varname].isel(level=ind_level)
        else:
            ds_subset = ds[varname]

        # use float64 for more accurate computation
        if flag_float64:
            ds_subset = ds_subset.astype('float64', copy=False)
        
        # get mean and var for the current year
        mean_current_yr = float(ds_subset.mean(skipna=skipna).compute())
        var_current_yr = float(ds_subset.var(skipna=skipna).compute())
        L = len(ds_subset) * N_grids
        
        print('{} - {}'.format(mean_current_yr, var_current_yr))

        if np.isnan(mean_current_yr):
            print('NaN found in {}'.format(train_files[i_fn]))
            print('variable name: {}'.format(varname))
            raise
        
        if i_fn == 0:
            # if it is the first year, pass current year to the combined 
            mean_std_save[0] = mean_current_yr
            mean_std_save[1] = var_current_yr
            N_samples = L
            
        else:
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            mean_new = (L * mean_current_yr + N_samples * mean_std_save[0]) / (L + N_samples)
            var_new = ((L - 1) * var_current_yr + (N_samples - 1) * mean_std_save[1]) / (L + N_samples - 1)
            var_new_adjust = (L * N_samples * (mean_current_yr - mean_std_save[0])**2) / (L + N_samples) / (L + N_samples -1)
            
            mean_std_save[0] = mean_new
            mean_std_save[1] = var_new + var_new_adjust
            N_samples = N_samples + L
            
            print('{} - {}'.format(mean_std_save[0], mean_std_save[1]))
            
    if ind_level is not None:
        save_name = conf['zscore']['save_loc'] + '{}_level{}_mean_std_{}.npy'.format(conf['zscore']['prefix'], ind_level, varname)
    else:
        save_name = conf['zscore']['save_loc'] + '{}_mean_std_{}.npy'.format(conf['zscore']['prefix'], varname)
        
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_save)

def residual_zscore_var(conf, varname, ind_level=None, flag_float64=False, skipna=False):
    '''
    Given yearly zarr or nc files, compute the zscore of a variable, apply 
    np.diff on its 'time' coordinate, and compute the mean and std the resulting np.diff outputs.

    This function works for all variables
    Input coordinates should be (time, level, longitude, latitude)
    '''

    filenames = sorted(glob(conf['residual'][varname]))
    
    year_range = conf['residual']['years_range']
    train_years = [str(year) for year in range(year_range[0], year_range[1])]
    train_files = [file for file in filenames if any(year in file for year in train_years)]
    
    list_ds_train = []
    
    for fn in train_files:
        list_ds_train.append(get_forward_data(fn))
        
    # ------------------------------------------------------------------------------------ #
    ds_example = list_ds_train[0][varname]
    var_shape = ds_example.shape
    
    N_grids = var_shape[-1] * var_shape[-2]
    mean_std_save = np.empty((2,))
    mean_std_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # mean, std
    ds_mean = xr.open_dataset(conf['residual']['mean_loc'])
    ds_std = xr.open_dataset(conf['residual']['std_loc'])
    
    if ind_level is not None:
        ds_mean = ds_mean.isel(level=ind_level)
        ds_std = ds_std.isel(level=ind_level)

    for i_fn, ds in enumerate(list_ds_train):
        
        # use float64 for more accurate computation
        if flag_float64:
            ds.astype('float64', copy=False)
        
        if ind_level is not None:
            ds = ds.isel(level=ind_level)
            
        # ===================================================================== #
        # apply np.diff
        var_diff = xr.apply_ufunc(
            np.diff,
            (ds[varname] - ds_mean[varname]) / ds_std[varname],
            input_core_dims=[['time']],
            output_core_dims=[['time_diff']],  # Change this to a new dimension name
            vectorize=True,
            dask='allowed',
            output_dtypes=[ds[varname].dtype]
        )
        
        ds_out = var_diff.to_dataset(name='{}_diff'.format(varname))
        
        ds_out = ds_out.assign_coords(
            time_diff=ds_out['time_diff'])

        
        # Determine the coordinate names to use for transposing


        #if "level" in ds_out.coords:
        #    if "latitude" in ds_out.coords and "longitude" in ds_out.coords:
        #        ds_out = ds_out.transpose("time_diff", "level", "latitude", "longitude")
        #    elif "lat" in ds_out.coords and "lon" in ds_out.coords:
        #        ds_out = ds_out.transpose("time_diff", "level", "lat", "lon")
        #    else:
        #        raise ValueError("Expected coordinate names 'latitude/longitude' or 'lat/lon' not found in dataset.")
        #else:
        if "latitude" in ds_out.coords and "longitude" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "latitude", "longitude")
        elif "lat" in ds_out.coords and "lon" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "lat", "lon")
        else:
            raise ValueError("Expected coordinate names 'latitude/longitude' or 'lat/lon' not found in dataset.")

                
        # ===================================================================== #
        # compute the mean and std from the np.diff result
        
        ds_subset = ds_out['{}_diff'.format(varname)]
        
        # assign float64 again to make sure
        if flag_float64:
            ds_subset = ds_subset.astype('float64', copy=False)
        
        # get mean and var for the current year
        mean_current_yr = float(ds_subset.mean(skipna=skipna).compute())
        var_current_yr = float(ds_subset.var(skipna=skipna).compute())
        
        L = len(ds_subset) * N_grids
        
        print('{} - {}'.format(mean_current_yr, var_current_yr))

        if np.isnan(mean_current_yr):
            print('NaN found in {}'.format(train_files[i_fn]))
            print('variable name: {}'.format(varname))
            raise
                    
        if i_fn == 0:
            # if it is the first year, pass current year to the combined 
            mean_std_save[0] = mean_current_yr
            mean_std_save[1] = var_current_yr
            N_samples = L
            
        else:
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            mean_new = (L * mean_current_yr + N_samples * mean_std_save[0]) / (L + N_samples)
            var_new = ((L - 1) * var_current_yr + (N_samples - 1) * mean_std_save[1]) / (L + N_samples - 1)
            var_new_adjust = (L * N_samples * (mean_current_yr - mean_std_save[0])**2) / (L + N_samples) / (L + N_samples -1)
            
            mean_std_save[0] = mean_new
            mean_std_save[1] = var_new + var_new_adjust
            N_samples = N_samples + L
            
            print('{} - {}'.format(mean_std_save[0], mean_std_save[1]))
    
    if ind_level is not None:
        save_name = conf['residual']['save_loc'] + '{}_level{}_mean_std_{}.npy'.format(conf['residual']['prefix'], ind_level, varname)
    else:
        save_name = conf['residual']['save_loc'] + '{}_mean_std_{}.npy'.format(conf['residual']['prefix'], varname)
        
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_save)


def residual_zscore_var_split_years(conf, varname, year, ind_level=None, flag_float64=True):
    '''
    same as residual_zscore_var, but for a single year
    '''
    
    filenames = sorted(glob(conf['residual'][varname]))
        
    year_range = [str(year),]
    train_file = [file for file in filenames if any(year in file for year in year_range)][0]
        
    ds = ds_train = get_forward_data(train_file)
        
    # ------------------------------------------------------------------------------------ #
    var_shape = ds_train[varname].shape
    
    N_grids = var_shape[-1] * var_shape[-2]
    mean_std_N_save = np.empty((3,))
    mean_std_N_save.fill(np.nan)

    # ------------------------------------------------------------------------------------ #
    # mean, std
    ds_mean = xr.open_dataset(conf['residual']['mean_loc'])
    ds_std = xr.open_dataset(conf['residual']['std_loc'])
    
    if ind_level is not None:
        ds_mean = ds_mean.isel(level=ind_level)
        ds_std = ds_std.isel(level=ind_level)
        
    # use float64 for more accurate computation
    if flag_float64:
        ds.astype('float64', copy=False)
    
    if ind_level is not None:
        ds = ds.isel(level=ind_level)
            
    # ===================================================================== #
    print('applying np.diff ...')
    # apply np.diff
    var_diff = xr.apply_ufunc(
        np.diff,
        (ds[varname] - ds_mean[varname]) / ds_std[varname],
        input_core_dims=[['time']],
        output_core_dims=[['time_diff']],  # Change this to a new dimension name
        vectorize=True,
        dask='allowed',
        output_dtypes=[ds[varname].dtype]
    )
    
    ds_out = var_diff.to_dataset(name='{}_diff'.format(varname))
    
    ds_out = ds_out.assign_coords(
        time_diff=ds_out['time_diff'])
    
    # Determine the coordinate names to use for transposing
    if "level" in ds_out.coords:
        if "latitude" in ds_out.coords and "longitude" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "level", "latitude", "longitude")
        elif "lat" in ds_out.coords and "lon" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "level", "lat", "lon")
        else:
            raise ValueError("Expected coordinate names 'latitude/longitude' or 'lat/lon' not found in dataset.")
    else:
        if "latitude" in ds_out.coords and "longitude" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "latitude", "longitude")
        elif "lat" in ds_out.coords and "lon" in ds_out.coords:
            ds_out = ds_out.transpose("time_diff", "lat", "lon")
        else:
            raise ValueError("Expected coordinate names 'latitude/longitude' or 'lat/lon' not found in dataset.")

    # ===================================================================== #
    # compute the mean and std from the np.diff result
    
    ds_subset = ds_out['{}_diff'.format(varname)]
    
    # assign float64 again to make sure
    if flag_float64:
        ds_subset = ds_subset.astype('float64', copy=False)
    
    # get mean and var for the current year
    mean_current_yr = float(ds_subset.mean(skipna=False).compute())
    var_current_yr = float(ds_subset.var(skipna=False).compute())
    
    L = len(ds_subset) * N_grids
        
    print('{} - {}'.format(mean_current_yr, var_current_yr))
    
    if np.isnan(mean_current_yr):
        print('NaN found in {}'.format(train_files[i_fn]))
        print('variable name: {}'.format(varname))
        raise
                    
    mean_std_N_save[0] = mean_current_yr
    mean_std_N_save[1] = var_current_yr
    mean_std_N_save[2] = L
    
    if ind_level is not None:
        save_name = conf['residual']['save_loc'] + 'backup_{}_level{}_mean_std_{}_y{}.npy'.format(
            conf['residual']['prefix'], ind_level, varname, year)
    else:
        save_name = conf['residual']['save_loc'] + 'backup_{}_mean_std_{}_y{}.npy'.format(
            conf['residual']['prefix'], varname, year)
        
    print('Save to {}'.format(save_name))
    np.save(save_name, mean_std_N_save)


