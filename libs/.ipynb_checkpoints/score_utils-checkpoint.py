'''
A collection of functions for computing verification scores
-------------------------------------------------------
Content:
    - ETS()
    - freq_bias()
    - PIT_nan()
    - CRPS_1d_from_quantiles()
    - CRPS_1d()
    - CRPS_2d()
    - CRPS_1d_nan()
    - BS_binary_1d()
    - BS_binary_1d_nan()
    - score_bootstrap_1d()
    - bootstrap_confidence_intervals()
    - zonal_energy_spectrum_sph()
    
Yingkai Sha
ksha@ucar.edu
'''

import numba as nb
import numpy as np
import xarray as xr
import pyshtools

def ETS(TRUE, PRED):
    '''
    Computing Equitable Threat Score (ETS) from binary input and target.
    '''
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    TP_rnd = (TP+FN)*(TP+FP)/(TN+FP+FN+TP)
    return (TP-TP_rnd)/(TP+FN+FP-TP_rnd)

def freq_bias(TRUE, PRED):
    '''
    Computing frequency bias from binary input and target.
    '''
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    return (TP+FP)/(TP+FN)

def PIT_nan(fcst, obs, q_bins):
    '''
    Probability Integral Transform (PIT) of observations based on forecast
    '''
    obs = obs[~np.isnan(obs)]
    
    # CDF_fcst
    cdf_fcst = np.quantile(fcst, q_bins)
    
    # transforming obs to CDF_fcst 
    n_obs = np.searchsorted(cdf_fcst, obs)
    # an uniform distributed random variale
    p_obs = n_obs/len(q_bins)
    # estimate CDF_fcst(obs)
    p_bins = np.quantile(p_obs, q_bins)
    
    return p_bins
    
@nb.njit()
def CRPS_1d_from_quantiles(q_bins, CDFs, y_true):
    '''
    (experimental)
    Given quantile bins and values, computing CRPS from determinstic obs.
    
    Input
    ----------
        q_bins: quantile bins. `shape=(num_bins,)`.
        CDFs: quantile values corresponded to the `q_bins`.
              `shape=(num_bins, grid_points)`.
        y_true: determinstic true values. `shape=(obs_time, grid_points)`.
        
    Output
    ----------
        CRPS
        
    * `y_true` is 2-d. That said, one CDF paired for multiple obs.
      This is commonly applied for climatology CDFs vs. real-time obs. 
    
    '''
    
    L = len(q_bins)-1
    H_func = np.zeros((L+1,))
    d_bins = q_bins[1] - q_bins[0]
    
    N_days, N_grids = y_true.shape
    
    CRPS = np.empty((N_days, N_grids))
    
    for day in range(N_days):
        for n in range(N_grids):
            
            cdf = CDFs[:, n]
            obs = y_true[day, n]    
            step = np.searchsorted(cdf, obs)
            if step > L: step = L
                
            H_func[:] = 0.0
            H_func[step:] = 1.0
            
            CRPS[day, n] = np.trapz((q_bins-H_func)**2, x=cdf)
            #np.sum(np.diff(cdf)*(np.abs(q_bins-H_func)[:-1]))
    
    return CRPS

@nb.njit()
def CRPS_1d(y_true, y_ens):
    '''
    Given one-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    
    CRPS, MAE, pairwise_abs_diff = CRPS_1d(y_true, y_ens)
    
    Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A., 2006. The continuous ranked probability score 
    for circular variables and its application to mesoscale forecast ensemble verification. Quarterly Journal of 
    the Royal Meteorological Society, 132(621C), pp.2925-2942.
    
    Input
    ----------
        y_true: a numpy array with shape=(time, grids) and represents the (observed) truth 
        y_pred: a numpy array with shape=(time, ensemble_members, grids), represents the ensemble forecast
    
    Output
    ----------
        CRPS: the continuous ranked probability score, shape=(time, grids)
        MAE: mean absolute error
        SPREAD: pairwise absolute difference among ensemble members (not the spread)
        
    '''
    N_day, EN, N_grids = y_ens.shape
    M = 2*EN*EN
    
    # allocate outputs
    MAE = np.empty((N_day, N_grids),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, N_grids),); SPREAD[...] = np.nan
    
    # loop over grid points
    for n in range(N_grids):
        # loop over days
        for day in range(N_day):
            # calc MAE
            MAE[day, n] = np.mean(np.abs(y_true[day, n]-y_ens[day, :, n]))
            # calc SPREAD
            spread_temp = 0
            for en1 in range(EN):
                for en2 in range(EN):
                    spread_temp += np.abs(y_ens[day, en1, n]-y_ens[day, en2, n])
            SPREAD[day, n] = spread_temp/M
            
    CRPS = MAE-SPREAD
    
    return CRPS, MAE, SPREAD

@nb.njit()
def CRPS_2d(y_true, y_ens, land_mask=None):
    
    '''
    Given two-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    
    CRPS, MAE, pairwise_abs_diff = CRPS_2d(y_true, y_ens, land_mask='none')
    
    Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A., 2006. The continuous ranked probability score 
    for circular variables and its application to mesoscale forecast ensemble verification. Quarterly Journal of 
    the Royal Meteorological Society, 132(621C), pp.2925-2942.
    
    Input
    ----------
        y_true: a numpy array with shape=(time, gridx, gridy) and represents the (observed) truth.
        y_pred: a numpy array with shape=(time, ensemble_members, gridx, gridy), represents the ensemble forecast.
        land_mask: a numpy array with shape=(gridx, gridy). 
                   True elements indicate where CRPS will be computed.
                   Positions of False elements will be filled with np.nan
                   *if land_mask='none', all grid points will participate.
    
    Output
    ----------
        CRPS: the continuous ranked probability score, shape=(time, grids)
        MAE: mean absolute error
        SPREAD: pairwise absolute difference among ensemble members (not the spread)
        
    '''
    
    N_day, EN, Nx, Ny = y_ens.shape
    M = 2*EN*EN
    
    if land_mask is None:
        land_mask_ = np.ones((Nx, Ny)) > 0
    else:
        land_mask_ = land_mask
    
    # allocate outputs
    MAE = np.empty((N_day, Nx, Ny),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, Nx, Ny),); SPREAD[...] = np.nan
    
    # loop over grid points
    for i in range(Nx):
        for j in range(Ny):
            if land_mask_[i, j]:
                # loop over days
                for day in range(N_day):
                    # calc MAE
                    MAE[day, i, j] = np.mean(np.abs(y_true[day, i, j]-y_ens[day, :, i, j]))
                    # calc SPREAD
                    spread_temp = 0
                    for en1 in range(EN):
                        for en2 in range(EN):
                            spread_temp += np.abs(y_ens[day, en1, i, j]-y_ens[day, en2, i, j])
                    SPREAD[day, i, j] = spread_temp/M
    CRPS = MAE-SPREAD

    return CRPS, MAE, SPREAD

@nb.njit()
def CRPS_1d_nan(y_true, y_ens):
    '''
    Given one-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    np.nan will not propagate.
    
    CRPS, MAE, pairwise_abs_diff = CRPS_1d(y_true, y_ens)
    
    Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A., 2006. The continuous ranked probability score 
    for circular variables and its application to mesoscale forecast ensemble verification. Quarterly Journal of 
    the Royal Meteorological Society, 132(621C), pp.2925-2942.
    
    Input
    ----------
        y_true: a numpy array with shape=(time, grids) and represents the (observed) truth 
        y_pred: a numpy array with shape=(time, ensemble_members, grids), represents the ensemble forecast
    
    Output
    ----------
        CRPS: the continuous ranked probability score, shape=(time, grids)
        MAE: mean absolute error
        SPREAD: pairwise absolute difference among ensemble members (not the spread)
        
    '''
    N_day, EN, N_grids = y_ens.shape
    M = 2*EN*EN
    
    # allocate outputs
    MAE = np.empty((N_day, N_grids),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, N_grids),); SPREAD[...] = np.nan
    
    # loop over grid points
    for n in range(N_grids):
        # loop over days
        for day in range(N_day):
            # if obs is nan, then mark result as nan
            if np.isnan(y_true[day, n]):
                MAE[day, n] = np.nan
                SPREAD[day, n] = np.nan
            else:
                # calc MAE
                MAE[day, n] = np.mean(np.abs(y_true[day, n]-y_ens[day, :, n]))
                # calc SPREAD
                spread_temp = 0
                for en1 in range(EN):
                    for en2 in range(EN):
                        spread_temp += np.abs(y_ens[day, en1, n]-y_ens[day, en2, n])
                SPREAD[day, n] = spread_temp/M
            
    CRPS = MAE-SPREAD
    
    return CRPS, MAE, SPREAD



@nb.njit()
def BS_binary_1d(y_true, y_ens):
    '''
    Brier Score.
    
    BS_binary_1d(y_true, y_ens)
    
    ----------
    Hamill, T.M. and Juras, J., 2006. Measuring forecast skill: Is it real skill 
    or is it the varying climatology?. Quarterly Journal of the Royal Meteorological Society: 
    A journal of the atmospheric sciences, applied meteorology and physical oceanography, 132(621C), pp.2905-2923.
    
    Input
    ----------
        y_true: determinstic and binary true values. `shape=(obs_time, grid_points)`.
        y_ens: ensemble forecast. `shape=(time, ensemble_memeber, gird_points)`.
        
    Output
    ----------
        BS: Brier Score as described in Hamill and Juras (2006). 
        i.e., not scaled by `ensemble_memeber`, so can be applied for spatial-averaged analysis.
    
    '''
    
    N_days, EN, N_grids = y_ens.shape
    
    # allocation
    BS = np.empty((N_days, N_grids))

    # loop over initialization days
    for day in range(N_days):

        ens_vector = y_ens[day, ...]
        obs_vector = y_true[day, :]

        for n in range(N_grids):
            BS[day, n] = (obs_vector[n] - np.sum(ens_vector[:, n])/EN)**2

    return BS

@nb.njit()
def BS_binary_1d_nan(y_true, y_ens):
    '''
    Brier Score. np.nan will not propagate.
    
    BS_binary_1d_nan(y_true, y_ens)
    
    ----------
    Hamill, T.M. and Juras, J., 2006. Measuring forecast skill: Is it real skill 
    or is it the varying climatology?. Quarterly Journal of the Royal Meteorological Society: 
    A journal of the atmospheric sciences, applied meteorology and physical oceanography, 132(621C), pp.2905-2923.
    
    Input
    ----------
        y_true: determinstic and binary true values. `shape=(obs_time, grid_points)`.
        y_ens: ensemble forecast. `shape=(time, ensemble_memeber, gird_points)`.
        
    Output
    ----------
        BS: Brier Score as described in Hamill and Juras (2006). 
        i.e., not scaled by `ensemble_memeber`, so can be applied for spatial-averaged analysis.
    
    '''
    
    N_days, EN, N_grids = y_ens.shape
    
    # allocation
    BS = np.empty((N_days, N_grids))

    # loop over initialization days
    for day in range(N_days):

        ens_vector = y_ens[day, ...]
        obs_vector = y_true[day, :]

        for n in range(N_grids):
            if np.isnan(obs_vector[n]):
                BS[day, n] = np.nan
            else:
                BS[day, n] = (obs_vector[n] - np.sum(ens_vector[:, n])/EN)**2

    return BS


@nb.njit()
def score_bootstrap_1d(data, bootstrap_n=100):
    '''
    Bootstrapping all dimensions EXCEPT the last dimension of an array.
    
    Can be applied for the bootstrap replication of metrics.
    '''
    
    dim = data.shape[-1]
    temp = np.empty((dim, bootstrap_n))

    for i in range(dim):
        
        raw = data[..., i].ravel()
        flag_nan = np.logical_not(np.isnan(raw))
        raw = raw[flag_nan]
        L = np.sum(flag_nan)
        
        # bootstrap cycles
        for b in range(bootstrap_n):
            
            ind_bagging = np.random.choice(L, size=L, replace=True)
            temp[i, b] = np.mean(raw[ind_bagging])
            
    return temp



def bootstrap_confidence_intervals(rmse_t2m, 
                                   num_bootstrap_samples=1000, 
                                   lower_quantile=0.05, 
                                   upper_quantile=0.95,
                                   random_seed=None):
    """
    Compute confidence intervals over the 'day' dimension of rmse_t2m using bootstrapping.

    Parameters:
    - rmse_t2m: numpy array of shape (n_days, n_lead_times)
    - num_bootstrap_samples: int, number of bootstrap samples to draw
    - lower_percentile, upper_percentile: float, confidence intervals
    - random_seed: int or None, seed for the random number generator for reproducibility

    Returns:
    - ci_lower: numpy array of shape (n_lead_times,), lower bounds of the confidence intervals
    - ci_upper: numpy array of shape (n_lead_times,), upper bounds of the confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # call the numba-optimized function
    bootstrap_data = bootstrap_core(rmse_t2m, num_bootstrap_samples)
    
    # Compute confidence intervals outside numba
    ci_lower = np.quantile(bootstrap_data, lower_quantile, axis=0)
    ci_upper = np.quantile(bootstrap_data, upper_quantile, axis=0)
    mean_score = np.mean(bootstrap_data, axis=0)
    
    return mean_score, ci_lower, ci_upper

@nb.njit()
def bootstrap_core(rmse_t2m, num_bootstrap_samples):
    n_days, n_lead_times = rmse_t2m.shape
    bootstrap_data = np.empty((num_bootstrap_samples, n_lead_times))
    
    for i in range(num_bootstrap_samples):
        ind = np.random.randint(0, n_days)
        bootstrap_data[i, :] = rmse_t2m[ind, :]  # Shape: (n_days, n_lead_times)
        
    return bootstrap_data

def zonal_energy_spectrum_sph(ds_input: xr.Dataset, 
                              varname: str,
                              grid_type: str ='DH',
                              rescale=False)-> xr.DataArray:
    '''
    Computes the Zonal Energy Spectrum of a variable in an xarray.Dataset 
    using spherical harmonic transform. The output is rescaled by the 
    mean circumference per longitude grid.
    
    Parameters:
    - ds_input: xarray.Dataset containing the data.
    - varname: Name of the variable to compute the spectrum for.
    - grid_type: 'GLQ' or 'DH'
    - rescale: produce m * unit result based on circumference

    Returns:
    - spectrum: xarray.DataArray containing the zonal energy spectrum.
    '''
    RAD_EARTH = 6371000
    
    data = ds_input[varname]

    # check 'latitude' and 'longitude' cooridnate names
    if 'latitude' not in data.dims or 'longitude' not in data.dims:
        raise ValueError("Data must have 'latitude' and 'longitude' dimensions")
        
    latitudes = data['latitude'].values
    longitudes = data['longitude'].values

    # check latitudes for [90, -90] descending order
    # if not flip data and latitude
    if latitudes[0] < latitudes[-1]:
        data = data.isel(latitude=slice(None, None, -1))
        latitudes = data['latitude'].values
        
    # check longitudes for [0, 360] order
    # if not re-organize
    if np.any(longitudes < 0):
        longitudes = (longitudes + 360) % 360
        sorted_indices = np.argsort(longitudes)
        data = data.isel(longitude=sorted_indices)
        longitudes = data['longitude'].values

    # number of grids
    nlat = len(latitudes)
    nlon = len(longitudes)
    
    # max wavenumber is half of the latitude grids -1
    max_wavenum = (nlat - 1) // 2  # int divide
    
    # allocate zonal wavenumbers ranges
    zonal_wavenumbers = np.arange(max_wavenum + 1)

    def compute_power_m(data_array_2d):
        '''
        Computes the power spectrum for a 2D data array using spherical harmonics.

        Parameters:
        - data_array_2d: 2D numpy array of shape (nlat, nlon)

        Returns:
        - power_m: 1D numpy array of power corresponding to each zonal wavenumber m
        '''
        # initialize SHGrid
        grid = pyshtools.SHGrid.from_array(data_array_2d, grid=grid_type)
        
        # expand the grid to spherical harmonic coefs
        coeffs = grid.expand(normalization='ortho', lmax_calc=max_wavenum)

        # power per degree per order. shape=(lmax+1, lmax+1)
        coeffs_squared = coeffs.coeffs[0]**2 + coeffs.coeffs[1]**2
        
        # allocate power array for each zonal wavenumber m
        power_m = np.zeros(max_wavenum + 1)
        
        # sum over degrees l > m for each order m to get the total power
        # -l < m < l
        for l in range(max_wavenum + 1):
            power_m[l] = np.sum(coeffs_squared[l:, l])
        
        return power_m

    # xr.apply_ufunc scope
    spectrum = xr.apply_ufunc(
        compute_power_m,
        data,
        input_core_dims=[['latitude', 'longitude']],
        output_core_dims=[['zonal_wavenumber']],
        vectorize=True,
        dask='parallelized',  # <-- dask parallelization
        output_dtypes=[float],
    )

    # assign new coordinate 'zonal_wavenumber'
    spectrum = spectrum.assign_coords(zonal_wavenumber=zonal_wavenumbers)

    if rescale:
        # re-scale power spectrum based on the mean circumference per longitude
        cos_latitudes = np.cos(np.deg2rad(latitudes))
        normalization_factor = (RAD_EARTH * np.sum(cos_latitudes)) / nlon
        
        spectrum = spectrum * normalization_factor
    
    return spectrum
