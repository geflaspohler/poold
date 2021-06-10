# Utility functions supporting experiments
import os
import warnings
import numpy as np
import pandas as pd
import netCDF4
# import xarray as xr
import subprocess
from datetime import datetime, timedelta
import collections
import itertools
import time
import sys
from filelock import FileLock
from functools import partial
from src.utils.general_util import printf, tic, toc

# Some global variables
# Forecast id to file name
forecast_id_to_fname_mapping = {
    "subx_cfsv2-precip": "subx-cfsv2-precip-all_leads-8_periods_avg",
    "subx_cfsv2-tmp2m": "subx-cfsv2-tmp2m-all_leads-8_periods_avg",
    "subx_geos_v2p1-precip": "subx-geos_v2p1-precip-all_leads-4_periods_avg",
    "subx_geos_v2p1-tmp2m": "subx-geos_v2p1-tmp2m-all_leads-4_periods_avg",
    "subx_nesm-precip": "subx-nesm-precip-all_leads-4_periods_avg",
    "subx_nesm-tmp2m": "subx-nesm-tmp2m-all_leads-4_periods_avg",
    "subx_ccsm4-precip": "subx-ccsm4-precip-all_leads-4_periods_avg",
    "subx_ccsm4-tmp2m": "subx-ccsm4-tmp2m-all_leads-4_periods_avg",
    "subx_cfsv2-precip-us": "subx-cfsv2-precip-all_leads-8_periods_avg-us",
    "subx_cfsv2-tmp2m-us": "subx-cfsv2-tmp2m-all_leads-8_periods_avg-us",
    "subx_geos_v2p1-precip-us": "subx-geos_v2p1-precip-all_leads-4_periods_avg-us",
    "subx_geos_v2p1-tmp2m-us": "subx-geos_v2p1-tmp2m-all_leads-4_periods_avg-us",
    "subx_nesm-precip-us": "subx-nesm-precip-all_leads-4_periods_avg-us",
    "subx_nesm-tmp2m-us": "subx-nesm-tmp2m-all_leads-4_periods_avg-us",
    "subx_ccsm4-precip-us": "subx-ccsm4-precip-all_leads-4_periods_avg-us",
    "subx_ccsm4-tmp2m-us": "subx-ccsm4-tmp2m-all_leads-4_periods_avg-us"
}
# Add nmme to mapping
nmme_ids = ["nmme{idx}-{var}-{target_horizon}".format(
    idx=idx, var=var, target_horizon=target_horizon) for idx, var, target_horizon in itertools.product(
        ["", "0"], ["prate", "tmp2m"], ["34w", "56w"])
]
forecast_id_to_fname_mapping.update({k: k for k in nmme_ids})


def pandas2file(df_to_file_func, out_file):
    """Writes pandas dataframe or series to file, makes file writable by all,
    creates parent directories with 777 permissions if they do not exist,
    and changes file group ownership to sched_mit_hill

    Args:
      df_to_file_func - function that writes dataframe to file when invoked,
        e.g., df.to_feather
      out_file - file to which df should be written
    """
    # Create parent directories with 777 permissions if they do not exist
    dirname = os.path.dirname(out_file)
    if dirname != '':
        os.umask(0)
        os.makedirs(dirname, exist_ok=True, mode=0o777)
    printf("Saving to "+out_file)
    with FileLock(out_file+'lock'):
        tic()
        df_to_file_func(out_file)
        toc()
    subprocess.call(f"rm {out_file}lock", shell=True)
    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)


def pandas2hdf(df, out_file, key="data", format="fixed"):
    """Write pandas dataframe or series to HDF; see pandas2file for other
    side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      key - key to use when writing to HDF
      format - format argument of to_hdf
    """
    pandas2file(partial(df.to_hdf, key=key, format=format, mode='w'), out_file)


def pandas2feather(df, out_file):
    """Write pandas dataframe or series to feather file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
    """
    pandas2file(df.to_feather, out_file)


def pandas2csv(df, out_file, index=False, header=True):
    """Write pandas dataframe or series to CSV file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      index - write index to file?
      header - write header row to file?
    """
    pandas2file(partial(df.to_csv, index=index, header=header), out_file)


def subsetlatlon(df, lat_range, lon_range):
    """Subsets df to rows where lat and lon fall into lat_range and lon_range

    Args:
       df: dataframe with columns 'lat' and 'lon'
       lat_range: range of latitude values, as xrange
       lon_range: range of longitude values, as xrange

    Returns:
       Subsetted dataframe
    """
    return df.loc[df['lat'].isin(lat_range) & df['lon'].isin(lon_range)]


def createmaskdf(mask_file):
    """Loads netCDF4 mask file and creates an equivalent dataframe in tall/melted
    format, with columns 'lat' and 'lon' and rows corresponding to (lat,lon)
    combinations with mask value == 1

    Args:
       mask_file: name of netCDF4 mask file

    Returns:
       Dataframe with one row for each (lat,lon) pair with mask value == 1
    """
    fh = netCDF4.Dataset(mask_file, 'r')
    # fh = xr.open_dataset(mask_file)
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:] + 360
    mask = fh.variables['mask'][:]
    lon, lat = np.meshgrid(lon, lat)
    # mask_df = pd.DataFrame({'lat': lat.flatten(),
    #                         'lon': lon.flatten(),
    #                         'mask': mask.flatten()})
    mask_df = pd.DataFrame({'lat': lat.flatten(),
                            'lon': lon.flatten(),
                            'mask': mask.data.flatten()})
    # Retain only those entries with a mask value of 1
    mask_df = mask_df.loc[mask_df['mask'] == 1]
    # Drop unnecessary 'mask' column
    return mask_df.drop('mask', axis=1)


def get_contest_mask():
    """Returns forecast rodeo contest mask as a dataframe

    Columns of dataframe are lat, lon, and mask, where mask is a {0,1} variable
    indicating whether the grid point should be included (1) or excluded (0).
    """
    return createmaskdf("data/masks/fcstrodeo_mask.nc")


def get_us_mask():
    """Returns contiguous U.S. mask as a dataframe

    Columns of dataframe are lat, lon, and mask, where mask is a {0,1} variable
    indicating whether the grid point should be included (1) or excluded (0).
    """
    return createmaskdf("data/masks/us_mask.nc")


def subsetmask(df, mask_df=get_contest_mask()):
    """Subsets df to rows with lat,lon pairs included in both df and mask_df

    Args:
        df: dataframe with columns 'lat' and 'lon'
        mask_df: dataframe created by createmaskdf

    Returns:
        Subsetted dataframe
    """
    return pd.merge(df, mask_df, on=['lat', 'lon'], how='inner')


def get_measurement_variable(gt_id, shift=None):
    """Returns measurement variable name for the given ground truth id

    Args:
       gt_id: ground truth data string accepted by get_ground_truth
       shift: (optional) Number of days by which ground truth measurements
           should be shifted forward
    """
    suffix = "" if shift is None or shift == 0 else "_shift"+str(shift)
    valid_names = ["tmp2m", "tmin", "tmax", "precip", "sst", "icec",
                   "mei", "mjo", "sce", "sst_2010", "icec_2010"]
    for name in valid_names:
        if gt_id.endswith(name):
            return name+suffix
    # for wind or hgt variables, measurement variable name is the same as the
    # gt id
    if "hgt" in gt_id or "uwnd" in gt_id or "vwnd" in gt_id:
        return gt_id+suffix
    # for NCEP/NCAR reanalysis surface variables, remove contest_ prefix and
    # take the first part of the variable name, before the first period
    if gt_id in ["contest_slp", "contest_pr_wtr.eatm", "contest_rhum.sig995",
                 "contest_pres.sfc.gauss", "contest_pevpr.sfc.gauss"]:
        return gt_id.replace("contest_", "").split(".")[0]+suffix
    elif gt_id in ["us_slp", "us_pr_wtr.eatm", "us_rhum.sig995",
                   "us_pres.sfc.gauss", "us_pevpr.sfc.gauss"]:
        return gt_id.replace("us_", "").split(".")[0]+suffix
    raise ValueError("Unrecognized gt_id "+gt_id)


def get_forecast_variable(gt_id):
    """Returns forecast variable name for the given ground truth id

    Args:
       gt_id: ground truth data string ending in "precip" or "tmp2m"
    """
    if gt_id.endswith("tmp2m"):
        return "tmp2m"
    if gt_id.endswith("precip"):
        return "prate"
    raise ValueError("Unrecognized gt_id "+gt_id)


def shift_df(df, shift=None, date_col='start_date', groupby_cols=['lat', 'lon'],
             rename_cols=True):
    """Returns dataframe with all columns save for the date_col and groupby_cols
    shifted forward by a specified number of days within each group

    Args:
       df: dataframe to shift
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting;
          if shift is None or shift == 0, original df is returned, unmodified
       date_col: (optional) name of datetime column
       groupby_cols: (optional) if all groupby_cols exist, shifting performed
          separately on each group; otherwise, shifting performed globally on
          the dataframe
       rename_cols: (optional) if True, rename columns to reflect shift
    """
    if shift is not None and shift != 0:
        # Get column names of all variables to be shifted
        # If any of groupby_cols+[date_col] do not exist, ignore error
        cols_to_shift = df.columns.drop(
            groupby_cols+[date_col], errors='ignore')
        # Function to shift data frame by shift and extend index

        def shift_grp_df(grp_df): return grp_df[cols_to_shift].set_index(
            grp_df[date_col]).shift(int(shift), freq="D")
        if set(groupby_cols).issubset(df.columns):
            # Shift ground truth measurements for each group
            df = df.groupby(groupby_cols).apply(shift_grp_df).reset_index()
        else:
            # Shift ground truth measurements
            df = shift_grp_df(df).reset_index()
        if rename_cols:
            # Rename variables to reflect shift
            df.rename(columns=dict(
                list(zip(cols_to_shift, [col+"_shift"+str(shift) for col in cols_to_shift]))),
                inplace=True)
    return df


def load_measurement(file_name, mask_df=None, shift=None):
    """Loads measurement data from a given file name and returns as a dataframe

    Args:
       file_name: name of HDF5 file from which measurement data will be loaded
       mask_df: (optional) mask dataframe of the form returned by subsetmask();
         if specified, returned dataframe will be restricted to those lat, lon
         pairs indicated by the mask
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting
    """
    # Load ground-truth data
    df = pd.read_hdf(file_name, 'data')

    # Convert to dataframe if necessary
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    # Replace multiindex with start_date, lat, lon columns if necessary
    if isinstance(df.index, pd.MultiIndex):
        df.reset_index(inplace=True)
    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        df = subsetmask(df, mask_df)

    # Return dataframe with desired shift
    return shift_df(df, shift=shift, date_col='start_date', groupby_cols=['lat', 'lon'])


def get_first_year(data_id):
    """Returns first year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or ground truth identifier
         accepted by get_ground_truth
    """
    if data_id.startswith("global"):
        return 2011
    if data_id.endswith("precip"):
        return 1948
    if data_id.startswith("nmme"):
        return 1982
    if data_id.endswith("tmp2m") or data_id.endswith("tmin") or data_id.endswith("tmax"):
        return 1979
    if "sst" in data_id or "icec" in data_id:
        return 1981
    if data_id.endswith("mei"):
        return 1979
    if data_id.endswith("mjo"):
        return 1974
    if data_id.endswith("sce"):
        return 1966
    if "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        return 1948
    if ("slp" in data_id or "pr_wtr" in data_id or "rhum" in data_id or
            "pres" in data_id or "pevpr" in data_id):
        return 1948
    if data_id.startswith("subx_cfsv2"):
        return 1999
    raise ValueError("Unrecognized data_id "+data_id)


def get_last_year(data_id):
    """Returns last year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or
         ground truth identifier accepted by get_ground_truth
    """
    return 2019


def get_ground_truth(gt_id, mask_df=None, shift=None):
    """Returns ground truth data as a dataframe

    Args:
       gt_id: string identifying which ground-truth data to return;
         valid choices are "global_precip", "global_tmp2m", "us_precip",
         "contest_precip", "contest_tmp2m", "contest_tmin", "contest_tmax",
         "contest_sst", "contest_icec", "contest_sce",
         "pca_tmp2m", "pca_precip", "pca_sst", "pca_icec", "mei", "mjo",
         "pca_hgt_{}", "pca_uwnd_{}", "pca_vwnd_{}",
         "pca_sst_2010", "pca_icec_2010", "pca_hgt_10_2010",
         "contest_rhum.sig995", "contest_pres.sfc.gauss", "contest_pevpr.sfc.gauss",
         "wide_contest_sst", "wide_hgt_{}", "wide_uwnd_{}", "wide_vwnd_{}",
         "us_tmp2m", "us_tmin", "us_tmax", "us_sst", "us_icec", "us_sce",
         "us_rhum.sig995", "us_pres.sfc.gauss", "us_pevpr.sfc.gauss"
       mask_df: (optional) see load_measurement
       shift: (optional) see load_measurement
    """
    gt_file = os.path.join("data", "dataframes", "gt-"+gt_id+"-14d.h5")
    printf(f"Loading {gt_file}")
    if gt_id.endswith("mei"):
        # MEI does not have an associated number of days
        gt_file = gt_file.replace("-14d", "")
    if gt_id.endswith("mjo"):
        # MJO is not aggregated to a 14-day period
        gt_file = gt_file.replace("14d", "1d")
    return load_measurement(gt_file, mask_df, shift)


def get_ground_truth_unaggregated(gt_id, mask_df=None, shifts=None):
    """Returns daily ground-truth data as a dataframe, along with one column
    per shift in shifts
    """
    first_year = get_first_year(gt_id)
    last_year = get_last_year(gt_id)
    gt_file = os.path.join("data", "dataframes",
                           "gt-"+gt_id+"-1d-{}-{}.h5".format(
                               first_year, last_year))
    gt = load_measurement(gt_file, mask_df)
    if shifts is not None:
        measurement_variable = get_measurement_variable(gt_id)
        for shift in shifts:
            # Shift ground truth measurements by shift for each lat lon and extend index
            gt_shift = gt.groupby(['lat', 'lon']).apply(
                lambda df: df[[measurement_variable]].set_index(df.start_date).shift(shift, freq="D")).reset_index()
            # Rename variable to reflect shift
            gt_shift.rename(columns={measurement_variable: measurement_variable +
                                     "_shift"+str(shift)}, inplace=True)
            # Merge into the main dataframe
            gt = pd.merge(gt, gt_shift, on=[
                          "lat", "lon", "start_date"], how="outer")
    return gt


def get_climatology(gt_id, mask_df=None):
    """Returns climatology data as a dataframe

    Args:
       gt_id: see load_measurement
       mask_df: (optional) see load_measurement
    """
    # Load global climatology if US climatology requested
    climatology_file = os.path.join("data", "dataframes",
                                    "official_climatology-"+gt_id+".h5")
    return load_measurement(climatology_file, mask_df)


def get_ground_truth_anomalies(gt_id, mask_df=None, shift=None):
    """Returns ground truth data, climatology, and ground truth anomalies
    as a dataframe

    Args:
       gt_id: see get_climatology
       mask_df: (optional) see get_climatology
       shift: (optional) see get_climatology
    """
    date_col = "start_date"
    # Get shifted ground truth column names
    gt_col = get_measurement_variable(gt_id, shift=shift)
    # Load unshifted ground truth data
    tic()
    gt = get_ground_truth(gt_id, mask_df=mask_df)
    toc()
    printf("Merging climatology and computing anomalies")
    tic()
    # Load associated climatology
    climatology = get_climatology(gt_id, mask_df=mask_df)
    if shift is not None and shift != 0:
        # Rename unshifted gt columns to reflect shifted data name
        cols_to_shift = gt.columns.drop(
            ['lat', 'lon', date_col], errors='ignore')
        gt.rename(columns=dict(
            list(zip(cols_to_shift, [col+"_shift"+str(shift) for col in cols_to_shift]))),
            inplace=True)
        unshifted_gt_col = get_measurement_variable(gt_id)
        # Rename unshifted climatology column to reflect shifted data name
        climatology.rename(columns={unshifted_gt_col: gt_col},
                           inplace=True)
    # Merge climatology into dataset
    gt = pd.merge(gt, climatology[[gt_col]],
                  left_on=['lat', 'lon', gt[date_col].dt.month,
                           gt[date_col].dt.day],
                  right_on=[climatology.lat, climatology.lon,
                            climatology[date_col].dt.month,
                            climatology[date_col].dt.day],
                  how='left', suffixes=('', '_clim')).drop(['key_2', 'key_3'], axis=1)
    clim_col = gt_col+"_clim"
    # Compute ground-truth anomalies
    anom_col = gt_col+"_anom"
    gt[anom_col] = gt[gt_col] - gt[clim_col]
    toc()
    printf("Shifting dataframe")
    tic()
    # Shift dataframe without renaming columns
    gt = shift_df(gt, shift=shift, rename_cols=False)
    toc()
    return gt


def in_month_day_range(test_datetimes, target_datetime, margin_in_days=0):
    """For each test datetime object, returns whether month and day is
    within margin_in_days days of target_datetime month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       test_datetimes: pandas Series of datetime.datetime objects
       target_datetime: target datetime.datetime object (must not be Feb. 29!)
       margin_in_days: number of days allowed between target
         month and day and test date month and day
    """
    # Compute target day of year in a year that is not a leap year
    non_leap_year = 2017
    target_day_of_year = pd.Timestamp(target_datetime.
                                      replace(year=non_leap_year)).dayofyear
    # Compute difference between target and test days of year
    # after adjusting leap year days of year to match non-leap year days of year;
    # This has the effect of treating Feb. 29 as the same date as Feb. 28
    leap_day_of_year = 60
    day_delta = test_datetimes.dt.dayofyear
    day_delta -= (test_datetimes.dt.is_leap_year &
                  (day_delta >= leap_day_of_year))
    day_delta -= target_day_of_year
    # Return true if test day within margin of target day when we account for year
    # wraparound
    return ((np.abs(day_delta) <= margin_in_days) |
            ((365 - margin_in_days) <= day_delta) |
            (day_delta <= (margin_in_days - 365)))


def month_day_subset(data, target_datetime, margin_in_days=0,
                     start_date_col="start_date"):
    """Returns subset of dataframe rows with start date month and day
    within margin_in_days days of the target month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       data: pandas dataframe with start date column containing datetime values
       target_datetime: target datetime.datetime object providing target month
         and day (will treat Feb. 29 like Feb. 28)
       start_date_col: name of start date column
       margin_in_days: number of days allowed between target
         month and day and start date month and day
    """
    if (target_datetime.day == 29) and (target_datetime.month == 2):
        target_datetime = target_datetime.replace(day=28)
    return data.loc[in_month_day_range(data[start_date_col], target_datetime,
                                       margin_in_days)]
    # return data.loc[(data[start_date_col].dt.month == target_datetime.month) &
    #                (data[start_date_col].dt.day == target_datetime.day)]


def load_forecast_from_file(file_name, mask_df=None):
    """Loads forecast data from file and returns as a dataframe

    Args:
       file_name: HDF5 file containing forecast data
       forecast_variable: name of forecasted variable (see get_forecast_variable)
       target_horizon: target forecast horizon
         ("34w" for 3-4 weeks or "56w" for 5-6 weeks)
       mask_df: (optional) see load_measurement
    """
    # Load forecast dataframe
    forecast = pd.read_hdf(file_name)

    # PY37
    if 'start_date' in forecast.columns:
        forecast.start_date = pd.to_datetime(forecast.start_date)
    if 'target_date' in forecast.columns:
        forecast.target_date = pd.to_datetime(forecast.target_date)

    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        forecast = subsetmask(forecast, mask_df)
    return forecast


def get_forecast(forecast_id, mask_df=None, shift=None):
    """Returns forecast data as a dataframe

    Args:
       forecast_id: forecast identifier of the form "{1}-{2}-{3}"
         where {1} is the forecast name in {nmme, nmme0, subx_cfsv2},
         {2} is the forecast variable (see get_forecast_variable),
         and {3} is the target forecast horizon in {34w, 56w}
       mask_df: (optional) see load_measurement
       shift: (optional) number of days by which ground truth measurements
         should be shifted forward; date index will be extended upon shifting
    """
    forecast_file = os.path.join("data", "dataframes",
                                 forecast_id_to_fname_mapping[forecast_id]+".h5")
    printf(f"Loading {forecast_file}")
    forecast = load_forecast_from_file(forecast_file, mask_df)

    if forecast_id.startswith("nmme0"):
        models = ['cancm3_0', 'cancm4_0', 'ccsm4_0', 'gfdl_0',
                  'gfdl-flor-a_0', 'gfdl-flor-b_0', 'cfsv2_0']
        forecast['nmme0_wo_ccsm3_nasa'] = forecast[models].mean(axis=1)
        forecast.drop(models, axis=1, inplace=True)
    elif forecast_id.startswith("nmme"):
        models = ['cancm3', 'cancm4', 'ccsm4', 'gfdl',
                  'gfdl-flor-a', 'gfdl-flor-b', 'cfsv2']
        forecast['nmme_wo_ccsm3_nasa'] = forecast[models].mean(axis=1)
        forecast.drop(models, axis=1, inplace=True)

    return shift_df(forecast, shift=shift,
                    groupby_cols=['lat', 'lon'])


def get_nmme(forecast_variable, target_horizon, mask_df=None):
    """Returns NMME forecast data as a dataframe

    Args:
       forecast_variable: name of forecasted variable (see get_forecast_variable)
       target_horizon: target forecast horizon
         ("34w" for 3-4 weeks or "56w" for 5-6 weeks)
       mask_df: (optional) see load_measurement
    """
    # Read in the contest bounding box NMME dataframe if it exists
    forecast_file = 'data/dataframes/nmme-{}-{}.h5'.format(
        forecast_variable, target_horizon)
    forecast = load_forecast_from_file(forecast_file, mask_df)
    return forecast


def get_contest_id(gt_id, horizon):
    """Returns contest task identifier string for the given ground truth
    identifier and horizon identifier

    Args:
       gt_id: ground truth data string ending in "precip" or "tmp2m" or
          belonging to {"prate", "apcp", "temp"}
       horizon: string in {"34w","56w","week34","week56"} indicating target
          horizon for prediction
    """
    # Map gt_id to standard contest form
    if gt_id.endswith("tmp2m") or gt_id == "temp":
        gt_id = "temp"
    elif gt_id.endswith("precip") or gt_id == "apcp" or gt_id == "prate":
        gt_id = "apcp"
    else:
        raise ValueError("Unrecognized gt_id "+gt_id)
    # Map horizon to standard contest form
    if horizon == "34w" or horizon == "week34":
        horizon = "week34"
    elif horizon == "56w" or horizon == "week56":
        horizon = "week56"
    else:
        raise ValueError("Unrecognized horizon "+horizon)
    # Return contest task identifier
    return gt_id+"_"+horizon


def get_contest_template_file(gt_id, horizon):
    """Returns name of contest template netcdf file for a given ground truth
    descriptor and horizon for prediction

    Args:
       gt_id: see get_contest_id
       horizon: see get_contest_id
    """
    return os.path.join("data", "fcstrodeo_nctemplates",
                        get_contest_id(gt_id, horizon)+"_template.nc")


def get_contest_netcdf(filename, contest_id, variable_name):
    """Loads a file that is in the same format as the contest submission templates
    (i.e., a netcdf file with variables 'lat, 'lon', and contest_id) and returns a
    pandas dataframe with columns 'lat', 'lon', and variable_name

    Args:
      filename: name of netcdf file
      contest_id: name of data variable in netcdf file (e.g., apcp_week34)
      variable_name: desired name of data variable in pandas dataframe
    """
    fh = netCDF4.Dataset(filename)
    # fh = xr.open_dataset(filename)
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]
    var = fh.variables[contest_id][:]
    lon, lat = np.meshgrid(lon, lat)
    df = pd.DataFrame({'lat': lat.flatten(),
                       'lon': lon.flatten(),
                       variable_name: var.flatten()})
    return df


def contest_pandas_to_netcdf(data, output_nc_file, gt_id, horizon):
    """Writes a pandas dataframe to a netcdf file copied from a template

    Args:
       data: pandas dataframe with three columns ("lat", "lon", and one
          representing measurement to be stored)
       output_nc_file: Output nc file name
       gt_id: see get_contest_id
       horizon: see get_contest_id

    Example usage:
       data = preds[preds.year == 2017][["lat","lon","ols_pred"]]
       output_nc_file = "deleteme.nc"
       gt_id = "contest_tmp2m"
       horizon = "34w"
       contest_pandas_to_netcdf(data, output_nc_file, gt_id, horizon)
    """
    # Get template file name from variable and horizon specification
    template_nc_file = get_contest_template_file(gt_id, horizon)

    # Copy contest netcdf template to output file
    command = 'cp {} {}'.format(template_nc_file, output_nc_file)
    subprocess.call(command, shell=True)

    # Open output file for reading and writing
    fh = netCDF4.Dataset(output_nc_file, 'r+')
    # fh = xr.open_dataset(output_nc_file)

    # Read in desired latitudes and longitudes
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]

    # Look up contest variable identifier
    contest_id = get_contest_id(gt_id, horizon)

    # Find weather variable in netcdf file
    weather_var = fh.variables[contest_id]

    # For each lat, lon pair
    for ii in range(len(lat)):
        for jj in range(len(lon)):
            # If data value exists, store it in netcdf object
            value = data[(data.lat == lat[ii]) & (data.lon == lon[jj])].drop(
                ["lat", "lon"], axis=1)
            if not value.empty:
                weather_var[ii, jj] = value.squeeze()

    # Write updated values to file
    fh.close()


def get_deadline_delta(target_horizon):
    """Returns number of days between official contest submission deadline date
    and start date of target period
    (0 for weeks 1-2 target, as it's 0 days away,
    14 for weeks 3-4 target, as it's 14 days away,
    28 for weeks 5-6 target, as it's 28 days away)

    Args:
       target_horizon: "12w", "34w", or "56w" indicating whether target period is
          weeks 1 & 2, weeks 3 & 4, or weeks 5 & 6
    """
    if target_horizon == "12w":
        deadline_delta = 0
    elif target_horizon == "34w":
        deadline_delta = 14
    elif target_horizon == "56w":
        deadline_delta = 28
    else:
        raise ValueError("Unrecognized target_horizon "+target_horizon)
    return deadline_delta


def get_forecast_delta(target_horizon, days_early=1):
    """Returns number of days between forecast date and start date of target period
    (deadline_delta + days_early, as we submit early)

    Args:
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
       days_early: how many days early is forecast submitted?
    """
    return get_deadline_delta(target_horizon) + days_early


def get_measurement_lag(data_id):
    """Returns the number of days of lag (e.g., the number of days over
    which a measurement is aggregated plus the number of days
    late that a measurement is released) for a given ground truth data
    measurement

    Args:
       data_id: forecast identifier beginning with "subx-cfsv2" or
         ground truth identifier accepted by get_ground_truth
    """
    # Every measurement is associated with its start date, and measurements
    # are aggregated over one or more days, so, on a given date, only the measurements
    # from at least aggregation_days ago are fully observed.
    # Most of our measurements require 14 days of aggregation
    aggregation_days = 14
    # Some of our measurements are also released a certain number of days late
    days_late = 0
    if data_id.endswith("mjo"):
        # MJO uses only a single day of aggregation and is released one day late
        aggregation_days = 1
        days_late = 1
    elif "sst" in data_id:
        # SST measurements are released one day late
        days_late = 1
    elif data_id.endswith("mei"):
        # MEI measurements are released at most 30 days late
        # (since they are released monthly) but are not aggregated
        aggregation_days = 0
        days_late = 30
    elif "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        # Wind / hgt measurements are released one day late
        days_late = 1
    elif "icec" in data_id:
        days_late = 1
    elif ("slp" in data_id or "pr_wtr.eatm" in data_id or "rhum.sig995" in data_id or
          "pres.sfc.gauss" in data_id or "pevpr.sfc.gauss" in data_id):
        # NCEP/NCAR measurements are released one day late
        days_late = 1
    elif data_id.startswith("subx_cfsv2"):
        # No aggregation required for subx cfsv2 forecasts
        aggregation_days = 0
    return aggregation_days + days_late


def get_start_delta(target_horizon, data_id):
    """Returns number of days between start date of target period and start date
    of observation period used for prediction. One can subtract this number
    from a target date to find the last viable training date.

    Args:
       target_horizon: see get_forecast_delta()
       data_id: see get_measurement_lag()
    """
    if data_id.startswith("nmme"):
        # Special case: NMME is already shifted to match target period
        return None
    return get_measurement_lag(data_id) + get_forecast_delta(target_horizon)


def get_target_date(deadline_date_str, target_horizon):
    """Returns target date (as a datetime object) for a given deadline date
    and target horizon

    Args:
       deadline_date_str: string in YYYYMMDD format indicating official
          contest submission deadline (note: we often submit a day before
          the deadline, but this variable should be the actual deadline)
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
    """
    # Get deadline date datetime object
    deadline_date_obj = datetime.strptime(deadline_date_str, "%Y%m%d")
    # Compute target date object
    return deadline_date_obj + timedelta(days=get_deadline_delta(target_horizon))


def df_merge(left, right, on=["lat", "lon", "start_date"], how="outer"):
    """Returns merger of pandas dataframes left and right on 'on'
    with merge type determined by 'how'. If left == None, simply returns right.
    """
    if left is None:
        return right
    else:
        return pd.merge(left, right, on=on, how=how)


def clim_merge(df, climatology, date_col="start_date",
               on=["lat", "lon"], how="left", suffixes=('', '_clim')):
    """Returns merger of pandas dataframe df and climatology on
    the columns 'on' together with the month and day indicated by date_col
    using merge type 'how' and the given suffixes.
    The date_col of clim is not preserved.
    """
    return pd.merge(df, climatology.drop(columns=date_col),
                    left_on=on + [df[date_col].dt.month,
                                  df[date_col].dt.day],
                    right_on=on + [climatology[date_col].dt.month,
                                   climatology[date_col].dt.day],
                    how=how, suffixes=suffixes).drop(['key_2', 'key_3'], axis=1)


def year_slice(df, first_year=None, date_col='start_date'):
    """Returns slice of df containing all rows with df[date_col].dt.year >= first_year;
    returns df if first_year is None
    """
    if first_year is None:
        return df
    ###years = pd.to_datetime(df[date_col]).dt.year
    #years = df[date_col].dt.year
    if first_year <= df[date_col].min().year:
        # No need to slice
        return df
    return df[df[date_col] >= f"{first_year}-01-01"]


def get_lat_lon_date_features(gt_ids=[], gt_masks=None, gt_shifts=None,
                              forecast_ids=[], forecast_masks=None, forecast_shifts=None,
                              anom_ids=[], anom_masks=None, anom_shifts=None,
                              first_year=None):
    """Returns dataframe of features associated with (lat, lon, start_date)
    values

    Args:
       gt_ids: list of ground-truth variable identifiers to include as
          features
       gt_masks: a mask dataframe, the value None, or list of masks that should
          be applied to each ground-truth feature
       gt_shifts: shift in days, the value None, or list of shifts that should
          be used to shift each ground truth time series forward to produce
          feature
       forecast_ids: list of forecast identifiers to include as features
          (see get_forecast)
       forecast_masks: a mask dataframe, the value None, or list of masks that
          should be applied to each forecast feature
       forecast_shifts: shift in days, the value None, or list of shifts that
          should be used to shift each forecast time series forward to produce
          each forecast feature
       anom_ids: for each ground-truth variable identifier in this list,
          returned dataframe will include ground truth, climatology, and
          ground truth anomaly columns with names measurement_variable,
          measurement_variable+"_clim", and measurement_variable+"_anom"
          for measurement_variable = get_measurement_variable(gt_id);
          only applicable to ids ending in "tmp2m" or "precip"
       anom_masks: a mask dataframe, the value None, or list of masks that should
          be applied to each feature in anom_ids, as well as to the associated
          climatology
       anom_shifts: shift in days, the value None, or list of shifts that should
          be used to shift each ground truth anomaly time series forward to produce
          feature
       first_year: only include rows with year >= first_year; if None, do
          not prune rows by year
    """
    # If particular arguments aren't lists, replace with repeating iterators
    if not isinstance(gt_masks, list):
        gt_masks = itertools.repeat(gt_masks)
    if not isinstance(gt_shifts, list):
        gt_shifts = itertools.repeat(gt_shifts)
    if not isinstance(forecast_masks, list):
        forecast_masks = itertools.repeat(forecast_masks)
    if not isinstance(forecast_shifts, list):
        forecast_shifts = itertools.repeat(forecast_shifts)
    if not isinstance(anom_masks, list):
        anom_masks = itertools.repeat(anom_masks)
    if not isinstance(anom_shifts, list):
        anom_shifts = itertools.repeat(anom_shifts)

    # Define canonical name for target start date column
    date_col = "start_date"
    # Function that warns if any lat-lon-date combinations are duplicated

    def warn_if_duplicated(df):
        if df[['lat', 'lon', date_col]].duplicated().any():
            print(
                "Warning: dataframe contains duplicated lat-lon-date combinations", file=sys.stderr)
    # Add each ground truth feature to dataframe
    df = None
    printf("\nAdding ground truth features to dataframe")
    for gt_id, gt_mask, gt_shift in zip(gt_ids, gt_masks, gt_shifts):
        printf(f"\nGetting {gt_id}_shift{gt_shift}")
        tic()
        # Load ground truth data
        gt = get_ground_truth(gt_id, gt_mask, shift=gt_shift)
        # Discard years prior to first_year
        gt = year_slice(gt, first_year=first_year)
        # Use outer merge to include union of (lat,lon,date_col)
        # combinations across all features
        df = df_merge(df, gt)
        toc()
        tic()
        warn_if_duplicated(df)
        toc()

    # Add each forecast feature to dataframe
    printf("\nAdding forecast features to dataframe")
    for forecast_id, forecast_mask, forecast_shift in zip(forecast_ids,
                                                          forecast_masks,
                                                          forecast_shifts):
        printf("\nGetting {}_shift{}".format(forecast_id, forecast_shift))
        tic()
        # Load forecast with years >= first_year
        forecast = get_forecast(
            forecast_id, forecast_mask, shift=forecast_shift)
        # Discard years prior to first_year
        forecast = year_slice(forecast, first_year=first_year)
        # Use outer merge to include union of (lat,lon,date_col)
        # combinations across all features
        df = df_merge(df, forecast)
        toc()
        tic()
        warn_if_duplicated(df)
        toc()

    # Add anomaly features and climatology last so that climatology
    # is produced for all previously added start dates
    printf("\nAdding anomaly features to dataframe")
    for anom_id, anom_mask, anom_shift in zip(anom_ids, anom_masks, anom_shifts):
        printf("\nGetting {}_shift{} with anomalies".format(anom_id, anom_shift))
        # Add masked ground truth anomalies
        gt = get_ground_truth_anomalies(
            anom_id, mask_df=anom_mask, shift=anom_shift)
        # Discard years prior to first_year
        printf(f"Discarding years prior to {first_year}")
        tic()
        gt = year_slice(gt, first_year=first_year)
        toc()
        # Use outer merge to include union of (lat,lon,date_col)
        # combinations across all features
        printf("Merging in features")
        tic()
        df = df_merge(df, gt)
        toc()
        printf("Checking for row duplications")
        tic()
        warn_if_duplicated(df)
        toc()

    return df


def get_date_features(gt_ids=[], gt_masks=None, gt_shifts=None, first_year=None):
    """Returns dataframe of features associated with start_date values. If
    any of the input dataframes contains columns (lat, lon), it is converted
    to wide format, with one column for each (lat, lon) grid point.

    Args:
       gt_ids: list of ground-truth variable identifiers to include as
          features, choose from {"contest_tmp2m", "pca_tmp2m", "contest_precip",
          "pca_precip", "contest_sst", "pca_sst", "contest_icec", "pca_icec",
          "mei", "mjo", "pca_sst_2010", "pca_icec_2010", "pca_hgt_{}",
          "pca_uwnd_{}", "pca_vwnd_{}",
          "us_tmp2m", "us_precip", "us_sst", "us_icec"}
       gt_masks: a mask dataframe, the value None, or list of masks that should
          be applied to each ground-truth feature
       gt_shifts: shift in days, the value None, or list of shifts that should
          be used to shift each ground truth time series forward to produce
          feature
       first_year: only include rows with year >= first_year; if None, do
          not prune rows by year
    """
    # If particular arguments aren't lists, replace with repeating iterators
    if not isinstance(gt_masks, list):
        gt_masks = itertools.repeat(gt_masks)
    if not isinstance(gt_shifts, list):
        gt_shifts = itertools.repeat(gt_shifts)

    # Add each ground truth feature to dataframe
    df = None
    for gt_id, gt_mask, gt_shift in zip(gt_ids, gt_masks, gt_shifts):
        # Load ground truth data
        printf("\nGetting {}_shift{}".format(gt_id, gt_shift))
        tic()
        gt = get_ground_truth(gt_id, gt_mask, gt_shift)
        toc()
        # Discard years prior to first_year
        printf(f"Discarding years prior to {first_year}")
        tic()
        gt = year_slice(gt, first_year=first_year)
        toc()
        # If lat, lon columns exist, pivot to wide format
        if 'lat' in gt.columns and 'lon' in gt.columns:
            if gt_shift == None:
                measurement_variable = get_measurement_variable(gt_id)
            else:
                measurement_variable = get_measurement_variable(
                    gt_id)+'_shift'+str(gt_shift)
            printf("Transforming to wide format")
            tic()
            gt = gt.set_index(['lat', 'lon', 'start_date']
                              ).unstack(['lat', 'lon'])
            gt = pd.DataFrame(gt.to_records())
            toc()

        # Use outer merge to include union of start_date values across all features
        # combinations across all features
        printf("Merging")
        tic()
        df = df_merge(df, gt, on="start_date")
        toc()

    return df


def get_lat_lon_gt(gt_id, mask_df=None):
    """Returns dataframe with lat_lon feature gt_id.

    Args:
        gt_id: ground truth data string; either "elevation" or "climate_regions"
        mask_df: mask dataframe of the form returned by subsetmask();
          if specified, returned dataframe will be restricted to those lat, lon
          pairs indicated by the mask
    """
    gt_file = os.path.join("data", "dataframes", "gt-{}.h5".format(gt_id))
    df = load_measurement(gt_file, mask_df)
    return df


def get_lat_lon_features(gt_ids=[], gt_masks=None):
    """ Returns dataframe with lat_lon features gt_ids.
        gt_ids: vector with gt data string; e.g. ["elevation", "climate_regions"]
        mask_df: mask dataframe of the form returned by subsetmask();
          if specified, returned dataframe will be restricted to those lat, lon
          pairs indicated by the mask
    """
    # If particular arguments aren't lists, replace with repeating iterators
    if not isinstance(gt_masks, list):
        gt_masks = itertools.repeat(gt_masks)

    df = None
    for gt_id, gt_mask in zip(gt_ids, gt_masks):
        printf("Getting {}".format(gt_id))
        tic()
        # Load ground truth data
        gt = get_lat_lon_gt(gt_id, gt_mask)
        # Use outer merge to include union of (lat,lon,date_col)
        # combinations across all features
        df = df_merge(df, gt, on=["lat", "lon"])
        toc()
    return df


def get_combined_data_dir(model="default"):
    """Returns path to combined data directory for a given model

    Args:
      model: "default" for default versions of data or name of model for model-
        specific versions of data
    """
    if model == 'default':
        cache_dir = os.path.join('data', 'combined_dataframes')
    else:
        cache_dir = os.path.join('models', model, 'data')
    return cache_dir


def get_combined_data_filename(file_id, gt_id, target_horizon,
                               model="default"):
    """Returns path to directory or file of interest

    Args:
      file_id: string identifier defining data file of interest;
        valid values include
        {"lat_lon_date_data","lat_lon_data","date_data","all_data","all_data_no_NA"}
      gt_id: "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
      target_horizon: "34w" or "56w"
      model: "default" for default versions of data or name of model for model
        specific versions of data
    """
    combo_dir = get_combined_data_dir(model)
    suffix = "feather"
    return os.path.join(combo_dir, "{}-{}_{}.{}".format(file_id, gt_id,
                                                        target_horizon, suffix))


def create_lat_lon_date_data(gt_id,
                             target_horizon,
                             model="default",
                             past_gt_ids=["contest_precip", "contest_tmp2m"],
                             # TODO: resolve NMME download issues
                             # ["nmme", "nmme0", "subx_cfsv2"],
                             forecast_models=["subx_cfsv2"],
                             other_lat_lon_date_features=["contest_rhum.sig995",
                                                          "contest_pres.sfc.gauss"]):
    """Generates a dataframe of lat-lon-date features and saves it to file.
    Returns a list with the features included in the saved dataframe.

    Args:
       gt_id: variable to predict; either "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
       target_horizon: either "34w" or "56w"
       model: if "default", saves data in default shared data location;
        otherwise, string indicates model name and saves data in
        model-specific location
       past_gt_ids: list of strings identifying which members of the set
        {"contest_precip","contest_tmp2m"} to include as features
       forecast_models: list of strings identifying which forecast models
        to include as features
       other_lat_lon_date_features: list of other lat-lon-date gt_ids to
        include as features

    Returns:
       List with lat-lon-date features included in the matrix created
    """

    tt = time.time()  # total time counter

    # Add forecasts to list of forecast IDs
    forecast_variable = get_forecast_variable(gt_id)  # 'prate' or 'tmp2m'
    forecast_ids = ['{}-{}-{}'.format(forecast, forecast_variable, target_horizon)
                    for forecast in forecast_models if "nmme" in forecast]
    if "subx_cfsv2" in forecast_models:
        subx_cfsv2_id = "subx_cfsv2-{}".format(
            forecast_variable).replace("prate", "precip")
        if gt_id.startswith('us'):
            subx_cfsv2_id += '-us'
        forecast_ids.append(subx_cfsv2_id)

    # -----------
    # Generate relevant variable and column names
    # -----------

    # Identify measurement variable name
    measurement_variable = get_measurement_variable(
        gt_id)  # 'tmp2m' or 'prate'

    # Keep track of relevant column names
    gt_col = measurement_variable
    clim_col = measurement_variable+"_clim"
    anom_col = measurement_variable+"_anom"

    # Inverse of standard deviation of anomalies for each start_date
    anom_inv_std_col = anom_col+"_inv_std"

    # --------
    # Prepare file name for saved data
    # --------

    lat_lon_date_data_file = get_combined_data_filename(
        "lat_lon_date_data", gt_id, target_horizon, model=model)

    # --------
    # Load mask indicating which grid points count in the contest or in contiguous U.S. (1=in, 0=out)
    # --------
    if gt_id.startswith("contest_"):
        printf("Loading contest mask...")
        tic()
        mask_df = get_contest_mask()
        toc()
    elif gt_id.startswith("us_"):
        printf("Loading contiguous U.S. mask...")
        tic()
        mask_df = get_us_mask()
        toc()
    # --------
    # Creates and saves lat_lon_date_data dataframe
    # --------
    # Load masked lat lon date features restricted to years >= get_first_year(gt_id)
    # Note: contest lat lon date features and forecasts are pre-masked, so there
    # is no need to mask explcitily
    printf("\nLoading lat lon date features...")
    num_gt_ids = len(past_gt_ids)
    # For each measurement,
    # get number of days between start date of observation period used for prediction
    # (2 weeks + 1 submission day behind for most predictors) and start date of
    # target period (2 or 4 weeks ahead)
    past_start_deltas = [get_start_delta(target_horizon, past_gt_id)
                         for past_gt_id in past_gt_ids]
    forecast_start_deltas = [get_start_delta(target_horizon, forecast_id)
                             for forecast_id in forecast_ids]
    other_start_deltas = [get_start_delta(target_horizon, other_gt_id)
                          for other_gt_id in other_lat_lon_date_features]

    # Additionally keep track of days between forecast date and start date of
    # target period
    forecast_delta = get_forecast_delta(target_horizon)

    lat_lon_date_data = get_lat_lon_date_features(
        gt_ids=other_lat_lon_date_features + other_lat_lon_date_features
        + other_lat_lon_date_features,
        gt_masks=mask_df,
        gt_shifts=other_start_deltas +
        [2*delta for delta in other_start_deltas] +
        [365]*len(other_lat_lon_date_features),
        forecast_ids=forecast_ids,
        forecast_masks=mask_df,
        forecast_shifts=forecast_start_deltas,
        anom_ids=[gt_id] + past_gt_ids + past_gt_ids + past_gt_ids,
        anom_masks=mask_df,
        anom_shifts=[None] + past_start_deltas +
                    [2*delta for delta in past_start_deltas] +
                    [365]*len(past_gt_ids),
        first_year=get_first_year(gt_id)
    )

    printf("\nLoading additional lat lon date features")
    tic()
    # Add inverse of standard deviation of anomalies for each start_date
    lat_lon_date_data[anom_inv_std_col] = \
        1.0/lat_lon_date_data.groupby(["start_date"]
                                      )[anom_col].transform('std')
    toc()

    # Save lat lon date features to disk
    tic()
    pandas2feather(lat_lon_date_data, lat_lon_date_data_file)
    toc()
    printf("Finished generating lat_lon_date_data matrix.")
    printf("--total time elapsed: {} seconds\n".format(time.time()-tt))
    return list(lat_lon_date_data)


def load_combined_data(file_id, gt_id,
                       target_horizon,
                       model="default",
                       target_date_obj=None,
                       columns=None):
    """Loads and returns a previously saved combined data dataset

    Args:
      file_id: string identifier defining data file of interest;
        valid values include
        {"lat_lon_date_data","lat_lon_data","date_data","all_data","all_data_no_NA"}
      gt_id: "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
      target_horizon: "34w" or "56w"
      model: "default" for default versions of data or name of model for model
        specific versions of data
      target_date_obj: if not None, print any columns in loaded data that are
        missing on this date in datetime format
      columns: list of column names to load or None to load all
    Returns:
       Loaded dataframe
    """
    data_file = get_combined_data_filename(
        file_id, gt_id, target_horizon, model=model)

    # ---------------
    # Read data_file from disk
    # ---------------
    col_arg = "all columns" if columns is None else columns
    printf(f"Reading {col_arg} from file {data_file}")
    tic()
    data = pd.read_feather(data_file, columns=columns)
    toc()
    # Print any data columns missing on target date
    if target_date_obj is not None:
        print_missing_cols_func(data, target_date_obj, True)

    return data


def create_lat_lon_data(gt_id,
                        target_horizon,
                        model="default",
                        lat_lon_features=["elevation", "climate_regions"]):
    """Generates a dataframe of lat-lon data features and saves it to file.
    Returns a list with the features included in the saved dataframe.

    Args:
       gt_id: variable to predict; either "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
       target_horizon: either "34w" or "56w"
       model: if "default", saves data in default shared data location;
        otherwise, string indicates model name and saves data in
        model-specific location
       lat_lon_features: list of lat-lon gt_ids to include as features

    Returns:
       List with lat-lon features included in the dataframe created
    """

    tt = time.time()  # total time counter

    # --------
    # Prepare model data file name
    # --------

    # Filename for data file to be stored in combo_dir
    lat_lon_data_file = get_combined_data_filename(
        "lat_lon_data", gt_id, target_horizon, model=model)

    # --------
    # Load mask indicating which grid points count in the contest or in contiguous U.S. (1=in, 0=out)
    # --------
    if gt_id.startswith("contest_"):
        printf("Loading contest mask")
        tic()
        mask_df = get_contest_mask()
        toc()
    elif gt_id.startswith("us_"):
        printf("Loading contiguous U.S. mask")
        tic()
        mask_df = get_us_mask()
        toc()

    # --------
    # Creates lat_lon_data dataframe.
    # --------
    # Load masked lat lon features
    printf("\nLoading lat lon features...")
    lat_lon_data = get_lat_lon_features(
        gt_ids=lat_lon_features, gt_masks=mask_df)
    # Add one-hot encoding of climate_region variable to dataframe
    if lat_lon_features:
        if "climate_regions" in lat_lon_features:
            lat_lon_data = pd.concat(
                [lat_lon_data,
                 pd.get_dummies(lat_lon_data[['climate_region']],
                                drop_first=False)], axis=1)
        tic()
        pandas2feather(lat_lon_data, lat_lon_data_file)
        toc()
    else:
        printf("No lat lon features requested")
        # Delete any old version of the data
        if os.path.isfile(lat_lon_data_file):
            os.remove(lat_lon_data_file)

    printf("Finished generating lat_lon_data matrix.")
    printf("--total time elapsed: {} seconds\n".format(time.time()-tt))
    return list(lat_lon_data) if lat_lon_features else 0


def create_date_data(gt_id,
                     target_horizon,
                     model="default",
                     date_features=["mjo", "mei",
                                    "pca_sst_2010", "pca_icec_2010",
                                    "pca_hgt_10_2010",
                                    "pca_hgt_100_2010",
                                    "pca_hgt_500_2010",
                                    "pca_hgt_850_2010"]):
    """Generates a dataframe of requested date features plus
    "ones","year","month","day","dayofyear", and "season" and saves it to file.
    Returns a list with the features included in the saved matrix.

    Args:
       gt_id: variable to predict; either "contest_precip" or "contest_tmp2m"
               or "us_precip" or "us_tmp2m"
       target_horizon: either "34w" or "56w"
       model: if "default", saves data in default shared data location;
        otherwise, string indicates model name and saves data in
        model-specific location
       date_features: which gt_id date features to include

    Returns:
       List with date features included in the matrix created
    """

    tt = time.time()  # total time counter

    # --------
    # Prepare file name for saved data
    # --------

    date_data_file = get_combined_data_filename(
        "date_data", gt_id, target_horizon, model=model)

    # --------
    # Creates date_data dataframe.
    # --------
    # Get number of days between start date of observation period used for prediction
    # (2 weeks behind) and start date of target period (2 or 4 weeks ahead)
    start_deltas = [get_start_delta(target_horizon, gt_id)
                    for gt_id in date_features]

    # Load masked date features
    printf("Loading date features...")
    date_data = get_date_features(gt_ids=date_features, gt_shifts=start_deltas,
                                  first_year=get_first_year(gt_id))

    printf("Loading additional date features")
    if 'mjo' in date_features:
        tic()
        # Add cosine and sine transforms of MJO phase
        mjo_phase_name = 'phase_shift' + \
            str(get_start_delta(target_horizon, 'mjo'))
        date_data['cos_' +
                  mjo_phase_name] = np.cos((2*np.pi*date_data[mjo_phase_name])/8)
        date_data['sin_' +
                  mjo_phase_name] = np.sin((2*np.pi*date_data[mjo_phase_name])/8)
        toc()
    # Add additional standard features
    tic()
    date_data['ones'] = 1
    date_data['year'] = date_data.start_date.dt.year
    date_data['month'] = date_data.start_date.dt.month
    date_data['day'] = date_data.start_date.dt.day
    date_data['dayofyear'] = date_data.start_date.dt.dayofyear
    # Winter = 0, spring = 1, summer = 2, fall = 3
    date_data['season'] = (
        ((date_data.month >= 3) & (date_data.month <= 5)) +
        2 * ((date_data.month >= 6) & (date_data.month <= 8)) +
        3 * ((date_data.month >= 9) & (date_data.month <= 11)))
    toc()
    # Save date features to disk
    tic()
    pandas2feather(date_data, date_data_file)
    toc()

    printf("Finished generating date_data matrix.")
    printf("--total time elapsed: {} seconds\n".format(time.time()-tt))
    return list(date_data)


def create_sub_data(gt_id,
                    target_horizon,
                    submission_date,
                    margin_in_days,
                    model="default"):
    """Generates a sub data matrix which concatenates the lat_lon_date_data,
    date_data and lat_lon_data_file into a single dataframe, and then subsets
    the dates to those matching the day and month of the submission date plus
    dates within margin_in_days distance of the target.  Also restricts data
    to rows with year >= get_first_year(gt_id).

    Args:
       gt_id: variable to predict; either "contest_precip" or "contest_tmp2m"
       target_horizon: either "34w" or "56w"
       model: name of model in which this data will be used
       submission_date: official contest submission deadline (note: we often
            submit a day before the deadline, but this variable should be the
            actual deadline)
       margin_in_days: include targets with months and days within this
            many days of target month and day; to only train on a single target
            month and day, set margin_in_days equal to 0

    Returns:
       List with features included in the subdata matrix created
    """
    tt = time.time()  # total time counter
    # Get target date as a datetime object
    target_date_obj = get_target_date(submission_date, target_horizon)

    # --------
    # Read saved features from disk
    # --------
    if experiment == 'default':
        cache_dir = os.path.join('data', 'combined_dataframes')
    else:
        cache_dir = os.path.join('models', experiment, 'data')

    # Filenames for data files in cache_dir
    lat_lon_date_data_file = os.path.join(
        cache_dir, "lat_lon_date_data-{}_{}.h5".format(gt_id, target_horizon))
    date_data_file = os.path.join(
        cache_dir, "date_data-{}_{}.h5".format(gt_id, target_horizon))
    lat_lon_data_file = os.path.join(
        cache_dir, "lat_lon_data-{}_{}.h5".format(gt_id, target_horizon))

    printf("Reading saved features from {}".format(date_data_file))
    tic()
    date_data = pd.read_hdf(date_data_file)
    toc()
    if os.path.isfile(lat_lon_data_file):
        printf("Reading saved features from {}".format(lat_lon_data_file))
        tic()
        lat_lon_data = pd.read_hdf(lat_lon_data_file)
        toc()
    else:
        lat_lon_data = None
        printf("No lat lon data")
    printf("Reading saved features from {}".format(lat_lon_date_data_file))
    tic()
    lat_lon_date_data = pd.read_hdf(lat_lon_date_data_file)
    toc()

    # ---------------
    # Prepare saved file name for sub_data
    # ---------------
    # Name of cache directory for storing submission date-specific results
    # and intermediate files
    if experiment == 'default':
        submission_cache_dir = os.path.join('data', 'combined_dataframes',
                                            '{}'.format(submission_date))
    else:
        submission_cache_dir = os.path.join('models', experiment, 'data',
                                            '{}'.format(submission_date))
    # if submission_cache_dir doesn't exist, create it
    if not os.path.isdir(submission_cache_dir):
        os.makedirs(submission_cache_dir)
    # Note that saved data subset only depends on margin_in_days,gt_id,
    # target_horizon,submission_date
    sub_data_file = os.path.join(
        submission_cache_dir,
        "sub_data-margin{}-{}_{}-{}.h5".format(margin_in_days, gt_id,
                                               target_horizon, submission_date))

    # ---------------
    # Subset data
    # ---------------
    # Only include rows with year >= the first year in which gt target data is available
    first_year = get_first_year(gt_id)
    printf("Subsetting lat lon date data with margin_in_days {}".format(margin_in_days))
    tic()
    # Restrict data to entries matching target month and day (all years)
    # First, subset lat_lon_date_data
    sub_data = month_day_subset(
        lat_lon_date_data[lat_lon_date_data.start_date.dt.year >= first_year],
        target_date_obj, margin_in_days)
    toc()
    # Second, integrate date_data
    printf("Subsetting date data with margin_in_days {}".format(margin_in_days))
    tic()
    sub_date_data = month_day_subset(
        date_data[date_data.start_date.dt.year >= first_year],
        target_date_obj, margin_in_days)
    toc()
    # Use outer merge to merge lat_lon_date_data and date_data,
    # including the union of start dates
    printf("Merging sub date data into sub data")
    tic()
    sub_data = pd.merge(sub_data, sub_date_data, on="start_date", how="outer")
    toc()
    # Third, integrate lat_lon_data
    sub_lat_lon_data = lat_lon_data
    if sub_lat_lon_data is not None:
        printf("Merging sub lat lon data into sub data")
        tic()
        sub_data = pd.merge(sub_data, sub_lat_lon_data,
                            on=["lat", "lon"], how="outer")
        toc()

    printf("Adding additional sub data features")
    tic()
    # Add year column to dataset
    sub_data['year'] = sub_data.start_date.dt.year
    # Add month column to dataset
    sub_data['month'] = sub_data.start_date.dt.month
    # Add season column to dataset
    # Winter = 0, spring = 1, summer = 2, fall = 3
    sub_data['season'] = (
        ((sub_data.month >= 3) & (sub_data.month <= 5)) +
        2 * ((sub_data.month >= 6) & (sub_data.month <= 8)) +
        3 * ((sub_data.month >= 9) & (sub_data.month <= 11)))
    # Add column of all ones (can be used in place of fitting intercept)
    sub_data['ones'] = 1.0
    # Add column of all zeros (can be used as dummy base_col)
    sub_data['zeros'] = 0.0
    toc()

    # Save subset data to disk
    printf("Saving subset data to "+sub_data_file)
    tic()
    sub_data.to_hdf(sub_data_file, key="data", mode="w")
    subprocess.call("chmod a+w "+sub_data_file, shell=True)
    toc()

    printf("Finished generating sub data matrix.")
    printf("--total time elapsed: {} seconds\n".format(time.time()-tt))
    return list(sub_data)


def load_sub_data(gt_id,
                  target_horizon,
                  target_date_obj,
                  submission_date,
                  margin_in_days,
                  model="default",
                  regen=True,
                  print_missing_cols=True):
    """Loads, and if necessary, generates a sub data matrix which concatenates
    the lat_lon_date_data, date_data and lat_lon_data_file into a single
    dataframe, and then subsets the dates to those matching the day and month of
    the submission date plus dates within margin_in_days distance of the target

    Args:
       gt_id: variable to predict; either "contest_precip" or "contest_tmp2m"
       target_horizon: either "34w" or "56w"
       submission_date: official contest submission deadline (note: we often
            submit a day before the deadline, but this variable should be the
            actual deadline)
       margin_in_days: include targets with months and days within this
            many days of target month and day; to only train on a single target
            month and day, set margin_in_days equal to 0
      regen: if True, sub_data is always regenerated; if False, and
            sub_data file already exists, the stored sub_data is loaded
            and returned

    Returns:
       Subdata matrix created
    """
    # Name of cache directory and file with submission date-specific results
    if experiment == 'default':
        submission_cache_dir = os.path.join('data', 'combined_dataframes',
                                            '{}'.format(submission_date))
    else:
        submission_cache_dir = os.path.join('models', experiment, 'data',
                                            '{}'.format(submission_date))

    sub_data_file = os.path.join(
        submission_cache_dir,
        "sub_data-margin{}-{}_{}-{}.h5".format(margin_in_days, gt_id,
                                               target_horizon, submission_date))

    # ---------------
    # Check if subdata matrix already exists, otherwise regenerate it
    # ---------------
    if regen or not os.path.isfile(sub_data_file):
        printf("Creating sub_data")
        create_sub_data(gt_id, target_horizon, submission_date,
                        model, margin_in_days)
        printf("")

    # ---------------
    # Read saved subset features from disk
    # ---------------
    printf("Reading saved subset features from "+sub_data_file)
    tic()
    sub_data = pd.read_hdf(sub_data_file)
    toc()

    # print any data missing in target_date
    print_missing_cols_func(sub_data, target_date_obj, print_missing_cols)

    return sub_data


def print_missing_cols_func(df, target_date_obj, print_missing_cols):
    if print_missing_cols is True:
        missing_cols_in_target_date = df.loc[df["start_date"] == target_date_obj].isnull(
        ).any()
        if sum(missing_cols_in_target_date) > 0:
            printf("")
            printf("There is missing data for target_date. The following variables are missing: {}"
                   .format(df.columns[missing_cols_in_target_date].tolist()))
            printf("")


def get_id_name(gt_id, None_ok=False):
    if gt_id in ["tmp2m", "temp", "contest_tmp2m", "contest_temp"]:
        return "contest_tmp2m"
    elif gt_id in ["precip", "prate", "contest_prate", "contest_precip"]:
        return "contest_precip"
    elif gt_id in ["us_tmp2m", "us_temp"]:
        return "us_tmp2m"
    elif gt_id in ["us_prate", "us_precip"]:
        return "us_precip"
    elif gt_id is None and None_ok:
        return None
    else:
        raise Exception(f"gt_id not recognized. Value passed: {gt_id}.")


def get_th_name(target_horizon, None_ok=False):
    if target_horizon in ["12", "12w"]:
        return "12w"
    elif target_horizon in ["34", "34w"]:
        return "34w"
    elif target_horizon in ["56", "56w"]:
        return "56w"
    elif target_horizon is None and None_ok:
        return None
    raise Exception(
        f"target_horizon not recognized. Value passed: {target_horizon}.")


def get_conditioning_cols(gt_id, horizon, mei=True, mjo=True):
    conditioning_columns = []
    if mei:
        conditioning_columns += [f"nip_shift{get_forecast_delta(horizon)+30}"]
    if mjo:
        conditioning_columns += [f"phase_shift{get_forecast_delta(horizon)+2}"]
    return conditioning_columns


def cond_indices(conditioning_data, conditioning_columns, target_conditioning_val):
    df_aux = (conditioning_data[conditioning_columns]
              == target_conditioning_val)
    indic = df_aux[conditioning_columns[0]].values
    for c in df_aux.columns[1:]:
        indic = (indic & df_aux[c].values)
    return indic

