import numpy as np
import pandas as pd
import os
import pdb
from datetime import datetime

def get_and_merge_experts(expert_filenames, default_value=None):
    """Merges forecasts from experts. Returns a forecast_size by number 
    of experts merged pd DataFrame. Additionally returns a list of experts 
    who are missing a prediction for the target date. Missing predictions are 
    filled with a default_value if provided, or skipped if default_value is None.

    Args:
       expert_filenames: dict from model_name to full paths to the forecast 
           files a given target date and contest objective 
       expert_models: a list of expert model names that provides an ordering
           for the columns of the returned np array
       default_value: when an expert prediction is missing, the corresponding
           column is filled with default_value. 
    """   
    # Read in each experts predictions
    merged_df = None
    missing_experts = []
    
    cols_to_select = ["start_date", "lat", "lon", "pred"]
    for a, m, fname in expert_filenames:
        # If expert is missing predications, fill with default value
        if not os.path.exists(fname):
            missing_experts.append(a)
            if default_value is not None:
                merged_df[f"{a}"] = default_value             
            continue

        if merged_df is None:
            merged_df = pd.read_hdf(fname).rename(columns={"pred": f"{a}"})
            if merged_df.isna().any(axis=None): # If any of expert predictions are NaN
                missing_experts.append(a)
                if default_value is not None:
                    merged_df[f"{a}"] = default_value             
                continue
        else:
            df = pd.read_hdf(fname).rename(columns={"pred": f"{a}"})
            if df.isna().any(axis=None): # If any of expert predictions are NaN
                missing_experts.append(a)
                if default_value is not None:
                    merged_df[f"{a}"] = default_value             
                continue

            merged_df = pd.merge(merged_df,
                            df, on=["start_date", "lat", "lon"])
   
    # Important to sort df in order to ensure lat/lon points are in consistant order 
    if merged_df is not None: # occurs if all experts are missing
        merged_df = merged_df.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()    
    return merged_df, missing_experts

def generate_expert_df(get_filename_fn, target_date_objs, expert_models, expert_submodels, default_value=None):
    """ Generates a merged expert dataframe for all target dates in 
    target_date_objs. Utility funciton, not possible for real-time funciton

    Args:
       get_filename_fn: partial of get_forecast_filename that takes a model,
           submodel, and target date as input and produces forecast filename
       target_date_objs: a pd Series of target date time objects 
       expert_models: a list of expert model names that provides an ordering
           for the columns of the returned np array
       expert_submodels: a dictionary from model name to selected submodel name          
       default_value: when an expert prediction is missing, the corresponding
           column is filled with default_value. If None, that expert is skipped. 
    """       
    expert_df = None
    for target_date_obj in target_date_objs:
        # Convert target date to string
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')  

        # Get names of submodel forecast files using the selected submodel
        expert_filenames = {m: get_filename_fn(model=m, 
                                               submodel=s,
                                               target_date_str=target_date_str) 
                               for (m, s) in expert_submodels.items()}
        
        # Get expert merged df for target date
        merged_df, missing_experts = get_and_merge_experts(expert_filenames, default_value)
        
        if len(missing_experts) > 0 and default_value is None:
            printf(f"warning: experts {missing_experts} unavailable for target={target_date_obj}; skipping")
            continue

        # Update full expert_df
        if expert_df is None:
            expert_df = merged_df
        else:
            expert_df = pd.merge(expert_df, merged_df, 
                                 left_index=True, right_index=True, how='outer', 
                                 on=expert_models)
    return expert_df.sort_index()
