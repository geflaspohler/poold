# Model attributes
import json
import os
from src.utils.general_util import set_file_permissions, hash_strings
from filelock import FileLock

MODEL_NAME="llr"
SELECTED_SUBMODEL_PARAMS_FILE=os.path.join(
    "src","models",MODEL_NAME,"selected_submodel.json")
SUBMODEL_NAME_TO_PARAMS_FILE = os.path.join(
    "src", "models", MODEL_NAME, "submodel_name_to_params.json")

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args)

def get_submodel_name(train_years="all", margin_in_days=None,
                      use_cfsv2="False"):
    """Returns submodel name for a given setting of model parameters
    """
    submodel_name = (f"{MODEL_NAME}-years{train_years}_margin{margin_in_days}"
                     f"_cfsv2{use_cfsv2}")

    return submodel_name
