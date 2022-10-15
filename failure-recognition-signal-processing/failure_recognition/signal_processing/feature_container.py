"""Module providing the feature container class"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Union

from failure_recognition.signal_processing import PATH_DICT
from failure_recognition.signal_processing.feature import Feature
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
import pandas as pd
import datetime
from failure_recognition.signal_processing.my_property import MyProperty
from failure_recognition.signal_processing.signal_helper import find_peaks_feature


@dataclass
class FeatureContainer:
    """Container class for tsfresh features

    Examples
    --------
    Feature class, e.g. agg_ autocorrelation
    """
    logger: logging.Logger = None
    feature_list: List[Feature] = field(default_factory=list)
    incumbent: dict = field(default_factory=dict)
    feature_state: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    random_forest_params: List[MyProperty] = field(default_factory=list)
    custom_features: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger("FEAT_CONT_LOGGER")
        self.custom_features["find_peaks_feature"] = find_peaks_feature
        self.history = pd.DataFrame(
            {
                "datetime": [datetime.datetime.now()],
                "timespan": [datetime.timedelta(0)],
                "action": ["startup"],
                "orig-value": [0],
                "opt-value": [0],
            })

    def __str__(self):
        return f"Feature Container with {len(self.feature_list)} elements"

    @property
    def enabled_features(self) -> List[Feature]:
        """Get all enabled features"""
        return [f for f in self.feature_list if f.enabled]

    @property
    def paramless_features(self) -> List[Feature]:
        """Get all enabled features without(!) parameters"""
        return [f for f in self.enabled_features if not f.has_params()]

    @property
    def param_features(self) -> List[Feature]:
        """Get all enabled features with parameters"""
        return [f for f in self.enabled_features if f.has_params()]
    
    @property
    def opt_features(self) -> List[Feature]:
        """Get all enabled features with parameters that are being optimized"""
        return [f for f in self.feature_list if f.enabled and f.is_optimized()]
    
    @property
    def no_opt_features(self) -> List[Feature]:
        """Get all enabled features that are not being optimized"""
        return [f for f in self.feature_list if f.enabled and not f.is_optimized()]

    def column_update(self, new_sensor_state: pd.DataFrame, drop_opt_col: bool = True):
        """
        Add columns from newDFState that do not exist in feature_state to feature_state.
        updates columns from newDFState if they do exist in feature_state.
        
        Parameters
        ---
        new_sensor_state:
            Data frame with feature columns
        drop_opt_col:
            Drop columns belonging to features with parameters that are being optimized
        """
        if len(new_sensor_state) == 0:
            return
        if len(self.feature_state) == 0:
            self.feature_state = {}
            self.feature_state = new_sensor_state
            return
        old_cols_cnt = len(self.feature_state.columns)        
        self.logger.info(f"old columns \n {self.feature_state.columns.values}")
        self.logger.info(f"new columns \n {new_sensor_state.columns.values}")
        if drop_opt_col:             
            parameter_columns = [c for c in self.feature_state.columns for f in self.opt_features if f"__{f.name}" in c]
            self.feature_state = self.feature_state.drop(parameter_columns, axis=1)
        
        for overwrite_col in [c for c in new_sensor_state.columns if c in self.feature_state.columns]:
            del self.feature_state[overwrite_col]
        self.feature_state = pd.concat([self.feature_state, new_sensor_state], axis=1)

        self.logger.info(f"result columns \n{self.feature_state.columns.values}")
        self.logger.info(f"Column update\n {old_cols_cnt} => {len(self.feature_state.columns.values)}")

    def load(self, tsfresh_features: Union[Path, str], random_forest_parameters: Union[Path, str]):
        """Load features/rf params from file"""
        with open(tsfresh_features, 'r', encoding="utf-8") as features_file:
            feature_list = json.load(features_file)
        for feature in feature_list:
            feat = Feature.from_json(feature)
            self.feature_list.append(feat)
        self.random_forest_params.clear()
        with open(random_forest_parameters, 'r', encoding="utf-8") as features_file:
            forest_parameters_json = json.load(features_file)
        for forest_parameter_json in forest_parameters_json:
            self.random_forest_params.append(
                MyProperty.from_json(forest_parameter_json))

    def reset_feature_state(self):
        """Reset the feature state"""
        self.feature_state = {}

    @property
    def column_names(self):
        return list(self.feature_state.columns.values)

    def compute_feature_state(self, timeseries: pd.DataFrame, cfg: dict = None, compute_for_all_features: bool = False):
        """
        Computes the feature matrix for sensor and the incumbent configuration.
        Attention: Changes within "rf_from_cfg" are not persistent.
        If cfg is not given, then the feature state is computed  with default values

        Parameters
        ----------
        timeseries: pd.DataFrame
            timeseries
        cfg: dict
            difeature param / value dictionary. If None, use default values
        compute_for_all_features: bool
            If true, compute the feature state for all features (including non-opt features)

        """
        sensors = timeseries.columns[2:]

        # get the sensor-feature dictionary for all features to extract
        kind_to_fc_parameters = self.get_feature_dictionary(sensors, cfg,  compute_for_all_features, True)    
        #print("kind_to_fc_parameters", kind_to_fc_parameters)                   

        if len(kind_to_fc_parameters[sensors[0]]) > 0:
            x = extract_features(
                timeseries, column_id="id", column_sort="time", kind_to_fc_parameters=kind_to_fc_parameters
            )
            X = impute(x)
            self.column_update(X)

    def get_feature_dictionary(self, sensors: list, cfg: dict, non_opt: bool, opt: bool) -> dict:
        """
        This method returns a dictionary providing information of all features per sensor and their hyperparameters
        (including the incumbent hyperparameter values).

        Parameters
        ---
        non_opt: get feature dict for parameterless features
        opt: get feature dict for all features with at least one hyperparameter          
        
        Returns
        ---
        "sensor_0": 
            "feature_0":
                "input_var_0": designated_value
        ...
        """        
        paramless_features, param_features = [], []
        if non_opt:
            paramless_features = [f for f in self.no_opt_features if not f.has_params()]
            param_features = [f for f in self.no_opt_features if f.has_params()]
        if opt:
            param_features += self.opt_features

        def merge_with_coeffi(feat: Feature, params: Union[dict, None]) -> List[Dict]:
            """Return a list of param dicts for all coefficients"""
            coeffi = feat.coefficients
            if coeffi is None:
                if params is None:
                    return None
                return [params]
            merged_list = []
            coeffi_values = coeffi.get_values()
            if len(coeffi_values) == 0:
                raise ValueError("merge_with_coeffi: Zero coefficients")
            for value in coeffi_values:
                value: int
                coeffi_dict = dict(params) if params is not None else {}
                coeffi_dict[coeffi.name] = value
                merged_list.append(coeffi_dict)
            return merged_list

        feature_dict = {}
        for sensor in sensors:
            feature_dict[sensor] = {}

            for feat in paramless_features:               
                feature_dict[sensor][feat.name] = merge_with_coeffi(feat, None)           

            for feat in param_features:
                param_dict = feat.get_parameter_dict(cfg, sensor)
                feature_dict[sensor][feat.name] = merge_with_coeffi(feat, param_dict)

            for name, func in self.custom_features.items():
                if name in feature_dict[sensor]:
                    feature_dict[sensor][func] = feature_dict[sensor].pop(name)

        #print("feature_dict", feature_dict)
        return feature_dict


if __name__ == "__main__":
    container = FeatureContainer()
    container.load(PATH_DICT["features"], PATH_DICT["forest_params"])
    pass
