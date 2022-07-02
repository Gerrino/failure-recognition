"""Module providing the feature container class"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Union
from failure_recognition.signal_processing.feature import Feature
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
import pandas as pd
import datetime

from failure_recognition.signal_processing.my_property import MyProperty


#

@dataclass
class FeatureContainer:
    """Container class for tsfresh features

    Examples
    --------
    Feature class, e.g. agg_ autocorrelation
    """
    feature_list: List[Feature] = field(default_factory=list)
    incumbent: dict = field(default_factory=dict)
    feature_state: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    random_forest_params: List[MyProperty] = field(default_factory=list)

    def __post_init__(self):
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

    def column_update(self, new_sensor_state: pd.DataFrame):
        """
        adds columns from newDFState that do not exist in feature_state to feature_state.
        updates columns from newDFState if they do exist in feature_state
        """
        if len(new_sensor_state) == 0:
            return
        if len(self.feature_state) == 0:
            self.feature_state = {}
            self.feature_state = new_sensor_state
            return
        # if not sensor in self.feature_state:
        # self.feature_state = newSensorState
        # return
        old_cols = len(self.feature_state.columns)

        for col2_name in filter(lambda c2: (c2 in self.feature_state.columns), new_sensor_state.columns):
            del self.feature_state[col2_name]
        for col2_name in filter(lambda c2: not c2 in self.feature_state.columns, new_sensor_state.columns):
            self.feature_state = pd.concat(
                [self.feature_state, new_sensor_state[col2_name]], axis=1)
        print(f"update with {old_cols} => {len(self.feature_state.columns)}")

    def load(self, tsfresh_features: Union[Path, str], random_forest_parameters: Union[Path, str]):
        with open(tsfresh_features, 'r', encoding="utf-8") as features_file:
            feature_list = json.load(features_file)
        for feature in feature_list:
            feat = Feature.from_json(feature)
            self.feature_list.append(feat)
        self.random_forest_params.clear()
        with open(random_forest_parameters, 'r', encoding="utf-8") as features_file:
            forest_parameters_json = json.load(features_file)
        for forest_parameter_json in forest_parameters_json:
            self.random_forest_params.append(MyProperty.from_json(forest_parameter_json))

    def reset_feature_state(self):
        self.feature_state = {}

    def compute_feature_state(self, timeseries: pd.DataFrame, cfg: dict = None, compute_for_all_features: bool = False):
        """
        Computes the feature matrix for sensor and the incumbent configuration.
        Attention: Changes within "rf_from_cfg" are not persistent.
        If cfg is not given, then the feature state is computed  with default values
        """
        sensors = timeseries.columns[2:]
        if compute_for_all_features and cfg is not None:
            self.compute_feature_state(
                timeseries, cfg=None, compute_for_all_features=True)
        kind_to_fc_parameters = self.get_feature_dictionary(
            sensors, cfg, not compute_for_all_features)
        if len(kind_to_fc_parameters[sensors[0]]) > 0:
            x = extract_features(
                timeseries, column_id="id", column_sort="time", kind_to_fc_parameters=kind_to_fc_parameters
            )
            X = impute(x)
            self.column_update(X)

    def get_feature_dictionary(self, sensors: list, cfg: dict = None, use_default_values: bool = True) -> dict:
        """
        This method returns a dictionary providing information of all features per sensor and their hyperparameters
        (including the incumbent hyperparameter values).
        cfg given: get feature dict for all features with at least one hyperparam.
        cfg not given: get feature dict for all features (use default values for features with hyperparameters)
        """
        feature_dict = {}
        for sensor in sensors:
            feature_dict[sensor] = {}
            if cfg is not None:
                for feat in filter(lambda f: f.enabled and len(f.input_parameters) > 0, self.feature_list):
                    feature_dict[sensor][feat.name] = [
                        feat.get_parameter_dict(cfg, sensor)]
            else:
                for feat in filter(lambda f: f.enabled and len(f.input_parameters) == 0, self.feature_list):
                    feature_dict[sensor][feat.name] = None
                if use_default_values:
                    for feat in filter(lambda f: f.enabled and len(f.input_parameters) > 0, self.feature_list):
                        feature_dict[sensor][feat.name] = [
                            feat.get_parameter_dict(None, sensor)]
        return feature_dict


if __name__ == "__main__":
    container = FeatureContainer()
    pass
