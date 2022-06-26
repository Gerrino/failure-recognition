"""Module providing the feature container class"""

import json
from pathlib import Path
from typing import List, Union
from Feature import Feature
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
import pandas as pd
import datetime


#


class FeatureContainer:
    """Container class for tsfresh features

    Examples
    --------
    Feature class, e.g. agg_autocorrleation
    """

    def __init__(self):
        self.Incumbent = {}
        self.FeatureState: pd.DataFrame = pd.DataFrame()
        self.FeatureList: List[Feature] = []
        self.History = pd.DataFrame(
            {
                "datetime": [datetime.datetime.now()],
                "timespan": [datetime.timedelta(0)],
                "action": ["startup"],
                "orig-value": [0],
                "opt-value": [0],
            }
        )

    def __str__(self):
        return f"Feature Container with {len(self.FeatureList)} elements"

    def columnUpdate(self, newSensorState: pd.DataFrame):
        """
        adds columns from newDFState that do not exist in FeatureState to FeatureState.
        updates columns from newDFState if they do exist in FeatureState
        """
        if len(newSensorState) == 0:
            return
        if len(self.FeatureState) == 0:
            self.FeatureState = {}
            self.FeatureState = newSensorState
            return
        # if not sensor in self.FeatureState:
        # self.FeatureState = newSensorState
        # return
        old_cols = len(self.FeatureState.columns)

        for col2_name in filter(lambda c2: (c2 in self.FeatureState.columns), newSensorState.columns):
            del self.FeatureState[col2_name]
        for col2_name in filter(lambda c2: not c2 in self.FeatureState.columns, newSensorState.columns):
            self.FeatureState = pd.concat([self.FeatureState, newSensorState[col2_name]], axis=1)
        print(f"update with {old_cols} => {len(self.FeatureState.columns)}")

    def load(self, path: Union[Path, str]):
        tsfreshlist = open(path)
        objectList = json.loads(
            tsfreshlist.read(),
        )
        for obj in objectList:
            feat = Feature(obj)
            self.FeatureList.append(feat)

    def registerHyperparameters(self, cs, sensors):
        for sensor in sensors:
            for f in filter(lambda f: f.Enabled, self.FeatureList):
                for i in f.InputParameters:
                    hyp = i.GetHyperParameterList(sensor)
                    print(f"Added Hyper Parameter: {hyp}")
                    cs.add_hyperparameters(hyp)

    def resetFeatureState(self):
        self.FeatureState = {}

    def computeFeatureState(self, timeseries: pd.DataFrame, cfg: dict = None, computeForAllFeatures: bool = False):
        """
        Computes the feature matrix for sensor and the incumbent configuration.
        Attention: Changes within "rf_from_cfg" are'nt persistent.
        If cfg is not given, then the feature state is computed  with default values
        """
        sensors = timeseries.columns[2:]
        if computeForAllFeatures and cfg != None:
            self.computeFeatureState(timeseries, cfg=None, computeForAllFeatures=True)
        kind_to_fc_parameters = self.GetFeatureDictionary(sensors, cfg, not computeForAllFeatures)
        if len(kind_to_fc_parameters[sensors[0]]) > 0:
            x = extract_features(
                timeseries, column_id="id", column_sort="time", kind_to_fc_parameters=kind_to_fc_parameters
            )
            X = impute(x)
            self.columnUpdate(X)

    def GetFeatureDictionary(self, sensors: list, cfg: dict = None, useDefaultValues: bool = True) -> dict:
        """
        This method returns a dictionary providing information of all features per sensor and their hyperparameters
        (including the incumbent hyperparameter values).
        cfg given: get feature dict for all features with at least one hyperparam.
        cfg not given: get feature dict for all features (use default values for features with hyperparameters)
        """
        featureDict = {}
        for sensor in sensors:
            featureDict[sensor] = {}
            if cfg != None:
                for feat in filter(lambda f: f.Enabled and len(f.InputParameters) > 0, self.FeatureList):
                    featureDict[sensor][feat.Name] = [feat.GetParameterDict(cfg, sensor)]
            else:
                for feat in filter(lambda f: f.Enabled and len(f.InputParameters) == 0, self.FeatureList):
                    featureDict[sensor][feat.Name] = None
                if useDefaultValues:
                    for feat in filter(lambda f: f.Enabled and len(f.InputParameters) > 0, self.FeatureList):
                        featureDict[sensor][feat.Name] = [feat.GetParameterDict(None, sensor)]
        return featureDict
