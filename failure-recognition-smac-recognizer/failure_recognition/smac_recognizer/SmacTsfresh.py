#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 02:05:17 2021

@author: gerritnoske
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
import RandomForestFromCFG
from smac.initial_design.latin_hypercube_design import LHDesign
from FeatureContainer import FeatureContainer
import pandas as pd
import datetime


def registerDecisionTreeHyperparams(cs, incumbent: dict = None):
    print("\nregister\n numTrees" + str(incumbent["num_trees"] if "num_trees" in incumbent else 10))
    num_trees = UniformIntegerHyperparameter(
        "num_trees",
        10,
        50,
        default_value=incumbent["num_trees"] if "num_trees" in incumbent else 10,
    )
    min_weight_frac_leaf = UniformFloatHyperparameter(
        "min_weight_frac_leaf",
        0.0,
        0.5,
        default_value=incumbent["min_weight_frac_leaf"] if "min_weight_frac_leaf" in incumbent else 0.0,
    )
    criterion = CategoricalHyperparameter(
        "criterion",
        ["squared_error", "absolute_error"],
        default_value=incumbent["criterion"] if "criterion" in incumbent else "squared_error",
    )
    min_samples_to_split = UniformIntegerHyperparameter(
        "min_samples_to_split",
        2,
        20,
        default_value=incumbent["min_samples_to_split"] if "min_samples_to_split" in incumbent else 2,
    )
    min_samples_in_leaf = UniformIntegerHyperparameter(
        "min_samples_in_leaf",
        1,
        20,
        default_value=incumbent["min_samples_in_leaf"] if "min_samples_in_leaf" in incumbent else 1,
    )
    max_leaf_nodes = UniformIntegerHyperparameter(
        "max_leaf_nodes",
        10,
        1000,
        default_value=incumbent["max_leaf_nodes"] if "max_leaf_nodes" in incumbent else 100,
    )
    bootstrapping = CategoricalHyperparameter(
        "do_bootstrapping",
        ["False", "True"],
        default_value=incumbent["do_bootstrapping"] if "do_bootstrapping" in incumbent else "True",
    )
    cs.add_hyperparameters(
        [
            num_trees,
            min_weight_frac_leaf,
            criterion,
            min_samples_to_split,
            min_samples_in_leaf,
            max_leaf_nodes,
            bootstrapping,
        ]
    )
    print(str(cs))


def CreateSmac(
    cs,
    timeseries: pd.DataFrame,
    testSettings: pd.DataFrame,
    y,
    featureContainer,
    scenarioDict,
    seed,
    window_size_ratio,
) -> SMAC4HPO:
    rf_from_cfg = lambda cfg, seed: (
        RandomForestFromCFG.rf_from_cfg_extended(
            cfg, seed, timeseries, testSettings, y, featureContainer, window_size_ratio
        )
    )
    scenarioDict["cs"] = cs
    scenario = Scenario(scenarioDict)
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(
        scenario=scenario,
        rng=seed,
        tae_runner=rf_from_cfg,
        initial_design=LHDesign,
    )
    return smac


def LoadFeatureContainer(path_feat_list: Union[Path, str]):
    """Create the feature container using a tsfresh feature list"""
    featureContainer = FeatureContainer()
    featureContainer.load(path_feat_list)
    print()
    print(f"There are {sum(1 for f in featureContainer.FeatureList if f.Enabled)} enabled Features!")
    print("Features with parameters:")
    print(f"   {', '.join(f.Name for f in featureContainer.FeatureList if f.Enabled and len(f.InputParameters) > 0)}")
    print("Features without parameters:")
    print(
        f"   clear{', '.join(f.Name for f in featureContainer.FeatureList if f.Enabled and len(f.InputParameters) == 0)}"
    )
    print()
    return featureContainer


def registerLogger():
    logger = logging.getLogger("RF-example")
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
    logger.info(
        "Running random forest example for SMAC. If you experience " "difficulties, try to decrease the memory-limit."
    )
    return logger


def createConfigurationSpace(featureContainer, sensors):
    cs = ConfigurationSpace()
    registerDecisionTreeHyperparams(cs, featureContainer.Incumbent)
    windowSizePercent = UniformIntegerHyperparameter("windowSizePercent", 10, 100, default_value=50)
    windowOffsetPercent = UniformIntegerHyperparameter("windowOffsetPercent", 0, 90, default_value=0)
    cs.add_hyperparameters([windowSizePercent, windowOffsetPercent])
    featureContainer.registerHyperparameters(cs, sensors)
    return cs


def Smac_tsfresh_optimize(
    timeseries: pd.DataFrame,
    testSettings: pd.DataFrame,
    y,
    featureContainer: FeatureContainer,
    scenarioDict,
    seed,
    window_size_ratio,
):
    sensors = timeseries.columns[2:]
    featureList = featureContainer.FeatureList
    dateTimeOptStart = datetime.datetime.now()
    numOptFeat = sum(1 for f in featureList if f.Enabled and len(f.InputParameters) > 0)
    nameOptFeat = ", ".join(f.Name for f in featureContainer.FeatureList if f.Enabled)
    for sensor in sensors:
        for f in filter(
            lambda f: f.Enabled and len(f.InputParameters) > 0,
            featureContainer.FeatureList,
        ):  # drop existing default values of enabled features
            print("will it drop")
            drop_name = None
            for col_name in featureContainer.FeatureState.columns:
                drop_name = col_name if col_name.startswith(f"{sensor}__{f.Name}__") else drop_name
            if drop_name != None:
                print("try dropping" + drop_name)
                featureContainer.FeatureState = featureContainer.FeatureState.drop(drop_name, axis=1)
    cs = createConfigurationSpace(featureContainer, sensors)
    smac = CreateSmac(
        cs,
        timeseries,
        testSettings,
        y,
        featureContainer,
        scenarioDict,
        seed,
        window_size_ratio,
    )
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    print("Value for default configuration: %.2f" % def_value)
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    featureContainer.Incumbent = incumbent
    print("Optimized Value: %.2f" % inc_value)
    newHistoryRow = pd.DataFrame(
        {
            "datetime": [datetime.datetime.now()],
            "timespan": [datetime.datetime.now() - dateTimeOptStart],
            "action": [f"Optimize {nameOptFeat}"],
            "orig-value": [def_value],
            "opt-value": [inc_value],
        }
    )
    featureContainer.History = featureContainer.History.append(newHistoryRow)
    featureContainer.computeFeatureState(timeseries, incumbent)  # get parameterless feature matrix
    return incumbent


def smac_tsfresh_window_opt(
    timeseries: pd.DataFrame,
    testSettings: pd.DataFrame,
    y,
    pathDict,
    scenarioDict,
    window_size,
    overlap,
    seed,
    window_size_ratio,
):
    print(f"Starting Optimization window_size: {window_size}, overlap: {overlap}")
    featureContainer = LoadFeatureContainer(pathDict["features"])
    print("Compute feature state for parameterless features")
    featureContainer.computeFeatureState(timeseries, cfg=None)  # get default feature matrix
    featureList = featureContainer.FeatureList
    allEnabledFeatures = list(filter(lambda f: f.Enabled, featureContainer.FeatureList))
    enabledFeatures = list(
        filter(
            lambda f: f.Enabled and len(f.InputParameters) > 0,
            featureContainer.FeatureList,
        )
    )
    enabled_feature_count = len(enabledFeatures)
    incr = window_size - overlap
    if incr <= 0:
        raise Exception("FATAL ERROR: Overlap must be smaller than window_size!")
    window_start_pntr = 0
    logger = registerLogger()
    tot_it = round(
        min(1, np.ceil(enabled_feature_count / window_size))
        + max(0, np.ceil((enabled_feature_count - window_size) / incr))
    )
    it = 1
    incumbent = {}
    while window_start_pntr < enabled_feature_count:
        for f in filter(lambda f: f.Enabled, featureList):
            f.Enabled = False
        cur_window = [
            window_start_pntr,
            min(window_start_pntr + window_size - 1, enabled_feature_count - 1),
        ]
        for i, item in enumerate(enabledFeatures):
            if i >= cur_window[0] and i <= cur_window[1] and item in enabledFeatures:
                item.Enabled = True
            else:
                item.Enabled = False
        print()
        print(f"Iteration {it}/{tot_it}")
        print(
            "|".join("█" if f.Enabled else "░" for f in enabledFeatures)
            + f" -> {', '.join(f.Name for f in featureContainer.FeatureList if f.Enabled)}"
        )
        print()
        window_start_pntr += incr
        it += 1
        incumbent.update(
            Smac_tsfresh_optimize(
                timeseries,
                testSettings,
                y,
                featureContainer,
                scenarioDict,
                seed,
                window_size_ratio,
            )
        )
    for f in allEnabledFeatures:
        f.Enabled = True
    return incumbent, featureContainer
