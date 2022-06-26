#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 02:05:17 2021

@author: gerritnoske
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn.datasets import load_boston
from smac.configspace import ConfigurationSpace
from smac.configspace import InCondition
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
import random_forest_from_cfg
from tsfresh.utilities.dataframe_functions import impute
from smac.initial_design.latin_hypercube_design import LHDesign
import pandas as pd
from tsfresh import extract_features
from tsfresh import feature_extraction
from itertools import islice
from feature_container import FeatureContainer
import datetime


def registerDecisionTreeHyperparams(cs):
    num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default_value=10)
    min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default_value=0.0)
    criterion = CategoricalHyperparameter("criterion", ["squared_error", "absolute_error"], default_value="squared_error")
    min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default_value=2)
    min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default_value=1)
    max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)
    cs.add_hyperparameters([num_trees, min_weight_frac_leaf, criterion,
                            min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])


def CreateSmac(cs, sensor, timeseries, y, feature_container, scenario_dict) -> SMAC4HPO:
    rf_from_cfg = lambda cfg, seed: (
        RandomForestFromCFG.rf_from_cfg_extended(cfg, seed, "0", timeseries, y, feature_container))
    scenario_dict["cs"] = cs
    scenario = Scenario(scenario_dict)
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario,
                    rng=np.random.RandomState(42),
                    tae_runner=rf_from_cfg,
                    initial_design=LHDesign,
                    )
    return smac


def LoadFeatureContainer(path_feat_list):
    featureContainer = FeatureContainer()
    featureContainer.load(path_feat_list)
    print()
    print(f"There are {sum(1 for f in featureContainer.feature_list if f.enabled)} enabled Features!")
    print("Features with parameters:")
    print(f"   {', '.join(f.name for f in featureContainer.feature_list if f.enabled and len(f.input_parameters) > 0)}")
    print("Features without parameters:")
    print(
        f"   clear{', '.join(f.name for f in featureContainer.feature_list if f.enabled and len(f.input_parameters) == 0)}")
    print()
    return featureContainer


def registerLogger():
    logger = logging.getLogger("RF-example")
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
    logger.info("Running random forest example for SMAC. If you experience "
                "difficulties, try to decrease the memory-limit.")
    return logger


def createConfigurationSpace(featureContainer):
    cs = ConfigurationSpace()
    registerDecisionTreeHyperparams(cs)
    featureContainer.register_hyperparameters(cs)
    return cs


def Smac_tsfresh_optimize(timeseries, y, featureContainer, scenarioDict):
    featureList = featureContainer.feature_list
    # dateTimeOptStart = datetime.now()
    numOptFeat = sum(1 for f in featureList if f.enabled and len(f.input_parameters) > 0)
    # nameOptFeat = ''#', '.join(f.Name for f in featureContainer.FeatureList if f.Enabled)
    cs = createConfigurationSpace(featureContainer)
    smac = CreateSmac(cs, "0", timeseries, y, featureContainer, scenarioDict)
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    print("Value for default configuration: %.2f" % def_value)
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    print("Optimized Value: %.2f" % inc_value)
    # newHistoryRow = pd.DataFrame({'datetime': datetime.now(), 'timespan': datetime.now() - dateTimeOptStart, 'action': f"Optimize {nameOptFeat}", 'opt-value': inc_value})
    # featureContainer.History.append(newHistoryRow)
    featureContainer.compute_feature_state("0", timeseries, incumbent)  # get parameterless feature matrix
    return incumbent


def smac_tsfresh_window_opt(timeseries, y, pathDict, scenarioDict, window_size, overlap):
    print(f"Starting Optimization window_size: {window_size}, overlap: {overlap}")
    featureContainer = LoadFeatureContainer(pathDict["features"])
    print("Compute feature state for parameterless features")
    featureContainer.compute_feature_state("0", timeseries, cfg=None)  # get parameterless feature matrix
    featureList = featureContainer.feature_list
    enabledFeatures = list(filter(lambda f: f.enabled and len(f.input_parameters) > 0, featureContainer.feature_list))
    enabled_feature_count = len(enabledFeatures)
    incr = window_size - overlap
    if incr <= 0:
        raise Exception("FATAL ERROR: Overlap must be smaller than window_size!")
    window_start_pntr = 0
    logger = registerLogger()
    tot_it = round(min(1, np.ceil(enabled_feature_count / window_size)) + max(0, np.ceil(
        (enabled_feature_count - window_size) / incr)))
    it = 1
    incumbent = {}
    while window_start_pntr < enabled_feature_count:
        for f in filter(lambda f: f.enabled, featureList):
            f.enabled = False
        cur_window = [window_start_pntr, min(window_start_pntr + window_size - 1, enabled_feature_count - 1)]
        for i, item in enumerate(enabledFeatures):
            if i >= cur_window[0] and i <= cur_window[1] and item in enabledFeatures:
                item.enabled = True
            else:
                item.enabled = False
        print()
        print(f"Iteration {it}/{tot_it}")
        print('|'.join('█' if f.enabled else '░' for f in
                       enabledFeatures) + f" -> {', '.join(f.name for f in featureContainer.feature_list if f.enabled)}")
        print()
        window_start_pntr += incr
        it += 1
        incumbent.update(Smac_tsfresh_optimize(timeseries, y, featureContainer, scenarioDict))
    return incumbent, featureContainer.feature_state
