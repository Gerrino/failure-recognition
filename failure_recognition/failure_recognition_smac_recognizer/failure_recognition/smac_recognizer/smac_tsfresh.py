#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 02:05:17 2021

@author: gerritnoske
"""

from argparse import ArgumentError
import logging
from pathlib import Path
from typing import List, Tuple, Union
import ConfigSpace
import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from failure_recognition.signal_processing.my_property import MyProperty, MyType
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign
from failure_recognition.signal_processing.random_forest_from_cfg import rf_from_cfg_extended
from failure_recognition.signal_processing.feature_container import FeatureContainer
import pandas as pd
import datetime


def hyperparameter_from_type(name: str, type: MyType, default_value) -> ConfigSpace.hyperparameters:
    """Create a hyperparameter object for the given type and initialized with the given default value
    """
    default_value = type.default_value if default_value is None else default_value
    if type.system_type == "int":
        return UniformIntegerHyperparameter(name, *type.range, default_value=default_value)
    if type.system_type == "float":
        return UniformFloatHyperparameter(name, *type.range, default_value=default_value)
    if type.system_type == "string":
        return CategoricalHyperparameter(name, type.range, default_value=default_value)
    raise ArgumentError(
        message="hyperparameter_from_type: unknown system type " + type.system_type)


def generate_decision_tree_hyperparams(cs, type_parameters: List[MyProperty], incumbent: dict):
    """Create and add hyperparameters for every given type to the cs.
     Their default values are either in incumbent or saved in the type object
    """
    print(f"register num_tress={incumbent.get('num_trees', 10)}")
    hyper_parameters = [hyperparameter_from_type(p.name,
        p.type, incumbent.get(p.name)) for p in type_parameters]

    cs.add_hyperparameters(hyper_parameters)
    print("generate_decision_tree_hyperparams", str(cs))

def generate_feature_hyperparams(feature_container: FeatureContainer, cs, sensors, incumbent: dict):
    """Create and register the hyperparameters in the given configuration state for very sensor
    """
    for sensor in sensors:
        for i in [i for f in feature_container.opt_features for i in f.input_parameters]:
            hyp = i.get_hyper_parameter_list(sensor, incumbent)
            print(f"Added Hyper Parameter: {hyp}")
            cs.add_hyperparameters(hyp)
    pass


def create_smac(
    cs,
    timeseries: pd.DataFrame,
    test_settings: pd.DataFrame,
    y,
    feature_container,
    scenario_dict,
    seed,
    window_size_ratio
) -> SMAC4HPO:
    """Create a smac optimization object using the the optimized random forest regressor
    """
    def rf_from_cfg(cfg, seed): return (
        rf_from_cfg_extended(
            cfg, seed, timeseries, test_settings, y, feature_container, window_size_ratio
        )
    )
    scenario_dict["cs"] = cs
    scenario = Scenario(scenario_dict)
    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(
        scenario=scenario,
        rng=seed,
        tae_runner=rf_from_cfg,
        initial_design=LHDesign
    )
    return smac


def load_feature_container(path_feat_list: Union[Path, str], path_random_forest: Union[Path, str], logger: logging.Logger) -> FeatureContainer:
    """Create the feature container using a tsfresh feature list
    """
    feature_container = FeatureContainer(logger=logger)
    feature_container.load(path_feat_list, path_random_forest)
    print()
    print(
        f"There are {sum(1 for f in feature_container.feature_list if f.enabled)} enabled Features!")
    print("Features with parameters:")
    print(f"   {', '.join(f.name for f in feature_container.param_features)}")
    print("Features without parameters:")
    print(
        f"   clear{', '.join(f.name for f in feature_container.paramless_features)}"
    )
    print()
    return feature_container


def register_logger():
    """Create the logging object (INFO)"""
    logger = logging.getLogger("SmacOpt")
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Running random forest example for SMAC. If you experience " "difficulties, try to decrease the memory-limit."
    )
    return logger


def create_configuration_space(feature_container: FeatureContainer, sensors: list) -> ConfigurationSpace:
    """Create a smac configuration space object using the incumbent state, the
    window parameters and random forest parameters
    """
    cs = ConfigurationSpace()
    generate_decision_tree_hyperparams(
        cs, feature_container.random_forest_params, feature_container.incumbent)
    window_size_percent = UniformIntegerHyperparameter(
        "window_size_percent", 10, 100, default_value=50)
    window_offset_percent = UniformIntegerHyperparameter(
        "window_offset_percent", 0, 90, default_value=0)
    cs.add_hyperparameters([window_size_percent, window_offset_percent])
    generate_feature_hyperparams(feature_container, cs, sensors, feature_container.incumbent)
    return cs


def smac_tsfresh_optimize(
    timeseries: pd.DataFrame,
    test_settings: pd.DataFrame,
    y,
    feature_container: FeatureContainer,
    scenario_dict,
    seed,
    window_size_ratio,
):
    """
    Performs the optimization problem on timeseries using smac given a feature window

    Parameters
    ---
    timeseries: pd.DataFrame
        time series
    test_settings: pd.Dat   aFrame
        test settings corresponding to the time series
    y: pd.DataFrame
        label data frame
    feature_container: FeatureContainer
        feature container object with the current feature state
    scenario_dict: dict
        dictionary containing special options for smac e.g. memory limit
    seed: float
        seed for the smac opt
    window_size_ratio: float
        #window_featuers / #all_features

    Returns
    ---
    incumbent: dict
    """
    sensors = timeseries.columns[2:]
    feature_list = feature_container.feature_list
    date_time_opt_start = datetime.datetime.now()
    num_opt_feat = sum(1 for f in feature_container.opt_features)
    name_opt_feat = ", ".join(
        f.name for f in feature_container.opt_features)
    feature_container.new_columns.clear()
    cs = create_configuration_space(feature_container, sensors)
    smac = create_smac(
        cs,
        timeseries,
        test_settings,
        y,
        feature_container,
        scenario_dict,
        seed,
        window_size_ratio,
    )
    def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
    print("Value for default configuration: %.2f" % def_value)
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    print("Added colums", feature_container.new_columns)
    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    feature_container.incumbent = incumbent
    print("Optimized Value: %.3f" % inc_value)
    new_history_row = pd.DataFrame(
        {
            "datetime": [datetime.datetime.now()],
            "timespan": [datetime.datetime.now() - date_time_opt_start],
            "action": [f"Optimize {name_opt_feat}"],
            "orig-value": [def_value],
            "opt-value": [inc_value],
        }
    )
    feature_container.history = feature_container.history.append(
        new_history_row)
    feature_container.compute_feature_state(
        timeseries, incumbent)  # get parameterless feature matrix
    return incumbent


def smac_tsfresh_window_opt(
    timeseries: pd.DataFrame,
    test_settings: pd.DataFrame,
    y: pd.DataFrame,
    path_dict: dict,
    scenario_dict: dict,
    window_size: int,
    overlap: int,
    seed: float,
    window_size_ratio: float,
) -> Tuple[dict, FeatureContainer]:
    """
    Performs the optimization problem on timeseries using smac and a feature window

    Parameters
    ---
    timeseries: pd.DataFrame
        time series
    test_settings: pd.Dat   aFrame
        test settings corresponding to the time series
    y: pd.DataFrame
        label data frame
    path_dict: dict
        dictionary containing paths for feature and random forest parameters
    window_size: int
        size of the feature window
    overlap: int
        feature overlap
    seed: float
        seed for the smac opt
    window_size_ratio: float
        #window_featuers / #all_features

    Returns
    ---
    incumbent: dict, feature_container: FeatureContainer
    """
    print(
        f"Starting Optimization window_size: {window_size}, overlap: {overlap}")
    feature_container = load_feature_container(
        path_dict["features"], path_dict["forest_params"], logger=register_logger())
    print("Compute feature state for parameterless features")
    feature_container.compute_feature_state(
        timeseries, cfg=None, compute_for_all_features=True)  # get default feature matrix

    opt_features = list([f for f in feature_container.opt_features])
    param_features_cnt = len(opt_features)
    incr = window_size - overlap
    if incr <= 0:
        raise Exception(
            "FATAL ERROR: Overlap must be smaller than window_size!")
    window_start_pntr = 0
    tot_it = round(
        min(1, np.ceil(param_features_cnt / window_size))
        + max(0, np.ceil((param_features_cnt - window_size) / incr))
    )
    it = 1
    incumbent = {}
    while window_start_pntr - incr + window_size < param_features_cnt:
        cur_window = [
            window_start_pntr,
            min(window_start_pntr + window_size - 1, param_features_cnt - 1),
        ]
        for i, item in enumerate(opt_features):
            item.optimize = cur_window[0] <= i <= cur_window[1] and item in opt_features

        print()
        print(f"Iteration {it}/{tot_it}")
        print(
            "|".join("█" if f.is_optimized() else "░" for f in opt_features)
            + f" -> {', '.join(f.name for f in feature_container.opt_features)}"
        )
        print()
        window_start_pntr += incr
        it += 1
        incumbent.update(
            smac_tsfresh_optimize(
                timeseries,
                test_settings,
                y,
                feature_container,
                scenario_dict,
                seed,
                window_size_ratio,
            )
        )

    return incumbent, feature_container
