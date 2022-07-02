#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 02:05:17 2021

@author: gerritnoske
"""

from argparse import ArgumentError
import logging
from pathlib import Path
from typing import List, Union
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


def hyperparameter_from_type(type: MyType, default_value) -> ConfigSpace.hyperparameters:
    default_value = type.default_value if default_value is None else default_value
    if type.system_type == "int":
        return UniformIntegerHyperparameter(type, *type.range, default_value=default_value)
    if type.system_type == "float":
        return UniformFloatHyperparameter(type, *type.range, default_value=default_value)
    if type.system_type == "string":
        return CategoricalHyperparameter(type, type.range, default_value=default_value)
    raise ArgumentError(
        message="hyperparameter_from_type: unknown system type " + type.system_type)


def register_decision_tree_hyperparams(cs, type_parameters: List[MyProperty], incumbent: dict = None):
    """Create and add hyperparameters for every given type to the cs.
     Their default values are either in incumbent or saved in the type object
    """
    print("\nregister\n numTrees" +
          str(incumbent["num_trees"] if "num_trees" in incumbent else 10))
    hyper_parameters = [hyperparameter_from_type(
        p.type, incumbent.get(p.name)) for p in type_parameters]

    cs.add_hyperparameters(hyper_parameters)
    print(str(cs))


def create_smac(
    cs,
    timeseries: pd.DataFrame,
    test_settings: pd.DataFrame,
    y,
    feature_container,
    scenario_dict,
    seed,
    window_size_ratio,
) -> SMAC4HPO:
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
        initial_design=LHDesign,
    )
    return smac


def load_feature_container(path_feat_list: Union[Path, str], path_random_forest: Union[Path, str]):
    """Create the feature container using a tsfresh feature list"""
    feature_container = FeatureContainer()
    feature_container.load(path_feat_list, path_random_forest)
    print()
    print(
        f"There are {sum(1 for f in feature_container.feature_list if f.enabled)} enabled Features!")
    print("Features with parameters:")
    print(f"   {', '.join(f.name for f in feature_container.feature_list if f.enabled and len(f.input_parameters) > 0)}")
    print("Features without parameters:")
    print(
        f"   clear{', '.join(f.name for f in feature_container.feature_list if f.enabled and len(f.input_parameters) == 0)}"
    )
    print()
    return feature_container


def register_logger():
    logger = logging.getLogger("RF-example")
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
    logger.info(
        "Running random forest example for SMAC. If you experience " "difficulties, try to decrease the memory-limit."
    )
    return logger


def register_hyperparameters(feature_container, cs, sensors):
    for sensor in sensors:
        for f in filter(lambda f: f.enabled, feature_container.feature_list):
            for i in f.input_parameters:
                hyp = i.get_hyper_parameter_list(sensor)
                print(f"Added Hyper Parameter: {hyp}")
                cs.add_hyperparameters(hyp)


def create_configuration_space(feature_container, sensors):
    cs = ConfigurationSpace()
    register_decision_tree_hyperparams(
        cs, feature_container.random_forest_params, feature_container.incumbent)
    window_size_percent = UniformIntegerHyperparameter(
        "window_size_percent", 10, 100, default_value=50)
    window_offset_percent = UniformIntegerHyperparameter(
        "window_offset_percent", 0, 90, default_value=0)
    cs.add_hyperparameters([window_size_percent, window_offset_percent])
    register_hyperparameters(feature_container, cs, sensors)
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
    sensors = timeseries.columns[2:]
    feature_list = feature_container.feature_list
    date_time_opt_start = datetime.datetime.now()
    num_opt_feat = sum(
        1 for f in feature_list if f.enabled and len(f.input_parameters) > 0)
    name_opt_feat = ", ".join(
        f.name for f in feature_container.feature_list if f.enabled)
    for sensor in sensors:
        for f in filter(
            lambda f: f.enabled and len(f.input_parameters) > 0,
            feature_container.feature_list,
        ):  # drop existing default values of enabled features
            print("will it drop")
            drop_name = None
            for col_name in feature_container.feature_state.columns:
                drop_name = col_name if col_name.startswith(
                    f"{sensor}__{f.name}__") else drop_name
            if drop_name is not None:
                print("try dropping" + drop_name)
                feature_container.feature_state = feature_container.feature_state.drop(
                    drop_name, axis=1)
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
    inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
    feature_container.incumbent = incumbent
    print("Optimized Value: %.2f" % inc_value)
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
    y,
    path_dict,
    scenario_dict,
    window_size,
    overlap,
    seed,
    window_size_ratio,
):
    print(
        f"Starting Optimization window_size: {window_size}, overlap: {overlap}")
    feature_container = load_feature_container(
        path_dict["features"], path_dict["forest_params"])
    print("Compute feature state for parameterless features")
    feature_container.compute_feature_state(
        timeseries, cfg=None)  # get default feature matrix
    feature_list = feature_container.feature_list
    all_enabled_features = list(
        filter(lambda f: f.enabled, feature_container.feature_list))
    enabled_features = list(
        filter(
            lambda f: f.enabled and len(f.input_parameters) > 0,
            feature_container.feature_list,
        )
    )
    enabled_feature_count = len(enabled_features)
    incr = window_size - overlap
    if incr <= 0:
        raise Exception(
            "FATAL ERROR: Overlap must be smaller than window_size!")
    window_start_pntr = 0
    logger = register_logger()
    tot_it = round(
        min(1, np.ceil(enabled_feature_count / window_size))
        + max(0, np.ceil((enabled_feature_count - window_size) / incr))
    )
    it = 1
    incumbent = {}
    while window_start_pntr < enabled_feature_count:
        for f in filter(lambda f: f.enabled, feature_list):
            f.enabled = False
        cur_window = [
            window_start_pntr,
            min(window_start_pntr + window_size - 1, enabled_feature_count - 1),
        ]
        for i, item in enumerate(enabled_features):
            if cur_window[0] <= i <= cur_window[1] and item in enabled_features:
                item.enabled = True
            else:
                item.enabled = False
        print()
        print(f"Iteration {it}/{tot_it}")
        print(
            "|".join("█" if f.enabled else "░" for f in enabled_features)
            + f" -> {', '.join(f.name for f in feature_container.feature_list if f.enabled)}"
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
    for f in all_enabled_features:
        f.enabled = True
    return incumbent, feature_container
