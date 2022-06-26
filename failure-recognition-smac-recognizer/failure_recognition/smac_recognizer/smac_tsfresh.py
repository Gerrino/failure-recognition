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
from random_forest_from_cfg import rf_from_cfg_extended
from smac.initial_design.latin_hypercube_design import LHDesign
from feature_container import FeatureContainer
import pandas as pd
import datetime


def register_decision_tree_hyperparams(cs, incumbent: dict = None):
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
    rf_from_cfg = lambda cfg, seed: (
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


def load_feature_container(path_feat_list: Union[Path, str]):
    """Create the feature container using a tsfresh feature list"""
    feature_container = FeatureContainer()
    feature_container.load(path_feat_list)
    print()
    print(f"There are {sum(1 for f in feature_container.feature_list if f.enabled)} enabled Features!")
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


def create_configuration_space(feature_container, sensors):
    cs = ConfigurationSpace()
    register_decision_tree_hyperparams(cs, feature_container.incumbent)
    window_size_percent = UniformIntegerHyperparameter("window_size_percent", 10, 100, default_value=50)
    window_offset_percent = UniformIntegerHyperparameter("window_offset_percent", 0, 90, default_value=0)
    cs.add_hyperparameters([window_size_percent, window_offset_percent])
    feature_container.register_hyperparameters(cs, sensors)
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
    num_opt_feat = sum(1 for f in feature_list if f.enabled and len(f.input_parameters) > 0)
    name_opt_feat = ", ".join(f.name for f in feature_container.feature_list if f.enabled)
    for sensor in sensors:
        for f in filter(
            lambda f: f.enabled and len(f.input_parameters) > 0,
            feature_container.feature_list,
        ):  # drop existing default values of enabled features
            print("will it drop")
            drop_name = None
            for col_name in feature_container.feature_state.columns:
                drop_name = col_name if col_name.startswith(f"{sensor}__{f.name}__") else drop_name
            if drop_name is not None:
                print("try dropping" + drop_name)
                feature_container.feature_state = feature_container.feature_state.drop(drop_name, axis=1)
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
    feature_container.history = feature_container.history.append(new_history_row)
    feature_container.compute_feature_state(timeseries, incumbent)  # get parameterless feature matrix
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
    print(f"Starting Optimization window_size: {window_size}, overlap: {overlap}")
    feature_container = load_feature_container(path_dict["features"])
    print("Compute feature state for parameterless features")
    feature_container.compute_feature_state(timeseries, cfg=None)  # get default feature matrix
    feature_list = feature_container.feature_list
    all_enabled_features = list(filter(lambda f: f.enabled, feature_container.feature_list))
    enabled_features = list(
        filter(
            lambda f: f.enabled and len(f.input_parameters) > 0,
            feature_container.feature_list,
        )
    )
    enabled_feature_count = len(enabled_features)
    incr = window_size - overlap
    if incr <= 0:
        raise Exception("FATAL ERROR: Overlap must be smaller than window_size!")
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
