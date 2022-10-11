#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:16:42 2021

@author: gerritnoske
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import time
import pandas as pd

from failure_recognition.signal_processing.feature_container import FeatureContainer


def rf_from_cfg_extended(
    cfg,
    seed,
    timeseries: pd.DataFrame,
    test_settings: pd.DataFrame,
    y: pd.DataFrame,
    feature_container: FeatureContainer,
    window_size_ratio: float = None,
) -> float:
    """
    Creates a random forest regressor from sklearn and fits the given data on it.
    This is the function-call we can try to optimize (smac). Chosen values are stored in
    the configuration (cfg).vnc

    Parameters:
    -----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        used to initialize the rf's random generator

    Returns:
    -----------
    np.mean(rmses): float
        mean of root mean square errors of random-forest test predictions
        per cv-fold
    """
    max_time = max(timeseries["time"])
    windowed_time_series, window_left, window_right = timeseries, 0, max_time   
    if window_size_ratio is not None:
        window_size = window_size_ratio * max_time
        window_left = cfg["window_offset_percent"] / 100.0 * (max_time - window_size)
        window_right = window_left + window_size
        windowed_time_series = timeseries.query(f"time >= {window_left} and time <= {window_right}")
    rfr = get_rfr(cfg, seed)
    # features
    start_time = time.time()
    feature_container.compute_feature_state(windowed_time_series, cfg)

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y) ** 2))

    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    feature_matrix = pd.concat([feature_container.feature_state.reset_index(drop=True), test_settings.reset_index(drop=True)], axis=1)
    print("Shape feature matrix: " + str(feature_matrix.shape))
    score = cross_val_score(rfr, feature_matrix, y, cv=10, scoring=rmse_scorer)
    duration = time.time() - start_time
    print(f"score: {score}")
    print(f"Eval FeatureState: ({duration})s " + str(feature_container.feature_state.columns))
    print("")
    print("**")
    print(f"size of windowedTimeSeries {len(windowed_time_series)}")
    if (win_size:=cfg.get('window_size_percent')) is not None:
        print(
            f"Size {win_size} Duration {round(duration)}s, window_left:{window_left}, "
            f"window_right:{window_right}/{max_time}"
        )
    print("**")
    cost = -1 * np.mean(score)  # + 0.01 * duration
    print(f"cost: {cost}")
    return cost  # Because cross_validation sign-flips the score


def get_prediction(
    cfg: dict,
    seed: int,
    feature_container: FeatureContainer,
    x_train: pd.DataFrame,
    test_settings_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    test_settings_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, any]:
    """
    Calculate the feature state given the training data, and the test data for all features.
    Fit a random forest regressor to the feature state (training set) and return the predicted output for the test data
    
    Returns
    ---
    y_pred: DataFrame, importances"""
    rfr = get_rfr(cfg, seed)
    feature_container.reset_feature_state()
    feature_container.compute_feature_state(x_train, cfg, compute_for_all_features=True)
    feature_matrix_train = pd.concat([feature_container.feature_state, test_settings_train], axis=1)
    rfr.fit(feature_matrix_train, y_train)
    feature_container.reset_feature_state()
    feature_container.compute_feature_state(x_test, cfg, compute_for_all_features=True)
    feature_matrix_test = pd.concat([feature_container.feature_state, test_settings_test], axis=1)
    y_pred = rfr.predict(feature_matrix_test)
    importances = rfr.steps[1][1].feature_importances_
    return y_pred, importances


def get_rfr(cfg: dict, seed: int) -> Pipeline:
    """
    Create a random forest regrssion pipeline

    Returns
    ---
    p: pipeline
    """
    return make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=cfg["num_trees"],
            criterion=cfg["criterion"],
            min_samples_split=cfg["min_samples_to_split"],
            min_samples_leaf=cfg["min_samples_in_leaf"],
            min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
            # max_features=cfg["max_features"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            bootstrap=cfg["do_bootstrapping"],
            random_state=seed,
        ),
    )
