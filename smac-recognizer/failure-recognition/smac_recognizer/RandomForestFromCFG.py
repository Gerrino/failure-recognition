#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:16:42 2021

@author: gerritnoske
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import time
import pandas as pd

def rf_from_cfg_extended(cfg, seed, timeseries, testSettings, y, feature_container, window_size_ratio):
    """
        Creates a random forest regressor from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
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
    maxTime = max(timeseries["time"])
    windowSize = window_size_ratio*maxTime#cfg["windowSizePercent"] / 100.0 * maxTime
    windowLeft = cfg["windowOffsetPercent"] / 100.0 * (maxTime - windowSize)
    windowRight = windowLeft + windowSize
    windowedTimeSeries = timeseries.query(f"time >= {windowLeft} and time <= {windowRight}")
    rfr = getRFR(cfg, seed)    
    #features
    startTime = time.time()
    feature_container.computeFeatureState(windowedTimeSeries, cfg) 
    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y) ** 2))
    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    feature_matrix = pd.concat([feature_container.FeatureState, testSettings], 1)
    print("Len Featmat" + str(len(feature_matrix)))
    score = cross_val_score(rfr, feature_matrix, y, cv=10, scoring=rmse_scorer)
    duration = time.time() - startTime
    print("")
    print(f"Eval FeatureState: ({duration})s " + str(feature_container.FeatureState.columns))
    print("")
    print("**")
    print(f"size of windowedTimeSeries {len(windowedTimeSeries)}")
    print(f"Size {cfg['windowSizePercent']} Duration {round(duration)}s, window_left:{windowLeft}, window_right:{windowRight}/{maxTime}")
    print("**")
    cost = -1 * np.mean(score)# + 0.01 * duration
    return cost  # Because cross_validation sign-flips the score

def getPrediction(cfg, seed, feature_container, x_train, testSettings_train, y_train, x_test, testSettings_test):
    rfr = getRFR(cfg, seed)
    feature_container.resetFeatureState()
    feature_container.computeFeatureState(x_train, cfg, computeForAllFeatures = True)
    feature_matrix_train = pd.concat([feature_container.FeatureState, testSettings_train], 1)
    rfr.fit(feature_matrix_train, y_train)
    feature_container.resetFeatureState()
    feature_container.computeFeatureState(x_test, cfg, computeForAllFeatures = True)
    feature_matrix_test = pd.concat([feature_container.FeatureState, testSettings_test], 1)
    y_pred = rfr.predict(feature_matrix_test)
    importances = rfr.steps[1][1].feature_importances_
    return y_pred, importances

def getRFR(cfg, seed):
    return make_pipeline(        
        StandardScaler(),        
        RandomForestRegressor(
        n_estimators=cfg["num_trees"],
        criterion=cfg["criterion"],
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        #max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed
        )
    )