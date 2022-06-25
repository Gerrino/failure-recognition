#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:45:56 2021

@author: gerritnoske
"""
import matplotlib.pyplot as plt
import pandas as pd
import SmacTsfresh
from smac.scenario.scenario import Scenario
import numpy as np
import RandomForestFromCFG
plt.close("all")
pathDict = {
    "features" : "/home/gerritnoske/Documents/Projekt_Henne/tsfreshlist.txt",
    "timeSeries": '/home/gerritnoske/Documents/Projekt_Henne/NASA_tsfresh_format_all.csv',
    "label" : '/home/gerritnoske/Documents/Projekt_Henne/label.csv',
    "testSettings" : '/home/gerritnoske/Documents/Projekt_Henne/testsettings.csv'
    }


timeseries = pd.read_csv(pathDict["timeSeries"], decimal='.', sep=',', header=0)
testSettings = pd.read_csv(pathDict["testSettings"], decimal='.', sep=',', header=0)
y = pd.read_csv(pathDict["label"], decimal='.', sep=',', header=None)
y = y.iloc[:,0] 

scenarioDict = { "run_obj": "quality",  # we optimize quality (alternative runtime)
                 "runcount-limit": 20,  # max. number of function evaluations; for this example set to a low number
                 "deterministic": "true",
                 "memory_limit": 3072,  # adapt this to reasonable value for your hardware               
                 #"cutoff": .1,                 
                 }
testSettings.index += 1
(incumbent, featureContainer) = SmacTsfresh.smac_tsfresh_window_opt(timeseries, testSettings, y, pathDict, scenarioDict, window_size = 2, overlap = 0, seed = np.random.RandomState(42),
                                                                    window_size_ratio = 0.2)
featureState = featureContainer.FeatureState
history = featureContainer.History.reset_index()
y_pred, importances = RandomForestFromCFG.getPrediction(incumbent, np.random.RandomState(42), featureContainer, timeseries, testSettings, y, timeseries.query("id <= 50"), testSettings[:50])

optDF = pd.DataFrame(history, columns=["opt-value"])
optDF["opt-value"][0] = history["orig-value"][1]
optDF["T"] = history["datetime"]
optDF.plot(x = "T", y="opt-value")
print()
print("_____________________________________")
print()
print(f"Windowed Optimization finished in {(history['datetime'].max()-history['datetime'].min()).seconds} seconds!")
print()
print(incumbent)