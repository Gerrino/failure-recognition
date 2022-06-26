"""Module to start the smac optimization"""

import matplotlib.pyplot as plt
import pandas as pd
import SmacTsfresh
from failure_recognition.smac_recognizer import PATH_DICT, SCENARIODICT
from smac.scenario.scenario import Scenario
import numpy as np
import RandomForestFromCFG

plt.close("all")
timeseries = pd.read_csv(PATH_DICT["timeSeries"], decimal=".", sep=",", header=0)
testSettings = pd.read_csv(PATH_DICT["testSettings"], decimal=".", sep=",", header=0)
y = pd.read_csv(PATH_DICT["label"], decimal=".", sep=",", header=None)
y = y.iloc[:, 0]

testSettings.index += 1
(incumbent, featureContainer) = SmacTsfresh.smac_tsfresh_window_opt(
    timeseries,
    testSettings,
    y,
    PATH_DICT,
    SCENARIODICT,
    window_size=2,
    overlap=0,
    seed=np.random.RandomState(42),
    window_size_ratio=0.2,
)
featureState = featureContainer.FeatureState
history = featureContainer.History.reset_index()
y_pred, importances = RandomForestFromCFG.getPrediction(
    incumbent,
    np.random.RandomState(42),
    featureContainer,
    timeseries,
    testSettings,
    y,
    timeseries.query("id <= 50"),
    testSettings[:50],
)

optDF = pd.DataFrame(history, columns=["opt-value"])
optDF["opt-value"][0] = history["orig-value"][1]
optDF["T"] = history["datetime"]
optDF.plot(x="T", y="opt-value")

print()
print("_____________________________________")
print()
print(f"Windowed Optimization finished in {(history['datetime'].max() - history['datetime'].min()).seconds} seconds!")
print()
print(incumbent)
