"""Module to start the smac optimization"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import failure_recognition.signal_processing.random_forest_from_cfg as random_forest_from_cfg
from smac.scenario.scenario import Scenario
from failure_recognition.smac_recognizer import PATH_DICT, SCENARIO_DICT

import smac_tsfresh



plt.close("all")
timeseries = pd.read_csv(PATH_DICT["timeSeries"], decimal=".", sep=",", header=0)
test_settings = pd.read_csv(PATH_DICT["testSettings"], decimal=".", sep=",", header=0)
y = pd.read_csv(PATH_DICT["label"], decimal=".", sep=",", header=None)
y = y.iloc[:, 0]

test_settings.index += 1
(incumbent, feature_container) = smac_tsfresh.smac_tsfresh_window_opt(
    timeseries,
    test_settings,
    y,
    PATH_DICT,
    SCENARIO_DICT,
    window_size=2,
    overlap=0,
    seed=np.random.RandomState(42),
    window_size_ratio=0.2,
)

featureState = feature_container.feature_state
history = feature_container.history.reset_index()
y_pred, importances = random_forest_from_cfg.get_prediction(
    incumbent,
    np.random.RandomState(42),
    feature_container,
    timeseries,
    test_settings,
    y,
    timeseries.query("id <= 50"),
    test_settings[:50],
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
