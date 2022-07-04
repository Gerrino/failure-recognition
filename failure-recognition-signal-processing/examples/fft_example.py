import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from failure_recognition.signal_processing import PATH_DICT
from failure_recognition.signal_processing import feature_container
from failure_recognition.signal_processing.feature_container import FeatureContainer
from failure_recognition.signal_processing.random_forest_from_cfg import rf_from_cfg_extended

from failure_recognition.signal_processing.signal_helper import FindPeaksMode, get_fft, find_signal_peaks


def show_fft():
    A = 2.756
    f = 1.0
    fs = 200.0
    ts = 1 / fs
    t = np.arange(0, 10, ts)
    peaks_f = [3, 44, 55]
    signal = np.zeros(t.shape)
    for peak_f in peaks_f:
        signal += A * np.sin(2 * math.pi * peak_f * t)
    plt.plot(t, signal)
    plt.show()

    t = np.arange(0, 10, ts)
    xf, yyf = get_fft(ts, signal, 40, 60)

    peaks_x, peaks_y = find_signal_peaks(xf, yyf, 5, FindPeaksMode.DISTANCE)
    plt.plot(xf, yyf)
    plt.show()


def example_prediction():
    plt.close("all")
    timeseries = pd.read_csv(
        PATH_DICT["timeSeries"], decimal=".", sep=",", header=0)
    test_settings = pd.read_csv(
        PATH_DICT["testSettings"], decimal=".", sep=",", header=0)
    y = pd.read_csv(PATH_DICT["label"], decimal=".", sep=",", header=None)
    y = y.iloc[:, 0]

    container = FeatureContainer()
    container.load(PATH_DICT["features"], PATH_DICT["forest_params"])
    #container.compute_feature_state(timeseries, cfg=None)
    cfg = {p.name: p.get_default_value()
           for p in container.random_forest_params}
    rf_from_cfg_extended(cfg, np.random.seed(
        42), timeseries, test_settings, y, container)

    pass


if __name__ == "__main__":
    # show_fft()
    example_prediction()
    pass
