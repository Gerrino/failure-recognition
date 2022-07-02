import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from failure_recognition.signal_processing import PATH_DICT
from failure_recognition.signal_processing.feature_container import FeatureContainer

from failure_recognition.signal_processing.signal_helper import get_fft

def show_fft():
    A = 2.75
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


    xf, yyf = get_fft(ts, signal, 40, 60)
    plt.plot(xf, yyf)
    plt.show()

def example_prediction():
    plt.close("all")
    timeseries = pd.read_csv(PATH_DICT["timeSeries"], decimal=".", sep=",", header=0)
    test_settings = pd.read_csv(PATH_DICT["testSettings"], decimal=".", sep=",", header=0)
    y = pd.read_csv(PATH_DICT["label"], decimal=".", sep=",", header=None)
    y = y.iloc[:, 0]

    container = FeatureContainer()
    container.load(PATH_DICT["features"], PATH_DICT["forest_params"])
    container.compute_feature_state(timeseries, cfg=None)
    pass

if __name__ == "__main__":
    example_prediction()
    pass
