"""
Created on Wed Jun 22 10:25:29 2022

@author: Noske
"""

from cgi import parse_multipart
from typing import Dict, Union
from enum import Enum
import math
from typing import Callable, Tuple
import numpy as np
from numpy import typing as np_typing
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
from tsfresh.utilities.string_manipulation import convert_to_output_format
from tsfresh.feature_extraction.feature_calculators import set_property
import failure_recognition.signal_processing.signal_helper


def __get_bin_search_dir(fun_x: Callable, y_target: float, x_i: float, pos_slope: bool, epsilon: float) -> int:
    """Eval a single the position in the binary search

    Returns
    -------
    int
         0: target reached
        +1: in upper half
        -1: in lower half
    """

    diff = fun_x(x_i) - y_target
    if abs(diff) <= epsilon:
        return 0
    if diff < 0 and pos_slope or diff > 0 and not pos_slope:
        return 1
    return -1


def cond_max(val: float, min: Union[None, float]):
    """Call max function with val and min if min is not None"""
    if min is None:
        return val
    return max(val, min)


def cond_min(val: float, max: Union[None, float]):
    """Call min function with val and max if max is not None"""
    if max is None:
        return val
    return min(val, max)


def binary_search(
    fun_x: Callable,
    xi_range: Tuple[float, float],
    y_target: float,
    epsilon: float,
    max_iter: int,
    xi_range_0: Tuple[float, float] = None,
    pos_slope: bool = True,
):
    """
    For a monotonous function, find x such that |f(x) - y_target| <= epsilon

    Parameters
    ----------
    fun_x: Callable: function
    xi_range_0: Tuple[float, float]
        (low, high): Start Interval
    xi_range: Tuple[float, float]
        (x_min, x_max)
    y_target: float
        target function value
    epsilon: float
        max. error
    max_iter: int
        max. number of iterations
    pos_slope: bool
        specifies whether the function is monotonically increasing or decreasing

    Returns
    -------
    float
    """
    checked_boundaries = []
    xi_range_0 = tuple(xi_range) if xi_range_0 is None else xi_range_0
    a, b = xi_range_0
    if a is None:
        a = b if b is not None else 0
    b = a if b is None else b
    xi_range_0 = (cond_max(a, xi_range[0]), cond_min(b, xi_range[1]))
    low, high = xi_range_0
    if high == low:
        low, high = cond_max(low - 0.5 * abs(low) - 1,
                             xi_range[0]), cond_min(2 * abs(high) + 1, xi_range[1])
    for _ in range(max_iter):
        x_i = (low + high) / 2
        bin_eval = __get_bin_search_dir(
            fun_x, y_target, x_i, pos_slope, epsilon)
        if bin_eval == 0:
            return x_i
        if bin_eval == 1:
            if high not in checked_boundaries:
                checked_boundaries.append(high)
                if __get_bin_search_dir(fun_x, y_target, high, pos_slope, epsilon) == 1:
                    high = cond_min(2 * abs(high) + 1, xi_range[1])
            low = x_i
        else:
            if low not in checked_boundaries:
                checked_boundaries.append(low)
                if __get_bin_search_dir(fun_x, y_target, high, pos_slope, epsilon) == -1:
                    low = cond_max(low - 0.5 * abs(low) - 1, xi_range[0])
            high = x_i
    return x_i


class FindPeaksMode(Enum):
    PROMINENCE = {"name": "prominence", "pos_slope": False, "range": [0, None]}
    DISTANCE = {"name": "distance", "pos_slope": False, "range": [1, None]}
    THRESHOLD = {"name": "threshold", "pos_slope": False, "range": [0, None]}


def __find_peaks_feature(x, param: Dict[str, str]):
    """Function to calculate a feature vector from fft peaks and combine it with tsfresh parameters """
    param = dict(param)
    ts = param["ts"]
    num_peaks = param["num_peaks"]
    xf, yyf = get_fft(ts, x, 50)
    mode = FindPeaksMode.PROMINENCE
    if param["mode"].lower() == "distance":
        mode = FindPeaksMode.DISTANCE
    elif param["mode"].lower() == "threshold":
        mode = FindPeaksMode.THRESHOLD
    peaks_x, peaks_y = find_signal_peaks(
        xf, yyf, num_peaks=num_peaks, mode=mode, x_0=dict(param["x_0"]), max_iterations=param["max_iterations"])
    if peaks_x.shape[0] < num_peaks:
        peaks_x, peaks_y = np.pad(peaks_x, (num_peaks-peaks_x.shape[0], 0), 'constant'), \
            np.pad(peaks_y, (num_peaks-peaks_y.shape[0], 0), 'constant')
    if peaks_x.shape[0] > num_peaks:
        peaks_x, peaks_y = peaks_x[:num_peaks], peaks_y[:num_peaks]
    peaks_x = peaks_x / np.median(xf)
    peaks_y = peaks_y / np.median(yyf)
    peaks_x_y = np.concatenate((peaks_x, peaks_y), dtype=x.dtype)
    if peaks_x_y.shape != (num_peaks * 2,):
        raise Exception("")
    return peaks_x_y


@set_property("fctype", "combiner")
def find_peaks_feature(x, param):
    """
    Function to calculate a feature vector from fft peaks and combine it with tsfresh parameters 

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param c: the time series name
    :type c: str
    :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
    :type param: list
    :return: list of tuples (s, f) where s are the parameters, serialized as a string,
            and f the respective feature value as bool, int or float
    :return type: pandas.Series
    """
    #return [(convert_to_output_format(config), __find_peaks_feature(x, config)) for config in param]
    peaks_x_y = __find_peaks_feature(x, param[0])
    result = [(convert_to_output_format(param[0].update({"no": i}) or param[0]), peak)  for i, peak in enumerate(peaks_x_y)]
    return result


def get_fft(
    ts: float, signal_t: np_typing.ArrayLike, f_min: float = None, f_max: float = None
) -> Tuple[np_typing.ArrayLike, np_typing.ArrayLike]:
    """Get the fft of the given signal"""
    yf = fft(signal_t)
    N = len(yf)
    xf = fftfreq(N, ts)[: N // 2]
    yyf = 2.0 / N * np.abs(yf[0: N // 2])
    _N = len(xf)
    # just use frequency spectrum between 10 and 50 HZ
    # plt.plot(xf, yyf)
    # plt.show()
    f_min = 0 if f_min is None else f_min
    f_max = xf.max() if f_max is None else f_max

    limited_range = range(int(_N / xf.max() * f_min),
                          int(_N / xf.max() * f_max))
    xf = xf[limited_range]
    yyf = yyf[limited_range]
    return (xf, yyf)


def find_signal_peaks(
    t: np_typing.ArrayLike,
    signal: np_typing.ArrayLike,
    num_peaks: int,
    mode: FindPeaksMode,
    x_0: dict = None,
    xi_range_0: Tuple[float, float] = None,
    max_iterations: int = 20,
) -> Tuple[np_typing.ArrayLike, np_typing.ArrayLike]:
    """Find given (max.) number of peaks in the given signal

    Parameters
    ----------
    t: ArrayLike
        input vector if the signal
    signal: ArrayLike
        signal vector
    num_peaks: int
        max number of peaks that is reached by varying the given peaks finding parameter
    mode: FindPeaksMode
        specifies the variable in numpy's find_peaks method that is optimized to reach the desired number of peaks
    x_0: dict
        (initial) set of key word arguments that is used by numpy's find_peaks method
    number_peaks: int
        range of the optimized parameter
    Returns
    -------
    x_peak: ArrayLike, y_peak: ArrayLike
    """
    x_0 = {} if x_0 is None else x_0
    opt = binary_search(
        lambda p: len(find_peaks(
            signal, **(x_0.update({mode.value["name"]: p}) or x_0))[0]),
        mode.value["range"],
        num_peaks,
        0,
        max_iterations,
        xi_range_0,
        mode.value["pos_slope"],
    )
    all_peaks = find_peaks(
        signal, **(x_0.update({mode.value["name"]: opt}) or x_0))
    peaks = all_peaks[0]
    x_peak = t[peaks]
    y_peak = signal[peaks]
    return x_peak, y_peak


if __name__ == "__main__":

    A = 2.75
    f = 100
    fs = 5000.0
    ts = 1 / fs
    t = np.arange(0, 10, ts)
    sinc_10 = A * np.sinc(f * (t))  # + A*np.sin(2*math.pi*50*t)

    xf, yf = get_fft(ts, sinc_10)
    x_peaks, y_peaks = find_signal_peaks(
        t, sinc_10, 55, FindPeaksMode.PROMINENCE, max_iterations=200)
    plt.plot(x_peaks, y_peaks, marker=11)
    plt.plot(t, sinc_10)
    plt.show()
    pass
