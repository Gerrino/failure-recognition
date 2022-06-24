"""
Created on Wed Jun 22 10:25:29 2022

@author: Noske
"""

import math
from typing import Tuple
import numpy as np
from numpy import typing as np_typing
from numpy.fft import fft, fftfreq 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def get_fft(ts: float, signal_t: np_typing.ArrayLike, f_min: float = None, f_max: float = None) -> Tuple[np_typing.ArrayLike, np_typing.ArrayLike]:
    """Get the fft of the given signal
    """
    yf = fft(signal_t)
    N = len(yf)
    xf = fftfreq(N, ts)[:N//2]
    yyf = 2.0/N * np.abs(yf[0:N//2])
    _N = len(xf)
    #just use frequency spectrum between 10 and 50 HZ
    #plt.plot(xf, yyf)
    #plt.show()
    f_min = 0 if f_min is None else f_min
    f_max = xf.max() if f_max is None else f_max

    limited_range = range(int(_N/xf.max()*f_min), int(_N/xf.max()*f_max))
    xf = xf[limited_range]
    yyf = yyf[limited_range]
    return (xf, yyf)

def find_peaks_fft(ts: float, train_data: np_typing.ArrayLike, number_peaks: int = 5):
    """Find given (max.) number of peaks in the given signal

    Parameters
    ----------
    train_data: np_typing.ArrayLike
    number_peaks: int
    """
    dom_peaks_freq = []
    dom_peaks_ampl = []    
    
    xf, yyf = get_fft(ts, train_data, 45, 55)

    # find the dominant peaks


    k = 0

    prom = 0.1

    while k == 0:

        peaks, _ = find_peaks(yyf, prominence=prom)
        prom -= 0.0001
        print(prom)

        if len(peaks) >= number_peaks:
            k = 1


        if prom <= 0:
            peaks = np.zeros(number_peaks, dtype=int)
            k = 1

    x_peak = xf[peaks]
    y_peak = yyf[peaks]

    peaks_freq = x_peak[0:number_peaks]
    peaks_ampl = y_peak[0:number_peaks]

    dom_peaks_freq.append(peaks_freq)
    dom_peaks_ampl.append(peaks_ampl)


    # end for
    dom_peaks_freq = np.array(dom_peaks_freq)
    dom_peaks_freq = dom_peaks_freq[:,0:number_peaks]

    dom_peaks_ampl = np.array(dom_peaks_ampl)
    dom_peaks_ampl = dom_peaks_ampl[:,0:number_peaks]


if __name__ == "__main__":
    A = 2.75
    f = 1.0
    fs = 200.0
    ts = 1/fs
    t = np.arange(0, 10, ts)
    sinc_10 = A*np.sin(2*math.pi*f*t) + A*np.sin(2*math.pi*50*t)
   
    plt.plot(t, sinc_10)
    plt.show()
    find_peaks_fft(ts, sinc_10)