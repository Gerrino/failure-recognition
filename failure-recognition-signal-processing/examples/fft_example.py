
import math
from matplotlib import pyplot as plt
import numpy as np

from failure_recognition.signal_processing.findpeaks import get_fft


A = 2.75
f = 1.0
fs = 200.0
ts = 1/fs
t = np.arange(0, 10, ts)
peaks_f = [3, 44, 55]
signal = np.zeros(t.shape)
for peak_f in  peaks_f:
    signal += A*np.sin(2*math.pi*peak_f*t)
plt.plot(t, signal)
plt.show()


xf, yyf = get_fft(ts, signal, 40, 60)
plt.plot(xf, yyf)
plt.show()