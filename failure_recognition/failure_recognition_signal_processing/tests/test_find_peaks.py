"""module for testing the die casting failure detection"""

import math
import unittest

import numpy as np

from failure_recognition.signal_processing.signal_helper import (
    FindPeaksMode,
    find_signal_peaks,
    get_fft,
)


class FindPeaksTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        self.A = 2.75
        self.f = 1.0
        self.fs = 200.0
        self.ts = 1 / self.fs
        self.t = np.arange(0, 10, self.ts)
        self.peaks_f = [3, 44, 55]
        self.signal = np.zeros(self.t.shape)
        for peak_f in self.peaks_f:
            self.signal += self.A * np.sin(2 * math.pi * peak_f * self.t)
        super().__init__(methodName)

    def test_find_all_peaks(self):
        self.xf, self.yyf = get_fft(self.ts, self.signal, 40, 60)
        for mode in FindPeaksMode:
            x_peaks, y_peaks = find_signal_peaks(
                self.xf,
                self.yyf,
                100,
                mode,
                x_0={"prominence": 0.01, "threshold": 2, "distance": 1},
                max_iterations=500,
            )
            self.assertTrue(len(x_peaks) == len(y_peaks))
            self.assertEqual(len(x_peaks), 2)
            self.assertAlmostEqual(x_peaks[0], 44, 1)
            self.assertAlmostEqual(x_peaks[1], 55, 1)
            self.assertAlmostEqual(y_peaks[0], self.A, 1)
            self.assertAlmostEqual(y_peaks[1], self.A, 1)
            pass

    def test_sinc_peaks(self):
        sinc_10 = self.A * np.sinc(100 * (self.t - math.pi / 4))
        for mode in FindPeaksMode:
            x_peaks, y_peaks = find_signal_peaks(
                self.t, sinc_10, 55, mode, max_iterations=500
            )
            self.assertTrue(len(x_peaks) == len(y_peaks))
            self.assertAlmostEqual(len(x_peaks) / 55.0, 1.0, 1)
            pass


if __name__ == "__main__":
    unittest.main()
