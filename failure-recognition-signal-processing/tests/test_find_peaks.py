"""module for testing the die casting failure detection"""

import math
import unittest

import numpy as np


class FindPeaksTest(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        self.A = 2.75
        self.f = 1.0
        self.fs = 200.0
        self.ts = 1/self.fs
        self.t = np.arange(0, 10, self.ts)
        self.peaks_f = [3, 44, 55]
        self.signal = np.zeros(self.t.shape)
        for peak_f in  self.peaks_f:
            self.signal += self.A*np.sin(2*math.pi*peak_f*self.t)
        super().__init__(methodName)

    

    def test_find_peaks(self):
        ...
       

if __name__ == '__main__':
    unittest.main()