# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:25:29 2022

@author: Noske
"""

# find peaks

#Number of dominant peaks to find
number_peaks = 5

dom_peaks_freq = []
dom_peaks_ampl = []

for i in range(0, Num_Pro):

    N = 1000
    T = .01
    yf = fft(train_data[:,i])
    xf = fftfreq(N, T)[:N//2]
    yyf = 2.0/N * np.abs(yf[0:N//2])
    
    #just use frequency spectrum between 10 and 50 HZ
    
    xf = xf[100::]
    yf = yf[100::]
    yyf = yyf[100::]
    
    # find the dominant peaks
    
      
    k = 0
    
    prom = 0.1
    
    while k == 0:
        
        peaks, _ = find_peaks(yyf, prominence=prom)
        prom -= 0.0001
        print(prom)
        
        if len(peaks) >= number_peaks:
            k =1
        #end if
        
        if prom <= 0:
            peaks = np.zeros(number_peaks, dtype=int)
            k = 1
        #end if
        
    #end while
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