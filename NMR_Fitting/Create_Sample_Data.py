import pandas as pd
import numpy as np
# import random
# import sys
# from scipy import interpolate
# import cmath
import matplotlib.pyplot as plt
# import statistics as std
from scipy.stats import zscore
# import math
from Lineshape import *
from Misc_Functions import *
from Variables import *

df_filtered = exclude_outliers(df_rawsignal_noise)

R_arr = []
P_arr = []
SNR_arr = []
U = 0.1
for x in range(0,10):
    P = np.random.uniform(0,1)
    # Cknob = 0.180 + np.random.uniform(-.07,.07)
    # Cknob = .64 
    Cknob = .042
    cable = 22/2
    eta = 0.0104
    phi = 6.1319
    Cstray = 10**(-15)
    mu = 0.1
    gam = 0.1
    circ_params = (U,Cknob,cable,eta,phi,Cstray)
    noise = choose_random_row(df_filtered)
    lineshape = GenerateLineshape(P)
    # proton = voigt_profile(np.linspace(-1,1,500),.01,.0)
    signal = Signal(circ_constants, circ_params, function_input, scan_s, mu,gam, ranger)
    # shape = (np.array(signal) + np.array(baseline)) - noise
    # signal = baseline
    offset = np.array([x - max(signal) for x in signal])
    # noise = np.zeros(500)
    amplitude_shift = np.ones(500,)
    # sinusoidal_shift = Sinusoidal_Noise(500,)
    # sig = offset + noise + np.multiply(amplitude_shift,np.random.uniform(-0.01,0.01)) + sinusoidal_shift
    sig = offset 
    x_sig = max(list(map(abs, signal)))
    y_sig = max(list(map(abs,noise)))
    SNR = (x_sig/y_sig)
    R_arr.append(sig)
    P_arr.append(np.round(P,6))
    SNR_arr.append(SNR)
df = pd.DataFrame(R_arr)
df['P'] = P_arr
df['SNR'] = SNR_arr
# df.to_csv('Testing_Data_v5/Sample_Data' + str(sys.argv[1]) + '.csv',index=False)
df.to_csv('Test.csv',index=False)
