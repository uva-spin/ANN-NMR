
### This file should be ran with the associated jobscript.sh file within this directory. Adequate amounts of sample data
### are most efficiently created on Rivanna using batch job submissions. Instructions for how to 
### Create large amounts of sample data are present in the .sh file.

import numpy as np
import pandas as pd
from Lineshape import *
from scipy.integrate import quad
from math import log, floor
from tqdm import tqdm
import os 
import sys

Signal_arr = []
Area_arr = []
SNR_arr = []


for i in tqdm(range(1000)):  

    # U = 2.4283e1 + np.random.uniform(-0.01, 0.01)
    # Cknob = .0647 + np.random.uniform(-0.005, 0.005)
    # cable = 22/2
    # eta = 1.04e-2 + np.random.uniform(-0.001, 0.001)
    # phi = 6.1319 + np.random.uniform(-0.1, 0.1)
    # Cstray = 10**(-15)
    # shift = -2.0464e-2 + np.random.uniform(-0.001, 0.001)
    # sig = 0.01 + np.random.uniform(-0.009, 0.1)              # Gaussian width variation
    # gam = 0.01 + np.random.uniform(-0.009, 0.1)              # Lorentzian width variation
    # amp = .002 + np.random.uniform(-0.0019, 0.005)
    # center = 213 + np.random.uniform(-.3, .3)

    U = 2.4283e1 
    Cknob = .0647 
    cable = 22/2
    eta = 1.04e-2 
    phi = 6.1319
    Cstray = 10**(-15)
    shift = -2.0464e-2
    sig = 0.01            
    gam = 0.01         
    amp = .002 + np.random.uniform(-0.0019, 0.005)
    center = 213 

    x, lower_bound, upper_bound = FrequencyBound(212.6)

    signal = Voigt(x, amp, sig, gam, center)

    baseline = Baseline(x, U, Cknob, eta, cable, Cstray, phi, shift)

    combined_signal = signal + baseline

    noise = np.random.normal(0, 0.0005, size=x.shape)
    # noise = 0

    noisy_signal = combined_signal + noise

    x_sig = np.max(np.abs(combined_signal))
    y_sig = np.max(np.abs(noise))
    SNR = x_sig / y_sig

    area, _ = quad(Voigt, lower_bound, upper_bound, args=(amp, sig, gam, center))

    Signal_arr.append(noisy_signal)
    Area_arr.append(area)
    SNR_arr.append(SNR)

df = pd.DataFrame(Signal_arr)
df['Area'] = Area_arr
df['SNR'] = SNR_arr

file_path = 'Training_Data'

os.makedirs(file_path, exist_ok = True)

try:
    df.to_csv(os.path.join(file_path,'Testing_Data_' + str(sys.argv[1]) + '.csv'), index=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")