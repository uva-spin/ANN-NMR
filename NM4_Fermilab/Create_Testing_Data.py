
### This file should be ran as:
# C:> python Create_Testing_Data.py <number of data events>

import numpy as np
import pandas as pd
from Lineshape import *
from scipy.integrate import quad
from math import log, floor
from tqdm import tqdm
import os 
import sys

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000
    magnitude = int(floor(log(number, k)))
    return '%i%s' % (number / k**magnitude, units[magnitude])

Signal_arr = []
P_arr = []
#Area_arr = []
SNR_arr = []

for i in tqdm(range(int(sys.argv[1]))):

    U = 2.4283 + np.random.uniform(-0.01, 0.01)
    Cknob = .0647 + np.random.uniform(-0.005, 0.005)
    cable = 22/2
    eta = 1.04e-2 + np.random.uniform(-0.001, 0.001)
    phi = 6.1319 + np.random.uniform(-0.1, 0.1)
    Cstray = 10**(-15)
    shift = -2.0464e-2 + np.random.uniform(-0.001, 0.001)

    sig = 0.1 + np.random.uniform(-0.009, 0.001)       
    gam = 0.1 + np.random.uniform(-0.009, 0.001)         
    amp = .005 + np.random.uniform(-0.005, 0.01)
    center = 213 + np.random.uniform(-.1, .1)

    x, lower_bound, upper_bound = FrequencyBound(212.6)

    # signal = Voigt(x, amp, sig, gam, center)

    P = np.random.uniform(0,1)
    
    signal = GenerateLineshape(P)/100.0

    baseline = Baseline(x, U, Cknob, eta, cable, Cstray, phi, shift)
    # baseline = -0.001717*x **2 + 0.732*x - 77.99

    
    combined_signal = signal + baseline

    noise = np.random.normal(0, 0.0005, size=x.shape)
    # noise = 0

    noisy_signal = combined_signal + noise

    x_sig = np.max(np.abs(combined_signal))
    y_sig = np.max(np.abs(noise))
    SNR = x_sig / y_sig

    area, _ = quad(Voigt, lower_bound, upper_bound, args=(amp, sig, gam, center))

    Signal_arr.append(noisy_signal)
    P_arr.append(P)
    #Area_arr.append(area)
    SNR_arr.append(SNR)

df = pd.DataFrame(Signal_arr)
df['P'] = P_arr
#df['Area'] = Area_arr
df['SNR'] = SNR_arr

file_path = '/home/devin/Documents/Big_Data/Testing_Data_Deuteron'

os.makedirs(file_path, exist_ok = True)

version = 'v4'

try:
    df.to_csv(os.path.join(file_path,f'Sample_Testing_Data_{version}_' + str(human_format(int(sys.argv[1])) + '.csv')), index=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")