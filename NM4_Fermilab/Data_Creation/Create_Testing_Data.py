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

def generate_signals(i):
    U = 2.4283e1 + np.random.uniform(-1, 1)
    Cknob = .0647 + np.random.uniform(-0.01, 0.01)
    cable = 22/2
    eta = 1.04e-2 + np.random.uniform(-0.001, 0.001)
    phi = 6.1319
    Cstray = 10**(-15)
    shift = -2.0464e-2 + np.random.uniform(-0.001, 0.001)

    x, lower_bound, upper_bound = FrequencyBound(32.32)
    P = np.random.uniform(.01, 1)

    signal = GenerateLineshape(P)/10.0
    baseline = Baseline(x, U, Cknob, eta, cable, Cstray, phi, shift)
    combined_signal = signal + baseline

    noise = np.random.normal(0, 0.000002, size=x.shape)
    noisy_signal = combined_signal + noise

    x_sig = np.max(np.abs(combined_signal))
    y_sig = np.max(np.abs(noise))
    SNR = x_sig / y_sig

    return noisy_signal, P, SNR

Signal_arr, P_arr, SNR_arr = zip(*list(map(generate_signals, tqdm(range(int(sys.argv[1]))))))

df = pd.DataFrame(Signal_arr)
df['P'] = P_arr
df['SNR'] = SNR_arr

file_path = '/home/devin/Documents/Big_Data/Testing_Data_Deuteron'
# file_path = '/home/ptgroup/Documents/Devin/Big_Data/Testing_Data_Deuteron'
os.makedirs(file_path, exist_ok=True)

version = 'Deuteron_v7'

try:
    df.to_csv(os.path.join(file_path, f'Sample_Testing_Data_{version}_' + str(human_format(int(sys.argv[1]))) + '.csv'), index=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")
