import numpy as np
import pandas as pd
from Lineshape import *
from scipy.integrate import quad
from tqdm import tqdm
import os 

# Initialize arrays to store results
Signal_arr = []
Area_arr = []
SNR_arr = []

# Loop to generate data with varying parameters
for i in tqdm(range(50000)):

    U = 2.4283e1 + np.random.uniform(-0.01, 0.01)
    Cknob = .0647 + np.random.uniform(-0.005, 0.005)
    cable = 22/2
    eta = 1.04e-2 + np.random.uniform(-0.001, 0.001)
    phi = 6.1319 + np.random.uniform(-0.1, 0.1)
    Cstray = 10**(-15)
    shift = -2.0464e-2 + np.random.uniform(-0.001, 0.001)
    sig = 0.01 + np.random.uniform(-0.009, 0.1)              # Gaussian width variation
    gam = 0.01 + np.random.uniform(-0.009, 0.1)              # Lorentzian width variation
    amp = .002 + np.random.uniform(-0.0019, 0.005)
    center = 213 + np.random.uniform(-.3, .3)

    # Generate x values
    x, lower_bound, upper_bound = FrequencyBound(212.6)

    # Generate signal using Voigt profile with varying parameters
    signal = Voigt(x, amp, sig, gam, center)

    # Generate baseline with varying parameters
    baseline = Baseline(x, U, Cknob, eta, cable, Cstray, phi, shift)

    # Combine signal and baseline
    combined_signal = signal + baseline

    # Generate some noise (e.g., Gaussian noise)
    noise = np.random.normal(0, 0.0005, size=x.shape)
    # noise = 0

    # Combine signal, baseline, and noise
    noisy_signal = combined_signal + noise

    # Calculate SNR
    x_sig = np.max(np.abs(combined_signal))
    y_sig = np.max(np.abs(noise))
    SNR = x_sig / y_sig

    # Calculate the area under the Voigt curve
    area, _ = quad(Voigt, lower_bound, upper_bound, args=(amp, sig, gam, center))

    # Store the results
    Signal_arr.append(noisy_signal)
    Area_arr.append(area)
    SNR_arr.append(SNR)

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(Signal_arr)
df['Area'] = Area_arr
df['SNR'] = SNR_arr
# Save DataFrame to CSV

file_path = 'Testing_Data'

os.makedirs(file_path, exist_ok = True)

try:
    df.to_csv(os.path.join(file_path,'Sample_Testing_Data.csv'), index=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")