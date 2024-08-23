import numpy as np
import pandas as pd
from Lineshape import *
from scipy.integrate import quad


# Initialize arrays to store results
Signal_arr = []
Area_arr = []
SNR_arr = []

# Loop to generate data with varying parameters
for i in range(10):
    # Randomly vary the parameters slightly for both Voigt and Baseline
    # amp = 0.00005 + np.random.uniform(-0.00001, 0.00001)  # Amplitude variation
    # center = np.random.uniform(-0.05, 0.05)               # Center shift

    # U = 0.1 + np.random.uniform(-0.01, 0.01)
    # Cknob = 0.042 + np.random.uniform(-0.005, 0.005)
    # eta = 0.0104 + np.random.uniform(-0.001, 0.001)
    # cable = 22/2 + np.random.uniform(-1, 1)
    # Cstray = 10**(-15)  # Keeping this constant
    # phi = 6.1319 + np.random.uniform(-0.1, 0.1)
    # shift = np.random.uniform(-0.01, 0.01)
    U = 2.4283e1 + np.random.uniform(-0.01, 0.01)
    Cknob = .0647 + np.random.uniform(-0.005, 0.005)
    cable = 22/2
    eta = 1.04e-2 + np.random.uniform(-0.001, 0.001)
    phi = 6.1319 + np.random.uniform(-0.1, 0.1)
    Cstray = 10**(-15)
    shift = -2.0464e-2 + np.random.uniform(-0.001, 0.001)
    sig = 0.01 + np.random.uniform(0.001, 0.1)              # Gaussian width variation
    gam = 0.01 + np.random.uniform(0.001, 0.1)              # Lorentzian width variation
    amp = .005 + np.random.uniform(-0.00001, 0.0001)
    center = 213

    # Generate x values
    x, lower_bound, upper_bound = FrequencyBound(212.6)

    # Generate signal using Voigt profile with varying parameters
    signal = Voigt(x, amp, sig, gam, center)

    # Generate baseline with varying parameters
    baseline = Baseline(x, U, Cknob, eta, cable, Cstray, phi, shift)

    # Combine signal and baseline
    combined_signal = signal + baseline

    # Generate some noise (e.g., Gaussian noise)
    # noise = np.random.normal(0, 0.000001, size=x.shape)
    noise = 0

    # Combine signal, baseline, and noise
    noisy_signal = combined_signal + noise

    # Calculate SNR
    x_sig = np.max(np.abs(combined_signal))
    y_sig = np.max(np.abs(noise))
    SNR = x_sig / y_sig

    # Calculate the area under the Voigt curve
    area, _ = quad(Voigt, lower_bound, upper_bound, args=(amp, s, g, center))

    # Store the results
    Signal_arr.append(noisy_signal)
    Area_arr.append(area)
    SNR_arr.append(SNR)

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(Signal_arr)
df['Area'] = Area_arr
df['SNR'] = SNR_arr
# Save DataFrame to CSV
try:
    df.to_csv(r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NM4-Fermilab\data\Generated_Sample_Data_with_Area.csv', index=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# Optionally plot one of the generated signals
import matplotlib.pyplot as plt
plt.plot(x, noisy_signal)
plt.title('Generated Signal with Noise')
plt.xlabel('x')
plt.ylabel('Signal + Noise')
plt.show()
