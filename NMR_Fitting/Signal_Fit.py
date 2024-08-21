import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Misc_Functions import *
from Lineshape import *

file_path = find_file('2024-02-07_14h58m46s-RawSignal.csv')

if file_path:
    baseline_data = pd.read_csv(file_path)
    print(f"File found: {file_path}")
else:
    print("File not found.")


Larmor_Frequency = 212.6

Frequency = FrequencyBound(Larmor_Frequency)
# Define the domain to fit (bins 100 to 400)
fit_start_bin, fit_end_bin = 0, 500
y_full = baseline_data.iloc[0, 1:501].values 
yerr_full = 0.0005  
y = y_full[fit_start_bin:fit_end_bin+1]


# Define initial parameter guesses for curve_fit
# U=0.37 Cknob=13.6 eta=0.707 trim=25.2 Cstray=2.272 phi_const=1.42
# initial_params = [0.382652652, 13.6565062, 9.85632350e-02, 24.3720560, 400, 265.575865,-0.018]


# Set bounds for the parameters
#param_bounds = ([.1, 0.01, 0.001, 5, 1e-15, 0],  # Lower bounds
#                [1.8, 30, 1, 15, 50, 3])  # Upper bounds
param_bounds = ([.05, 0.01, 0.001, 1, 1e-15, 0,-0.1],  # Lower bounds
                [0.4, 30, 1, 30, 400, 360, -0.001])  # Upper bounds

# Define initial parameter guesses
initial_params = {
    "U": 0.382652652,
    "Cknob": 13.6565062,
    "eta": 9.85632350e-02,
    "trim": 24.3720560,
    "Cstray": 400,
    "phi_const": 265.575865,
    "DC_offset": -0.018,
}


p0 = list(initial_params.values())
print(p0)


# Perform the curve fitting with bounds
popt, pcov = curve_fit(Baseline, Frequency, y, p0 = p0, bounds = param_bounds)

# Extract parameter names
param_names = list(initial_params.keys())

print_optimized_params_with_names(popt, param_names)
print_cov_matrix_with_param_names(pcov, param_names)

# Plot the original data and the resulting fit function
plt.errorbar(Frequency, y_full, yerr=yerr_full, fmt='o', markersize=1, label='Data', color='black')  # Plot full data range
# plt.plot(x_freq, Baseline(x_freq, *popt), label='Fit', color='red', linewidth=2)  # Fit for selected domain
plt.plot(Frequency, Baseline(Frequency, initial_params.keys()), label='Fit', color='red', linewidth=2)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Dependent Variable')
plt.legend()
plt.title('Fit of Data between Frequencies')
plt.show()
