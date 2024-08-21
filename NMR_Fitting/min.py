import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

def find_file(filename, start_dir='.'):
    current_dir = os.path.abspath(start_dir)
    levels_up = 0
    
    while levels_up <= 2:  # Limit to two directory levels up
        # Search in the current directory and its subdirectories
        for file in glob.glob(os.path.join(current_dir, '**', filename), recursive=True):
            return file
        
        # Move up one directory level
        current_dir = os.path.abspath(os.path.join(current_dir, '..'))
        levels_up += 1
    
    return None

file_path = find_file('single_event_data.csv')

if file_path:
    baseline_data = pd.read_csv(file_path)
    print(f"File found: {file_path}")
else:
    print("File not found.")

# Print optimized parameters
def print_optimized_params_with_names(params, param_names):
    print("Optimized Parameters:")
    for name in param_names:
        print(f"{name:>16}: {params[name]:16.5e} ± {param_errors[name]:16.5e}")

# Print approximate covariance matrix
def print_cov_matrix_with_param_names(minuit_obj, param_names):
    print("Approximate Covariance Matrix (with parameter names):")
    cov_matrix = minuit_obj.covariance
    if cov_matrix is None:
        print("Covariance matrix not available.")
        return
    
    # Check if cov_matrix is 1D and reshape it if necessary
    if cov_matrix.ndim == 1:
        size = len(param_names)
        cov_matrix = cov_matrix.reshape((size, size))
    
    # Print formatted covariance matrix
    print(f"{'':>16}  " + "  ".join(f"{name:>16}" for name in param_names))
    for i, row in enumerate(cov_matrix):
        row_label = param_names[i]
        formatted_row = "  ".join(f"{value:16.5e}" for value in row)
        print(f"{row_label:>16}  {formatted_row}")

# Load the baseline data from the CSV file
baseline_data = pd.read_csv('../data/single_event_data.csv')

# Define the domain to fit (bins 100 to 400)
fit_start_bin, fit_end_bin = 0, 500

# Frequency conversion factors
bin_to_freq = 0.0015287  # MHz per bin
start_freq = 212.6  # Starting frequency in MHz

# Create an independent variable (frequency in MHz) and extract the relevant data
x_full_bins = np.arange(500)  # Full range of bins
x_full_freq = start_freq + x_full_bins * bin_to_freq  # Convert bins to frequency

y_full = baseline_data.iloc[0, 1:501].values  # Full range of data
yerr_full = 0.0005  # Assuming a constant error, or replace with an array if errors are variable

# Restrict to the domain of interest (bins 100 to 400)
x_bins = x_full_bins[fit_start_bin:fit_end_bin+1]
x_freq = x_full_freq[fit_start_bin:fit_end_bin+1]
y = y_full[fit_start_bin:fit_end_bin+1]

# Define the Baseline function with frequency as input
def Baseline(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset):
    # Preamble
    circ_consts = (3*10**(-8), 0.35, 619, 50, 10, 0.0343, 4.752*10**(-9), 50, 1.027*10**(-10), 2.542*10**(-7), 0, 0, 0, 0)
    pi = np.pi
    im_unit = 1j  # Use numpy's complex unit (1j)
    sign = 1

    # Main constants
    L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = circ_consts

    I = U*1000/R  # Ideal constant current, mA
    w_res = 2 * pi * 213e6
    w_low = 2 * pi * (213 - 4) * 1e6
    w_high = 2 * pi * (213 + 4) * 1e6
    delta_w = 2 * pi * 4e6 / 500

    # Convert frequency to angular frequency (rad/s)
    w = 2 * pi * f * 1e6

    # Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope() * (w - w_res)

    def Cmain():
        return 20 * 1e-12 * Cknob

    def C(w):
        return Cmain() + Ctrim(w) * 1e-12

    def Z0(w):
        S = 2 * Z_cable * alpha
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.sqrt((S + w * M * im_unit) / (w * D * im_unit))
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def beta(w):
        return beta1 * w

    def gamma(w):
        return alpha + beta(w) * 1j  # Create a complex number using numpy

    def ZC(w):
        Cw = C(w)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cw != 0, 1 / (im_unit * w * Cw), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def vel(w):
        return 1 / beta(w)

    def l(w):
        return trim * vel(w_res) + delta_l

    def ic(w):
        return 0.11133

    def chi(w):
        return np.zeros_like(w)  # Placeholder for x1(w) and x2(w)

    def pt(w):
        return ic(w)

    def L(w):
        return L0 * (1 + sign * 4 * pi * eta * pt(w) * chi(w))

    def ZLpure(w):
        return im_unit * w * L(w) + Rcoil

    def Zstray(w):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cstray != 0, 1 / (im_unit * w * Cstray), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def ZL(w):
        return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))

    def ZT(w):
        epsilon = 1e-10  # Small constant to avoid division by zero
        return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)) + epsilon)

    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)))

    def parfaze(w):
        xp1 = w_low
        xp2 = w_res
        xp3 = w_high
        yp1 = 0
        yp2 = delta_phase
        yp3 = 0

        a = ((yp1 - yp2) * (w_low - w_high) - (yp1 - yp3) * (w_low - w_res)) / \
            (((w_low ** 2) - (w_res ** 2)) * (w_low - w_high) - ((w_low ** 2) - (w_high ** 2)) * (w_low - w_res))
        bb = (yp1 - yp3 - a * ((w_low ** 2) - (w_high ** 2))) / (w_low - w_high)
        c = yp1 - a * (w_low ** 2) - bb * w_low
        return a * w ** 2 + bb * w + c

    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))

    out_y = V_out(w)
    offset = np.array([x - min(out_y.real) for x in out_y.real])
    return offset.real + DC_offset

# Define initial parameter guesses
initial_params = {
    "U": 0.382652652,
    "Cknob": 13.6565062,
    "eta": 9.85632350e-02,
    "trim": 24.3720560,
    # "trim" : 22 / 2,
    "Cstray": 400,
    "phi_const": 265.575865,
    "DC_offset": -0.018,
}

# Define parameter bounds
param_bounds = {
    "U": (0.05, 0.4),
    "Cknob": (0.01, 30),
    "eta": (0.0001, 1),
    "trim": (1, 30),
    "Cstray": (1e-20, 600),
    "phi_const": (0, 360),
    "DC_offset": (-0.1, -0.001),
}

# Define the cost function for iminuit
least_squares = LeastSquares(x_freq, y, yerr_full, Baseline)

# Create a Minuit object for optimization
m = Minuit(least_squares, **initial_params)
# m.fixed['trim']

# Set parameter limits
for param, bounds in param_bounds.items():
    m.limits[param] = bounds

# Perform the fit
m.migrad()

# Check the fit status
try:
    fit_status = m.fmin
    print("Fit Status:")
    print(f"Value: {fit_status.fval:.5e}")
    print(f"Number of Evaluations: {fit_status.nfcn}")
    print(f"Converged: {fit_status.is_valid}")
except AttributeError as e:
    print(f"Error retrieving fit status: {e}")

optimized_params = {param: m.values[param] for param in initial_params.keys()}
param_errors = {param: m.errors[param] for param in initial_params.keys()}

# Extract parameter names
param_names = list(initial_params.keys())

print_optimized_params_with_names(optimized_params, param_names)
print_cov_matrix_with_param_names(m, param_names)

# Plot the original data and the resulting fit function
plt.errorbar(x_full_freq, y_full, yerr=yerr_full, fmt='o', markersize=1, label='Data', color='black')  # Plot full data range
plt.plot(x_freq, Baseline(x_freq, **optimized_params), label='Fit', color='red', linewidth=2)  # Fit for selected domain
plt.xlabel('Frequency (MHz)')
plt.ylabel('Dependent Variable')
plt.legend()
plt.title('Fit of Data between Frequencies')
plt.show()
