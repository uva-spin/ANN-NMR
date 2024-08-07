from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt
from iminuit import Minuit

# Load the baseline data
baseline = pd.read_csv(r"data\Baseline\2024-06-25_18h44m37s-base-RawSignal.csv", header=None)
baseline = baseline.iloc[0, 1:].tolist()
test = baseline

# Create an array for the x-axis values
x_array = np.linspace(211, 215, 500)

# Initial guess and bounds for the parameters to be optimized
U = 2.4283e1
Cknob = 2.4842e-1
eta = 0.0104
cable = 22 / 2
phi = 6.1319
Cstray = 10**(-15)

initial_guess = [U, Cknob, eta, Cstray]
# lower_bounds = [15, .05, 0, 9.5 * 10**(-16)]
# upper_bounds = [30, .3, 1, 2 * 10**(-15)]

# Define the Baseline function with fixed parameters trim and phi
def Baseline_fixed(x, U, Cknob, eta, Cstray):
    return Baseline(x, U, Cknob, eta, cable, Cstray, phi)

# Perform the curve fitting
# popt_baseline_voigt, pcov_baseline_voigt = scipy.optimize.curve_fit(
#     Baseline_fixed, x_array, test, p0=initial_guess, bounds=(lower_bounds, upper_bounds), method='trf'
# )


popt_baseline_voigt, pcov_baseline_voigt = scipy.optimize.curve_fit(
    Baseline_fixed, x_array, test, p0=initial_guess, method='trf'
)

# Calculate the standard deviation errors for the parameters
perr_1voigt = np.sqrt(np.diag(pcov_baseline_voigt))

# Print the optimized parameters and covariance matrix
print("Optimized parameters:", popt_baseline_voigt)
print("Covariance matrix:\n", pcov_baseline_voigt)

# Assign the fitted parameters to new variables
fitted_U, fitted_Cknob, fitted_eta, fitted_Cstray = popt_baseline_voigt

# Generate the simulated signal using the fitted parameters
signal = Baseline_fixed(x_array, fitted_U, fitted_Cknob, fitted_eta, fitted_Cstray)

# Plot the simulated signal against the experimental data
plt.plot(x_array, signal, label="Simulation")
plt.plot(x_array, test, label="Experiment")
plt.legend()
plt.show()
plt.savefig("Fitted_Signal.jpg")

import pandas as pd

# Create a DataFrame from the covariance matrix
param_names = ['U', 'Cknob', 'eta', 'Cstray']
cov_matrix_df = pd.DataFrame(pcov_baseline_voigt, columns=param_names, index=param_names)

# Print the covariance matrix in a nice format
print("Covariance Matrix:")
print(cov_matrix_df.to_string(index=True, float_format="%.4e"))

# Optionally, save the table to a file for easy inclusion in a presentation
cov_matrix_df.to_csv("covariance_matrix.csv")
