from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

# Load the true data
y_true = pd.read_csv(r"NMR_Fitting\data\Baseline\2024-06-25_18h44m37s-base-RawSignal.csv", header=None)
y_true = y_true.iloc[0, 1:].tolist()

errors = np.std(y_true, axis=0) 

x_array = np.linspace(211, 215, 500)

# Define the mean squared error function (not used in minimization)
def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Define the Chi-squared function
def ChiSquared(U, Cknob, eta, Cstray,phi_const):
    y_sim = Baseline(x_array, U, Cknob, eta, Cstray,phi_const)
    return np.sum(np.square((np.array(y_true) - np.array(y_sim))/(errors)))

# Initial parameter values

U = 2.4283e1
Cknob = 2.4842e-1
# cable = 22/2
eta = 1.04e-2
phi = 6.1319
# Cstray = 6.1183e-15
Cstray = 10**(-15)
shift = -2.0464e-2
signal = Baseline(x_array,U,Cknob,eta,Cstray,phi)
plt.plot(x_array,signal,label = "Simulation")
plt.plot(x_array,y_true, label = "Experiment")
plt.legend()
plt.show()

# # Initialize Minuit with the Chi-squared function and initial parameter values
# m = Minuit(ChiSquared, U=U, Cknob=Cknob, eta=eta, Cstray=Cstray, shift=shift)

# # Set parameter limits
# m.limits['U'] = (0.05, 30)
# m.limits['Cknob'] = (0.001, 0.5)
# m.limits['eta'] = (0.0001, 0.05)
# m.limits['Cstray'] = (1e-14, 1e-11)
# m.limits['shift'] = (-1e-3, 0.1)

# m.simplex()

# # Perform the minimization
# m.migrad()

# # Check if minimization was successful
# if not m.migrad():
#     print("Minimization did not converge properly.")
# else:
#     print(f"Best-fit parameters: {m.values}")
#     print(f"Chi-squared: {m.fval}")

#     # Try to compute errors and covariance matrix
#     m.hesse()

#     if m.covariance is not None:
#         print("Covariance matrix:")
#         cov_matrix = pd.DataFrame(m.covariance, columns=m.parameters, index=m.parameters)
#         print(cov_matrix)
#     else:
#         print("Covariance matrix could not be determined.")
