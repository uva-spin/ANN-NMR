import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Lineshape import *

# data = pd.read_csv("Deuteron_Low_No_Noise_500K.csv")

# data = data[(data["P"] >= 0.00045) & (data["P"] <= 0.00055)]

# X = data.drop(columns=["P", 'SNR']).astype('float64').values
# y = data["P"].astype('float64')


Polar = 0.00055
X = np.linspace(-3,3,500)
Line, _, _ = GenerateLineshape(Polar, X)

x_full_bins = np.linspace(-3, 3, 500)  

Line, _, _ = GenerateLineshape(Polar, x_full_bins)  


y = Line
yerr_full = 0.005  


plt.plot(x_full_bins, y)
plt.show()


initial_params = [0.00052]


param_bounds = ([0.0002],  
                [0.0007,]) 

def Baseline(x, P):
    Sig, _, _ = GenerateLineshape(P, x)
    return Sig


popt, pcov = curve_fit(Baseline, x_full_bins, y, p0=initial_params, bounds=param_bounds)

print("Covariance Matrix:")
print(pcov)

print("Optimized Parameters:")
print(popt)

Fitted_Sig = Baseline(x_full_bins, *popt)

plt.errorbar(x_full_bins, y, yerr=yerr_full, fmt='o', markersize=1, label='Data', color='black')  # Plot full data range
plt.plot(x_full_bins, Fitted_Sig, label='Fit', color='red', linewidth=2)  # Fit for selected domain
plt.xlabel('Frequency (MHz)')
plt.ylabel('Dependent Variable')
plt.legend()
plt.title('Fit of Data between Frequencies')
plt.show()
