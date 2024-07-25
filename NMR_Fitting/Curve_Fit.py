from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt

# circ_params = (U,Cknob,cable,eta,phi,Cstray)

baseline = pd.read_csv(r"data/2024-02-06_21h17m32s-RawSignal.csv")
baseline = baseline.iloc[:,1:]
test = baseline.iloc[7]
x_array  = np.linspace(209.13,215.13,500)

U = 0.1
Cknob = .042
ampG1 = 20
cenG1 = 50
sigmaG1 = 5
ampL1 = 80
cenL1 = 50
widL1 = 5

plt.plot(x_array,test)
plt.show()


popt_baseline_voigt, pcov_baseline_voigt = scipy.optimize.curve_fit(Signal, x_array, test)

perr_1voigt = np.sqrt(np.diag(pcov_baseline_voigt))

print(popt_baseline_voigt)