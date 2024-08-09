from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt
from iminuit import Minuit


baseline = pd.read_csv(r"data\Baseline\2024-06-26_15h51m39s-base-RawSignal.csv",header=None)
baseline = baseline.iloc[0,1:].tolist()
test = baseline

x_array = np.linspace(211,215,500)

# signal = Baseline(x_array,.8,1.,1.,213)

# plt.plot(x_array,signal)
# plt.show()
U = .1
Cknob = 0.2
eta = 0.0104
# cable = 22/2
# eta = 0.0104
# phi = 6.1319
# Cstray = 10**(-15)


initial_guess = [U,Cknob,eta]
lower_bounds = [ .05,.0005,0]
upper_bounds = [.5,2,1]


# popt_baseline_voigt, pcov_baseline_voigt = scipy.optimize.curve_fit(
#     Baseline, x_array, test, p0=initial_guess, bounds=(lower_bounds, upper_bounds), method = 'trf')

# perr_1voigt = np.sqrt(np.diag(pcov_baseline_voigt))

# print(popt_baseline_voigt)



# cap = np.linspace(.05,.7,30)
# sig = []
# for i in cap:
#     sig.append(Baseline(x_array,i,1.,1.,213))

# plt.figure()
# for ele in sig:
#     plt.plot(x_array,ele)
# plt.show()

# signal = Baseline(x_array,0.73302)
# U = 0.1
# Cknob = 0.6
# # cable = 22/2
# eta = 0.0104
# # phi = 6.1319
# # Cstray = 10**(-15)
signal = Baseline(x_array,U,Cknob,eta)
plt.plot(x_array,signal,label = "Simulation")
# plt.plot(x_array,test, label = "Experiment")
plt.legend()
plt.show()
