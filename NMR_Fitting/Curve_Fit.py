from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import optimize
import matplotlib.pyplot as plt
cable = 22/2
circ_params = (U,Cknob,cable,eta,phi,Cstray)


baseline = pd.read_csv(r"data\Baseline\2024-06-25_18h44m37s-base-RawSignal.csv",header=None)
baseline = baseline.iloc[0,1:].tolist()
test = baseline

x_array = np.linspace(209,214,500)

# signal = Baseline(x_array,.8,1.,1.,213)

# plt.plot(x_array,signal)
# plt.show()

Cknob = 0.73302
sigma = 1.
gamma = 1.
center = 213



# initial_guess = [Cknob, sigma,gamma,center]
# lower_bounds = [ .3, .9,.9,212]
# upper_bounds = [.98, 1.,1.,214]

initial_guess = [Cknob]
lower_bounds = [ .3]
upper_bounds = [2]


popt_baseline_voigt, pcov_baseline_voigt = scipy.optimize.curve_fit(
    Baseline, x_array, test, p0=initial_guess, bounds=(lower_bounds, upper_bounds), method = 'trf')

perr_1voigt = np.sqrt(np.diag(pcov_baseline_voigt))

print(popt_baseline_voigt)

# cap = np.linspace(.05,.7,30)
# sig = []
# for i in cap:
#     sig.append(Baseline(x_array,i,1.,1.,213))

# plt.figure()
# for ele in sig:
#     plt.plot(x_array,ele)
# plt.show()

# signal = Baseline(x_array,0.73302)
signal = Baseline(x_array,0.7335583)
plt.plot(x_array,signal,label = "Simulation")
# plt.plot(x_array,test, label = "Experiment")
plt.legend()
plt.show()

