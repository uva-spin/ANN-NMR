import numpy as np
import matplotlib.pyplot as plt
from Lineshape import *
# from phase_lineshape import *

x_freq = np.linspace(30.88, 34.48, 500) 

x = np.linspace(-3, 3, 500)

bound = 0.0

U = 2.4283
eta = 1.04e-2
# self.phi = 6.1319
Cstray = 10**(-20)
shift = 0
Cknob = 0.220
cable = 6/2
center_freq = 32.32

P = 0.5

### Tensor Polarization ###

# P = np.sqrt((4-(P-2)**2)/3)

r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)

phi_deg=0

baseline = Baseline(x_freq, U, Cknob, eta, shift, Cstray, phi_deg, 0)

signal, Iplus, Iminus = GenerateTensorLineshape(x, P, phi_deg)

plt.plot(x, signal/100 + baseline)
plt.show()