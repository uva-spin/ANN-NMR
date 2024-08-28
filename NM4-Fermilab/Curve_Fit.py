from Variables import *
from Lineshape import *
import pandas as pd
import numpy as np
from lmfit import Model

x = np.linspace(210,216,500)
# circ_params = (U,Cknob,cable,eta,phi,Cstray)

mod = Model(LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, .1, 0., ranger))

params = mod.make_params(circ_params,mu = .1, gamma = 0.)
