import numpy as np
import pandas as pd
g = 0.05
s = 0.04
bigy=(3-s)**0.5


###Circuit Parameters###
U = 0.1
Cknob = 0.180 
# Cknob = 0.011
cable = 22
eta = 0.0104
phi = 6.1319
Cstray = 10**(-15)

k_range = 500
circ_constants = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)
circ_params = (U,Cknob,cable,eta,phi,Cstray)
# function_input = 32000000
function_input = 213000000
scan_s = .25