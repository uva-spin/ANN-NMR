import numpy as np
import pandas as pd
g = 0.05
s = 0.04
bigy=(3-s)**0.5


###Circuit Parameters###
U = 0.1
Cknob = 0.42
cable = 22/2
eta = 0.0104
phi = 6.1319
Cstray = 10**(-15)
shift = -2.0464e-2

k_range = 500
circ_consts = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)
params = (U,Cknob,cable,eta,phi,Cstray,shift)
# function_input = 32000000
function_input = 213000000
scansize = .25
rangesize = 0
# ---- Data Files ---- #
Icoil = np.loadtxt(r'data\New_Current.csv', unpack = True)
df_rawsignal_noise = pd.read_csv(r"J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\Noise_RawSignal.csv",header=None)
df_rawsignal_noise = df_rawsignal_noise.drop([0],axis=1)