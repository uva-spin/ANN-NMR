import numpy as np
import matplotlib.pyplot as mp
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
from Misc_Functions import *
from Lineshape import *
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

g = 0.05
s = 0.04
bigy = (3 - s)**0.5

x_freq = np.linspace(30.88, 34.48, 500) 

bound = 0.0

sig1 = Sampling_Lineshape(0.0005, x_freq, bound)
sig2 = Sampling_Lineshape(0.0005, x_freq, bound)
sig3 = Sampling_Lineshape(0.001, x_freq, bound)

scaler1 = MinMaxScaler(feature_range=(-1, 1))
scaler2 = StandardScaler()

sig2_MinMax = scaler1.fit_transform(sig1.reshape(-1, 1))
sig2_StandardScalar = scaler2.fit_transform(sig2.reshape(-1, 1))

scaler3 = RobustScaler()
scaler4 = PowerTransformer(method='box-cox')

sig2_Robust = scaler3.fit_transform(sig2.reshape(-1, 1))
sig2_BoxCox = scaler4.fit_transform(sig2.reshape(-1, 1))


sig2_LogScale = np.log1p(sig2.reshape(-1, 1))

sig3_MinMax = scaler1.fit_transform(sig3.reshape(-1, 1))
sig3_StandardScalar = scaler2.fit_transform(sig3.reshape(-1, 1))    
sig3_Robust = scaler3.fit_transform(sig3.reshape(-1, 1))
sig3_BoxCox = scaler4.fit_transform(sig3.reshape(-1, 1))

sig3_LogScale = np.log1p(sig3.reshape(-1, 1))

minor_ticks_x = np.arange(30, 50, 0.25)
# minor_ticks_y = np.arange(0, 0.6, 0.05)
fig, (ax1, ax2, ax3, ax4, ax5) = mp.subplots(5, 1, figsize=(16, 40))

axisFontSize = 38
legendFontSize = 16

font = font_manager.FontProperties(family='serif', size=legendFontSize)

ax1.plot(x_freq, sig2_MinMax, color='red', linewidth=4, linestyle='--', label='Signal with MinMaxScaler, P = 0.05%')
ax1.plot(x_freq, sig3_MinMax, color='blue', linewidth=4, linestyle=':', label='Signal with MinMaxScaler, P = 0.1%')
ax1.set_xlabel('f', fontsize=axisFontSize)
ax1.set_ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
ax1.grid(True, which='both', axis='both', linewidth=2)
ax1.legend(prop=font)

ax2.plot(x_freq, sig2_StandardScalar, color='red', linewidth=4, linestyle='.', label='Signal with StandardScaler, P = 0.05%')
ax2.plot(x_freq, sig3_StandardScalar, color='blue', linewidth=4, linestyle=':', label='Signal with StandardScaler, P = 0.1%')
ax2.set_xlabel('f', fontsize=axisFontSize)
ax2.set_ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
ax2.grid(True, which='both', axis='both', linewidth=2)
ax2.legend(prop=font)

ax3.plot(x_freq, sig2_LogScale, color='red', linewidth=4, linestyle='--', label='Signal with LogScale, P = 0.05%')
ax3.plot(x_freq, sig3_LogScale, color='blue', linewidth=4, linestyle=':', label='Signal with LogScale, P = 0.1%')
ax3.set_xlabel('f', fontsize=axisFontSize)
ax3.set_ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
ax3.grid(True, which='both', axis='both', linewidth=2)
ax3.legend(prop=font)

ax4.plot(x_freq, sig2_Robust, color='red', linewidth=4, linestyle='--', label='Signal with RobustScaler, P = 0.05%')
ax4.plot(x_freq, sig3_Robust, color='blue', linewidth=4, linestyle=':', label='Signal with RobustScaler, P = 0.1%')
ax4.set_xlabel('f', fontsize=axisFontSize)
ax4.set_ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
ax4.grid(True, which='both', axis='both', linewidth=2)
ax4.legend(prop=font)

ax5.plot(x_freq, sig2_BoxCox, color='red', linewidth=4, linestyle='--', label='Signal with BoxCoxScaler, P = 0.05%')
ax5.plot(x_freq, sig3_BoxCox, color='blue', linewidth=4, linestyle=':', label='Signal with BoxCoxScaler, P = 0.1%')
ax5.set_xlabel('f', fontsize=axisFontSize)
ax5.set_ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
ax5.grid(True, which='both', axis='both', linewidth=2)
ax5.legend(prop=font)

mp.tight_layout()  # Adjust spacing between subplots
fig.savefig('signal_f_domain.pdf', dpi=600)
fig.clear()
