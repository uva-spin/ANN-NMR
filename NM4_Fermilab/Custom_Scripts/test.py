import numpy as np
import matplotlib.pyplot as mp
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
from Misc_Functions import *
from Lineshape import *
import matplotlib.cm as cm
# Given parameters
g = 0.05
s = 0.04
bigy = (3 - s)**0.5

x_freq = np.linspace(30.88, 34.48, 500) 

sigs = []
bound = .05
for i in range(30):
    sigs.append(Sampling_Lineshape(0.5, x_freq, bound))

sigs = np.array(sigs)


minor_ticks_x = np.arange(30, 50, 0.25)
minor_ticks_y = np.arange(0, 0.6, 0.05)
fig = mp.figure()
ax = fig.add_subplot(1, 1, 1)

axisFontSize = 38
legendFontSize = 38

cmap = mp.get_cmap('plasma')

color_gradients = cmap(np.linspace(0, 1, len(sigs)))

font = font_manager.FontProperties(family='serif', size=legendFontSize)
for i in range(len(sigs)):
    mp.plot(x_freq, sigs[i], color=color_gradients[i], linewidth=4, label='Signal')
mp.xlabel('f', fontsize=axisFontSize)
mp.ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
mp.grid(True, which='both', axis='both', linewidth=2)
# mp.xticks(minor_ticks_x, fontsize=axisFontSize)
# mp.yticks(minor_ticks_y, fontsize=axisFontSize)


fig.set_size_inches(16, 16)
fig.savefig('signal_f_domain.pdf', dpi=100)
fig.clear()