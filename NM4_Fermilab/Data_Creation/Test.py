import numpy as np
import matplotlib.pyplot as plt
from Lineshape import Baseline, Voigt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager
g = 0.05
s = 0.04
bigy = (3 - s)**0.5



# sigs = []
# bound = .05
# for i in range(30):
#     sigs.append(Sampling_Lineshape(0.5, x_freq, bound))

# sigs = np.array(sigs)


# minor_ticks_x = np.arange(30, 50, 0.25)
# minor_ticks_y = np.arange(0, 0.6, 0.05)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# axisFontSize = 28
# legendFontSize = 38

# cmap = plt.get_cmap('plasma')

# color_gradients = cmap(np.linspace(0, 1, len(sigs)))

# # font = font_manager.FontProperties(family='serif', size=legendFontSize)
# # for i in range(len(sigs)):
# #     plt.plot(x_freq, sigs[i], color=color_gradients[i], linewidth=1.5)
# # plt.xlabel(r'Frequency [MHz]', fontsize=axisFontSize)
# # plt.ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
# # plt.axvline(x=32.68, color='black', linestyle='--', linewidth=2, label=r'Larmor Frequency (ND$_{3}$)')
# # plt.grid(True, which='both', axis='both', linewidth=2)
# # plt.xticks(fontsize=axisFontSize)
# # plt.yticks(fontsize=axisFontSize)

# # fig.set_size_inches(16, 16)
# # # plt.tight_layout()
# # plt.legend(fontsize=24)
# # plt.show()
# # fig.savefig('signal_f_domain.pdf', dpi=100)
# # fig.clear()


# ### Generate shifts with baseline ###


# U = 2.4283
# eta = 1.04e-2
# phi = 6.1319
# Cstray = 10**(-20)
# Cknob = 0.1899
# cable = 6/2
# center_freq = 32.32

# baseline = Baseline(x_freq, U, Cknob, eta, cable, Cstray, phi, 0)

# combined_signal = (sigs/1500.0) + baseline

# font = font_manager.FontProperties(family='serif', size=legendFontSize)
# for i in range(len(sigs)):
#     plt.plot(x_freq, combined_signal[i], color=color_gradients[i], linewidth=1.5)
# plt.plot(x_freq, baseline, color='black', linewidth=2.5, label='Baseline')
# plt.xlabel(r'Frequency [MHz]', fontsize=axisFontSize)
# plt.ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
# plt.axvline(x=32.68, color='black', linestyle='--', linewidth=2, label=r'Larmor Frequency (ND$_{3}$)')
# plt.grid(True, which='both', axis='both', linewidth=2)
# plt.xticks(fontsize=axisFontSize)
# plt.yticks(fontsize=axisFontSize)

# fig.set_size_inches(16, 16)
# # plt.tight_layout()
# plt.legend(fontsize=24)
# plt.show()
# import os
# os.makedirs('Data_Creation/Shifting', exist_ok=True)
# fig.savefig('Data_Creation/Shifting/baseline_shifts.pdf', dpi=100)
# fig.clear()



            # Common baseline parameters
U = 0.24283
eta = 1.04e-2
phi = 6.1319
Cstray = 10**(-20)
shift = 0

Cknob = 2.547
cable = 22/2
center_freq = 213

x_freq = np.linspace(211.8, 214.2, 500)

baseline = Baseline(x_freq, U, Cknob, eta, cable, Cstray, phi, shift)

proton = Voigt(x_freq, 0.5, 0.05, 0.1, 213)

combined_signal = baseline + proton

plt.plot(x_freq, combined_signal)
plt.show()


