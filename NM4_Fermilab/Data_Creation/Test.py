import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Lineshape import *

df = pd.read_csv('/home/devin/Documents/Big_Data/ANN_Sample_Data/Deuteron_Oversampled_1M.csv', index_col=False)

df = df[:-400000]

df.to_csv('/home/devin/Documents/Big_Data/ANN_Sample_Data/Deuteron_Oversampled_500K.csv', index=False)


# # Calculate statistics for df['P']
# lowest_value = df['P'].min()
# median_value = df['P'].median()
# mean_value = df['P'].mean()
# highest_value = df['P'].max()

# Sig_Lowest, _, _ = GenerateLineshape(0.0005, np.linspace(-3, 3, 500))
# Sig_Median, _, _ = GenerateLineshape(median_value, np.linspace(-3, 3, 500))
# Sig_Mean, _, _ = GenerateLineshape(mean_value, np.linspace(-3, 3, 500))
# Sig_Highest, _, _ = GenerateLineshape(0.01, np.linspace(-3, 3, 500))

# Sig_Lowest = Sig_Lowest 
# Sig_Median = Sig_Median / 1500.0
# Sig_Mean = Sig_Mean / 1500.0
# Sig_Highest = Sig_Highest 

# print(lowest_value)
# print(median_value)
# print(mean_value)
# print(highest_value)

# # # Get the corresponding rows
# lowest_row = df[df['P'] == lowest_value].drop(columns=['P', 'SNR']).values.reshape(500, -1)
# median_row = df[df['P'] == median_value].drop(columns=['P', 'SNR']).values.reshape(500, -1)
# mean_row = df[df['P'] == mean_value].drop(columns=['P', 'SNR']).values.reshape(500, -1)
# highest_row = df[df['P'] == highest_value].drop(columns=['P', 'SNR']).values.reshape(500, -1)

# # print(lowest_row.values.flatten().shape)

# # Plot the signals for lowest, median, mean, and highest values
# plt.figure(figsize=(12, 6))
# plt.plot(np.linspace(-3, 3, 500), Sig_Lowest, label='0.05%')
# # plt.plot(np.linspace(-3, 3, 500), Sig_Median, label='Median Value')
# # plt.plot(np.linspace(-3, 3, 500), Sig_Mean, label='Mean Value')
# plt.plot(np.linspace(-3, 3, 500), Sig_Highest, label='1%')
# # plt.plot(np.linspace(-3, 3, 500), lowest_row, label='Lowest Value')
# # plt.plot(np.linspace(-3, 3, 500), median_row, label='Median Value')
# # plt.plot(np.linspace(-3, 3, 500), mean_row, label='Mean Value')
# # plt.plot(np.linspace(-3, 3, 500), highest_row, label='Highest Value')

# plt.title('Lineshape at different P values')
# plt.xlabel('R')
# plt.ylabel('Intensity')
# plt.legend()    
# plt.grid()
# plt.show()