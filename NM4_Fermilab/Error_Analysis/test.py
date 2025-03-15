import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Custom_Scripts.Misc_Functions import *


data_path = find_file("Deuteron_Low_Oversampled_1M.csv")

data = pd.read_csv(data_path)

X = data.drop(columns=["P", 'SNR']).astype('float64').values

plt.plot(np.linspace(0, 1, 500), X[0])
plt.show()





