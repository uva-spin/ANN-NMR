import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import h5py as h
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import sys

Area_Model = load_model(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Models\v1\best_model_v1.keras')

df_Area = pd.read_csv(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Testing_Data\Sample_Testing_Data.csv')

y_Area = df_Area['Area']
x_Area = df_Area.drop(['Area'], axis=1)
X_Area = df_Area.drop(['Area', 'SNR'], axis=1)
Y_Area = Area_Model.predict(X_Area)
Area = np.array(df_Area['Area'])  
SNR = np.array(df_Area['SNR'])
Y_Area = Y_Area.reshape((len(Y_Area),))

rel_err = (np.abs((Y_Area - Area)) / Area) * 100 
abs_err = np.abs(Y_Area - Area)

result = pd.DataFrame({'Area_Predicted': Y_Area})
result['Area_True'] = Area.tolist()
result['rel_err'] = rel_err.tolist()
result['abs_err'] = abs_err.tolist()
result.to_csv(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Results.csv', index=False)

num_bins = 100

n, bins, patches = plt.hist(rel_err, num_bins, 
                            density=True, 
                            color='green',
                            alpha=0.7)

(mu, sigma) = norm.fit(rel_err)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, '--', color='black')

plt.xlabel('Difference in Area')
plt.ylabel('Count')
plt.title(f"Histogram of Area Difference: mu={mu:.4f}, sigma={sigma:.4f}")
plt.grid(True)

plt.savefig(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Histogram_Relative_Error.png')
plt.close()  

n, bins, patches = plt.hist(abs_err, num_bins, 
                            density=True, 
                            color='blue',
                            alpha=0.7)

(mu, sigma) = norm.fit(abs_err)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, '--', color='black')

plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.title(f"Histogram of Absolute Error: mu={mu:.4f}, sigma={sigma:.4f}")
plt.grid(True)

plt.savefig(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Histogram_Absolute_Error.png')



