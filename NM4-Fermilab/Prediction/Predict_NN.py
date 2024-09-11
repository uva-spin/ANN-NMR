import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import h5py as h
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
import random
from keras.layers import GaussianNoise
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import matplotlib.mlab as mlab
import sys
from scipy import interpolate
import cmath
import statistics as std

Polarization_Model = tf.keras.models.load_model('/project/ptgroup/Devin/Neural_Network/Trained_Models_v14/trained_model_1M_v3_tuned.h5')

df_Pol = pd.read_csv('Testing_Data_v4/Sample_Data_50K.csv')
df_Err = pd.read_csv('Blahblahblahblahblahblahbl')


### Predicted Polarization
y_Pol = df_Pol['P']
x_Pol = df_Pol.drop(['P'],axis=1)
X_Pol = df_Pol.drop(['P','SNR'],axis=1)
Y_Pol = Polarization_Model.predict(X_Pol)
P = np.array(df_Pol['P'])
SNR = np.array(df_Pol['SNR'])
Y_Pol = Y_Pol.reshape((len(Y_Pol),))

### Predicted Errors
y_Err = df_Err['Rel','Abs']
# x_Err = df_Err.drop(['P','Rel','Abs','SNR'],axis=1)
X_Err = df_Err.drop(['P','Rel','Abs','SNR'],axis=1)
Y_Rel,Y_Abs = Polarization_Model.predict(X_Pol)
rel_err = np.array(df_Pol['P'])
SNR = np.array(df_Pol['SNR'])
Y_Pol = Y_Pol.reshape((len(Y_Pol),))

rel_err = (np.abs((Y_Pol-P))/(P))*100
abs_err = np.abs(Y_Pol-P)
result = pd.DataFrame(Y_Pol, columns ={'P'})
result = result.rename(columns={df_Pol.columns[0]:'P'},inplace=False)
result['P_True'] = P.tolist()
result['rel_err'] = rel_err.tolist()
result['abs_err'] = abs_err.tolist()
result.to_csv('Results.csv')

result['P_diff'] = result['P'] - result['P_True']
x_Pol = np.array(result['P_diff'])*100
np.savetxt("Histogram_Data_v3.csv",x_Pol,delimiter = ',')


num_bins = 100
   
n, bins, patches = plt.hist(x_Pol, num_bins, 
                            density = True, 
                            color ='green',
                            alpha = 0.7)
   
(mu, sigma) = norm.fit(x_Pol)

### Plot Histogram of Percentage Difference

# y = norm.pdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2)
  
# plt.plot(bins, y, '--', color ='black')
# plt.hist(x,num_bins,density=True,color='green',alpha=0.7)
# plt.xlabel('Percentage Difference (%)')
# plt.ylabel('Count')
# # plt.xlim([-5,5])
# plt.title("Histogram of Percentage Difference: mu=%.4f, sigma=%.4f" %(mu, sigma))
# # plt.title(r'$\mathrm{Histogram\ of\ I0%% Noise:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

# # plt.grid(True)
  
# plt.savefig('Trained_Models_v14/Histogram_1M_v3.png')

# plt.figure()
# plt.plot(result['P_True'],result['err'], '.')
# plt.xlabel('Polarization')
# plt.ylabel('Relative Percent Error (%)')
# plt.savefig('Accuracy_Plot.pdf')

