import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import h5py as h
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import sys

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
np.savetxt("Histogram_Data_v3.csv",x_Pol,delimiter = ',') ### This saves the relative difference to a .csv file


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


# acc = []
# f_1 = []
# f_2 = []
# for i in range(0,200):
#     p = P[i]
#     p_pred = Y[i]
#     accuracy = ((P[i] - Y[i])/P[i])*100
#     acc.append(accuracy)
#     snr = SNR[i]
#     r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
#     r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
#     center = 250
#     length = range(500)
#     # ratio = Iminus/Iplus
#     array = r*Iminus
#     array_pred = r_pred*Iminus
#     array_flipped = np.flip(array)
#     array_pred_flipped = np.flip(array_pred)
#     element_1 = array_flipped+Iminus
#     sum_array = np.sum(array_flipped)*(12/500)
#     element_2 = 1/sum_array
#     element_3 = p
#     element_1_pred = array_pred_flipped + Iminus
#     sum_array_pred = np.sum(array_pred_flipped)*(12/500)
#     element_2_pred = 1/sum_array_pred
#     element_3_pred = p_pred
#     result = element_1*element_2*element_3
#     result_pred = element_1_pred*element_2_pred*element_3_pred
#     result_new = result.reshape(500,)
#     result_pred_new = result_pred.reshape(500,)
#     # base = np.zeros((500,))
#     # baseline = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, base, base, Backgmd, Backreal,Current, ranger)
#     lineshape = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, result_pred_new, result_pred_new, Backgmd, Backreal,Current, ranger)
#     offset = [x - max(lineshape) for x in lineshape]
#     # offset = np.subtract(lineshape,baseline)
#     # offset += np.full((500,),0.00011769998)
#     arr = np.array(X.iloc[[i]]).reshape(500,)
#     arr_original  = [x - max(arr) for x in arr]
#     # arr_original = np.subtract(arr,baseline)
#     offset = np.append(offset,snr)
#     offset = np.append(offset,p)
#     offset = np.append(offset,p_pred)
#     arr_original = np.append(arr_original,snr)
#     arr_original = np.append(arr_original,p)
#     arr_original = np.append(arr_original,p_pred)
#     f_1.append(arr_original)
#     f_2.append(offset)
#     # plt.figure()
#     # # plt.plot(norm_array,arr)
#     # plt.plot(norm_array,result_pred_new)
#     # plt.xlim(-3,3)
#     # plt.xlabel('R')
#     # plt.ylabel('Intensity')
#     # plt.legend(['True','Predicted'])
#     # plt.ylim(0,.015)
#     # p = p*100
#     # plt.title("P_true:" + "%.4f %%" %(p) + "\n P_pred:" + str(np.round(p_pred*100,3)) + "%%"+ "\n SNR:" + str(np.round(snr,3)))
#     # plt.savefig('Test_Plots_TE_v3/Model_Prediction_50K_V1_After_No'+ str(i) +'.png')
#     # plt.figure()
#     # # plt.plot(norm_array,result_new)
#     # arr = np.array(X.iloc[[i]]).reshape(500,)
#     # plt.plot(x_arr,arr)
#     # # plt.ylim(0,1)
#     # # plt.xlim(-3,3)
#     # plt.xlabel('Frequency')
#     # plt.ylabel('Intensity')
#     # # plt.legend(['True','Sample_Noise'])
#     # plt.title("P_pred:" + str(np.round(p_pred*100,3)) + "%%" + "\n P_true:" + "%.4f %%" %(p) + "\n SNR:" + str(np.round(snr,3)))
#     # plt.savefig('Test_Plots_TE_v3/_Model_Prediction_50K_V1_Before_No' + str(i) + '.png')
    
# df1 = pd.DataFrame(f_1)
# df2 = pd.DataFrame(f_2)
# df1.to_csv("Before_v3.csv",header=False,index=False)
# df2.to_csv("After_v3.csv",header=False,index=False)
# plt.figure()
# plt.plot(acc)
# plt.savefig('Accuracy.png')
