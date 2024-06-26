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

testmodel = tf.keras.models.load_model('/project/ptgroup/Devin/Neural_Network/Trained_Models_v14/trained_model_1M_v3_tuned.h5')

df = pd.read_csv('Testing_Data_v4/Sample_Data_50K.csv')

y = df['P']
x = df.drop(['P'],axis=1)
# train_X, test_X, train_y, test_y = split_data(x,y)

X = df.drop(['P','SNR'],axis=1)
Y = testmodel.predict(X)
P = np.array(df['P'])
SNR = np.array(df['SNR'])
Y = Y.reshape((len(Y),))

err = ((P-Y)/(P))*100
# plt.scatter(len(err),err)
# plt.savefig('Accuracy.png')
result = pd.DataFrame(Y, columns ={'P'})
result = result.rename(columns={df.columns[0]:'P'},inplace=False)
result['P_True'] = P.tolist()
result['err'] = err.tolist()
# plt.plot(np.arange(len(result['P_True'])),result['P_True'],'.',label = 'True')
result.to_csv('Results.csv')

# ### Plotting ###

g = 0.05
s = 0.04
bigy=(3-s)**0.5
labelfontsize = 30

def cosal(x,eps):
    return (1-eps*x-s)/bigxsquare(x,eps)


def c(x):
    return ((g**2+(1-x-s)**2)**0.5)**0.5


def bigxsquare(x,eps):
    return (g**2+(1-eps*x-s)**2)**0.5


def mult_term(x,eps):
    return float(1)/(2*np.pi*np.sqrt(bigxsquare(x,eps)))


def cosaltwo(x,eps):
    return ((1+cosal(x,eps))/2)**0.5


def sinaltwo(x,eps):
    return ((1-cosal(x,eps))/2)**0.5


def termone(x,eps):
    return np.pi/2+np.arctan((bigy**2-bigxsquare(x,eps))/((2*bigy*(bigxsquare(x,eps))**0.5)*sinaltwo(x,eps)))


def termtwo(x,eps):
    return np.log((bigy**2+bigxsquare(x,eps)+2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps))/(bigy**2+bigxsquare(x,eps)-2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps)))

def icurve(x,eps):
    return mult_term(x,eps)*(2*cosaltwo(x,eps)*termone(x,eps)+sinaltwo(x,eps)*termtwo(x,eps))

xvals = np.linspace(-6,6,500)
yvals = icurve(xvals,1)/10
yvals2 = icurve(-xvals,1)/10
x_arr = np.linspace(31.5,32.5,500)


###Circuit Parameters###
U = 0.1
# Cknob = 0.125
Cknob = .120
cable = 23/2
eta = 0.0104
phi = 6.1319
Cstray = 10**(-15)

k_range = 5000
circ_constants = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0) ### Don't change these, these are constants of the Liverpool
circ_params = (U,Cknob,cable,eta,phi,Cstray)
function_input = 32000000
# function_input = 213000000
scan_s = .25
ranger = 0


    

def cosal(x,eps):
    return (1-eps*x-s)/bigxsquare(x,eps)


def c(x):
    return ((g**2+(1-x-s)**2)**0.5)**0.5


def bigxsquare(x,eps):
    return (g**2+(1-eps*x-s)**2)**0.5


def mult_term(x,eps):
    return float(1)/(2*np.pi*np.sqrt(bigxsquare(x,eps)))


def cosaltwo(x,eps):
    return ((1+cosal(x,eps))/2)**0.5


def sinaltwo(x,eps):
    return ((1-cosal(x,eps))/2)**0.5


def termone(x,eps):
    return np.pi/2+np.arctan((bigy**2-bigxsquare(x,eps))/((2*bigy*(bigxsquare(x,eps))**0.5)*sinaltwo(x,eps)))


def termtwo(x,eps):
    return np.log((bigy**2+bigxsquare(x,eps)+2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps))/(bigy**2+bigxsquare(x,eps)-2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps)))

def icurve(x,eps):
    return mult_term(x,eps)*(2*cosaltwo(x,eps)*termone(x,eps)+sinaltwo(x,eps)*termtwo(x,eps))

### All simulation above, can just ignore

xvals = np.linspace(-6,6,500)
yvals = icurve(xvals,1)/10
yvals2 = icurve(-xvals,1)/10
center = 250
length = range(500)
norm_array = []
for x in length:
    norm_array = np.append(norm_array,(x - center)*(12/500))  
Iplus = icurve(norm_array,1)
Iminus = icurve(norm_array,-1)

result['P_diff'] = result['P'] - result['P_True']
x = np.array(result['P_diff'])*100
np.savetxt("Histogram_Data_v3.csv",x,delimiter = ',') ### This saves the relative difference to a .csv file


num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = True, 
                            color ='green',
                            alpha = 0.7)
   
(mu, sigma) = norm.fit(x)

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

plt.figure()
plt.plot(result['P_True'],result['err'], '.')
plt.xlabel('Polarization')
plt.ylabel('Relative Percent Error (%)')
plt.savefig('Accuracy_Plot.pdf')


acc = []
f_1 = []
f_2 = []
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
