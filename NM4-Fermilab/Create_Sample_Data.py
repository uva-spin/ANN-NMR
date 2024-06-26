import pandas as pd
import numpy as np
import random
import sys
from scipy import interpolate
import cmath
import matplotlib.pyplot as plt
import statistics as std
from scipy.stats import zscore
import math
g = 0.05
s = 0.04
bigy=(3-s)**0.5
labelfontsize = 30

###Circuit Parameters###
U = 0.1
Cknob = 0.180 
# Cknob = 0.011
cable = 23
eta = 0.0104
phi = 6.1319
Cstray = 10**(-15)

k_range = 500
circ_constants = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)
circ_params = (U,Cknob,cable,eta,phi,Cstray)
function_input = 32000000
# function_input = 213000000
scan_s = .25
ranger = 0
# ---- Data Files ---- #
Backgmd = np.loadtxt(r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\data\Backgmd.dat', unpack = True)
Backreal = np.loadtxt(r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\data\Backreal.dat', unpack = True)
Current = np.loadtxt(r'J:\Users\Devin\Desktop\Spin Physics Work\Deuteron\New_Current.csv', unpack = True)
df_rawsignal_noise = pd.read_csv(r"J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\noise-20240527T224337Z-001\noise\2024-05-24_16h12m33s-RawSignal.csv",header=None)
df_rawsignal_noise = df_rawsignal_noise.drop([0],axis=1)

def choose_random_row(csv_file):
    df = csv_file
    if df.empty:
        return None  # If the DataFrame is empty
    random_index = np.random.randint(0, len(df))  # Generate a random index
    random_row = df.iloc[random_index]  # Get the row at the random index
    return random_row

def exclude_outliers(df, threshold=2):
    # Compute Z-scores for each row
    z_scores = df.apply(zscore, axis=0, result_type='broadcast')
    
    # Check if any Z-score exceeds the threshold
    is_outlier = (z_scores.abs() > threshold).any(axis=1)
    
    # Exclude outliers
    df_filtered = df[~is_outlier].apply(lambda x: x / 1000)
    
    return df_filtered

def Baseline_Polynomial_Curve(w):
    return -1.84153246e-07*w**2 + 8.42855076e-05*w - 1.11342243e-04

def random_sign():
    return random.choice([-1, 1])

def Sinusoidal_Noise(shape):
    # Generate an array of random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2*np.pi, shape)
    
    # Calculate cosine and sine of each angle
    cos_values = np.random.uniform(-0.0005,0.0005)*np.cos(angles)
    sin_values = np.random.uniform(-0.0005,0.0005)*np.sin(angles)
    
    # Sum cosine and sine
    result = cos_values + sin_values
    
    return result

df_filtered = exclude_outliers(df_rawsignal_noise)


def LabviewCalculateXArray(f_input, scansize, rangesize):
    
    #---------------------preamble----------------
    
    pi = np.pi
    im_unit = complex(0,1)

    #----------------------main------------------
    
    
    

    #Derived quantities
    w_res = 2*pi*f_input
    f_small = f_input/(1e6)
    w_low = 2 * pi * (f_small - scansize) * (1e6)
    w_high = 2 * pi * (f_small + scansize) * (1e6)
    delta_w = 2 * pi * 500 * ((1e3)/500)

    
    #Variables for creating splines
    k_ints = range(0,500)
    k = np.array(k_ints,float)
    x = (k*delta_w)+(w_low)
    
    larger_k = range(0,k_range)
    larger_x = np.array(larger_k, float)
    w_range = w_high - w_low
    larger_range = (delta_w*larger_x)+(w_low-5*w_range)
    larger_range /= (2 * pi)*(1e6)
    
    x /= (2*pi)*(1e6)
    return_val = x
    if (rangesize == 1):
        return_val = larger_range
    return return_val

def getArrayFromFunc(func,inputs):
    output = []
    for input in inputs:
        output.append((func(input)).real)
    return output

def LabviewCalculateYArray(circ_consts, params, f_input, scansize, current_sig, rangesize):
    
    #---------------------preamble----------------
    
    pi = np.pi
    im_unit = complex(0,1)
    sign = 1
    
    #----------------------main------------------

    L0 = circ_consts[0]
    Rcoil = circ_consts[1]
    R = circ_consts[2]
    R1 = circ_consts[3]
    r = circ_consts[4]
    alpha = circ_consts[5]
    beta1 = circ_consts[6]
    Z_cable = circ_consts[7]
    D = circ_consts[8]
    M = circ_consts[9]
    delta_C = circ_consts[10]
    delta_phi = circ_consts[11]
    delta_phase = circ_consts[12]
    delta_l = circ_consts[13]
    
    
    f = f_input
    
    U = params[0]
    knob = params[1]
    trim = params[2]
    eta = params[3]
    phi_const = params[4]
    Cstray = params[5]
    
    I = U*1000/R #Ideal constant current, mA

    #Derived quantities
    w_res = 2*pi*f
    w_low = 2 * pi * (213 - scansize) * (1e6)
    w_high = 2 * pi * (213 + scansize) * (1e6)
    delta_w = 2 * pi * 500 * ((1e3)/500)
    
    #Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope()*(w - w_res)

    def Cmain():
        return 20*(1e-12)*knob

    def C(w):
        return Cmain() + Ctrim(w)*(1e-12)

    def Cpf():
        return C(w_res)*(1e12)
    
    
    #--------------------Cable characteristics-------------


    #Derived quantities
    S = 2*Z_cable*alpha

    #Functions

    def Z0(w):
        return cmath.sqrt( (S + w*M*im_unit) / (w*D*im_unit))

    def beta(w):
        return beta1*w

    def gamma(w):
        return complex(alpha,beta(w))

    def ZC(w):
        if  w != 0 and C(w) != 0:
            return 1/(im_unit*w*C(w))
        return 1
  
    #More derived quantities
    vel = 1/beta(1)

    #More functions
    def lam(w):
        return vel/f
    
    #Even more derived quantities
    l_const = trim*lam(w_res)

    #Even more functions
    def l(w):
        return l_const + delta_l
        
 
        
    
    #Variables for creating splines
    k_ints = range(0,500)
    k = np.array(k_ints,float)
    x = (k*delta_w)+(w_low)
    Icoil_TE = 0.11133
    
    # butxi = []
    # butxii = []
    # vback = []
    # vreal = []    
    Icoil = []
    
    # for item in deriv_sig:
    #     butxi.append(item)
    # for item in main_sig:
    #     butxii.append(item)
    # for item in backgmd_sig:
    #     vback.append(item)
    # for item in backreal_sig:
    #     vreal.append(item)
    for item in current_sig:
        Icoil.append(item)
    
    # x1 = interpolate.interp1d(x,butxi,fill_value=0.0,bounds_error=False)
    # x2 = interpolate.interp1d(x,butxii,fill_value=0.0,bounds_error=False)
    # b = interpolate.interp1d(x,vback,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    # rb = interpolate.interp1d(x,vreal,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    ic = interpolate.interp1d(x,Icoil,fill_value="extrapolate",kind="linear",bounds_error=False)
    x1 = Baseline_Polynomial_Curve
    x2 = Baseline_Polynomial_Curve

    
    def chi(w):
        return complex(x1(w),-1*x2(w))

    def pt(w):
        return ic(w)/Icoil_TE

    def L(w):
        return L0*(1+(sign*4*pi*eta*pt(w)*chi(w)))

    def real_L(w):
        return L(w).real

    def imag_L(w):
        return L(w).imag

    def ZLpure(w):
        return im_unit*w*L(w) + Rcoil

    def Zstray(w):
        if w != 0 and Cstray !=0:
            return 1/(im_unit*w*Cstray)
        return 1

    def ZL(w):
        return ZLpure(w)*Zstray(w)/(ZLpure(w)+Zstray(w))

    def ZT(w):
        return Z0(w)*(ZL(w) + Z0(w)*np.tanh(gamma(w)*l(w)))/(Z0(w) + ZL(w)*np.tanh(gamma(w)*l(w)))


    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)) )

    #Adding parabolic term

    xp1 = w_low
    xp2 = w_res
    xp3 = w_high
    yp1 = 0
    yp2 = delta_phase
    yp3 = 0

    alpha_1 = yp1-yp2
    alpha_2 = yp1-yp3
    beta_1 = (xp1*xp1) - (xp2*xp2)
    beta_2 = (xp1*xp1) - (xp3*xp3)
    gamma_1 = xp1-xp2
    gamma_2 = xp1-xp3
    temp=(beta_1*(gamma_1/gamma_2) - beta_2)
    a= (gamma_2 *(alpha_1/gamma_1) - alpha_2)/temp
    bb = (alpha_2 - a*beta_2)/gamma_2
    c = yp1 - a*xp1*xp1 - bb*xp1

    def parfaze(w):
        return a*w*w + bb*w + c

    def phi_trim(w):
        return slope_phi()*(w-w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1*(I*Ztotal(w)*np.exp(im_unit*phi(w)*pi/180))

    
    larger_k = range(0,k_range)
    larger_x = np.array(larger_k, float)
    w_range = w_high - w_low
    larger_range = (delta_w*larger_x)+(w_low-5*w_range)
    
    out_y = getArrayFromFunc(V_out,x)
    if (rangesize == 1):
        out_y = getArrayFromFunc(V_out,larger_range)
    return out_y
    

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

center = 250
length = range(500)
norm_array = []
for x in length:
    norm_array = np.append(norm_array,(x - center)*(12/500))  
Iplus = icurve(norm_array,1)
Iminus = icurve(norm_array,-1)
ratio = Iminus/Iplus

R_arr = []
P_arr = []
SNR_arr = []
U = 0.1
for x in range(0,10):
    P = np.random.uniform(0,1)
    # Cknob = 0.180 + np.random.uniform(-.07,.07)
    # Cknob = .01522 + np.random.uniform(-.5,.5)
    Cknob = 1.38 + np.random.uniform(-.5,.5)
    cable = 23/2
    eta = 0.0104
    phi = 6.1319
    Cstray = 10**(-15)
    circ_params = (U,Cknob,cable,eta,phi,Cstray)
    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    Iminus = icurve(norm_array,-1)
    array = r*Iminus
    array_flipped = np.flip(array)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = P
    signal = element_1*element_2*element_3/1000
    baseline = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, Current, ranger)
    shape = np.array(signal) + np.array(baseline)
    # shape = np.array(baseline)
    offset = np.array([x - min(shape) for x in shape])/200
    noise = choose_random_row(df_filtered)
    # noise = np.zeros(500)
    amplitude_shift = np.ones(500,)
    # sinusoidal_shift = Sinusoidal_Noise(500,)
    # sig = offset + noise + np.multiply(amplitude_shift,np.random.uniform(-0.01,0.01)) + sinusoidal_shift
    sig = offset + noise 
    x_sig = max(list(map(abs, shape)))
    y_sig = max(list(map(abs,noise)))
    SNR = (x_sig/y_sig)
    R_arr.append(sig)
    P_arr.append(np.round(P,6))
    SNR_arr.append(SNR)
df = pd.DataFrame(R_arr)
df['P'] = P_arr
df['SNR'] = SNR_arr
# df.to_csv('Testing_Data_v5/Sample_Data' + str(sys.argv[1]) + '.csv',index=False)
df.to_csv('Test.csv',index=False)
