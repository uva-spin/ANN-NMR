from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import random
import os.path
import datetime
from dateutil.parser import parse
import json
import pytz
from scipy import optimize
import numpy as np
from scipy import interpolate
import cmath
import matplotlib.pyplot as plt
import statistics as std
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.python.keras.models import load_model

Current = np.loadtxt(r'J:\Users\Devin\Desktop\Spin Physics Work\Deuteron\New_Current.csv', unpack = True)


def choose_random_row(csv_file):
    df = csv_file
    if df.empty:
        return None  # If the DataFrame is empty
    random_index = np.random.randint(0, len(df))  # Generate a random index
    random_row = df.iloc[random_index]  # Get the row at the random index
    return random_row

def Sinusoidal_Noise(shape):
    # Generate an array of random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2*np.pi, shape)
    
    # Calculate cosine and sine of each angle
    cos_values = np.random.uniform(-0.005,0.005)*np.cos(angles)
    sin_values = np.random.uniform(-0.005,0.005)*np.sin(angles)
    
    # Sum cosine and sine
    result = cos_values + sin_values
    
    return result


class Config():

    ''' Contains Relevant Parameter Data for NMR Simulation
    Arguments:
        U: Voltage
        Cknob: Capacitance
        Cable: Cable length
        Eta: Filling factor
        Phi: Const
        Cstray: stray capacitance

        k_range: sweep range
        circ_constants: circuitry constant tuple
        f_input: frequency median

        scansize: sweep size
        ranger: range covered

        Current: baseline current simulation

        Other constants: g, s, bigy
    '''

    def __init__(self, params, k_range, f_input, scansize, ranger, current,noise):

        self.circ_consts = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)

        self.L0 = self.circ_consts[0]
        self.Rcoil = self.circ_consts[1]
        self.R = self.circ_consts[2]
        self.R1 = self.circ_consts[3]
        self.r = self.circ_consts[4]
        self.alpha = self.circ_consts[5]
        self.beta1 = self.circ_consts[6]
        self.Z_cable = self.circ_consts[7]
        self.D = self.circ_consts[8]
        self.M = self.circ_consts[9]
        self.delta_C = self.circ_consts[10]
        self.delta_phi = self.circ_consts[11]
        self.delta_phase = self.circ_consts[12]
        self.delta_l = self.circ_consts[13]

        # self.f = f_input
    
        # self.U = params[0]
        # self.knob = params[1]
        # self.trim = params[2]
        # self.eta = params[3]
        # self.phi_const = params[4]
        # self.Cstray = params[5]

        self.g = 0.05
        self.s = 0.04
        self.bigy=(3-self.s)**0.5

        # self.scansize = scansize
        # self.k_range = k_range
        # self.rangesize = ranger

        # self.main_sig = main_sig
        # self.deriv_sig = deriv_sig
        self.current_sig = current
        self.noise_df = noise
 
class Simulate():

    def __init__(self,configuration):
        self.Inputs = configuration

    def LabviewCalculateXArray(self,circ_params, scansize, k_range, ranger, f_input):
        
        #---------------------preamble----------------
        
        pi = np.pi
        im_unit = complex(0,1)

            
        U = circ_params[0]
        knob = circ_params[1]
        trim = circ_params[2]
        eta = circ_params[3]
        phi_const = circ_params[4]
        Cstray = circ_params[5]

        
        scansize = scansize
        k_range = k_range
        rangesize = ranger
        f = f_input

        #----------------------main------------------
        
        
        

        #Derived quantities
        w_res = 2*pi*f
        f_small = f/(1e6)
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
    
    def getArrayFromFunc(self,func,inputs):
        output = []
        for input in inputs:
            output.append((func(input)).real)
        return output
    
    def Baseline_Polynomial_Curve(w):
        return -1.84153246e-07*w**2 + 8.42855076e-05*w - 1.11342243e-04

    def LabviewCalculateYArray(self, circ_params,f_input, scansize, current_sig, k_range,rangesize):
        
        #---------------------preamble----------------
        
        pi = np.pi
        im_unit = complex(0,1)
        sign = 1
        
        #----------------------main------------------

        L0 = self.Inputs.L0
        Rcoil = self.Inputs.Rcoil
        R = self.Inputs.R
        R1 = self.Inputs.R1
        r = self.Inputs.r
        alpha = self.Inputs.alpha
        beta1 = self.Inputs.beta1
        Z_cable = self.Inputs.Z_cable
        D = self.Inputs.D
        M = self.Inputs.M
        delta_C = self.Inputs.delta_C
        delta_phi = self.Inputs.delta_phi
        delta_phase = self.Inputs.delta_phase
        delta_l = self.Inputs.delta_l

        self.circ_consts = ()
        
        
        f = f_input
        
        U = circ_params[0]
        knob = circ_params[1]
        trim = circ_params[2]
        eta = circ_params[3]
        phi_const = circ_params[4]
        Cstray = circ_params[5]

        current_sig = self.Inputs.current_sig
        # main_sig = self.Inputs.main_sig
        # deriv_sig = self.Inputs.deriv_sig
        
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
        
        butxi = []
        butxii = []
        vback = []
        vreal = []    
        Icoil = []
        
        # for item in main_sig:
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
        x1 = Simulate.Baseline_Polynomial_Curve
        x2 = Simulate.Baseline_Polynomial_Curve

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
        
        out_y = Simulate.getArrayFromFunc(self,V_out,x)
        if (rangesize == 1):
            out_y = Simulate.getArrayFromFunc(self,V_out,larger_range)
        return out_y

    def cosal(self,x,eps):
        return (1-eps*x-self.Inputs.s)/Simulate.bigxsquare(self,x,eps)

    def c(self,x):
        return ((self.Inputs.g**2+(1-x-self.Inputs.s)**2)**0.5)**0.5

    def bigxsquare(self,x,eps):
        return (self.Inputs.g**2+(1-eps*x-self.Inputs.s)**2)**0.5

    def mult_term(self,x,eps):
        return float(1)/(2*np.pi*np.sqrt(Simulate.bigxsquare(self,x,eps)))

    def cosaltwo(self,x,eps):
        return ((1+Simulate.cosal(self,x,eps))/2)**0.5

    def sinaltwo(self,x,eps):
        return ((1-Simulate.cosal(self,x,eps))/2)**0.5

    def termone(self,x,eps):
        return np.pi/2+np.arctan((self.Inputs.bigy**2-Simulate.bigxsquare(self,x,eps))/((2*self.Inputs.bigy*(Simulate.bigxsquare(self,x,eps))**0.5)*Simulate.sinaltwo(self,x,eps)))


    def termtwo(self,x,eps):
        return np.log((self.Inputs.bigy**2+Simulate.bigxsquare(self,x,eps)+2*self.Inputs.bigy*(Simulate.bigxsquare(self,x,eps)**0.5)*Simulate.cosaltwo(self,x,eps))/(self.Inputs.bigy**2+Simulate.bigxsquare(self,x,eps)-2*self.Inputs.bigy*(Simulate.bigxsquare(self,x,eps)**0.5)*Simulate.cosaltwo(self,x,eps)))
    def icurve(self,x,eps):
        return Simulate.mult_term(self,x,eps)*(2*Simulate.cosaltwo(self,x,eps)*Simulate.termone(self,x,eps)+Simulate.sinaltwo(self,x,eps)*Simulate.termtwo(self,x,eps))
    
    def Lineshape(self,P, circ_params, f_input, scan_s, ranger, k_range,noise_df):
        # xvals = np.linspace(-6,6,500)
        # yvals = Simulate.icurve(self,xvals,1)/10
        # yvals2 = Simulate.icurve(self,-xvals,1)/10

        center = 250
        length = range(500)
        norm_array = []
        for x in length:
            norm_array = np.append(norm_array,(x - center)*(12/500))  
        # Iplus = Simulate.icurve(self,norm_array,1)
        # Iminus = Simulate.icurve(self,norm_array,-1)
        # ratio = Iminus/Iplus
        # r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
        # Iminus = Simulate.icurve(self,norm_array,-1)
        # array = r*Iminus
        # array_flipped = np.flip(array)
        # element_1 = array_flipped+Iminus
        # sum_array = np.sum(array_flipped)*(12/500)
        # element_2 = 1/sum_array
        # element_3 = P
        # signal = element_1*element_2*element_3
        # lineshape = Simulate.LabviewCalculateYArray(self,signal,circ_params,scan_s,k_range,ranger,f_input)
        # offset = [x - max(lineshape) for x in lineshape]
        # offset = np.array(offset)
        # noise = choose_random_row(noise_df)
        # sig = offset + noise
        r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
        Iminus = Simulate.icurve(self,norm_array,-1)
        array = r*Iminus
        array_flipped = np.flip(array)
        element_1 = array_flipped+Iminus
        sum_array = np.sum(array_flipped)*(12/500)
        element_2 = 1/sum_array
        element_3 = P
        signal = -element_1*element_2*element_3/1000
        baseline = Simulate.LabviewCalculateYArray(self,circ_params, f_input, scan_s, Current, k_range,ranger)
        shape = np.array(signal) + np.array(baseline)
        offset = np.array([x - min(shape) for x in shape])
        noise = choose_random_row(noise_df)
        # noise = np.zeros(500)
        amplitude_shift = np.ones(500,)
        sinusoidal_shift = Sinusoidal_Noise(500,)
        sig = offset + noise + np.multiply(amplitude_shift,np.random.uniform(-0.01,0.01)) + sinusoidal_shift
        # sig = offset + noise + np.multiply(amplitude_shift,np.random.uniform(-0.01,0.01))
        sig = offset + noise
        return sig


def Predict(X,Polarization, testmodel):
    #X = np.array(X)
    acc = []
    X = np.reshape(X,(1,500))
    Y = testmodel.predict(X,batch_size=10000)
    Y = Y.reshape((len(Y),))
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
    
    p = Polarization
    p_pred = Y
    accuracy = ((p - Y)/p)*100
    acc.append(accuracy)
    #snr = SNR[i]
    r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
    r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
    center = 250
    length = range(500)
    norm_array = []
    norm_array_pred = []
    for x in length:
        norm_array = np.append(norm_array,(x - center)*(12/500))  
        norm_array_pred = norm_array
    Iplus = icurve(norm_array,1)
    Iminus = icurve(norm_array,-1)
    ratio = Iminus/Iplus
    array = r*Iminus
    array_pred = r_pred*Iminus
    array_flipped = np.flip(array)
    array_pred_flipped = np.flip(array_pred)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = p
    element_1_pred = array_pred_flipped + Iminus
    sum_array_pred = np.sum(array_pred_flipped)*(12/500)
    element_2_pred = 1/sum_array_pred
    element_3_pred = p_pred
    result = element_1*element_2*element_3
    result_pred = element_1_pred*element_2_pred*element_3_pred
    result_new = result.reshape(500,)
    result_pred_new = result_pred.reshape(500,)
    return p, p_pred, result_new, result_pred_new