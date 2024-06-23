import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFormLayout, QLineEdit, QPushButton
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import random
import os.path
import datetime
from dateutil.parser import parse
import json
import pytz
from scipy import optimize, interpolate
import cmath
import matplotlib.pyplot as plt
import pandas as pd
import statistics as std

Current = np.loadtxt(r'J:\Users\Devin\Desktop\Spin Physics Work\Deuteron\New_Current.csv', unpack=True)

def choose_random_row(df):
    if df.empty:
        return None
    random_index = np.random.randint(0, len(df))
    random_row = df.iloc[random_index]
    return random_row

def Sinusoidal_Noise(shape):
    angles = np.random.uniform(0, 2*np.pi, shape)
    cos_values = np.random.uniform(-0.005, 0.005) * np.cos(angles)
    sin_values = np.random.uniform(-0.005, 0.005) * np.sin(angles)
    result = cos_values + sin_values
    return result

class Config:
    def __init__(self, params, k_range, f_input, scansize, ranger, current, noise):
        self.circ_consts = (3*10**(-8), 0.35, 619, 50, 10, 0.0343, 4.752*10**(-9), 50, 1.027*10**(-10), 2.542*10**(-7), 0, 0, 0, 0)
        self.L0, self.Rcoil, self.R, self.R1, self.r, self.alpha, self.beta1, self.Z_cable, self.D, self.M, self.delta_C, self.delta_phi, self.delta_phase, self.delta_l = self.circ_consts
        self.g = 0.05
        self.s = 0.04
        self.bigy = (3 - self.s) ** 0.5
        self.current_sig = current
        self.noise_df = noise

class Simulate:
    def __init__(self, configuration):
        self.Inputs = configuration

    def LabviewCalculateXArray(self, circ_params, scansize, k_range, ranger, f_input):
        pi = np.pi
        U, knob, trim, eta, phi_const, Cstray = circ_params
        w_res = 2 * pi * f_input
        w_low = 2 * pi * (f_input - scansize) * 1e6
        w_high = 2 * pi * (f_input + scansize) * 1e6
        delta_w = 2 * pi * 500 * (1e3 / 500)
        k_ints = range(0, 500)
        k = np.array(k_ints, float)
        x = (k * delta_w) + w_low
        larger_k = range(0, k_range)
        larger_x = np.array(larger_k, float)
        w_range = w_high - w_low
        larger_range = (delta_w * larger_x) + (w_low - 5 * w_range)
        larger_range /= 2 * pi * 1e6
        x /= 2 * pi * 1e6
        return larger_range if ranger == 1 else x

    def getArrayFromFunc(self, func, inputs):
        return [func(input).real for input in inputs]

    def LabviewCalculateYArray(self, circ_params, f_input, scansize, current_sig, k_range, ranger):
        pi = np.pi
        im_unit = complex(0, 1)
        sign = 1
        L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = self.Inputs.circ_consts
        U, knob, trim, eta, phi_const, Cstray = circ_params
        I = U * 1000 / R
        w_res = 2 * pi * f_input
        w_low = 2 * pi * (213 - scansize) * 1e6
        w_high = 2 * pi * (213 + scansize) * 1e6
        delta_w = 2 * pi * 500 * (1e3 / 500)
        k_ints = range(0, 500)
        k = np.array(k_ints, float)
        x = (k * delta_w) + w_low
        ic = interpolate.interp1d(x, current_sig, fill_value="extrapolate", kind="linear", bounds_error=False)
        def slope():
            return delta_C / (0.25 * 2 * pi * 1e6)
        def slope_phi():
            return delta_phi / (0.25 * 2 * pi * 1e6)
        def Ctrim(w):
            return slope() * (w - w_res)
        def Cmain():
            return 20 * 1e-12 * knob
        def C(w):
            return Cmain() + Ctrim(w) * 1e-12
        def Z0(w):
            return cmath.sqrt((2 * Z_cable * alpha + w * M * im_unit) / (w * D * im_unit))
        def beta(w):
            return beta1 * w
        def gamma(w):
            return complex(alpha, beta(w))
        def ZC(w):
            return 1 / (im_unit * w * C(w)) if w != 0 and C(w) != 0 else 1
        def ZLpure(w):
            return im_unit * w * L(w) + Rcoil
        def Zstray(w):
            return 1 / (im_unit * w * Cstray) if w != 0 and Cstray != 0 else 1
        def ZL(w):
            return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))
        def ZT(w):
            return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)))
        def Zleg1(w):
            return r + ZC(w) + ZT(w)
        def Ztotal(w):
            return R1 / (1 + (R1 / Zleg1(w)))
        def V_out(w):
            return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))
        larger_k = range(0, k_range)
        larger_x = np.array(larger_k, float)
        w_range = w_high - w_low
        larger_range = (delta_w * larger_x) + (w_low - 5 * w_range)
        out_y = self.getArrayFromFunc(V_out, x)
        if ranger == 1:
            out_y = self.getArrayFromFunc(V_out, larger_range)
        return out_y

    def Lineshape(self, P, circ_params, f_input, scan_s, ranger, k_range, noise_df):
        center = 250
        length = range(500)
        norm_array = np.array([(x - center) * (12 / 500) for x in length])
        r = (np.sqrt(4 - 3 * P ** 2) + P) / (2 - 2 * P)
        Iminus = self.icurve(norm_array, -1)
        array = r * Iminus
        array_flipped = np.flip(array)
        element_1 = array_flipped + Iminus
        sum_array = np.sum(array_flipped) * (12 / 500)
        element_2 = 1 / sum_array
        element_3 = P
        signal = -element_1 * element_2 * element_3 / 1000
        baseline = self.LabviewCalculateYArray(circ_params, f_input, scan_s, Current, k_range, ranger)
        shape = np.array(signal) + np.array(baseline)
        offset = np.array([x - min(shape) for x in shape])
        noise = choose_random_row(noise_df)
        amplitude_shift = np.ones(500,)
        sinusoidal_shift = Sinusoidal_Noise(500,)
        sig = offset + noise + np.multiply(amplitude_shift, np.random.uniform(-0.01, 0.01)) + sinusoidal_shift
        return sig

def Predict(X, Polarization, testmodel):
    acc = []
    X = np.reshape(X, (1, 500))
    Y = testmodel.predict(X, batch_size=10000)
    Y = Y.reshape((len(Y),))
    g = 0.05
    s = 0.04
    bigy = (3 - s) ** 0.5
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
