import numpy as np
from scipy.special import wofz
import sys
import os
import tensorflow as tf
from scipy.signal import hilbert
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Variables import *

def FrequencyBound(f):
    # Define the domain to fit (bins 100 to 400)
    fit_start_bin, fit_end_bin = 0, 500

    # Frequency conversion factors
    bin_to_freq = 0.0015287  # MHz per bin
    start_freq = f  # Starting frequency in MHz

    x_full_bins = np.arange(500)  # Full range of bins
    x_full_freq = start_freq + x_full_bins * bin_to_freq  # Convert bins to frequency

    x_bins = x_full_bins[fit_start_bin:fit_end_bin+1]
    x_freq = x_full_freq[fit_start_bin:fit_end_bin+1]

    return x_freq,  x_full_freq[0], x_full_freq[-1]

def FrequencyBoundTensor(f):
    """
    Calculate frequency bounds based on Larmor frequency.
    Converted to TensorFlow operations.
    """
    # Convert input to tensor
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    
    # Define the domain to fit
    fit_start_bin, fit_end_bin = 0, 500
    
    # Frequency conversion factors
    bin_to_freq = tf.constant(0.0015287, dtype=tf.float32)  # MHz per bin
    start_freq = f  # Starting frequency in MHz
    
    # Create full range of bins using TensorFlow
    x_full_bins = tf.range(500, dtype=tf.float32)
    
    # Convert bins to frequency
    x_full_freq = start_freq + x_full_bins * bin_to_freq
    
    return x_full_freq, x_full_freq[0], x_full_freq[-1]

def Voigt(x, amp, s, g, x0):
    """
    Voigt profile function with an adjustable center (x0).
    
    :param x: Array of x values
    :param amp: Amplitude of the Voigt profile
    :param s: Width of the Gaussian component (sigma)
    :param g: Width of the Lorentzian component (gamma)
    :param x0: Center of the Voigt profile
    :return: Voigt profile values
    """
    z = (x - x0 + 1j * g) / (s * np.sqrt(2.0))
    v = wofz(z)  # Faddeeva function for Voigt profile
    out = amp * (np.real(v) / (s * np.sqrt(2 * np.pi)))
    return out


def Signal(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset,ampG1, sigmaG1, ampL1, widL1, center):
    # Preamble
    circ_consts = (3*10**(-8), 0.35, 619, 50, 10, 0.0343, 4.752*10**(-9), 50, 1.027*10**(-10), 2.542*10**(-7), 0, 0, 0, 0)
    pi = np.pi
    im_unit = 1j  # Use numpy's complex unit (1j)
    sign = 1

    # Main constants
    L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = circ_consts

    I = U*1000/R  # Ideal constant current, mA
    w_res = 2 * pi * 213e6
    w_low = 2 * pi * (213 - 4) * 1e6
    w_high = 2 * pi * (213 + 4) * 1e6
    delta_w = 2 * pi * 4e6 / 500
    
    trim = tf.cast(trim, tf.complex64)

    # Convert frequency to angular frequency (rad/s)
    w = 2 * pi * f * 1e6

    # Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope() * (w - w_res)

    def Cmain():
        return 20 * 1e-12 * Cknob

    def C(w):
        return Cmain() + Ctrim(w) * 1e-12

    def Z0(w):
        S = 2 * Z_cable * alpha
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.sqrt((S + w * M * im_unit) / (w * D * im_unit))
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def beta(w):
        return beta1 * w

    def gamma(w):
        return alpha + beta(w) * 1j  # Create a complex number using numpy

    def ZC(w):
        Cw = C(w)
        # Ensure both parts are of the same type
        real_part = tf.cast(1.0, tf.complex64)  # Cast to complex64
        imaginary_part = tf.cast(w, tf.complex64) * tf.cast(Cw, tf.complex64)
        return real_part / tf.complex(0.0, imaginary_part)

    def vel(w):
        return 1 / beta(w)

    def l(w):
        return trim * vel(w_res) + delta_l

    def ic(w):
        return 1.113325582555695081e-01
    
    def x1(x):
        return Voigt(x,ampG1, sigmaG1, ampL1, widL1, center)
    
    def x2(x):
        return Voigt(x,ampG1, sigmaG1, ampL1, widL1, center)
    
    def chi(x):
        return complex(x1(x),-x2(x))

    def pt(w):
        return ic(w)

    def L(w):
        return L0 * (1 + sign * 4 * pi * eta * pt(w) * chi(w))

    def ZLpure(w):
        return im_unit * w * L(w) + Rcoil

    def Zstray(w):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cstray != 0, 1 / (im_unit * w * Cstray), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def ZL(w):
        return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))

    def ZT(w):
        epsilon = 1e-10  # Small constant to avoid division by zero
        return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)) + epsilon)

    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)))

    def parfaze(w):
        xp1 = w_low
        xp2 = w_res
        xp3 = w_high
        yp1 = 0
        yp2 = delta_phase
        yp3 = 0

        a = ((yp1 - yp2) * (w_low - w_high) - (yp1 - yp3) * (w_low - w_res)) / \
            (((w_low ** 2) - (w_res ** 2)) * (w_low - w_high) - ((w_low ** 2) - (w_high ** 2)) * (w_low - w_res))
        bb = (yp1 - yp3 - a * ((w_low ** 2) - (w_high ** 2))) / (w_low - w_high)
        c = yp1 - a * (w_low ** 2) - bb * w_low
        return a * w ** 2 + bb * w + c

    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))

    out_y = V_out(w)
    offset = np.array([x - min(out_y.real) for x in out_y.real])
    return offset.real + DC_offset

def Baseline(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset):
    # Preamble
    circ_consts = (3*10**(-8), 0.35, 619, 50, 10, 0.0343, 4.752*10**(-9), 50, 1.027*10**(-10), 2.542*10**(-7), 0, 0, 0, 0)
    pi = np.pi
    im_unit = 1j  
    sign = 1

    # Main constants
    L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = circ_consts

    I = U*1000/R  # Ideal constant current, mA
    # w_res = 2 * pi * 213e6
    # w_low = 2 * pi * (213 - 4) * 1e6
    # w_high = 2 * pi * (213 + 4) * 1e6
    # delta_w = 2 * pi * 4e6 / 500

    w_res = 2 * pi * 32e6
    w_low = 2 * pi * (32 - 4) * 1e6
    w_high = 2 * pi * (32 + 4) * 1e6
    delta_w = 2 * pi * 4e6 / 500

    # Convert frequency to angular frequency (rad/s)
    w = 2 * pi * f * 1e6

    # Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope() * (w - w_res)

    def Cmain():
        return 20 * 1e-12 * Cknob

    def C(w):
        return Cmain() + Ctrim(w) * 1e-12

    def Z0(w):
        S = 2 * Z_cable * alpha
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.sqrt((S + w * M * im_unit) / (w * D * im_unit))
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def beta(w):
        return beta1 * w

    def gamma(w):
        return alpha + beta(w) * 1j  # Create a complex number using numpy

    def ZC(w):
        Cw = C(w)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cw != 0, 1 / (im_unit * w * Cw), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def vel(w):
        return 1 / beta(w)

    def l(w):
        return trim * vel(w_res) + delta_l

    def ic(w):
        return 0.11133

    def chi(w):
        return np.zeros_like(w)  # Placeholder for x1(w) and x2(w)

    def pt(w):
        return ic(w)

    def L(w):
        return L0 * (1 + sign * 4 * pi * eta * pt(w) * chi(w))

    def ZLpure(w):
        return im_unit * w * L(w) + Rcoil

    def Zstray(w):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(Cstray != 0, 1 / (im_unit * w * Cstray), 0)
        return np.where(w == 0, 0, result)  # Avoid invalid values for w=0

    def ZL(w):
        return ZLpure(w) * Zstray(w) / (ZLpure(w) + Zstray(w))

    def ZT(w):
        epsilon = 1e-10  # Small constant to avoid division by zero
        return Z0(w) * (ZL(w) + Z0(w) * np.tanh(gamma(w) * l(w))) / (Z0(w) + ZL(w) * np.tanh(gamma(w) * l(w)) + epsilon)

    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)))

    def parfaze(w):
        xp1 = w_low
        xp2 = w_res
        xp3 = w_high
        yp1 = 0
        yp2 = delta_phase
        yp3 = 0

        a = ((yp1 - yp2) * (w_low - w_high) - (yp1 - yp3) * (w_low - w_res)) / \
            (((w_low ** 2) - (w_res ** 2)) * (w_low - w_high) - ((w_low ** 2) - (w_high ** 2)) * (w_low - w_res))
        bb = (yp1 - yp3 - a * ((w_low ** 2) - (w_high ** 2))) / (w_low - w_high)
        c = yp1 - a * (w_low ** 2) - bb * w_low
        return a * w ** 2 + bb * w + c

    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1 * (I * Ztotal(w) * np.exp(im_unit * phi(w) * pi / 180))

    out_y = V_out(w)
    offset = np.array([x - min(out_y.real) for x in out_y.real])
    
    return offset.real + DC_offset

# @tf.function
def BaselineTensor(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset):
    
    tf.config.run_functions_eagerly(True)

    """
    Calculate baseline signal with TensorFlow operations.
    
    Parameters:
    -----------
    f : float or tensor
        Frequency in MHz
    U : float or tensor
        Voltage
    Cknob : float or tensor
        Capacitance knob setting
    eta : float or tensor
        Eta parameter
    trim : float or tensor
        Trim parameter
    Cstray : float or tensor
        Stray capacitance
    phi_const : float or tensor
        Phase constant (degrees)
    DC_offset : float or tensor
        DC offset
        
    Returns:
    --------
    tensor
        Baseline signal with offset applied
    """
    # f = tf.convert_to_tensor(f, dtype=tf.float32)
    U = tf.complex(U, 0.0)
    Cknob = tf.complex(Cknob, 0.0)
    eta = tf.complex(eta, 0.0)
    trim = tf.complex(trim, 0.0)
    Cstray = tf.complex(Cstray, 0.0)
    phi_const = tf.complex(phi_const, 0.0)
    # DC_offset = tf.complex(DC_offset, 0.0)
    
    f = tf.complex(f, 0.0)
    
    # Constants
    pi = tf.complex(3.14159265358979323846, 0.0)
    
    # Circuit constants
    L0 = tf.complex(3e-8, 0.0)       # Inductance
    Rcoil = tf.complex(0.35, 0.0)    # Coil resistance
    R = tf.complex(619.0, 0.0)       # Resistance
    R1 = tf.complex(50.0, 0.0)       # R1 resistance
    r = tf.complex(10.0, 0.0)        # r resistance
    alpha = tf.complex(0.0343, 0.0)  # Alpha parameter
    beta1 = tf.complex(4.752e-9, 0.0) # Beta parameter
    Z_cable = tf.complex(50.0, 0.0)  # Cable impedance
    D = tf.complex(1.027e-10, 0.0)   # D parameter
    M = tf.complex(2.542e-7, 0.0)    # M parameter
    
    delta_C = tf.complex(0.0, 0.0)
    delta_phi = tf.complex(0.0, 0.0)
    delta_phase = tf.complex(0.0, 0.0)
    delta_l = tf.complex(0.0, 0.0)
    
    sign = tf.complex(1.0, 0.0)
    
    # Calculate ideal constant current (mA)
    I = U * 1000.0 / R
    
    w_res = 2.0 * pi * (32 - 4) * 1e6            # Resonant angular frequency
    w_low = 2.0 * pi * (32 - 4) * 1e6              # Lower bound angular frequency
    w_high = 2.0 * pi * (32 + 4) * 1e6             # Upper bound angular frequency
    
    # Convert frequency to angular frequency (rad/s)
    w = 2.0 * pi * f * 1e6
    
    w = tf.cast(w, tf.complex64)
    
    # Slope functions
    def slope():
        return delta_C / (0.25 * 2.0 * pi * 1e6)
    
    def slope_phi():
        return delta_phi / (0.25 * 2.0 * pi * 1e6)
    
    # Capacitance functions
    def Cmain():
        return 20.0 * 1e-12 * Cknob
    
    def Ctrim(w):
        return slope() * (w - w_res)
    
    def C(w):
        return Cmain() + Ctrim(w) * 1e-12
    
    # Impedance and related functions
    def Z0(w):
        
        S = 2.0 * Z_cable * alpha
        return tf.sqrt((S + w * M) / (w * D))
    
    def beta(w):
        return tf.cast(beta1 * w, tf.complex64)
    
    def gamma(w):
        return tf.complex(
            tf.cast(alpha, tf.float32),
            tf.cast(beta1 * w, tf.float32)
        )
    
    def ZC(w):
        Cw = C(w)
        return 1.0 / w * Cw
    
    def vel(w):
        beta_value = beta(w)
        return tf.cast(1.0, tf.complex64) / beta_value
    
    def l(w):
        return tf.cast(trim, tf.complex64) * tf.cast(vel(w_res), tf.complex64) + tf.cast(delta_l, tf.complex64)
    
    # Material property functions
    def ic(w):
        return tf.ones_like(w) * tf.complex(0.11133, 0.0)
    
    def chi(w):
        return tf.zeros_like(w)
    
    def pt(w):
        return ic(w)
    
    def L(w):
        return L0 * (1.0 + sign * 4.0 * pi * eta * pt(w) * chi(w))
    
    # Impedance functions
    def ZLpure(w):
        return tf.complex(0.0, 1.0) * tf.cast(w, tf.complex64) * tf.cast(L(w), tf.complex64) + tf.cast(Rcoil, tf.complex64)
    
    def Zstray(w):
        return tf.cast(1.0, tf.complex64) / (tf.complex(0.0, 1.0) * tf.cast(w, tf.complex64) * tf.cast(Cstray, tf.complex64))
    
    def ZL(w):
        ZLp = ZLpure(w)
        Zs = Zstray(w)
        
        numerator = ZLp * Zs
        denominator = ZLp + Zs
        return numerator / denominator
    
    def ZT(w):
        Z0w = Z0(w)
        ZLw = ZL(w)
        gamma_l = gamma(w) * l(w)
        tanh_term = tf.tanh(gamma_l)
        
        numerator = Z0w * (ZLw + Z0w * tanh_term)
        denominator = Z0w + ZLw * tanh_term
        
        return numerator / denominator
    
    def Zleg1(w):
        return tf.cast(r, tf.complex64) + ZC(w) + ZT(w)
    
    def Ztotal(w):
        Zleg = Zleg1(w)
        
        one = tf.complex(1.0, 0.0)
        R1_complex = tf.cast(R1, tf.complex64)
        
        denominator = one + (R1_complex / Zleg)
        return R1_complex / denominator
    
    def parfaze(w):
        xp1 = w_low
        xp2 = w_res
        xp3 = w_high
        yp1 = tf.complex(0.0, 0.0)
        yp2 = delta_phase
        yp3 = tf.complex(0.0, 0.0)
        
        
        a = ((yp1 - yp2) * (xp1 - xp3) - (yp1 - yp3) * (xp1 - xp2)) / (((xp1 ** 2) - (xp2 ** 2)) * (xp1 - xp3) - ((xp1 ** 2) - (xp3 ** 2)) * (xp1 - xp2))
        bb = (yp1 - yp3 - a * ((xp1 ** 2) - (xp3 ** 2))) / (xp1 - xp3)
        c = yp1 - a * (xp1 ** 2) - bb * xp1
        # return a * w ** 2 + bb * w + c
        return 0.0
    
    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)
    
    def phi(w):
        return phi_trim(w) + phi_const
    
    # Output voltage calculation
    def V_out(w):
        Z_total = Ztotal(w)
        
        I_complex = tf.cast(I, tf.complex64)
        neg_one = tf.cast(-1.0, tf.complex64)
        
        # Phase calculation in radians
        phi_value = tf.math.real(phi(w))
        pi_float = tf.cast(pi, tf.float32)
        phi_radians = phi_value * pi_float / 180.0
        
        # Complex exponential for phase
        cos_phi = tf.math.cos(phi_radians)
        sin_phi = tf.math.sin(phi_radians)
        exp_term = tf.math.exp(tf.complex(cos_phi, sin_phi))
        
        # Final output calculation
        return neg_one * I_complex * Z_total * exp_term
    
    # Calculate output and apply offset
    out_y = V_out(w)
    real_part = tf.math.real(out_y)
    min_val = tf.reduce_min(real_part)
    offset = real_part - min_val
    parfaze_test = parfaze(w)
    print("Parfaze:", parfaze_test)
    
    return offset + DC_offset


def Lineshape(x,eps):
    def cosal(x, eps):
        return (1 - eps * x - s) / bigxsquare(x, eps)

    def c(x):
        return np.sqrt(np.sqrt(g**2 + (1 - x - s)**2))

    def bigxsquare(x, eps):
        return np.sqrt(g**2 + (1 - eps * x - s)**2)

    def mult_term(x, eps):
        return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return np.sqrt((1 + cosal(x, eps)) / 2)

    def sinaltwo(x, eps):
        return np.sqrt((1 - cosal(x, eps)) / 2)

    def termone(x, eps):
        return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

    def termtwo(x, eps):
        return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                    (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))
    
    return icurve(x,eps)/10



def GenerateVectorLineshape(P,x):
    
    x = (x - 32.68) / 0.6

    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)

    Iplus = r*Lineshape(x,1)
    Iminus = Lineshape(x,-1)

    signal = Iplus + Iminus

    return signal,Iplus,Iminus


def GenerateTensorLineshape(x, P, phi_deg):
    """
    Calculate the total signal for given x, polarization P, and phase angle phi.
    
    Parameters:
    -----------
    x : float or array-like
        The x-coordinate value(s)
    P : float
        Input polarization (between 0 and 1)
    phi_deg : float
        Phase angle in degrees
        
    Returns:
    --------
    float or array-like
        The total signal value(s)
    """
    # System parameters
    g = 0.05
    s = 0.04
    bigy = np.sqrt(3 - s)

    # x = (x - 32.68) / 0.6
    
    # Calculate r from P
    r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
    
    # Convert phase to radians
    phi_rad = np.deg2rad(phi_deg)
    
    # Calculate absorptive signals
    yvals_absorp1 = Lineshape(x, 1)        # χ''₊
    yvals_absorp2 = Lineshape(-x, 1)       # χ''₋
    
    # Calculate dispersive signals using Hilbert transform
    yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
    yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋
    
    # Calculate phase-sensitive linear combination
    Iplus = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    Iminus = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)

    signal = Iplus + Iminus
    
    # Return total signal
    return signal, Iplus, Iminus

        

def SamplingVectorLineshape(P, x, bound):
    """Sampling the lineshape with a stochastic shift to frequency bins.

    Args:
        P (float): Polarization
        x (list): Frequency range
        bound (float): Bound of the shift

    Returns:
        signal (list): Generated lineshape with a stochastic shift
    """
    shift = np.full(len(x),np.random.uniform( -bound , bound))
    x += shift
    ### Generate the lineshape with the shifted 
    signal, _, _ = GenerateVectorLineshape(P,x)
    return signal

def SamplingTensorLineshape(P, x, bound, phi=0):
    """Sampling the lineshape with a stochastic shift to frequency bins.

    Args:
        P (float): Polarization
        x (list): Frequency range
        bound (float): Bound of the shift
        phi (float): Phase angle in degrees

    Returns:
        signal (list): Generated lineshape with a stochastic shift
    """
    shift = np.full(len(x),np.random.uniform( -bound , bound))
    x += shift
    ### Generate the lineshape with the shifted 
    signal, _, _ = GenerateTensorLineshape(x, P, phi)
    return signal

def GenerateLineshapeTensor(P, x): ### Working now
    """
    Generate lineshape based on polarization and frequency range.
    Converted to TensorFlow operations.
    """
    
    P = tf.constant(P, dtype=tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32) 
    
    g = tf.constant(0.05, dtype=tf.float32)
    s = tf.constant(0.04, dtype=tf.float32)
    bigy = tf.constant((3-s)**0.5, dtype=tf.float32)
    
    def bigxsquare(x, eps):
        eps_tensor = tf.constant(eps, dtype=tf.float32)
        return tf.sqrt(g**2 + (1.0 - eps_tensor * x - s)**2)
    
    def cosal(x, eps):
        eps_tensor = tf.constant(eps, dtype=tf.float32)
        return (1.0 - eps_tensor * x - s) / bigxsquare(x, eps)
    
    def c(x):
        return tf.sqrt(tf.sqrt(g**2 + (1.0 - x - s)**2))
    
    def mult_term(x, eps):
        return 1.0 / (2.0 * tf.constant(3.14159265358979323846, dtype=tf.float32) * tf.sqrt(bigxsquare(x, eps)))
    
    def cosaltwo(x, eps):
        return tf.sqrt((1.0 + cosal(x, eps)) / 2.0)
    
    def sinaltwo(x, eps):
        return tf.sqrt((1.0 - cosal(x, eps)) / 2.0)
    
    def termone(x, eps):
        bigx = bigxsquare(x, eps)
        sin_term = sinaltwo(x, eps)
        # Handle division by zero
        safe_denominator = tf.where(
            tf.abs(2.0 * bigy * tf.sqrt(bigx) * sin_term) < 1e-10,
            tf.ones_like(sin_term) * 1e-10,
            2.0 * bigy * tf.sqrt(bigx) * sin_term
        )
        return tf.constant(3.14159265358979323846 / 2.0, dtype=tf.float32) + tf.atan((bigy**2 - bigx) / safe_denominator)
    
    def termtwo(x, eps):
        bigx = bigxsquare(x, eps)
        cos_term = cosaltwo(x, eps)
        sqrt_bigx = tf.sqrt(bigx)
        
        numerator = bigy**2 + bigx + 2.0 * bigy * sqrt_bigx * cos_term
        denominator = bigy**2 + bigx - 2.0 * bigy * sqrt_bigx * cos_term
        
        # Handle division by zero
        safe_denominator = tf.where(
            tf.abs(denominator) < 1e-10,
            tf.ones_like(denominator) * 1e-10,
            denominator
        )
        
        return tf.math.log(numerator / safe_denominator)
    
    def icurve(x, eps):
        return mult_term(x, eps) * (2.0 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))
    

    safe_P = tf.where(tf.abs(P) < 1e-10, tf.ones_like(P) * 1e-10, P)
    r = (tf.sqrt(4.0 - 3.0 * tf.pow(safe_P, 2)) + safe_P) / (2.0 - 2.0 * safe_P)
    
    r_expanded = tf.expand_dims(r, -1)
    
    Iplus = r_expanded * icurve(x, 1) 
    Iminus = icurve(x, -1) 
    signal = Iplus + Iminus
    
    return signal

def Baseline_Polynomial_Curve(w):
    return -1.84153246e-07*w**2 + 8.42855076e-05*w - 1.11342243e-04





