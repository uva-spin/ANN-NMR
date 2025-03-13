import numpy as np
from scipy.special import wofz
import sys
import os
import tensorflow as tf

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

    return x_full_freq,  x_full_freq[0], x_full_freq[-1]

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
    im_unit = 1j  # Use numpy's complex unit (1j)
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

@tf.function
def BaselineTensor(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset):
    """
    Calculate baseline signal. Converted to TensorFlow operations.
    """
    # Convert all inputs to tensors
    f = tf.convert_to_tensor(f, dtype=tf.float32)
    U = tf.convert_to_tensor(U, dtype=tf.float32)
    Cknob = tf.convert_to_tensor(Cknob, dtype=tf.float32)
    eta = tf.convert_to_tensor(eta, dtype=tf.float32)
    trim = tf.convert_to_tensor(trim, dtype=tf.float32)
    Cstray = tf.convert_to_tensor(Cstray, dtype=tf.float32)
    phi_const = tf.convert_to_tensor(phi_const, dtype=tf.float32)
    DC_offset = tf.convert_to_tensor(DC_offset, dtype=tf.float32)
    
    # Preamble with TensorFlow constants
    circ_consts = (
        tf.constant(3e-8, dtype=tf.float32),       # L0
        tf.constant(0.35, dtype=tf.float32),       # Rcoil
        tf.constant(619.0, dtype=tf.float32),      # R
        tf.constant(50.0, dtype=tf.float32),       # R1
        tf.constant(10.0, dtype=tf.float32),       # r
        tf.constant(0.0343, dtype=tf.float32),     # alpha
        tf.constant(4.752e-9, dtype=tf.float32),   # beta1
        tf.constant(50.0, dtype=tf.float32),       # Z_cable
        tf.constant(1.027e-10, dtype=tf.float32),  # D
        tf.constant(2.542e-7, dtype=tf.float32),   # M
        tf.constant(0.0, dtype=tf.float32),        # delta_C
        tf.constant(0.0, dtype=tf.float32),        # delta_phi
        tf.constant(0.0, dtype=tf.float32),        # delta_phase
        tf.constant(0.0, dtype=tf.float32)         # delta_l
    )
    
    pi = tf.constant(3.14159265358979323846, dtype=tf.float32)
    sign = tf.constant(1.0, dtype=tf.float32)
    
    # Unpack circuit constants
    L0, Rcoil, R, R1, r, alpha, beta1, Z_cable, D, M, delta_C, delta_phi, delta_phase, delta_l = circ_consts
    
    # Calculate ideal constant current (mA)
    I = U * 1000.0 / R
    
    # Define frequency ranges
    w_res = 2.0 * pi * 32e6
    w_low = 2.0 * pi * (32.0 - 4.0) * 1e6
    w_high = 2.0 * pi * (32.0 + 4.0) * 1e6
    delta_w = 2.0 * pi * 4e6 / 500.0
    
    # Convert frequency to angular frequency (rad/s)
    w = 2.0 * pi * f * 1e6
    
    def slope():
        denominator = 0.25 * 2.0 * pi * 1e6
        return delta_C / denominator
    
    def slope_phi():
        denominator = 0.25 * 2.0 * pi * 1e6
        return delta_phi / denominator
    
    # Define capacitance functions
    def Ctrim(w):
        return slope() * (w - w_res)
    
    def Cmain():
        return 20.0 * 1e-12 * Cknob
    
    def C(w):
        return Cmain() + Ctrim(w) * 1e-12
    
    # Define impedance and related functions
    def Z0(w):
        S = 2.0 * Z_cable * alpha
        
        return tf.sqrt((S + w * M * tf.complex(0.0,1.0))/(w*D*tf.complex(0.0,1.0)))
    
    def beta(w):
        return tf.cast(beta1 * w , tf.complex32)
    
    def gamma(w):
        return tf.complex(alpha, beta(w))
    
    def ZC(w):
        Cw = C(w)
        
        return (tf.constant(1.0) / tf.complex(0.0,w*Cw))
    def vel(w):
        return tf.constant(1.0) / beta(w)
    
    def l(w):
        return trim * vel(w_res) + delta_l
    
    def ic(w):
        return tf.ones_like(w) * tf.constant(0.11133)
    
    def chi(w):
        return tf.zeros_like(w)
    
    def pt(w):
        return ic(w)
    
    def L(w):
        return L0 * (1.0 + sign * 4.0 * pi * eta * pt(w) * chi(w))
    
    def ZLpure(w):
        return tf.complex(0.0, 1.0) * w * L(w) + Rcoil
    
    def Zstray(w):
        return (tf.constant(1.0)/(tf.complex(0.0,1.0)*w*Cstray))
    
    def ZL(w):
        ZLp = ZLpure(w)
        Zs = Zstray(w)
        # Handle division by zero
        safe_denominator = tf.where(tf.abs(ZLp + Zs) < 1e-10, 
                                   tf.complex(1e-10, 0.0), 
                                   ZLp + Zs)
        return ZLp * Zs / safe_denominator
    
    def ZT(w):
        Z0w = Z0(w)
        ZLw = ZL(w)
        gamma_l = gamma(w) * l(w)
        tanh_term = tf.tanh(gamma_l)
        
        numerator = Z0w * (ZLw + Z0w * tanh_term)
        denominator = Z0w + ZLw * tanh_term
        
        # Handle division by zero
        safe_denominator = tf.where(tf.abs(denominator) < 1e-10, 
                                   tf.complex(1e-10, 0.0), 
                                   denominator)
        return numerator / safe_denominator
    
    def Zleg1(w):
        return r + ZC(w) + ZT(w)
    
    def Ztotal(w):
        Zleg = Zleg1(w)
        # Handle division by zero
        safe_denominator = tf.where(tf.abs(1.0 + (R1 / Zleg)) < 1e-10, 
                                   tf.complex(1e-10, 0.0), 
                                   1.0 + (R1 / Zleg))
        return R1 / safe_denominator
    
    def parfaze(w):
        xp1 = w_low
        xp2 = w_res
        xp3 = w_high
        yp1 = tf.constant(0.0, dtype=tf.float32)
        yp2 = delta_phase
        yp3 = tf.constant(0.0, dtype=tf.float32)
        
        denominator1 = ((xp1**2 - xp2**2) * (xp1 - xp3) - (xp1**2 - xp3**2) * (xp1 - xp2))
        # Handle division by zero
        safe_denominator1 = tf.where(tf.abs(denominator1) < 1e-10, 1e-10, denominator1)
        
        a = ((yp1 - yp2) * (xp1 - xp3) - (yp1 - yp3) * (xp1 - xp2)) / safe_denominator1
        
        denominator2 = xp1 - xp3
        # Handle division by zero
        safe_denominator2 = tf.where(tf.abs(denominator2) < 1e-10, 1e-10, denominator2)
        
        bb = (yp1 - yp3 - a * (xp1**2 - xp3**2)) / safe_denominator2
        c = yp1 - a * xp1**2 - bb * xp1
        
        return a * w**2 + bb * w + c
    
    def phi_trim(w):
        return slope_phi() * (w - w_res) + parfaze(w)
    
    def phi(w):
        return phi_trim(w) + phi_const
    
    def V_out(w):
        Z_total = Ztotal(w)
        phi_radians = phi(w) * pi / 180.0
        cos_phi = tf.cos(phi_radians)
        sin_phi = tf.sin(phi_radians)
        exp_term = tf.complex(cos_phi, sin_phi)
        return -1.0 * I * Z_total * exp_term
    
    # Calculate output
    out_y = V_out(w)
    real_part = tf.math.real(out_y)
    min_val = tf.reduce_min(real_part)
    offset = real_part - min_val
    
    return offset + DC_offset

def GenerateLineshape(P,x):
    
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
    
    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    Iplus = r*icurve(x,1)/10
    Iminus = icurve(x,-1)/10
    signal = Iplus + Iminus
    return signal,Iplus,Iminus

def GenerateLineshapeTensor(P, x):
    """
    Generate lineshape based on polarization and frequency range.
    Converted to TensorFlow operations.
    """
    # Ensure inputs are tensors
    P = tf.convert_to_tensor(P, dtype=tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Define constants
    g = tf.constant(0.15, dtype=tf.float32)  # Assuming g value - add your actual value
    s = tf.constant(0.0, dtype=tf.float32)   # Assuming s value - add your actual value
    bigy = tf.constant(1.0, dtype=tf.float32)  # Assuming bigy value - add your actual value
    
    # Define helper functions using TensorFlow operations
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
    
    # Calculate r 
    # Handle special case for P to avoid division by zero
    safe_P = tf.where(tf.abs(P) < 1e-10, tf.ones_like(P) * 1e-10, P)
    r = (tf.sqrt(4.0 - 3.0 * tf.pow(safe_P, 2)) + safe_P) / (2.0 - 2.0 * safe_P)
    
    # Expand dimensions for broadcasting
    r_expanded = tf.expand_dims(r, -1)
    
    # Calculate signal components
    Iplus = r_expanded * icurve(x, 1) 
    Iminus = icurve(x, -1) 
    signal = Iplus + Iminus
    
    return signal

def Baseline_Polynomial_Curve(w):
    return -1.84153246e-07*w**2 + 8.42855076e-05*w - 1.11342243e-04



