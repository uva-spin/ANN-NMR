import numpy as np
from scipy.special import wofz
from Variables import *

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

def GenerateLineshape(P):
    
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
    
    
    center = 250
    length = range(500)
    norm_array = []
    for x in length:
        norm_array = np.append(norm_array,(x - center)*(12/500))  
    Iplus = icurve(norm_array,1)
    Iminus = icurve(norm_array,-1)
    
    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    Iminus = icurve(norm_array,-1)
    array = r*Iminus
    array_flipped = np.flip(array)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = P
    signal = element_1*element_2*element_3
    result = signal
    return result