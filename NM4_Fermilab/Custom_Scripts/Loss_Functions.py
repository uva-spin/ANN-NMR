import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Custom_Scripts.Misc_Functions import *

def log_cosh_precision_loss(y_true, y_pred):
    error = y_true - y_pred
    precision_weights = tf.math.exp(-10.0 * y_true) + 1e-6  
    return tf.reduce_mean(precision_weights * tf.math.log(cosh(error)))

def cosh(x):
    return (tf.math.exp(x) + tf.math.exp(-x)) / 2

def balanced_precision_loss(y_true, y_pred):
    error = y_true - y_pred
    precision_weights = 1 / (tf.math.log1p(y_true + 1e-2) + 1.0) 
    return tf.reduce_mean(precision_weights * tf.math.log(tf.cosh(error)))

@tf.function(jit_compile=True)
def adaptive_weighted_huber_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = tf.abs(y_true - y_pred)

    small_value_penalty = tf.where((y_true < 0.01) & (y_pred > y_true), 20.0, 1.0)

    huber = tf.where(error < 1e-3, 0.5 * tf.square(error), 1e-3 * (error - 0.5 * 1e-3))

    return tf.reduce_mean(tf.cast(small_value_penalty, tf.float32) * huber)


def scaled_mse(y_true, y_pred):
    return tf.reduce_mean((100000 * (y_true - y_pred)) ** 2)  # Amplify small differences


def weighted_mse(y_true, y_pred):

    mse = tf.square(y_true - y_pred)
    

    center_weight = tf.exp(-200.0 * tf.square(y_true - 0.0005)) * 10.0
    
    small_value_weight = tf.exp(-5.0 * y_true) + 1.0
    
    weights = small_value_weight + center_weight
    
    weighted_loss = mse * weights
    
    return tf.reduce_mean(weighted_loss)


def Lineshape_Loss(y_true, y_pred):

    #### Constants #####
    U = tf.constant(2.4283, dtype=tf.float32)
    Cknob = tf.constant(0.1899, dtype=tf.float32)
    cable = tf.constant(6.0 / 2.0, dtype=tf.float32)
    eta = tf.constant(1.04e-2, dtype=tf.float32)
    phi = tf.constant(6.1319, dtype=tf.float32)
    Cstray = tf.constant(1e-20, dtype=tf.float32)
    shift = tf.constant(0.0, dtype=tf.float32)
    
    X = tf.linspace(-3.0, 3.0, 500)
    
    signal_true = GenerateLineshapeTensor(y_true, X) / 1500.0
    signal_pred = GenerateLineshapeTensor(y_pred, X) / 1500.0
    
    x, lower_bound, upper_bound = FrequencyBoundTensor(32.32)
    baseline = BaselineTensor(x, U, Cknob, eta, cable, Cstray, phi, shift)
    
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    
    combined_signal_true = signal_true + baseline
    combined_signal_pred = signal_pred + baseline
    
    combined_signal_true = tf.convert_to_tensor(combined_signal_true, dtype=tf.float32)
    combined_signal_pred = tf.convert_to_tensor(combined_signal_pred, dtype=tf.float32)
    
    return tf.reduce_mean(tf.square(combined_signal_true - combined_signal_pred))

def Polarization_Loss(y_true, y_pred):

    return tf.reduce_mean(tf.square(y_true - y_pred))

def Polarization_Lineshape_Loss(y_true, y_pred):
 
    return Polarization_Loss(y_true, y_pred) + 0.1 * Lineshape_Loss(y_true, y_pred)
