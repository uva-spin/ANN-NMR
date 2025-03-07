import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Custom_Scripts.Misc_Functions import *

def log_cosh_precision_loss(y_true, y_pred):
    """Hybrid loss combining log-cosh and precision weighting"""
    error = y_true - y_pred
    precision_weights = tf.math.exp(-10.0 * y_true) + 1e-6  
    return tf.reduce_mean(precision_weights * tf.math.log(cosh(error)))

def cosh(x):
    return (tf.math.exp(x) + tf.math.exp(-x)) / 2

def balanced_precision_loss(y_true, y_pred):
    """Custom loss that ensures equal precision across the entire range."""
    error = y_true - y_pred
    precision_weights = 1 / (tf.math.log1p(y_true + 1e-2) + 1.0) 
    return tf.reduce_mean(precision_weights * tf.math.log(tf.cosh(error)))

@tf.function(jit_compile=True)
def adaptive_weighted_huber_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = tf.abs(y_true - y_pred)

    # **Penalty for underestimating small values**
    small_value_penalty = tf.where((y_true < 0.01) & (y_pred > y_true), 20.0, 1.0)

    # **Huber Loss with Adaptive Weights**
    huber = tf.where(error < 1e-3, 0.5 * tf.square(error), 1e-3 * (error - 0.5 * 1e-3))

    # Ensure output remains float32
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
    X = np.linspace(-3, 3, 500)
    Lineshape_True = GenerateLineshape(y_true, X)
    Lineshape_Pred = GenerateLineshape(y_pred, X)
    
    return tf.reduce_mean(tf.square(Lineshape_True - Lineshape_Pred))

def Polarization_Loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def Custom_Polarization_Loss(y_true, y_pred):
    return Polarization_Loss(y_true, y_pred) + Lineshape_Loss(y_true, y_pred)
