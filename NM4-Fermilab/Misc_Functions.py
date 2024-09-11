import numpy as np
import random
from scipy.stats import zscore
import os
import glob as glob
import scipy.integrate as spi

def choose_random_row(csv_file):
    df = csv_file
    if df.empty:
        return None 
    random_index = np.random.randint(0, len(df))  
    random_row = df.iloc[random_index]  
    return random_row

def exclude_outliers(df, threshold=1.5):

    z_scores = df.apply(zscore, axis=0, result_type='broadcast')
    
    is_outlier = (z_scores.abs() > threshold).any(axis=1)
    
    df_filtered = df[~is_outlier]
    # df_filtered = df[~is_outlier].apply(lambda x: x / 1000)
    
    return df_filtered

def Baseline_Polynomial_Curve(w):
    return -1.84153246e-07*w**2 + 8.42855076e-05*w - 1.11342243e-04

def random_sign():
    return random.choice([-1, 1])

def Sinusoidal_Noise(shape):

    angles = np.random.uniform(0, 2*np.pi, shape)
    
    cos_values = np.random.uniform(-0.0005,0.0005)*np.cos(angles)
    sin_values = np.random.uniform(-0.0005,0.0005)*np.sin(angles)
    
    result = cos_values + sin_values
    
    return result

def apply_distortion(signal, alpha):
    distorted_signal = signal + alpha * np.power(signal, 3)
    return distorted_signal

def find_file(filename, start_dir='.'):
    current_dir = os.path.abspath(start_dir)
    levels_up = 0
    
    while levels_up <= 2:  

        for file in glob.glob(os.path.join(current_dir, '**', filename), recursive=True):
            return file
        
  
        current_dir = os.path.abspath(os.path.join(current_dir, '..'))
        levels_up += 1
    
    return None


def print_cov_matrix_with_param_names(matrix, param_names):

    print("Covariance Matrix (with parameter names):")
    print(f"{'':>16}  " + "  ".join(f"{name:>16}" for name in param_names))
    
    for i, row in enumerate(matrix):
        row_label = param_names[i]
        formatted_row = "  ".join(f"{value:16.5e}" for value in row)
        print(f"{row_label:>16}  {formatted_row}")

def print_optimized_params_with_names(params, param_names):
    print("Optimized Parameters:")
    for name, param in zip(param_names, params):
        print(f"{name:>16}: {param:16.5e}")

