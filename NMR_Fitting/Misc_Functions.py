import numpy as np
import random
from scipy.stats import zscore

def choose_random_row(csv_file):
    df = csv_file
    if df.empty:
        return None  # If the DataFrame is empty
    random_index = np.random.randint(0, len(df))  # Generate a random index
    random_row = df.iloc[random_index]  # Get the row at the random index
    return random_row

def exclude_outliers(df, threshold=1.5):
    # Compute Z-scores for each row
    z_scores = df.apply(zscore, axis=0, result_type='broadcast')
    
    # Check if any Z-score exceeds the threshold
    is_outlier = (z_scores.abs() > threshold).any(axis=1)
    
    # Exclude outliers
    df_filtered = df[~is_outlier]
    # df_filtered = df[~is_outlier].apply(lambda x: x / 1000)
    
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