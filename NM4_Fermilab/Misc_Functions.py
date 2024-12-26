import numpy as np
import random
from scipy.stats import zscore
import os
import glob as glob
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats import norm

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

def find_directory(directory_name, start_dir='.'):
    current_dir = os.path.abspath(start_dir)
    levels_up = 0
    
    while levels_up <= 2:  
        print(f"Searching for '{directory_name}' in {current_dir}...")  
        for root, dirs, _ in os.walk(current_dir):
            if directory_name in dirs:
                print(f"Found '{directory_name}' in {root}")  
                return os.path.join(root, directory_name)
        
        
        current_dir = os.path.abspath(os.path.join(current_dir, '..'))
        levels_up += 1
    
    print(f"Could not find '{directory_name}' within {levels_up} levels up from {start_dir}")  
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

def plot_histogram(data, title, xlabel, ylabel, color, ax, num_bins=100, plot_norm=True):
    n, bins, patches = plt.hist(data, num_bins, density=True, color=color, alpha=0.7)
    mu, sigma = norm.fit(data)

    if plot_norm:
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, '--', color='black')

    plt.title(f"{title}: μ={mu:.4f}, σ={sigma:.4f}", fontsize = 16,weight='bold')
    plt.xlabel(xlabel, fontsize = 16,weight='bold')
    plt.ylabel(ylabel, fontsize = 16,weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)  
    ax.tick_params(axis='both', which='minor', labelsize=12)  
    plt.grid(False)
    # plt.savefig(save_path)
    # plt.close()


def data_generator(file_path, chunk_size=10000, batch_size=1024):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        target_variable = "P"
        X = chunk.drop([target_variable, 'SNR'], axis=1).values  
        y = chunk[target_variable].values
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for batch in dataset:
            yield batch

def split_data_in_batches(data_generator, val_fraction=0.1):
    for X_batch, y_batch in data_generator:
        split_index = int(X_batch.shape[0] * (1 - val_fraction))
        X_train_batch, X_val_batch = X_batch[:split_index], X_batch[split_index:]
        y_train_batch, y_val_batch = y_batch[:split_index], y_batch[split_index:]
        yield (X_train_batch, y_train_batch), (X_val_batch, y_val_batch)

def test_data_generator(file_path, chunk_size=10000, test_fraction=0.1):
    test_data = pd.read_csv(file_path, chunksize=chunk_size)
    test_df = pd.concat([chunk for chunk in test_data]).sample(frac=test_fraction)
    target_variable = "P"
    X_test = test_df.drop([target_variable, 'SNR'], axis=1).values
    y_test = test_df[target_variable].values
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1024).prefetch(tf.data.experimental.AUTOTUNE)
    return test_dataset


