import numpy as np
import random
from scipy.stats import zscore
import os
import sys
import glob as glob
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.layers import Layer
from Lineshape import *
import sys
from io import StringIO

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
    """
    Search for a file by name in the starting directory and up to 2 parent directories.
    
    Args:
        filename (str): Name of the file to find
        start_dir (str): Directory to start the search from (default: current directory)
        
    Returns:
        str or None: Full path to the first matching file, or None if not found
    """
    # Start from absolute path of the starting directory
    current_dir = os.path.abspath(start_dir)
    
    # Search up to 2 levels up from the starting directory
    for _ in range(3):  # 0, 1, 2 levels up
        # Use glob to find the file in current directory and all subdirectories
        matches = glob.glob(os.path.join(current_dir, '**', filename), recursive=True)
        
        # Return the first match if found
        if matches:
            return matches[0]
            
        # Move up to parent directory
        parent_dir = os.path.dirname(current_dir)
        
        # Stop if we've reached the root directory
        if parent_dir == current_dir:
            break
            
        current_dir = parent_dir
    
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


import tensorflow as tf

class LookaheadWrapper:
    def __init__(self, optimizer, sync_period=5, alpha=0.5):
        """
        Implements Lookahead by wrapping a base optimizer.

        Parameters:
        - optimizer: The base optimizer (e.g., AdamW, Adamax, RMSprop)
        - sync_period: Number of steps before slow weights update
        - alpha: Interpolation factor (0.0 = no effect, 1.0 = full Lookahead)
        """
        self.optimizer = optimizer
        self.sync_period = sync_period
        self.alpha = alpha
        self.step_counter = 0
        self.slow_weights = None

    def apply_gradients(self, grads_and_vars):
        """Apply gradients using the base optimizer and perform Lookahead updates."""
        self.optimizer.apply_gradients(grads_and_vars)
        self.step_counter += 1

        if self.step_counter % self.sync_period == 0:
            if self.slow_weights is None:
                self.slow_weights = [tf.Variable(var.read_value(), trainable=False) for _, var in grads_and_vars]

            # Perform Lookahead update: slow_weight = slow_weight + α * (fast_weight - slow_weight)
            for slow_var, (_, fast_var) in zip(self.slow_weights, grads_and_vars):
                slow_var.assign(slow_var + self.alpha * (fast_var - slow_var))
                fast_var.assign(slow_var)

    def get_config(self):
        """Return config for serialization (optional)."""
        return {
            "sync_period": self.sync_period,
            "alpha": self.alpha,
            "optimizer": self.optimizer.get_config()
        }




# 🔹 Custom TensorBoard Callback to Log Weights & Gradients
class CustomTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='./logs', validation_data=None):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.validation_data = validation_data  # Store validation data for computing metrics

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            # 🔹 Log Weight Histograms
            for weight in self.model.trainable_weights:
                tf.summary.histogram(weight.name, weight.numpy(), step=epoch)

            # 🔹 Compute & Log Gradients using `GradientTape`
            if self.validation_data is not None:
                X_val, y_val = self.validation_data  # Unpack validation data
                with tf.GradientTape() as tape:
                    predictions = self.model(X_val, training=True)
                    loss_value = self.model.loss(y_val, predictions)  # Use validation data
                gradients = tape.gradient(loss_value, self.model.trainable_weights)

                for grad, weight in zip(gradients, self.model.trainable_weights):
                    if grad is not None:
                        tf.summary.histogram(f'gradient/{weight.name}', grad.numpy(), step=epoch)

            # 🔹 Log Learning Rate
            tf.summary.scalar("learning_rate", self.model.optimizer.learning_rate.numpy(), step=epoch)

        self.writer.flush()


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.epoch_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        training_loss = logs.get('loss', None)
        validation_loss = logs.get('val_loss', None)
        loss_diff = None
        if training_loss is not None and validation_loss is not None:
            loss_diff = training_loss - validation_loss
        self.epoch_data.append({
            'Epoch': epoch + 1,
            'Learning Rate': lr,
            'Training Loss': training_loss,
            'Validation Loss': validation_loss,
            'Loss Difference': loss_diff
        })

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.epoch_data)
        df.to_csv(self.log_path, index=False)
        print(f"Custom metrics log saved to {self.log_path}")


# Compute Differences Between Consecutive Voltage Values (X data)
def compute_differences(X):
    # Compute the differences between consecutive points
    diffs = np.diff(X, axis=1)
    
    # Compute the error bars (absolute differences between consecutive points)
    error_bars = np.hstack([np.zeros((X.shape[0], 1)), np.abs(diffs)])
    
    # Set the first point to 0 for the differences
    differences = np.hstack([np.zeros((X.shape[0], 1)), diffs])
    
    return differences, error_bars


def Binning_Errors(y_true, y_pred, feature_space, bins=500, bin_range=(-3, 3)):
    """
    Custom loss function that divides each feature space neuron by its respective binning error.
    
    Parameters:
    -----------
    y_true : tensor
        True values (ground truth).
    y_pred : tensor
        Predicted values.
    feature_space : tensor
        The feature space (input features).
    bins : int
        Number of bins to create.
    bin_range : tuple
        Range of values to consider for binning (min, max).
    
    Returns:
    --------
    tensor
        Computed loss value.
    """
    # Initialize binned errors
    
    n = 10000  
    num_bins = 500  
    data_min = -3  
    data_max = 3 
    
    x_values = np.linspace(data_min, data_max, n) 
    bin_edges = np.linspace(data_min, data_max, num_bins + 1)  # Bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers for plotting
    
    y_true = y_true.numpy()

    binned_errors = np.zeros((len(y_true), num_bins))
    
    # Loop through each true value to generate the corresponding signal
    for i in range(len(y_true)):
        P = y_true[i]  # Get the true value for this sample
        # x_values = feature_space.numpy()  # Get the corresponding feature space for this sample
        
        # Generate the signal using the GenerateLineshape function
        signal, _, _ = GenerateLineshape(P, x_values)
        
        # Create bins
        bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1)
        
        # Digitize the generated signal into bins
        bin_indices = np.digitize(signal, bin_edges) - 1  # -1 to make it zero-indexed
        bin_indices = np.clip(bin_indices, 0, bins - 1)  # Clip to valid indices
        
        # Calculate the bin counts (standard deviation as error)
        for j in range(bins):
            mask = (bin_indices == j)
            if np.any(mask):
                binned_errors[i, j] = np.std(signal[mask])  # Standard deviation as error

    # Normalize the errors
    min_error = np.min(binned_errors)
    max_error = np.max(binned_errors)
    normalized_errors = (binned_errors - min_error) / (max_error - min_error + 1e-8)  # Avoid division by zero

    # Calculate the loss by dividing each predicted value by its corresponding binning error
    loss = tf.reduce_mean(tf.square(y_pred / normalized_errors))

    return loss


# def calculate_binned_errors(signal, num_bins=500, data_min=-3, data_max=3):
#     """
#     Calculate binned errors for a given signal.

#     Parameters:
#     - signal: The signal to calculate the binned errors for. Should be a 1D array.
#     - num_bins: Number of bins for binning the data.
#     - data_min: Minimum value for x_values.
#     - data_max: Maximum value for x_values.

#     Returns:
#     - bin_centers: The centers of the bins.
#     - binned_errors: The standard deviation of the signal in each bin.
#     - bin_indices: The indices of the bins.
#     """
#     signal = np.asarray(signal).flatten()

#     samples = len(signal)
#     x_values = np.linspace(data_min, data_max, samples)

#     # Bin edges
#     bin_edges = np.linspace(data_min, data_max, num_bins + 1)  # Bin edges
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

#     # Bin the data
#     bin_indices = np.digitize(x_values, bin_edges) - 1
#     bin_indices = np.clip(bin_indices, 0, num_bins - 1)  

#     binned_errors = np.zeros(num_bins)

#     # Calculate the standard deviation for each bin
#     for i in range(num_bins):
#         mask = (bin_indices == i)
#         if np.any(mask):
#             # Calculate the standard deviation only if there are valid samples in the bin
#             binned_errors[i] = np.std(signal[mask]) if np.any(signal[mask]) else 0.0

#     return bin_centers, binned_errors, bin_indices 

def calculate_binned_errors(signal, num_bins=500, data_min=-3, data_max=3):
    """
    Calculate binned errors for a given signal.

    Parameters:
    - signal: The signal to calculate the binned errors for. Should be a 1D array.
    - num_bins: Number of bins for binning the data.
    - data_min: Minimum value for x_values.
    - data_max: Maximum value for x_values.

    Returns:
    - bin_centers: The centers of the bins.
    - binned_errors: The standard deviation of the signal in each bin.
    - bin_indices: The indices of the bins.
    """
    signal = np.asarray(signal).flatten()

    samples = len(signal)
    x_values = np.linspace(data_min, data_max, samples)

    # Bin edges
    bin_edges = np.linspace(data_min, data_max, num_bins + 1)  # Bin edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    # Bin the data
    bin_indices = np.digitize(x_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  

    binned_errors = np.zeros(num_bins)

    # Calculate the standard deviation for each bin
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            # Calculate the standard deviation only if there are valid samples in the bin
            binned_errors[i] = np.std(signal[mask]) if np.any(signal[mask]) else 0.0
            
            # Multiply binned errors by 1.5 for bins corresponding to x < -1.5 or x > 1.5
            if bin_centers[i] < -1.5 or bin_centers[i] > 1.5:
                binned_errors[i] *= 1000

    return bin_centers, binned_errors, bin_indices 

def relative_percent_error(y_true, y_pred):
    """
    Custom loss function: Relative Percent Error (RPE).
    
    Args:
        y_true: True values (ground truth).
        y_pred: Predicted values.
    
    Returns:
        RPE loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    non_zero_mask = tf.not_equal(y_true, 0)
    rpe = tf.where(non_zero_mask, 
                   tf.abs((y_true - y_pred) / y_true) * 100.0, 
                   tf.zeros_like(y_true)) 
    
    # Return the mean RPE over the batch
    return tf.reduce_mean(rpe)

def save_model_summary(model, performance_dir, version):
    # Create the Models directory if it doesn't exist
    models_dir = os.path.join(performance_dir, 'Models')
    os.makedirs(models_dir, exist_ok=True)

    # Define the filename based on the version
    filename = os.path.join(models_dir, f'model_summary_{version}.txt')

    # Redirect stdout to capture the model summary
    string_io = StringIO()
    sys.stdout = string_io
    model.summary()
    sys.stdout = sys.__stdout__  # Reset redirect

    # Write the model summary to the file
    with open(filename, 'w') as f:
        f.write(string_io.getvalue())

    print(f"Model summary saved to {filename}")

