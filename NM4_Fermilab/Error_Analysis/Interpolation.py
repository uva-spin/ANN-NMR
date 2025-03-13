import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as skm
from tensorflow.keras import regularizers
import random
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import glob  # Import glob to find files
import datetime
# Add path to custom scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))

from Custom_Scripts.Lineshape import GenerateLineshape
from Custom_Scripts.Misc_Functions import calculate_binned_errors

# Set seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

version = 'varied_polarization_PI_Loss'

# Number of points in lineshape
samples = 100000

R = np.linspace(-3.5, 3.5, samples)

# Define P range to explore (30 steps around 0.0005 ± 0.0001)
P_center = 0.0005
P_range = 0.0001
num_steps = 10
P_values = np.linspace(P_center - P_range, P_center + P_range, num_steps)

# Generate the reference lineshape at P = 0.0005
reference_P = 0.0005
reference_X, _, _ = GenerateLineshape(reference_P, R)
reference_X_log = np.log(reference_X)

def create_model(errF, input_shape=(1,)):

    inputs = keras.Input(shape=input_shape)
    
    x = layers.Dense(256, activation=None, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    
    for units in [128, 64, 32]:

        y = layers.Dense(units, activation=None, kernel_regularizer=regularizers.l2(0.001))(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('swish')(y)
        y = layers.Dense(units, activation=None, kernel_regularizer=regularizers.l2(0.001))(y)
        y = layers.BatchNormalization()(y)
        
        if x.shape[-1] != units:
            x = layers.Dense(units, kernel_regularizer=regularizers.l2(0.001))(x)
        
        # Add skip connection
        x = layers.Add()([x, y])
        x = layers.Activation('swish')(x)
        x = layers.Dropout(0.1)(x)
    
    # Final output layer
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    def Loss(y_true, y_pred, errF):
        errF = tf.convert_to_tensor(errF, dtype=tf.float32)
        squared_diff = tf.square(y_true - y_pred)
        
        # # Basic binned MSE
        loss = tf.reduce_mean(squared_diff / tf.square(errF))
        
        return loss 
    
    def lineshape_accuracy(y_true, y_pred):


        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        y_true_exp = tf.exp(y_true_flat)
        y_pred_exp = tf.exp(y_pred_flat)
        
        # Normalize both curves for shape comparison
        y_true_sum = tf.reduce_sum(y_true_exp) + tf.keras.backend.epsilon()
        y_pred_sum = tf.reduce_sum(y_pred_exp) + tf.keras.backend.epsilon()
        
        y_true_norm = y_true_exp / y_true_sum
        y_pred_norm = y_pred_exp / y_pred_sum
        
        similarity = 1.0 - tf.reduce_mean(tf.abs(y_true_norm - y_pred_norm))
        return similarity
        
    loss_function = Loss(errF)
    
    initial_learning_rate = 0.01
    model.compile(
        optimizer=keras.optimizers.AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=1e-5),
        loss=loss_function,
        metrics=[lineshape_accuracy])
    
    return model

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

all_predictions = []
all_models = []
all_true_lineshapes = []
predicted_lineshapes = []

models_dir = os.path.join(current_dir, 'varied_p_models')
os.makedirs(models_dir, exist_ok=True)

for i, p_value in enumerate(tqdm(P_values, desc="Training models")):
    print(f"\nTraining model {i+1}/{num_steps} with P = {p_value:.7f}")
    
    X, _, _ = GenerateLineshape(p_value, R)
    X_log = np.log(X) + np.random.normal(0, 0.1, size=X.shape)
    
    X_log_reshaped = X_log
    
    
    _, errF, _ = calculate_binned_errors(X_log_reshaped, num_bins=10000)
    
    
    model = create_model(errF)
    
    lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        min_delta=1e-8,
        mode='min',
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    R_reshaped = R.reshape(-1, 1)
    
    # Train the model
    history = model.fit(
        R_reshaped, X_log_reshaped,
        epochs=100,  
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler, reduce_lr],
        verbose=1
    )
    
    # Save the model
    model_path = os.path.join(models_dir, f'model_p_{p_value:.7f}.keras')
    model.save(model_path)
    all_models.append(model)
    
    x_new = np.linspace(-3, 3, 10000)  
    x_new_reshaped = x_new.reshape(-1, 1)
    predictions_log = model.predict(x_new_reshaped, verbose=0)
    predictions = np.exp(predictions_log)
    
    all_predictions.append(predictions.flatten())
    
    true_X, _, _ = GenerateLineshape(p_value, x_new)
    all_true_lineshapes.append(true_X)
    
    
df = pd.DataFrame(all_predictions)
df["P_true"] = P_values
csv_path = os.path.join(current_dir, f'predictions_p_{p_value:.7f}.csv')
df.to_csv(csv_path, index=False)

print(f"Saved predictions for P = {p_value:.7f} to {csv_path}")
    
# Now create a comprehensive plot comparing all models
plt.figure(figsize=(15, 10))

colors = cm.rainbow(np.linspace(0, 1, num_steps))

for i, (predictions, p_value, true_lineshape) in enumerate(zip(all_predictions, P_values, all_true_lineshapes)):
    x_new = np.linspace(-3, 3, len(predictions))
    
    plt.plot(x_new, predictions, color=colors[i], alpha=0.7, 
             label=f'Predicted (P = {p_value:.7f})' if i % 5 == 0 else "")
    
    if i % 5 == 0:  
        plt.plot(x_new, true_lineshape, '--', color=colors[i], alpha=0.5, 
                 label=f'True (P = {p_value:.7f})')

plt.xlabel('R', fontsize=14)
plt.ylabel('Intensity', fontsize=14)
plt.title('Lineshape Interpolation for Various P Values', fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(current_dir, 'polarization_variation_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.show()

print(f"Comparison plot saved to {plot_path}")

summary_data = {
    'P_value': P_values,
    'Model_path': [os.path.join(models_dir, f'model_p_{p:.7f}.keras') for p in P_values],
    'Predictions_path': [os.path.join(current_dir, f'predictions_p_{p:.7f}.csv') for p in P_values]
}

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(current_dir, 'polarization_variation_summary.csv')
summary_df.to_csv(summary_path, index=False)

print(f"Summary information saved to {summary_path}")
print(f"Completed analysis of {num_steps} P values between {P_values[0]:.7f} and {P_values[-1]:.7f}")

df = pd.read_csv(os.path.join(current_dir, 'Lineshape_Predictions.csv'))

predictions = df.drop(columns=["P_true"]).values
P_values = df["P_true"].values

def Baseline(x, P):
    Sig, _, _ = GenerateLineshape(P, x)
    return Sig

X = np.linspace(-3, 3, 10000)

initial_params = [0.0004183, 0.000432, 0.00046, 0.000532, 0.0005443, 0.000556, 0.000568, 0.00058, 0.00062, 0.00062]

lower_bounds = [p - 0.00005 for p in initial_params]
upper_bounds = [p + 0.00005 for p in initial_params]
param_bounds = (lower_bounds, upper_bounds)

covs = []
popts = []
residuals = []  # Initialize a list to store residuals

for i, (lower_bound, upper_bound) in enumerate(zip(lower_bounds, upper_bounds)):
    popt, pcov = curve_fit(Baseline, X, predictions[i], p0=initial_params[i], bounds=(lower_bound, upper_bound))
    covs.append(pcov)
    popts.append(popt)
    
    # Calculate residuals
    P_true = P_values[i]*100
    popt = popt*100
    residual = P_true - popt  # Calculate the residual
    residuals.append(residual)  # Store the residual
    

# Calculate statistics for residuals
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
residuals_array = np.array(residuals).flatten()
# Plot histogram of residuals with Gaussian fit
plt.figure(figsize=(10, 6))

# Create histogram
n, bins, patches = plt.hist(residuals_array, bins=min(10, num_steps), 
                            color='skyblue', edgecolor='black', 
                            alpha=0.7, density=True)

# Fit a Gaussian curve to the histogram
def gaussian(x, mean, std, amplitude):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

# Create x values for the Gaussian curve
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

# Initial guess for the fit
p0 = [residual_mean, residual_std, 1.0/(residual_std * np.sqrt(2 * np.pi))]

try:
    # Try to fit a Gaussian to the histogram data
    params, covs_gauss = curve_fit(gaussian, bin_centers, n, p0=p0)
    fitted_mean, fitted_std, amplitude = params
    
    # Create smooth curve for plotting
    x_smooth = np.linspace(min(bins), max(bins), 100)
    y_smooth = gaussian(x_smooth, fitted_mean, fitted_std, amplitude)
    
    # Plot the fitted Gaussian curve
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                label=f'Gaussian Fit\nMean = {fitted_mean:.5f}\nStd Dev = {fitted_std:.5f}')
    
    # Store the fitted parameters
    gauss_fit_params = {
        'mean': fitted_mean,
        'std_dev': fitted_std,
        'amplitude': amplitude
    }
except:
    # If fitting fails, use the calculated statistics
    plt.axvline(residual_mean, color='r', linestyle='--', 
                label=f'Mean = {residual_mean:.5f}')
    plt.axvline(residual_mean + residual_std, color='g', linestyle=':', 
                label=f'±Std Dev = {residual_std:.5f}')
    plt.axvline(residual_mean - residual_std, color='g', linestyle=':')
    
    gauss_fit_params = {
        'mean': residual_mean,
        'std_dev': residual_std,
        'amplitude': None,
        'note': 'Gaussian fitting failed, using sample statistics'
    }

plt.xlabel('Residual (P_true - P_optimized) × 100')
plt.ylabel('Probability Density')
plt.title('Histogram of Fitting Residuals with Gaussian Fit')
plt.grid(alpha=0.3)
plt.legend()

# Save histogram with Gaussian fit
histogram_path = os.path.join(current_dir, "residuals_histogram_NN_with_gaussian.png")
plt.savefig(histogram_path)
plt.close()

# Save results to a file
def save_results_to_file(P_true=None, filename="optimization_results_NN.txt"):
    with open(filename, 'w') as f:
        # Write header
        f.write("="*50 + "\n")
        f.write("OPTIMIZATION RESULTS\n")
        f.write("="*50 + "\n\n")
        
        # Write optimized parameters
        f.write("OPTIMIZED PARAMETERS:\n")
        f.write("-"*50 + "\n")
        for i, (P_true, param_value) in enumerate(zip(P_values, popts)):
            # Handle both single value and array cases
            if hasattr(param_value, '__iter__') and not isinstance(param_value, str):
                formatted_value = f"{param_value[0]:.8f}"
            else:
                formatted_value = f"{param_value:.8f}"
            
            f.write(f"Parameter {i+1}: {P_true} | {formatted_value}\n")
        f.write("-"*50 + "\n\n")
        
        # Write residuals if P_true is provided
        if P_true is not None:
            f.write("RESIDUALS (P_true - P_optimized):\n")
            f.write("-"*50 + "\n")
            for i, (P_true, opt_val) in enumerate(zip(P_values, popts)):
                # Handle both single value and array cases for optimized values
                if hasattr(opt_val, '__iter__') and not isinstance(opt_val, str):
                    opt_val = opt_val[0]
                
                # Calculate residual
                residual = P_true - opt_val
                percent_error = (residual / P_true) * 100 if P_true != 0 else float('inf')
                
                f.write(f"Parameter {i+1}: {P_true}\n")
                f.write(f"    True value:      {P_true:.8f}\n")
                f.write(f"    Optimized value: {opt_val:.8f}\n")
                f.write(f"    Residual:        {residual:.8f}\n")
                f.write(f"    Percent error:   {percent_error:.4f}%\n\n")
            f.write("-"*50 + "\n\n")
        
        # Write covariance matrices
        f.write("COVARIANCE MATRICES:\n")
        f.write("-"*50 + "\n")
        for i, cov in enumerate(covs):
            f.write(f"\nMatrix {i+1} (Parameter: {P_values[i]}):\n")
            # Check the type of cov and handle accordingly
            if np.isscalar(cov):
                # If it's a scalar value
                f.write(f"    {cov:.8e}\n")
            elif isinstance(cov, (list, np.ndarray)):
                # If it's an array or matrix
                if hasattr(cov, 'shape') and len(cov.shape) == 2:
                    # For 2D matrices
                    for row in cov:
                        formatted_row = "    " + " ".join([f"{val:.4e}" for val in row])
                        f.write(formatted_row + "\n")
                else:
                    # For 1D arrays or other structures
                    f.write(f"    {cov}\n")
            else:
                # Fallback for any other type
                f.write(f"    {cov}\n")
        f.write("-"*50 + "\n")
        
        f.write("\n\nFile generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"Results saved to '{filename}'")


save_results_to_file(P_true = P_values, filename=os.path.join(current_dir, "NN_Optimization_Results.txt"))