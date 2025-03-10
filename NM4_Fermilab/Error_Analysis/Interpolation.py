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

version = 'varied_polarization'

# Number of points in lineshape
samples = 100000

R = np.linspace(-3.5, 3.5, samples)

# Define P range to explore (30 steps around 0.0005 ± 0.0001)
P_center = 0.0005
P_range = 0.0001
num_steps = 10
# P_values = np.linspace(P_center - P_range, P_center + P_range, num_steps)
P_values = np.array([0.000543])

# Generate the reference lineshape at P = 0.0005
reference_P = 0.0005
reference_X, _, _ = GenerateLineshape(reference_P, R)
reference_X_log = np.log(reference_X)

# Function to create and compile model
def create_model(errF):
    model = keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(256, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(128, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(128, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(64, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(64, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(32, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)
    ])
    
    def Binned_MSE(y_true, y_pred, errF):
        errF = tf.convert_to_tensor(errF, dtype=tf.float32)
        squared_diff = tf.square(y_true - y_pred)
        loss = tf.reduce_mean(squared_diff / tf.square(errF))
        return loss

    def Binning(errF):
        def loss(y_true, y_pred):
            return Binned_MSE(y_true, y_pred, errF)
        return loss
        
    loss_function = Binning(errF)
    
    initial_learning_rate = 0.01
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=initial_learning_rate),
        loss=loss_function)
    
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

models_dir = os.path.join(current_dir, 'varied_p_models')
os.makedirs(models_dir, exist_ok=True)

for i, p_value in enumerate(tqdm(P_values, desc="Training models")):
    print(f"\nTraining model {i+1}/{num_steps} with P = {p_value:.7f}")
    
    X, _, _ = GenerateLineshape(p_value, R)
    X_log = np.log(X)
    
    X_log_reshaped = X_log
    
    
    _, errF, _ = calculate_binned_errors(X_log_reshaped, num_bins=10000)
    
    
    model = create_model(errF)
    
    lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        min_delta=1e-8,
        mode='max',
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
        batch_size=128,
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
    
    pred_df = pd.DataFrame({
        'R': x_new,
        'True_Lineshape': true_X,
        'Predicted_Lineshape': predictions.flatten()
    })
    
    csv_path = os.path.join(current_dir, f'predictions_p_{p_value:.7f}.csv')
    pred_df.to_csv(csv_path, index=False)
    
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
plt.show()

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