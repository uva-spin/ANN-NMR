import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers, Model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import matplotlib.pyplot as plt
import random
from datetime import datetime
from Custom_Scripts.Misc_Functions import *
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

### Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)

set_seeds(42)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)

# Mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(physical_devices)} GPU(s): {physical_devices}")
else:
    print("No GPU found, using CPU")

tf.keras.backend.set_floatx('float32')

# File paths and versioning
data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  # Update with your path function
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
version = f'Deuteron_Advanced_ResNet_V1_{timestamp}'
performance_dir = f"Model_Performance/{version}"
model_dir = f"Models/{version}"
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Advanced loss functions
def huber_precision_loss(y_true, y_pred, delta=0.1):
    """Huber loss that emphasizes precision"""
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    # Calculate the relative error for small true values
    rel_error = tf.where(
        tf.abs(y_true) > 1e-6,
        error / tf.abs(y_true),
        error
    )
    
    # Weight based on the magnitude of y_true (smaller values get higher weights)
    weights = 1.0 / (tf.abs(y_true) + 0.1)
    
    # Combine absolute and relative error with weighting
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear) + 0.2 * tf.reduce_mean(tf.abs(rel_error) * weights)

def combined_precision_loss(y_true, y_pred):
    """Combination of MSE, MAE, and relative error for high precision"""
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Safe relative percentage error
    safe_y_true = y_true + tf.where(tf.abs(y_true) < epsilon, epsilon, 0.0)
    rpe = tf.abs((y_true - y_pred) / safe_y_true)
    
    # Weight smaller values
    weights = tf.exp(-tf.abs(y_true) * 5.0) + 0.5
    weighted_rpe = tf.reduce_mean(rpe * weights)
    
    return 0.2 * mse + 0.3 * mae + 0.5 * weighted_rpe
    
    # Final loss combines all three with emphasis on relative precision
    return 0.2 * mse + 0.3 * mae + 0.5 * weighted_rpe

def attention_block(x, filters):
    """Self-attention mechanism using TensorFlow's MultiHeadAttention"""
    # Add sequence dimension with Lambda layer
    x_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
    
    # Create MultiHeadAttention with proper key_dim
    attention_layer = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=filters//4,  # Ensure key_dim is at least 1
        attention_axes=(1,)  # Attend along sequence dimension
    )
    
    # Apply attention
    attention_output = attention_layer(query=x_expanded, value=x_expanded)
    
    # Remove sequence dimension
    attention_output = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(attention_output)
    
    # Add residual connection with scaling
    output = layers.Add()([x, 0.1 * attention_output])  # Scale attention output
    return output

def residual_block_enhanced(x, units, dropout_rate=0.1):
    """Enhanced residual block with normalization and regularization"""
    shortcut = x
    
    # First dense layer with batch normalization
    x = layers.Dense(units, 
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second dense layer 
    x = layers.Dense(units, 
                     kernel_initializer=initializers.HeNormal(),
                     kernel_regularizer=regularizers.l2(1e-5),
                     kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut dimension if needed
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, kernel_initializer=initializers.HeNormal())(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Combine with shortcut
    x = layers.Add()([x, shortcut])
    # x = layers.Activation('swish')(x)
    return x

def feature_extraction_block(x, filters):
    """Extract hierarchical features using convolutions for the frequency domain"""
    # Reshape for 1D convolutions (treating input as frequency domain)
    x_reshaped = layers.Reshape((-1, 1))(x)  # Add a new axis at the end
    
    # Multi-scale feature extraction
    conv1 = layers.Conv1D(filters, kernel_size=3, padding='same', activation='swish')(x_reshaped)
    conv3 = layers.Conv1D(filters, kernel_size=7, padding='same', activation='swish')(x_reshaped)
    conv5 = layers.Conv1D(filters, kernel_size=11, padding='same', activation='swish')(x_reshaped)
    
    # Concatenate different scales
    x_concat = layers.Concatenate()([conv1, conv3, conv5])
    
    # Global pooling to create fixed-size feature vector
    x_pooled = layers.GlobalAveragePooling1D()(x_concat)
    
    return x_pooled

def build_advanced_polarization_model(input_shape=(500,), depth=6):
    """Advanced model architecture with residual connections, attention, and multi-path design"""
    inputs = layers.Input(shape=input_shape, dtype='float32', name='spectrum_input')
    
    # Initial normalization
    x = layers.LayerNormalization()(inputs)
    
    # Feature extraction path using convolutional approach
    conv_features = feature_extraction_block(inputs, 64)
    
    # Deep residual path
    units = [512, 256, 256, 128, 128, 64]
    
    # First dense block
    x = layers.Dense(units[0], activation='swish',
                    kernel_initializer=initializers.HeNormal(),
                    kernel_regularizer=regularizers.l2(1e-5),
                    kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    x = layers.BatchNormalization()(x)
    
    # Residual blocks with progressive reduction
    for i in range(min(depth, len(units))):
        x = residual_block_enhanced(x, units[i], dropout_rate=0.1)
        
        # Add attention every 2 blocks
        if i % 2 == 1:
            x = attention_block(x, units[i])
    
    # Merge conv features with dense features
    x = layers.Concatenate()([x, conv_features])
    
    # Final layers for regression with progressive narrowing
    x = layers.Dense(64, activation='swish',
                    kernel_initializer=initializers.GlorotNormal(),
                    kernel_regularizer=regularizers.l2(1e-4),
                    kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(32, activation='swish',
                    kernel_initializer=initializers.GlorotNormal(),
                    kernel_regularizer=regularizers.l2(1e-4),
                    kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1, activation='linear',
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4),
                          kernel_constraint=tf.keras.constraints.MaxNorm(3.0))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0, 
    )
    
    model.compile(
        optimizer=optimizer,
        loss=combined_precision_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    return model

# Advanced data preprocessing
print("Loading data...")
data = pd.read_csv(data_path)

# Data splitting - use stratified split for better distribution
# To stratify polarization values, create bins
data['P_bin'] = pd.qcut(data['P'], q=10, labels=False)  # 10 quantile bins
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['P_bin'])
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=temp_data['P_bin'])

# Prepare features and targets
X_train = train_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_train = train_data["P"].astype('float32').values
X_val = val_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_val = val_data["P"].astype('float32').values
X_test = test_data.drop(columns=["P", 'SNR', 'P_bin']).astype('float32').values
y_test = test_data["P"].astype('float32').values

# Enhanced normalization pipeline
def normalize_data(X_train, X_val, X_test):
    """Advanced normalization pipeline with robust scaling and power transform"""
    # Remove outliers from training data for better scaling
    scaler1 = RobustScaler().fit(X_train)
    X_train_robust = scaler1.transform(X_train)
    X_val_robust = scaler1.transform(X_val)
    X_test_robust = scaler1.transform(X_test)
    
    # Apply power transform for approximate normality
    scaler2 = PowerTransformer(method='yeo-johnson').fit(X_train_robust)
    X_train_final = scaler2.transform(X_train_robust).astype('float32')
    X_val_final = scaler2.transform(X_val_robust).astype('float32')
    X_test_final = scaler2.transform(X_test_robust).astype('float32')
    
    return X_train_final, X_val_final, X_test_final, (scaler1, scaler2)

X_train, X_val, X_test, scalers = normalize_data(X_train, X_val, X_test)

# Save scalers for later use
import joblib
joblib.dump(scalers, os.path.join(model_dir, 'scalers.pkl'))

# Advanced callbacks
early_stopping = EarlyStopping(
    monitor='val_mae',
    patience=100,  # More patience for complex models
    min_delta=1e-6,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=30,
    min_lr=1e-7,
    verbose=1
)

def cosine_annealing_with_warmup_and_restarts(epoch, lr):
    """Advanced learning rate schedule with warmup and restarts"""
    warmup_epochs = 10
    cycle_length = 100
    min_lr = 1e-7
    max_lr = 1e-4
    
    # Determine which cycle we're in
    cycle = epoch // cycle_length
    cycle_epoch = epoch % cycle_length
    
    # Warmup during first cycle only
    if cycle == 0 and cycle_epoch < warmup_epochs:
        return min_lr + (max_lr - min_lr) * (cycle_epoch / warmup_epochs)
    
    # Cosine decay for the remainder of each cycle
    progress = (cycle_epoch - warmup_epochs) / (cycle_length - warmup_epochs) if cycle == 0 else cycle_epoch / cycle_length
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    
    # Reduce max_lr slightly after each restart
    cycle_max_lr = max_lr * (0.9 ** cycle)
    
    return min_lr + (cycle_max_lr - min_lr) * cosine_decay

model_checkpoint = ModelCheckpoint(
    os.path.join(model_dir, 'best_model.keras'),
    monitor='val_mae',
    save_best_only=True,
    verbose=1
)

csv_logger = CSVLogger(os.path.join(performance_dir, 'training_log.csv'))

callbacks_list = [
    early_stopping,
    model_checkpoint,
    reduce_lr,
    tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup_and_restarts),
    csv_logger,
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.LambdaCallback(
        on_batch_end=lambda batch, logs: 
            tf.debugging.check_numerics(logs['loss'], 'Invalid loss detected'))
]

# Build and train the model
print("Building and training model...")
model = build_advanced_polarization_model(input_shape=(X_train.shape[1],), depth=8)

# Save model architecture summary
with open(os.path.join(performance_dir, f'model_summary_{version}.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Train the model with larger batch for faster convergence but not too large to maintain precision
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,  # Train longer
    batch_size=128,
    callbacks=callbacks_list,
    verbose=2
)

# Evaluate the model
print("Evaluating model...")
model = tf.keras.models.load_model(
    os.path.join(model_dir, 'best_model.keras'),
    custom_objects={'combined_precision_loss': combined_precision_loss}
)

y_test_pred = model.predict(X_test).flatten()
residuals = y_test - y_test_pred

# Calculate various metrics for thorough evaluation
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(np.square(residuals)))
max_error = np.max(np.abs(residuals))
rpe = np.abs((y_test - y_test_pred) / np.maximum(np.abs(y_test), 1e-10)) * 100
median_rpe = np.median(rpe)
p95_rpe = np.percentile(rpe, 95)

print(f"\nTest Set Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")
print(f"Maximum Absolute Error: {max_error:.8f}")
print(f"Median Relative Percentage Error: {median_rpe:.3f}%")
print(f"95th Percentile RPE: {p95_rpe:.3f}%")

# Save detailed metrics
metrics_dict = {
    'mae': mae,
    'rmse': rmse,
    'max_error': max_error,
    'median_rpe': median_rpe,
    'p95_rpe': p95_rpe
}

with open(os.path.join(performance_dir, 'test_metrics.txt'), 'w') as f:
    for metric, value in metrics_dict.items():
        f.write(f"{metric}: {value}\n")

# Save test predictions with high precision
test_results_df = pd.DataFrame({
    'Actual': y_test.round(8),
    'Predicted': y_test_pred.round(8),
    'Residuals': residuals.round(8),
    'RPE': rpe.round(4)
})
test_results_df.to_csv(os.path.join(performance_dir, f'test_event_results_{version}.csv'), index=False)

# Plot detailed evaluation metrics
def plot_enhanced_results(y_true, y_pred, output_dir, version_name):
    """Create detailed plots for model evaluation"""
    residuals = y_true - y_pred
    rpe = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10)) * 100
    
    # Create figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. True vs Predicted
    axs[0, 0].scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r-')
    axs[0, 0].set_xlabel('True Polarization (%)')
    axs[0, 0].set_ylabel('Predicted Polarization (%)')
    axs[0, 0].set_title('True vs Predicted Values')
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    axs[0, 1].hist(residuals, bins=100, alpha=0.7)
    axs[0, 1].axvline(x=0, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Residual Value')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title(f'Residuals Distribution (MAE={np.mean(np.abs(residuals)):.8f})')
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals vs True Values
    axs[1, 0].scatter(y_true, residuals, alpha=0.3, s=10)
    axs[1, 0].axhline(y=0, color='r', linestyle='--')
    axs[1, 0].set_xlabel('True Polarization (%)')
    axs[1, 0].set_ylabel('Residual')
    axs[1, 0].set_title('Residuals vs True Values')
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. RPE Distribution
    axs[1, 1].hist(rpe[rpe < np.percentile(rpe, 99)], bins=100, alpha=0.7)  # Exclude outliers
    axs[1, 1].set_xlabel('Relative Percentage Error (%)')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title(f'RPE Distribution (Median={np.median(rpe):.f}%)')
    axs[1, 1].grid(True, alpha=0.3)
    
    # 5. RPE vs True Values
    axs[2, 0].scatter(y_true, rpe, alpha=0.3, s=10)
    axs[2, 0].set_ylim(0, np.percentile(rpe, 99))  # Limit to exclude extreme outliers
    axs[2, 0].set_xlabel('True Polarization (%)')
    axs[2, 0].set_ylabel('Relative Percentage Error (%)')
    axs[2, 0].set_title('RPE vs True Values')
    axs[2, 0].grid(True, alpha=0.3)
    
    # 6. Log-scaled error analysis
    sorted_errors = np.sort(np.abs(residuals))
    cum_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    axs[2, 1].plot(sorted_errors, cum_pct)
    axs[2, 1].set_xscale('log')
    axs[2, 1].set_xlabel('Absolute Error (log scale)')
    axs[2, 1].set_ylabel('Cumulative Percentage')
    axs[2, 1].set_title('Error Distribution (Cumulative)')
    axs[2, 1].grid(True, alpha=0.3)
    
    # Add key metrics as text
    metrics_text = (
        f"MAE: {np.mean(np.abs(residuals)):.8f}\n"
        f"RMSE: {np.sqrt(np.mean(np.square(residuals))):.8f}\n"
        f"Max Error: {np.max(np.abs(residuals)):.8f}\n"
        f"Median RPE: {np.median(rpe):.4f}%\n"
        f"95th Percentile RPE: {np.percentile(rpe, 95):.4f}%\n"
        f"% within 0.001 abs error: {(np.abs(residuals) < 0.001).mean()*100:.2f}%\n"
        f"% within 0.1% rel error: {(rpe < 0.1).mean()*100:.2f}%"
    )
    
    fig.text(0.5, 0.01, metrics_text, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.suptitle(f"Model Evaluation - {version_name}", fontsize=16, y=0.995)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{version_name}_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot training history with learning rate
def plot_enhanced_training_history(history, output_dir, version_name):
    """Plot training history with extended metrics"""
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot loss
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot metrics
    axs[1].plot(history.history['mae'], label='Training MAE')
    axs[1].plot(history.history['val_mae'], label='Validation MAE')
    if 'rmse' in history.history:
        axs[1].plot(history.history['rmse'], label='Training RMSE')
        axs[1].plot(history.history['val_rmse'], label='Validation RMSE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Metric Value')
    axs[1].set_title('Training and Validation Metrics')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{version_name}_training_history.png'), dpi=300)
    plt.close()

# Generate plots
plot_enhanced_results(y_test, y_test_pred, performance_dir, version)
plot_enhanced_training_history(history, performance_dir, version)

print(f"Evaluation complete. Results saved to {performance_dir}")

# Create an ensemble model for even better precision (optional)
def create_ensemble(X_train, y_train, X_val, y_val, n_models=5):
    """Create an ensemble of models for improved prediction"""
    models = []
    kf = KFold(n_splits=n_models, shuffle=True, random_state=42)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining ensemble model {i+1}/{n_models}")
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Create a model with slightly different hyperparameters
        model = build_advanced_polarization_model(
            input_shape=(X_train.shape[1],), 
            depth=8 if i % 2 == 0 else 7
        )
        
        # Use callbacks specific to this fold
        callbacks = [
            EarlyStopping(monitor='val_mae', patience=50, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=20, min_lr=1e-7)
        ]
        
        # Train with different batch sizes
        batch_size = 128 if i % 2 == 0 else 256
        
        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=200,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Save model
        model.save(os.path.join(model_dir, f'ensemble_model_{i}.keras'))
        models.append(model)
    
    return models

# Create and evaluate ensemble (uncomment to use)

print("\nCreating ensemble model...")
ensemble_models = create_ensemble(X_train, y_train, X_val, y_val, n_models=5)

# Ensemble prediction
ensemble_preds = np.zeros_like(y_test)
for model in ensemble_models:
    ensemble_preds += model.predict(X_test).flatten()
ensemble_preds /= len(ensemble_models)

# Calculate ensemble metrics
ensemble_residuals = y_test - ensemble_preds
ensemble_mae = np.mean(np.abs(ensemble_residuals))
ensemble_rpe = np.abs((y_test - ensemble_preds) / np.maximum(np.abs(y_test), 1e-10)) * 100

print(f"\nEnsemble Model Metrics:")
print(f"Ensemble MAE: {ensemble_mae:.8f}")
print(f"Ensemble Median RPE: {np.median(ensemble_rpe):.4f}%")

# Save ensemble results
ensemble_results_df = pd.DataFrame({
    'Actual': y_test.round(8),
    'Predicted': ensemble_preds.round(8),
    'Residuals': ensemble_residuals.round(8),
    'RPE': ensemble_rpe.round(4)
})
ensemble_results_df.to_csv(os.path.join(performance_dir, f'ensemble_results_{version}.csv'), index=False)


# Save final model and preprocessing steps
model.save(os.path.join(model_dir, 'final_model.keras'))


# Create deployment-ready prediction function
# def create_prediction_script():
#     """Create a standalone script for making predictions with the trained model"""
#     script_path = os.path.join(model_dir, 'predict.py')
    
#     with open(script_path, 'w') as f:
#         f.write("""

def load_model_and_scalers(model_dir):
    # Load the trained model
    model = tf.keras.models.load_model(
        os.path.join(model_dir, 'final_model.keras'),
        compile=False  # Custom loss not needed for inference
    )
    
    # Load the scalers
    scalers = joblib.load(os.path.join(model_dir, 'scalers.pkl'))
    return model, scalers

def preprocess_data(X, scalers):
    # Apply the same preprocessing as during training
    scaler1, scaler2 = scalers
    X_robust = scaler1.transform(X)
    X_preprocessed = scaler2.transform(X_robust).astype('float32')
    return X_preprocessed

def predict_polarization(model, scalers, input_data):
    # Ensure input is numpy array
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    # Preprocess the input
    preprocessed_input = preprocess_data(input_data, scalers)
    
    # Make prediction
    predictions = model.predict(preprocessed_input).flatten()
    return predictions

