import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import sys
import os
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Misc_Functions import *
from Custom_Scripts.Loss_Functions import *
from Custom_Scripts.Lineshape import *
from Plotting.Plot_Script import *

# Set global precision policy to float64
tf.keras.mixed_precision.set_global_policy('float64')
tf.keras.backend.set_floatx('float64')

# Set deterministic behavior for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Enable XLA JIT compilation for much faster training
tf.config.optimizer.set_jit(True)  
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


data_path = find_file("Deuteron_TE_60_Noisy_Shifted_100K.parquet")  
version = 'Deuteron_TE_60_Noisy_Shifted_100K_CNN_XGBoost_V1'  
performance_dir = f"Model_Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the polarization threshold for classification
P_THRESHOLD = 0.1

start_time = time.time()

try:
    data = pd.read_parquet(data_path, engine='pyarrow')
    print("Data loaded successfully from Parquet file!")
except Exception as e:
    print(f"Error loading data: {e}")
    
# Shuffle data once with fixed seed for reproducibility
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Data loading time: {time.time() - start_time:.2f} seconds")
start_time = time.time()

# Use stratified split to maintain class distribution
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, 
                                         stratify=data["P"] < P_THRESHOLD)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, 
                                       stratify=temp_data["P"] < P_THRESHOLD)

# Convert data to float64
X_train = train_data.drop(columns=["P", 'SNR']).values.astype(np.float64)
y_train = train_data["P"].values.astype(np.float64)
X_val = val_data.drop(columns=["P", 'SNR']).values.astype(np.float64)
y_val = val_data["P"].values.astype(np.float64)
X_test = test_data.drop(columns=["P", 'SNR']).values.astype(np.float64)
y_test = test_data["P"].values.astype(np.float64)
snr_test = test_data["SNR"].values if "SNR" in test_data.columns else None

print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")
start_time = time.time()

# Use RobustScaler for better handling of outliers in noisy data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# Reshape targets and convert to float32 for better precision
y_train = y_train.reshape(-1, 1).astype(np.float32)
y_val = y_val.reshape(-1, 1).astype(np.float32)
y_test = y_test.reshape(-1, 1).astype(np.float32)

# Create classification targets
y_train_class = (y_train < P_THRESHOLD).astype(np.int32)
y_val_class = (y_val < P_THRESHOLD).astype(np.int32)
y_test_class = (y_test < P_THRESHOLD).astype(np.int32)

# Calculate class weights to handle imbalance
class_counts = np.bincount(y_train_class.flatten().astype(int))
total_samples = len(y_train_class)
class_weights = {
    0: total_samples / (2 * class_counts[0]),  
    1: total_samples / (2 * class_counts[1])  
}
print(f"Class distribution: {class_counts}")
print(f"Class weights: {class_weights}")

# Split regression data based on threshold
X_train_low = X_train[y_train_class.flatten() == 1]
y_train_low = y_train[y_train_class.flatten() == 1]
X_train_high = X_train[y_train_class.flatten() == 0]
y_train_high = y_train[y_train_class.flatten() == 0]

X_val_low = X_val[y_val_class.flatten() == 1]
y_val_low = y_val[y_val_class.flatten() == 1]
X_val_high = X_val[y_val_class.flatten() == 0]
y_val_high = y_val[y_val_class.flatten() == 0]

print(f"Data splitting time: {time.time() - start_time:.2f} seconds")

# Custom smooth L1/Huber loss with adaptive delta based on noise level
def adaptive_smooth_l1_loss():
    """Smooth L1 loss that adapts to the noise level in the data"""
    def loss_fn(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        # Use smaller delta for higher precision
        delta = 0.03
        
        # Smooth L1 / Huber loss formula
        less_than_delta = 0.5 * tf.square(abs_error)
        greater_than_delta = delta * (abs_error - 0.5 * delta)
        
        # Combine both parts
        loss = tf.where(abs_error < delta, less_than_delta, greater_than_delta)
        
        return tf.reduce_mean(loss)
    return loss_fn

def noise_resistant_conv_block(x, filters, kernel_size=3, dilation_rate=1):
    """Enhanced convolution block with multi-scale feature extraction and residual connections"""
    # Multi-scale feature extraction with different kernel sizes
    conv1 = layers.Conv1D(filters//4, kernel_size, padding='same', 
                         activation='relu', dilation_rate=dilation_rate,
                         dtype='float64')(x)
    conv2 = layers.Conv1D(filters//4, kernel_size*2+1, padding='same', 
                         activation='relu', dilation_rate=dilation_rate,
                         dtype='float64')(x)
    conv3 = layers.Conv1D(filters//4, kernel_size*3+2, padding='same', 
                         activation='relu', dilation_rate=dilation_rate,
                         dtype='float64')(x)
    conv4 = layers.Conv1D(filters//4, 1, padding='same', 
                         activation='relu', dtype='float64')(x)  # Pointwise convolution
    
    # Concatenate different kernel sizes
    x = layers.Concatenate()([conv1, conv2, conv3, conv4])
    
    # Enhanced normalization
    x = layers.BatchNormalization(dtype='float64')(x)
    x = layers.LayerNormalization(dtype='float64')(x)  # Additional normalization
    
    # Spatial dropout for better regularization
    x = layers.SpatialDropout1D(0.1)(x)
    
    # Residual connection if dimensions match
    if x.shape[-1] == filters:
        x = layers.Add()([x, x])
    
    return x

def enhanced_attention(x, ratio=8):
    """Enhanced attention mechanism with multi-head attention and channel attention"""
    channel_dim = x.shape[-1]
    
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=channel_dim//4, dtype='float64'
    )(x, x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(dtype='float64')(x)
    
    # Channel attention
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    
    # Enhanced channel attention with multiple layers
    combined = layers.Concatenate()([avg_pool, max_pool])
    attention = layers.Dense(channel_dim // ratio, activation='relu', dtype='float64')(combined)
    attention = layers.Dense(channel_dim // ratio, activation='relu', dtype='float64')(attention)
    attention = layers.Dense(channel_dim, activation='sigmoid', dtype='float64')(attention)
    
    # Apply attention
    attention = layers.Reshape((1, channel_dim))(attention)
    return layers.Multiply()([x, attention])

def build_precision_regression_model(input_shape=(500,), l2_reg=1e-5):
    """Optimized regression model for high precision polarization prediction"""
    inputs = Input(shape=input_shape, dtype='float64')
    
    # Initial feature extraction with frequency-aware processing
    x = layers.Reshape((input_shape[0], 1))(inputs)
    
    # First stage: Multi-scale feature extraction
    # Use parallel convolutions with different kernel sizes to capture various frequency patterns
    conv1 = layers.Conv1D(32, 3, padding='same', activation='relu', 
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    conv2 = layers.Conv1D(32, 5, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    conv3 = layers.Conv1D(32, 7, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()([conv1, conv2, conv3])
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    
    # Second stage: Feature refinement with dilated convolutions
    x = layers.Conv1D(64, 3, padding='same', dilation_rate=2, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.1)(x)
    
    # Third stage: Global context with attention
    # Self-attention for capturing long-range dependencies
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=16)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Global pooling with multiple strategies
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Dense layers with residual connections
    dense1 = layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.2)(dense1)
    
    dense2 = layers.Dense(32, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.2)(dense2)
    
    # Output layer with specialized initialization for regression
    outputs = layers.Dense(1, activation=None,
                          kernel_initializer=tf.keras.initializers.GlorotNormal(),
                          bias_initializer='zeros')(dense2)
    
    return models.Model(inputs=inputs, outputs=outputs)

def precision_focused_loss():
    """Loss function optimized for high precision regression"""
    def loss_fn(y_true, y_pred):
        # Calculate absolute error
        abs_error = tf.abs(y_true - y_pred)
        
        # Calculate relative error with numerical stability
        rel_error = abs_error / (tf.abs(y_true) + 1e-8)
        
        # Weight the loss based on the magnitude of the true value
        # This gives more importance to small values
        weights = 1.0 / (tf.abs(y_true) + 1e-8)
        weights = tf.clip_by_value(weights, 1.0, 100.0)  # Limit weight range
        
        # Combine absolute and relative errors
        loss = weights * (0.5 * tf.square(abs_error) + 0.5 * rel_error)
        
        return tf.reduce_mean(loss)
    return loss_fn

# Create TensorFlow datasets for efficient training
def create_tf_dataset(X, y, batch_size=32, shuffle=True, buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    # Define optimized hyperparameters
    params = {
        'learning_rate': 0.001,
        'batch_size': 128,  # Increased for better gradient estimates
        'l2_reg': 1e-5,
        'epochs': 200,
        'patience': 30
    }
    
    # Create regression models
    print("Creating regression models...")
    reg_model_low = build_precision_regression_model(input_shape=(X_train.shape[1],))
    reg_model_high = build_precision_regression_model(input_shape=(X_train.shape[1],))
    
    # Configure learning rate schedule with warmup
    initial_learning_rate = params['learning_rate']
    decay_steps = params['epochs'] * (len(X_train) // params['batch_size'])
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=decay_steps // 4,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    # Compile regression models with optimized settings
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    reg_model_low.compile(
        optimizer=optimizer,
        loss=precision_focused_loss(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    reg_model_high.compile(
        optimizer=optimizer,
        loss=precision_focused_loss(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Create TensorFlow datasets
    train_dataset_low = create_tf_dataset(X_train_low, y_train_low, params['batch_size'], shuffle=True)
    val_dataset_low = create_tf_dataset(X_val_low, y_val_low, params['batch_size'], shuffle=False)
    
    train_dataset_high = create_tf_dataset(X_train_high, y_train_high, params['batch_size'], shuffle=True)
    val_dataset_high = create_tf_dataset(X_val_high, y_val_high, params['batch_size'], shuffle=False)
    
    # Train models with improved callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=params['patience'],
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model_{}_polarization.keras'.format('low' if 'low' in locals() else 'high')),
            monitor='val_mae',
            save_best_only=True
        )
    ]
    
    # Create XGBoost classifier model
    print("Creating XGBoost classifier model...")
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=class_weights[1]/class_weights[0],  # Handle class imbalance
        objective='binary:logistic',
        eval_metric=['auc', 'error', 'logloss'],
        use_label_encoder=False,
        random_state=42
    )
    
    # Set up CSV loggers for tracking metrics
    os.makedirs(f"{performance_dir}/logs", exist_ok=True)
    low_reg_csv_logger = tf.keras.callbacks.CSVLogger(
        f"{performance_dir}/logs/low_reg_training_log.csv"
    )
    high_reg_csv_logger = tf.keras.callbacks.CSVLogger(
        f"{performance_dir}/logs/high_reg_training_log.csv"
    )
    
    # Time history callback
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []
            self.epoch_start_time = time.time()
        
        def on_epoch_end(self, batch, logs={}):
            epoch_time = time.time() - self.epoch_start_time
            self.times.append(epoch_time)
            print(f" - {epoch_time:.2f}s/epoch")
            self.epoch_start_time = time.time()
    
    time_callback = TimeHistory()
    
    # Train XGBoost classifier model
    print("Training XGBoost classifier model...")
    start_time = time.time()
    
    # Flatten y_train_class for XGBoost
    y_train_class_flat = y_train_class.flatten()
    y_val_class_flat = y_val_class.flatten()
    
    # Train XGBoost with early stopping
    xgb_classifier.fit(
        X_train, y_train_class_flat,
        eval_set=[(X_val, y_val_class_flat)],
        verbose=True
    )
    
    print(f"XGBoost classifier training time: {time.time() - start_time:.2f} seconds")
    
    # Save XGBoost model
    joblib.dump(xgb_classifier, f"{model_dir}/xgb_classifier.pkl")
    
    # Train low region model
    print("Training low region regression model...")
    start_time = time.time()
    low_reg_history = reg_model_low.fit(
        train_dataset_low,
        validation_data=val_dataset_low,
        epochs=params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mae', 
                patience=params['patience'],
                restore_best_weights=True,
                mode='min'
            ),
            low_reg_csv_logger,
            time_callback
        ],
        verbose=1
    )
    print(f"Low region training time: {time.time() - start_time:.2f} seconds")
    
    # Train high region model
    print("Training high region regression model...")
    start_time = time.time()
    high_reg_history = reg_model_high.fit(
        train_dataset_high,
        validation_data=val_dataset_high,
        epochs=params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_mae', 
                patience=params['patience'],
                restore_best_weights=True,
                mode='min'
            ),
            high_reg_csv_logger,
            time_callback
        ],
        verbose=1
    )
    print(f"High region training time: {time.time() - start_time:.2f} seconds")

    # Save models
    reg_model_low.save(f"{model_dir}/low_reg_model.keras")
    reg_model_high.save(f"{model_dir}/high_reg_model.keras")

    # Evaluate models on test set
    print("Evaluating models on test set...")
    start_time = time.time()
    
    # Get XGBoost predictions
    test_class_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
    test_class_binary = xgb_classifier.predict(X_test)
    
    # Use masks for targeted predictions
    test_low_mask = test_class_binary == 1
    test_high_mask = test_class_binary == 0
    
    # Initialize prediction array
    test_pred = np.zeros_like(y_test, dtype=np.float64)
    
    # Only predict samples classified as low
    if np.any(test_low_mask):
        test_pred_low = reg_model_low.predict(
            tf.data.Dataset.from_tensor_slices(X_test[test_low_mask]).batch(params['batch_size'])
        )
        test_pred[test_low_mask] = test_pred_low
    
    # Only predict samples classified as high
    if np.any(test_high_mask):
        test_pred_high = reg_model_high.predict(
            tf.data.Dataset.from_tensor_slices(X_test[test_high_mask]).batch(params['batch_size'])
        )
        test_pred[test_high_mask] = test_pred_high
    
    # Calculate metrics with high precision
    test_mae = np.mean(np.abs(test_pred - y_test), dtype=np.float64)
    test_rmse = np.sqrt(np.mean(np.square(test_pred - y_test), dtype=np.float64))
    test_r2 = 1 - (np.sum(np.square(test_pred - y_test), dtype=np.float64) / 
                   np.sum(np.square(y_test - np.mean(y_test)), dtype=np.float64))
    
    # Calculate additional precision metrics
    precision_errors = []
    for pred, true in zip(test_pred.flatten(), y_test.flatten()):
        if abs(true) > 1e-10:  # Avoid division by zero
            rel_error = abs(pred - true) / abs(true)
            precision_errors.append(rel_error)
    
    mean_rel_error = np.mean(precision_errors) if precision_errors else 0
    
    # Calculate classifier metrics
    accuracy = accuracy_score(y_test_class.flatten(), test_class_binary)
    auc = roc_auc_score(y_test_class.flatten(), test_class_pred_proba)
    precision = precision_score(y_test_class.flatten(), test_class_binary)
    recall = recall_score(y_test_class.flatten(), test_class_binary)
    f1 = f1_score(y_test_class.flatten(), test_class_binary)
    
    print(f"Test MAE: {test_mae:.12f}")
    print(f"Test RMSE: {test_rmse:.12f}")
    print(f"Test R²: {test_r2:.12f}")
    print(f"Mean Relative Error: {mean_rel_error:.8f}")
    print(f"XGBoost Classifier Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
    
    # Flatten for plotting
    y_test_flat = y_test.flatten()
    y_pred_flat = test_pred.flatten()

    # Plot overall performance metrics and results
    plot_enhanced_performance_metrics(y_test_flat, y_pred_flat, snr_test, performance_dir, version)
    plot_enhanced_results(y_test_flat, y_pred_flat, performance_dir, version)
    
    # Evaluate and plot performance for low and high polarization models separately
    if np.any(test_low_mask):
        y_test_low = y_test[test_low_mask].flatten()
        y_pred_low = test_pred[test_low_mask].flatten()
        snr_test_low = snr_test[test_low_mask] if snr_test is not None else None
        
        # Calculate metrics for low polarization model
        low_mae = np.mean(np.abs(y_pred_low - y_test_low), dtype=np.float64)
        low_rmse = np.sqrt(np.mean(np.square(y_pred_low - y_test_low), dtype=np.float64))
        low_r2 = 1 - (np.sum(np.square(y_pred_low - y_test_low), dtype=np.float64) / 
                     np.sum(np.square(y_test_low - np.mean(y_test_low)), dtype=np.float64))
        
        # Calculate relative error for low polarization
        low_rel_errors = []
        for pred, true in zip(y_pred_low, y_test_low):
            if abs(true) > 1e-10:
                rel_error = abs(pred - true) / abs(true)
                low_rel_errors.append(rel_error)
        
        low_mean_rel_error = np.mean(low_rel_errors) if low_rel_errors else 0
        
        print(f"\nLow Polarization Model Performance:")
        print(f"  MAE: {low_mae:.12f}")
        print(f"  RMSE: {low_rmse:.12f}")
        print(f"  R²: {low_r2:.12f}")
        print(f"  Mean Relative Error: {low_mean_rel_error:.8f}")
        
        # Plot performance for low polarization model
        plot_enhanced_performance_metrics(y_test_low, y_pred_low, snr_test_low, 
                                         performance_dir, f"{version}_low_polarization")
        plot_enhanced_results(y_test_low, y_pred_low, 
                             performance_dir, f"{version}_low_polarization")
    
    if np.any(test_high_mask):
        y_test_high = y_test[test_high_mask].flatten()
        y_pred_high = test_pred[test_high_mask].flatten()
        snr_test_high = snr_test[test_high_mask] if snr_test is not None else None
        
        # Calculate metrics for high polarization model
        high_mae = np.mean(np.abs(y_pred_high - y_test_high), dtype=np.float64)
        high_rmse = np.sqrt(np.mean(np.square(y_pred_high - y_test_high), dtype=np.float64))
        high_r2 = 1 - (np.sum(np.square(y_pred_high - y_test_high), dtype=np.float64) / 
                      np.sum(np.square(y_test_high - np.mean(y_test_high)), dtype=np.float64))
        
        # Calculate relative error for high polarization
        high_rel_errors = []
        for pred, true in zip(y_pred_high, y_test_high):
            if abs(true) > 1e-10:
                rel_error = abs(pred - true) / abs(true)
                high_rel_errors.append(rel_error)
        
        high_mean_rel_error = np.mean(high_rel_errors) if high_rel_errors else 0
        
        print(f"\nHigh Polarization Model Performance:")
        print(f"  MAE: {high_mae:.12f}")
        print(f"  RMSE: {high_rmse:.12f}")
        print(f"  R²: {high_r2:.12f}")
        print(f"  Mean Relative Error: {high_mean_rel_error:.8f}")
        
        # Plot performance for high polarization model
        plot_enhanced_performance_metrics(y_test_high, y_pred_high, snr_test_high, 
                                         performance_dir, f"{version}_high_polarization")
        plot_enhanced_results(y_test_high, y_pred_high, 
                             performance_dir, f"{version}_high_polarization")

    # Create a function to save a figure with training metrics
    def plot_combined_training_history():
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot XGBoost feature importance
        xgb.plot_importance(xgb_classifier, ax=axs[0, 0], max_num_features=10)
        axs[0, 0].set_title('XGBoost Feature Importance')
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test_class.flatten(), test_class_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High P', 'Low P'])
        disp.plot(ax=axs[0, 1], cmap='Blues', values_format='d')
        axs[0, 1].set_title('Classifier Confusion Matrix')
        
        # Plot regression model MAE
        axs[1, 0].plot(low_reg_history.history['mae'])
        axs[1, 0].plot(low_reg_history.history['val_mae'])
        axs[1, 0].plot(high_reg_history.history['mae'])
        axs[1, 0].plot(high_reg_history.history['val_mae'])
        axs[1, 0].set_title('Regression Models MAE')
        axs[1, 0].set_ylabel('MAE')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['low train', 'low val', 'high train', 'high val'], loc='upper left')
        
        # Plot regression model loss
        axs[1, 1].plot(low_reg_history.history['loss'])
        axs[1, 1].plot(low_reg_history.history['val_loss'])
        axs[1, 1].plot(high_reg_history.history['loss'])
        axs[1, 1].plot(high_reg_history.history['val_loss'])
        axs[1, 1].set_title('Regression Models Loss')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend(['low train', 'low val', 'high train', 'high val'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/training_history.png")
        plt.close()
    
    # Plot training history
    plot_combined_training_history()
    
    # Save classification report
    report = classification_report(y_test_class.flatten(), test_class_binary, 
                                 target_names=['High P', 'Low P'])
    print("Classification Report:")
    print(report)
    
    # Save report to file
    with open(f"{performance_dir}/classification_report.txt", 'w') as f:
        f.write(report)
    
    # Save test results to CSV
    results_df = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': test_pred.flatten(),
        'Residuals': (y_test - test_pred).flatten(),
        'Relative_Error': np.array(precision_errors) if precision_errors else np.zeros(len(y_test)),
        'Class_Actual': y_test_class.flatten(),
        'Class_Predicted': test_class_binary,
        'Class_Probability': test_class_pred_proba
    })
    results_df.to_csv(f"{performance_dir}/test_results.csv", index=False)
    
    # Print completion message with summary
    print("\nTraining and evaluation complete.")
    print(f"Models and results saved to: {performance_dir} and {model_dir}")
    print(f"Final test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")
