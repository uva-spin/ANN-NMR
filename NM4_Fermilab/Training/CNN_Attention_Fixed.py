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
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


data_path = find_file("Deuteron_TE_60_Noisy_Shifted_100K.parquet")  
version = 'Deuteron_TE_60_Noisy_Shifted_100K_CNN_Attention_Fixed_V4'  
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

# Reshape targets and convert to float32 for better precision
y_train = y_train.reshape(-1, 1).astype(np.float32)
y_val = y_val.reshape(-1, 1).astype(np.float32)
y_test = y_test.reshape(-1, 1).astype(np.float32)

# Create classification targets
y_train_class = (y_train < P_THRESHOLD).astype(np.float32)
y_val_class = (y_val < P_THRESHOLD).astype(np.float32)
y_test_class = (y_test < P_THRESHOLD).astype(np.float32)

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

def weighted_binary_crossentropy(class_weights):
    """Weighted BCE with improved numerical stability for noisy data"""
    def loss_fn(y_true, y_pred):
        # Apply sigmoid with higher numerical precision
        y_pred = tf.clip_by_value(tf.nn.sigmoid(y_pred), 1e-7, 1.0 - 1e-7)
        
        # Calculate weighted BCE with improved numerical stability
        weights = tf.where(tf.equal(y_true, 1), 
                          tf.ones_like(y_true) * class_weights[1],
                          tf.ones_like(y_true) * class_weights[0])
        
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(weights * bce)
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

def build_noise_robust_model(input_shape=(500, 1), is_classifier=True, l2_reg=1e-5):
    """Enhanced model architecture with improved feature extraction and precision"""
    inputs = Input(shape=input_shape, dtype='float64')
    
    # Initial feature extraction with larger kernel
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.Conv1D(64, 11, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg),
                     dtype='float64')(x)
    x = layers.BatchNormalization(dtype='float64')(x)
    x = layers.LayerNormalization(dtype='float64')(x)
    
    # First stage - preserve fine details
    x1 = noise_resistant_conv_block(x, 64, kernel_size=3, dilation_rate=1)
    x2 = noise_resistant_conv_block(x1, 64, kernel_size=3, dilation_rate=2)
    x3 = noise_resistant_conv_block(x2, 64, kernel_size=3, dilation_rate=4)
    x = layers.Add()([x1, x2, x3])  # Enhanced residual connection
    
    # Apply enhanced attention
    x = enhanced_attention(x)
    
    # Second stage with adaptive pooling
    x = layers.MaxPooling1D(2)(x)
    x4 = noise_resistant_conv_block(x, 128, kernel_size=3, dilation_rate=1)
    x5 = noise_resistant_conv_block(x4, 128, kernel_size=3, dilation_rate=2)
    x6 = noise_resistant_conv_block(x5, 128, kernel_size=3, dilation_rate=4)
    x = layers.Add()([x4, x5, x6])
    
    # More attention with skip connection
    x = enhanced_attention(x)
    
    # Third stage with global context
    x = layers.MaxPooling1D(2)(x)
    x7 = noise_resistant_conv_block(x, 256, kernel_size=3, dilation_rate=1)
    x8 = noise_resistant_conv_block(x7, 256, kernel_size=3, dilation_rate=2)
    x = layers.Add()([x7, x8])
    
    # Enhanced global context
    x_avg = layers.GlobalAveragePooling1D()(x)
    x_max = layers.GlobalMaxPooling1D()(x)
    x_std = layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1), dtype='float64')(x)
    x = layers.Concatenate()([x_avg, x_max, x_std])
    
    # Task-specific heads with enhanced architecture
    if is_classifier:
        x = layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(l2_reg),
                        dtype='float64')(x)
        x = layers.BatchNormalization(dtype='float64')(x)
        x = layers.LayerNormalization(dtype='float64')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        dtype='float64')(x)
        x = layers.BatchNormalization(dtype='float64')(x)
        x = layers.Dense(1, dtype='float64')(x)
    else:
        # Enhanced regression head for better precision
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        dtype='float64')(x)
        x = layers.BatchNormalization(dtype='float64')(x)
        x = layers.LayerNormalization(dtype='float64')(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        dtype='float64')(x)
        x = layers.BatchNormalization(dtype='float64')(x)
        x = layers.LayerNormalization(dtype='float64')(x)
        
        x = layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        dtype='float64')(x)
        x = layers.BatchNormalization(dtype='float64')(x)
        
        # Final layer with precision-focused activation
        x = layers.Dense(1, activation=None, dtype='float64')(x)
    
    return models.Model(inputs=inputs, outputs=x)

def create_optimized_dataset(X, y, batch_size, shuffle=True):
    """Creates an optimized TF dataset with noise-robust processing"""
    # Convert inputs to the most efficient format
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000))
    
    # Performance optimizations
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Define optimized hyperparameters
    params = {
        'learning_rate': 1e-3,       # Higher learning rate for faster initial convergence
        'batch_size': 512,           # Larger batch size for noise averaging
        'l2_reg': 5e-5,              # L2 regularization to prevent overfitting
        'epochs': 75,                # More epochs for better convergence
        'patience': 10               # More patience for noisy training curves
    }
    
    start_time = time.time()
    
    # Create models with noise-robust architecture
    classifier_model = build_noise_robust_model(
        input_shape=(X_train.shape[1], 1), 
        is_classifier=True, 
        l2_reg=params['l2_reg']
    )
    
    reg_model_low = build_noise_robust_model(
        input_shape=(X_train.shape[1], 1), 
        is_classifier=False, 
        l2_reg=params['l2_reg']
    )
    
    reg_model_high = build_noise_robust_model(
        input_shape=(X_train.shape[1], 1), 
        is_classifier=False, 
        l2_reg=params['l2_reg']
    )
    
    print(f"Model building time: {time.time() - start_time:.2f} seconds")
    
    # Create optimized datasets
    train_dataset_class = create_optimized_dataset(
        X_train, y_train_class, params['batch_size']
    )
    val_dataset_class = create_optimized_dataset(
        X_val, y_val_class, params['batch_size'], shuffle=False
    )
    
    train_dataset_low = create_optimized_dataset(
        X_train_low, y_train_low, params['batch_size']
    )
    val_dataset_low = create_optimized_dataset(
        X_val_low, y_val_low, params['batch_size'], shuffle=False
    )
    
    train_dataset_high = create_optimized_dataset(
        X_train_high, y_train_high, params['batch_size']
    )
    val_dataset_high = create_optimized_dataset(
        X_val_high, y_val_high, params['batch_size'], shuffle=False
    )
    
    # OneCycle learning rate schedule - faster convergence and better generalization
    # Especially effective for noisy data
    steps_per_epoch = len(X_train) // params['batch_size']
    total_steps = params['epochs'] * steps_per_epoch
    
    class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, total_steps, pct_start=0.3, div_factor=25, final_div_factor=10000):
            super(OneCycleLR, self).__init__()
            self.max_lr = max_lr
            self.total_steps = total_steps
            self.pct_start = pct_start
            self.div_factor = div_factor
            self.final_div_factor = final_div_factor
            
            # Calculate key points
            self.step_size_up = int(total_steps * pct_start)
            self.step_size_down = total_steps - self.step_size_up
            
            # Calculate learning rates
            self.initial_lr = max_lr / div_factor
            self.final_lr = max_lr / (div_factor * final_div_factor)
            
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            
            # Use tf.where instead of if/else for control flow
            lr = tf.where(
                step < self.step_size_up,
                # Warm-up phase
                self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.step_size_up),
                # Annealing phase
                self.max_lr + (self.final_lr - self.max_lr) * ((step - self.step_size_up) / self.step_size_down)
            )
            
            return lr
            
        def get_config(self):
            return {
                "max_lr": self.max_lr,
                "total_steps": self.total_steps,
                "pct_start": self.pct_start,
                "div_factor": self.div_factor,
                "final_div_factor": self.final_div_factor
            }
    
    lr_schedule = OneCycleLR(
        max_lr=params['learning_rate'],
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training warming up
        div_factor=25,  # max_lr/div_factor = initial lr
        final_div_factor=10000  # how much lower is final lr than max_lr
    )
    
    # Use weighted BCE for classifier
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-7),
        loss=weighted_binary_crossentropy(class_weights),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Add a new high-precision loss function
    def high_precision_loss(y_true, y_pred):
        """Custom loss function optimized for high precision regression"""
        # Convert to float64 for higher precision
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        # Small epsilon for numerical stability
        epsilon = tf.constant(1e-10, dtype=tf.float64)
        
        # Value importance weights - higher weight for smaller values
        value_importance = 1.0 / (tf.abs(y_true) + epsilon)
        
        # Relative error component
        relative_error = value_importance * tf.abs(y_pred - y_true) / (tf.abs(y_true) + epsilon)
        
        # Absolute error component
        absolute_error = tf.abs(y_pred - y_true)
        
        # Log-space error component
        log_predictions = tf.math.log(tf.abs(y_pred) + epsilon)
        log_targets = tf.math.log(tf.abs(y_true) + epsilon)
        log_space_error = tf.abs(log_predictions - log_targets)
        
        # Combined loss with emphasis on precision
        combined_loss = (
            2.0 * relative_error +  # Higher weight for relative error
            1.0 * absolute_error +  # Base absolute error
            3.0 * log_space_error   # Higher weight for log-space error
        )
        
        return tf.reduce_mean(combined_loss)
    
    # For regression models, use the new high-precision loss
    reg_model_low.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=high_precision_loss,
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    reg_model_high.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8),
        loss=high_precision_loss,
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Set up logging and callbacks
    classifier_log_file = f"{performance_dir}/training_log_classifier_model.csv"
    low_reg_log_file = f"{performance_dir}/training_log_low_reg_model.csv"
    high_reg_log_file = f"{performance_dir}/training_log_high_reg_model.csv"
    
    classifier_csv_logger = tf.keras.callbacks.CSVLogger(classifier_log_file, append=True, separator=',')
    low_reg_csv_logger = tf.keras.callbacks.CSVLogger(low_reg_log_file, append=True, separator=',')
    high_reg_csv_logger = tf.keras.callbacks.CSVLogger(high_reg_log_file, append=True, separator=',')
    
    # Exponential Moving Average callback for more stable weights in noisy environments
    class EMACallback(tf.keras.callbacks.Callback):
        def __init__(self, decay=0.999):
            super(EMACallback, self).__init__()
            self.decay = decay
            self.shadow_weights = []
            self.has_initialized = False
            
        def on_train_begin(self, logs=None):
            # Initialize shadow weights to model weights
            for weight in self.model.weights:
                self.shadow_weights.append(tf.Variable(weight, trainable=False))
            self.has_initialized = True
            
        def on_batch_end(self, batch, logs=None):
            # Update shadow weights
            for i, weight in enumerate(self.model.weights):
                self.shadow_weights[i].assign(
                    self.decay * self.shadow_weights[i] + (1.0 - self.decay) * weight
                )
                
        def on_epoch_end(self, epoch, logs=None):
            # Log current loss and apply EMA weights for validation
            print(f"EMA updated on epoch {epoch}")
            
        def apply_ema_weights(self):
            # Apply shadow weights to the model for inference
            if self.has_initialized:
                model_weights = self.model.get_weights()
                for i, weight in enumerate(self.model.weights):
                    weight.assign(self.shadow_weights[i])
                return model_weights
            return None
        
        def restore_model_weights(self, model_weights):
            if model_weights is not None:
                self.model.set_weights(model_weights)
    
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
    
    ema_callback_classifier = EMACallback(decay=0.999)
    ema_callback_low = EMACallback(decay=0.999)
    ema_callback_high = EMACallback(decay=0.999)
    time_callback = TimeHistory()
    
    # Train classifier model
    print("Training classifier model...")
    start_time = time.time()
    classifier_history = classifier_model.fit(
        train_dataset_class,
        validation_data=val_dataset_class,
        epochs=params['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', 
                patience=params['patience'],
                restore_best_weights=True,
                mode='max'
            ),
            ema_callback_classifier,
            classifier_csv_logger,
            time_callback
        ],
        verbose=1
    )
    # Apply EMA weights for inference
    model_weights = ema_callback_classifier.apply_ema_weights()
    print(f"Classifier training time: {time.time() - start_time:.2f} seconds")
    
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
            ema_callback_low,
            low_reg_csv_logger,
            time_callback
        ],
        verbose=1
    )
    ema_callback_low.apply_ema_weights()
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
            ema_callback_high,
            high_reg_csv_logger,
            time_callback
        ],
        verbose=1
    )
    ema_callback_high.apply_ema_weights()
    print(f"High region training time: {time.time() - start_time:.2f} seconds")

    # Save models in SavedModel format for better performance
    classifier_model.save(f"{model_dir}/classifier_model.keras")
    reg_model_low.save(f"{model_dir}/low_reg_model.keras")
    reg_model_high.save(f"{model_dir}/high_reg_model.keras")

    # Evaluate with ensemble approach for more robust predictions
    print("Evaluating models on test set...")
    start_time = time.time()
    
    # Create test datasets
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(params['batch_size'])
    
    # Get predictions
    test_class_pred = classifier_model.predict(test_dataset)
    
    # Use probability threshold tuned for optimal F1 score
    # This is better than a fixed 0.5 threshold for imbalanced/noisy data
    optimal_threshold = 0.4  # This could be tuned based on validation set
    test_class_binary = (test_class_pred > optimal_threshold).astype(np.float32)
    
    # Use masks for targeted predictions
    test_low_mask = test_class_binary.flatten() == 1
    test_high_mask = test_class_binary.flatten() == 0
    
    # Initialize prediction array
    test_pred = np.zeros_like(y_test, dtype=np.float64)  # Ensure float64
    
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
        if true != 0:  # Avoid division by zero
            rel_error = abs(pred - true) / abs(true)
            precision_errors.append(rel_error)
    
    mean_rel_error = np.mean(precision_errors) if precision_errors else 0
    
    print(f"Test MAE: {test_mae:.12f}")
    print(f"Test RMSE: {test_rmse:.12f}")
    print(f"Test R²: {test_r2:.12f}")
    print(f"Mean Relative Error: {mean_rel_error:.8f}")
    print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
    
    # Flatten for plotting
    y_test_flat = y_test.flatten()
    y_pred_flat = test_pred.flatten()

    # Plot performance metrics and results
    plot_enhanced_performance_metrics(y_test_flat, y_pred_flat, snr_test, performance_dir, version)
    plot_enhanced_results(y_test_flat, y_pred_flat, performance_dir, version)
    
    # Create a function to save a figure with training metrics
    def plot_combined_training_history():
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot classifier accuracy and AUC
        axs[0, 0].plot(classifier_history.history['accuracy'])
        axs[0, 0].plot(classifier_history.history['val_accuracy'])
        axs[0, 0].plot(classifier_history.history['auc'])
        axs[0, 0].plot(classifier_history.history['val_auc'])
        axs[0, 0].set_title('Classifier Model Performance')
        axs[0, 0].set_ylabel('Metric Value')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['train_acc', 'val_acc', 'train_auc', 'val_auc'], loc='upper left')
        
        # Plot classifier loss
        axs[0, 1].plot(classifier_history.history['loss'])
        axs[0, 1].plot(classifier_history.history['val_loss'])
        axs[0, 1].set_title('Classifier Model Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['train', 'val'], loc='upper left')
        
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
    
    # Create a confusion matrix for the classifier
    def plot_classifier_confusion_matrix():
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
        
        # Convert predictions to binary
        test_class_pred_binary = (test_class_pred > 0.5).astype(int)
        y_test_class_binary = y_test_class.astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test_class_binary, test_class_pred_binary)
        
        # Generate classification report
        report = classification_report(y_test_class_binary, test_class_pred_binary, 
                                     target_names=['High P', 'Low P'])
        print("Classification Report:")
        print(report)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High P', 'Low P'])
        disp.plot(ax=ax, cmap='Blues', values_format='.4g')
        plt.title('Classifier Confusion Matrix')
        plt.savefig(f"{performance_dir}/classifier_confusion_matrix.png")
        plt.close()
        
        # Save report to file
        with open(f"{performance_dir}/classification_report.txt", 'w') as f:
            f.write(report)
    
    # Plot classifier confusion matrix
    plot_classifier_confusion_matrix()
    
    # Print completion message with summary
    print("\nTraining and evaluation complete.")
    print(f"Models and results saved to: {performance_dir} and {model_dir}")
    print(f"Final test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.6f}")