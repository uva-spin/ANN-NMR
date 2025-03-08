import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, optimizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from Misc_Functions import *
import random
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from Plotting.RPE_Histograms import analyze_model_errors

### Let's set a specific seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
tf.config.optimizer.set_jit(True)

tf.keras.mixed_precision.set_global_policy('float64')
tf.keras.backend.set_floatx('float64')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")

data_path = find_file("Deuteron_Low_No_Noise_500K.csv")  
version = 'Deuteron_Low_Noise_HighPrecision_V1'  
performance_dir = f"Model Performance/{version}"  
model_dir = f"Models/{version}"  
os.makedirs(performance_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def residual_block(x, units, dropout_rate=0.1, l1=1e-6, l2=1e-5, use_attention=True):
    shortcut = x
    
    x = layers.Dense(units, 
                     activation='swish', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                     dtype='float64')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, 
                     activation=None, 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                     dtype='float64')(x)
    x = layers.LayerNormalization()(x)
    
    if use_attention:
        attention = layers.Dense(units, activation='sigmoid', dtype='float64')(x)
        x = layers.Multiply()([x, attention])
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, 
                                activation=None, 
                                kernel_initializer='he_normal', 
                                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), 
                                dtype='float64')(shortcut)
        shortcut = layers.LayerNormalization()(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('swish')(x)
    
    return x

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=0.1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = scale
        
    def call(self, inputs):
        return inputs * self.scale
    
    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        config.update({"scale": self.scale})
        return config

tf.keras.utils.get_custom_objects().update({'ScaleLayer': ScaleLayer})

def weighted_mse(y_true, y_pred):

    mse = tf.square(y_true - y_pred)
    

    center_weight = tf.exp(-200.0 * tf.square(y_true - 0.0005)) * 10.0
    
    small_value_weight = tf.exp(-5.0 * y_true) + 1.0
    
    weights = small_value_weight + center_weight
    
    weighted_loss = mse * weights
    
    return tf.reduce_mean(weighted_loss)

def create_high_precision_model():

    inputs = layers.Input(shape=(500,), dtype='float64')
    
    x = layers.LayerNormalization()(inputs)
    
    x = layers.Dense(512, activation='swish', kernel_initializer='he_normal', dtype='float64')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    units = [512, 256, 256, 128, 128, 64, 64, 32]
    for i, u in enumerate(units):
        x = residual_block(x, u, 
                          dropout_rate=0.1 if i < 4 else 0.05,  
                          l1=1e-7,  
                          l2=1e-6, 
                          use_attention=True)  # Use attention in all blocks
    
    # Additional feature extraction layers
    x = layers.Dense(32, activation='swish', kernel_initializer='he_normal', dtype='float64')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(16, activation='swish', kernel_initializer='he_normal', dtype='float64')(x)
    x = layers.LayerNormalization()(x)
    
    outputs = layers.Dense(1, 
                          activation='sigmoid',
                          kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                          kernel_regularizer=regularizers.l1_l2(l1=1e-7, l2=1e-6),
                          dtype='float64')(x)
    
    outputs = ScaleLayer(scale=0.1)(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.AdamW(
        learning_rate=0.00005, 
        weight_decay=1e-6,      
        epsilon=1e-9,         
        clipnorm=0.05,      
    )
    
    model.compile(
        optimizer=optimizer,
        loss=weighted_mse,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    return model

class RangeSpecificMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir):
        super(RangeSpecificMetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.writer = tf.summary.create_file_writer(log_dir + '/range_metrics')
        self.range_mae = []
        self.range_rmse = []
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        
        y_pred = self.model.predict(x_val, verbose=0)
        
        mask = y_val <= 0.1
        if np.sum(mask) > 0:  
            y_val_range = y_val[mask]
            y_pred_range = y_pred[mask]
            
            mae_range = mean_absolute_error(y_val_range, y_pred_range)
            rmse_range = np.sqrt(mean_squared_error(y_val_range, y_pred_range))
            
            self.range_mae.append(mae_range)
            self.range_rmse.append(rmse_range)
            
            with self.writer.as_default():
                tf.summary.scalar('range_0_to_0.1_mae', mae_range, step=epoch)
                tf.summary.scalar('range_0_to_0.1_rmse', rmse_range, step=epoch)
            
            print(f"\nRange 0-0.1 - MAE: {mae_range:.6f}, RMSE: {rmse_range:.6f}")

def cosine_decay_with_warmup(epoch, lr):
    warmup_epochs = 5
    total_epochs = 1000
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

print("Loading and preparing data...")
data = pd.read_csv(data_path)

critical_range_data = data[(data["P"] >= 0.0004) & (data["P"] <= 0.0006)]  # Around 0.05%
low_range_data = data[data["P"] <= 0.1]

if len(critical_range_data) > 0:
    n_critical = len(critical_range_data) * 5  # 5x oversampling for critical range
    n_low = len(low_range_data) * 2      # 2x oversampling for general low range
    
    oversampled_dfs = []
    
    batch_size = 50000
    for i in range(0, n_critical, batch_size):
        current_batch_size = min(batch_size, n_critical - i)
        batch_indices = np.random.choice(critical_range_data.index, size=current_batch_size, replace=True)
        oversampled_batch = data.loc[batch_indices].copy()
        oversampled_dfs.append(oversampled_batch)
    
    for i in range(0, n_low, batch_size):
        current_batch_size = min(batch_size, n_low - i)
        batch_indices = np.random.choice(low_range_data.index, size=current_batch_size, replace=True)
        oversampled_batch = data.loc[batch_indices].copy()
        oversampled_dfs.append(oversampled_batch)
    
    data = pd.concat([data] + oversampled_dfs, ignore_index=True)
    print(f"Final data size after oversampling: {len(data)}")

print("Splitting data...")
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

print("Preparing feature matrices...")
X_train = train_data.drop(columns=["P", 'SNR']).astype('float64').values
y_train = train_data["P"].astype('float64').values
X_val = val_data.drop(columns=["P", 'SNR']).astype('float64').values
y_val = val_data["P"].astype('float64').values
X_test = test_data.drop(columns=["P", 'SNR']).astype('float64').values
y_test = test_data["P"].astype('float64').values

# Free up memory
del data, train_data, val_data, test_data
import gc
gc.collect()

# Normalize data
print("Normalizing data...")
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype('float64')
X_val = scaler.transform(X_val).astype('float64')
X_test = scaler.transform(X_test).astype('float64')

model = create_high_precision_model()
model.summary()

tensorboard_callback = CustomTensorBoard(log_dir=f'./logs/{version}')
range_metrics_callback = RangeSpecificMetricsCallback(
    validation_data=(X_val, y_val),
    log_dir=f'./logs/{version}'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    min_delta=1e-7,
    mode='min',
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(model_dir, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(performance_dir, 'training_log.csv')
)

callbacks_list = [
    early_stopping,
    model_checkpoint,
    lr_scheduler,
    tensorboard_callback,
    range_metrics_callback,
    csv_logger
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=128,  
    callbacks=callbacks_list,
    verbose=2
)

tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.keras'), compile=False)

history_file = os.path.join(performance_dir, 'training_log.csv')
history = pd.read_csv(history_file)


optimizer = optimizers.AdamW(
    learning_rate=0.0001,
    weight_decay=1e-5,
    epsilon=1e-8,
    clipnorm=0.1,
)

def weighted_mse(y_true, y_pred):
    mse = tf.square(y_true - y_pred)
    weights = tf.exp(-5.0 * y_true) + 1.0
    weighted_loss = mse * weights
    return tf.reduce_mean(weighted_loss)

model.compile(
    optimizer=optimizer,
    loss=weighted_mse,
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]
)

print("Evaluating model on test set...")
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {test_results[0]:.6f}")
print(f"Test MAE: {test_results[1]:.6f}")
print(f"Test RMSE: {test_results[2]:.6f}")

y_test_pred = model.predict(X_test).flatten()
y_test_pred *= 100
y_test *= 100
residuals = y_test - y_test_pred
relative_errors = (abs(y_test_pred - y_test) / y_test) * 100

test_results_df = pd.DataFrame({
    'Actual': y_test.round(6),
    'Predicted': y_test_pred.round(6),
    'Residuals': residuals.round(6),
    'RelativeError': np.round(relative_errors, 6)
})

test_results_file = os.path.join(performance_dir, f'test_results_{version}.csv')
test_results_df.to_csv(test_results_file, index=False)
print(f"Test results saved to {test_results_file}")

plots_dir = os.path.join(performance_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)


# Plot training history
plt.figure(figsize=(12, 10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{version}_training_history.png'), dpi=300)
plt.close()

# Plot MAE and RMSE metrics over epochs
plt.figure(figsize=(12, 10))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.plot(history.history['rmse'], label='Training RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Model Metrics Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f'{version}_metrics_history.png'), dpi=300)
plt.close()


stats = analyze_model_errors(
    actual=y_test,
    predicted=y_test_pred,
    save_dir=plots_dir,
    prefix=f'{version}_',
    threshold=0.2
)

def save_model_summary(model, version, performance_dir):
    """Generate and save a detailed model summary"""
    
    summary_path = os.path.join(performance_dir, f'{version}_architecture_summary.txt')
    
    # Redirect print statements to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
    
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Save original stdout and redirect to Logger
    original_stdout = sys.stdout
    sys.stdout = Logger(summary_path)
    
    try:
        print("\n" + "="*50)
        print("DETAILED MODEL ARCHITECTURE SUMMARY")
        print("="*50)
        
        print("\nGENERAL CONFIGURATION:")
        print("-"*30)
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print(f"Data Type: {model.dtype}")
        print(f"Total Parameters: {model.count_params():,}")
        
        print("\nLAYER ARCHITECTURE:")
        print("-"*30)
        model.summary()
        
        # Get optimizer config
        optimizer_config = model.optimizer.get_config()
        
        print("\nOPTIMIZER CONFIGURATION:")
        print("-"*30)
        print(f"Type: {model.optimizer.__class__.__name__}")
        for key, value in optimizer_config.items():
            print(f"{key}: {value}")
        
        print("\nLOSS FUNCTION:")
        print("-"*30)
        print("Type: Custom Weighted MSE")
        print("Components:")
        # Inspect the actual loss function used
        loss_fn = model.loss
        print(f"Loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else str(loss_fn)}")
        
        print("\nTRAINING CONFIGURATION:")
        print("-"*30)
        # These values are taken from the actual training configuration
        print(f"Batch size: {model.fit_args['batch_size'] if hasattr(model, 'fit_args') else '32 (default)'}")
        
        print("\nCALLBACKS:")
        print("-"*30)
        for callback in callbacks_list:
            print(f"- {callback.__class__.__name__}")
            if hasattr(callback, 'get_config'):
                config = callback.get_config()
                for key, value in config.items():
                    print(f"  {key}: {value}")
        
        print("\nDATA PREPROCESSING:")
        print("-"*30)
        print(f"Scaler type: {scaler.__class__.__name__}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        print("\nMONITORING METRICS:")
        print("-"*30)
        for metric in model.metrics:
            print(f"- {metric.name}")
        
        print("\nMODEL PERFORMANCE METRICS:")
        print("-"*30)
        if hasattr(model, 'history'):
            for metric_name, values in model.history.history.items():
                final_value = values[-1]
                best_value = min(values) if 'loss' in metric_name else max(values)
                print(f"{metric_name}:")
                print(f"  Final: {final_value:.6f}")
                print(f"  Best: {best_value:.6f}")
        
        print("\n" + "="*50 + "\n")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
    
    print(f"Model summary saved to: {summary_path}")

save_model_summary(model, version, performance_dir)