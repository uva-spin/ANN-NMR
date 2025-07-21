
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
    Add, Input, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# Enhanced GPU Memory Management
def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to 80% of available GPU memory
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 8GB limit
                # )
            print(f"Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
    else:
        print("No GPUs found, using CPU")

# Configure GPU memory at import
configure_gpu_memory()

# Enhanced environment variables for memory optimization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Enable mixed precision for better performance and memory usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class HeUniformConv2D(Conv2D):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding='valid', 
                 data_format=None, 
                 dilation_rate=(1, 1), 
                 activation=None, 
                 use_bias=True, 
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 **kwargs):
        
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer='he_uniform',
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

version = "Inception_Block_V1_TF_DataPipeline_Memory_Optimized"

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_metadata_from_parquet(file_path):
    """Extract metadata from parquet file."""
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        
        # Try to get metadata from file metadata
        metadata_str = parquet_file.metadata.metadata.get(b'metadata', b'{}')
        if metadata_str:
            metadata = eval(metadata_str.decode('utf-8'))
            print("Successfully read metadata using pyarrow direct approach")
        else:
            # Fallback: read first row to get metadata
            first_batch = next(parquet_file.iter_batches(batch_size=1))
            df_sample = first_batch.to_pandas()
            metadata = eval(df_sample.attrs.get('metadata', '{}'))
            print("Successfully read metadata from first row")
            
    except Exception as e:
        print(f"Error reading metadata: {e}")
        # Default metadata
        metadata = {
            'polarization_type': 'tensor',
            'mode': 'deuteron',
            'frequency_bins': 500,
            'phi_bins': 500,
            'signal_shape': (500, 500, 1),
            'is_flattened': True,
            'frequency_range': (-3, 3),
            'phi_range': (0, 180),
            'num_samples': 0,
            'columns': ['signal', 'P', 'SNR']
        }
    
    # Ensure metadata has required fields
    if 'polarization_type' not in metadata:
        metadata['polarization_type'] = 'tensor'
    if 'frequency_bins' not in metadata:
        metadata['frequency_bins'] = 500
    if 'phi_bins' not in metadata:
        metadata['phi_bins'] = 500
    if 'columns' not in metadata:
        metadata['columns'] = ['signal', 'P', 'SNR']
    
    return metadata

def create_memory_optimized_dataset(file_path, batch_size=16,
                                   max_samples=None, metadata=None):
    """
    Create memory-optimized TensorFlow dataset with aggressive memory management.
    """
    print(f"Creating memory-optimized dataset from {file_path}")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Get metadata if not provided
    if metadata is None:
        metadata = get_metadata_from_parquet(file_path)
    
    if metadata['polarization_type'] != 'tensor':
        raise ValueError(f"Unsupported polarization type: {metadata['polarization_type']}")
    
    # Get total number of rows
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows in file: {total_rows}")
    except Exception as e:
        print(f"Error reading row count: {e}")
        total_rows = max_samples or 100000
    
    if max_samples:
        total_rows = min(total_rows, max_samples)
    
    print(f"Processing {total_rows} samples with batch_size={batch_size}")
    
    # Memory-optimized generator function
    def load_parquet_chunk():
        """Memory-optimized generator function."""
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            
            # Use smaller batch size for memory efficiency
            for batch in parquet_file.iter_batches(batch_size=500):  # Reduced from 2000
                df_batch = batch.to_pandas()
                
                for _, row in df_batch.iterrows():
                    # Process signal with memory-efficient operations
                    signal_data = np.array(row['signal'], dtype=np.float32)
                    signal = signal_data.reshape(metadata['frequency_bins'], metadata['phi_bins'])
                    
                    # Add channel dimension
                    signal = np.expand_dims(signal, axis=-1)
                    
                    # Get P value
                    p_value = float(row['P'])
                    
                    yield signal, p_value
                    
                # Force garbage collection after each batch
                del df_batch
                gc.collect()
                    
        except Exception as e:
            print(f"Error in data loading: {e}")
            return
    
    # Create TensorFlow dataset with memory optimizations
    dataset = tf.data.Dataset.from_generator(
        load_parquet_chunk,
        output_signature=(
            tf.TensorSpec(shape=(metadata['frequency_bins'], metadata['phi_bins'], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    # Apply memory-optimized transformations
    dataset = dataset.take(total_rows)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)  # Reduced prefetch buffer
    
    print(f"Memory-optimized dataset created. Memory usage: {get_memory_usage():.2f} GB")
    
    return dataset, metadata, total_rows

def create_memory_optimized_train_test_datasets(file_path, test_size=0.2, batch_size=16, max_samples=None, random_state=42):
    """
    Create memory-optimized train and test datasets.
    """
    print("Creating memory-optimized train/test datasets...")
    
    # Get metadata first
    metadata = get_metadata_from_parquet(file_path)
    
    # Get total samples
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error reading row count: {e}")
        total_rows = max_samples or 100000
    
    if max_samples:
        total_rows = min(total_rows, max_samples)
    
    # Calculate split indices
    train_size = int(total_rows * (1 - test_size))
    test_size_actual = total_rows - train_size
    
    print(f"Total samples: {total_rows}")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {test_size_actual}")
    
    # Create full dataset with memory optimizations
    full_dataset, _, _ = create_memory_optimized_dataset(
        file_path, batch_size=batch_size,
        max_samples=max_samples, metadata=metadata
    )
    
    # Split into train and test
    train_dataset = full_dataset.take(train_size // batch_size)
    test_dataset = full_dataset.skip(train_size // batch_size)
    
    # Apply final optimizations
    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)
    
    print(f"Memory-optimized datasets created. Memory usage: {get_memory_usage():.2f} GB")
    
    return train_dataset, test_dataset, metadata, total_rows

def simple_conv_block(x, filters, kernel_size=3, pool_size=2):
    """Simple convolutional block with pooling."""
    x = HeUniformConv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size, padding='same')(x)
    return x

def Tensor_model_memory_optimized(input_shape=(500, 500, 1)):
    """Memory-optimized tensor model with reduced complexity."""
    
    inputs = Input(shape=input_shape)
    
    # More aggressive downsampling to reduce memory usage
    x = simple_conv_block(inputs, 16, kernel_size=7, pool_size=8)  # 500x500 -> 63x63
    x = Dropout(0.1)(x)
    
    x = simple_conv_block(x, 32, kernel_size=5, pool_size=4)      # 63x63 -> 16x16
    x = Dropout(0.1)(x)
    
    x = simple_conv_block(x, 64, kernel_size=3, pool_size=2)      # 16x16 -> 8x8
    x = Dropout(0.2)(x)
    
    # Final feature extraction
    x = HeUniformConv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, padding='same')(x)  # 8x8 -> 4x4
    x = Dropout(0.2)(x)
    
    # Global pooling and classification
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def cosine_decay_schedule(epoch, initial_lr=0.001, decay_steps=500, warmup_steps=50):
    if epoch < warmup_steps:
        # Linear warmup
        return float(initial_lr * (epoch + 1) / warmup_steps)
    else:
        # Cosine decay
        progress = (epoch - warmup_steps) / (decay_steps - warmup_steps)
        return float(initial_lr * 0.5 * (1 + np.cos(np.pi * progress)))

def train_model_with_memory_optimization(train_dataset, test_dataset, epochs=100):
    """Train model with memory optimization."""
    
    print("Creating and training memory-optimized model...")
    print(f"Memory usage before model creation: {get_memory_usage():.2f} GB")
    
    # Clear GPU memory before creating model
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Get input shape from dataset
    for signals, _ in train_dataset.take(1):
        input_shape = signals.shape[1:]
        break
    
    model = Tensor_model_memory_optimized(input_shape=input_shape)
    
    print("Model Summary:")
    model.summary()
    
    # Memory-optimized callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
        LearningRateScheduler(lambda epoch: cosine_decay_schedule(epoch, initial_lr=0.001, decay_steps=epochs, warmup_steps=25))
    ]
    
    print(f"Training with memory-optimized datasets...")
    print(f"Steps per epoch: {len(list(train_dataset.as_numpy_iterator()))}")

    # Use memory-optimized training
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    print(f"Memory usage after training: {get_memory_usage():.2f} GB")
    
    return model, history

def evaluate_model_memory_optimized(model, test_dataset, metadata):
    """Evaluate model with memory optimization."""
    
    print("Evaluating model with memory optimization...")
    
    # Collect predictions and actual values with memory-efficient approach
    y_pred_list = []
    y_test_list = []
    
    for signals, labels in test_dataset:
        # Make predictions with smaller batch size
        batch_pred = model.predict(signals, verbose=0, batch_size=8)  
        
        y_pred_list.append(batch_pred.flatten())
        y_test_list.append(labels.numpy())
        
        # Force garbage collection
        gc.collect()
    
    # Combine results
    y_pred_flat = np.concatenate(y_pred_list)
    y_test_flat = np.concatenate(y_test_list)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = np.mean(np.abs(y_test_flat - y_pred_flat))
    residuals = y_test_flat - y_pred_flat
    
    # Calculate precision metrics
    abs_errors = np.abs(residuals)
    precision_5_decimal = np.sum(abs_errors < 1e-5) / len(abs_errors)
    precision_4_decimal = np.sum(abs_errors < 1e-4) / len(abs_errors)
    precision_3_decimal = np.sum(abs_errors < 1e-3) / len(abs_errors)
    
    # Calculate percentage errors
    percentage_errors = np.abs(residuals / (y_test_flat + 1e-8)) * 100
    
    print(f"\n=== Model Evaluation Results ===")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Mean Absolute Error: {mae:.8f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.8f}")
    print(f"Actual P range: [{y_test_flat.min():.6f}, {y_test_flat.max():.6f}]")
    print(f"Predicted P range: [{y_pred_flat.min():.6f}, {y_pred_flat.max():.6f}]")
    print(f"Mean Percentage Error: {np.mean(percentage_errors):.4f}%")
    print(f"Precision within 1e-5: {precision_5_decimal:.4f}")
    print(f"Precision within 1e-4: {precision_4_decimal:.4f}")
    print(f"Precision within 1e-3: {precision_3_decimal:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'actual': y_test_flat,
        'predicted': y_pred_flat,
        'residuals': residuals,
        'abs_error': abs_errors,
        'percentage_error': percentage_errors
    })
    
    os.makedirs(f'Tensor_Results_{version}', exist_ok=True)
    results_df.to_csv(f'Tensor_Results_{version}/detailed_evaluation_results.csv', index=False)
    
    # Create simplified visualizations to save memory
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Predictions vs Actual
    axes[0].scatter(y_test_flat, y_pred_flat, alpha=0.6, s=10)  # Reduced point size
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predictions vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')  # Reduced bins
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Tensor_Results_{version}/evaluation_plots.png', dpi=150, bbox_inches='tight')  # Reduced DPI
    plt.close()  # Close figure to free memory
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'precision_5_decimal': precision_5_decimal,
        'precision_4_decimal': precision_4_decimal,
        'precision_3_decimal': precision_3_decimal,
        'mean_percentage_error': np.mean(percentage_errors)
    }

if __name__ == "__main__":
    print("=== Memory-Optimized TensorFlow Training Test ===")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Create results directory
    os.makedirs(f'Tensor_Results_{version}', exist_ok=True)
    
    # Create memory-optimized train and test datasets
    train_dataset, test_dataset, metadata, total_samples = create_memory_optimized_train_test_datasets(
        file_path='/home/ptgroup/Documents/Devin/Big_Data/Tensor_Data_100K.parquet',
        test_size=0.2,
        batch_size=8,  # Drastically reduced batch size
        max_samples=5000  # Reduced sample count for testing
    )
    
    print(f"Memory-optimized datasets created. Memory usage: {get_memory_usage():.2f} GB")
    print(f"Total samples: {total_samples}")
    
    # Train model with memory optimization
    model, history = train_model_with_memory_optimization(train_dataset, test_dataset, epochs=50)  # Reduced epochs
    
    # Evaluate model with memory optimization
    results = evaluate_model_memory_optimized(model, test_dataset, metadata)
    
    print(f"\n=== Final Summary ===")
    print(f"Training completed successfully!")
    print(f"Test MAE: {results['mae']:.8f}")
    print(f"Test RMSE: {results['rmse']:.8f}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    print(f"Results saved to: Tensor_Results_{version}/") 