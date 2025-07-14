import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Custom_Scripts.Variables import *
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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow.keras.backend as K
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def load_tensor_data_efficient(file_path, chunk_size=1000, max_samples=None):
    """
    Memory-efficient loading of tensor polarization data from a Parquet file.
    Enhanced version with better error handling for PyArrow issues.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
    chunk_size : int
        Number of samples to load at once
    max_samples : int, optional
        Maximum number of samples to load (for testing with large files)
        
    Returns:
    --------
    tuple
        (signals, P_values, SNR_values, metadata)
        - signals: numpy array of shape (n_samples, frequency_bins, phi_bins)
        - P_values: numpy array of polarization values
        - SNR_values: numpy array of SNR values (if available)
        - metadata: dictionary containing data structure information
    """
    print(f"Loading data from {file_path}")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Try different approaches to read the file
    print("Attempting to read file metadata...")
    
    # Approach 1: Try reading just metadata first using pyarrow directly
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
            
    except Exception as e1:
        print(f"PyArrow approach failed: {e1}")
        try:
            # Approach 2: Try reading with pandas head
            df_sample = pd.read_parquet(file_path).head(1)
            metadata = eval(df_sample.attrs.get('metadata', '{}'))
            print("Successfully read metadata using pandas head approach")
        except Exception as e2:
            print(f"Pandas approach failed: {e2}")
            try:
                # Approach 3: Try reading specific columns only
                df_sample = pd.read_parquet(file_path, columns=['P']).head(1)
                # Try to get metadata from a different approach
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                metadata_str = parquet_file.metadata.metadata.get(b'metadata', b'{}')
                metadata = eval(metadata_str.decode('utf-8'))
                print("Successfully read metadata using columns approach")
            except Exception as e3:
                print(f"Columns approach failed: {e3}")
                # Final fallback: create default metadata
                print("Using default metadata")
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
    
    if metadata['polarization_type'] != 'tensor':
        raise ValueError(f"Unsupported polarization type: {metadata['polarization_type']}")
    
    # Get total number of rows using a safe approach
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"Got row count using pyarrow: {total_rows}")
    except Exception as e:
        print(f"Error reading row count with pyarrow: {e}")
        try:
            df_count = pd.read_parquet(file_path, columns=['P'])
            total_rows = len(df_count)
            del df_count  # Clean up memory
            print(f"Got row count using pandas: {total_rows}")
        except Exception as e2:
            print(f"Error reading row count with pandas: {e2}")
            # Fallback: estimate from file size
            file_size_gb = os.path.getsize(file_path) / (1024**3)
            estimated_rows = int(file_size_gb * 1000000)  # Rough estimate
            print(f"Using estimated row count: {estimated_rows}")
            total_rows = estimated_rows
    
    if max_samples:
        total_rows = min(total_rows, max_samples)
    
    print(f"Total samples to load: {total_rows}")
    print(f"Frequency bins: {metadata['frequency_bins']}, Phi bins: {metadata['phi_bins']}")
    
    # Pre-allocate arrays
    signals = np.zeros((total_rows, metadata['frequency_bins'], metadata['phi_bins']), dtype=np.float32)
    P_values = np.zeros(total_rows, dtype=np.float32)
    SNR_values = None
    if 'SNR' in metadata.get('columns', []):
        SNR_values = np.zeros(total_rows, dtype=np.float32)
    
    # Load data in chunks using different strategies
    loaded_samples = 0
    
    # Strategy 1: Try pyarrow directly with row groups
    try:
        print("Using pyarrow direct approach...")
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        
        for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            if loaded_samples >= total_rows:
                break
            
            # Convert batch to pandas
            df_chunk = batch.to_pandas()
            current_chunk_size = len(df_chunk)
            
            if loaded_samples + current_chunk_size > total_rows:
                current_chunk_size = total_rows - loaded_samples
                df_chunk = df_chunk.head(current_chunk_size)
            
            print(f"Loading chunk {chunk_idx + 1} (pyarrow): samples {loaded_samples} to {loaded_samples + current_chunk_size - 1}")
            
            # Process signals
            for i, signal_data in enumerate(df_chunk['signal'].values):
                signal_array = np.array(signal_data, dtype=np.float32)
                signals[loaded_samples + i] = signal_array.reshape(metadata['frequency_bins'], metadata['phi_bins'])
            
            # Process P values
            P_values[loaded_samples:loaded_samples + current_chunk_size] = df_chunk['P'].values.astype(np.float32)
            
            # Process SNR values if available
            if SNR_values is not None and 'SNR' in df_chunk.columns:
                SNR_values[loaded_samples:loaded_samples + current_chunk_size] = df_chunk['SNR'].values.astype(np.float32)
            
            loaded_samples += current_chunk_size
            
            # Memory cleanup
            del df_chunk, batch
            gc.collect()
            
            print(f"Memory usage after chunk: {get_memory_usage():.2f} GB")
            
            if loaded_samples >= total_rows:
                break
                
    except Exception as e:
        print(f"PyArrow approach failed: {e}")
        print("Trying pandas chunking approach...")
        
        # Strategy 2: Try pandas chunking
        try:
            for chunk_idx, df_chunk in enumerate(pd.read_parquet(file_path, chunksize=chunk_size)):
                if loaded_samples >= total_rows:
                    break
                    
                current_chunk_size = len(df_chunk)
                if loaded_samples + current_chunk_size > total_rows:
                    current_chunk_size = total_rows - loaded_samples
                    df_chunk = df_chunk.head(current_chunk_size)
                
                print(f"Loading chunk {chunk_idx + 1}: samples {loaded_samples} to {loaded_samples + current_chunk_size - 1}")
                
                # Process signals
                for i, signal_data in enumerate(df_chunk['signal'].values):
                    signal_array = np.array(signal_data, dtype=np.float32)
                    signals[loaded_samples + i] = signal_array.reshape(metadata['frequency_bins'], metadata['phi_bins'])
                
                # Process P values
                P_values[loaded_samples:loaded_samples + current_chunk_size] = df_chunk['P'].values.astype(np.float32)
                
                # Process SNR values if available
                if SNR_values is not None and 'SNR' in df_chunk.columns:
                    SNR_values[loaded_samples:loaded_samples + current_chunk_size] = df_chunk['SNR'].values.astype(np.float32)
                
                loaded_samples += current_chunk_size
                
                # Memory cleanup
                del df_chunk
                gc.collect()
                
                print(f"Memory usage after chunk: {get_memory_usage():.2f} GB")
                
                if loaded_samples >= total_rows:
                    break
                    
        except Exception as e2:
            print(f"Pandas chunking also failed: {e2}")
            raise ValueError(f"All data loading strategies failed. PyArrow error: {e}, Pandas error: {e2}")
    
    print(f"Data loading completed. Final memory usage: {get_memory_usage():.2f} GB")
    print(f"Signal shape: {signals.shape}")
    print(f"P_values shape: {P_values.shape}")
    print(f"Sample signal range: [{signals[0].min():.6f}, {signals[0].max():.6f}]")
    
    return signals, P_values, SNR_values, metadata

def create_data_generator(signals, P_values, batch_size=32, augment=True, shuffle=True):
    """
    Create a memory-efficient data generator for training.
    
    Parameters:
    -----------
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target values
    batch_size : int
        Batch size for training
    augment : bool
        Whether to apply data augmentation
    shuffle : bool
        Whether to shuffle the data
        
    Yields:
    -------
    tuple
        (X_batch, y_batch) - Batches of data
    """
    n_samples = len(signals)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Load batch data
        X_batch = signals[batch_indices].copy()
        y_batch = P_values[batch_indices].copy()
        
        # Add channel dimension
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
        
        # Normalize signals to [0, 1] range
        X_batch = (X_batch - X_batch.min()) / (X_batch.max() - X_batch.min() + 1e-8)
        
        # Apply augmentation if requested
        if augment:
            X_batch, y_batch = augment_batch(X_batch, y_batch)
        
        yield X_batch, y_batch

def augment_batch(X_batch, y_batch, augmentation_prob=0.5):
    """
    Apply data augmentation to a batch of samples.
    
    Parameters:
    -----------
    X_batch : numpy.ndarray
        Batch of input signals
    y_batch : numpy.ndarray
        Batch of target values
    augmentation_prob : float
        Probability of applying augmentation to each sample
        
    Returns:
    --------
    tuple
        (X_augmented, y_augmented) - Augmented batch
    """
    X_aug = X_batch.copy()
    y_aug = y_batch.copy()
    
    for i in range(len(X_batch)):
        if np.random.random() < augmentation_prob:
            # Random horizontal flip
            if np.random.random() > 0.5:
                X_aug[i] = np.flip(X_aug[i], axis=1)
            
            # Random vertical flip
            if np.random.random() > 0.5:
                X_aug[i] = np.flip(X_aug[i], axis=0)
            
            # Random rotation (90, 180, or 270 degrees)
            if np.random.random() > 0.7:
                k = np.random.randint(1, 4)
                X_aug[i] = np.rot90(X_aug[i], k=k)
            
            # Add small random noise
            if np.random.random() > 0.5:
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_level, X_aug[i].shape)
                X_aug[i] = X_aug[i] + noise
                X_aug[i] = np.clip(X_aug[i], 0, 1)
            
            # Small random brightness adjustment
            if np.random.random() > 0.7:
                brightness_factor = np.random.uniform(0.9, 1.1)
                X_aug[i] = X_aug[i] * brightness_factor
                X_aug[i] = np.clip(X_aug[i], 0, 1)
    
    return X_aug, y_aug

def prepare_data_efficient(signals, P_values, test_size=0.2, random_state=42, 
                          chunk_size=1000, max_samples=None):
    """
    Memory-efficient data preparation for CNN training.
    
    Parameters:
    -----------
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target polarization values
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed for reproducibility
    chunk_size : int
        Size of chunks for processing
    max_samples : int, optional
        Maximum number of samples to use
        
    Returns:
    --------
    tuple
        (train_indices, test_indices, metadata) - Indices for train/test split and metadata
    """
    print(f"Preparing data efficiently...")
    print(f"Memory usage before preparation: {get_memory_usage():.2f} GB")
    
    # Limit samples if specified
    if max_samples and len(signals) > max_samples:
        print(f"Limiting to {max_samples} samples from {len(signals)} total")
        indices = np.random.choice(len(signals), max_samples, replace=False)
        signals = signals[indices]
        P_values = P_values[indices]
    
    # Create train/test split indices
    n_samples = len(signals)
    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # Calculate normalization parameters from training data only
    print("Calculating normalization parameters from training data...")
    train_signals = signals[train_indices]
    
    # Calculate global min/max for normalization
    signal_min = train_signals.min()
    signal_max = train_signals.max()
    
    print(f"Signal normalization range: [{signal_min:.6f}, {signal_max:.6f}]")
    print(f"P_values range: [{P_values.min():.6f}, {P_values.max():.6f}]")
    
    metadata = {
        'signal_min': signal_min,
        'signal_max': signal_max,
        'n_samples': n_samples,
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'input_shape': (signals.shape[1], signals.shape[2], 1)
    }
    
    print(f"Memory usage after preparation: {get_memory_usage():.2f} GB")
    
    return train_indices, test_indices, metadata

def create_efficient_data_generators(signals, P_values, train_indices, test_indices, 
                                   metadata, batch_size=32, augment_train=True):
    """
    Create efficient data generators for training and validation.
    
    Parameters:
    -----------
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target values
    train_indices : numpy.ndarray
        Training data indices
    test_indices : numpy.ndarray
        Test data indices
    metadata : dict
        Data metadata
    batch_size : int
        Batch size for training
    augment_train : bool
        Whether to augment training data
        
    Returns:
    --------
    tuple
        (train_gen, val_gen, steps_per_epoch, validation_steps) - Data generators and steps
    """
    def train_generator():
        while True:
            # Shuffle training indices
            shuffled_indices = train_indices.copy()
            np.random.shuffle(shuffled_indices)
            
            for start_idx in range(0, len(shuffled_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(shuffled_indices))
                batch_indices = shuffled_indices[start_idx:end_idx]
                
                # Load and process batch
                X_batch = signals[batch_indices].copy()
                y_batch = P_values[batch_indices].copy()
                
                # Add channel dimension and normalize
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
                X_batch = (X_batch - metadata['signal_min']) / (metadata['signal_max'] - metadata['signal_min'] + 1e-8)
                
                # Apply augmentation
                if augment_train:
                    X_batch, y_batch = augment_batch(X_batch, y_batch)
                
                yield X_batch, y_batch
    
    def val_generator():
        while True:
            for start_idx in range(0, len(test_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(test_indices))
                batch_indices = test_indices[start_idx:end_idx]
                
                # Load and process batch
                X_batch = signals[batch_indices].copy()
                y_batch = P_values[batch_indices].copy()
                
                # Add channel dimension and normalize
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
                X_batch = (X_batch - metadata['signal_min']) / (metadata['signal_max'] - metadata['signal_min'] + 1e-8)
                
                yield X_batch, y_batch
    
    steps_per_epoch = len(train_indices) // batch_size
    validation_steps = len(test_indices) // batch_size
    
    return train_generator(), val_generator(), steps_per_epoch, validation_steps

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Create a residual block with skip connections for better gradient flow.
    
    Parameters:
    -----------
    x : tensor
        Input tensor
    filters : int
        Number of filters in the convolutional layers
    kernel_size : int
        Size of the convolutional kernel
    stride : int
        Stride for the first convolution
        
    Returns:
    --------
    tensor
        Output tensor with residual connection
    """
    # Store the input for the skip connection
    shortcut = x
    
    # First convolution
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Second convolution
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If the input and output have different shapes, adjust the shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add the residual connection
    x = Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def adaptive_lr_schedule(epoch, initial_lr=0.0005):
    """
    Adaptive learning rate schedule that combines warmup, exponential decay, and plateau detection.
    
    Parameters:
    -----------
    epoch : int
        Current epoch
    initial_lr : float
        Initial learning rate
        
    Returns:
    --------
    float
        Learning rate for the current epoch
    """
    # Warmup phase (first 5 epochs)
    if epoch < 5:
        return initial_lr * (epoch + 1) / 5
    
    # Exponential decay phase
    decay_rate = 0.95
    decay_epochs = epoch - 5
    lr = initial_lr * (decay_rate ** decay_epochs)
    
    # Minimum learning rate
    min_lr = 1e-8
    return max(lr, min_lr)

def create_resnet_model(input_shape=(500, 500, 1), num_outputs=1):
    """
    Create a ResNet-style model with residual connections for high-precision regression.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (height, width, channels)
    num_outputs : int
        Number of output neurons (1 for regression)
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled ResNet model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = Dropout(0.2)(x)
    
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)
    x = Dropout(0.2)(x)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = Dropout(0.2)(x)
    
    # Global average pooling to reduce spatial dimensions
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with residual connections
    dense_input = x
    
    # First dense block
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second dense block with residual connection
    dense_residual = x
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, dense_residual])  # Residual connection
    x = Dropout(0.3)(x)
    
    # Final dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer - use linear activation for regression
    outputs = Dense(num_outputs, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom loss function for high-precision regression
    def precision_loss(y_true, y_pred):
        """
        Custom loss function that emphasizes high precision for small differences.
        Combines MSE with Huber loss for better handling of outliers.
        """
        # Standard MSE loss
        mse_loss = K.mean(K.square(y_true - y_pred))
        
        # Huber loss for better handling of outliers
        delta = 0.1
        abs_diff = K.abs(y_true - y_pred)
        quadratic = K.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        huber_loss = K.mean(0.5 * K.square(quadratic) + delta * linear)
        
        # Range penalty to encourage full range predictions
        range_penalty = K.mean(K.square(y_pred - 0.5))  # Encourage predictions around 0.5
        
        # Soft boundary penalty (very minimal)
        boundary_penalty = K.mean(K.square(K.relu(y_pred - 1.0))) + K.mean(K.square(K.relu(-y_pred)))
        
        # Combine losses with weights (focus on MSE and range)
        total_loss = 0.8 * mse_loss + 0.15 * huber_loss + 0.03 * range_penalty + 0.02 * boundary_penalty
        
        return total_loss
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=precision_loss,
        metrics=['mae', 'mse']
    )
    
    return model

def train_resnet_model_efficient(signals, P_values, train_indices, test_indices, metadata, 
                                model_save_path=None, batch_size=32, epochs=100):
    """
    Train the ResNet model with memory-efficient data generators.
    
    Parameters:
    -----------
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target values
    train_indices : numpy.ndarray
        Training data indices
    test_indices : numpy.ndarray
        Test data indices
    metadata : dict
        Data metadata
    model_save_path : str, optional
        Path to save the trained model
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
        
    Returns:
    --------
    tuple
        (model, history) - Trained model and training history
    """
    print("Creating ResNet model...")
    model = create_resnet_model(input_shape=metadata['input_shape'])
    
    # Print model summary
    print("Model Summary:")
    model.summary()
    
    # Create data generators
    train_gen, val_gen, steps_per_epoch, validation_steps = create_efficient_data_generators(
        signals, P_values, train_indices, test_indices, metadata, 
        batch_size=batch_size, augment_train=True
    )
    
    # Define comprehensive callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True, 
            min_delta=1e-6,
            verbose=1
        ),
        
        ModelCheckpoint(
            filepath=model_save_path if model_save_path else 'best_resnet_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        LearningRateScheduler(
            lambda epoch: adaptive_lr_schedule(epoch, initial_lr=0.0005),
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-8,
            verbose=1,
            cooldown=3
        )
    ]

    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(test_indices)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Input shape: {metadata['input_shape']}")
    print(f"Memory usage before training: {get_memory_usage():.2f} GB")
    
    # Train the model
    print(f"\nStarting training with initial learning rate: {model.optimizer.learning_rate.numpy():.6f}")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print final learning rate
    final_lr = model.optimizer.learning_rate.numpy()
    print(f"\nTraining completed. Final learning rate: {final_lr:.8f}")
    print(f"Memory usage after training: {get_memory_usage():.2f} GB")
    
    return model, history

def evaluate_model_efficient(model, signals, P_values, test_indices, metadata, batch_size=32):
    """
    Memory-efficient model evaluation with detailed metrics and visualizations.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target values
    test_indices : numpy.ndarray
        Test data indices
    metadata : dict
        Data metadata
    batch_size : int
        Batch size for evaluation
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("Evaluating model efficiently...")
    print(f"Memory usage before evaluation: {get_memory_usage():.2f} GB")
    
    # Make predictions in batches
    y_pred_list = []
    y_test_list = []
    
    for start_idx in range(0, len(test_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        
        # Load and process batch
        X_batch = signals[batch_indices].copy()
        y_batch = P_values[batch_indices].copy()
        
        # Add channel dimension and normalize
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
        X_batch = (X_batch - metadata['signal_min']) / (metadata['signal_max'] - metadata['signal_min'] + 1e-8)
        
        # Make predictions
        batch_pred = model.predict(X_batch, verbose=0)
        
        y_pred_list.append(batch_pred.flatten())
        y_test_list.append(y_batch)
        
        # Memory cleanup
        del X_batch, y_batch, batch_pred
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
    
    os.makedirs('Tensor_Results', exist_ok=True)
    results_df.to_csv('Tensor_Results/detailed_evaluation_results.csv', index=False)
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Predictions vs Actual
    axes[0, 0].scatter(y_test_flat, y_pred_flat, alpha=0.6)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Absolute errors histogram
    axes[0, 2].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Absolute Errors')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Absolute Errors Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Percentage errors histogram
    axes[1, 0].hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Percentage Errors (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Percentage Errors Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Residuals vs Predicted
    axes[1, 1].scatter(y_pred_flat, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Precision analysis
    precision_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    precision_values = [np.sum(abs_errors < level) / len(abs_errors) for level in precision_levels]
    axes[1, 2].bar(range(len(precision_levels)), precision_values)
    axes[1, 2].set_xlabel('Precision Level')
    axes[1, 2].set_ylabel('Fraction of Predictions')
    axes[1, 2].set_title('Precision Analysis')
    axes[1, 2].set_xticks(range(len(precision_levels)))
    axes[1, 2].set_xticklabels([f'{level:.0e}' for level in precision_levels])
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Tensor_Results/evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Memory usage after evaluation: {get_memory_usage():.2f} GB")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'precision_5_decimal': precision_5_decimal,
        'precision_4_decimal': precision_4_decimal,
        'precision_3_decimal': precision_3_decimal,
        'mean_percentage_error': np.mean(percentage_errors)
    }

def evaluate_precision(y_true, y_pred):
    """Evaluate precision at different decimal places."""
    abs_errors = np.abs(y_true - y_pred)
    
    precision_metrics = {}
    for i in range(1, 7):  # 1 to 6 decimal places
        threshold = 10**(-i)
        precision = np.sum(abs_errors < threshold) / len(abs_errors)
        precision_metrics[f'precision_{i}_decimal'] = precision
    
    return precision_metrics

def comprehensive_testing_efficient(model, signals, P_values, test_indices, metadata, batch_size=32):
    """
    Memory-efficient comprehensive testing of the model with detailed precision analysis.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    signals : numpy.ndarray
        Input signals
    P_values : numpy.ndarray
        Target values
    test_indices : numpy.ndarray
        Test data indices
    metadata : dict
        Data metadata
    batch_size : int
        Batch size for evaluation
        
    Returns:
    --------
    dict
        Comprehensive test results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL TESTING (EFFICIENT)")
    print("="*60)
    
    # Make predictions in batches
    y_pred_list = []
    y_test_list = []
    
    for start_idx in range(0, len(test_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(test_indices))
        batch_indices = test_indices[start_idx:end_idx]
        
        # Load and process batch
        X_batch = signals[batch_indices].copy()
        y_batch = P_values[batch_indices].copy()
        
        # Add channel dimension and normalize
        X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
        X_batch = (X_batch - metadata['signal_min']) / (metadata['signal_max'] - metadata['signal_min'] + 1e-8)
        
        # Make predictions
        batch_pred = model.predict(X_batch, verbose=0)
        
        y_pred_list.append(batch_pred.flatten())
        y_test_list.append(y_batch)
        
        # Memory cleanup
        del X_batch, y_batch, batch_pred
        gc.collect()
    
    # Combine results
    y_pred_flat = np.concatenate(y_pred_list)
    y_test_flat = np.concatenate(y_test_list)
    
    # Calculate basic metrics
    mse = np.mean((y_test_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_test_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    
    print(f"\n=== Basic Performance Metrics ===")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Mean Absolute Error: {mae:.8f}")
    print(f"Root Mean Squared Error: {rmse:.8f}")
    print(f"Actual P range: [{y_test_flat.min():.6f}, {y_test_flat.max():.6f}]")
    print(f"Predicted P range: [{y_pred_flat.min():.6f}, {y_pred_flat.max():.6f}]")
    
    # Evaluate precision at different decimal places
    print(f"\n=== Precision Analysis ===")
    precision_metrics = evaluate_precision(y_test_flat, y_pred_flat)
    
    for i in range(1, 6):  # Show up to 5 decimal places
        key = f'precision_{i}_decimal'
        precision = precision_metrics[key]
        threshold = 10**(-i)
        print(f"Precision within {threshold:.0e} ({i} decimal place): {precision:.4f} ({precision*100:.2f}%)")
    
    # Show some example predictions
    print(f"\n=== Example Predictions ===")
    print("Actual\t\tPredicted\t\tAbsolute Error\t\tRelative Error (%)")
    print("-" * 80)
    
    for i in range(min(10, len(y_test_flat))):
        actual = y_test_flat[i]
        predicted = y_pred_flat[i]
        abs_error = abs(actual - predicted)
        rel_error = abs_error / actual * 100 if actual > 0 else 0
        print(f"{actual:.6f}\t\t{predicted:.6f}\t\t{abs_error:.8f}\t\t{rel_error:.4f}%")
    
    # Create comprehensive visualization
    print(f"\n=== Creating Comprehensive Visualization ===")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Predictions vs Actual
    axes[0, 0].scatter(y_test_flat, y_pred_flat, alpha=0.7, s=50)
    axes[0, 0].plot([y_test_flat.min(), y_test_flat.max()], 
                    [y_test_flat.min(), y_test_flat.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predictions vs Actual Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_test_flat - y_pred_flat
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Absolute Errors
    abs_errors = np.abs(residuals)
    axes[0, 2].hist(abs_errors, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Absolute Errors')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Absolute Errors Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision Analysis
    precision_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    precision_values = [precision_metrics[f'precision_{i}_decimal'] for i in range(1, 6)]
    axes[1, 0].bar(range(len(precision_levels)), precision_values, alpha=0.7)
    axes[1, 0].set_xlabel('Precision Level')
    axes[1, 0].set_ylabel('Fraction of Predictions')
    axes[1, 0].set_title('Precision Analysis')
    axes[1, 0].set_xticks(range(len(precision_levels)))
    axes[1, 0].set_xticklabels([f'{level:.0e}' for level in precision_levels])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Residuals vs Predicted
    axes[1, 1].scatter(y_pred_flat, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Error Distribution by Value Range
    value_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    range_errors = []
    range_labels = []
    
    for low, high in value_ranges:
        mask = (y_test_flat >= low) & (y_test_flat < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(abs_errors[mask])
            range_errors.append(range_mae)
            range_labels.append(f'{low:.1f}-{high:.1f}')
    
    if range_errors:
        axes[1, 2].bar(range_labels, range_errors, alpha=0.7)
        axes[1, 2].set_xlabel('Value Range')
        axes[1, 2].set_ylabel('Mean Absolute Error')
        axes[1, 2].set_title('Error by Value Range')
        axes[1, 2].grid(True, alpha=0.3)
        plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('Tensor_Results/comprehensive_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'actual': y_test_flat,
        'predicted': y_pred_flat,
        'residuals': residuals,
        'abs_error': abs_errors,
        'percentage_error': np.abs(residuals / (y_test_flat + 1e-8)) * 100
    })
    
    os.makedirs('Tensor_Results', exist_ok=True)
    results_df.to_csv('Tensor_Results/comprehensive_test_results.csv', index=False)
    
    print(f"\n=== Summary ===")
    print(f"The improved ResNet model achieves:")
    print(f"- MSE: {mse:.8f}")
    print(f"- MAE: {mae:.8f}")
    print(f"- RMSE: {rmse:.8f}")
    print(f"- Precision within 1e-5: {precision_metrics['precision_5_decimal']:.4f}")
    
    # Check if we meet the 5-decimal precision goal
    if precision_metrics['precision_5_decimal'] > 0.5:  # More than 50% of predictions within 1e-5
        print(f"✅ SUCCESS: Model achieves high precision (5 decimal places) for majority of predictions!")
    else:
        print(f"⚠️  Model needs further optimization for consistent 5-decimal precision.")
    
    print(f"\nResults saved to Tensor_Results/ directory")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'precision_metrics': precision_metrics
    }

# Main execution
if __name__ == "__main__":
    print("=== High-Precision Tensor Polarization Prediction (Memory Efficient) ===")
    
    # Configuration parameters
    CHUNK_SIZE = 1000  # Number of samples to load at once
    MAX_SAMPLES = None  # Set to a number to limit samples for testing
    BATCH_SIZE = 32
    EPOCHS = 100
    TEST_SIZE = 0.3
    
    print(f"Configuration:")
    print(f"- Chunk size: {CHUNK_SIZE}")
    print(f"- Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Test size: {TEST_SIZE}")
    
    # Load data efficiently
    print("\n=== Loading Data ===")
    signals, P_values, SNR_values, metadata = load_tensor_data_efficient(
        '../Data_Creation/Training_Data/Sample_tensor.parquet',
        chunk_size=CHUNK_SIZE,
        max_samples=MAX_SAMPLES
    )
    
    print(f"Data loaded - Signals shape: {signals.shape}")
    print(f"P_values shape: {P_values.shape}")
    print(f"Metadata: {metadata}")
    
    # Prepare data efficiently
    print("\n=== Preparing Data ===")
    train_indices, test_indices, data_metadata = prepare_data_efficient(
        signals, P_values, 
        test_size=TEST_SIZE, 
        random_state=42,
        max_samples=MAX_SAMPLES
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Train the ResNet model efficiently
    print("\n=== Training ResNet Model ===")
    model, history = train_resnet_model_efficient(
        signals, P_values, train_indices, test_indices, data_metadata,
        batch_size=BATCH_SIZE, epochs=EPOCHS
    )
    
    # Evaluate the model efficiently
    print("\n=== Model Evaluation ===")
    val_metrics = evaluate_model_efficient(
        model, signals, P_values, test_indices, data_metadata, batch_size=BATCH_SIZE
    )
    
    # Comprehensive testing
    test_results = comprehensive_testing_efficient(
        model, signals, P_values, test_indices, data_metadata, batch_size=BATCH_SIZE
    )
    
    # Plot training history
    print("\n=== Training History ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE
    axes[1, 0].plot(history.history['mse'], label='Training MSE', linewidth=2)
    axes[1, 0].plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
    axes[1, 0].set_title('Model MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', linewidth=2)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate History\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('Tensor_Results/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Training completed successfully!")
    print(f"Final validation loss: {min(history.history['val_loss']):.8f}")
    print(f"Final validation MAE: {min(history.history['val_mae']):.8f}")
    print(f"Test set MAE: {test_results['mae']:.8f}")
    print(f"Test set precision within 1e-5: {test_results['precision_metrics']['precision_5_decimal']:.4f}")
    print(f"Model saved as: best_resnet_model.keras")
    print(f"All results saved to Tensor_Results/ directory")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    print("="*60)















