
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def load_tensor_data(file_path, chunk_size=1000, max_samples=None):
    """
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


def residual_block(x, filters, kernel_size=3, stride=1):

    shortcut = x
    
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def Tensor_model(input_shape=(500, 500, 1)):

    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(32, 7, strides=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = Dropout(0.2)(x)
    
    x = residual_block(x, 64, stride=3)
    x = residual_block(x, 64)
    x = Dropout(0.2)(x)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
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

def prepare_data(signals, P_values, test_size=0.2, random_state=42):
    print("Preparing data...")
    
    # Normalize signals
    signal_min = signals.min()
    signal_max = signals.max()
    signals_normalized = (signals - signal_min) / (signal_max - signal_min + 1e-8)
    
    # Add channel dimension
    signals_normalized = signals_normalized.reshape(signals_normalized.shape[0], 
                                                   signals_normalized.shape[1], 
                                                   signals_normalized.shape[2], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        signals_normalized, P_values, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test, epochs=10):

    print("Creating and training model...")
    
    model = Tensor_model(input_shape=X_train.shape[1:])
    
    print("Model Summary:")
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")
    
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    rpe = np.abs(y_test - y_pred) / (y_test + 1e-8) * 100
    
    print(f"\n=== Model Performance ===")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Mean Absolute Error: {mae:.8f}")
    print(f"Root Mean Squared Error: {rmse:.8f}")
    print(f"Mean Relative Percentage Error: {np.mean(rpe):.8f}")
    print(f"Actual P range: [{y_test.min():.6f}, {y_test.max():.6f}]")
    print(f"Predicted P range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    
    plt.figure(figsize=(12, 4))
    
    # Predictions vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred.flatten()
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    # Training history
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1,3,4)
    plt.hist(rpe, bins=20, alpha=0.7, edgecolor='black', color='red')
    plt.xlabel('Relative Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Relative Percentage Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_tensor_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

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

if __name__ == "__main__":
    print("=== Simple Tensor Training Test ===")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    signals, P_values, SNR_values,metadata = load_tensor_data(file_path='../Data_Creation/Training_Data/Sample_tensor.parquet', chunk_size=1000, max_samples=None)
    
    X_train, X_test, y_train, y_test = prepare_data(signals, P_values)
    
    # Train model
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=10)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    print(f"\n=== Final Summary ===")
    print(f"Training completed successfully!")
    print(f"Test MAE: {results['mae']:.8f}")
    print(f"Test RMSE: {results['rmse']:.8f}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    print("Results saved to: simple_tensor_test_results.png") 