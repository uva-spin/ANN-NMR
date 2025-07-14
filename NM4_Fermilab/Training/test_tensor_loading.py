#!/usr/bin/env python3
"""
Test script for tensor data loading to diagnose the PyArrow list index overflow error.
"""

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

def test_parquet_file(file_path, max_samples=100):
    """
    Test function to diagnose Parquet file issues.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
    max_samples : int
        Maximum number of samples to test with
    """
    print(f"Testing Parquet file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Test 1: Try to read metadata
    print("\n=== Test 1: Reading Metadata ===")
    try:
        # Try reading just the first row
        df_sample = pd.read_parquet(file_path).head(1)
        print("✓ Successfully read first row")
        
        if 'metadata' in df_sample.attrs:
            metadata = eval(df_sample.attrs['metadata'])
            print(f"✓ Metadata found: {metadata}")
        else:
            print("✗ No metadata found in attrs")
            
    except Exception as e:
        print(f"✗ Failed to read first row: {e}")
        return False
    
    # Test 2: Try to read just P column
    print("\n=== Test 2: Reading P Column ===")
    try:
        df_p = pd.read_parquet(file_path, columns=['P']).head(max_samples)
        print(f"✓ Successfully read P column: {len(df_p)} rows")
        print(f"P range: [{df_p['P'].min():.6f}, {df_p['P'].max():.6f}]")
    except Exception as e:
        print(f"✗ Failed to read P column: {e}")
        return False
    
    # Test 3: Try to read signal column for a few samples
    print("\n=== Test 3: Reading Signal Column ===")
    try:
        df_signal = pd.read_parquet(file_path, columns=['signal']).head(5)
        print(f"✓ Successfully read signal column: {len(df_signal)} rows")
        
        # Check signal structure
        first_signal = df_signal['signal'].iloc[0]
        if isinstance(first_signal, list):
            print(f"Signal is list with {len(first_signal)} elements")
            signal_array = np.array(first_signal)
            print(f"Signal array shape: {signal_array.shape}")
        else:
            print(f"Signal type: {type(first_signal)}")
            
    except Exception as e:
        print(f"✗ Failed to read signal column: {e}")
        return False
    
    # Test 4: Try to read with different engines
    print("\n=== Test 4: Testing Different Engines ===")
    engines = ['pyarrow']  # Remove fastparquet since it's not installed
    
    for engine in engines:
        try:
            df_test = pd.read_parquet(file_path, engine=engine).head(10)
            print(f"✓ {engine} engine works: {len(df_test)} rows")
        except Exception as e:
            print(f"✗ {engine} engine failed: {e}")
    
    # Test 5: Try pyarrow directly
    print("\n=== Test 5: PyArrow Direct Access ===")
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        print(f"✓ PyArrow file opened successfully")
        print(f"Number of rows: {parquet_file.metadata.num_rows}")
        print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
        print(f"Schema: {parquet_file.schema}")
        
        # Try reading first batch
        first_batch = next(parquet_file.iter_batches(batch_size=5))
        print(f"✓ First batch read successfully: {len(first_batch)} rows")
        
    except Exception as e:
        print(f"✗ PyArrow direct access failed: {e}")
        return False
    
    print(f"\nFinal memory usage: {get_memory_usage():.2f} GB")
    return True

def create_small_test_dataset():
    """
    Create a small test dataset to verify the training pipeline works.
    """
    print("\n=== Creating Small Test Dataset ===")
    
    # Create a simple tensor dataset
    num_samples = 1000
    frequency_bins = 500
    phi_bins = 500
    
    print(f"Creating {num_samples} samples with shape ({frequency_bins}, {phi_bins})")
    
    # Generate random signals
    signals = np.random.randn(num_samples, frequency_bins, phi_bins).astype(np.float32)
    
    # Generate random P values between 0 and 1
    P_values = np.random.uniform(0, 1, num_samples).astype(np.float32)
    
    # Generate random SNR values
    SNR_values = np.random.uniform(10, 50, num_samples).astype(np.float32)
    
    # Create DataFrame
    df = pd.DataFrame({
        'signal': [sig.flatten() for sig in signals],  # Flatten for storage
        'P': P_values,
        'SNR': SNR_values
    })
    
    # Add metadata
    metadata = {
        'polarization_type': 'tensor',
        'mode': 'deuteron',
        'frequency_bins': frequency_bins,
        'phi_bins': phi_bins,
        'signal_shape': (frequency_bins, phi_bins, 1),
        'is_flattened': True,
        'frequency_range': (-3, 3),
        'phi_range': (0, 180),
        'num_samples': num_samples
    }
    
    df.attrs['metadata'] = str(metadata)
    
    # Save to Parquet
    test_file_path = 'test_tensor_data.parquet'
    df.to_parquet(test_file_path, engine='pyarrow', compression='snappy')
    
    print(f"✓ Test dataset saved to {test_file_path}")
    print(f"File size: {os.path.getsize(test_file_path) / (1024**2):.2f} MB")
    
    return test_file_path

def test_training_pipeline(test_file_path):
    """
    Test the complete training pipeline with the small dataset.
    """
    print("\n=== Testing Training Pipeline ===")
    
    # Import the loading function from Tensor.py
    from Tensor import load_tensor_data_efficient, prepare_data_efficient, train_resnet_model_efficient
    
    # Load data
    print("Loading test data...")
    signals, P_values, SNR_values, metadata = load_tensor_data_efficient(
        test_file_path, chunk_size=100, max_samples=500
    )
    
    print(f"Data loaded successfully:")
    print(f"- Signals shape: {signals.shape}")
    print(f"- P_values shape: {P_values.shape}")
    print(f"- Metadata: {metadata}")
    
    # Prepare data
    print("Preparing data...")
    train_indices, test_indices, data_metadata = prepare_data_efficient(
        signals, P_values, test_size=0.2, random_state=42, max_samples=500
    )
    
    print(f"Data prepared:")
    print(f"- Training samples: {len(train_indices)}")
    print(f"- Test samples: {len(test_indices)}")
    
    # Train model (with reduced epochs for testing)
    print("Training model...")
    model, history = train_resnet_model_efficient(
        signals, P_values, train_indices, test_indices, data_metadata,
        batch_size=16, epochs=5  # Reduced for testing
    )
    
    print("✓ Training pipeline completed successfully!")
    return True

if __name__ == "__main__":
    print("=== Tensor Data Loading Test ===")
    
    # Test the original file
    original_file = '../Data_Creation/Training_Data/Sample_tensor.parquet'
    
    if os.path.exists(original_file):
        print("Testing original large file...")
        success = test_parquet_file(original_file, max_samples=100)
        
        if not success:
            print("\nOriginal file has issues. Creating small test dataset...")
            test_file = create_small_test_dataset()
            
            print("\nTesting with small dataset...")
            test_training_pipeline(test_file)
        else:
            print("\nOriginal file works! Testing training pipeline...")
            test_training_pipeline(original_file)
    else:
        print(f"Original file not found: {original_file}")
        print("Creating small test dataset...")
        test_file = create_small_test_dataset()
        test_training_pipeline(test_file) 