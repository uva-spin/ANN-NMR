import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from Custom_Scripts.Misc_Functions import find_file

def Pipeline(
    data_path,
    p_threshold=0.01,
    test_size=0.3,
    validation_split=1/3,
    batch_size=32,
    random_seed=42,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch_size=tf.data.AUTOTUNE
):
    """
    Load and prepare data from CSV for TensorFlow training with multiprocessing support.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the data
    p_threshold : float, optional
        Filter data where P is less than or equal to this value
    test_size : float, optional
        Proportion of data to use for testing and validation
    validation_split : float, optional
        Proportion of test_size to use for validation
    batch_size : int, optional
        Batch size for TensorFlow datasets
    random_seed : int, optional
        Random seed for reproducibility
    num_parallel_calls : int, optional
        Number of parallel calls for dataset mapping operations
    prefetch_size : int, optional
        Number of batches to prefetch
    
    Returns:
    --------
    dict
        Dictionary containing train, validation, and test datasets, along with data shapes
    """
    print("Loading data...")
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully! Total rows: {len(data)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Filter data based on P threshold
    if p_threshold is not None:
        data = data[data['P'] <= p_threshold]
        print(f"Filtered data to {len(data)} samples with P <= {p_threshold}")
    
    # Split data into train, validation, and test sets
    print("Splitting data...")
    train_data, temp_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    val_data, test_data = train_test_split(temp_data, test_size=validation_split, random_state=random_seed)
    print(f"Data split: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")
    
    # Extract features and targets
    print("Preparing features and targets...")
    columns_to_drop = ["P", "SNR"] if "SNR" in data.columns else ["P"]
    
    X_train = train_data.drop(columns=columns_to_drop).astype('float32').values
    y_train = train_data["P"].astype('float32').values
    
    X_val = val_data.drop(columns=columns_to_drop).astype('float32').values
    y_val = val_data["P"].astype('float32').values
    
    X_test = test_data.drop(columns=columns_to_drop).astype('float32').values
    y_test = test_data["P"].astype('float32').values
    
    # Get feature dimensions
    num_features = X_train.shape[1]
    print(f"Features prepared: {num_features} features per sample")
    
    # Convert to TensorFlow datasets with multiprocessing
    print("Creating TensorFlow datasets with multiprocessing support...")
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Apply performance optimizations
    train_dataset = (train_dataset
        .shuffle(buffer_size=len(X_train))
        .batch(batch_size)
        .map(lambda x, y: (x, y), num_parallel_calls=num_parallel_calls)
        .prefetch(prefetch_size))
    
    val_dataset = (val_dataset
        .batch(batch_size)
        .map(lambda x, y: (x, y), num_parallel_calls=num_parallel_calls)
        .prefetch(prefetch_size))
    
    test_dataset = (test_dataset
        .batch(batch_size)
        .map(lambda x, y: (x, y), num_parallel_calls=num_parallel_calls)
        .prefetch(prefetch_size))
    
    print("TensorFlow datasets created successfully")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "num_features": num_features,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    }


# Example usage:
if __name__ == "__main__":
    # Set the number of CPU cores to use
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    
    # Load the data
    data_info = Pipeline(
        data_path=find_file("Deuteron_Oversampled_1M.csv"),
        batch_size=64
    )
    
    if data_info:
        # Access the datasets
        train_ds = data_info["train_dataset"]
        val_ds = data_info["val_dataset"]
        test_ds = data_info["test_dataset"]
        
        print(f"Dataset info:")
        print(f"- Features per sample: {data_info['num_features']}")
        print(f"- Training samples: {data_info['train_samples']}")
        print(f"- Validation samples: {data_info['val_samples']}")
        print(f"- Test samples: {data_info['test_samples']}")
        
        # Example of iterating through batches
        for features, labels in train_ds.take(1):
            print(f"Batch shape: {features.shape}, Labels shape: {labels.shape}")