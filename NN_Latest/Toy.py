import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

def load_data_from_csv(file_path):
    """
    Load data from a CSV file and drop the first column.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    data (numpy array): Loaded data after dropping the first column.
    """
    df = pd.read_csv(file_path, header=None)  # Read CSV without headers
    data = df.iloc[1:, 1:] # Drop first row and first column
    return data

def save_data_to_csv(data, file_path):
    """
    Save data to a CSV file.
    
    Parameters:
    data (numpy array): Data to save.
    file_path (str): Path to the CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def split_data(X, y, test_size=0.2, random_state=None):
    """
    Splits the features and target data into training and testing sets.
    
    Parameters:
    X (pandas.DataFrame or numpy.ndarray): The input features.
    y (pandas.Series or numpy.ndarray): The target variable.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    random_state (int): Controls the shuffling applied to the data before applying the split (default is None).
    
    Returns:
    pandas.DataFrame: X_train
    pandas.DataFrame: X_test
    pandas.Series: y_train
    pandas.Series: y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def exclude_outliers(df, threshold=1.5):
    # Compute Z-scores for each row
    z_scores = df.apply(zscore, axis=0, result_type='broadcast')
    
    # Check if any Z-score exceeds the threshold
    is_outlier = (z_scores.abs() > threshold).any(axis=1)
    
    # Exclude outliers
    df_filtered = df[~is_outlier].apply(lambda x: x / 1000)
    
    return df_filtered

def normalize_row_wise(df):
    """
    Normalizes each row of the dataframe: (value - mean) / std
    
    Args:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Row-wise normalized dataframe
    """
    # Subtract the mean of each row and then divide by the standard deviation of each row
    return df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)


def create_linspace_array(n):
    # Initialize an empty list to store rows
    array_rows = []

    # Populate each row with np.linspace
    for i in range(n):
        # Generate a linearly spaced array from 0 to 1 with 500 elements
        linspace_row = np.linspace(31, 33, 500)
        # Append the row to the list
        array_rows.append(linspace_row)
    
    # Convert the list of rows into a numpy array
    result_array = np.array(array_rows)
    return result_array


file_path = r'C:\Work\ANN\Noise_RawSignal.csv'
real_data = normalize_row_wise(exclude_outliers(load_data_from_csv(file_path)))
y_df = real_data
n = 500
x = create_linspace_array(n)
x = x.reshape((n,500))

# Prepare the data
# x = x.reshape(-1, 1)  # Reshape x to be a 2D array with shape (n_samples, 1)

# Function to create and train a model for each y column
def create_and_train_model(x, y):
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x, y, epochs=100, batch_size=10, verbose=0)
    
    return model

# Dictionary to hold trained models for each y column
models = {}

# Train a model for each row in the DataFrame
for idx, row in y_df.iterrows():
    y = row.values.reshape(-1, 1)  # Reshape row to be a 2D array with shape (n_samples, 1)
    models[idx] = create_and_train_model(x, y)

# Example of how to use the models to make predictions up to a certain index
index = 80  # Specify the index up to which you want to plot predictions
x_new = x[:index]  # Use the original x values up to the specified index
predictions = {}
for column, model in models.items():
    predictions[column] = model.predict(x_new).flatten()

# Plot the original and predicted values
plt.figure(figsize=(14, 8))
for column in y_df.columns:
    plt.plot(x[:index], y_df[column].values[:index], label=f'Original {column}')
    plt.plot(x[:index], predictions[column], '--', label=f'Predicted {column}')
    
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original and Predicted Values')
plt.legend()
plt.show()

