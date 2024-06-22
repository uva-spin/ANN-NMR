import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_csv(file_path):
    """
    Load data from a CSV file and drop the first column.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    data (numpy array): Loaded data after dropping the first column.
    """
    df = pd.read_csv(file_path, header=None)  # Read CSV without headers
    data = df.iloc[1:, 1:].values  # Drop first row and first column
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



# Load real data from CSV
# file_path = r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\Noise_Data\Noise_RawSignal.csv'
file_path = r'C:\Work\ANN\Noise_RawSignal.csv'
real_data = load_data_from_csv(file_path)
y = real_data

Scaling = real_data.mean(axis=1)
scaler = StandardScaler()
Scaling = Scaling.reshape(-1, 1)
X = scaler.fit_transform(Scaling)
X_train, X_test, y_train, y_test = split_data(X, real_data, test_size=0.3, random_state=42)

# Define the neural network model
model = Sequential()
# Add multiple Dense layers to create a deeper network
model.add(tf.keras.Input(shape=(1,)))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# Output layer
model.add(Dense(500, activation='tanh'))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['accuracy'])

# Print the model summary
model.summary()

# Train the model with a larger batch size and more epochs
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.2)

# Evaluate the model on the test data
results = model.evaluate(X_test, y_test, verbose=1)
loss, accuracy = results if isinstance(results, (list, tuple)) else (results, None)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
# Predict using the model
n = 4
# predictions = []
predictions = model.predict(X_test[0:n])
    # predictions.append(prediction)
print("Predictions:")
print(predictions)

# Function to create n x 500 array with np.linspace elements
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
x = create_linspace_array(n)
x = x.reshape((n,500))
predictions = predictions.reshape((n,500))
print(predictions.shape)
print(x.shape)
plt.figure(figsize=(10, 6))
for i in range(n-1):
    plt.plot(x[i], predictions[i], color='blue', label='Synthetic Data Example')
    plt.plot(x[i], real_data[i], color='red',label='Real Data Example')
    plt.xlabel('Voltage')
    plt.ylabel('Frequency ')
    plt.title('Generated Synthetic Data')
    plt.legend()
    plt.grid(True)
plt.show()


# synthetic_file_path = r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\Noise_Data\Synthetic_Data.csv'
# save_data_to_csv(synthetic_data, synthetic_file_path)

# print(f"Synthetic data saved to {synthetic_file_path}")




# r'J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\Noise_Data\Noise_RawSignal.csv'