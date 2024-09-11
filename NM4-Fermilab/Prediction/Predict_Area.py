import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import norm
from Misc_Functions import find_file  # Importing your custom function

<<<<<<< HEAD
def find_directory(directory_name, start_dir='.'):
    current_dir = os.path.abspath(start_dir)
    levels_up = 0
    
    while levels_up <= 2:  # Limit the search to two directory levels up
        print(f"Searching for '{directory_name}' in {current_dir}...")  # Debug print
        for root, dirs, _ in os.walk(current_dir):
            if directory_name in dirs:
                print(f"Found '{directory_name}' in {root}")  # Debug print
                return os.path.join(root, directory_name)
        
        # Move up one directory level if not found
        current_dir = os.path.abspath(os.path.join(current_dir, '..'))
        levels_up += 1
    
    print(f"Could not find '{directory_name}' within {levels_up} levels up from {start_dir}")  # Debug print
    return None


# User input for version number
version_number = 'v3'  # Example: 'v1', 'v2', etc.

# Find the relevant directories
model_dir = find_directory('Models', start_dir='.')
data_dir = find_directory('Testing_Data', start_dir='.')
results_dir = find_directory('Model Performance', start_dir='.')

if not model_dir or not data_dir or not results_dir:
    raise FileNotFoundError("One or more required directories (Models, Testing_Data, Model Performance) could not be found.")

# Define the model file name and data file path
model_filename = f'best_model_{version_number}.keras'
data_file = os.path.join(data_dir, 'Sample_Testing_Data_10K.csv')

# Ensure the version-specific results directory exists
results_dir = os.path.join(results_dir, version_number)
os.makedirs(results_dir, exist_ok=True)

# Find and load the model file
model_file = find_file(model_filename, start_dir=model_dir)
if model_file is None:
    raise FileNotFoundError(f"Model file '{model_filename}' not found.")
Area_Model = load_model(model_file)

# Load the data
df_Area = pd.read_csv(data_file)

# Prepare data for prediction
X_Area = df_Area.drop(['Area', 'SNR'], axis=1)
Area = df_Area['Area'].to_numpy()
SNR = df_Area['SNR'].to_numpy()

Y_Area = Area_Model.predict(X_Area).reshape(-1)


# Calculate errors
rel_err = np.abs((Y_Area - Area) / Area) * 100 
abs_err = np.abs(Y_Area - Area)

# Save the results
result = pd.DataFrame({
    'Area_Predicted': Y_Area,
    'Area_True': Area,
    'rel_err': rel_err,
    'abs_err': abs_err
})
result_file = os.path.join(results_dir, 'Results.csv')
result.to_csv(result_file, index=False)

# Define a function for plotting histograms
def plot_histogram(data, title, xlabel, ylabel, color, save_path, num_bins=100):
    n, bins, patches = plt.hist(data, num_bins, density=True, color=color, alpha=0.7)
    mu, sigma = norm.fit(data)
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, '--', color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}: mu={mu:.4f}, sigma={sigma:.4f}")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
=======
Area_Model = load_model(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Models\v1\best_model_v1.keras')

df_Area = pd.read_csv(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Testing_Data\Sample_Testing_Data.csv')

y_Area = df_Area['Area']
x_Area = df_Area.drop(['Area'], axis=1)
X_Area = df_Area.drop(['Area', 'SNR'], axis=1)
Y_Area = Area_Model.predict(X_Area)
Area = np.array(df_Area['Area'])  
SNR = np.array(df_Area['SNR'])
Y_Area = Y_Area.reshape((len(Y_Area),))

rel_err = (np.abs((Y_Area - Area)) / Area) * 100 
abs_err = np.abs(Y_Area - Area)

result = pd.DataFrame({'Area_Predicted': Y_Area})
result['Area_True'] = Area.tolist()
result['rel_err'] = rel_err.tolist()
result['abs_err'] = abs_err.tolist()
result.to_csv(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Results.csv', index=False)

num_bins = 100

n, bins, patches = plt.hist(rel_err, num_bins, 
                            density=True, 
                            color='green',
                            alpha=0.7)

(mu, sigma) = norm.fit(rel_err)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, '--', color='black')

plt.xlabel('Difference in Area')
plt.ylabel('Count')
plt.title(f"Histogram of Area Difference: mu={mu:.4f}, sigma={sigma:.4f}")
plt.grid(True)

plt.savefig(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Histogram_Relative_Error.png')
plt.close()  

n, bins, patches = plt.hist(abs_err, num_bins, 
                            density=True, 
                            color='blue',
                            alpha=0.7)

(mu, sigma) = norm.fit(abs_err)
y = norm.pdf(bins, mu, sigma)
plt.plot(bins, y, '--', color='black')

plt.xlabel('Absolute Error')
plt.ylabel('Count')
plt.title(f"Histogram of Absolute Error: mu={mu:.4f}, sigma={sigma:.4f}")
plt.grid(True)

plt.savefig(r'C:\Program Files\Work\ANN\ANN-NMR\NM4-Fermilab\Prediction_Results\Histogram_Absolute_Error.png')
>>>>>>> 9f226776c453369d6cdece695b8b50a621a41467

# Plot and save relative error histogram
plot_histogram(
    rel_err, 
    'Histogram of Area Difference', 
    'Difference in Area', 
    'Count', 
    'green', 
    os.path.join(results_dir, 'Histogram_Relative_Error.png')
)

# Plot and save absolute error histogram
plot_histogram(
    abs_err, 
    'Histogram of Absolute Error', 
    'Absolute Error', 
    'Count', 
    'blue', 
    os.path.join(results_dir, 'Histogram_Absolute_Error.png')
)

