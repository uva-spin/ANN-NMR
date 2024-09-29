import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from scipy.stats import norm
from Misc_Functions import *
import sys 


version_number = str(sys.argv[1])
data_file = str(sys.argv[2])

# version_number = 'v9'
# data_file = 'Sample_Testing_Data_v7_50K.csv'


model_dir = find_directory('Models', start_dir='.')
data_dir = find_directory('Testing_Data_Deuteron', start_dir='.')
results_dir = find_directory('Model Performance', start_dir='.')

if not model_dir or not data_dir or not results_dir:
    raise FileNotFoundError("One or more required directories (Models, Testing_Data, Model Performance) could not be found.")

model_filename = f'final_model_{version_number}.keras'
data_file = os.path.join(data_dir, data_file)

results_dir = os.path.join(results_dir, version_number)
os.makedirs(results_dir, exist_ok=True)

model_file = find_file(model_filename, start_dir=model_dir)
if model_file is None:
    raise FileNotFoundError(f"Model file '{model_filename}' not found.")
P_Model = load_model(model_file)

df_P = pd.read_csv(data_file)

X_P = df_P.drop(['P', 'SNR'], axis=1)
P = df_P['P'].to_numpy()
SNR = df_P['SNR'].to_numpy()

scaler_X = MinMaxScaler()
# X_P_Scaled = scaler_X.fit_transform(X_P)

Y_P = P_Model.predict(X_P).reshape(-1)


# rel_err = np.abs((Y_P - P) / P) * 100 

P_diff = Y_P - P
abs_err = np.abs(Y_P - P)

result = pd.DataFrame({
    'P_Predicted': Y_P,
    'P_True': P,
    'rel_err': P_diff,
    'abs_err': abs_err
})
result_file = os.path.join(results_dir, 'Results.csv')
result.to_csv(result_file, index=False)
print("Results successfully saved!")


plot_histogram(
    P_diff, 
    'Histogram of P Difference', 
    'Difference in P', 
    'Count', 
    'green', 
    os.path.join(results_dir, 'Histogram_P_Difference.png')
)

plot_histogram(
    abs_err, 
    'Histogram of Absolute Error', 
    'Absolute Error', 
    'Count', 
    'blue', 
    os.path.join(results_dir, 'Histogram_Absolute_Error.png')
)

print("Histograms saved.")

