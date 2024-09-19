import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import norm
from Misc_Functions import *
import sys 


# version_number = str(sys.argv[1])
# data_file = str(sys.argv[2])

version_number = 'v7'
data_file = 'Sample_Testing_Data_v7_50K.csv'


model_dir = find_directory('Models', start_dir='.')
data_dir = find_directory('Testing_Data', start_dir='.')
results_dir = find_directory('Model Performance', start_dir='.')

if not model_dir or not data_dir or not results_dir:
    raise FileNotFoundError("One or more required directories (Models, Testing_Data, Model Performance) could not be found.")

model_filename = f'final_model_{version_number}.h5'
data_file = os.path.join(data_dir, data_file)

results_dir = os.path.join(results_dir, version_number)
os.makedirs(results_dir, exist_ok=True)

model_file = find_file(model_filename, start_dir=model_dir)
if model_file is None:
    raise FileNotFoundError(f"Model file '{model_filename}' not found.")
Area_Model = load_model(model_file)

df_Area = pd.read_csv(data_file)

X_Area = df_Area.drop(['Area', 'SNR'], axis=1)
Area = df_Area['Area'].to_numpy()
SNR = df_Area['SNR'].to_numpy()

Y_Area = Area_Model.predict(X_Area).reshape(-1)


rel_err = np.abs((Y_Area - Area) / Area) * 100 
abs_err = np.abs(Y_Area - Area)

result = pd.DataFrame({
    'Area_Predicted': Y_Area,
    'Area_True': Area,
    'rel_err': rel_err,
    'abs_err': abs_err
})
result_file = os.path.join(results_dir, 'Results.csv')
result.to_csv(result_file, index=False)
print("Results successfully saved!")


plot_histogram(
    rel_err, 
    'Histogram of Area Difference', 
    'Difference in Area', 
    'Count', 
    'green', 
    os.path.join(results_dir, 'Histogram_Area_Difference.png')
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

