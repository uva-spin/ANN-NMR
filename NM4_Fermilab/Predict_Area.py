import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import norm
from Misc_Functions import *
import sys 


version_number = str(sys.argv[1])
data_file = str(sys.argv[2])

# version_number = 'v9'
# data_file = 'Sample_Testing_Data_v7_50K.csv'


model_dir = find_directory('Models', start_dir='.')
data_dir = find_directory('Testing_Data', start_dir='.')
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
Area_Model = load_model(model_file)

df_Area = pd.read_csv(data_file)

X_Area = df_Area.drop(['Area', 'SNR'], axis=1)
Area = df_Area['Area'].to_numpy()
SNR = df_Area['SNR'].to_numpy()

Y_Area = Area_Model.predict(X_Area).reshape(-1)

if len(Y_Area) != len(Area):
    print("Error: Mismatch in lengths of predicted and true values.")
else:
    area_diff = Y_Area.flatten() - Area
    abs_err = np.abs(Y_Area.flatten() - Area)

    results = pd.DataFrame({
        'P_Predicted': Y_Area.flatten(),
        'P_True': Area,
        'rel_err': area_diff,
        'abs_err': abs_err
    })

    result_file = os.path.join(results_dir, 'Results.csv')
    results.to_csv(result_file, index=False)
    print("Results successfully saved!")

    fig = plt.figure(figsize=(16, 6))  

    gs = fig.add_gridspec(1, 2)  

    ax1 = fig.add_subplot(gs[0])

    plot_histogram(
        area_diff, 
        'Histogram of Area Difference', 
        'Difference in Area', 
        'Count', 
        'blue', 
        ax1,
        # os.path.join(results_dir, 'Proton_Histogram_Area_Difference.png'),
        plot_norm=False
    )
    subplot1_path = os.path.join(results_dir, 'Proton_Histogram_Area_Difference.png')
    ax1.figure.savefig(subplot1_path, dpi=600)
    print(f"Individual subplot saved: {subplot1_path}")

    ax2 = fig.add_subplot(gs[1])
    plot_histogram(
        abs_err,
        'Histogram of Absolute Error',
        'Absolute Error',
        '',
        'green',
        ax2,
        plot_norm=False
    )

    subplot2_path = os.path.join(results_dir, 'Proton_Histogram_Absolute_Error.png')
    ax2.figure.savefig(subplot2_path, dpi=600)
    print(f"Individual subplot saved: {subplot2_path}")

    ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes,
         ha='center', fontsize=16)
    ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes,
         ha='center', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    output_path = os.path.join(results_dir, 'Proton_Histograms.png')
    fig.savefig(output_path, dpi=600)
    print(f"Entire figure saved: {output_path}")

