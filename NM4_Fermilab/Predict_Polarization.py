import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from Misc_Functions import *
import sys 
import joblib

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

version_number = str(sys.argv[1])
data_file = str(sys.argv[2])

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

df_P = pd.read_csv(data_file, dtype={'P': 'float32', 'SNR': 'float32'})

scaler_X = joblib.load(f'Models/{version_number}/scaler_X.pkl')
scaler_y = joblib.load(f'Models/{version_number}/scaler_y.pkl')

X_P = df_P.drop(['P', 'SNR'], axis=1)
P = df_P['P'].to_numpy().astype('float32') 

X_P_Scaled = scaler_X.transform(X_P)

Y_P_Scaled = P_Model.predict(X_P_Scaled)

Y_P = scaler_y.inverse_transform(Y_P_Scaled.reshape(-1, 1))

print(f"Predictions size: {len(Y_P)}, True size: {len(P)}")

if len(Y_P) != len(P):
    print("Error: Mismatch in lengths of predicted and true values.")
else:
    P_diff = Y_P.flatten() - P
    abs_err = np.abs(Y_P.flatten() - P)

    results = pd.DataFrame({
        'P_Predicted': Y_P.flatten(),
        'P_True': P,
        'rel_err': P_diff,
        'abs_err': abs_err
    })

    result_file = os.path.join(results_dir, 'Results.csv')
    results.to_csv(result_file, index=False)
    print("Results successfully saved!")

    fig = plt.figure(figsize=(16, 6))  

    gs = fig.add_gridspec(1, 2)  

    ax1 = fig.add_subplot(gs[0])

    plot_histogram(
        P_diff, 
        'Histogram of Polarization Difference', 
        'Difference in Polarization', 
        'Count', 
        'blue', 
        ax1,
        plot_norm=False
    )
    subplot1_path = os.path.join(results_dir, 'Deuteron_Histogram_Polarization_Difference.png')
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

    subplot2_path = os.path.join(results_dir, 'Deuteron_Histogram_Absolute_Error.png')
    ax2.figure.savefig(subplot2_path, dpi=600)
    print(f"Individual subplot saved: {subplot2_path}")

    ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes,
         ha='center', fontsize=16)
    ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes,
         ha='center', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    output_path = os.path.join(results_dir, 'Deuteron_Histograms.png')
    fig.savefig(output_path, dpi=600)
    print(f"Entire figure saved: {output_path}")
