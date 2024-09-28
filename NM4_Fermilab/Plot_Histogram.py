from Misc_Functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

file_path = 'Model Performance/v9/Results.csv'  
df = pd.read_csv(file_path)

if 'Area_Predicted' not in df.columns or 'Area_True' not in df.columns:
    raise ValueError("The CSV must contain 'Area_Predicted' and 'Area_True' columns.")

df['Area_Diff'] = df['Area_Predicted'] - df['Area_True']
abs_err = np.abs(df['Area_Predicted'] - df['Area_True'])

plot_histogram(
    df['Area_Diff'], 
    'Histogram of Area Difference', 
    'Difference in Area', 
    'Count', 
    'green', 
    os.path.join(".", 'Proton_Histogram_Area_Difference.png'),
    plot_norm=False
)

plot_histogram(
    abs_err, 
    'Histogram of Absolute Error', 
    'Absolute Error', 
    'Count', 
    'blue', 
    os.path.join(".", 'Proton_Histogram_Absolute_Error.png'),
    plot_norm=False
)