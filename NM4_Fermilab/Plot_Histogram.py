from Misc_Functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

file_path = 'Model Performance/v6/Results.csv'  
df = pd.read_csv(file_path)

if 'Area_Predicted' not in df.columns or 'Area_True' not in df.columns:
    raise ValueError("The CSV must contain 'Area_Predicted' and 'Area_True' columns.")

df['Area_Diff'] = df['Area_Predicted'] - df['Area_True']

plot_histogram(
    df['Area_Diff'], 
    'Histogram of Area Difference', 
    'Difference in Area', 
    'Count', 
    'green', 
    os.path.join(".", 'Histogram_Relative_Error.png')
)
