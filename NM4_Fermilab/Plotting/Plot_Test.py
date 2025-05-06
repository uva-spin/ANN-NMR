import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Plot_Script as ps

# Load the CSV file
df = pd.read_csv('Training/Model Performance/Deuteron_Shifted_low_CNN_Attention_Optuna_V1/Deuteron_Shifted_low_CNN_Attention_Optuna_V1_filtered_results.csv', 
                 delimiter=',',
                 names = ['True', 'Predicted', 'Residuals', 'Percentage_Error', 'RPE'],
                 skiprows=1)

print("Original data shape:", df.shape)

# Calculate mean and standard deviation of RPE
mean_rpe = df['RPE'].mean()
std_rpe = df['RPE'].std()

# Define the threshold for outliers (3 standard deviations)
threshold = 3 * std_rpe

# Filter out outliers
df_filtered = df[abs(df['RPE'] - mean_rpe) <= threshold]

# Print statistics about removed outliers
num_outliers = len(df) - len(df_filtered)
print(f"Removed {num_outliers} outliers ({num_outliers/len(df)*100:.2f}% of data)")
print(f"Filtered data shape: {df_filtered.shape}")

# Use the filtered dataframe for plotting
print(df_filtered.head())
ps.plot_enhanced_performance_metrics(df_filtered['True'], df_filtered['Predicted'], 
                                    'Training/Model Performance/Deuteron_Shifted_low_CNN_Attention_Optuna_V1/New_Plots', 
                                    'Deuteron_CNN_Attention_Optuna_V1_filtered')