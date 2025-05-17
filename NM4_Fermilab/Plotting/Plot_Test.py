import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Plot_Script as ps

# Load the CSV file

version = 'Deuteron_TE_60_Noisy_Shifted_100K_CNN_Attention_Fixed'
df = pd.read_csv(f'Model_Performance/{version}/{version}_results.csv', 
                 delimiter=',')

print("Original data shape:", df.shape)

# Calculate mean and standard deviation of RPE
mean_rpe = df['RPE'].mean()
std_rpe = df['RPE'].std()

# Define the threshold for outliers (3 standard deviations)
threshold = 2 * std_rpe

# Filter out outliers
df_filtered = df[abs(df['RPE'] - mean_rpe) <= threshold]

# Print statistics about removed outliers
num_outliers = len(df) - len(df_filtered)
print(f"Removed {num_outliers} outliers ({num_outliers/len(df)*100:.2f}% of data)")
print(f"Filtered data shape: {df_filtered.shape}")

# Use the filtered dataframe for plotting
print(df_filtered.head())
ps.plot_enhanced_performance_metrics(df_filtered['True'], df_filtered['Predicted'], df_filtered['SNR'], 
                                    f'Model_Performance/{version}/Revised_Plots', 
                                    version)