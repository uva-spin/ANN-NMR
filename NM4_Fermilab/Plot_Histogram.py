from Misc_Functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.stats import norm

file_path = 'Model Performance/v9/Results.csv'  
df = pd.read_csv(file_path)

if 'Area_Predicted' not in df.columns or 'Area_True' not in df.columns:
    raise ValueError("The CSV must contain 'Area_Predicted' and 'Area_True' columns.")

df['Area_Diff'] = df['Area_Predicted'] - df['Area_True']
abs_err = np.abs(df['Area_Predicted'] - df['Area_True'])


fig = plt.figure(figsize=(16, 6))  # Adjusting the overall figure size

# Use gridspec to manage subplot sizes
gs = fig.add_gridspec(1, 2)  # Two rows for histograms

# Plot the first histogram for Area Difference
ax1 = fig.add_subplot(gs[0])

plot_histogram(
    df['Area_Diff'], 
    'Histogram of Area Difference', 
    'Difference in Area', 
    'Count', 
    'green', 
    ax1,
    os.path.join(".", 'Proton_Histogram_Area_Difference.png'),
    plot_norm=False
)
ax2 = fig.add_subplot(gs[1])
plot_histogram(
    abs_err, 
    'Histogram of Absolute Error', 
    'Absolute Error',
    '', 
    'blue', 
    ax2,
    os.path.join(".", 'Proton_Histogram_Absolute_Error.png'),
    plot_norm=False
)


ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, 
         ha='center', fontsize=16)
ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, 
         ha='center', fontsize=16)




plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
# plt.show()

output_path = os.path.join(".", 'Proton_Histograms.png')
fig.savefig(output_path,dpi=600)