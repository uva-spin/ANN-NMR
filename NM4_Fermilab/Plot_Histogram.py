from Misc_Functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from scipy.stats import norm

# file_path = 'Model Performance/v9/Results.csv'
file_path = 'Model Performance\Deuteron_v5\Results.csv'  
df = pd.read_csv(file_path)

# if 'Area_Predicted' not in df.columns or 'Area_True' not in df.columns:
#     raise ValueError("The CSV must contain 'Area_Predicted' and 'Area_True' columns.")

df['P_Diff'] = df['P_Predicted'] - df['P_True']
# abs_err = np.abs(df['Area_Predicted'] - df['Area_True'])
abs_err = np.abs(df['P_Predicted'] - df['P_True'])

fig = plt.figure(figsize=(16, 6))  

gs = fig.add_gridspec(1, 2)  

ax1 = fig.add_subplot(gs[0])

plot_histogram(
    df['P_Diff'], 
    'Histogram of Polarization Difference', 
    'Difference in Polarization', 
    'Count', 
    'red', 
    ax1,
    # os.path.join(".", 'Deuteron_Histogram_P_Difference.png'),
    plot_norm=False
)
ax2 = fig.add_subplot(gs[1])
plot_histogram(
    abs_err, 
    'Histogram of Absolute Error', 
    'Absolute Error',
    '', 
    'orange', 
    ax2,
    # os.path.join(".", 'Deuteron_Histogram_Absolute_Error.png'),
    plot_norm=False
)


ax1.text(0.5, -0.2, '(a)', transform=ax1.transAxes, 
         ha='center', fontsize=16,weight='bold')
ax2.text(0.5, -0.2, '(b)', transform=ax2.transAxes, 
         ha='center', fontsize=16,weight='bold')




plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

output_path = os.path.join(".", 'Deuteron_Histograms.png')
fig.savefig(output_path,dpi=600)