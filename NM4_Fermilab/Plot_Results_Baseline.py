import numpy as np
import pandas as pd
from Lineshape import *
from Misc_Functions import *
df = pd.read_csv(r'Model Performance\Deuteron_v8\Results.csv')

results = df.iloc[0]

print(results)