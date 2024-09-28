import pandas as pd

df = pd.read_csv(r"NM4_Fermilab\Model Performance\v9\Results.csv")
lowest, highest = df['Area_True'].min(), df['Area_True'].max()  # Replace 'your_column_name' with your actual column name
print(lowest, highest)