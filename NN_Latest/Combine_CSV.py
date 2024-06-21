import os
import glob
import pandas as pd
os.chdir(r"J:\Users\Devin\Desktop\Spin Physics Work\ANN Github\NMR-Fermilab\ANN-NMR\NN_Latest\Noise_Data")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "Noise_RawSignal.csv", index=False, encoding='utf-8-sig')