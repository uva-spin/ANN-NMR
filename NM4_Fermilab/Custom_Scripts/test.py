from Lineshape import *
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,5,500)
P = 0.5
bound = 0
signal = Sampling_Lineshape(P,x,bound)
plt.plot(x,signal) 
plt.show()