import numpy as np
import matplotlib.pyplot as plt


def Voigt(x, ampG1, sigmaG1, ampL1, widL1, center):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-center)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-center)**2+widL1**2)) )

x = np.linspace(-10,10,100)

voigt = Voigt(x,2,2,.0,10,0)

plt.plot(x,voigt)
plt.show()