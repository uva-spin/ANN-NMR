from scipy.special import voigt_profile
import numpy as np
import matplotlib.pyplot as plt

# Sample values for testing
circ_constants = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.01, 0.02, 0.03, 0.04]
circ_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
function_input = 1.0  # Replace with actual function input
scan_s = 0.1  # Replace with actual scan size
mu = 0.01
gamma = 0.1
ranger = 0  # Replace with actual range if necessary

def LabviewCalculateYArray(circ_consts, params, f_input, scansize, mu, gamma, rangesize):
    # Print inputs for debugging
    print("circ_consts:", circ_consts)
    print("params:", params)
    print("f_input:", f_input, type(f_input))
    print("scansize:", scansize, type(scansize))
    print("mu:", mu, type(mu))
    print("gamma:", gamma, type(gamma))
    print("rangesize:", rangesize, type(rangesize))

    x_values = np.linspace(-1, 1, 500).astype(np.float64)
    print("x_values:", x_values, type(x_values))

    # Calculate Voigt profile
    try:
        x1 = np.array(voigt_profile(x_values, mu, gamma), dtype=float)
        x2 = np.array(voigt_profile(x_values, mu, gamma), dtype=float)
        print("x1:", x1)
        print("x2:", x2)
    except Exception as e:
        print("Error calculating Voigt profile:", e)
        raise

    # Remaining function code...
    # For simplicity, we'll return x1 for now
    return x1,x_values

# Test the function with sample values
result,x = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, mu, gamma, ranger)
# print("Result:", result)

plt.plot(x,result)
plt.show()
