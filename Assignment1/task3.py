import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('weight-height(1).csv', delimiter=',', skip_header=1)

heights_in = data[:, 1]
weights_lb = data[:, 2]

heights_cm = heights_in * 2.54
weights_kg = weights_lb * 0.453592

avg_height = np.mean(heights_cm)
avg_weight = np.mean(weights_kg)

print(f"Mean height: {avg_height}")
