import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('weight-height(1).csv', delimiter=',', skip_header=1)

length_in = data[:, 1]
weight_lb = data[:, 2]

length_cm = length_in * 2.54
weight_kg = weight_lb * 0.453592

mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean height: {mean_length:.2f} cm")
print(f"Mean weight: {mean_weight:.2f} kg")

plt.hist(length_cm, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Student Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Number of Students')
plt.show()
