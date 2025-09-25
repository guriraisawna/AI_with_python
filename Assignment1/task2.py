import matplotlib.pyplot as plt
import numpy as np

x_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y_values = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.figure(figsize=(7, 5))
plt.scatter(x_values, y_values, marker="+", color="b", s=50)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()
