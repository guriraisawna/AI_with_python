import matplotlib.pyplot as plt
import numpy as np

points = np.linspace(0, 10, 400)

plt.figure(figsize=(10, 8))
plt.plot(points, 2*points + 1, 'b--', label="y = 2x + 1")
plt.plot(points, 2*points + 2, 'g-', label="y = 2x + 2")
plt.plot(points, 2*points + 3, 'r:', label="y = 2x + 3")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Lines y=2x+1, y=2x+2, y=2x+3")
plt.legend()
plt.grid(True)
plt.show()
