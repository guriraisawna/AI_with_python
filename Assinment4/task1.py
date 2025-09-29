import numpy as np
import matplotlib.pyplot as plt


trials = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for sample_size in trials:
    dice1 = np.random.randint(1, 7, sample_size)
    dice2 = np.random.randint(1, 7, sample_size)

    total = dice1 + dice2

    freq, bin_edges = np.histogram(total, bins=range(2, 14))

    plt.bar(bin_edges[:-1], freq / sample_size, width=0.9, color="skyblue", edgecolor="black")

    plt.title(f"Sample Size = {sample_size}")
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Relative Frequency")

    plt.show()
#guri