import numpy as np

A = np.array([[1, 2, 3],
                         [0, 1, 4],
                         [5, 6, 0]], dtype=float)

A_inv = np.linalg.inv(A)

print("Inverse of A:")
print(A_inv)

print("\nA @ A_inv:")
print(A@A_inv)

print("\nA_inv @ A:")
print(A_inv@A)


