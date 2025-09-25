import numpy as np

matrix = np.array([[1, 2, 3],
                   [0, 1, 4],
                   [5, 6, 0]], dtype=float)

matrix_inverse = np.linalg.inv(matrix)

print("Inverse of matrix:")
print(matrix_inverse)

print("\nmatrix @ matrix_inverse:")
print(matrix @ matrix_inverse)

print("\nmatrix_inverse @ matrix:")
print(matrix_inverse @ matrix)
