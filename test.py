import numpy as np
from wiener_filtering import add_zero_padding

# Suppose A is your non-square matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A = add_zero_padding(A, 3)
print(A)