import numpy as np
from scipy.linalg import eigh
A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
w, v = eigh(A)
print(v)
print(w)
s = np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
print(s)









