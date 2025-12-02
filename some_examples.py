import numpy as np
from scipy.linalg import eigh

from generate import generate_masks

np.random.seed(43)

A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
w, v = eigh(A)
print(v)
print(w)
s = np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
print(s)

s = np.allclose(v.T @ A @ v - np.diag(w), np.zeros((4, 4)))


d = 3
m = 1
n = 8
info_dic = {
                'beta_gt': np.random.rand(1, d),
                'n_train': [n], 
                'p_miss': [[0.9, 10/9 * (1/2), 2  * 0.1]]
            }

mm = generate_masks(info_dic)
print("result , \n", mm)
ratio = np.sum(mm, axis=(-1, -2))
print("ratio: , ", ratio / (n * d) )


m = np.random.randint(0, 2, (3, 4))
print(m)
#print(m[0:3, 1::])

idx = 1
X = np.random.randint(0, 5, (3, 4)) + 0.0
x = np.random.randint(5, 9,   np.sum(m[:, idx]  )   )  + 0.0
print(X)
print(x)
X[m[:, idx] == 1, idx] = x
print(X)






