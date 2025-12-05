import numpy as np
from scipy.sparse import csr_matrix



def serialization_first_idea(FF):
    # F: flip matrix (aka distance matrix between nodes)
    d, _ = FF.shape
    F = FF.copy()
    idx = np.random.randint(0, d)
    max_F = np.max(FF) + 1
    F = F + np.eye(d) * max_F  # now the diagonal does not attain the minimum anymore
    f = F[idx, :]
    res = np.zeros(d, dtype=np.int64) - 1
    #print("initial index ", idx)
    res[0] = idx
    #print(F)
    cost = 0
    for i in range(d-1):
        j = np.argmin(f)
#       print(j)
        #print("abc ", f)
        cost = cost + F[idx, j]
        res[i+1] = j
   #     print(res)
        F[:, idx] = max_F
        F[idx, :] = max_F
#        print(F)
        f = F[j, :]
 #       print(f, "\n")
        idx = j
    return res, cost

np.random.seed(44)
n, d = 50, 100
M = np.random.binomial(1, 0.5, size=(n, d))
M_s = csr_matrix(M)
ones_d = np.ones(d)
#FF = n * ones_d - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s

res = serialization_first_idea(F)
#print(res)

t = d
print(F)
superdiag = np.diag(F, k=1)
print("orignial cost ", np.sum(superdiag), "\n")
#for j in range(d):
#    res1, cost1 = serialization_first_idea(F)
    #print(res1)
    #print("cost1 ", cost1, "\n")










