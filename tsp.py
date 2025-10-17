import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from utils import flip_matrix, generate_binary_arrays





np.random.seed(53)
n = 5
d = 3
X = np.random.rand(n, d)
M = np.random.binomial(1, 0.5, size=(n, d))
sum = np.sum(1-M, axis=0)
print("nbr seen, ", sum)
#print(M)
ones = np.ones((d, d)) 
F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
#FF = flip_matrix(M.T)
print("flip matrix \n", F)
#print("test Flip matrix \n", FF)

permutation, distance = solve_tsp_local_search(F)
print(permutation, distance)

print(permutation)
print(X)

def matrix_switches(M):
    # this matrix encode which vectors should move
    # from one side to the other
    #M1 = M.copy()
    #m1 = M1[0, :]
    #M[0, :] = M[-1, :]
    #M[-1, :] = m1
    M1 = np.roll(M, -1, axis=1)
    print(M1)
    return M - M1

Ms = matrix_switches(M)
print(Ms)

m1 = Ms[:, 0]
vp = Ms[m1 == 1, :]
print(vp)

def split_upd(X, ms):
    # split the 1 rank perturbations in updates and downdates
    X_upd = X[ms == 1, :]
    X_dwd = X[ms == -1, :]
    return {'+': X_upd, '-': X_dwd} 

def seq_rk_1_upd():
    A_inv = 1


def impute_matrix(X, Q, M):
    



def gibb_sampl(info):
    # flip matrix
    X = info['data']
    M = info['masks']
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    print("who is X in gibb sampl \n", X)
    ones = np.ones((d, d)) 
    F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    permutation, distance = solve_tsp_local_search(F)
    print("optimal perm ", permutation, "optimal dist ", distance) 
    M = M[:, permutation]
    X = X[:, permutation]
    print("\n", X)
    print("\n", M)
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    print("\n ", first_mask)
    R = X[first_mask == 0, :]
    print("first set vct ", R)
    print("first set vct shape ", R.shape)
    Rt_R = R.T @ R + (1/n) * lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)


    for h in range(r):
        for i in range(d):
            x = 1
            X = impute_matrix(X, Q, M)


np.random.seed(53)
n = 5
d = 3
lbd = 1 + 0.0
X = np.random.randint(1, 9, size=(n, d))
M = np.random.binomial(1, 0.25, size=(n, d))
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': 2,
    'lbd_reg': lbd
}

print("\n\nnew exp ", )
gibb_sampl(info_dic)





