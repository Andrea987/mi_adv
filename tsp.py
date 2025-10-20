import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from utils import flip_matrix, generate_binary_arrays
from sklearn.linear_model import Ridge




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
    #print(M1)
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
    return X_upd, X_dwd
    #return {'+': X_upd, '-': X_dwd}


def test_split_upd():
    n, d = 9, 3
    X = np.random.randint(1, 5, size =(n, d))  + 0.0
    m = np.random.binomial(n=1, p=0.3, size= (n, d))
    #m1 = np.random.binomial(n=1, p=0.3, size=n)
    #m2 = np.random.binomial(n=1, p=0.3, size=n)
    #print(m1)
    #print(m2)
    #m = np.array([m1, m2])
    #print("masks\n\n ", m)
    print("masks\n ", m)
    ms = matrix_switches(m)
    print("ms \n ", ms)
    vpl, vmn = split_upd(X, ms[:, 0])
    print("X\n ", X)
    print("updates/downdates, \n", ms)
    print("v pl \n", vpl)
    print("v mn \n", vmn)

def impute_matrix(XX, Q, M, i):
    # X input matrix
    # Q current inverse, Q = (X.T@X + lbd*Id)^(-1)
    # M masks, 0 seen, 1 missing
    # i current iteration when sweeping the column
    X = XX.copy()
    print("masks \n", M)
    n, d = X.shape
    xi = X[:, i]
    X_i = np.delete(X, i, axis=1)
    Q_i = np.delete(Q, i, axis=0)
    v = np.zeros(d-1)
    v = -(1 / Q[i, i]) * Q_i[:, i]
    #if i == 0:
    #    v = -(1/Q[0, 0]) * Q[1:, 0]
    #elif i== d:
    #    v = -(1/Q[d, d]) * Q[0:d-1, 0]
    #else:
    #    v[0:i] = Q[0:i, 0]
    #    v[(i+1):d] = Q[(i+1):d, 0]
    #    v = -(1/Q[i, i]) * v
    
    prediction = X_i @ v[:, None]
    print(v[:, None])
    print("test in impute matrix, who is v\n ", v)
    print(-Q * (1 / Q[i, i]))
    print(prediction, prediction.shape)
    #print(prediction.squeeze())
    #print(X[:, i])
    #print(M[:, i])
    #print(X[:, i] * (1 - M[:, i]))
    #print(prediction * M[:, i])
    # print(X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i])
    # print(X[:, i])
    X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i] + 0.0
    #X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i]
    #X[:, i] = np.zeros_like(X[:, i])
    print("new X\n", X)
    return X


def rk_1_update_inverse(Q, u, c):
    w = Q @ u
    print(Q)
    print(u)
    print(w)
    return Q - np.outer(w, w) / (1/c + np.sum(u * w))

    

def test_rk_1_update_inverse():
    n, d, c = 5, 3, -0.345
    X = np.random.randint(1, 5, size =(n, d))  + 0.0
    u = np.random.randint(1, 5, size = d)  + 0.0
    A = X.T @ X  + np.eye(d)  # (d, d)
    Q = np.linalg.inv(A)
    A_upd = A + c * np.outer(u, u)
    Q_upd_inv = np.linalg.inv(A_upd)
    Q_tested = rk_1_update_inverse(Q, u, c)
    print("manually\n ", Q_upd_inv, "\n by function\n", Q_tested)
    #print(Q_tested @ A_upd)
    np.testing.assert_allclose(Q_upd_inv, Q_tested)


def test_impute_matrix(X, M):
    n, d = X.shape
    print("masks ", M)
    for i in range(d):
        xi = X[:, i]
        X_i = np.delete(X, i, axis=1)
        alpha = 1.563
        #Q = np.linalg.inv(X_i.T @ X_i + alpha * np.eye(d-1))
        Q = np.linalg.inv(X.T @ X + alpha * np.eye(d))
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_i, xi)
        X = impute_matrix(X, Q, M, i)
        print(clf.coef_)
    print("masks\n ", M)





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
            X = impute_matrix(X, Q, M, i)
            N = Ms[:, i]
            X_upd, X_dwd = split_upd(X, N)



np.random.seed(54)
n = 5
d = 3
lbd = 1 + 0.0
X = np.random.randint(1, 9, size=(n, d)) + 0.0
M = np.random.binomial(1, 0.5, size=(n, d))
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': 2,
    'lbd_reg': lbd
}


print("new exppp, test impute matrix")
test_impute_matrix(X, M)

print("new exp")
test_rk_1_update_inverse()

print("new test")
test_split_upd()

print("\n\nnew exp ")
#gibb_sampl(info_dic)





