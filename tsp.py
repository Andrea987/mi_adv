import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from utils import flip_matrix, generate_binary_arrays
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge
import time
from imputations_method import multiple_imputation
from scipy.linalg import cho_factor, cho_solve
#from itertools import batched




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
    #print("masks\n ", m)
    ms = matrix_switches(m)
    #print("ms \n ", ms)
    vpl, vmn = split_upd(X, ms[:, 0])
    #print("X\n ", X)
    #print("updates/downdates, \n", ms)
    #print("v pl \n", vpl)
    #print("v mn \n", vmn)

def impute_matrix(XX, Q, M, i):
    # X input matrix
    # Q current inverse, Q = (X.T@X + lbd*Id)^(-1)
    # M masks, 0 seen, 1 missing
    # i current iteration when sweeping the column
    X = XX.copy()
    #print("masks \n", M)
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
    #print(v[:, None])
    #print("test in impute matrix, who is v\n ", v)
    #print(-Q * (1 / Q[i, i]))
    #print(prediction, prediction.shape)
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
    #print("new X\n", X)
    return X

def batch_upd_dwt():
    x = 1






def swm_formula(Q, U, c):
    # sherman woodbury morrison formula
    # compute the inverse of (Q + c*U.T@U)ˆ(-1)
    if U.ndim == 1 or U.shape[0] == 1 or U.shape[1] == 1:
        ret = rk_1_update_inverse(Q, U, c)
    else:
        d, m = U.shape  # U = [u_1|..|u_m], size = (d, m)
        #print(U)
        #print(Q)
        w = U.T @ Q
        #print("w \n\n\n", w)
        #print(w @ U)
        #print(w.shape)
        #print(U.shape)
        #cc, low = cho_factor(np.eye(m) / c + w @ U)
        #sol = cho_solve((cc, low), w)
        sol = np.linalg.solve(np.eye(m) / c + w @ U, w)
        #print("sol ", sol)
        #print("trial ", (np.eye(m) / c + w @ U) @ sol)
        #print("w", w)
        ret = Q - w.T @ sol
    return ret


def test_swm():
    n, d, m, c = 6, 3, 2, 0.2
    X = np.random.randint(1, 5, size = (n, d))  + 0.0
    U = np.random.randint(1, 5, size = (m, d))  + 0.0
    A = X.T @ X  + np.eye(d)  # (d, d)
    Q = np.linalg.inv(A)
    A_upd = A + c * U.T @ U
    print(X)
    print(U)
    Q_upd_inv = np.linalg.inv(A_upd)
    Q_tested = swm_formula(Q, U.T, c)
    print("manually\n ", Q_upd_inv, "\n by function\n", Q_tested)
    #print(Q_tested @ A_upd)
    np.testing.assert_allclose(Q_upd_inv, Q_tested)

test_swm()
print("end test swm successfully")

def rk_1_update_inverse(Q, u, c):
    #print(Q)
    #print("u in rk. upd inverse ", u.ndim)
    if u.ndim > 1:
        #print("vector squeezed in rk 1 upd")
        u = np.squeeze(u)
    w = Q @ u
    #print(Q)
    #print(u)
    #print(w)
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
    b_s = int(np.sqrt(d))  # batch size
    #b_s = 1 
    print("batch size ", b_s)
    if b_s <= 0:
        b_s = 1
    #print("who is X in gibb sampl \n", X)
    ones = np.ones((d, d)) 
    F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    print("flip matrix ", F)
    permutation, distance = solve_tsp_local_search(F)
    print("optimal perm ", permutation, "optimal dist ", distance) 
    M = M[:, permutation]
    X = X[:, permutation]
    #print("\n", X)
    #print("\n", M)
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    #print("\n ", first_mask)
    R = X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    Rt_R = R.T @ R + (1/n) * lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)


    for h in range(r):
        print("iter ", h)
        for i in range(d):
            #print("index ", i)
            X = impute_matrix(X, Q, M, i)
            N = Ms[:, i]
            X_upd, X_dwd = split_upd(X, N)
            #print(N)
            #print("sequence of print")
            #print(X)
            #print(X_upd)
            #print(X_dwd)
            nupd, _ = X_upd.shape
            ndwd, _ = X_dwd.shape
            
            #for i_up in range(nupd):
            #    Q = rk_1_update_inverse(Q, X_upd[i_up, :], 1.0)
            #for i_dw in range(ndwd):
            #    Q = rk_1_update_inverse(Q, X_dwd[i_dw, :], -1.0)
            
            i_up = 0
            while (i_up + 1) * b_s < nupd:
                Q = swm_formula(Q, X_upd[i_up * b_s:(i_up + 1) * b_s, :].T, 1.0)
                i_up = i_up + 1
            Q = swm_formula(Q, X_upd[i_up * b_s:nupd, :].T, 1.0)

            i_dw = 0
            while (i_dw + 1) * b_s < nupd:
                Q = swm_formula(Q, X_upd[i_dw * b_s:(i_dw + 1) * b_s, :].T, -1.0)
                i_dw = i_dw + 1
            Q = swm_formula(Q, X_upd[i_dw * b_s:nupd, :].T, -1.0)





np.random.seed(54)
n = 1500
d = 100
lbd = 1 + 0.0
X = np.random.randint(1, 9, size=(n, d)) + 0.0
M = np.random.binomial(1, 0.2, size=(n, d))
X_nan = X.copy()
X_nan[M==1] = np.nan
R = 3
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': R,
    'lbd_reg': lbd
}


print("new exppp, test impute matrix")
#test_impute_matrix(X, M)

print("new exp")
#test_rk_1_update_inverse()

print("new test")
#test_split_upd()

print("\n\nnew exp ")
start_time = time.time()
gibb_sampl(info_dic)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.4f} seconds")


ice = IterativeImputer(estimator=BayesianRidge(), max_iter=d * R, initial_strategy='mean')
start2 = time.time()   # tic
res1 = ice.fit_transform(X_nan)
end2 = time.time()     # toc
print(f"Elapsed time no 1 simple imputer  prec: {end2 - start2:.4f} seconds")


start3 = time.time()   # tic
#res2 = multiple_imputation({'mi_nbr':1, 'nbr_feature':None, 'max_iter': d * R}, X_nan)
#print(res2)
end3 = time.time()     # toc
print(f"Elapsed time no 2 iter imputer  prec: {end3 - start3:.4f} seconds")


