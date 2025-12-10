import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge, Ridge
import time
from imputations_method import multiple_imputation
from scipy.linalg import cho_factor, cho_solve
#from itertools import batched
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pandas as pd
from utils import flip_matrix_manual, update_inverse_rk2_sym, matrix_switches, swm_formula, rk_1_update_inverse
from serialization import serialization_first_idea


np.random.seed(53)
'''
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


Ms = matrix_switches(M)
print(Ms)

m1 = Ms[:, 0]
vp = Ms[m1 == 1, :]
print(vp)
'''


def make_mask_with_bounded_flip(n, d, p_miss, p_flip):
    M = np.zeros((n, d))
    #print(M)
    mask = np.random.binomial(1, p_miss, size=n)
    #print("first mask in make mask with bounded flip", mask)
    for i in range(d):
        M[:, i] = mask
        flip = np.random.binomial(1, p_flip, size=n)
        #print(flip)
        mask = (mask + flip) % 2
    ones = np.ones((d, d)) 
    F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #FF = flip_matrix(M.T)
    #print("flip matrix in make mask with bounded flip\n", F)
    #permutation, distance = solve_tsp_local_search(F)
    #print(permutation, distance)
    #print("test Flip matrix \n", FF)
    return M        


def split_upd(X, ms):
    # split the 1 rank perturbations in updates and downdates
    X_upd = X[ms == 1, :]
    X_dwd = X[ms == -1, :]
    return X_upd, X_dwd
    #return {'+': X_upd, '-': X_dwd}


def impute_matrix_under_parametrized(XX, Q, M, i):
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
    #vvv = -30 * (vv < 4).astype(int) + 18 * (vv >= 6).astype(int) + vv * ((vv >= 4) & (vv < 6)).astype(int) 
    #thr = 1.8
    #prediction = -thr * (prediction < -thr).astype(int) + thr * (prediction >= thr).astype(int) + prediction * ((prediction >= -thr) & (prediction < thr)).astype(int) 
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
    return X, v  # imputed matrix, coeff


def swm_formula(Q, U, c):
    # sherman woodbury morrison formula
    # compute the inverse of (Q + c*U.T@U)ˆ(-1)
    # Q (d, d)
    if U.ndim == 1 or U.shape[0] == 1 or U.shape[1] == 1:
        ret = rk_1_update_inverse(Q, U, c)
    else:
        d, m = U.shape  # U = [u_1|..|u_m], size = (d, m)
        #print("shape U ", d, m)
        #print(U)
        #print(Q)
        #print(Q.dtype)
        #print(U.dtype)
        #print("cond numb ", np.linalg.cond(Q))
        with np.errstate(over='raise'):
            w = U.T @ Q  # (m, d) * (d, d) = (m, d), cost = O(mdˆ2)
            #cn = np.linalg.cond(Q)
            #print("cond nbr ", cn)
            #print("max Q: ", np.max(Q), ", min Q ", np.min(Q))
            #cn = 1e1
            #if cn > 1e8:
            #    print(cn)
            if not np.all(np.isfinite(w)):
                print("Q: ", Q)
                print("cond numb:", np.linalg.cond(Q))
                print("Overflow detected inside block.")
                input("Paused. Press Enter to continue...")
            #try:
            #    w = U.T @ Q  # x = np.exp(1000)
            #except FloatingPointError:
            #    print("cond numb ", np.linalg.cond(Q))
            #    print("Overflow detected inside block.")
            #    input("Paused. Press Enter to continue...")
        
        #print("w \n\n\n", w)
        #print(w @ U)
        #print(w.shape)
        #print(U.shape)
        #cc, low = cho_factor(np.eye(m) / c + w @ U)
        #sol = cho_solve((cc, low), w)
        
        ## w @ U: (m, d) * (d, m) = (m, m):  O(mˆ2d) 
        
        sol = np.linalg.solve(np.eye(m) / c + w @ U, w)  #  O(mˆ3) * d times (there are d vectors in w) = O(m^3d)
        #print("sol ", sol)
        #print("trial ", (np.eye(m) / c + w @ U) @ sol)
        #print("w", w)
        ret = Q - w.T @ sol # the identity should be cancelled, it is just to mitigate the numerical errors but it shouldn't be there
    
        # total cost: O(mdˆ2 + mˆ2d + mˆ3d)
    return ret
    

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


def gibb_sampl_no_modification(info):
    # flip matrix
    X = info['data']
    M = info['masks']
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant')
    X = imp_mean.fit_transform(X_nan)
    #print("simple imputer in gibb sample \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    #b_s = int(np.sqrt(d))  # batch size  
    #b_s = 10
    #b_s = 5
    b_s = info['batch_size']
    #print("batch size ", b_s)
    if b_s <= 0:
        b_s = 1
    #print("who is X in gibb sampl \n", X)
    #ones = np.ones((d, d)) 
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #print("flip matrix\n", F)
    #if info['tsp']:
    #    start_time = time.time()
    #    permutation, distance = solve_tsp_local_search(F)
    #    end_time = time.time()
    #    print("optimal perm ", permutation, "optimal dist ", distance) 
    #    print(f"Execution time tsp: {end_time - start_time:.4f} seconds")
    #    M = M[:, permutation]
    #    X = X[:, permutation]
    #print("\n", X)
    #print("\n", M)
    #Ms = matrix_switches(M)
    #first_mask = M[:, 0]
    #print("\n ", first_mask)
    #X = X * (1/np.sqrt(n))  # normalize the column, so that the final matrix will be the covariance matrix 
    R = X  # X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    start1 = time.time()
    Rt_R = R.T @ R + lbd * np.eye(d)
    end1 = time.time()
    print("building the matrix time: ", end1-start1)        
    Q = np.linalg.inv(Rt_R)
    start_gibb_s = time.time()
    upd_j = np.zeros((d, 2))
    #print("initial X \n", X)
    for h in range(r):
        #print("iter ", h)
        for i in range(d):
            #print("index gibb sampl no mod", i)
            X_pre_upd = X
            X, _ = impute_matrix_under_parametrized(X, Q, M, i)
            #print("new X ", X)
            if info['verbose'] > 0:
                print(X)
            upd_j[i, 0] = 1
            #start1 = time.time()
            upd_j[:, 1] = X.T @ (X[:, i] - X_pre_upd[:, i])
            #end1 = time.time()
            #print("multiplication time: ", end1-start1)
            upd_j[i, 1] = np.sum((X[:, i] - X_pre_upd[:, i]) * (X[:, i] + X_pre_upd[:, i])) / 2  
            Q = update_inverse_rk2_sym(Q, upd_j)
            upd_j[i, 0] = 0
            #QQ = X.T @ X + lbd * np.eye(d)
            #QQ = np.linalg.inv(QQ)
            #print("small check QQ\n", QQ)
            #print("small check Q\n", Q)
            #np.testing.assert_allclose(Q, QQ)
    return X      
            
                
    #end_gibb_s = time.time()
    #print("res my imp \n", X)
    #print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    #return X


'''
def gibb_sampl(info):
    # flip matrix
    X = info['data']
    M = info['masks']
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(X_nan)
    #print("simple imputer in gibb sample \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    #b_s = int(np.sqrt(d))  # batch size  
    #b_s = 10
    #b_s = 5
    b_s = info['batch_size']
    #print("batch size ", b_s)
    if b_s <= 0:
        b_s = 1
    #print("who is X in gibb sampl \n", X)
    ones = np.ones((d, d)) 
    F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #print("flip matrix\n", F)
    if info['tsp']:
        start_time = time.time()
        permutation, distance = solve_tsp_local_search(F)
        end_time = time.time()
        print("optimal perm ", permutation, "optimal dist ", distance) 
        print(f"Execution time tsp: {end_time - start_time:.4f} seconds")
        M = M[:, permutation]
        X = X[:, permutation]
    #print("\n", X)
    #print("\n", M)
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    #print("\n ", first_mask)
    #X = X * (1/np.sqrt(n))  # normalize the column, so that the final matrix will be the covariance matrix 
    R = X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    Rt_R = R.T @ R + lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)
    start_gibb_s = time.time()

    for h in range(r):
        #print("iter ", h)
        if info['recomputation']:
            R = X[first_mask == 0, :]
            #print("first set vct ", R)
            print("first set vct shape ", R.shape)
            Rt_R = R.T @ R + lbd * np.eye(d)
            Q = np.linalg.inv(Rt_R)
        for i in range(d):
            #print("index ", i)
            X, _ = impute_matrix(X, Q, M, i)
            if h < r-1 or i < d-1: 
                N = Ms[:, i]
                X_upd, X_dwd = split_upd(X, N)
                #print(N)
                #print("sequence of print")
                if info['verbose'] > 0:
                    print(X)
                #print(X_upd)
                #print(X_dwd)
                nupd, _ = X_upd.shape
                ndwd, _ = X_dwd.shape
                if nupd + ndwd > n: #* 0.2:
                    idx = i+1 if i<d-1 else 0
                    print(idx)
                    R = X[M[:, idx] == 0, :]
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    Rt_R = R.T @ R + lbd * np.eye(d)
                    Q = np.linalg.inv(Rt_R)
                else:
                    #print("nbr update ", nupd)
                    #print("nbr dwdate ", ndwd)
                    #print("it ", i, "total number flip ", nupd + ndwd)
                    
                    #for i_up in range(nupd):
                    #    Q = rk_1_update_inverse(Q, X_upd[i_up, :], 1.0)
                    #for i_dw in range(ndwd):
                    #    Q = rk_1_update_inverse(Q, X_dwd[i_dw, :], -1.0)
                
                    i_up = 0
                    while (i_up + 1) * b_s < nupd:
                        #print("current max ", (i_up + 1) * b_s, "total nbr upd ", nupd)
                        Q = swm_formula(Q, X_upd[i_up * b_s:(i_up + 1) * b_s, :].T, 1.0)
                        i_up = i_up + 1
                    Q = swm_formula(Q, X_upd[i_up * b_s:nupd, :].T, 1.0)
                    #print("cond nub Q before dwd: ", np.linalg.cond(Q))
                    i_dw = 0
                    while (i_dw + 1) * b_s < ndwd:
                        #print("current max ", (i_dw + 1) * b_s, "total nbr dw ", ndwd)
                        #print("shape dwd ", X_upd[i_dw * b_s:(i_dw + 1) * b_s, :].shape)
                        Q = swm_formula(Q, X_dwd[i_dw * b_s:(i_dw + 1) * b_s, :].T, -1.0)
                        i_dw = i_dw + 1
                    #print("outside the cycle ", i_dw * b_s)
                    Q = swm_formula(Q, X_dwd[i_dw * b_s:ndwd, :].T, -1.0)
                    #print("QQ\n ", QQ)
                    #print("Q\n", Q)
                    #print("cond nub Q in gibb sampl: ", np.linalg.cond(Q))
    end_gibb_s = time.time()
    #print("res my imp \n", X)
    print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    return X
'''


def impute_matrix_overparametrized(X, M, K ,K_inv, lbd, idx):
    n_m = np.sum(M[:, idx])  #  nbr missing, M_ij = 1 iff component is missing
    n_s = np.sum(1-M[:, idx])
    _, d = X.shape

    #print("K\n", K)
    #print("K_inv\n", K_inv)
    #print("nplinalg.inv(K)\n", np.linalg.inv(K))

    X_idx = X[:, idx]
    X_s = X_idx[M[:, idx] == 0]  # (n_s,)
    A = K_inv[M[:, idx] == 1][:, M[:, idx] == 0]  # (n_m, n_s)
    #print("dim A ", A.shape, ", nbr miss ", n_m, "nbr seen ", n_s) 
    if A.ndim == 1:
        A = np.array([A])
    if n_m < n_s:  # not many missing components 
        #print("n_m < n_s")
        #print("M \n", M)
        #X_del = np.delete(X, idx, axis=1)
        
        #X.copy()
        #C = K_inv[M[:, idx] == 1, :]  #  (n_m, n)
        #print("C \n", C)
        #S_C = C[0:n_m, 0:n_m]  #  Schur Complement
        #print(K_inv)
        S_C = K_inv[M[:, idx] == 1, :][:, M[:, idx] == 1]
        if S_C.ndim == 1:
            S_C = np.array([S_C])
        #print("S_C \n", S_C)
        #print("A\n", A)
        
        #print("X_s ", X_s)
        x = np.linalg.solve(S_C, A @ X_s)
        #print("x\n ", x)
        #print("check ", np.sqrt( np.sum( (S_C @ x - A@X_s)**2) ) )
        #print("M \n ", M[:, idx])
        #X[M[:, idx] == 1, idx] = -x
    else:  # many missing components, it's better to work with the the submatrix of seen components
        #print("n_s < n_m")
        K_S = K[M[:, idx] == 0, :][:, M[:, idx] == 0]  # submatrix of seen components
        #print("K_S\n", K_S)
        v = A @ X_s  # (n_m, n_s) * (n_s,) = (n_m,) 
        X_del = np.delete(X, idx, axis=1)
        Xm = X_del[M[:, idx] == 1, :]  # (n_m, d-1) 
        Xs = X_del[M[:, idx] == 0, :]  # (n_s, d-1)
        #print("Xs @ Xs.T + lbd * Id\n", Xs @ Xs.T + lbd * np.eye(n_s))
        w = Xm.T @ v  # (d-1, n_m) * (n_m,) = (d-1,)
        partial = w - Xs.T @ np.linalg.solve(K_S, Xs @ w ) 
        x = Xm @ partial + lbd * v
#        print("x\n ", x)
#        S_CC = Xm @ Xm.T - (Xm @ Xs.T) @ np.linalg.inv(K_S) @ (Xs @ Xm.T) + lbd * np.eye(n_m)
#        S_C_inv = np.linalg.inv(S_CC)
#        S_C_true = K_inv[M[:, idx] == 1, :][:, M[:, idx] == 1]  # this works in the other branch
#        print("SC_inv\n", S_C_inv)
#        print("K_inv\n", K_inv)
#        print("\n\nSC true\n ", S_C_true)
#        print("S_C_inv @ v\n ", S_CC @ v)
#        S_C_true_inv = np.linalg.inv(S_C_true)
#        print("SC true inv (from K_inv)\n", S_C_true_inv)
#        print("SCS computed by hand\n", S_CC)
#        x = np.linalg.inv(S_C_true) @ v
    X[M[:, idx] == 1, idx] = -x

    '''
    Xm = X_del[M[:, idx] == 1, :]
    Xs = X_del[M[:, idx] == 0, :]
    #print("split X\n", X)
    #print("\n", Xm)
    #print(Xs)
    one = Xm @ Xs.T
    two = np.linalg.inv((Xs @ Xs.T + lbd * np.eye(n_s)))
    res = -one @ two @ X_s
    #print("check result1 in impute matrix overparamettrized\n", res)

    one1 = Xm
    two1 = np.linalg.inv((Xs.T @ Xs + lbd * np.eye(d-1)))
    res1 = -one1 @ two1 @ (Xs.T @ X_s)
    #print("check result1 in impute matrix overparamettrized\n", res1)
    '''
    return X


def gibb_sampl_over_parametrized(info):
    ## Gibb sampling in an overparametrized setting
    X = info['data']
    M = info['masks']
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = imp_mean.fit_transform(X_nan)
    #print("simple imputer in gibb sample overparametrized \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    nbr_it_gs = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape  # suppose n < d
    X_del = np.delete(X, 0, axis=1)
    K = X_del @ X_del.T + lbd * np.eye(n)  # (n, n)
    K_inv = np.linalg.inv(K)
    for h in range(nbr_it_gs):
        for i in range(d):
            #print("index ", i)
            #idx = i if i<d-1 else 0
            X = impute_matrix_overparametrized(X=X, M=M, K=K, K_inv=K_inv, lbd=lbd, idx=i)
            #print("round ", i, ": imputed matrix gs overp\n", X)
            if h < nbr_it_gs-1 or i < d-1:
                v_to_add = X[:, i]
                v_to_remove = X[:,(i+1)] if i<d-1 else X[:, 0]
                K = K + np.outer(v_to_add, v_to_add) - np.outer(v_to_remove, v_to_remove)
                #if i == d-1:
                #    v_to_remove = X[:, 0]
                K_inv = swm_formula(K_inv, v_to_add, 1.0)
                K_inv = swm_formula(K_inv, v_to_remove, -1.0)
    return X


def gibb_sampl_under_parametrized(info):
    # flip matrix
    X = info['data']
    M = info['masks']
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = imp_mean.fit_transform(X_nan)
    #print("simple imputer in gibb sample under param \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    #b_s = int(np.sqrt(d))  # batch size  
    #b_s = 10
    #b_s = 5
    #print("who is X in gibb sampl \n", X)
    ones = np.ones((d, d)) 
    #start_algo_gibb_s_partial = time.time()
    s = np.ones_like(M.T)
    ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #FF = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M.T @ M
    #np.testing.assert_allclose(F, FF)
    #end_algo_gibb_s_partial = time.time()
    #print(f"Elapsed time gibb sampl, cov matrix masks: {end_algo_gibb_s_partial - start_algo_gibb_s_partial:.4f} seconds\n\n")
    start_algo_gibb_s_partial = time.time()
    s = np.ones_like(M.T)
    ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M.T @ M
    #np.testing.assert_allclose(F, FF)
    end_algo_gibb_s_partial = time.time()
    print(f"Elapsed time gibb sampl, cov matrix, M: {end_algo_gibb_s_partial - start_algo_gibb_s_partial:.4f} seconds\n\n")
    start_algo_gibb_s_partial_sparse = time.time()
    M_s = csr_matrix(M)
    ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
    #print("type flip matrix ", type(F))
    print("flip matrix head\n" , F[0:8, 0:8])
    #np.testing.assert_allclose(F, FF)
    end_algo_gibb_s_partial_sparse = time.time()
    print(f"Elapsed time gibb sampl, cov matrix, M sparse: {end_algo_gibb_s_partial_sparse - start_algo_gibb_s_partial_sparse:.4f} seconds\n\n")
    
    #print("flip matrix\n", F)
    if info['tsp']:
        start_time = time.time()
        #permutation, distance = solve_tsp_local_search(F)
        permutation, distance = serialization_first_idea(F)
        current_distance = distance
        current_permutation = permutation
        
        original_cost = np.sum(np.diag(F, k=1))
        print("original cost ", original_cost)
        #print("optimal perm ", permutation, "optimal dist ", distance) 
        distances = []
        distances.append(distance)
        s = int(np.floor(np.sqrt(d)))
        for i in range(s):
            permutation, distance = serialization_first_idea(F)
            distances.append(distance)
            if distance < current_distance:
                current_distance = distance
                current_permutation = permutation
        M = M[:, current_permutation]
        X = X[:, current_permutation]
        print("distances tsp ", np.array(distances))
        end_time = time.time()
        print(f"Execution time tsp: {end_time - start_time:.4f} seconds")

    #print("exponent d ", info['exponent_d'])
    #print("\n", X)
    #print("\n", M)
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    #print("\n ", first_mask)
    #X = X * (1/np.sqrt(n))  # normalize the column, so that the final matrix will be the covariance matrix 
    R = X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    start_gibb_s = time.time()
    Rt_R = R.T @ R + lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)
    counter_upd_dwd = 0
    counter_recomputation = 0
    counter_swm_formula = 0 
    counter_reinversion = 0
    print("d ** exp: ", d ** info['exponent_d'])
    for h in range(r):
        for i in range(d):
            #print("index ", i)
            X, _ = impute_matrix_under_parametrized(X, Q, M, i)
            #print("round ", i, "who is X gs\n", X)
            #v = X.T @ X[:, i]
            #Rt_R[i, :] = v
            #Rt_R[:, i] = v
            #Rt_R
            #print("who is Rt_R \n", Rt_R)
            if h < r-1 or i < d-1:
                N = Ms[:, i]
                X_upd, X_dwd = split_upd(X, N)
                #print(N)
                #print("sequence of print")
                if info['verbose'] > 0:
                    print(X)
                #print(X_upd)
                #print(X_dwd)
                nupd, _ = X_upd.shape
                ndwd, _ = X_dwd.shape
                '''
                if nupd + ndwd > n:
                    idx = i+1 if i<d-1 else 0
                    print(idx)
                    R = X[M[:, idx] == 0, :]
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    Rt_R = R.T @ R + lbd * np.eye(d)
                    Q = np.linalg.inv(Rt_R)
                '''
                idx = i+1 if i<d-1 else 0
                #print("nbr seen ", n - np.sum(M[:, 0]), " nbr flip ", nupd + ndwd)
                if n - np.sum(M[:, idx]) < nupd + ndwd:  # if nbr seen component is less than nbr of flips
                    #print("recompute the matrix with the missing components")
                    counter_recomputation = counter_recomputation + 1
                    R = X[M[:, idx] == 0, :]
                    Rt_R = R.T @ R + lbd * np.eye(d)

                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd

                else:
                    counter_upd_dwd = counter_upd_dwd + 1
                    #print("update the covariance matrix") 
                    Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                    #RR = X[M[:, idx] == 0, :]
                    #Rt_RR = RR.T @ RR + lbd * np.eye(d)
                    #np.testing.assert_allclose(Rt_R, Rt_RR)
                if nupd + ndwd > d ** info['exponent_d']:
                    #print("invert the matrix")
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd too big, invert the matrix ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    #idx = i+1 if i<d-1 else 0
                    #print(idx)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    counter_reinversion = counter_reinversion + 1
                    Q = np.linalg.inv(Rt_R)
                else:
                    counter_swm_formula = counter_swm_formula + 1
                    #print("low rank upd of the inverse")
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd small, swm formula.          ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    Q = swm_formula(Q, X_upd.T, 1.0)
                    Q = swm_formula(Q, X_dwd.T, -1.0)
                    #for i_up in range(nupd):
                    #    Q = rk_1_update_inverse(Q, X_upd[i_up, :], 1.0)
                    #for i_dw in range(ndwd):
                    #    Q = rk_1_update_inverse(Q, X_dwd[i_dw, :], -1.0)
                    #print("QQ\n ", QQ)
                    #print("Q\n", Q)
                    #print("cond nub Q in gibb sampl: ", np.linalg.cond(Q))
    end_gibb_s = time.time()
    print("counter recomp ", counter_recomputation/r)
    print("counter upd dwd ", counter_upd_dwd/r)
    print("counter reinv", counter_reinversion/r)
    print("counter swm ", counter_swm_formula/r)
    #print("res my imp \n", X)
    print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    return X


def gibb_sampl(info):
    X = info['data']
    n, d = X.shape
    return gibb_sampl_under_parametrized(info) if n>=d else gibb_sampl_over_parametrized(info)


np.random.seed(543)


def plot_some_graph():
    print("\n\nstarting plot some graph()\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [1000, 2000, 4000, 6000]  # increasing order
    list_d = [100]  # increasing order
    lbd = 1 + 0.0
    n, d = list_n[-1], list_d[-1]
    print("sqrt n ", np.sqrt(n), "n ** (3/4) / n", (n ** (3/4)) / n)
    print("n ** (3/4)", n ** (3/4))
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    X = (X_orig - mean) / std
    X = X_orig
    X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    print(np.max(X))
    print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    exponent = (n ** (3/4)) / n
    print("exponent", exponent)
    p1 = 1/2 - np.sqrt(1 - 2 * d/n)/2 if 2 * d/n>0 else d/(2 * n)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.1, p_flip=p1)
    p1 = 0.4
    #print("p1:   ", p1)
    M = np.random.binomial(n=1, p=p1, size= (n, d))
    #p_missing = [0.8 , 0.6, 0.3]
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    tsp_switch = False
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_miss'])
    print(df)
    total_time_gibb_sampl = np.zeros((len(list_n), len(list_d)))
    total_time_ridge = np.zeros_like(total_time_gibb_sampl)
    total_time_baseline = np.zeros_like(total_time_gibb_sampl)
    for i, d_i in enumerate(list_d):
        print("\ncurrent dimension ", d_i)
        for j, n_j in enumerate(list_n):
            print("\n\n current size ", n_j)
            ones = np.ones((d_i, d_i))
            MM = M[0:n_j, 0:d_i]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            print("nbr seen components ", n_j - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n_j * p1 * (1-p1))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n_j, 0:d_i],
                'masks': M[0:n_j, 0:d_i],
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': tsp_switch,
                'recomputation': False,
                'batch_size': 64,
                'verbose': 0
            }
            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl(info_dic)
            end_time_gibb_sampl = time.time()
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
        #   print(X_my)
            total_time_gibb_sampl[j, i] = end_time_gibb_sampl - start_time_gibb_sampl
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            start44 = time.time()   # tic
            ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='mean', verbose=0)
            end44 = time.time()   # tic
            print(f"Elapsed time no 4 iterative imputer definition: {end44 - start44:.4f} seconds\n\n")

            start4 = time.time()
            res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            #print("result IterativeImptuer with Ridge\n", res4)
            end4 = time.time()     # toc
            total_time_ridge[j, i] = end4 - start4 
            print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds\n\n")
            np.testing.assert_allclose(X_my, res4)

            start_baseline = time.time()   # tic
            #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            X_my_baseline = gibb_sampl_no_modification(info_dic)  
            # print("result IterativeImptuer with Ridge\n", res4)
            end_baseline = time.time()     # toc
            total_time_baseline[j, i] = end_baseline - start_baseline
            print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
            #if not info_dic['tsp']:
            #np.testing.assert_allclose(X_my, res4)
            #np.testing.assert_allclose(X_my, res4)
            print("test gibb sampl ended successfully")    
    print("total time gibb sampl\n", total_time_gibb_sampl)
    print("total time ridge\n", total_time_ridge)
    print("total time baseline\n", total_time_baseline)
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    for i, d_i in enumerate(list_d):
        plt.plot(list_n, total_time_gibb_sampl[:, i], label="our_gibb, dim: " + str(d_i), marker="o", color=clr[i])
        plt.plot(list_n, total_time_ridge[:, i], label="ridge  , dim: " + str(d_i), marker="*", color=clr[i+1])
        plt.plot(list_n, total_time_baseline[:, i], label="baseline  , dim: " + str(d_i), marker="s", color=clr[i+2])
        #plt.plot(iterations, accuracy, label="Accuracy", color="blue")
        plt.xlabel("train size")
        plt.ylabel("time")
    plt.title("Time in function of training size")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.text(5.05, 0.5, "ciao sono un testo", rotation=0)
    text = "MCAR p_miss: " + str(p1) + "\n\n"
    #text = "prob flip: " + str(p1) + "\n\n"
    text1 = "tsp: " + str(tsp_switch) + "nbr it: " + str(R)

    plt.figtext(0.71, 0.65, "Extra info about curves:\n" + text + text1, fontsize=10)
    plt.tight_layout() 
    #plt.legend()
    plt.show()

## to do: run experiments with something like n=700, d=500, and study the result
## you can see that if the percentage of missing is greater than 0.5, running tsp is actually useful

def plot_some_graph_2():
    print("\n\nstarting plot some graph 2(). In this function we go through the probabilities\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [4000]  # increasing order
    list_d = [200]  # increasing order
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    list_p_seen_true = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.05, 0.01, 0.005]
    list_p_seen = list_p_seen_true[:-1]
    list_p_seen.insert(0, 1.0)
    print("list p seen true ", list_p_seen_true)
    print("list prob        ", list_p_seen)
    list_p_seen = [list_p_seen_true[i] / list_p_seen[i] for i in range(len(list_p_seen))]
    print("list p _seen ", list_p_seen)
    print("true probabilities, cumprod ", np.cumprod(list_p_seen)) 
    lbd = 1.01 + 0.0
    n, d = list_n[-1], list_d[-1]
    #print("sqrt n ", np.sqrt(n), "n ** (3/4) / n", (n ** (3/4)) / n)
    #print("n ** (3/4)", n ** (3/4))
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    X = (X_orig - mean) / std
    X = X_orig
    X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    print(np.max(X))
    print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    #p1 = 1/2 - np.sqrt(1 - 2 * d/n)/2 if 2 * d/n>0 else d/(2 * n)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.1, p_flip=p1)
    #p1 = 0.4
    #print("p1:   ", p1)
    #M = np.random.binomial(n=1, p=p1, size= (n, d))
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    
    #X_nan = X.copy()
    #X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    tsp_switch = False
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print(df)
    total_time_gibb_sampl = np.zeros((len(list_n), len(list_d)))
    total_time_ridge = np.zeros_like(total_time_gibb_sampl)
    total_time_baseline = np.zeros_like(total_time_gibb_sampl)
    for i, d_i in enumerate(list_d):
        print("\ncurrent dimension ", d_i)
        for j, n_j in enumerate(list_n):
            for s in list_p_seen:
                print(s)
            masks = np.array([np.random.binomial(1, 1-pr, (n_j, d_i)) for pr in list_p_seen])
            masks = np.cumsum(masks, axis=0)  # each round
            masks[masks>1] = 1
            for k, p_k in enumerate(list_p_seen_true):
                print("\n\n current size ", n_j)
                M = masks[k, :, :]
                for ii in range(d_i):
                    nbr = np.random.randint(0, n_j)
                    #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
                    if np.sum(M[:, ii]) == n_j:
                        print("add a random seen component")
                        M[nbr, ii] = 0
                X_nan = X.copy()
                X_nan[M==1] = np.nan
                print("X_nan \n", X_nan)
                ones = np.ones((d_i, d_i))
                MM = M[0:n_j, 0:d_i]
                #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
                print("nbr seen components ", n_j - np.sum(MM, axis=0))
                print("nbr missing components ", np.sum(MM, axis=0))
                print("2 * n * p1 * (1-p1):   ", 2 * n_j * p_k * (1-p_k))
                
                #FF = flip_matrix(M.T)
                #ones_d = np.ones(d_i)
                #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
                #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
                #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
                info_dic = {
                    'data': X[0:n_j, 0:d_i],
                    'masks': M,  #M[k, :, :],
                    'nbr_it_gibb_sampl': R,
                    'lbd_reg': lbd,
                    'tsp': tsp_switch,
                    'recomputation': False,
                    'batch_size': 64,
                    'verbose': 0,
                    'initial_strategy': 'constant',
                    'exponent_d': 0.75
                }
                start_time_gibb_sampl = time.time()
                X_my = gibb_sampl(info_dic)
                end_time_gibb_sampl = time.time()
                print("current prob seen ", p_k)
                print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
            #   print(X_my)
                t_my = end_time_gibb_sampl - start_time_gibb_sampl
                total_time_gibb_sampl[j, i] = t_my  # end_time_gibb_sampl - start_time_gibb_sampl
                print("\nend my gibb sampling\n")

                print("It imputer Ridge Reg")
                #start_skl = time.time()   # tic
                ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='constant', verbose=0)
                #end_skl = time.time()   # tic
                #print(f"Elapsed time no 4 iterative imputer definition: {end_skl_ptl - start_skl_ptl:.4f} seconds\n\n")

                start_skl = time.time()
                res_skl = ice_skl.fit_transform(X_nan[0:n_j, 0:d_i])
                #print("result IterativeImptuer with Ridge\n", res4)
                end_skl = time.time()     # toc
                t_skl = end_skl - start_skl
                total_time_ridge[j, i] = end_skl - start_skl 
                print("current prob seen ", p_k)
                print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end_skl - start_skl:.4f} seconds\n\n")
                if info_dic['tsp'] == False:
                    np.testing.assert_allclose(X_my, res_skl)
                #np.testing.assert_allclose(X_my, res_skl)

                start_baseline = time.time()   # tic
                #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
                #X_my_baseline = gibb_sampl_no_modification(info_dic)
                info_dic['tsp'] = False
                X_my_baseline = gibb_sampl(info_dic)  
                # print("result IterativeImptuer with Ridge\n", res4)
                end_baseline = time.time()     # toc
                t_bsl = end_baseline - start_baseline
                total_time_baseline[j, i] = end_baseline - start_baseline

                df.loc[len(df)] = [n_j, d_i, p_k, t_my, t_skl, t_bsl]
                print("current prob seen ", p_k)
                print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
                #if not info_dic['tsp']:
                #np.testing.assert_allclose(X_my, res4)
                #print("test baseline ended successfully")   
    print("\n\n SHOW THE RESULTS")
    dd = d ** info_dic['exponent_d']
    p1 = 1/2 - np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
    p2 = 1/2 + np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
    ## when probability = p1 or p2, then 2n(p-1)p ~ d
    ## observe, 2n(p-1)p < np if p > (1/2), so if p greater than (1/2),
    ## the average number of seen component is grater than the average number of flip 
    ## if prob = d/n, the number of seen components is ~ d = n * (d/n)
    print("d ** ", info_dic['exponent_d'], ": ", dd)
    print("p1 ", p1, ",  p2 ", p2,  ",   d/n ", d/n)
    print("df \n", df) 
    print("total time gibb sampl\n", total_time_gibb_sampl)
    print("total time ridge\n", total_time_ridge)
    print("total time baseline\n", total_time_baseline)
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    for i, d_i in enumerate(list_d):
        plt.plot(list_p_seen_true, df['time_my'], label="our_gibb, dim: " + str(d_i), marker="o", color=clr[i])
        plt.plot(list_p_seen_true, df['time_skl'], label="ridge  , dim: " + str(d_i), marker="*", color=clr[i+1])
        plt.plot(list_p_seen_true, df['time_bsl'], label="baseline  , dim: " + str(d_i), marker="s", color=clr[i+2])
        #plt.plot(iterations, accuracy, label="Accuracy", color="blue")
        plt.axvline(x = p1, linestyle='--', linewidth=2, label="p1: sol 2np(1-p)=d^" +  str(info_dic['exponent_d']))
        plt.axvline(x = p2, linestyle='--', linewidth=2, label="p2: sol 2np(1-p)=d^" +  str(info_dic['exponent_d']))
        plt.axvline(x = d/n, linestyle='--', linewidth=1, label="d/n")
        plt.axvline(x = 1/2, linestyle='--', linewidth=0.5, label="1/2")
        #plt.axvline(x = d ** (3/4)/n, linestyle='--', linewidth=0.5)
        #plt.axvline(x = (1-d/n) * (d/n), linestyle='--', linewidth=2)
        #plt.axvline(x = 1-(1-d/n) * (d/n), linestyle='--', linewidth=2)
        #plt.axvline(x = (1-d/n) * d/n * (1/2), linestyle='--', linewidth=3)
        #plt.axvline(x = 1-(1-d/n) * d/n * (1/2), linestyle='--', linewidth=3)
        plt.xlabel("prob seen")
        plt.ylabel("time")
    plt.title("Time in function of training size")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.text(5.05, 0.5, "ciao sono un testo", rotation=0)
    #text = "MCAR p_miss: " + str(p1) + "\n\n"
    #text = "prob flip: " + str(p1) + "\n\n"
    #text1 = "tsp: " + str(tsp_switch) + "nbr it: " + str(R)

    #plt.figtext(0.71, 0.65, "Extra info about curves:\n" + text + text1, fontsize=10)
    text = "Extra info about curves\n"
    text0 = "nbr train: " + str(n) + "\n\n"
    text1 = "right of the line d/n: nbr_seen> d\n\n"
    text2 = "left of the line d/n : nbr_seen< d\n\n"
    text3 = "between the lines p1,p2: nbr_flip > d ** " + str(info_dic['exponent_d']) + " = " + str(d ** info_dic['exponent_d']) + "\n\n"
    text4 = "right line (1/2): number seen greater than number flips\n\n"
    text5 = "left line (1/2):  number seen smaller than number flips\n\n"
    plt.figtext(0.68, 0.5, text1 + text2 + text3 + text4, fontsize=10)
    plt.tight_layout()
    #plt.legend()
    plt.show()



plot_some_graph_2()



'''
n = 8
d = 3
lbd = 1 + 0.0
X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
#X_orig = np.random.rand(n, d) + 0.0
print(X_orig.dtype)
print("max min ", )
mean = np.mean(X_orig, axis=0)
std = np.std(X_orig, axis=0)
# Standardize
X = (X_orig - mean) / std
X = X_orig
print(np.max(X))
print(np.min(X))
M = np.random.binomial(1, 0.2, size=(n, d))
X_nan = X.copy()
X_nan[M==1] = np.nan
print("X_nan \n", X_nan)
R = 1
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': R,
    'lbd_reg': lbd,
    'tsp': False,
    'recomputation': False
}
'''



'''
print("new exppp, test impute matrix")
#test_impute_matrix(X, M)

print("new exp")
#test_rk_1_update_inverse()

print("new test")
#test_split_upd()

print("\n\nnew exp ")
start_time_gibb_sampl = time.time()
#gibb_sampl(info_dic)
end_time_gibb_sampl = time.time()

print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
print("\nend my gibb sampling\n")

#ice = IterativeImputer(estimator=BayesianRidge(), max_iter=R, initial_strategy='mean')
start2 = time.time()   # tic
#res1 = ice.fit_transform(X_nan)
end2 = time.time()     # toc
print(f"Elapsed time no 1 simple imputer  prec: {end2 - start2:.4f} seconds")

print("ciao")
start3 = time.time()   # tic
#res2 = multiple_imputation({'mi_nbr':1, 'nbr_feature':None, 'max_iter': R}, X_nan)
#print(res2)
end3 = time.time()     # toc
print(f"Elapsed time no 2 iter imputer  prec: {end3 - start3:.4f} seconds")


print("It imputer Ridge Reg")
ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False), max_iter=R, initial_strategy='mean', verbose=0)
start4 = time.time()   # tic
res4 = ice4.fit_transform(X_nan)
print("result IterativeImptuer \n", res4)
end4 = time.time()     # toc
#print("res aftet Ridge \n", res4)
print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds")
'''

