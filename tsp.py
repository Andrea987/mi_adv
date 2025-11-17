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

from utils import flip_matrix, update_inverse_rk2_sym, matrix_switches, swm_formula, rk_1_update_inverse



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
    print("first mask in make mask with bounded flip", mask)
    for i in range(d):
        M[:, i] = mask
        flip = np.random.binomial(1, p_flip, size=n)
        #print(flip)
        mask = (mask + flip) % 2
    ones = np.ones((d, d)) 
    F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #FF = flip_matrix(M.T)
    print("flip matrix in make mask with bounded flip\n", F)
    #permutation, distance = solve_tsp_local_search(F)
    #print(permutation, distance)
    #print("test Flip matrix \n", FF)
    return M        

maskkk = make_mask_with_bounded_flip(n=2, d=10, p_miss=0.3, p_flip=0.1)
#print("masksss\n", maskkk)


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
            w = U.T @ Q  # x = np.exp(1000)
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
        sol = np.linalg.solve(np.eye(m) / c + w @ U, w)
        #print("sol ", sol)
        #print("trial ", (np.eye(m) / c + w @ U) @ sol)
        #print("w", w)
        ret = Q - w.T @ sol # the identity should be cancelled, it is just to mitigate the numerical errors but it shouldn't be there
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

#test_swm()
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


def test_impute_matrix():
    print("beginning test impute matrix ")
    n, d = 30, 5
    X = np.random.randint(1, 5, size=(n, d))
    M = np.random.binomial(1, 0.2, size=(n, d))
    #print("masks in test impute matrix\n", M)
    for i in range(d):
        xi = X[:, i]
        X_i = np.delete(X, i, axis=1)
        alpha = 0.3213
        #Q = np.linalg.inv(X_i.T @ X_i + alpha * np.eye(d-1))
        Q = np.linalg.inv(X.T @ X + alpha * np.eye(d))
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_i, xi)
        X, my_coeff = impute_matrix(X, Q, M, i)
        #print("my coeff from Ridge  ", my_coeff)
        #print("coeff from Ridge fit ", clf.coef_)
        np.testing.assert_allclose(my_coeff, clf.coef_)
    #print("masks in test impute matrix\n", M)
    print("test impute matrix ended successfully")

test_impute_matrix()


def gibb_sampl_no_modification(info):
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
    R = X  # X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    Rt_R = R.T @ R + lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)
    start_gibb_s = time.time()
    upd_j = np.zeros((d, 2))
    print("initial X \n", X)
    for h in range(r):
        #print("iter ", h)
        for i in range(d):
            print("index gibb sampl no mod", i)
            X_pre_upd = X
            X, _ = impute_matrix(X, Q, M, i)
            print("new X ", X)
            if info['verbose'] > 0:
                print(X)
            upd_j[i, 0] = 1
            upd_j[:, 1] = X.T @ (X[:, i] - X_pre_upd[:, i])
            upd_j[i, 1] = np.sum((X[:, i] - X_pre_upd[:, i]) * (X[:, i] + X_pre_upd[:, i])) / 2  
            Q = update_inverse_rk2_sym(Q, upd_j)
            upd_j[i, 0] = 0
            QQ = X.T @ X + lbd * np.eye(d)
            QQ = np.linalg.inv(QQ)
            #print("small check QQ\n", QQ)
            #print("small check Q\n", Q)
            np.testing.assert_allclose(Q, QQ)
    return X      
            
                
    end_gibb_s = time.time()
    #print("res my imp \n", X)
    print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    return X


def test_gibb_sampl_no_modification():
    print("beginning test gibb samp no modification")
    n = 7
    print("sqrt n ", np.sqrt(n))
    print("n ** (3/4)", n ** (3/4))
    print("n ** (3/4) / n", (n ** (3/4)) / n)
    d = 3
    lbd = 1 + 0.0
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    #X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    #X = (X_orig - mean) / std
    X = X_orig
    #X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    print(np.max(X))
    print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    exponent = (n ** (3/4)) / n
    print("exponent", exponent)
    M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.4, p_flip=exponent)
    print("masks in test gibb sampl no modification\n", M)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    info_dic = {
        'data': X,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'batch_size': 64,
        'verbose': 0
    }
    res = gibb_sampl_no_modification(info_dic)
    print("test impute matrix ended successfully")

test_gibb_sampl_no_modification()


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


def test_gibb_sampl():
    # the test consists in running IterativeImputer with Ridge Regression,
    # and our handmade gibb sampling function
    n = 240
    print("sqrt n ", np.sqrt(n))
    print("n ** (3/4)", n ** (3/4))
    print("n ** (3/4) / n", (n ** (3/4)) / n)
    d = 25
    lbd = 1 + 0.0
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
    M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    info_dic = {
        'data': X,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'batch_size': 64,
        'verbose': 0
    }
    start_time_gibb_sampl = time.time()
    X_my = gibb_sampl(info_dic)
    end_time_gibb_sampl = time.time()
    print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
#    print(X_my) 
    print("\nend my gibb sampling\n")
    
    print("It imputer Ridge Reg")
    ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='mean', verbose=0)
    start4 = time.time()   # tic
    res4 = ice4.fit_transform(X_nan)
#    print("result IterativeImptuer with Ridge\n", res4)
    end4 = time.time()     # toc
    print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds\n\n")
    if not info_dic['tsp']:
        np.testing.assert_allclose(X_my, res4)
    print("test gibb sampl ended successfully")

np.random.seed(54321)

# test_gibb_sampl()


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

