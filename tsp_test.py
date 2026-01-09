import numpy as np
import time
from generate import generate_mask_with_bounded_flip
from tsp_imputation import impute_matrix_under_parametrized, impute_matrix_overparametrized
from tsp import gibb_sampl_no_modification, gibb_sampl_over_parametrized, gibb_sampl_under_parametrized
from utils import flip_matrix_manual, rk_1_update_inverse, swm_formula, matrix_switches, split_upd, s, update_covariance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge 
from scipy.sparse import csr_matrix
from hyppo.ksample import Energy


def test_flip_matrix():
    print("test flip matrix")
    n, d = 10, 5
    M = np.random.binomial(1, 0.5, size=(n, d))
    M_s = csr_matrix(M)
    ones_d = np.ones(d)
    #FF = n * ones_d - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
    MM = flip_matrix_manual(M.T)
    np.testing.assert_allclose(F, MM)
    print("test flip matrix ended successfully")
    

def test_swm():
    print("test swm started")
    n, d, m, c = 20, 30, 5, 0.6
    X = np.random.randint(1, 5, size = (n, d))  + 0.0
    U = np.random.randint(1, 5, size = (m, d))  + 0.0
    A = X.T @ X  + np.eye(d)  # (d, d)
    Q = np.linalg.inv(A)
    A_upd = A + c * U.T @ U
    #print(X)
    #rint(U)
    Q_upd_inv = np.linalg.inv(A_upd)
    Q_tested = swm_formula(Q, U.T, c)
    #print("manually\n ", Q_upd_inv, "\n by function\n", Q_tested)
    #print(Q_tested @ A_upd)
    np.testing.assert_allclose(Q_upd_inv, Q_tested)
    print("test swm ended successfully\n\n")


def test_split_upd():
    ## the test work in the following way
    ## start from a matrix C = sum_{i: M_i0 = 0} xi(xi.T)
    ## then compute the update and downdate to pass to the matrix C = sum_{i: M_i1 = 0} xi(xi.T)
    ## sum the updates and subtract the updates to C
    ## check with the original definition of C if the results are correct
    print("\n\ntest split upd")
    n, d = 5, 2
    X = np.random.randint(1, 5, size=(n, d)) + 0.0
    M = np.random.binomial(n=1, p=0.5, size= (n, d))
    #M = np.array( [ [1] * n, [0] * n  ]   ).T
    #M = np.array( [ [0] * n, [1] * n  ]   ).T
    #print(M)
    #m1 = np.random.binomial(n=1, p=0.3, size=n)
    #m2 = np.random.binomial(n=1, p=0.3, size=n)
    #print(m1)
    #print(m2)
    #m = np.array([m1, m2])
    #print("masks\n\n ", m)
    #print("masks\n ", m)
    ms = matrix_switches(M)
    #print("ms \n ", ms)
    Xs = X[M[:, 0] == 0, :]
    C = Xs.T @ Xs #if Xs.shape[0] > 0 else np.zeros((d, d))  
    #print(C)
    for i in range(d):
        vpl, vmn = split_upd(X, ms[:, i])
        #print(C)
        C = C + vpl.T @ vpl - vmn.T @ vmn
        #print(C)
        idx = i+1 if i<d-1 else 0
        Xs = X[M[:, idx] == 0, :]
        CC = Xs.T @ Xs
        #print(C)
        #print(CC)
        np.testing.assert_allclose(C, CC)

    #print("X\n ", X)
    #print("updates/downdates, \n", ms)
    #print("v pl \n", vpl)
    #print("v mn \n", vmn)
    print("end test split upd ended successfully\n\n")


def test_s():
    print("test_s started")
    n, d = 40, 6
    C = np.random.randint(0, 5, (n, d)) + 0.0
    v = np.random.randint(0, 5, d) + 0.0
    u = np.ones(n)
    print(C)
    CC = C - np.outer(u, v)
    print(v)
    print(CC)
    S = CC.T @ CC 
    S_test = C.T @ C + s(C, v)
    print(S_test)
    np.testing.assert_allclose(S, S_test)
    print("test_s ended successfully")


def test_update_covariance():
    print("test update covariance started")
    n, d = 6, 4
    nu = 3  # nd <= n 
    C1 = np.random.randint(0, 5, (n, d)) + 0.0
    v1 = np.random.randint(0, 5, d) + 0.0   
    C_upd = np.random.randint(0, 5, (nu, d)) + 0.0
    v2 = np.random.randint(0, 5, d) + 0.0
    m = np.array([1, 1, -1, 1, -1, 1])
    nd = np.sum(m == -1)
    C_fix, C_dwd = split_upd(C1, m) 
    C2 = np.vstack((C_fix, C_upd))
    print("C2 shape ", C2.shape)

    u1 = np.ones(n)
    u2 = np.ones(n - nd + nu)

    C11 = C1 - np.outer(u1, v1)
    Cov1 = C11.T @ C11

    C22 = C2 - np.outer(u2, v2)
    Cov2 = C22.T @ C22

    Cov2_test = update_covariance(Cov1, C1, C2, v1, v2, C_upd, C_dwd)
    np.testing.assert_allclose(Cov2, Cov2_test)
    print("test update covariance ended successfully")


test_update_covariance()
input()


def test_rk_1_update_inverse():
    print("\n\ntest rk_1 update inverse started")
    n, d, c = 5, 3, -0.345
    X = np.random.randint(1, 5, size =(n, d))  + 0.0
    u = np.random.randint(1, 5, size = d)  + 0.0
    A = X.T @ X  + np.eye(d)  # (d, d)
    Q = np.linalg.inv(A)
    A_upd = A + c * np.outer(u, u)
    Q_upd_inv = np.linalg.inv(A_upd)
    Q_tested = rk_1_update_inverse(Q, u, c)
    #print("manually\n ", Q_upd_inv, "\n by function\n", Q_tested)
    #print(Q_tested @ A_upd)
    np.testing.assert_allclose(Q_upd_inv, Q_tested)
    print("test rk_1 upd inverse ended successfully\n\n")


def test_impute_matrix_under_parametrized():
    print("\n\nbeginning test impute matrix ")
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
        X, my_coeff = impute_matrix_under_parametrized(X, Q, M, i)
        #print("my coeff from Ridge  ", my_coeff)
        #print("coeff from Ridge fit ", clf.coef_)
        np.testing.assert_allclose(my_coeff, clf.coef_)
    #print("masks in test impute matrix\n", M)
    print("test impute matrix ended successfully\n\n")


def test_gibb_sampl_no_modification():
    print("\n\nbeginning test gibb samp no modification\n")
    n = 7
    #print("sqrt n ", np.sqrt(n))
    #rint("n ** (3/4)", n ** (3/4))
    #print("n ** (3/4) / n", (n ** (3/4)) / n)
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
    #print(np.max(X))
    #print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    exponent = (n ** (3/4)) / n
    #print("exponent", exponent)
    M = generate_mask_with_bounded_flip(n=n, d=d, p_miss=0.4, p_flip=exponent)
    #print("masks in test gibb sampl no modification\n", M)
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
    print("test gibb sampl no modif ended successfully\n\n")


def test_gibb_sampl_under_parametrized():
    # the test consists in running IterativeImputer with Ridge Regression,
    # and our handmade gibb sampling function
    print("test gibb sampl under parametr started")
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
    M = np.random.binomial(1, 0.1, size=(n, d))
    #exponent = (n ** (3/4)) / n
    #print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 4
    info_dic = {
        'data': X,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75
    }
    start_time_gibb_sampl = time.time()
    X_my = gibb_sampl_under_parametrized(info_dic)
    end_time_gibb_sampl = time.time()
    print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
#    print(X_my) 
    print("\nend my gibb sampling\n")
    
    print("It imputer Ridge Reg")
    ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
    start4 = time.time()   # tic
    res4 = ice4.fit_transform(X_nan)
#    print("result IterativeImptuer with Ridge\n", res4)
    end4 = time.time()     # toc
    print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds\n\n")
    #if not info_dic['tsp']:
    np.testing.assert_allclose(X_my, res4)
    print("test gibb sampl under parametr ended successfully\n")


def test_gibb_sampl_over_parametrized():
    print("\ntest gibb sample over parametr started")
    n = 39
    d = 55
    lbd = 1.6321 + 0.0
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    #X_orig = np.random.rand(n, d) + 0.0
    #print(X_orig.dtype)
    #print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    #X = (X_orig - mean) / std
    X = X_orig
    #X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    print(np.max(X))
    print(np.min(X))
    M = np.random.binomial(1, 0.5, size=(n, d))
    for ii in range(d):
        nbr = np.random.randint(0, n)
        #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
        if np.sum(M[:, ii]) == n:
            print("add a random seen component")
            M[nbr, ii] = 0
    exponent = (n ** (3/4)) / n
    #M[-1, 0] = 0
    #print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    #print(X_nan)
    R = 2
    info_dic = {
        'data': X,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'mean',
        'exponent_d': 0.75
    }
    res = gibb_sampl_over_parametrized(info_dic)
    #res_std = gibb_sampl(info_dic)
    #print("final res_std\n", res_std)
    #print("final res\n", res)
    #np.testing.assert_allclose(res, res_std)

    print("It imputer Ridge Reg")
    ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
    res_skl = ice_skl.fit_transform(X_nan)
    np.testing.assert_allclose(res, res_skl)
    print("check skl vs my under parametrized passed successfully")
    print("test gibb sample over parametr ended successfully")


test_flip_matrix()
test_split_upd()
test_swm()
test_split_upd()
test_s()
test_rk_1_update_inverse()
test_impute_matrix_under_parametrized()
test_gibb_sampl_no_modification()
test_gibb_sampl_under_parametrized()
test_gibb_sampl_over_parametrized()

