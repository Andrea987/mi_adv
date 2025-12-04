import numpy as np
import time
from tsp import swm_formula, matrix_switches, split_upd, make_mask_with_bounded_flip
from tsp import rk_1_update_inverse, impute_matrix, gibb_sampl_no_modification, gibb_sampl, gibb_sampl_over_parametrized
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge 




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



def test_gibb_sampl_no_modification():
    print("\n\nbeginning test gibb samp no modification\n")
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
    print("test gibb sampl no modif ended successfully")


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
        'initial_strategy': 'constant'
    }
    start_time_gibb_sampl = time.time()
    X_my = gibb_sampl(info_dic)
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
    print("test gibb sampl ended successfully")



def test_gibb_sampl_over_parametrized():
    n = 100
    d = 200
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
    M = np.random.binomial(1, 0.5, size=(n, d))
    for ii in range(d):
        nbr = np.random.randint(0, n)
        #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
        if np.sum(M[:, ii]) == n:
            print("add a random seen component")
            M[nbr, ii] = 0
    exponent = (n ** (3/4)) / n
    #M[-1, 0] = 0
    print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    print(X_nan)
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
        'initial_strategy': 'mean'
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
    print("check skl vs my underparametrized passed successfully")


#test_gibb_sampl()

test_gibb_sampl_over_parametrized()

