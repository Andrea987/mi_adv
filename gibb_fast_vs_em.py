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
from tsp import gibb_sampl_fast_sampling
from em_miss import em_miss
import copy
from hyppo.ksample import Energy
import ot







def test_gibb_fast_vs_em():
    print("gibb fast vs em")
    n = 600
    d = 5
    lbd = 0.1 + 0.0
    X_orig = np.random.randint(0, 6, size=(n, d)) + 0.0
    mean = np.random.rand(d)
    cov1 = np.random.rand(n, d)
    #cov1 = np.random.randint(0, 5, (n, d))
    cov = (cov1.T @ cov1)/n + np.eye(d) * 0.1
    X_orig = np.random.multivariate_normal(mean, cov, size=n)

    true_cov_full_data = np.cov(X_orig, rowvar=False, bias=True)
    true_mean_full_data = np.mean(X_orig, axis=0)
    print("true mean and cov full data")
    print(true_mean_full_data)
    print(true_cov_full_data)

    X = X_orig
    M = np.random.binomial(1, 0.45, size=(n, d))
    for i in range(n):
        m = M[i, :]
        j = np.random.randint(0, d)
        if np.sum(m) == d:  # full missing
            M[i, j] = 0
    #print("M\n", M)
    print("M in gibb sampling fast sampling, tsp_test.py\n", M) if n<=10 and d<=10 else print("")
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    it_em = 500
    info_em = {
        'data': X_orig,
        'masks': M,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'tolerance': 1e-2,
        'nbr_it_em': it_em,
        'sampling': True,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0, 
        'cov_gt': cov,
        'mean_gt': mean
    }
    res_em = em_miss(info_em)
    cov_em = res_em['cov_em']
    mean_em = res_em['mean_em']

    R = 4
    info_gibb_fast = {
        'data': X_orig,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'sampling': True,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0
    }

    res_fast_gibbs = gibb_sampl_fast_sampling(info_gibb_fast)
    cov_gbsf = np.cov(res_fast_gibbs, rowvar=False, bias=True)
    mean_gbsf = np.mean(res_fast_gibbs, axis=0)

    print("EM")
    print(cov_em)
    print(mean_em)

    print("\ngibb fast")
    print(cov_gbsf)
    print(mean_gbsf)

    print("\ntrue mean and cov full data")
    print(true_cov_full_data)
    print(true_mean_full_data)
    
    
    print("\nend test gibb fast vs em")



test_gibb_fast_vs_em()


