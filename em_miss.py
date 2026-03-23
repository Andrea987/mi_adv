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
from tsp_imputation import impute_matrix_overparametrized, impute_matrix_under_parametrized_sampling
from tsp_imputation import impute_matrix_under_parametrized, impute_matrix_over_parametrized_sampling
from utils import flip_matrix_manual, update_inverse_rk2_sym, matrix_switches, swm_formula, split_upd, split_up_fx_dw, update_covariance
from utils import s as s_prod
from utils import make_centered_kernel_matrix, compute_centered_kernel_matrix_regulirized_manually, compute_centered_kernel_matrix_regulirized_manually_2
from serialization import serialization_first_idea
import copy
from hyppo.ksample import Energy
import ot

def log_lkh(S_inv, H):
    # S is a dxd matrix
    # compute -log |S| - Tr(S^1H)
    return np.linalg.slogdet(S_inv)[1] - np.sum(S_inv * H)


def obs_log_lkh(S, mu, M, X):
    # S_inv inverse kernel matrix
    # M masks,
    # X, observations
    n, d = X.shape
    res = 0
    for i in range(n):
        m = M[i, :]
        l = np.sum(1-m)  # nbr seen
        x = X[i, :]
        xo = x[m==0]
        Soo = S[m==0, :][:, m==0]
        mu0 = mu[m==0]
        Soo_inv = np.linalg.inv(Soo)
        h = np.outer(xo-mu0, xo-mu0)
        res = res + np.linalg.slogdet(Soo_inv)[1] - np.sum(Soo_inv * h)
    return res

def em_miss(info):
    # compute mean and covariance matrix of some data
    # X = [x1|..|xn]ˆT, xi \in Rˆd
    # M = [m1|..|mn]ˆT, mi \in {0,1}ˆd
    # mij = 0 iff xij seen, mij = 1 iff xij missing
    X = info['data']
    n, d = X.shape
    M = info['masks']
    lbd = info['lbd_reg']
    current_mean1 = np.mean(X, axis=0)
    #S1 = info['starting_point'] if 'starting_point' in info else np.eye(d)
    S1 = X.T @ X /n - np.outer(current_mean1, current_mean1) + lbd * np.eye(d)
    print("mean with full data, no missing \n", current_mean1)
    print("covariance matrix with full data\n", S1)
    #print("X in em_miss ", X)
    original_X = X
    M = info['masks']
    print("sampling :::::: ", info['sampling'])
    sampling = info['sampling'] if 'sampling' in info else False
    intercept = info['intercept'] if 'intercept' in info else True
    if original_X.shape[1] == 2:
        plt.scatter(original_X[:, 0], original_X[:, 1])
        plt.scatter(original_X[M[:, 0] == 1, 0], original_X[M[:, 0] == 1, 1])
        plt.scatter(original_X[M[:, 1] == 1, 0], original_X[M[:, 1] == 1, 1])
        plt.show()
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = initial_imputation.fit_transform(X_nan)
    current_mean = np.mean(X, axis=0)
    S = info['starting_point'] if 'starting_point' in info else np.eye(d)
    S = X.T @ X /n - np.outer(current_mean, current_mean) + lbd * np.eye(d)
    SS = np.cov(X, rowvar=False, bias=True)
    #print("covariance matrices")
    #print(S)
    #print(SS)
    #mu = np.mean(X, axis=0)
    tol = info['tolerance']    
    R = info['nbr_it_em']
    err = tol
    it = 0
    #M[0, :] = np.array([0, 0, 0, 0])
    #M[1, :] = np.array([0, 1, 1, 1])
    old_cov = S
    new_cov = S
    Q = np.linalg.inv(new_cov)
    print("prints imputed dataset \n", X)
    while it<R and err>=tol:
        if it % 20 == 0:
            print(it)
        current_cov = np.zeros((d, d))
        current_cov1 = np.zeros((d, d))
        for i in range(n):
            m = M[i, :]
            #print("current mask ", m)
            x = X[i, :]
            #print(m==0)
            #print(S)
            #print(x[m==0])
            xo, xm, muo, mum = x[m==0], x[m==1], current_mean[m==0], current_mean[m==1]
            Qmm, Qmo = Q[m==1,:][: , m==1], Q[m==1, :][: , m==0]
            #print("Qmo ", Qmo)
            Soo, Som, Smo, Smm = S[m==0, :][:, m==0], S[m==0, :][:, m==1], S[m==1, :][:, m==0], S[m==1, :][:, m==1]
            #print("submatrices")
            #print(Soo), print(Som), print(Smo), print(Smm)
            #print("end submatrices")
            #Soo_inv = np.linalg.inv(Soo)
            #mu_cond_check = mum + Smo @ Soo_inv @ (xo - muo)
            #S_cond_check = Smm - Smo @ Soo_inv @ Som
            S_cond = np.linalg.inv(Qmm)
            #print("S cond check")
            #print(S_cond)
            #print(S_cond_check)
            mu_cond = mum - S_cond @ Qmo @ (xo - muo)
            #print("mu cond vs mu cond check")
            #print(mu_cond)
            #print(mu_cond_check)
            embed_cond_cov = np.zeros_like(current_cov)
            if sampling and np.sum(m)>0:
                sample =  np.random.multivariate_normal(mu_cond, S_cond)
                x[m==1] = sample
                #print(sampling)
                #print(x)
                #print("some prints, you are sampling")
                #input()
            else:
                x[m==1] = mu_cond
                #print(x)
                #print(np.ix_(m==1,m==1))
                embed_cond_cov[np.ix_(m==1, m==1)] = S_cond
                #print(embed_cond_cov[m==1, :][:, m==1])
                #print("embed cond conv \n", embed_cond_cov)
                #input()
            X[i, :] = x
            #print(X)
            v = np.outer(x, x) + embed_cond_cov
            current_cov = v if i==0 else (current_cov + v/i) * i/(i+1)
            current_cov1 = current_cov1 + v
            #print(np.outer(x, x))
            #print(current_cov)
            #print("submatrices")
            #print(Soo), print(Som), print(Smo), print(Smm)
            #print("end submatrices")
            #input()
        #print("end of loop")
        current_mean = np.mean(X, axis=0)
        old_cov = new_cov
        new_cov = current_cov - np.outer(current_mean, current_mean)
        S = new_cov
        current_obs_log_lkh = obs_log_lkh(new_cov, current_mean, M, original_X) 
        #current_obs_log_lkh1 = obs_log_lkh(new_cov, current_mean, M, X) 
        #print("current obs log lkh (shoyld be increasing)", current_obs_log_lkh)
        #print("current obs log lkh (shoyld be increasing)", current_obs_log_lkh1) 
        err = np.sqrt(np.sum((old_cov - new_cov)**2))
        #print("err ", err)
        #current_cov1 = current_cov1 / n
        #print(current_cov1)
        #print(current_cov) 
        #np.testing.assert_allclose(current_cov, current_cov1)
        Q_S_old__S_old = log_lkh(Q, new_cov)  # obs: Q(S|S_old) = const + log_lkh = logdet(S^{-1}) - Tr(S^1S_new)
        Q = np.linalg.inv(new_cov)
        Q_S_new__S_old = log_lkh(Q, new_cov) 
        diff = Q_S_new__S_old - Q_S_old__S_old
        diff = (n/2) * diff
        #print("difference, should be >=0 ", diff)
        it = it + 1
    print("fin res \n", new_cov)
    print("gt cov \n", info['cov_gt'])
    print("current mean ", current_mean)
    print("gt mean ", info['mean_gt'])
    return {'cov_em':new_cov, 'mean_em':current_mean}


def small_test_em_miss():
    print("small test em miss")
    n = 10
    d = 3
    lbd = 0.0 + 0.0
    X_orig = np.random.randint(0, 6, size=(n, d)) + 0.0
    mean = np.random.rand(d)
    cov1 = np.random.rand(n, d)
    #cov1 = np.random.randint(0, 5, (n, d))
    cov = (cov1.T @ cov1)/n + np.eye(d) * 0.1
    X_orig = np.random.multivariate_normal(mean, cov, size=n)
    X = X_orig
    M = np.random.binomial(1, 0.2, size=(n, d))
    for i in range(n):
        m = M[i, :]
        j = np.random.randint(0, d)
        if np.sum(m) == d:  # full missing
            M[i, j] = 0
    #print("M\n", M)
    print("M in gibb sampling fast sampling, tsp_test.py\n", M) if n<=10 and d<=10 else print("")
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    R = 500
    info_dic = {
        'data': X_orig,
        'masks': M,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'tolerance': 1e-1,
        'nbr_it_em': R,
        'sampling': True,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0, 
        'sampling': True,
        'cov_gt': cov,
        'mean_gt': mean
    }
    em_miss(info_dic)
    print("end small test em miss")

#small_test_em_miss()











