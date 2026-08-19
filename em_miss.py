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
        quad = np.sum(Soo_inv * h)
        #res = res + np.linalg.slogdet(Soo_inv)[1] - np.sum(Soo_inv * h)
        res = res - 0.5 * (l * np.log(2 * np.pi) + np.linalg.slogdet(Soo)[1] + quad)
    return res

def em_miss(info):
    # compute mean and covariance matrix of some data
    # X = [x1|..|xn]ˆT, xi \in Rˆd
    # M = [m1|..|mn]ˆT, mi \in {0,1}ˆd
    # mij = 0 iff xij seen, mij = 1 iff xij missing
    X_orig = info['data'].copy() #if info['imputed_data'] is None else info['imputed_data']
    n, d = X_orig.shape
    M = info['masks'].copy()
    lbd = info['lbd_reg']
    sampling = info['sampling'] if 'sampling' in info else False
    X_nan = X_orig.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = initial_imputation.fit_transform(X_nan) if info['imputed_data'] is None else info['imputed_data']
    #print("em miss\n ", X)
    mu = np.nanmean(X_nan, axis=0)
    S = np.cov(X, rowvar=False) + lbd * np.eye(d)
    tol = info['tolerance']    
    R = info['nbr_it_em']
    err = tol
    it = 0
    list_cov = []
    list_mean = []
    while it<R and err>=tol:
        S_sum = np.zeros((d, d))
        mu_sum = np.zeros(d)
        for i in range(n):
            m = M[i, :]
            x = X[i, :]
            if np.sum(m) == 0:
                mu_i = x
                Si = np.zeros((d, d))
            else:
                xo, xm, muo, mum = x[m==0], x[m==1], mu[m==0], mu[m==1]
                Soo, Som, Smo, Smm = S[m==0, :][:, m==0], S[m==0, :][:, m==1], S[m==1, :][:, m==0], S[m==1, :][:, m==1]
                
                mu_cond = mum + Smo @ np.linalg.solve(Soo, (xo - muo))
                S_cond = Smm - Smo @ np.linalg.solve(Soo, Som)
                #print("mu cond em miss")
                #print(mu_cond)

                mu_i = np.zeros(d)
                Si = np.zeros((d, d))
                mu_i[m==0], mu_i[m==1] = xo, mu_cond
                Si[np.ix_(m==1, m==1)] = S_cond

                #if sampling:  # do not use now
                #    sample =  np.random.multivariate_normal(mu_cond, S_cond)
                #    x[m==1] = sample
                    
            mu_sum = mu_sum + mu_i     
            S_sum += np.outer(mu_i, mu_i) + Si
        mu_new = mu_sum / n
        #print(mu_new)
        S_new = S_sum / n - np.outer(mu_new, mu_new)
        S_new += lbd * np.eye(d)

        # convergence
        err = np.linalg.norm(mu_new - mu) + np.linalg.norm(S_new - S)

        mu, S = mu_new, S_new
        if info['save_all_iterations']:
            list_mean.append(mu)
            list_cov.append(S)
        #obs_log_lkl = obs_log_lkh(S, mu, M, X)
        #print("obs lkl in em miss: ", obs_log_lkl)
        it = it + 1
        #print(it)
    return {'cov_em':S , 'mean_em':mu, 'list_mean': list_mean, 'list_cov': list_cov}




















def em_gaussian_missing(X, M, max_iter=100, tol=1e-6, ridge=0):
    """
    EM for multivariate Gaussian with missing data.

    Parameters
    ----------
    X : (n, d) array
        Data with arbitrary values in missing entries (ignored).
    M : (n, d) array
        Mask: 1 = missing, 0 = observed
    max_iter : int
    tol : float
    ridge : float
        Regularization for covariance

    Returns
    -------
    mu, Sigma
    """

    n, d = X.shape

    # ---- init: simple imputation ----
    X_filled = X.copy()
    X_filled[M == 1] = np.nan
    mu = np.nanmean(X_filled, axis=0)

    # fill missing with mean
    inds = np.where(np.isnan(X_filled))
    X_filled[inds] = np.take(mu, inds[1])
    print("em gaussian missing\n ", X_filled)

    Sigma = np.cov(X_filled, rowvar=False) + ridge * np.eye(d)

    log_likelihood_prev = -np.inf

    for it in range(max_iter):

        mu_sum = np.zeros(d)
        S_sum = np.zeros((d, d))

        # precompute inverse once per iteration
        Sigma_inv = np.linalg.inv(Sigma)

        for i in range(n):

            obs = (M[i] == 0)
            miss = ~obs

            x_obs = X[i, obs]
            mu_obs = mu[obs]

            if np.sum(miss) == 0:
                xi = X[i]
                mu_i = xi
                Si = np.zeros((d, d))
            else:
                Sigma_oo = Sigma[np.ix_(obs, obs)]
                Sigma_mo = Sigma[np.ix_(miss, obs)]
                Sigma_mm = Sigma[np.ix_(miss, miss)]

                # conditional mean
                mu_miss = mu[miss] + Sigma_mo @ np.linalg.solve(Sigma_oo, (x_obs - mu_obs))

                #print("mu miss em gauss miss")
                #print(mu_miss)
                mu_i = mu.copy()
                mu_i[obs] = x_obs
                mu_i[miss] = mu_miss

                # conditional covariance
                Sigma_cond = Sigma_mm - Sigma_mo @ np.linalg.solve(Sigma_oo, Sigma_mo.T)

                Si = np.zeros((d, d))
                Si[np.ix_(miss, miss)] = Sigma_cond

            # accumulate E[x]
            mu_sum += mu_i

            # accumulate E[xx^T]
            S_sum += np.outer(mu_i, mu_i) + Si

        # ---- M-step ----
        mu_new = mu_sum / n
        #print(mu_new)
        Sigma_new = S_sum / n - np.outer(mu_new, mu_new)
        Sigma_new += ridge * np.eye(d)

        # convergence
        diff = np.linalg.norm(mu_new - mu) + np.linalg.norm(Sigma_new - Sigma)

        mu, Sigma = mu_new, Sigma_new
        obs_log_lkl = obs_log_lkh(Sigma, mu, M, X)
        print("obs lkl in em gaussian missing: ", obs_log_lkl)
        if diff < tol:

            break

    return mu, Sigma



def small_test_em_gaussian_mixture():
    print("small test em miss")
    n = 500
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
    R = 20
    info_dic = {
        'data': X_orig,
        'imputed_data': None,
        'starting_point': None,
        'masks': M,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'tolerance': 1e-10,
        'nbr_it_em': R,
        'sampling': False,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0, 
        'sampling': True,
        'cov_gt': cov,
        'mean_gt': mean
    }
    em_gaussian_missing(X, M)
    print("end small test em miss")




def small_test_em_miss():
    print("small test em miss")
    n = 100
    d = 15
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
    R = 300
    info_dic = {
        'data': X_orig,
        'imputed_data': None,
        'starting_point': None,
        'masks': M,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'tolerance': 1e-10,
        'nbr_it_em': R,
        'sampling': False,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0, 
        'sampling': False,
        'cov_gt': cov,
        'mean_gt': mean
    }
    em_miss(info_dic)
    print("end small test em miss")

#small_test_em_gaussian_mixture()
#small_test_em_miss()













