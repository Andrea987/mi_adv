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
from utils import make_centered_kernel_matrix, plot2D, compute_centered_kernel_matrix_regulirized_manually, compute_centered_kernel_matrix_regulirized_manually_2
from serialization import serialization_first_idea
import copy
from hyppo.ksample import Energy
import ot



def gibb_sampl_fast_sampling(info):
    ## this code implement the fast Gibb sampler, that is consider as covariance matrix the 
    ## the one obtained by summing all the tensor associated to the vectors of the dataset.
    ## In the MICE versione, each dataset is composed by a subset of observations
    X = info['data']
    M = info['masks']
    n, d = X.shape
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = imp_mean.fit_transform(X_nan)
    sampling = info['sampling'] if 'sampling' in info else False
    intercept = info['intercept'] if 'intercept' in info else True
    print("sampling ", sampling, "intercept ", intercept)
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
    R = X.copy()  # X[first_mask == 0, :]
    #print("first set vct ", R)
    #print("first set vct shape ", R.shape)
    mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1])
    u = np.ones(R.shape[0])
    #print(a)
    #print("\n", np.outer(u, a))
    R_centered = R - np.outer(u, mean)
    start1 = time.time()
    cov = R_centered.T @ R_centered + lbd * np.eye(d)
    end1 = time.time()
    print("building the matrix time: ", end1-start1)        
    Q = np.linalg.inv(cov)
    start_gibb_s = time.time()
    upd_j = np.zeros((d, 2))
    #print("initial X \n", X)
    gamma = np.ones(r) / np.arange(1, r+1)
    gamma = np.ones(r)
    cov_gamma = cov / R_centered.shape[0]
    cov_test = cov
    #print(gamma)
    #input()
    for h in range(r):
        #print("iter ", h)
        for i in range(d):
            #print("index gibb sampl no mod", i)
            X_pre_upd = X
            #X, _ = impute_matrix_under_parametrized(X, Q, M, i)
            alpha = R_centered.shape[0]
            X, _ = impute_matrix_under_parametrized_sampling(X, mean, cov / alpha, Q * alpha, M, i, sampling, intercept)
            old_mean_rescaled = np.sqrt(n) * mean  # old_mean, before making the update 
            mean = np.mean(X, axis=0)  # new mean
            mean_rescaled = np.sqrt(n) * mean  # new mean, after the update
            if info['verbose'] > 0:
                print("print X\n in gibb sampling fast ", X)
            upd_j[i, 0] = 1
            #start1 = time.time()
            upd_j[:, 1] = X.T @ (X[:, i] - X_pre_upd[:, i])
            #end1 = time.time()
            #print("multiplication time: ", end1-start1)
            upd_j[i, 1] = np.sum((X[:, i] - X_pre_upd[:, i]) * (X[:, i] + X_pre_upd[:, i])) / 2  
            cov_test = cov_test + np.outer(old_mean_rescaled, old_mean_rescaled) - np.outer(mean_rescaled, mean_rescaled) 
            cov_test = cov_test + np.outer(upd_j[:, 1], upd_j[:, 0]) + np.outer(upd_j[:, 0], upd_j[:, 1])
            #print(cov_test)
            Q = update_inverse_rk2_sym(Q, upd_j)
            upd_j[i, 0] = 0
            Q = swm_formula(Q, old_mean_rescaled, 1.0)  # updates
            Q = swm_formula(Q, mean_rescaled, -1.0)  # downdates
            
            ## small test
            mmean = np.mean(X, axis=0)
            #cov = X.T @ X - n * np.outer(mmean, mmean) + lbd * np.eye(d)
            #print(cov)
            cov = cov_test
            #input()
            #print("alpha ", alpha)
            cov_gamma = cov_gamma + gamma[h] * (cov/alpha - cov_gamma)
            cov = cov_gamma * alpha
            QQ = np.linalg.inv(cov)
            #if d<=8 and n<=10:
                #print("small check QQ\n", QQ)
                #print("small check Q\n", Q)
            #np.testing.assert_allclose(Q, QQ)
            #input()
    res = {'imputed_dts': X, 'RM_cov': cov_gamma}  # RM: Robbins_Monro
    return res      


def gibb_sampl_under_parametrized_sampling(info):
    # flip matrix
    #if info['ml_or_bs'] not in ['bayesian', 'max_lh']:
    #    print("please specify a correct approach, bayesian or max_lh")
    #    input()
    X = info['data']
    original_X = X
    M = info['masks']
    sampling = info['sampling'] if 'sampling' in info else False
    intercept = info['intercept'] if 'intercept' in info else True
    #print(M)
    #plot2D(X, M) if original_X.shape[1] == 2 else print("dimension too high, no 2D plot")
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = initial_imputation.fit_transform(X_nan) if info['imputed_data'] is None else info['imputed_data']
    print(X)
    #plot2D(X, M) if info['plot2D_it'] else print("no 2D plot")
    #print("simple imputer in gibb sample under param \n", X)
    #print("shape M", M.shape)
    #print("nbr masks ", np.sum(M, axis=0).shape)
    #print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    
    if info['tsp']:
        start_time = time.time()
        MM = M if np.mean(M) >= 1/2 else 1 - M
        M_s = csr_matrix(MM)
        ones_d = np.ones(d)
        #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
        #M_ss = csr_matrix(M)
        F = np.outer(ones_d, np.sum(MM, axis=0)) + np.outer(np.sum(MM.T, axis=1), ones_d) - 2 * M_s.T @ M_s
        #FF = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_ss.T @ M_ss
        #np.testing.assert_allclose(F, FF)
        #permutation, distance = solve_tsp_local_search(F)
        permutation, distance = serialization_first_idea(F)
        current_distance = distance
        current_permutation = permutation
        
        original_cost = np.sum(np.diag(F, k=1))
        print("original cost in tsp2", original_cost)
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
    mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1])
    u = np.ones(R.shape[0])
    #print(a)
    #print("\n", np.outer(u, a))
    R_centered = R - np.outer(u, mean)
    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1/R_centered.shape[0] 
    #print("alpha ", alpha)
    Cov = R_centered.T @ R_centered + lbd * np.eye(d)  ## look how to add the (1/n), where it is better to be added 
    Q = np.linalg.inv(Cov)
    current_info ={
        'inverse': Q,
        'vectors': R
    } 
    counter_upd_dwd = 0
    counter_recomputation = 0
    counter_swm_formula = 0 
    counter_reinversion = 0
    old_X = X
    print("d ** exp: ", d ** info['exponent_d'])
    for h in range(r):
        print("\n\n CURRENT ITERATION GIBBS SANMPLING IN UNDERPARAMETRIZED SAMPLING ", h, "\n")
        old_X = X
        #diff = np.sum((X - old_X)**2)
        #stat, pvalue = Energy().test(original_X, old_X)
        #print("stat ", stat , "pvalue ", pvalue)
        #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
        #plot2D(X, M, extra_info) if info['plot2D_it'] else print("no 2D plot")
        #aa, bb = np.ones((n,)) / n, np.ones((n,)) / n
        #MM = ot.dist(original_X, old_X)
        #G0 = ot.sinkhorn2(aa, bb, MM, 0.1)
        #print("G0 ", G0)
        #input()
        #print("\ndifference old vs new \n", np.sqrt(diff))
        #plot2D(X, M) if original_X.shape[1] == 2 else print("dimension too high, no 2D plot")
        for i in range(d):
            #alpha = R_centered.shape[0] if info['ml_or_bs'] == 'bayesian' else 1
            alpha = R_centered.shape[0]  # we need to correct by this alpha to get a correct imputation
            X, _ = impute_matrix_under_parametrized_sampling(X, mean, Cov / alpha, Q * alpha, M, i, sampling, intercept)
            #print("stopppp")
            #input()
            #print("round ", i, "who is X gs\n", X)
            #v = X.T @ X[:, i]
            #Rt_R[i, :] = v
            #Rt_R[:, i] = v
            #Rt_R
            #print("who is Rt_R \n", Rt_R)
            if h < r-1 or i < d-1:
                N = Ms[:, i]
                #print("flip vector ", N)
                X_up, X_fx, X_dw = split_up_fx_dw(X, N)  # observe, in the fix there are also vector that were not present in neither of masks, i.e rows with (1, 1)
                #print(N)
                #print("sequence of print")
                #if info['verbose'] > 0:
                #    print(X)
                #print(X_upd)
                #print(X_dwd)
                nup, _ = X_up.shape
                nfx, _ = X_fx.shape
                ndw, _ = X_dw.shape
                #ns = nfx + nup
                #print(nfx + nup + ndw)
                #print("ns outside", ns)
                #print("nup nfx ndw, nup", nup, ", nfx ", nfx, ", ndw ", ndw)
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
                #print("idx ", idx)
                #print("nbr seen ", n - np.sum(M[:, 0]), " nbr flip ", nup + ndw)
                #print("masks \n", M[:, i:(i+2)])
                old_R = R  # old_seen components, not centered
                ns_old = old_R.shape[0]
                R = X[M[:, idx] == 0, :]  # seen components, not centered
                ns = R.shape[0]
                #print("ns true ", R.shape[0])
                old_mean = mean #if intercept else np.zeros(R.shape[1])
                mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1]) # new mean
                old_R_centered = R_centered
                u = np.ones(R.shape[0])
                R_centered = R - np.outer(u, mean)
                if ns < nup + ndw:  # if nbr seen component is less than nbr of flips
                    print("recompute the matrix with the missing components")
                    counter_recomputation = counter_recomputation + 1
                    #mean = np.mean(R, axis=0)
                    u = np.ones(R.shape[0])
                    #print(a)
                    #print("\n", np.outer(u, a))
                    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1 / R_centered.shape[0]
                    Cov = R_centered.T @ R_centered + lbd * np.eye(d)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                else:
                    counter_upd_dwd = counter_upd_dwd + 1
                    #print("update the covariance matrix") 
                    #old_mean = mea
                    #mean = np.mean(R, axis=0)
                    old_Cov = Cov
                    Cov = Cov + np.outer(old_mean, old_mean) * ns_old + X_up.T @ X_up - X_dw.T @ X_dw - np.outer(mean, mean) * ns
                    Cov_test = R_centered.T @ R_centered + lbd * np.eye(d)
                    #np.testing.assert_allclose(Cov_test, Cov)
                    #Cov = ((Cov - lbd * np.eye(d) + np.outer(old_mean, old_mean)) * ns_old + X_up.T @ X_up - X_dw.T @ X_dw) / ns - np.outer(mean, mean) + lbd * np.eye(d)
                if  nup + ndw > d ** info['exponent_d']:
                    print("invert the matrix")
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd too big, invert the matrix ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    #idx = i+1 if i<d-1 else 0
                    #print(idx)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                    #print("first set vct ", R)
                    #print("first set vct shape ", R.shape)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    counter_reinversion = counter_reinversion + 1
                    Q = np.linalg.inv(Cov)
                else:
                    counter_swm_formula = counter_swm_formula + 1
                    print("low rank upd of the inverse")
                    #print("approach: ", info['ml_or_bs'])
                    #print("nupd + nded ", nupd + ndwd, " number upd + dwd small, swm formula.          ", "nbr seen ", n - np.sum(M[:, idx]), " nbr flip ", nupd + ndwd)
                    #alpha = 1 if info['ml_or_bs'] == 'bayesian' else 1 / R_centered.shape[0]
                    old_mean_rescaled = np.sqrt(ns_old) * old_mean #if info['ml_or_bs'] == 'bayesian' else old_mean 
                    mean_rescaled = np.sqrt(ns) * mean #if info['ml_or_bs'] == 'bayesian' else mean
                    #old_mean_rescaled = np.sqrt(ns_old) * old_mean 
                    #mean_rescaled = np.sqrt(ns) * mean
                    X_up_ext = np.vstack((X_up, old_mean_rescaled))
                    X_dw_ext = np.vstack((X_dw, mean_rescaled))

                    #QQ = swm_formula(Q, old_mean_rescaled, 1.0)
                    #QQ_test = np.linalg.inv(old_R.T @ old_R + np.eye(d) * lbd)
                    #print("QQ_inv\n ", QQ)
                    #print("QQ_test_inv \n", QQ_test)

                    #QQQ = swm_formula(QQ, X_up.T, 1.0)
                    #QQQ = swm_formula(QQQ, X_dw.T, -1.0) 
                    #QQQ_test = np.linalg.inv(R.T @ R + np.eye(d) * lbd)
                    #print("QQQ_inv\n ", QQQ)
                    #print("QQQ_test_inv\n ", QQQ_test)

                    #QQQQ = swm_formula(QQQ, mean_rescaled, -1.0)
                    #QQQQ_test = np.linalg.inv(R_centered.T @ R_centered + np.eye(d) * lbd)
                    #print("\n\nQQQQ_inv\n ", QQQQ)
                    #print("QQQQ_test_inv \n", QQQQ_test)


                    Q = swm_formula(Q, X_up_ext.T, 1.0)
                    Q = swm_formula(Q, X_dw_ext.T, -1.0)
                    #Q_test = np.linalg.inv(Cov)

                    #print("Q_inv\n ", Q)
                    #print("Q_test_inv \n", Q_test)
                    #np.testing.assert_allclose(Q, Q_test)
                    #input()
                    #for i_up in range(nupd):
                    #    Q = rk_1_update_inverse(Q, X_upd[i_up, :], 1.0)
                    #for i_dw in range(ndwd):
                    #    Q = rk_1_update_inverse(Q, X_dwd[i_dw, :], -1.0)
                    #print("QQ\n ", QQ)
                    #print("Q\n", Q)
                    #print("cond nub Q in gibb sampl: ", np.linalg.cond(Q))
    end_gibb_s = time.time()
    #stat, pvalue = Energy().test(original_X, old_X)
    #print("stat ", stat , "pvalue ", pvalue)
    #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
    #plot2D(X, M, extra_info) if info['plot2D_it'] else print("no 2D plot")
    print("counter recomp ", counter_recomputation/r)
    print("counter upd dwd ", counter_upd_dwd/r)
    print("counter reinv", counter_reinversion/r)
    print("counter swm ", counter_swm_formula/r)
    #print("res my imp \n", X)
    print(f"Execution time gibb sampler: {end_gibb_s - start_gibb_s:.4f} seconds")
    return X


def gibb_sampl_over_parametrized_sampling(info):
    ## Gibb sampling in an overparametrized setting
    X = info['data']
    M = info['masks']
    u = np.ones(X.shape[0])
    M_original = M.copy()
    sampling = info['sampling'] if 'sampling' in info else False
    intercept = info['intercept'] if 'intercept' in info else True
    print("\nintercept: ", intercept, "\n\n")
    X_nan = X.copy()
    original_X = X.copy()
    X_nan[M==1] = np.nan
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = imp_mean.fit_transform(X_nan)
    nbr_it_gs = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape  # suppose n < d
    X_del = np.delete(X, 0, axis=1)
    K = X_del @ X_del.T  #+ np.eye(n) * lbd
    K_centered, K_m, m_K_m = make_centered_kernel_matrix(K, M[:, 0]) if intercept else (K, np.zeros(n), 0)
    K_centered_reg = K_centered + np.eye(n) * lbd
    #K_centered_test = compute_centered_kernel_matrix_regulirized_manually_2(X, M[:, 0], lbd, intercept)
    #K_centered_test2 = compute_centered_kernel_matrix_regulirized_manually(K, M[:, 0], lbd, intercept)
    #np.testing.assert_allclose(K_centered_reg, K_centered_test)
    #np.testing.assert_allclose(K_centered_reg, K_centered_test2)
    K_centered_reg_inv = np.linalg.inv(K_centered_reg)
    for h in range(nbr_it_gs):
        print("\n\n CURRENT ITERATION GIBBS SANMPLING OVER PARAMETRIZED", h, "\n")
        old_X = X
        stat, pvalue = Energy().test(original_X, old_X)
        print("stat ", stat , "p value ", pvalue)
        #diff = np.sum((X - old_X)**2)
        #stat, pvalue = Energy().test(original_X, old_X)
        #print("stat ", stat , "p value ", pvalue)
        #input()
        for i in range(d):
            #print("index ", i)
            #idx = i if i<d-1 else 0
            #print("i: ", i)
            X = impute_matrix_over_parametrized_sampling(X=X, m=M[:, i], K=K_centered_reg, K_inv=K_centered_reg_inv, lbd=lbd, idx=i, sampling=sampling, intercept=intercept)
            #print("round ", i, ": imputed matrix gs overp\n", X)
            #input()
            if h < nbr_it_gs-1 or i < d-1:
                v_to_add = X[:, i]
                v_to_remove = X[:,(i+1)] if i<d-1 else X[:, 0]
                current_mask = M[:,(i+1)] if i<d-1 else M[:, 0]
                
                K_centered_reg_inv = swm_formula(K_centered_reg_inv, v_to_add, 1.0)
                K_centered_reg_inv = swm_formula(K_centered_reg_inv, v_to_remove, -1.0)
                K_centered_reg_inv = swm_formula(K_centered_reg_inv, u * np.sqrt(m_K_m), -1.0)
                U = np.array([K_m, u]).T
                K_centered_reg_inv = update_inverse_rk2_sym(K_centered_reg_inv, U)  # now we should have K_reg_inv = (K + lbd Id)^(-1)

                K = K + np.outer(v_to_add, v_to_add) - np.outer(v_to_remove, v_to_remove)
                K_centered, K_m, m_K_m = make_centered_kernel_matrix(K, current_mask) if intercept else (K, np.zeros(n), 0)
                K_centered_reg = K_centered + np.eye(n) * lbd
                U = np.array([K_m, -u]).T

                K_centered_reg_inv = swm_formula(K_centered_reg_inv, u * np.sqrt(m_K_m), 1.0)
                K_centered_reg_inv = update_inverse_rk2_sym(K_centered_reg_inv, U)

                #K_centered_reg_test2 = compute_centered_kernel_matrix_regulirized_manually(K, current_mask, lbd)
                #K_centered_reg_test2_inv = np.linalg.inv(K_centered_reg_test2)
                #np.testing.assert_allclose(K_centered_reg_inv, K_centered_reg_test2_inv)
                #if not intercept:
                    #print("check if intercept is false")
                    #np.testing.assert_allclose(K + np.eye(n) * lbd, K_centered_reg)
                    #KKK_inv = np.linalg.inv(K + np.eye(n) * lbd)
                    #np.testing.assert_allclose(KKK_inv, K_centered_reg_inv)
                #input()
    return X


'''some tests'''

def test_gibb_sampl_under_parametrized_sampling():
    # the test consists in running IterativeImputer with Ridge Regression,
    # and our handmade gibb sampling function
    print("test gibb sampl under parametr started")
    n = 200
    print("sqrt n ", np.sqrt(n))
    print("n ** (3/4)", n ** (3/4))
    print("n ** (3/4) / n", (n ** (3/4)) / n)
    d = 2
    gaussian = True
    lbd = 0.1 + 0.0
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    #print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    X = (X_orig - mean) / std
    X = X_orig
    X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    #print(np.max(X))
    #print(np.min(X))
    if d == 2:
        mean = np.array([4, -5])
        cov = np.array([[4, -0.95],[-0.95, 0.25]])
        X = np.random.multivariate_normal(mean, cov, size=n)
    if gaussian and d>2:
        mean = np.random.rand(d)
        cov = np.random.rand(n, d)
        cov = cov.T @ cov + np.eye(d) * 0.1
        #print(cov)
        X = np.random.multivariate_normal(mean, cov, size=n)
        #print(X)
        #input()
    M = np.random.binomial(1, 0.5, size=(n, d))
    #print(M)
    for i in range(n):
        if np.sum(M[i, :]) == 0:
 #           ss = np.random.rand()
 #           print(ss)
#            input()
            M[i, 0] = 0 if np.random.rand()>0.5 else 1
    #exponent = (n ** (3/4)) / n
    #print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 10
    info_dic = {
        'data': X,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        #'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'ml_or_bs': 'bayesian'
    }
    #start_time_gibb_sampl = time.time()
    X_my = gibb_sampl_under_parametrized_sampling(info_dic)
    if d == 2:
        plt.scatter(X_my[:, 0], X_my[:, 1])
        plt.scatter(X_my[M[:, 0] == 1, 0], X_my[M[:, 0] == 1, 1])
        plt.scatter(X_my[M[:, 1] == 1, 0], X_my[M[:, 1] == 1, 1])
        plt.show()
    #end_time_gibb_sampl = time.time()
    #print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
#    print(X_my) 
    print("\nend my gibb sampling\n")
    
    print("It imputer Ridge Reg")
    #ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
    #start4 = time.time()   # tic
    #res4 = ice4.fit_transform(X_nan)


def test_gibb_sampl_over_parametrized_sampling():
    # the test consists in running IterativeImputer with Ridge Regression,
    # and our handmade gibb sampling function
    print("test gibb sampl under parametr started")
    n = 20
    print("sqrt n ", np.sqrt(n))
    print("n ** (3/4)", n ** (3/4))
    print("n ** (3/4) / n", (n ** (3/4)) / n)
    d = 17
    gaussian = True
    lbd = 1.10 + 0.0
    X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    #print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    # Standardize
    X = (X_orig - mean) / std
    X = X_orig
    X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    #print(np.max(X))
    #print(np.min(X))
    if d == 2:
        mean = np.array([4, -5])
        cov = np.array([[4, -0.95],[-0.95, 0.25]])
        X = np.random.multivariate_normal(mean, cov, size=n)
    if gaussian and d>2:
        mean = np.random.rand(d)
        cov = np.random.rand(n, d)
        cov = cov.T @ cov + np.eye(d) * 0.5
        #print(cov)
        X = np.random.multivariate_normal(mean, cov, size=n)
        #print(X)
        #input()
    M = np.random.binomial(1, 0.4, size=(n, d))
    #print(M)
    for i in range(n):
        if np.sum(M[i, :]) == 0:
#           ss = np.random.rand()
#           print(ss)
#            input()
            M[i, 0] = 0 if np.random.rand()>0.5 else 1
    #exponent = (n ** (3/4)) / n
    #print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
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
        #'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'ml_or_bs': 'bayesian'
    }
    #start_time_gibb_sampl = time.time()
    X_my = gibb_sampl_over_parametrized_sampling(info_dic)
    if d == 2:
        plt.scatter(X_my[:, 0], X_my[:, 1])
        plt.scatter(X_my[M[:, 0] == 1, 0], X_my[M[:, 0] == 1, 1])
        plt.scatter(X_my[M[:, 1] == 1, 0], X_my[M[:, 1] == 1, 1])
        plt.show()
    #end_time_gibb_sampl = time.time()
    #print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
#    print(X_my) 
    print("\nend my gibb sampling\n")
    
    print("It imputer Ridge Reg")
    #ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
    #start4 = time.time()   # tic
    #res4 = ice4.fit_transform(X_nan)


