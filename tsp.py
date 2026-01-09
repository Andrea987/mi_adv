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
from tsp_imputation import impute_matrix_overparametrized, impute_matrix_under_parametrized, impute_matrix_under_parametrized_sampling
from utils import flip_matrix_manual, update_inverse_rk2_sym, matrix_switches, swm_formula, split_upd, split_up_fx_dw, update_covariance
from utils import s as s_prod
from serialization import serialization_first_idea
import copy
from hyppo.ksample import Energy
import ot


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
    #ones = np.ones((d, d)) 
    #start_algo_gibb_s_partial = time.time()
    #s = np.ones_like(M.T)
    #ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #FF = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M.T @ M
    #np.testing.assert_allclose(F, FF)
    #end_algo_gibb_s_partial = time.time()
    #print(f"Elapsed time gibb sampl, cov matrix masks: {end_algo_gibb_s_partial - start_algo_gibb_s_partial:.4f} seconds\n\n")
    #start_algo_gibb_s_partial = time.time()
    #s = np.ones_like(M.T)
    #ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M.T @ M
    #np.testing.assert_allclose(F, FF)
    #end_algo_gibb_s_partial = time.time()
    #print(f"Elapsed time gibb sampl, cov matrix, M: {end_algo_gibb_s_partial - start_algo_gibb_s_partial:.4f} seconds\n\n")
    #start_algo_gibb_s_partial_sparse = time.time()
    #M_s = csr_matrix(M)
    #ones_d = np.ones(d)
    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
    #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
    #print("type flip matrix ", type(F))
    #print("flip matrix head\n" , F[0:8, 0:8])
    #np.testing.assert_allclose(F, FF)
    #end_algo_gibb_s_partial_sparse = time.time()
    #print(f"Elapsed time gibb sampl, cov matrix, M sparse: {end_algo_gibb_s_partial_sparse - start_algo_gibb_s_partial_sparse:.4f} seconds\n\n")
    
    #print("flip matrix\n", F)
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



'''let's add the sampling part'''

def gibb_sampl_under_parametrized_sampling(info):
    # flip matrix
    X = info['data']
    original_X = X
    M = info['masks']
    if original_X.shape[1] == 2:
        plt.scatter(original_X[:, 0], original_X[:, 1])
        plt.scatter(original_X[M[:, 0] == 1, 0], original_X[M[:, 0] == 1, 1])
        plt.scatter(original_X[M[:, 1] == 1, 0], original_X[M[:, 1] == 1, 1])
        plt.show()
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info['initial_strategy'])
    X = initial_imputation.fit_transform(X_nan)
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
    mean = np.mean(R, axis=0)
    u = np.ones(R.shape[0])
    #print(a)
    #print("\n", np.outer(u, a))
    R_centered = R - np.outer(u, mean)
    Cov = R_centered.T @ R_centered * (1/R_centered.shape[0]) + lbd * np.eye(d)  ## look how to add the (1/n), where it is better to be added 
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
        old_X = X
        #diff = np.sum((X - old_X)**2)
        stat, pvalue = Energy().test(original_X, old_X)
        print("stat ", stat , "p value ", pvalue)
        aa, bb = np.ones((n,)) / n, np.ones((n,)) / n
        MM = ot.dist(original_X, old_X)
        G0 = ot.sinkhorn2(aa, bb, MM, 0.1)
        print("G0 ", G0)
        input()
        #print("\ndifference old vs new \n", np.sqrt(diff))
        if X.shape[1] == 2:
            plt.scatter(X[:, 0], X[:, 1])
            plt.scatter(X[M[:, 0] == 1, 0], X[M[:, 0] == 1, 1])
            plt.scatter(X[M[:, 1] == 1, 0], X[M[:, 1] == 1, 1])
            plt.show()
        for i in range(d):
            X, _ = impute_matrix_under_parametrized_sampling(X, mean, Cov, Q, M, i)
            print("stopppp")
            #input()
            #print("round ", i, "who is X gs\n", X)
            #v = X.T @ X[:, i]
            #Rt_R[i, :] = v
            #Rt_R[:, i] = v
            #Rt_R
            #print("who is Rt_R \n", Rt_R)
            if h < r-1 or i < d-1:
                N = Ms[:, i]
                X_up, X_fx, X_dw = split_up_fx_dw(X, N)
                #print(N)
                #print("sequence of print")
                #if info['verbose'] > 0:
                #    print(X)
                #print(X_upd)
                #print(X_dwd)
                nup, _ = X_up.shape
                nfx, _ = X_fx.shape
                ndw, _ = X_dw.shape
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
                old_R = R  # old_seen components, not centered
                R = X[M[:, idx] == 0, :]  # seen components, not centered
                old_mean = mean
                mean = np.mean(R, axis=0)  # new mean
                old_R_centered = R_centered
                u = np.ones(R.shape[0])
                R_centered = R - np.outer(u, mean)
                if nup + nfx < nup + ndw:  # if nbr seen component is less than nbr of flips
                    print("recompute the matrix with the missing components")
                    counter_recomputation = counter_recomputation + 1
                    #mean = np.mean(R, axis=0)
                    u = np.ones(R.shape[0])
                    #print(a)
                    #print("\n", np.outer(u, a))
                    Cov = R_centered.T @ R_centered * (1/R_centered.shape[0]) + lbd * np.eye(d)
                    #Rt_R = R.T @ R + lbd * np.eye(d)
                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                else:
                    counter_upd_dwd = counter_upd_dwd + 1
                    print("update the covariance matrix") 
                    #old_mean = mea
                    #mean = np.mean(R, axis=0)
                    old_Cov = Cov
                    #Cov = update_covariance(Cov * old_R_centered.shape[0], old_R, R, old_mean, mean, X_up, X_dw)
                    Cov = update_covariance((Cov - lbd * np.eye(d)) * old_R_centered.shape[0], old_R, R, old_mean, mean, X_up, X_dw) * (1/R_centered.shape[0]) + lbd * np.eye(d)
                                        
                    P1 = ((old_Cov - lbd * np.eye(d)) * old_R_centered.shape[0] - s_prod(old_R, old_mean)) 
                    P2 = old_R.T @ old_R
                    print("P1 \n", P1)
                    print("P2 \n", P2)


                    #Rt_R = Rt_R + X_upd.T @ X_upd - X_dwd.T @ X_dwd
                    RR = X[M[:, idx] == 0, :]
                    mean = np.mean(RR, axis=0)
                    u = np.ones(RR.shape[0])
                    #print(a)
                    #print("\n", np.outer(u, a))
                    RR = RR - np.outer(u, mean)
                    print("RR\n", RR)
                    print("R centered ", R_centered)
                    RRt_RR = RR.T @ RR * (1/RR.shape[0]) + lbd * np.eye(d)
                    #Rt_RR = RR.T @ RR + lbd * np.eye(d)
                    print("error? \n")
                    print("index ", idx, "\n")
                    np.testing.assert_allclose(Cov, RRt_RR)
                    print("testing passed")
                    #input()
                if nup + ndw > -d ** info['exponent_d']:
                    #print("invert the matrix")
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


def gibb_sampl_over_parametrized_sampling(info):
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













def test_gibb_sampl_under_parametrized_sampling():
    # the test consists in running IterativeImputer with Ridge Regression,
    # and our handmade gibb sampling function
    print("test gibb sampl under parametr started")
    n = 200
    print("sqrt n ", np.sqrt(n))
    print("n ** (3/4)", n ** (3/4))
    print("n ** (3/4) / n", (n ** (3/4)) / n)
    d = 10
    gaussian = True
    lbd = 0.001 + 0.0
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
    #print(np.max(X))
    #print(np.min(X))
    if d == 2:
        mean = np.array([4, -5])
        cov = np.array([[4, -0.95],[-0.95, 0.25]])
        X = np.random.multivariate_normal(mean, cov, size=n)
    if gaussian:
        mean = np.random.rand(d)
        cov = np.random.rand(n, d)
        cov = cov.T @ cov + np.eye(d) * 0.1
        #print(cov)
        X = np.random.multivariate_normal(mean, cov, size=n)
        #print(X)
        #input()
    M = np.random.binomial(1, 0.5, size=(n, d))
    print(M)
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
    print("X_nan \n", X_nan)
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
        'exponent_d': 0.75
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
#    print("result IterativeImptuer with Ridge\n", res4)
    #end4 = time.time()     # toc
    #print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds\n\n")
    #if not info_dic['tsp']:
    #np.testing.assert_allclose(X_my, res4)
    #print("test gibb sampl under parametr ended successfully\n")



test_gibb_sampl_under_parametrized_sampling()


