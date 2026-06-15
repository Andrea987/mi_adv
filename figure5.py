import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl_under_parametrized_sampling, gibb_sampl_under_parametrized
from generate import generate_mask_with_bounded_flip, generate_masks_mnar
from utils import generate_matrix_with_bounded_flip
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge, Ridge
import pandas as pd



np.random.seed(54321)

### Assumption: to avoid numerical issue, we should renormalize the data
### that's not super realistic of course



def time_comparison_tsp_works():
    print("\n\nstarting plot some graph()\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [450, 600, 750, 900, 1050, 1200]  # increasing order
    list_d = [400]  # increasing order
    lbd = 0.8765 + 0.0
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
    #print(np.max(X))
    #print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    exponent = (n ** (3/4)) / n
    print("exponent", exponent)
    p1 = 1/2 - np.sqrt(1 - 2 * d/n)/2 if 2 * d/n>0 else d/(2 * n)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.1, p_flip=p1)
    p1 = 0.01
    #print("p1:   ", p1)
    #M = np.random.binomial(n=1, p=p1, size= (n, d))
    M = np.zeros((n, d))
    ns = int(n * p1)
    for i in range(d):  # n > d
        M[0:ns, i] = np.zeros(ns) if i % 2 == 0 else np.ones(ns)
    #M = np.random.binomial(n=1, p=p1, size= (n, d))
    #p_missing = [0.8 , 0.6, 0.3]
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    tsp_switch = True
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_miss'])
    #print(df)
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
            print("nbr seen components \n", n_j - np.sum(MM, axis=0))
            print("nbr missing components  \n", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n_j * p1 * (1-p1))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n_j, 0:d_i],
                'masks': M[0:n_j, 0:d_i],
                'imputed_data': None,
                'initial_strategy': 'constant',
                'exponent_d': 0.75,
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': tsp_switch,
                'recomputation': False,
                'batch_size': 64,
                'verbose': 0,
                'sampling': False,
                'intercept': True
            }
            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl_under_parametrized_sampling(info_dic)
            end_time_gibb_sampl = time.time()
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
        #   print(X_my)
            total_time_gibb_sampl[j, i] = end_time_gibb_sampl - start_time_gibb_sampl
            print("\nend my gibb sampling, tsp was \n", info_dic['tsp'])


            info_dic_baseline = copy.deepcopy(info_dic)
            info_dic_baseline['tsp'] = False
            start_baseline = time.time()   # tic
            #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            X_my_baseline = gibb_sampl_under_parametrized(info_dic_baseline)  
            # print("result IterativeImptuer with Ridge\n", res4)
            end_baseline = time.time()     # toc
            #np.testing.assert_allclose(X_my_baseline, res4) if info_dic['intercept'] is False else print("no test X_my_baseline vs res4")
            total_time_baseline[j, i] = end_baseline - start_baseline
            print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
            #if not info_dic['tsp']:
            #np.testing.assert_allclose(X_my, res4)
            #np.testing.assert_allclose(X_my, res4)
            print("test gibb sampl ended successfully")    
    print("total time gibb sampl\n", total_time_gibb_sampl)
    print("total time ridge\n", total_time_ridge)
    print("total time baseline\n", total_time_baseline)
    total_time_gibb_sampl = total_time_gibb_sampl / R
    total_time_ridge = total_time_ridge / R
    total_time_baseline = total_time_baseline / R

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    for i, d_i in enumerate(list_d):
        plt.plot(list_n, total_time_gibb_sampl[:, i], label="our_gibb, dim: " + str(d_i), marker="o", color=clr[i])
        #plt.plot(list_n, total_time_ridge[:, i], label="ridge  , dim: " + str(d_i), marker="*", color=clr[i+1])
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


time_comparison_tsp_works()



