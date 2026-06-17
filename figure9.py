import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl_over_parametrized_sampling, gibb_sampl_over_parametrized, gibb_sampl_sampling
from generate import generate_mask_with_bounded_flip, generate_masks_mnar
from utils import generate_matrix_with_bounded_flip
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge, Ridge
import pandas as pd
from pathlib import Path



np.random.seed(54321)

### Assumption: to avoid numerical issue, we should renormalize the data
### that's not super realistic of course


def time_comparison_classical_cleaned():
    print("Suppose d>n")
    print("\n\nstarting plot some graph 2(). In this function we go through the probabilities\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_d = [150]  # increasing order
    list_n = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]  # increasing order
    n, d = list_n[-1], list_d[-1]
    print(n, d)
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.05, 0.01, 0.005]
    lbd = 1.01 + 0.0
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
    
    exponent = (n ** (3/4)) / n
    #p_miss = 0.6  # prob miss
    M = np.zeros((n, d))
    for i in range(d):  # n > d
        M[i, i] = 1
        M[i+1, i] = 1
    
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    R = 2
    tsp_switch = False
    intercept_switch = True
    df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
    list_df = []
    rep = 6
    time_my_array = np.zeros((rep, len(list_n)))
    time_skl_array = np.zeros((rep, len(list_n)))
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
        for k, n_k in enumerate(list_n):
            print("\n\n CURRENT DIMENSION ", n_k)
            MM = M[0:n, 0:n_k]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            print("nbr seen components ", n - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            #print("2 * n * p1 * (1-p1):   ", 2 * n * p_miss * (1-p_miss))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n, 0:n_k],
                'masks': M[0:n, 0:n_k],
                'imputed_data': None,
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': tsp_switch,
                #'recomputation': False,
                'batch_size': 64,
                'verbose': 0,
                'initial_strategy': 'constant',
                'exponent_d': 0.75,
                'sampling': False,
                'intercept': intercept_switch
            }
            info_dic_baseline = copy.deepcopy(info_dic)
            info_dic_baseline['tsp'] = False

            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl_sampling(info_dic)
            end_time_gibb_sampl = time.time()
            print("current dimension ", n_k)
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
            t_my = end_time_gibb_sampl - start_time_gibb_sampl  # total time my
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=intercept_switch, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='constant', verbose=0)
            start_skl = time.time()
            res_skl = ice_skl.fit_transform(X_nan[0:n, 0:n_k])
            end_skl = time.time()     # toc
            t_skl = end_skl - start_skl  # total time iterative imputer ridge
            print("current dimension ", n_k)
            print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end_skl - start_skl:.4f} seconds\n\n")
            np.testing.assert_allclose(X_my, res_skl)  ## TESTING IF OUTPUT IS THE SAME
            print("END SKL,\n\n START BASELINE")
            start_baseline = time.time()   # tic
            #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            #X_my_baseline = gibb_sampl_no_modification(info_dic)
            #info_dic['tsp'] = False
            #X_my_baseline = gibb_sampl(info_dic_baseline)  
            # print("result IterativeImptuer with Ridge\n", res4)
            end_baseline = time.time()     # toc
            t_bsl = end_baseline - start_baseline
            total_time_baseline = end_baseline - start_baseline
            if info_dic_baseline['tsp'] == False:
                print("CHECK IF WVERYTHING IS CORRECT")
                #np.testing.assert_allclose(X_my_baseline, res_skl)

            df.loc[len(df)] = [n_k, t_my, t_skl, t_bsl]
            print("current prob seen ", n_k)
            print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
            #if not info_dic['tsp']:
            #np.testing.assert_allclose(X_my, res4)
            #print("test baseline ended successfully")  
        list_df.append(df) 
    final_df = pd.DataFrame(np.zeros((len(list_d), 4)), columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print("FINAL DF\n")
    print(final_df)
    for s in list_df:
        final_df = final_df + s
        #print(final_df)
    final_df = final_df / rep
    print("list df \n ", list_df)
    print("\nfinal df\n", final_df)
    print("\n\n SHOW THE RESULTS")
    
    for j in range(rep):
        time_my_array[j, :] = list_df[j]['time_my']
        time_skl_array[j, :] = list_df[j]['time_skl']
    print(time_my_array)
    print(time_skl_array)
    folder = Path("results/experiment_9")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_9/time_my_array.npy", time_my_array)
    np.save("results/experiment_9/time_skl_array.npy", time_skl_array)
    np.save("results/experiment_9/list_n.npy", np.array(list_n))
    np.save("results/experiment_9/size.npy", np.array([n]))
    np.save("results/experiment_9/dim.npy", np.array([d]))
    np.save("results/experiment_9/R.npy", np.array([R]))
    np.save("results/experiment_9/rep.npy", np.array([rep]))
    #np.save("results/experiment_9/prob_miss.npy", np.array([p_miss]))
    


def plot_fig_9():

    n = np.load("results/experiment_9/size.npy")
    d = np.load("results/experiment_9/dim.npy")
    R = np.load("results/experiment_9/R.npy")
    rep = np.load("results/experiment_9/rep.npy")
    #p_miss = np.load("results/experiment_9/prob_miss.npy")
    list_n = np.load("results/experiment_9/list_n.npy")
    time_my_array = np.load("results/experiment_9/time_my_array.npy") / R  # Average time for one iteration
    time_skl_array = np.load("results/experiment_9/time_skl_array.npy") / R  # Average time for one iteration
    
    print("size dts: ", n, ", repetitons: ", rep)
    print("size dts's: ", list_n)

    time_my_array_mean = time_my_array.mean(axis=0)
    time_my_array_std = time_my_array.std(axis=0)
    time_skl_array_mean = time_skl_array.mean(axis=0)
    time_skl_array_std = time_skl_array.std(axis=0)

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(list_n, time_my_array_mean, label="Our implementation", marker="o", color=clr[0])
    plt.plot(list_n, time_skl_array_mean, label="Iterative Imputer Ridge", marker="*", color=clr[1])
    #plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])

    #plt.plot(x, mean, label="Mean")

    plt.fill_between(
        list_n,
        time_my_array_mean - time_my_array_std,
        time_my_array_mean + time_my_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )

    plt.fill_between(
        list_n,
        time_skl_array_mean - time_skl_array_std,
        time_skl_array_mean + time_skl_array_std,
        alpha=0.3,
        color="green",
        label="±1 std"
    )

    plt.legend(fontsize=18)
    plt.xlabel("Size Dataset", fontsize=24)
    plt.ylabel("Average Time", fontsize=24)
    plt.grid()
    plt.show()




#time_comparison_classical_cleaned()
plot_fig_9()





















