import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl, gibb_sampl_sampling
from generate import generate_mask_with_bounded_flip, generate_masks_mnar
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge, Ridge
import pandas as pd
from pathlib import Path





def time_vs_probabilities_missing_cleaned():
    print("Suppose d>n")
    print("\n\nstarting plot some graph 2(). In this function we go through the probabilities\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    n = 5000  # increasing order
    d = 350  # increasing order
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    list_p_seen_true = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    #list_p_seen_true = [0.05, 0.01, 0.005]
    list_p_seen = list_p_seen_true[:-1]
    list_p_seen.insert(0, 1.0)
    print("list p seen true ", list_p_seen_true)
    print("list prob        ", list_p_seen)
    list_p_seen = [list_p_seen_true[i] / list_p_seen[i] for i in range(len(list_p_seen))]
    print("list p _seen ", list_p_seen)
    print("true probabilities, cumprod ", np.cumprod(list_p_seen)) 
    lbd = 100.01 + 0.0
    #print("sqrt n ", np.sqrt(n), "n ** (3/4) / n", (n ** (3/4)) / n)
    #print("n ** (3/4)", n ** (3/4))
    #X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
    X_orig = np.random.rand(n, d) + 0.0
    print(X_orig.dtype)
    print("max min ")
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    print("mean\n ", mean, "\nstd\n", std)
    # Standardize
    X = (X_orig - mean) / std
    #X = X_orig
    #X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    
    for s in list_p_seen:
        print(s)
    masks = np.array([np.random.binomial(n=1, p=1-pr, size=(n, d)) for pr in list_p_seen])
    masks = np.cumsum(masks, axis=0)  # each round
    masks[masks>1] = 1

    #masks = np.array([generate_masks_mnar(n_j, d_i, pr, 0.5) for pr in list_p_seen_true]) 
    #M = np.random.binomial(1, 0.01, size=(n, d))
    #p1 = 1/2 - np.sqrt(1 - 2 * d/n)/2 if 2 * d/n>0 else d/(2 * n)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.1, p_flip=p1)
    #p1 = 0.4
    #print("p1:   ", p1)
    #M = np.random.binomial(n=1, p=p1, size= (n, d))
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    
    #X_nan = X.copy()
    #X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 1
    tsp_switch = False
    intercept_switch = True
    df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print(df)
    total_time_gibb_sampl = np.zeros((n, d))
    total_time_ridge = np.zeros_like(total_time_gibb_sampl)
    total_time_baseline = np.zeros_like(total_time_gibb_sampl)
    list_df = []
    rep = 20
    time_my_array = np.zeros((rep, len(list_p_seen_true)))
    time_skl_array = np.zeros((rep, len(list_p_seen_true)))
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
        for k, p_k in enumerate(list_p_seen_true):
            print("\n\n CURRENT PROBABILITY ", p_k)
            M = masks[k, :, :]
            print("check the masks ", M[0:8, 0:8])
            for ii in range(d):
                nbr = np.random.randint(0, n)
                #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
                if np.sum(M[:, ii]) == n:
                    print("add a random seen component")
                    M[nbr, ii] = 0
            X_nan = X.copy()
            X_nan[M==1] = np.nan
            #print("X_nan \n", X_nan)
            ones = np.ones((d, d))
            MM = M[0:n, 0:d]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            #print("nbr seen components ", n_j - np.sum(MM, axis=0))
            #print("nbr missing components ", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n * p_k * (1-p_k))
            
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n, 0:d],
                'masks': M,  #M[k, :, :],
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
            #for key, values in info_dic.items():
            #    if key not in ['data', 'masks']:
            #        print(info_dic[key])
            #        print(info_dic_baseline[key])
            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl_sampling(info_dic)
            end_time_gibb_sampl = time.time()
            print("current prob seen ", p_k)
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
        #   print(X_my)
            t_my = end_time_gibb_sampl - start_time_gibb_sampl
            total_time_gibb_sampl = t_my  # end_time_gibb_sampl - start_time_gibb_sampl
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            #start_skl = time.time()   # tic
            ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=intercept_switch, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='constant', verbose=0)
            #end_skl = time.time()   # tic
            #print(f"Elapsed time no 4 iterative imputer definition: {end_skl_ptl - start_skl_ptl:.4f} seconds\n\n")

            start_skl = time.time()
            res_skl = ice_skl.fit_transform(X_nan)
            #print("result IterativeImptuer with Ridge\n", res4)
            end_skl = time.time()     # toc
            t_skl = end_skl - start_skl
            total_time_ridge = end_skl - start_skl 
            print("current prob seen ", p_k)
            print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end_skl - start_skl:.4f} seconds\n\n")
            
            #np.testing.assert_allclose(X_my, res_skl)  ## TESTING IF OUTPUT IS THE SAME
            
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

            df.loc[len(df)] = [p_k, t_my, t_skl, t_bsl]
            print("current prob seen ", p_k)
            print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
            #if not info_dic['tsp']:
            #np.testing.assert_allclose(X_my, res4)
            #print("test baseline ended successfully")  
        list_df.append(df) 
    final_df = pd.DataFrame(np.zeros((len(list_p_seen_true), 4)), columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print("FINAL DF\n")
    print(final_df)
    for s in list_df:
        final_df = final_df + s
        #print(final_df)
    final_df = final_df / rep
    print("list df \n ", list_df)
    print("\nfinal df\n", final_df)
    print("\n\n SHOW THE RESULTS")
    dd = d ** info_dic['exponent_d']
    p1 = 1/2 - np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
    p2 = 1/2 + np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)

    for j in range(rep):
        time_my_array[j, :] = list_df[j]['time_my']
        time_skl_array[j, :] = list_df[j]['time_skl']
    print(time_my_array)
    print(time_skl_array)
    time_my_array_mean = time_my_array.mean(axis=0)
    time_my_array_std = time_my_array.std(axis=0)
    time_skl_array_mean = time_skl_array.mean(axis=0)
    time_skl_array_std = time_skl_array.std(axis=0)
    folder = Path("results/experiment_12")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_12/time_my_array.npy", time_my_array)
    np.save("results/experiment_12/time_skl_array.npy", time_skl_array)
    np.save("results/experiment_12/list_p_seen_true.npy", np.array(list_p_seen_true))
    np.save("results/experiment_12/size.npy", np.array([n]))
    np.save("results/experiment_12/dim.npy", np.array([d]))
    np.save("results/experiment_12/R.npy", np.array([R]))
    np.save("results/experiment_12/rep.npy", np.array([rep]))
    
    
    

    




def plot_fig_12():
    n = np.load("results/experiment_12/size.npy")
    d = np.load("results/experiment_12/dim.npy")
    R = np.load("results/experiment_12/R.npy")
    rep = np.load("results/experiment_12/rep.npy")
    list_p_seen_true = np.load("results/experiment_12/list_p_seen_true.npy") 
    time_my_array = np.load("results/experiment_12/time_my_array.npy") / R
    time_skl_array = np.load("results/experiment_12/time_skl_array.npy") / R
    
    
    time_my_array_mean = time_my_array.mean(axis=0)
    time_my_array_std = time_my_array.std(axis=0)
    time_skl_array_mean = time_skl_array.mean(axis=0)
    time_skl_array_std = time_skl_array.std(axis=0)

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(list_p_seen_true, time_my_array_mean, label="Our implementation", marker="o", color=clr[0])
    plt.plot(list_p_seen_true, time_skl_array_mean, label="Iterative Imputer Ridge", marker="*", color=clr[1])
    #plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])

    #plt.plot(x, mean, label="Mean")

    plt.fill_between(
        list_p_seen_true,
        time_my_array_mean - time_my_array_std,
        time_my_array_mean + time_my_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )

    plt.fill_between(
        list_p_seen_true,
        time_skl_array_mean - time_skl_array_std,
        time_skl_array_mean + time_skl_array_std,
        alpha=0.3,
        color="green",
        label="±1 std"
    )

    plt.legend()
    plt.xlabel("Probability of observation")
    plt.ylabel("Average Time")
    plt.grid()
    plt.show()





#time_vs_probabilities_missing_cleaned()
plot_fig_12()
    

