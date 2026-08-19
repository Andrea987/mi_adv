import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl_sampling
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


'''
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
            #print(X_my)
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

    folder = Path("results/experiment_5")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_5/time_tsp.npy", time_my_array)
    np.save("results/experiment_5/time_no_tsp.npy", time_skl_array)
    np.save("results/experiment_5/list_d.npy", np.array(list_d))
    np.save("results/experiment_5/size.npy", np.array([n]))
    np.save("results/experiment_5/dim.npy", np.array([d]))
    np.save("results/experiment_5/R.npy", np.array([R]))
    np.save("results/experiment_5/rep.npy", np.array([rep]))
    np.save("results/experiment_5/prob_miss.npy", np.array([p_miss]))


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
'''

np.random.seed(54321)

def time_comparison_tsp_works_cleaned():
    print("Suppose n>d")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    list_n = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]  # increasing order
    list_d = [200]  # increasing order
    #list_n = [20]  # increasing order
    #list_d = [8]  # increasing order
    n, d = list_n[-1], list_d[-1]
    ns1 = 150
    ns2 = 150
    print(n, d)
    lbd = 1.01 + 0.0
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
    
    #M = np.random.binomial(n=1, p=p_miss, size= (n, d))
    M = np.zeros((n, d))
    for i in range(d):
        M[0:ns1, i] = np.zeros(ns1) if i % 2 == 0 else np.ones(ns1)
        M[ns1:(ns1+ns2), i] = np.zeros(ns2) if i % 2 == 1 else np.ones(ns2)
    #print("M in figure 5:\n", M[0:10, 0:6])
    #input()
    #for ii in range(d):
    #    nbr = np.random.randint(0, n)
    #    if np.sum(M[:, ii]) == n:
    #        print("add a random seen component")
    #        M[nbr, ii] = 0

    X_nan = X.copy()
    X_nan[M==1] = np.nan
    R = 2
    intercept_switch = True
    df = pd.DataFrame(columns=['size', 'time_tsp_true', 'time_tsp_false'])
    list_df = []
    rep = 20
    time_tsp_true_array = np.zeros((rep, len(list_n)))
    time_tsp_false_array = np.zeros((rep, len(list_n)))
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        df = pd.DataFrame(columns=['size', 'time_tsp_true', 'time_tsp_false'])
        for k, n_k in enumerate(list_n):
            print("\n\n CURRENT SIZE DTS ", n_k)
            MM = M[0:n_k, 0:d]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            print("nbr seen components ", n_k - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n_k, 0:d],
                'masks': M[0:n_k, 0:d],
                'imputed_data': None,
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': True,
                #'recomputation': False,
                'batch_size': 64,
                'verbose': 0,
                'initial_strategy': 'constant',
                'exponent_d': 0.75,
                'sampling': False,
                'intercept': intercept_switch
            }
            info_dic_tsp_false = copy.deepcopy(info_dic)
            info_dic_tsp_false['tsp'] = False

            start_time_tsp_true = time.time()
            X_my = gibb_sampl_sampling(info_dic)
            end_time_tsp_true = time.time()
            print("current size dts ", n_k)
            t_tsp_true = end_time_tsp_true - start_time_tsp_true  # total time my
            print(f"Execution time gs tsp true: {t_tsp_true:.4f} seconds")
            print("\nend my gibb sampling tsp true\n")

            print("starting gs tsp false")
            start_tsp_false = time.time()
            X_my_tsp_false = gibb_sampl_sampling(info_dic_tsp_false)
            end_tsp_false = time.time()     # toc
            t_tsp_false = end_tsp_false - start_tsp_false  # total time gs tsp false
            print("current size ", n_k)
            print(f"Elapsed time gs tsp false: {t_tsp_false:.4f} seconds\n\n")
            #np.testing.assert_allclose(X_my, res_skl)  ## TESTING IF OUTPUT IS THE SAME
            #print("END SKL,\n\n START BASELINE")
            df.loc[len(df)] = [n_k, t_tsp_true, t_tsp_false]
        list_df.append(df) 
    final_df = pd.DataFrame(np.zeros((len(list_d), 3)), columns=['size', 'time_tsp_true', 'time_tsp_false'])
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
        time_tsp_true_array[j, :] = list_df[j]['time_tsp_true']
        time_tsp_false_array[j, :] = list_df[j]['time_tsp_false']
    print(time_tsp_true_array)
    print(time_tsp_false_array)
    folder = Path("results/experiment_5")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_5/time_tsp_true_array.npy", time_tsp_true_array)
    np.save("results/experiment_5/time_tsp_false_array.npy", time_tsp_false_array)
    np.save("results/experiment_5/list_n.npy", np.array(list_n))
    np.save("results/experiment_5/size.npy", np.array([n]))
    np.save("results/experiment_5/dim.npy", np.array([d]))
    np.save("results/experiment_5/R.npy", np.array([R]))
    np.save("results/experiment_5/rep.npy", np.array([rep]))
    #np.save("results/experiment_5/prob_miss.npy", np.array([p_miss]))

    
    

def plot_fig_5():

    n = np.load("results/experiment_5/size.npy")
    d = np.load("results/experiment_5/dim.npy")
    R = np.load("results/experiment_5/R.npy")
    rep = np.load("results/experiment_5/rep.npy")
    #p_miss = np.load("results/experiment_5/prob_miss.npy")
    list_d = np.load("results/experiment_5/list_n.npy")
    time_tsp_true_array = np.load("results/experiment_5/time_tsp_true_array.npy") / R  # Average time for one iteration
    time_tsp_false_array = np.load("results/experiment_5/time_tsp_false_array.npy") / R  # Average time for one iteration
    
    print("size dts: ", n, ", repetitons: ", rep)
    print("dimemsions: ", list_d)

    time_tsp_true_array_mean = time_tsp_true_array.mean(axis=0)
    time_tsp_true_array_std = time_tsp_true_array.std(axis=0)
    time_tsp_false_array_mean = time_tsp_false_array.mean(axis=0)
    time_tsp_false_array_std = time_tsp_false_array.std(axis=0)

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(list_d, time_tsp_true_array_mean, label="With TSP heuristic", marker="o", color=clr[0])
    plt.plot(list_d, time_tsp_false_array_mean, label="Baseline", marker="*", color=clr[1])
    #plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])

    #plt.plot(x, mean, label="Mean")

    plt.fill_between(
        list_d,
        time_tsp_true_array_mean - time_tsp_true_array_std,
        time_tsp_true_array_mean + time_tsp_true_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )

    plt.fill_between(
        list_d,
        time_tsp_false_array_mean - time_tsp_false_array_std,
        time_tsp_false_array_mean + time_tsp_false_array_std,
        alpha=0.3,
        color="green",
        label="±1 std"
    )

    plt.legend(fontsize=18)
    plt.xlabel("Dimension", fontsize=24)
    plt.ylabel("Average Time", fontsize=24)
    plt.grid()
    plt.show()




#time_comparison_tsp_works_cleaned()
plot_fig_5()



