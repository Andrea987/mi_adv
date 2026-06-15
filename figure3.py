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



def time_comparison_high_dimensional_test():
    print("\n\nstarting plot some graph()\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [100]  # increasing order
    list_d = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]  # increasing order
    lbd = 1.2 + 0.0
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
    p1 = 0.3  # prob missing
    #print("p1:   ", p1)
    M = np.random.binomial(n=1, p=p1, size= (n, d))
    for ii in range(d):
        nbr = np.random.randint(0, n)
        #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
        if np.sum(M[:, ii]) == n:
            print("add a random seen component")
            M[nbr, ii] = 0
    #M = np.zeros((n, d))
    #for i in range(d):  # n > d
    #    M[i, i] = 1
    #    M[i+1, i] = 1
    #M = np.random.binomial(n=1, p=p1, size= (n, d))
    #p_missing = [0.8 , 0.6, 0.3]
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    tsp_switch = False
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_miss'])
    #print(df)
    total_time_gibb_sampl = np.zeros((len(list_n), len(list_d)))
    total_time_ridge = np.zeros_like(total_time_gibb_sampl)
    total_time_baseline = np.zeros_like(total_time_gibb_sampl)
    for i, d_i in enumerate(list_d):
        print("\ncurrent dimension ", d_i)
        for j, n_j in enumerate(list_n):
            #print("\n\n current size ", n_j)
            ones = np.ones((d_i, d_i))
            MM = M[0:n_j, 0:d_i]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            print("nbr seen components ", n_j - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n_j * p1 * (1-p1))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            intercept_switch = False 
            info_dic = {
                'data': X[0:n_j, 0:d_i],
                'masks': M[0:n_j, 0:d_i],
                'imputed_data': None,
                'initial_strategy': 'constant',
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': tsp_switch,
                'recomputation': False,
                'batch_size': 64,
                'verbose': 0,
                'sampling': False,
                'intercept': intercept_switch
            }
            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl_over_parametrized_sampling(info_dic)
            end_time_gibb_sampl = time.time()
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
        #   print(X_my)
            total_time_gibb_sampl[j, i] = end_time_gibb_sampl - start_time_gibb_sampl
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            start44 = time.time()   # tic
            ice4 = IterativeImputer(estimator=Ridge(fit_intercept=intercept_switch, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
            end44 = time.time()   # tic
            print(f"Elapsed time no 4 iterative imputer definition: {end44 - start44:.4f} seconds\n\n")

            start4 = time.time()
            res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            #print("result IterativeImptuer with Ridge\n", res4)
            end4 = time.time()     # toc
            total_time_ridge[j, i] = end4 - start4 
            print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end4 - start4:.4f} seconds\n\n")
            np.testing.assert_allclose(X_my, res4)

            start_baseline = time.time()   # tic
            #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
            X_my_baseline = gibb_sampl_over_parametrized(info_dic)  
            # print("result IterativeImptuer with Ridge\n", res4)
            end_baseline = time.time()     # toc
            np.testing.assert_allclose(X_my_baseline, res4) if info_dic['intercept'] is False else print("no test X_my_baseline vs res4")
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
    for i, n_i in enumerate(list_n):
        plt.plot(list_d, total_time_gibb_sampl[i, :], label="our_gibb, n_train: " + str(n_i), marker="o", color=clr[i])
        plt.plot(list_d, total_time_ridge[i, :], label="ridge  , n_train: " + str(n_i), marker="*", color=clr[i+1])
        plt.plot(list_d, total_time_baseline[i, :], label="baseline  , n_train: " + str(n_i), marker="s", color=clr[i+2])
        #plt.plot(iterations, accuracy, label="Accuracy", color="blue")
        plt.xlabel("dimension")
        plt.ylabel("time")
    plt.title("Time in function of training size")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.text(5.05, 0.5, "ciao sono un testo", rotation=0)
    #text = "MCAR p_miss: " + str(p1) + "\n\n"
    #text = "prob flip: " + str(p1) + "\n\n"
    text1 = "tsp: " + str(tsp_switch) + ", nbr it: " + str(R)

    #plt.figtext(0.71, 0.65, "Extra info about curves:\n" + text + text1, fontsize=10)
    plt.tight_layout() 
    #plt.legend()
    plt.show()


def time_comparison_high_dimensional_cleaned():
    print("Suppose d>n")
    print("\n\nstarting plot some graph 2(). In this function we go through the probabilities\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [50]  # increasing order
    list_d = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]  # increasing order
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
    p1 = 0.3  # prob missing
    M = np.random.binomial(n=1, p=p1, size= (n, d))
    for ii in range(d):
        nbr = np.random.randint(0, n)
        if np.sum(M[:, ii]) == n:
            print("add a random seen component")
            M[nbr, ii] = 0

    X_nan = X.copy()
    X_nan[M==1] = np.nan
    R = 2
    tsp_switch = False
    intercept_switch = True
    df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
    list_df = []
    rep = 6
    time_my_array = np.zeros((rep, len(list_d)))
    time_skl_array = np.zeros((rep, len(list_d)))
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        df = pd.DataFrame(columns=['p_seen', 'time_my', 'time_skl', 'time_bsl'])
        for k, d_k in enumerate(list_d):
            print("\n\n CURRENT DIMENSION ", d_k)
            MM = M[0:n, 0:d_k]
            #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
            print("nbr seen components ", n - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n * p1 * (1-p1))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n, 0:d_k],
                'masks': M[0:n, 0:d_k],
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
            print("current dimension ", d_k)
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
            t_my = end_time_gibb_sampl - start_time_gibb_sampl  # total time my
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=intercept_switch, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='constant', verbose=0)
            start_skl = time.time()
            res_skl = ice_skl.fit_transform(X_nan[0:n, 0:d_k])
            end_skl = time.time()     # toc
            t_skl = end_skl - start_skl  # total time iterative imputer ridge
            print("current dimension ", d_k)
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

            df.loc[len(df)] = [d_k, t_my, t_skl, t_bsl]
            print("current prob seen ", d_k)
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
    dd = d ** info_dic['exponent_d']
    p1 = 1/2 - np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
    p2 = 1/2 + np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)

    for j in range(rep):
        time_my_array[j, :] = list_df[j]['time_my']
        time_skl_array[j, :] = list_df[j]['time_skl']
    print(time_my_array)
    print(time_skl_array)
    folder = Path("results/experiment_3")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_3/time_my_array.npy", time_my_array)
    np.save("results/experiment_3/time_skl_array.npy", time_skl_array)
    np.save("results/experiment_3/list_d.npy", np.array(list_d))
    np.save("results/experiment_3/size.npy", np.array([n]))
    np.save("results/experiment_3/dim.npy", np.array([d]))
    np.save("results/experiment_3/R.npy", np.array([R]))
    np.save("results/experiment_3/rep.npy", np.array([rep]))
    
    
    '''
    print("mean and var")
    print(time_my_array_mean)
    print(time_my_array_std)
    print("check if you have extraced the right experients")
    #input()

    ## when probability = p1 or p2, then 2n(p-1)p ~ d
    ## observe, 2n(p-1)p < np if p > (1/2), so if p greater than (1/2),
    ## the average number of seen component is grater than the average number of flip 
    ## if prob = d/n, the number of seen components is ~ d = n * (d/n)
    print("d ** ", info_dic['exponent_d'], ": ", dd)
    print("p1 ", p1, ",  p2 ", p2,  ",   d/n ", d/n)
    print("df \n", df) 
    print("total time gibb sampl\n", total_time_gibb_sampl)
    print("total time ridge\n", total_time_ridge)
    print("total time baseline\n", total_time_baseline)
    
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    
    plt.plot(list_p_seen_true, time_my_array_mean, label="our_gibb, dim: " + str(d), marker="o", color=clr[0])
    plt.plot(list_p_seen_true, time_skl_array_mean, label="ridge  , dim: " + str(d), marker="*", color=clr[1])
    #plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])

    #plt.plot(x, mean, label="Mean")

    plt.fill_between(
        list_p_seen_true,
        time_my_array_mean - time_my_array_std,
        time_my_array_mean + time_my_array_std,
        alpha=0.3,
        label="±1 std"
    )

    plt.fill_between(
        list_p_seen_true,
        time_skl_array_mean - time_skl_array_std,
        time_skl_array_mean + time_skl_array_std,
        alpha=0.3,
        label="±1 std"
    )

    plt.grid()
    plt.legend()
    plt.xlabel("Probability of observation")
    plt.ylabel("Averate Time")
    plt.show()
    '''

    '''
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    
    plt.plot(list_p_seen_true, final_df['time_my'], label="our_gibb, dim: " + str(d), marker="o", color=clr[0])
    plt.plot(list_p_seen_true, final_df['time_skl'], label="ridge  , dim: " + str(d), marker="*", color=clr[1])
    plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])
    #plt.plot(iterations, accuracy, label="Accuracy", color="blue")
    plt.axvline(x = p1, linestyle='--', linewidth=2, label="p1: sol 2np(1-p)=d^" +  str(info_dic['exponent_d']))
    plt.axvline(x = p2, linestyle='--', linewidth=2, label="p2: sol 2np(1-p)=d^" +  str(info_dic['exponent_d']))
    if n >= d:
        plt.axvline(x = d/n, linestyle='--', linewidth=1, label="d/n")
    plt.axvline(x = 1/2, linestyle='--', linewidth=0.5, label="1/2")
    #plt.axvline(x = d ** (3/4)/n, linestyle='--', linewidth=0.5)
    #plt.axvline(x = (1-d/n) * (d/n), linestyle='--', linewidth=2)
    #plt.axvline(x = 1-(1-d/n) * (d/n), linestyle='--', linewidth=2)
    #plt.axvline(x = (1-d/n) * d/n * (1/2), linestyle='--', linewidth=3)
    #plt.axvline(x = 1-(1-d/n) * d/n * (1/2), linestyle='--', linewidth=3)
    plt.xlabel("prob seen")
    plt.ylabel("time")
    plt.title("Time in function of training size")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #plt.text(5.05, 0.5, "ciao sono un testo", rotation=0)
    #text = "MCAR p_miss: " + str(p1) + "\n\n"
    #text = "prob flip: " + str(p1) + "\n\n"
    #text1 = "tsp: " + str(tsp_switch) + "nbr it: " + str(R)

    #plt.figtext(0.71, 0.65, "Extra info about curves:\n" + text + text1, fontsize=10)
    text = "Extra info about curves\n"
    text0 = "nbr train: " + str(n) + "\n\n"
    text1 = "right of the line d/n: nbr_seen> d\n\n"
    text2 = "left of the line d/n : nbr_seen< d\n\n"
    text3 = "between the lines p1,p2: nbr_flip > d ** " + str(info_dic['exponent_d']) + " = " + str(d ** info_dic['exponent_d']) + "\n\n"
    text4 = "right line (1/2): number seen greater than number flips\n\n"
    text5 = "left line (1/2):  number seen smaller than number flips\n\n"
    text6 = "nbr repetitions: " + str(rep) + "\n\n"
    plt.figtext(0.65, 0.37, text0 + text1 + text2 + text3 + text4 + text5 + text6, fontsize=10)
    plt.tight_layout()
    #plt.legend()
    plt.show()
    '''



def plot_fig_3():

    n = np.load("results/experiment_3/size.npy")
    d = np.load("results/experiment_3/dim.npy")
    R = np.load("results/experiment_3/R.npy")
    rep = np.load("results/experiment_3/rep.npy")
    list_d = np.load("results/experiment_3/list_d.npy")
    time_my_array = np.load("results/experiment_3/time_my_array.npy") / R  # Average time for one iteration
    time_skl_array = np.load("results/experiment_3/time_skl_array.npy") / R  # Average time for one iteration
    n = np.load("results/experiment_3/size.npy")
    d = np.load("results/experiment_3/dim.npy")
    R = np.load("results/experiment_3/R.npy")
    rep = np.load("results/experiment_3/rep.npy")

    print("size dts: ", n, ", repetitons: ", rep)
    print("dimemsions: ", list_d)

    time_my_array_mean = time_my_array.mean(axis=0)
    time_my_array_std = time_my_array.std(axis=0)
    time_skl_array_mean = time_skl_array.mean(axis=0)
    time_skl_array_std = time_skl_array.std(axis=0)

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(list_d, time_my_array_mean, label="Our implementation", marker="o", color=clr[0])
    plt.plot(list_d, time_skl_array_mean, label="Iterative Imputer Ridge", marker="*", color=clr[1])
    #plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d), marker="s", color=clr[2])

    #plt.plot(x, mean, label="Mean")

    plt.fill_between(
        list_d,
        time_my_array_mean - time_my_array_std,
        time_my_array_mean + time_my_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )

    plt.fill_between(
        list_d,
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




time_comparison_high_dimensional_cleaned()
plot_fig_3()


