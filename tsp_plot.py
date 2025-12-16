import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl, gibb_sampl_no_modification
from generate import generate_mask_with_bounded_flip, generate_masks_mnar
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge, Ridge
import pandas as pd




np.random.seed(54321)


def plot_some_graph():
    print("\n\nstarting plot some graph()\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [1000, 2000, 4000, 6000]  # increasing order
    list_d = [100]  # increasing order
    lbd = 1 + 0.0
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
    print(np.max(X))
    print(np.min(X))
    #M = np.random.binomial(1, 0.01, size=(n, d))
    exponent = (n ** (3/4)) / n
    print("exponent", exponent)
    p1 = 1/2 - np.sqrt(1 - 2 * d/n)/2 if 2 * d/n>0 else d/(2 * n)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.1, p_flip=p1)
    p1 = 0.4
    #print("p1:   ", p1)
    M = np.random.binomial(n=1, p=p1, size= (n, d))
    #p_missing = [0.8 , 0.6, 0.3]
    #M = np.array([np.random.binomial(1, 1-pr, (nbr_of_sample, dim)) for pr in p_missing])
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    R = 2
    tsp_switch = False
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_miss'])
    print(df)
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
            print("nbr seen components ", n_j - np.sum(MM, axis=0))
            print("nbr missing components ", np.sum(MM, axis=0))
            print("2 * n * p1 * (1-p1):   ", 2 * n_j * p1 * (1-p1))
            #FF = flip_matrix(M.T)
            #ones_d = np.ones(d_i)
            #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
            #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
            #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
            info_dic = {
                'data': X[0:n_j, 0:d_i],
                'masks': M[0:n_j, 0:d_i],
                'initial_strategy': 'constant',
                'exponent_d': 0.75,
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': tsp_switch,
                'recomputation': False,
                'batch_size': 64,
                'verbose': 0
            }
            start_time_gibb_sampl = time.time()
            X_my = gibb_sampl(info_dic)
            end_time_gibb_sampl = time.time()
            print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
        #   print(X_my)
            total_time_gibb_sampl[j, i] = end_time_gibb_sampl - start_time_gibb_sampl
            print("\nend my gibb sampling\n")

            print("It imputer Ridge Reg")
            start44 = time.time()   # tic
            ice4 = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy=info_dic['initial_strategy'], verbose=0)
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
            X_my_baseline = gibb_sampl_no_modification(info_dic)  
            # print("result IterativeImptuer with Ridge\n", res4)
            end_baseline = time.time()     # toc
            total_time_baseline[j, i] = end_baseline - start_baseline
            print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
            #if not info_dic['tsp']:
            #np.testing.assert_allclose(X_my, res4)
            #np.testing.assert_allclose(X_my, res4)
            print("test gibb sampl ended successfully")    
    print("total time gibb sampl\n", total_time_gibb_sampl)
    print("total time ridge\n", total_time_ridge)
    print("total time baseline\n", total_time_baseline)
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    for i, d_i in enumerate(list_d):
        plt.plot(list_n, total_time_gibb_sampl[:, i], label="our_gibb, dim: " + str(d_i), marker="o", color=clr[i])
        plt.plot(list_n, total_time_ridge[:, i], label="ridge  , dim: " + str(d_i), marker="*", color=clr[i+1])
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

## to do: run experiments with something like n=700, d=500, and study the result
## you can see that if the percentage of missing is greater than 0.5, running tsp is actually useful

def plot_some_graph_2():
    print("\n\nstarting plot some graph 2(). In this function we go through the probabilities\n")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_n = [800]  # increasing order
    list_d = [150]  # increasing order
    #list_p_seen_true = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
    list_p_seen_true = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
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
    lbd = 1.01 + 0.0
    n, d = list_n[-1], list_d[-1]
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
    print(np.max(X))
    print(np.min(X))
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
    R = 2
    tsp_switch = True
    df = pd.DataFrame(columns=['n_train', 'dim', 'p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print(df)
    total_time_gibb_sampl = np.zeros((len(list_n), len(list_d)))
    total_time_ridge = np.zeros_like(total_time_gibb_sampl)
    total_time_baseline = np.zeros_like(total_time_gibb_sampl)
    list_df = []
    rep = 1
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        df = pd.DataFrame(columns=['n_train', 'dim', 'p_seen', 'time_my', 'time_skl', 'time_bsl'])
        for i, d_i in enumerate(list_d):
            print("NEW RUN\n\ncurrent dimension ", d_i)
            for j, n_j in enumerate(list_n):
                for s in list_p_seen:
                    print(s)
                masks = np.array([np.random.binomial(1, 1-pr, (n_j, d_i)) for pr in list_p_seen])
                masks = np.cumsum(masks, axis=0)  # each round
                masks[masks>1] = 1

                masks = np.array([generate_masks_mnar(n_j, d_i, pr, 0.5) for pr in list_p_seen_true]) 
                
                for k, p_k in enumerate(list_p_seen_true):
                    print("\n\n CURRENT PROBABILITY ", p_k)
                    M = masks[k, :, :]
                    print("check the masks ", M[0:8, 0:8])
                    for ii in range(d_i):
                        nbr = np.random.randint(0, n_j)
                        #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
                        if np.sum(M[:, ii]) == n_j:
                            print("add a random seen component")
                            M[nbr, ii] = 0
                    X_nan = X.copy()
                    X_nan[M==1] = np.nan
                    #print("X_nan \n", X_nan)
                    ones = np.ones((d_i, d_i))
                    MM = M[0:n_j, 0:d_i]
                    #F = n_j * ones - MM.T @ MM - (np.ones_like(MM.T) - MM.T) @ (np.ones_like(MM) - MM)
                    #print("nbr seen components ", n_j - np.sum(MM, axis=0))
                    #print("nbr missing components ", np.sum(MM, axis=0))
                    print("2 * n * p1 * (1-p1):   ", 2 * n_j * p_k * (1-p_k))
                    
                    #FF = flip_matrix(M.T)
                    #ones_d = np.ones(d_i)
                    #F = n * ones - M.T @ M - (np.ones_like(M.T) - M.T) @ (np.ones_like(M) - M)
                    #F = np.outer(ones_d, np.sum(M, axis=0)) + np.outer(np.sum(M.T, axis=1), ones_d) - 2 * M_s.T @ M_s
                    #print("flip matrix in make mask with bounded flip\n", F[0:8, 0:8])
                    info_dic = {
                        'data': X[0:n_j, 0:d_i],
                        'masks': M,  #M[k, :, :],
                        'nbr_it_gibb_sampl': R,
                        'lbd_reg': lbd,
                        'tsp': tsp_switch,
                        #'recomputation': False,
                        'batch_size': 64,
                        'verbose': 0,
                        'initial_strategy': 'constant',
                        'exponent_d': 0.75
                    }
                    info_dic_baseline = copy.deepcopy(info_dic)
                    info_dic_baseline['tsp'] = False
                    #for key, values in info_dic.items():
                    #    if key not in ['data', 'masks']:
                    #        print(info_dic[key])
                    #        print(info_dic_baseline[key])
                    start_time_gibb_sampl = time.time()
                    X_my = gibb_sampl(info_dic)
                    end_time_gibb_sampl = time.time()
                    print("current prob seen ", p_k)
                    print(f"Execution time: {end_time_gibb_sampl - start_time_gibb_sampl:.4f} seconds")
                #   print(X_my)
                    t_my = end_time_gibb_sampl - start_time_gibb_sampl
                    total_time_gibb_sampl[j, i] = t_my  # end_time_gibb_sampl - start_time_gibb_sampl
                    print("\nend my gibb sampling\n")

                    print("It imputer Ridge Reg")
                    #start_skl = time.time()   # tic
                    ice_skl = IterativeImputer(estimator=Ridge(fit_intercept=False, alpha=lbd), imputation_order='roman', max_iter=R, initial_strategy='constant', verbose=0)
                    #end_skl = time.time()   # tic
                    #print(f"Elapsed time no 4 iterative imputer definition: {end_skl_ptl - start_skl_ptl:.4f} seconds\n\n")

                    start_skl = time.time()
                    res_skl = ice_skl.fit_transform(X_nan[0:n_j, 0:d_i])
                    #print("result IterativeImptuer with Ridge\n", res4)
                    end_skl = time.time()     # toc
                    t_skl = end_skl - start_skl
                    total_time_ridge[j, i] = end_skl - start_skl 
                    print("current prob seen ", p_k)
                    print(f"Elapsed time no 4 iterative imputer Ridge Reg prec: {end_skl - start_skl:.4f} seconds\n\n")
                    #np.testing.assert_allclose(X_my, res_skl)
                    print("END SKL,\n\n START BASELINE")
                    start_baseline = time.time()   # tic
                    #res4 = ice4.fit_transform(X_nan[0:n_j, 0:d_i])
                    #X_my_baseline = gibb_sampl_no_modification(info_dic)
                    #info_dic['tsp'] = False
                    X_my_baseline = gibb_sampl(info_dic_baseline)  
                    # print("result IterativeImptuer with Ridge\n", res4)
                    end_baseline = time.time()     # toc
                    t_bsl = end_baseline - start_baseline
                    total_time_baseline[j, i] = end_baseline - start_baseline
                    if info_dic_baseline['tsp'] == False:
                        print("CHECK IF WVERYTHING IS CORRECT")
                        np.testing.assert_allclose(X_my_baseline, res_skl)

                    df.loc[len(df)] = [n_j, d_i, p_k, t_my, t_skl, t_bsl]
                    print("current prob seen ", p_k)
                    print(f"Elapsed time no 4 iterative imputer baseline prec: {end_baseline - start_baseline:.4f} seconds\n\n")
                    #if not info_dic['tsp']:
                    #np.testing.assert_allclose(X_my, res4)
                    #print("test baseline ended successfully")  
        list_df.append(df) 
    final_df = pd.DataFrame(np.zeros((len(list_p_seen_true), 6)), columns=['n_train', 'dim', 'p_seen', 'time_my', 'time_skl', 'time_bsl'])
    print(final_df)
    for s in list_df:
        final_df = final_df + s
        print(final_df)
    final_df = final_df / rep
    print(list_df)
    print("\nfinal df\n", final_df)
    print("\n\n SHOW THE RESULTS")
    dd = d ** info_dic['exponent_d']
    p1 = 1/2 - np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
    p2 = 1/2 + np.sqrt(1 - 2 * dd/n)/2 if 2 * d/n>0 else d/(2 * n)
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
    for i, d_i in enumerate(list_d):
        plt.plot(list_p_seen_true, final_df['time_my'], label="our_gibb, dim: " + str(d_i), marker="o", color=clr[i])
        plt.plot(list_p_seen_true, final_df['time_skl'], label="ridge  , dim: " + str(d_i), marker="*", color=clr[i+1])
        plt.plot(list_p_seen_true, final_df['time_bsl'], label="baseline  , dim: " + str(d_i), marker="s", color=clr[i+2])
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


plot_some_graph()
#plot_some_graph_2()

