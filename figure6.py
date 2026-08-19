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
from dataset_load import dataset_loader
from sklearn.model_selection import train_test_split
import pandas as pd
from hyppo.ksample import Energy
from pathlib import Path

np.random.seed(54321)

DATASETS = ['iris', 'wine', 'boston', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white']

# check if all the variables are continuos
# 

def time_comparison_real_dataset_cleaned():
    print("Suppose n>d")
    #list_n = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    #list_d = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #list_n = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    #list_d = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    dataset = 'wine_quality_red'
    X_orig = dataset_loader(dataset)
    n, d = X_orig.shape
    print(X_orig.dtype)
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    lbd = 0.001 + 0.0
    # Standardize
    X = (X_orig - mean) / std
    #X = X_orig
    #X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    p_miss = 0.25
    R = 100
    intercept_switch = True
    df = pd.DataFrame(columns=['size', 'time_tsp_true', 'time_tsp_false'])
    list_df = []
    rep = 1
    p_value_array = np.zeros((rep, R))
    info_dic = {
            'data': None,
            'masks': None,
            'imputed_data': None,
            'nbr_it_gibb_sampl': 1,
            'lbd_reg': lbd,
            'tsp': False,
            #'recomputation': False,
            'batch_size': 64,
            'verbose': 0,
            'initial_strategy': 'constant',
            'exponent_d': 0.75,
            'sampling': True,
            'intercept': intercept_switch
        }
    M = np.random.binomial(n=1, p=p_miss, size= (n, d))
    for r in range(rep):
        print("\n\nREPETITION: ", r, "\n")
        X_train, X_test, M_train, M_test = train_test_split(X, M, test_size=0.30)
        n_tr, n_ts = X_train.shape[0], X_test.shape[0]
        info_dic['data'], info_dic['masks'] = X_train, M_train
        print("nbr seen components ", n_tr - np.sum(M_train, axis=0))
        print("nbr missing components ", np.sum(M_train, axis=0))
        for i in range(R):
            print("iterat gs", i)
            X_train_filled = gibb_sampl_sampling(info_dic)
            info_dic['imputed_data'] = X_train_filled
            stat, pvalue = Energy().test(X_test, X_train_filled)
            print("stat ", stat , "pvalue ", pvalue)
            #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
            p_value_array[r, i] = pvalue
        info_dic['data'], info_dic['masks'] = None, None
        
    print("\n\n SHOW THE RESULTS")
    
    folder = Path("results/experiment_6")
    folder.mkdir(parents=True, exist_ok=True)

    np.save("results/experiment_6/p_value_array.npy", p_value_array)
    np.save("results/experiment_6/dataset.npy", np.array([dataset]))
    np.save("results/experiment_6/size.npy", np.array([n]))
    np.save("results/experiment_6/dim.npy", np.array([d]))
    np.save("results/experiment_6/R.npy", np.array([R]))
    np.save("results/experiment_6/rep.npy", np.array([rep]))
    np.save("results/experiment_6/prob_miss.npy", np.array([p_miss]))

    
    

def plot_fig_6():

    dataset = np.load("results/experiment_6/dataset.npy")[0]
    n = np.load("results/experiment_6/size.npy")
    d = np.load("results/experiment_6/dim.npy")
    #R = np.load("results/experiment_6/R.npy")[0]
    #rep = np.load("results/experiment_6/rep.npy")
    p_miss = np.load("results/experiment_6/prob_miss.npy")
    p_value_array = np.load("results/experiment_6/p_value_array.npy")
    
    rep, R = p_value_array.shape
    
    p_value_array_mean = p_value_array.mean(axis=0)
    p_value_array_std = p_value_array.std(axis=0)

    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(np.arange(R), p_value_array_mean, label="p-value", marker="o", color=clr[0])
    
    plt.fill_between(
        np.arange(R),
        p_value_array_mean - p_value_array_std,
        p_value_array_mean + p_value_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )

    plt.legend(fontsize=18)
    plt.xlabel("Dimension", fontsize=24)
    plt.ylabel("Average Time", fontsize=24)
    plt.grid()
    plt.show()



time_comparison_real_dataset_cleaned()
plot_fig_6()






























