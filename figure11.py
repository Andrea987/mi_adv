import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl_fast_sampling
#from generate import generate_mask_with_bounded_flip, generate_masks_mnar
from em_miss import em_miss, em_gaussian_missing
from pathlib import Path


np.random.seed(54321)

def GibbsEM_vs_EM():
    n = 200
    d = 5
    lbd = 0.0 + 0.0
    mean = np.random.rand(d)
    cov1 = np.random.rand(n, d)
    #cov1 = np.random.randint(0, 5, (n, d))
    M = np.random.binomial(1, 0.4, size=(n, d))
    for i in range(n):
        m = M[i, :]
        j = np.random.randint(0, d)
        if np.sum(m) == d:  # full missing
            M[i, j] = 0
    print("M in GibbbsEM_vs_EM\n", M)
    print("M in gibb sampling fast sampling, tsp_test.py\n", M) if n<=10 and d<=10 else print("")

    sampling, intercept = True, True
    cov = (cov1.T @ cov1) / n + np.eye(d) * 0.1
    rep, nbr_it = 15, 50
    it_em = nbr_it
    it_gibb_sampl = nbr_it

    differences = np.zeros((rep, nbr_it))
    for i in range(rep):
        print("rep: ", i)
        X_orig = np.random.multivariate_normal(mean, cov, size=n)
        info_dic_em = {
            'data': X_orig,
            'imputed_data': None,
            'masks': M,
            'lbd_reg': lbd,
            'tsp': False,
            'initial_strategy': 'mean',
            'starting_point': None,
            'tolerance': 0.0,
            'nbr_it_em': it_em,
            'sampling': False,
            'intercept': intercept,
            'batch_size': 64,
            'verbose': 0, 
            'sampling': False,
            'cov_gt': cov,
            'mean_gt': mean
        }
        info_dic_gs = {
            'data': X_orig,
            'imputed_data': None,
            'masks': M,
            'lbd_reg': lbd,
            'tsp': False,
            'recomputation': False,
            'initial_strategy': 'mean',
            'starting_point': None,
            #'tolerance': 1e-1,
            'nbr_it_gibb_sampl': it_gibb_sampl,
            'save_all_iterations': True,
            'gamma': None,
            'sampling': sampling,
            'intercept': intercept,
            'batch_size': 64,
            'verbose': 0, 
            'sampling': True,
            'cov_gt': cov,
            'mean_gt': mean
        }
        dict_em_res = em_miss(info_dic_em)
        dict_gs_res = gibb_sampl_fast_sampling(info_dic_gs)
        
        it_em_mean, it_em_cov = dict_em_res['list_mean'], dict_em_res['list_cov']
        it_gs_mean, it_gs_cov = dict_gs_res['list_mean'], dict_gs_res['list_cov']
        
        #print(it_em_mean)
        #print(it_em_cov, "\n\n")
        #print(it_gs_mean)
        #print(it_gs_cov)

        it_em_mean, it_em_cov = np.stack(it_em_mean), np.stack(it_em_cov)
        it_gs_mean, it_gs_cov = np.stack(it_gs_mean), np.stack(it_gs_cov)
        diff_mean = it_em_mean - it_gs_mean
        diff_cov = it_em_cov- it_gs_cov
        rmse_mean = np.linalg.norm(diff_mean, axis=-1)
        rmse_cov = np.linalg.norm(diff_cov, axis=(-1, -2))
        rmse = rmse_cov + rmse_mean
        differences[i, :] = rmse
        #print(rmse)
        #input()
    #print(differences)
    folder = Path("results/experiment_11")
    folder.mkdir(parents=True, exist_ok=True)
    np.save("results/experiment_11/differences.npy", differences)
    #np.save("results/experiment_11/rmse.npy", rmse)
    np.save("results/experiment_11/size.npy", np.array([n]))
    np.save("results/experiment_11/dim.npy", np.array([d]))
    np.save("results/experiment_11/nbr_it.npy", np.array([nbr_it]))
    
    #np.save("results/experiment_11/M.npy", M)
    


def plot_fig_11():
    n = np.load("results/experiment_11/size.npy")
    d = np.load("results/experiment_11/dim.npy")
    differences = np.load("results/experiment_11/differences.npy")
    #rmse = np.load("results/experiment_11/differences.npy")
    rep, nbr_it = differences.shape
    #p_miss = np.load("results/experiment_9/prob_miss.npy")
    
    print("size dts: ", n, ", repetitons: ", d)
    
    differences_mean = differences.mean(axis=0)
    differences_std = differences.std(axis=0)
    clr = ['blue', 'green', 'red', "orange", "purple", "brown", 'black', 'cyan', 'magenta', 'yellow']
    plt.plot(np.arange(nbr_it), differences_mean, marker="o", color=clr[4])
    
    plt.fill_between(
        np.arange(nbr_it),
        differences_mean - differences_std,
        differences_mean + differences_std,
        alpha=0.3,
        color="purple",
        #label="±1 std"
    )

    #plt.legend(fontsize=18)
    plt.xlabel("Iterations", fontsize=24)
    plt.ylabel("", fontsize=24)
    plt.grid()
    plt.show()

    #plt.plot(np.arange(nbr_it), np.log(differences_mean), marker="o", color=clr[4])
    #plt.show()


#time_comparison_classical_cleaned()

GibbsEM_vs_EM()
plot_fig_11()
    
    




