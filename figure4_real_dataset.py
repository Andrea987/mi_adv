import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tsp import gibb_sampl_sampling
from generate import generate_mask_with_bounded_flip
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
    dataset = 'ecoli'
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
    R = 10
    intercept_switch = True
    df = pd.DataFrame(columns=['size', 'time_tsp_true', 'time_tsp_false'])
    list_df = []
    rep = 1
    p_value_array = np.zeros((rep, R))
    stat_array = np.zeros((rep, R))
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
            stat_array[r, i] = stat
        info_dic['data'], info_dic['masks'] = None, None
        
    print("\n\n SHOW THE RESULTS")
    
    folder = Path("results/experiment4_real_dataset")
    folder.mkdir(parents=True, exist_ok=True)

    np.save("results/experiment4_real_dataset/p_value_array.npy", p_value_array)
    np.save("results/experiment4_real_dataset/stat_array.npy", stat_array)
    np.save("results/experiment4_real_dataset/dataset.npy", np.array([dataset]))
    np.save("results/experiment4_real_dataset/size.npy", np.array([n]))
    np.save("results/experiment4_real_dataset/dim.npy", np.array([d]))
    np.save("results/experiment4_real_dataset/R.npy", np.array([R]))
    np.save("results/experiment4_real_dataset/rep.npy", np.array([rep]))
    np.save("results/experiment4_real_dataset/prob_miss.npy", np.array([p_miss]))

    
    

def plot_fig4_real_dataset():

    dataset = np.load("results/experiment4_real_dataset/dataset.npy")[0]
    n = np.load("results/experiment4_real_dataset/size.npy")
    d = np.load("results/experiment4_real_dataset/dim.npy")
    #R = np.load("results/experiment4_real_dataset/R.npy")[0]
    #rep = np.load("results/experiment4_real_dataset/rep.npy")
    p_miss = np.load("results/experiment4_real_dataset/prob_miss.npy")
    p_value_array = np.load("results/experiment4_real_dataset/p_value_array.npy")
    stat_array = np.load("results/experiment4_real_dataset/stat_array.npy")

    rep, R = p_value_array.shape

    p_value_array_mean = p_value_array.mean(axis=0)
    p_value_array_std = p_value_array.std(axis=0)
    stat_array_mean = stat_array.mean(axis=0)
    stat_array_std = stat_array.std(axis=0)

    fig, (ax_stat, ax_pval) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    ax_stat.plot(np.arange(R), stat_array_mean, label="energy distance", marker="o", color="green")
    ax_stat.fill_between(
        np.arange(R),
        stat_array_mean - stat_array_std,
        stat_array_mean + stat_array_std,
        alpha=0.3,
        color="green",
        label="±1 std"
    )
    ax_stat.set_ylabel("energy distance", fontsize=20)
    ax_stat.legend(fontsize=14)
    ax_stat.grid()

    ax_pval.plot(np.arange(R), p_value_array_mean, label="p-value", marker="o", color="blue")
    ax_pval.fill_between(
        np.arange(R),
        p_value_array_mean - p_value_array_std,
        p_value_array_mean + p_value_array_std,
        alpha=0.3,
        color="blue",
        label="±1 std"
    )
    ax_pval.set_xlabel("Iteration", fontsize=20)
    ax_pval.set_ylabel("p-value", fontsize=20)
    ax_pval.legend(fontsize=14)
    ax_pval.grid()

    fig.tight_layout()
    fig.savefig("results/experiment4_real_dataset/plot.png")
    plt.show()


def coverage_experiment_real_dataset():
    print("Empirical coverage experiment on real dataset (Gibbs sampler)")
    dataset = 'ecoli'
    X_orig = dataset_loader(dataset)
    n, d = X_orig.shape
    mean = np.mean(X_orig, axis=0)
    std = np.std(X_orig, axis=0)
    X = (X_orig - mean) / std
    p_miss = 0.25
    lbd = 0.001
    intercept_switch = True

    R_total = 2000  # total number of Gibbs sweeps (chain length)
    burn_in = 500  # sweeps discarded before collecting draws

    M = np.random.binomial(n=1, p=p_miss, size=(n, d))

    info_dic = {
        'data': X,
        'masks': M,
        'imputed_data': None,
        'nbr_it_gibb_sampl': 1,
        'lbd_reg': lbd,
        'tsp': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'sampling': True,
        'intercept': intercept_switch,
    }

    chain = []  # every post-burn-in sweep, unthinned, needed for the autocorrelation check
    for it in range(R_total):
        X_filled = gibb_sampl_sampling(info_dic)
        info_dic['imputed_data'] = X_filled
        if it >= burn_in:
            chain.append(X_filled[M == 1].copy())
        print(f"sweep {it + 1}/{R_total}")

    chain = np.stack(chain, axis=0)  # (n_sweeps_post_burn_in, n_missing)
    true_missing = X[M == 1]  # ground truth standardized values at missing positions

    folder = Path("results/experiment4_coverage")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "chain.npy", chain)
    np.save(folder / "true_missing.npy", true_missing)
    np.save(folder / "dataset.npy", np.array([dataset]))
    np.save(folder / "params.npy", np.array([n, d, p_miss, R_total, burn_in]))


def plot_coverage_experiment():
    folder = Path("results/experiment4_coverage")
    chain = np.load(folder / "chain.npy")
    true_missing = np.load(folder / "true_missing.npy")

    nominal_levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    empirical_coverage = []
    for level in nominal_levels:
        alpha = 1 - level
        lo = np.quantile(chain, alpha / 2, axis=0)
        hi = np.quantile(chain, 1 - alpha / 2, axis=0)
        covered = (true_missing >= lo) & (true_missing <= hi)
        empirical_coverage.append(covered.mean())
    empirical_coverage = np.array(empirical_coverage)

    print("nominal levels:   ", nominal_levels)
    print("empirical coverage:", empirical_coverage)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(nominal_levels, empirical_coverage, marker="o", label="Gibbs sampler")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal calibration")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(folder / "coverage_plot.png")
    plt.show()


def autocorrelation_diagnostics(max_lag=100):
    folder = Path("results/experiment4_coverage")
    chain = np.load(folder / "chain.npy")  # (n_sweeps, n_missing)
    n_sweeps, n_missing = chain.shape
    max_lag = min(max_lag, n_sweeps - 1)

    centered = chain - chain.mean(axis=0, keepdims=True)
    var = np.mean(centered ** 2, axis=0)  # (n_missing,)
    var_safe = np.where(var > 0, var, 1.0)

    acf = np.zeros((max_lag + 1, n_missing))
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        cov_lag = np.mean(centered[:-lag] * centered[lag:], axis=0)
        acf[lag] = cov_lag / var_safe

    mean_acf = acf.mean(axis=1)  # averaged across missing entries

    # per-entry ESS via Geyer's initial positive sequence: sum rho_k until first non-positive lag
    ess_per_entry = np.zeros(n_missing)
    for j in range(n_missing):
        rho = acf[1:, j]
        cutoff = np.argmax(rho <= 0) if np.any(rho <= 0) else max_lag
        tau = 1 + 2 * np.sum(rho[:cutoff])
        tau = max(tau, 1.0)
        ess_per_entry[j] = n_sweeps / tau

    report_lags = [l for l in [1, 5, 10, 20, 50, 100] if l <= max_lag]
    print(f"\nAutocorrelation diagnostics (chain length {n_sweeps}, {n_missing} missing entries):")
    for lag in report_lags:
        print(f"  mean ACF at lag {lag:3d}: {mean_acf[lag]:.4f}")
    print(f"  ESS across entries: min={ess_per_entry.min():.1f} "
          f"mean={ess_per_entry.mean():.1f} max={ess_per_entry.max():.1f} "
          f"(out of {n_sweeps} samples)")

    np.save(folder / "mean_acf.npy", mean_acf)
    np.save(folder / "ess_per_entry.npy", ess_per_entry)


time_comparison_real_dataset_cleaned()
#plot_fig4_real_dataset()

coverage_experiment_real_dataset()
#plot_coverage_experiment()
autocorrelation_diagnostics()






























