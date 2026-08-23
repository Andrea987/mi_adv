import numpy as np
import time
import traceback
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from hyppo.ksample import Energy

from tsp import gibb_sampl_sampling
from dataset_load import dataset_loader, DATASETS

np.random.seed(54321)

R_GOF = 10          # energy-distance goodness-of-fit sweeps
R_TOTAL = 1000       # coverage-chain length (reduced from 2000)
BURN_IN = 250        # keep same 25% burn-in fraction as before
P_MISS = 0.25
LBD = 0.001
INTERCEPT = True
MAX_N = 1000

OUT_ROOT = Path("results/experiment4_all_datasets")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_goodness_of_fit(X, n, d):
    M = np.random.binomial(n=1, p=P_MISS, size=(n, d))
    X_train, X_test, M_train, M_test = train_test_split(X, M, test_size=0.30)
    n_tr = X_train.shape[0]

    info_dic = {
        'data': X_train,
        'masks': M_train,
        'imputed_data': None,
        'nbr_it_gibb_sampl': 1,
        'lbd_reg': LBD,
        'tsp': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'sampling': True,
        'intercept': INTERCEPT,
    }

    p_values = np.zeros(R_GOF)
    stats = np.zeros(R_GOF)
    for i in range(R_GOF):
        X_train_filled = gibb_sampl_sampling(info_dic)
        info_dic['imputed_data'] = X_train_filled
        stat, pvalue = Energy().test(X_test, X_train_filled)
        p_values[i] = pvalue
        stats[i] = stat

    return p_values, stats


def run_coverage(X, n, d):
    M = np.random.binomial(n=1, p=P_MISS, size=(n, d))

    info_dic = {
        'data': X,
        'masks': M,
        'imputed_data': None,
        'nbr_it_gibb_sampl': 1,
        'lbd_reg': LBD,
        'tsp': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'sampling': True,
        'intercept': INTERCEPT,
    }

    chain = []
    for it in range(R_TOTAL):
        X_filled = gibb_sampl_sampling(info_dic)
        info_dic['imputed_data'] = X_filled
        if it >= BURN_IN:
            chain.append(X_filled[M == 1].copy())

    chain = np.stack(chain, axis=0)
    true_missing = X[M == 1]
    return chain, true_missing


def coverage_summary(chain, true_missing):
    nominal_levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    empirical = []
    for level in nominal_levels:
        alpha = 1 - level
        lo = np.quantile(chain, alpha / 2, axis=0)
        hi = np.quantile(chain, 1 - alpha / 2, axis=0)
        covered = (true_missing >= lo) & (true_missing <= hi)
        empirical.append(covered.mean())
    empirical = np.array(empirical)
    calib_error = np.mean(np.abs(empirical - nominal_levels))
    return nominal_levels, empirical, calib_error


def autocorrelation_summary(chain, max_lag=100):
    n_sweeps, n_missing = chain.shape
    max_lag = min(max_lag, n_sweeps - 1)

    centered = chain - chain.mean(axis=0, keepdims=True)
    var = np.mean(centered ** 2, axis=0)
    var_safe = np.where(var > 0, var, 1.0)

    acf = np.zeros((max_lag + 1, n_missing))
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        cov_lag = np.mean(centered[:-lag] * centered[lag:], axis=0)
        acf[lag] = cov_lag / var_safe

    ess_per_entry = np.zeros(n_missing)
    for j in range(n_missing):
        rho = acf[1:, j]
        cutoff = np.argmax(rho <= 0) if np.any(rho <= 0) else max_lag
        tau = 1 + 2 * np.sum(rho[:cutoff])
        tau = max(tau, 1.0)
        ess_per_entry[j] = n_sweeps / tau

    return acf.mean(axis=1), ess_per_entry


def main():
    datasets_to_run = [d for d in DATASETS]
    summary_rows = []

    for dataset in datasets_to_run:
        print(f"\n\n==== DATASET: {dataset} ====", flush=True)
        t0 = time.time()
        try:
            X_orig = dataset_loader(dataset)
            n, d = X_orig.shape

            if n > MAX_N:
                print(f"n={n} d={d} -- SKIPPED (n > {MAX_N})", flush=True)
                summary_rows.append({
                    'dataset': dataset,
                    'n': n,
                    'd': d,
                    'gof_p_mean': np.nan,
                    'gof_p_min': np.nan,
                    'gof_p_max': np.nan,
                    'gof_stat_mean': np.nan,
                    'coverage_calib_error': np.nan,
                    'mean_acf_lag1': np.nan,
                    'ess_min': np.nan,
                    'ess_mean': np.nan,
                    'ess_max': np.nan,
                    'elapsed_s': time.time() - t0,
                    'status': f'skipped: n={n} > {MAX_N}',
                })
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(OUT_ROOT / "summary.csv", index=False)
                continue

            mean = np.mean(X_orig, axis=0)
            std = np.std(X_orig, axis=0)
            std_safe = np.where(std > 0, std, 1.0)
            X = (X_orig - mean) / std_safe

            print(f"n={n} d={d}", flush=True)

            p_values, stats = run_goodness_of_fit(X, n, d)
            print(f"gof p-values: {np.round(p_values, 4)}", flush=True)

            chain, true_missing = run_coverage(X, n, d)
            nominal, empirical, calib_error = coverage_summary(chain, true_missing)
            mean_acf, ess_per_entry = autocorrelation_summary(chain)

            dataset_dir = OUT_ROOT / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            np.save(dataset_dir / "gof_p_values.npy", p_values)
            np.save(dataset_dir / "gof_stats.npy", stats)
            np.save(dataset_dir / "coverage_chain.npy", chain)
            np.save(dataset_dir / "coverage_true_missing.npy", true_missing)
            np.save(dataset_dir / "coverage_nominal.npy", nominal)
            np.save(dataset_dir / "coverage_empirical.npy", empirical)
            np.save(dataset_dir / "mean_acf.npy", mean_acf)
            np.save(dataset_dir / "ess_per_entry.npy", ess_per_entry)

            elapsed = time.time() - t0
            print(f"calib_error={calib_error:.4f} mean_ess={ess_per_entry.mean():.1f} "
                  f"elapsed={elapsed:.1f}s", flush=True)

            summary_rows.append({
                'dataset': dataset,
                'n': n,
                'd': d,
                'gof_p_mean': p_values.mean(),
                'gof_p_min': p_values.min(),
                'gof_p_max': p_values.max(),
                'gof_stat_mean': stats.mean(),
                'coverage_calib_error': calib_error,
                'mean_acf_lag1': mean_acf[1] if len(mean_acf) > 1 else np.nan,
                'ess_min': ess_per_entry.min(),
                'ess_mean': ess_per_entry.mean(),
                'ess_max': ess_per_entry.max(),
                'elapsed_s': elapsed,
                'status': 'ok',
            })
        except Exception as e:
            print(f"FAILED on {dataset}: {e}", flush=True)
            traceback.print_exc()
            summary_rows.append({
                'dataset': dataset,
                'n': np.nan,
                'd': np.nan,
                'gof_p_mean': np.nan,
                'gof_p_min': np.nan,
                'gof_p_max': np.nan,
                'gof_stat_mean': np.nan,
                'coverage_calib_error': np.nan,
                'mean_acf_lag1': np.nan,
                'ess_min': np.nan,
                'ess_mean': np.nan,
                'ess_max': np.nan,
                'elapsed_s': time.time() - t0,
                'status': f'failed: {e}',
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_ROOT / "summary.csv", index=False)

    print("\n\n==== DONE ====", flush=True)
    print(pd.DataFrame(summary_rows).to_string(), flush=True)


main()
