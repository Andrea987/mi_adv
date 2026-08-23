import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from hyppo.ksample import Energy

from tsp import gibb_sampl_sampling
from dataset_load import dataset_loader

np.random.seed(54321)

DATASETS = ['wine_quality_red', 'ecoli']
P_MISS = 0.25
LBD = 0.001
R = 10          # nbr sweeps for our sampler == max_iter for sklearn ICE, same budget
REP = 5         # independent repetitions (fresh mask + split + seed) per dataset
INTERCEPT = True

OUT_DIR = Path("results/experiment4_ed_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_gibbs(X_train, M_train, X_test):
    info_dic = {
        'data': X_train,
        'masks': M_train,
        'imputed_data': None,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': LBD,
        'tsp': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'sampling': True,
        'intercept': INTERCEPT,
        'save_all_iterations': True,
    }
    t0 = time.time()
    res = gibb_sampl_sampling(info_dic)
    elapsed = time.time() - t0
    history = res['list_imputed']  # imputed dataset after each of the R sweeps
    stat, pvalue = Energy().test(X_test, history[-1])
    return stat, pvalue, elapsed, history


def run_sklearn_ice(X_train, M_train, X_test, seed):
    X_nan = X_train.copy()
    X_nan[M_train == 1] = np.nan
    ice = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=True,
        max_iter=R,
        initial_strategy='constant',
        random_state=seed,
    )
    t0 = time.time()
    X_filled = ice.fit_transform(X_nan)
    elapsed = time.time() - t0
    stat, pvalue = Energy().test(X_test, X_filled)
    return stat, pvalue, elapsed


def main():
    rows = []
    for dataset in DATASETS:
        print(f"\n==== {dataset} ====", flush=True)
        X_orig = dataset_loader(dataset)
        n, d = X_orig.shape
        mean = np.mean(X_orig, axis=0)
        std = np.std(X_orig, axis=0)
        std_safe = np.where(std > 0, std, 1.0)
        X = (X_orig - mean) / std_safe

        for rep in range(REP):
            M = np.random.binomial(n=1, p=P_MISS, size=(n, d))
            X_train, X_test, M_train, M_test = train_test_split(X, M, test_size=0.30)

            gibbs_stat, gibbs_pval, gibbs_time, gibbs_history = run_gibbs(X_train, M_train, X_test)
            ice_stat, ice_pval, ice_time = run_sklearn_ice(X_train, M_train, X_test, seed=rep)

            lower = 'gibbs' if gibbs_stat < ice_stat else 'ice'
            print(f"  rep {rep}: gibbs ED={gibbs_stat:.4f} (p={gibbs_pval:.3f}, {gibbs_time:.2f}s)  "
                  f"ice ED={ice_stat:.4f} (p={ice_pval:.3f}, {ice_time:.2f}s)  lower={lower}", flush=True)

            rows.append(dict(dataset=dataset, rep=rep, n=n, d=d,
                              gibbs_ed=gibbs_stat, gibbs_pval=gibbs_pval, gibbs_time=gibbs_time,
                              ice_ed=ice_stat, ice_pval=ice_pval, ice_time=ice_time, lower=lower))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ed_comparison.csv", index=False)

    print("\n\n==== SUMMARY (mean over reps) ====")
    summary = df.groupby('dataset')[['gibbs_ed', 'ice_ed', 'gibbs_time', 'ice_time']].mean()
    print(summary)
    print("\nlower ED count per dataset:")
    print(df.groupby('dataset')['lower'].value_counts())


if __name__ == '__main__':
    main()
