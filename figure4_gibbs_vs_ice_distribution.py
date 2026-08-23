import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from hyppo.ksample import Energy

from tsp import gibb_sampl_sampling
from dataset_load import dataset_loader

DATASETS = ['wine_quality_red', 'ecoli']
P_MISS_LIST = [0.25, 0.5, 0.7]
LBD = 0.001
R_LIST = [500, 1000, 1500, 2000]
R_MAX = max(R_LIST)
NUM_SEEDS = 5
BASE_SEED = 54321
INTERCEPT = True

OUT_DIR = Path("results/experiment4_gibbs_vs_ice_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

energy = Energy()


def make_mask(rng, n, d, p_miss):
    # MCAR mask; a fully-missing row leaves nothing to condition on, so if one
    # occurs (rare unless p_miss is high), re-mark one random entry as seen
    M = rng.binomial(n=1, p=p_miss, size=(n, d))
    full_rows = np.where(M.sum(axis=1) == d)[0]
    if len(full_rows) > 0:
        cols = rng.integers(0, d, size=len(full_rows))
        M[full_rows, cols] = 0
    return M, len(full_rows)


def run_gibbs_history(X, M):
    info_dic = {
        'data': X,
        'masks': M,
        'imputed_data': None,
        'nbr_it_gibb_sampl': R_MAX,
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
    res = gibb_sampl_sampling(info_dic)
    return res['list_imputed']  # one imputed dataset per sweep, length R_MAX


def run_ice(X, M, max_iter, seed):
    X_nan = X.copy()
    X_nan[M == 1] = np.nan
    ice = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=True,
        max_iter=max_iter,
        initial_strategy='constant',
        random_state=seed,
    )
    return ice.fit_transform(X_nan)


def main():
    total_ice_fits = len(DATASETS) * len(P_MISS_LIST) * NUM_SEEDS * len(R_LIST)
    print(f"planned runs: {len(DATASETS)} datasets x {len(P_MISS_LIST)} p_miss levels x "
          f"{NUM_SEEDS} seeds x {len(R_LIST)} R-checkpoints = {total_ice_fits} ICE fits "
          f"(+ {len(DATASETS) * len(P_MISS_LIST) * NUM_SEEDS} gibbs chains)", flush=True)

    rows = []
    for dataset in DATASETS:
        print(f"\n==== {dataset} ====", flush=True)
        X_orig = dataset_loader(dataset)
        n, d = X_orig.shape
        mean = np.mean(X_orig, axis=0)
        std = np.std(X_orig, axis=0)
        std_safe = np.where(std > 0, std, 1.0)
        X = (X_orig - mean) / std_safe

        for p_miss in P_MISS_LIST:
            print(f"  -- p_miss={p_miss} --", flush=True)
            for seed in range(NUM_SEEDS):
                print(f"    seed {seed + 1}/{NUM_SEEDS}", flush=True)
                rng = np.random.default_rng(BASE_SEED + seed)
                M, n_fixed = make_mask(rng, n, d, p_miss)
                if n_fixed > 0:
                    print(f"      fixed {n_fixed} fully-missing row(s)", flush=True)

                t0 = time.time()
                history = run_gibbs_history(X, M)
                print(f"      gibbs: {R_MAX} sweeps done in {time.time() - t0:.2f}s", flush=True)

                for R in R_LIST:
                    X_gibbs = history[R - 1]

                    t1 = time.time()
                    X_ice = run_ice(X, M, max_iter=R, seed=seed)
                    ice_time = time.time() - t1

                    stat, pvalue = energy.test(X_gibbs, X_ice)
                    print(f"      R={R:4d}: ED={stat:.5f}  p={pvalue:.4f}  "
                          f"(ice fit {ice_time:.2f}s)", flush=True)

                    rows.append(dict(dataset=dataset, p_miss=p_miss, seed=seed, R=R, n=n, d=d,
                                      n_fixed_rows=n_fixed, stat=stat, pvalue=pvalue))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_seed.csv", index=False)

    print("\n\n==== SUMMARY (mean/std over seeds) ====", flush=True)
    summary = df.groupby(['dataset', 'p_miss', 'R']).agg(
        stat_mean=('stat', 'mean'), stat_std=('stat', 'std'),
        pvalue_mean=('pvalue', 'mean'), pvalue_min=('pvalue', 'min'),
        frac_p_below_05=('pvalue', lambda p: (p < 0.05).mean()),
    )
    print(summary, flush=True)
    summary.to_csv(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_summary.csv")

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(7 * len(DATASETS), 9), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_MISS_LIST)))

    for col, dataset in enumerate(DATASETS):
        ax_ed, ax_p = axes[0][col], axes[1][col]
        for color, p_miss in zip(colors, P_MISS_LIST):
            sub = summary.loc[(dataset, p_miss)]
            ax_ed.errorbar(sub.index, sub['stat_mean'], yerr=sub['stat_std'],
                            marker='o', capsize=3, color=color, label=f"p_miss={p_miss}")
            ax_p.plot(sub.index, sub['pvalue_mean'], marker='o', color=color,
                       label=f"p_miss={p_miss}")

        ax_ed.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_ed.set_xlabel("R (gibbs sweeps == ICE max_iter)")
        ax_ed.set_ylabel("energy distance (mean +/- std over seeds)")
        ax_ed.set_title(f"{dataset}: energy distance")
        ax_ed.legend()
        ax_ed.grid(alpha=0.3)

        ax_p.axhline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
        ax_p.set_xlabel("R (gibbs sweeps == ICE max_iter)")
        ax_p.set_ylabel("p-value (mean over seeds)")
        ax_p.set_title(f"{dataset}: p-value")
        ax_p.set_ylim(0, 1)
        ax_p.legend()
        ax_p.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "gibbs_vs_ice_by_R_pmiss.pdf")
    print(f"\nsaved plot to {OUT_DIR / 'gibbs_vs_ice_by_R_pmiss.pdf'}", flush=True)


if __name__ == '__main__':
    main()
