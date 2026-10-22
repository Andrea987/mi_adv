import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from hyppo.ksample import Energy

from tsp import gibb_sampl_sampling
from dataset_load import dataset_loader

DATASETS = ['wine_quality_red', 'ecoli']
P_MISS_LIST = [0.30, 0.40, 0.50, 0.60, 0.70]
LBD = 0.001
R_LIST = [0, 250, 500, 750, 1000]
R_MAX = max(R_LIST)
NUM_SEEDS = 5
BASE_SEED = 54321
INTERCEPT = True

OUT_DIR = Path("results/experiment4_gibbs_vs_ice_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMPUTED_DIR = OUT_DIR / "imputed"
IMPUTED_DIR.mkdir(parents=True, exist_ok=True)

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

    metadata = {
        "datasets": DATASETS,
        "p_miss_list": P_MISS_LIST,
        "R_list": R_LIST,
        "num_seeds": NUM_SEEDS,
        "base_seed": BASE_SEED,
        "lbd_reg": LBD,
        "intercept": INTERCEPT,
        "gibbs_sampling": True,
        "gibbs_initial_strategy": "constant",
        "gibbs_exponent_d": 0.75,
        "ice_estimator": "BayesianRidge",
        "ice_sample_posterior": True,
        "ice_initial_strategy": "constant",
        "zero_baseline_estimator": "SimpleImputer(strategy='constant')",
        "energy_test": "hyppo.ksample.Energy (bias=False, default)",
        "mask_type": "MCAR, binomial(1, p_miss) per cell; fully-missing rows "
                      "re-marked with one random seen entry",
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

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

                # gibb_sampl_sampling draws from numpy's global RNG (np.random.randn),
                # not from the local `rng` Generator above -- seed it here so the
                # sampler's stochastic draws are reproducible too, without touching tsp.py
                np.random.seed(BASE_SEED + seed)
                t0 = time.time()
                history = run_gibbs_history(X, M)
                print(f"      gibbs: {R_MAX} sweeps done in {time.time() - t0:.2f}s", flush=True)

                # trivial baseline: constant (=0) imputation via SimpleImputer, same
                # convention as initial_strategy='constant' used to seed gibbs/ICE.
                # Data is standardized, so 0 = per-column mean; this is a sanity-check
                # positive control -- a real imputer should be clearly distinguishable
                # from this, since constant imputation destroys all covariance structure.
                X_nan_zero = X.copy()
                X_nan_zero[M == 1] = np.nan
                X_zero = SimpleImputer(missing_values=np.nan, strategy='constant').fit_transform(X_nan_zero)

                ice_at_R = {}
                for R in R_LIST:
                    # R=0 means "no gibbs sweeps at all" -- gibb_sampl_sampling's own
                    # internal initialization is SimpleImputer(strategy='constant'),
                    # identical to X_zero (verified to match ICE(max_iter=0) too), so
                    # use it directly rather than history[R-1] which would wrap to the
                    # last sweep for R=0
                    X_gibbs = X_zero if R == 0 else history[R - 1]

                    t1 = time.time()
                    X_ice = run_ice(X, M, max_iter=R, seed=seed)
                    ice_time = time.time() - t1
                    ice_at_R[f"ice_R{R}"] = X_ice

                    stat, pvalue = energy.test(X_gibbs, X_ice)
                    stat_zero, pvalue_zero = energy.test(X_gibbs, X_zero)
                    print(f"      R={R:4d}: gibbs-vs-ICE ED={stat:.5f} p={pvalue:.4f}  |  "
                          f"gibbs-vs-zero ED={stat_zero:.5f} p={pvalue_zero:.4g}  "
                          f"(ice fit {ice_time:.2f}s)", flush=True)

                    rows.append(dict(dataset=dataset, p_miss=p_miss, seed=seed, R=R, n=n, d=d,
                                      n_fixed_rows=n_fixed, stat=stat, pvalue=pvalue,
                                      stat_zero=stat_zero, pvalue_zero=pvalue_zero))

                # persist every imputed dataset for this (dataset, p_miss, seed): our
                # method at each R checkpoint, ICE at each R checkpoint, the zero
                # baseline, and the mask -- so this experiment never needs a full
                # rerun (including the expensive ICE fits) to re-inspect or re-test
                gibbs_at_R = {f"gibbs_R{R}": (X_zero if R == 0 else history[R - 1]) for R in R_LIST}
                imputed_path = IMPUTED_DIR / f"{dataset}_pmiss{p_miss}_seed{seed}.npz"
                np.savez(imputed_path, mask=M, zero=X_zero, **gibbs_at_R, **ice_at_R)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_seed.csv", index=False)

    print("\n\n==== SUMMARY (mean/std over seeds) ====", flush=True)
    summary = df.groupby(['dataset', 'p_miss', 'R']).agg(
        stat_mean=('stat', 'mean'), stat_std=('stat', 'std'),
        pvalue_mean=('pvalue', 'mean'), pvalue_std=('pvalue', 'std'),
        pvalue_min=('pvalue', 'min'),
        frac_p_below_05=('pvalue', lambda p: (p < 0.05).mean()),
        stat_zero_mean=('stat_zero', 'mean'), stat_zero_std=('stat_zero', 'std'),
        pvalue_zero_mean=('pvalue_zero', 'mean'), pvalue_zero_max=('pvalue_zero', 'max'),
        frac_zero_below_05=('pvalue_zero', lambda p: (p < 0.05).mean()),
    )
    print(summary, flush=True)
    summary.to_csv(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_summary.csv")


PLOT_P_MISS = [0.3, 0.5, 0.7]  # p_miss levels shown in Figure 4


def plot_fig_4():
    # Figure 4 from the per-seed results saved by main(), without rerunning the experiment
    df = pd.read_csv(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_seed.csv")
    summary = df.groupby(['dataset', 'p_miss', 'R']).agg(
        stat_mean=('stat', 'mean'), stat_std=('stat', 'std'),
        pvalue_mean=('pvalue', 'mean'), pvalue_std=('pvalue', 'std'),
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(PLOT_P_MISS)))

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(6 * len(DATASETS), 11), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        ax_ed, ax_p = axes[0][col], axes[1][col]
        for color, p_miss in zip(colors, PLOT_P_MISS):
            sub = summary.loc[(dataset, p_miss)]

            ax_ed.plot(sub.index, sub['stat_mean'], marker='o', color=color, label=f"p_miss={p_miss}")
            ax_ed.fill_between(sub.index, sub['stat_mean'] - sub['stat_std'],
                                sub['stat_mean'] + sub['stat_std'], alpha=0.3, color=color)

            ax_p.plot(sub.index, sub['pvalue_mean'], marker='o', color=color, label=f"p_miss={p_miss}")
            pvalue_lo = (sub['pvalue_mean'] - sub['pvalue_std']).clip(lower=0)
            pvalue_hi = (sub['pvalue_mean'] + sub['pvalue_std']).clip(upper=1)
            ax_p.fill_between(sub.index, pvalue_lo, pvalue_hi, alpha=0.3, color=color)

        ax_ed.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_ed.set_ylabel("mean energy distance $\\pm$ 1 std", fontsize=18)
        ax_ed.set_title(f"{dataset}: energy distance", fontsize=22)
        ax_ed.tick_params(axis='both', labelsize=18)
        ax_ed.grid(alpha=0.3)

        ax_p.axhline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
        ax_p.set_ylabel("mean p-value $\\pm$ 1 std", fontsize=18)
        ax_p.set_ylim(-0.03, 1.05)
        ax_p.set_title(f"{dataset}: p-value", fontsize=22)
        ax_p.tick_params(axis='both', labelsize=18)
        ax_p.grid(alpha=0.3)

    # one shared x label and one shared legend under the four panels (p_miss levels + the p=0.05 line);
    # the x label is too long to fit under each panel at this font size
    fig.supxlabel("R (GSI-UD sweeps == ice + BR max_iter)", fontsize=20, y=0.055)
    fig.legend(*axes[1][0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=20, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT_DIR / "gibbs_vs_ice_by_R_pmiss_0.3_0.5_0.7.pdf")
    print(f"saved {OUT_DIR / 'gibbs_vs_ice_by_R_pmiss_0.3_0.5_0.7.pdf'}")


if __name__ == '__main__':
    # main()
    plot_fig_4()
