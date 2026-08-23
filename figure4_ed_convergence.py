import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
R = 500          # gibbs sweeps, and the top checkpoint for sklearn ICE's max_iter
INTERCEPT = True
CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]

OUT_DIR = Path("results/experiment4_ed_convergence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

energy = Energy()


def run_gibbs_trajectory(X_train, M_train, X_test):
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
    history = res['list_imputed']  # one imputed dataset per sweep, length R
    ed = np.array([energy.statistic(X_test, X_h) for X_h in history])
    return ed, elapsed


def run_ice_checkpoints(X_train, M_train, X_test):
    # IterativeImputer has no per-iteration history, so refit from scratch at each
    # checkpoint max_iter to get a comparable trajectory
    X_nan = X_train.copy()
    X_nan[M_train == 1] = np.nan
    eds = []
    t0 = time.time()
    for it in CHECKPOINTS:
        ice = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=it,
            initial_strategy='constant',
            random_state=0,
        )
        X_filled = ice.fit_transform(X_nan)
        eds.append(energy.statistic(X_test, X_filled))
    elapsed = time.time() - t0
    return np.array(eds), elapsed


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5))
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASETS):
        print(f"\n==== {dataset} ====", flush=True)
        X_orig = dataset_loader(dataset)
        n, d = X_orig.shape
        mean = np.mean(X_orig, axis=0)
        std = np.std(X_orig, axis=0)
        std_safe = np.where(std > 0, std, 1.0)
        X = (X_orig - mean) / std_safe

        M = np.random.binomial(n=1, p=P_MISS, size=(n, d))
        X_train, X_test, M_train, M_test = train_test_split(X, M, test_size=0.30)

        gibbs_ed, gibbs_time = run_gibbs_trajectory(X_train, M_train, X_test)
        print(f"  gibbs: {R} sweeps in {gibbs_time:.2f}s", flush=True)

        ice_ed, ice_time = run_ice_checkpoints(X_train, M_train, X_test)
        print(f"  ice: {len(CHECKPOINTS)} checkpoints in {ice_time:.2f}s", flush=True)

        np.save(OUT_DIR / f"{dataset}_gibbs_ed.npy", gibbs_ed)
        np.save(OUT_DIR / f"{dataset}_ice_ed.npy", ice_ed)
        np.save(OUT_DIR / f"{dataset}_ice_checkpoints.npy", np.array(CHECKPOINTS))

        ax.plot(np.arange(1, R + 1), gibbs_ed, label="gibbs (fast)", color="tab:blue")
        ax.plot(CHECKPOINTS, ice_ed, label="ICE (BayesianRidge)", color="tab:orange", marker="o")
        ax.set_xlabel("iteration / sweep")
        ax.set_ylabel("energy distance")
        ax.set_title(f"{dataset} (n={n}, d={d})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ed_convergence.pdf")
    print(f"\nsaved plot to {OUT_DIR / 'ed_convergence.pdf'}")


if __name__ == '__main__':
    main()
