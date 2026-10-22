import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


SEED = 54321
np.random.seed(SEED)


def run_one_missing_per_column_over_d_alpha(list_d, alpha, save_folder, nbr_trials=5, lbd=1.0, R=1):
    # n is tied to d through n = round(alpha * d), alpha > 1 fixed, so the ratio n/d
    # stays constant as d grows (instead of fixing n and only growing d).
    # for every feature (column) we hide exactly one component (one row picked at random),
    # and compare the time of our Gibbs sampler (under-parametrized, sampling) against
    # IterativeImputer(Ridge(fit_intercept=True)). Raw per-trial timings are saved to disk
    # so results can be replotted later without rerunning the experiment.
    print(f"\n\nstarting run_one_missing_per_column_over_d_alpha, d range {list_d[0]}-{list_d[-1]}, alpha={alpha}\n")
    assert alpha > 1, "alpha must be > 1 so that n > d always"
    list_n = [int(round(alpha * d)) for d in list_d]

    times_gibb_sampl = np.zeros((len(list_d), nbr_trials))
    times_ridge = np.zeros((len(list_d), nbr_trials))

    for j, (d, n) in enumerate(zip(list_d, list_n)):
        print(f"\n\ncurrent dim d = {d}, size n = {n} (alpha={alpha})")
        for trial in range(nbr_trials):
            print(f"  trial {trial + 1}/{nbr_trials}")
            mean = np.random.rand(d)
            cov = np.random.rand(d, d)
            cov = cov.T @ cov + np.eye(d) * 0.1
            X = np.random.multivariate_normal(mean, cov, size=n)
            X = X - np.mean(X, axis=0)  # center the columns
            X = X / np.sqrt(n)  # renormalize so X.T @ X stays close to the true covariance, avoiding numerical blow-up

            M = np.zeros((n, d), dtype=int)
            missing_rows = np.random.randint(0, n, size=d)  # one random row picked per column
            M[missing_rows, np.arange(d)] = 1

            X_nan = X.copy()
            X_nan[M == 1] = np.nan

            info_dic = {
                'data': X,
                'masks': M,
                'imputed_data': None,
                'initial_strategy': 'constant',
                'exponent_d': 0.75,
                'nbr_it_gibb_sampl': R,
                'lbd_reg': lbd,
                'tsp': False,
                'batch_size': 64,
                'verbose': 0,
                'sampling': False,
                'intercept': True
            }

            start_gibb = time.time()
            gibb_sampl_under_parametrized_sampling(info_dic)
            end_gibb = time.time()
            times_gibb_sampl[j, trial] = end_gibb - start_gibb
            print(f"  Execution time gibb sampl: {times_gibb_sampl[j, trial]:.4f} seconds")

            ice = IterativeImputer(estimator=Ridge(fit_intercept=True, alpha=lbd, tol=0.0),
                                    imputation_order='roman', max_iter=R,
                                    initial_strategy=info_dic['initial_strategy'], verbose=0)
            start_ridge = time.time()
            ice.fit_transform(X_nan)
            end_ridge = time.time()
            times_ridge[j, trial] = end_ridge - start_ridge
            print(f"  Execution time IterativeImputer(Ridge): {times_ridge[j, trial]:.4f} seconds")

        print(f"d={d}, n={n}: gibb sampl mean={times_gibb_sampl[j].mean():.4f}s std={times_gibb_sampl[j].std():.4f}s, "
              f"ridge mean={times_ridge[j].mean():.4f}s std={times_ridge[j].std():.4f}s")

    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "times_gibb_sampl.npy", times_gibb_sampl)
    np.save(folder / "times_ridge.npy", times_ridge)
    np.save(folder / "list_d.npy", np.array(list_d))
    np.save(folder / "list_n.npy", np.array(list_n))
    np.save(folder / "params.npy", np.array([alpha, lbd, R, nbr_trials]))

    metadata = {
        "alpha": alpha,
        "lbd_reg": lbd,
        "nbr_it_gibb_sampl": R,
        "nbr_trials": nbr_trials,
        "seed": SEED,
        "list_d": list(list_d),
        "list_n": list_n,
        "mask": "one missing entry per column (row picked uniformly at random per column)",
        "initial_strategy": "constant",
        "exponent_d": 0.75,
        "intercept": True,
        "sampling": False,
        "tsp": False,
        "batch_size": 64,
        "ridge_estimator": "Ridge(fit_intercept=True, tol=0.0)",
        "iterative_imputer_imputation_order": "roman",
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nsaved results and metadata.json to {folder}")


def plot_combined_from_folder(save_folder, title, out_file=None, max_d=None):
    # Figure 2: raw time vs d (left) and log-log with power-law fit (right), side by side
    folder = Path(save_folder)
    times_gibb_sampl = np.load(folder / "times_gibb_sampl.npy")
    times_ridge = np.load(folder / "times_ridge.npy")
    list_d = np.load(folder / "list_d.npy")
    alpha, lbd, R, nbr_trials = np.load(folder / "params.npy")

    if max_d is not None:
        keep = list_d <= max_d
        list_d = list_d[keep]
        times_gibb_sampl = times_gibb_sampl[keep]
        times_ridge = times_ridge[keep]

    mean_gibb = times_gibb_sampl.mean(axis=1)
    mean_ridge = times_ridge.mean(axis=1)
    std_gibb = times_gibb_sampl.std(axis=1)
    std_ridge = times_ridge.std(axis=1)

    log_d = np.log(list_d)
    log_gibb = np.log(mean_gibb)
    log_ridge = np.log(mean_ridge)
    # first-order (delta-method) propagation of std into log-space: d(ln x) ~= std(x) / x
    log_std_gibb = std_gibb / mean_gibb
    log_std_ridge = std_ridge / mean_ridge

    slope_gibb, intercept_gibb = np.polyfit(log_d, log_gibb, 1)
    slope_ridge, intercept_ridge = np.polyfit(log_d, log_ridge, 1)
    print(f"\nfitted slope (power-law exponent) gibb sampl: {slope_gibb:.4f}")
    print(f"fitted slope (power-law exponent) IterativeImputer(Ridge): {slope_ridge:.4f}")

    fig, (ax_top, ax_bottom) = plt.subplots(1, 2, figsize=(16, 6))

    ax_top.plot(list_d, mean_gibb, label="GSI-UD", marker="o", color="blue")
    ax_top.fill_between(list_d, mean_gibb - std_gibb, mean_gibb + std_gibb, alpha=0.3, color="blue")
    ax_top.plot(list_d, mean_ridge, label="ice + ridge", marker="*", color="green")
    ax_top.fill_between(list_d, mean_ridge - std_ridge, mean_ridge + std_ridge, alpha=0.3, color="green")
    ax_top.set_xlabel("d", fontsize=26)
    ax_top.set_ylabel("time (s)", fontsize=26)
    ax_top.tick_params(axis='both', labelsize=22)
    ax_top.grid()

    ax_bottom.plot(log_d, log_gibb, label="GSI-UD", marker="o", color="blue")
    ax_bottom.fill_between(log_d, log_gibb - log_std_gibb, log_gibb + log_std_gibb, alpha=0.3, color="blue")
    ax_bottom.plot(log_d, slope_gibb * log_d + intercept_gibb, color="blue", linestyle="--",
                    label=f"fit exponent GSI-UD: slope={slope_gibb:.2f}")
    ax_bottom.plot(log_d, log_ridge, label="ice + ridge", marker="*", color="green")
    ax_bottom.fill_between(log_d, log_ridge - log_std_ridge, log_ridge + log_std_ridge, alpha=0.3, color="green")
    ax_bottom.plot(log_d, slope_ridge * log_d + intercept_ridge, color="green", linestyle="--",
                    label=f"fit exponent ice + ridge: slope={slope_ridge:.2f}")
    ax_bottom.set_xlabel("log(d)", fontsize=26)
    ax_bottom.set_ylabel("log(time)", fontsize=26)
    ax_bottom.tick_params(axis='both', labelsize=22)
    ax_bottom.grid()

    # one shared legend under both panels; the log-log panel holds all four entries
    fig.legend(*ax_bottom.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=22, frameon=False)
    plt.tight_layout(rect=(0, 0.2, 1, 1))
    if out_file is not None:
        plt.savefig(out_file, dpi=150)
        print(f"saved plot to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    # Figure 2: n = round(alpha * d), d = 100..1200, 10 trials per d
    list_d = list(range(100, 1201, 100))
    alpha = 1.15
    lbd = 10.0
    save_folder = "results/experiment_2_one_missing_per_feature_alpha_sweep_100_1200_10trials"

    # run_one_missing_per_column_over_d_alpha(list_d, alpha, save_folder, nbr_trials=10, lbd=lbd)
    plot_combined_from_folder(save_folder, "",
                              out_file=f"{save_folder}/runtime_vs_dimension_gibbs_vs_ridge_alpha_sweep_100_1200_10trials_GSI-UD.pdf")
