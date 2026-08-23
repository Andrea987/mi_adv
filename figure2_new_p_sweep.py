import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


np.random.seed(54321)


def run_mcar_over_p(list_p, n, d, save_folder, nbr_trials=5, lbd=1.0, R=1):
    # n and d are fixed, and we grow the probability p that an entry is missing (MCAR).
    # compares the time of our Gibbs sampler (under-parametrized, sampling) against
    # IterativeImputer(Ridge(fit_intercept=True)), as p grows. We always keep n > d.
    # each point is repeated nbr_trials times; raw per-trial timings are saved to disk
    # so results can be replotted later without rerunning the experiment.
    print(f"\n\nstarting run_mcar_over_p, p range {list_p[0]}-{list_p[-1]}, n={n}, d={d}\n")
    assert d < n, "d must be smaller than n"

    times_gibb_sampl = np.zeros((len(list_p), nbr_trials))
    times_ridge = np.zeros((len(list_p), nbr_trials))

    for j, p in enumerate(list_p):
        print(f"\n\ncurrent p = {p}, n = {n}, d = {d}")
        for trial in range(nbr_trials):
            mean = np.random.rand(d)
            cov = np.random.rand(d, d)
            cov = cov.T @ cov + np.eye(d) * 0.1
            X = np.random.multivariate_normal(mean, cov, size=n)
            X = X - np.mean(X, axis=0)  # center the columns
            X = X / np.sqrt(n)  # renormalize so X.T @ X stays close to the true covariance, avoiding numerical blow-up

            M = np.random.binomial(1, p, size=(n, d))

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
                'recomputation': False,
                'batch_size': 64,
                'verbose': 0,
                'sampling': False,
                'intercept': True
            }

            start_gibb = time.time()
            gibb_sampl_under_parametrized_sampling(info_dic)
            end_gibb = time.time()
            times_gibb_sampl[j, trial] = end_gibb - start_gibb

            ice = IterativeImputer(estimator=Ridge(fit_intercept=True, alpha=lbd, tol=0.0),
                                    imputation_order='roman', max_iter=R,
                                    initial_strategy=info_dic['initial_strategy'], verbose=0)
            start_ridge = time.time()
            ice.fit_transform(X_nan)
            end_ridge = time.time()
            times_ridge[j, trial] = end_ridge - start_ridge

            print(f"  trial {trial + 1}/{nbr_trials}: gibb sampl {times_gibb_sampl[j, trial]:.4f}s, "
                  f"IterativeImputer(Ridge) {times_ridge[j, trial]:.4f}s")

        print(f"p={p}: gibb sampl mean={times_gibb_sampl[j].mean():.4f}s std={times_gibb_sampl[j].std():.4f}s, "
              f"ridge mean={times_ridge[j].mean():.4f}s std={times_ridge[j].std():.4f}s")

    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "times_gibb_sampl.npy", times_gibb_sampl)
    np.save(folder / "times_ridge.npy", times_ridge)
    np.save(folder / "list_p.npy", np.array(list_p))
    np.save(folder / "params.npy", np.array([n, d, lbd, R, nbr_trials]))
    print(f"\nsaved results to {folder}")


def plot_with_band(ax, list_p, mean_gibb, std_gibb, mean_ridge, std_ridge, title):
    ax.plot(list_p, mean_gibb, label="our gibb sampl (under-param, sampling)", marker="o", color="blue")
    ax.fill_between(list_p, mean_gibb - std_gibb, mean_gibb + std_gibb, alpha=0.3, color="blue", label="±1 std")

    ax.plot(list_p, mean_ridge, label="IterativeImputer(Ridge, intercept=True)", marker="*", color="green")
    ax.fill_between(list_p, mean_ridge - std_ridge, mean_ridge + std_ridge, alpha=0.3, color="green", label="±1 std")

    ax.set_xlabel("p (probability an entry is missing)")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    ax.grid()
    ax.legend(loc='upper left')


def plot_from_folder(save_folder, title):
    folder = Path(save_folder)
    times_gibb_sampl = np.load(folder / "times_gibb_sampl.npy")
    times_ridge = np.load(folder / "times_ridge.npy")
    list_p = np.load(folder / "list_p.npy")
    n, d, lbd, R, nbr_trials = np.load(folder / "params.npy")

    mean_gibb, std_gibb = times_gibb_sampl.mean(axis=1), times_gibb_sampl.std(axis=1)
    mean_ridge, std_ridge = times_ridge.mean(axis=1), times_ridge.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_with_band(ax, list_p, mean_gibb, std_gibb, mean_ridge, std_ridge, title)
    fig.suptitle(f"Time vs missingness p (n={int(n)}, d={int(d)}, n>d always, "
                 f"mean +/- std over {int(nbr_trials)} trials)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    list_p_coarse = [round(0.1 * k, 1) for k in range(1, 10)]  # 0.1, 0.2, ..., 0.9
    save_folder = "results/experiment_2_p_sweep_coarse"

    run_mcar_over_p(list_p_coarse, n=2000, d=300, save_folder=save_folder)
    plot_from_folder(save_folder, "coarse range 0.1-0.9")