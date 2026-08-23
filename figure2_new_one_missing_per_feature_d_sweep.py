import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


np.random.seed(54321)


def run_one_missing_per_column_over_d(list_d, n, save_folder, nbr_trials=5, lbd=1.0, R=1):
    # dataset size n is fixed, and we grow the number of features d.
    # for every feature (column) we hide exactly one component (one row picked at random),
    # instead of a random p-fraction, and compare the time of our Gibbs sampler
    # (under-parametrized, sampling) against IterativeImputer(Ridge(fit_intercept=True)),
    # as d grows. We always keep n > d. Raw per-trial timings are saved to disk so results
    # can be replotted later without rerunning the experiment.
    print(f"\n\nstarting run_one_missing_per_column_over_d, d range {list_d[0]}-{list_d[-1]}, n={n}\n")
    assert all(d < n for d in list_d), "d must always be smaller than n"

    times_gibb_sampl = np.zeros((len(list_d), nbr_trials))
    times_ridge = np.zeros((len(list_d), nbr_trials))

    for j, d in enumerate(list_d):
        print(f"\n\ncurrent dim d = {d}, size n = {n}")
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
            print(f"  Execution time gibb sampl: {times_gibb_sampl[j, trial]:.4f} seconds")

            ice = IterativeImputer(estimator=Ridge(fit_intercept=True, alpha=lbd, tol=0.0),
                                    imputation_order='roman', max_iter=R,
                                    initial_strategy=info_dic['initial_strategy'], verbose=0)
            start_ridge = time.time()
            ice.fit_transform(X_nan)
            end_ridge = time.time()
            times_ridge[j, trial] = end_ridge - start_ridge
            print(f"  Execution time IterativeImputer(Ridge): {times_ridge[j, trial]:.4f} seconds")

        print(f"d={d}: gibb sampl mean={times_gibb_sampl[j].mean():.4f}s std={times_gibb_sampl[j].std():.4f}s, "
              f"ridge mean={times_ridge[j].mean():.4f}s std={times_ridge[j].std():.4f}s")

    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "times_gibb_sampl.npy", times_gibb_sampl)
    np.save(folder / "times_ridge.npy", times_ridge)
    np.save(folder / "list_d.npy", np.array(list_d))
    np.save(folder / "params.npy", np.array([n, lbd, R, nbr_trials]))
    print(f"\nsaved results to {folder}")


def plot_from_folder(save_folder, title):
    folder = Path(save_folder)
    times_gibb_sampl = np.load(folder / "times_gibb_sampl.npy")
    times_ridge = np.load(folder / "times_ridge.npy")
    list_d = np.load(folder / "list_d.npy")
    n, lbd, R, nbr_trials = np.load(folder / "params.npy")

    mean_gibb = times_gibb_sampl.mean(axis=1)
    mean_ridge = times_ridge.mean(axis=1)

    log_d = np.log(list_d)
    log_gibb = np.log(mean_gibb)
    log_ridge = np.log(mean_ridge)

    slope_gibb, intercept_gibb = np.polyfit(log_d, log_gibb, 1)
    slope_ridge, intercept_ridge = np.polyfit(log_d, log_ridge, 1)
    print(f"\nfitted slope (power-law exponent) gibb sampl: {slope_gibb:.4f}")
    print(f"fitted slope (power-law exponent) IterativeImputer(Ridge): {slope_ridge:.4f}")

    plt.scatter(log_d, log_gibb, label="our gibb sampl (under-param, sampling)", marker="o", color="blue")
    plt.plot(log_d, slope_gibb * log_d + intercept_gibb, color="blue", linestyle="--",
              label=f"fit gibb: slope={slope_gibb:.2f}")
    plt.scatter(log_d, log_ridge, label="IterativeImputer(Ridge, intercept=True)", marker="*", color="green")
    plt.plot(log_d, slope_ridge * log_d + intercept_ridge, color="green", linestyle="--",
              label=f"fit ridge: slope={slope_ridge:.2f}")
    plt.xlabel("log(d)")
    plt.ylabel("log(time)")
    plt.title(f"{title}\nlog(time) vs log(d), averaged over {int(nbr_trials)} trials "
              f"(n={int(n)}, one missing component per column, n>d always)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    list_d = list(range(100, 501, 25))  # increasing order, always < n
    n = 2000
    save_folder = "results/experiment_2_one_missing_per_feature_d_sweep"

    run_one_missing_per_column_over_d(list_d, n, save_folder)
    plot_from_folder(save_folder, "Time vs feature dimension")
