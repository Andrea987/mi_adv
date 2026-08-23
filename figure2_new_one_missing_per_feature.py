import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


np.random.seed(54321)


def run_one_missing_per_column(list_n, d, save_folder, lbd=1.0, R=1):
    # for every feature (column) we hide exactly one component (one row picked at random),
    # instead of a random p-fraction, and compare the time of our Gibbs sampler
    # (under-parametrized, sampling) against IterativeImputer(Ridge(fit_intercept=True)),
    # as n grows. We always keep n > d. Raw timings are saved to disk so results can be
    # replotted later without rerunning the experiment.
    print(f"\n\nstarting run_one_missing_per_column, n range {list_n[0]}-{list_n[-1]}, d={d}\n")
    assert all(n > d for n in list_n), "n must always be greater than d"

    total_time_gibb_sampl = np.zeros(len(list_n))
    total_time_ridge = np.zeros(len(list_n))

    for j, n in enumerate(list_n):
        print(f"\n\ncurrent size n = {n}, dim d = {d}")
        mean = np.random.rand(d)
        cov = np.random.rand(d, d)
        cov = cov.T @ cov + np.eye(d) * 0.1
        X = np.random.multivariate_normal(mean, cov, size=n)
        X = X - np.mean(X, axis=0)  # center the columns
        X = X / np.sqrt(n)  # renormalize so X.T @ X stays close to the true covariance, avoiding numerical blow-up

        M = np.zeros((n, d), dtype=int)
        missing_rows = np.random.randint(0, n, size=d)  # one random row picked per column
        M[missing_rows, np.arange(d)] = 1
        print("nbr missing components per column ", np.sum(M, axis=0))

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
        total_time_gibb_sampl[j] = end_gibb - start_gibb
        print(f"Execution time gibb sampl: {total_time_gibb_sampl[j]:.4f} seconds")

        ice = IterativeImputer(estimator=Ridge(fit_intercept=True, alpha=lbd, tol=0.0),
                                imputation_order='roman', max_iter=R,
                                initial_strategy=info_dic['initial_strategy'], verbose=0)
        start_ridge = time.time()
        ice.fit_transform(X_nan)
        end_ridge = time.time()
        total_time_ridge[j] = end_ridge - start_ridge
        print(f"Execution time IterativeImputer(Ridge): {total_time_ridge[j]:.4f} seconds")

    folder = Path(save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "total_time_gibb_sampl.npy", total_time_gibb_sampl)
    np.save(folder / "total_time_ridge.npy", total_time_ridge)
    np.save(folder / "list_n.npy", np.array(list_n))
    np.save(folder / "params.npy", np.array([d, lbd, R]))
    print(f"\nsaved results to {folder}")


def plot_from_folder(save_folder, title):
    folder = Path(save_folder)
    total_time_gibb_sampl = np.load(folder / "total_time_gibb_sampl.npy")
    total_time_ridge = np.load(folder / "total_time_ridge.npy")
    list_n = np.load(folder / "list_n.npy")
    d, lbd, R = np.load(folder / "params.npy")

    plt.plot(list_n, total_time_gibb_sampl, label="our gibb sampl (under-param, sampling)", marker="o", color="blue")
    plt.plot(list_n, total_time_ridge, label="IterativeImputer(Ridge, intercept=True)", marker="*", color="green")
    plt.xlabel("train size n")
    plt.ylabel("time (s)")
    plt.title(f"{title}\n(d={int(d)}, one missing component per column, n>d always)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    list_n = list(range(500, 2001, 50))  # increasing order, always > d, kept well above d for numerical stability
    d = 300
    save_folder = "results/experiment_2_one_missing_per_feature"

    run_one_missing_per_column(list_n, d, save_folder)
    plot_from_folder(save_folder, "Time vs training size")
