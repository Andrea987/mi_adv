import numpy as np
import matplotlib.pyplot as plt
import time
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


np.random.seed(54321)


def time_comparison_mcar_fixed_n():
    # dataset size n is fixed, and we grow the number of features d.
    # for every feature (column) we hide a small percentage p of its components (MCAR),
    # and compare the time of our Gibbs sampler (under-parametrized, sampling) against
    # IterativeImputer(Ridge(fit_intercept=True)), as d grows. We always keep n > d.
    print("\n\nstarting time_comparison_mcar_fixed_n()\n")
    n = 800
    list_d = list(range(100, 601, 25))  # increasing order, always < n
    assert all(d < n for d in list_d), "d must always be smaller than n"
    p = 0.01  # small percentage of missing components per column
    lbd = 1.0
    R = 1
    nbr_trials = 5

    total_time_gibb_sampl = np.zeros(len(list_d))
    total_time_ridge = np.zeros(len(list_d))

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
            total_time_gibb_sampl[j] += end_gibb - start_gibb
            print(f"  Execution time gibb sampl: {end_gibb - start_gibb:.4f} seconds")

            ice = IterativeImputer(estimator=Ridge(fit_intercept=True, alpha=lbd, tol=0.0),
                                    imputation_order='roman', max_iter=R,
                                    initial_strategy=info_dic['initial_strategy'], verbose=0)
            start_ridge = time.time()
            ice.fit_transform(X_nan)
            end_ridge = time.time()
            total_time_ridge[j] += end_ridge - start_ridge
            print(f"  Execution time IterativeImputer(Ridge): {end_ridge - start_ridge:.4f} seconds")

        total_time_gibb_sampl[j] /= nbr_trials
        total_time_ridge[j] /= nbr_trials
        print(f"average over {nbr_trials} trials, gibb sampl: {total_time_gibb_sampl[j]:.4f} seconds")
        print(f"average over {nbr_trials} trials, IterativeImputer(Ridge): {total_time_ridge[j]:.4f} seconds")

    log_d = np.log(list_d)
    log_gibb = np.log(total_time_gibb_sampl)
    log_ridge = np.log(total_time_ridge)

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
    plt.title(f"log(time) vs log(d), averaged over {nbr_trials} trials (n={n}, p_miss={p} per column, n>d always)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


time_comparison_mcar_fixed_n()