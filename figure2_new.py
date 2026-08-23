import numpy as np
import matplotlib.pyplot as plt
import time
from tsp import gibb_sampl_under_parametrized_sampling
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


np.random.seed(54321)


def time_comparison_mcar_per_column():
    # for every feature (column) we hide a small percentage p of its components (MCAR),
    # and compare the time of our Gibbs sampler (under-parametrized, sampling) against
    # IterativeImputer(Ridge(fit_intercept=True)), as n grows. We always keep n > d.
    print("\n\nstarting time_comparison_mcar_per_column()\n")
    d = 300
    list_n = list(range(500, 1201, 50))  # increasing order, always > d, kept well above d for numerical stability
    assert all(n > d for n in list_n), "n must always be greater than d"
    p = 0.02  # small percentage of missing components per column
    lbd = 1.0
    R = 1

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

        M = np.random.binomial(1, p, size=(n, d))
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

    plt.plot(list_n, total_time_gibb_sampl, label="our gibb sampl (under-param, sampling)", marker="o", color="blue")
    plt.plot(list_n, total_time_ridge, label="IterativeImputer(Ridge, intercept=True)", marker="*", color="green")
    plt.xlabel("train size n")
    plt.ylabel("time (s)")
    plt.title(f"Time vs training size (d={d}, p_miss={p} per column, n>d always)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


time_comparison_mcar_per_column()