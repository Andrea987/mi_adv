import numpy as np
import jax.numpy as jnp

from tsp import gibb_sampl_fast_sampling, online_gibbs_sampling
from gaussian_log_likelihood_jax import gaussian_log_likelihood, gaussian_log_likelihood_grad


def test_gradient_norm_decreases_along_gibbs_iterates():
    rng = np.random.default_rng(0)
    n, d = 200, 10
    lbd = 0.0005

    A = rng.normal(size=(d, d))
    cov_true = A @ A.T + (d ** 2) * np.eye(d)
    mean_true = rng.normal(size=d)
    X = rng.multivariate_normal(mean_true, cov_true, size=n)

    M = rng.binomial(1, 0.3, size=(n, d))
    for j in range(d):
        if M[:, j].sum() == n:
            M[rng.integers(n), j] = 0

    R = 300
    info_dic = {
        'data': X,
        'imputed_data': None,
        'masks': M,
        'gamma': None,
        'save_all_iterations': True,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'sampling': True,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0,
    }

    res = gibb_sampl_fast_sampling(info_dic)

    X_j = jnp.array(X)
    M_j = jnp.array(M, dtype=jnp.float32)

    log_liks = []
    grad_norms = []
    for mean_it, cov_it in zip(res['list_mean'], res['list_cov']):
        S_j = jnp.array(cov_it)
        mu_j = jnp.array(mean_it)

        ll = gaussian_log_likelihood(S_j, mu_j, M_j, X_j)
        dS, dmu = gaussian_log_likelihood_grad(S_j, mu_j, M_j, X_j)
        grad_norm = float(jnp.sqrt(jnp.sum(dS ** 2) + jnp.sum(dmu ** 2)))

        log_liks.append(float(ll))
        grad_norms.append(grad_norm)

    print("log-likelihood per iterate:")
    for it, ll in enumerate(log_liks):
        print(f"  it {it}: {ll}")

    print("gradient magnitude per iterate:")
    for it, g in enumerate(grad_norms):
        print(f"  it {it}: {g}")

    k = max(1, R // 3)
    avg_first = np.mean(grad_norms[:k])
    avg_last = np.mean(grad_norms[-k:])
    print("avg grad norm, first third:", avg_first)
    print("avg grad norm, last third:", avg_last)

    assert avg_last < avg_first, (avg_first, avg_last)
    print("gradient magnitude decreased over iterations: OK")


def test_online_gibbs_decreasing_gamma_log_likelihood():
    rng = np.random.default_rng(0)
    n, d = 200, 10
    lbd = 0.0005

    A = rng.normal(size=(d, d))
    cov_true = A @ A.T + (d ** 2) * np.eye(d)
    mean_true = rng.normal(size=d)
    X = rng.multivariate_normal(mean_true, cov_true, size=n)

    M = rng.binomial(1, 0.3, size=(n, d))
    for j in range(d):
        if M[:, j].sum() == n:
            M[rng.integers(n), j] = 0

    R = 300
    gamma = 1.0 / np.arange(1, R * d + 1)  # gamma[h] = 1/h, h = 1, 2, ...

    info_dic = {
        'data': X,
        'imputed_data': None,
        'masks': M,
        'gamma': gamma,
        'save_all_iterations': True,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'initial_strategy': 'constant',
        'sampling': True,
        'intercept': True,
        'batch_size': 64,
        'verbose': 0,
    }

    res = online_gibbs_sampling(info_dic)

    X_j = jnp.array(X)
    M_j = jnp.array(M, dtype=jnp.float32)

    log_liks = []
    grad_norms = []
    for mean_it, cov_it in zip(res['list_mean'], res['list_cov']):
        S_j = jnp.array(cov_it)
        mu_j = jnp.array(mean_it)

        ll = gaussian_log_likelihood(S_j, mu_j, M_j, X_j)
        dS, dmu = gaussian_log_likelihood_grad(S_j, mu_j, M_j, X_j)
        grad_norm = float(jnp.sqrt(jnp.sum(dS ** 2) + jnp.sum(dmu ** 2)))

        log_liks.append(float(ll))
        grad_norms.append(grad_norm)

    print("log-likelihood per iterate:")
    for it, ll in enumerate(log_liks):
        print(f"  it {it}: {ll}")

    print("gradient magnitude per iterate:")
    for it, g in enumerate(grad_norms):
        print(f"  it {it}: {g}")


if __name__ == "__main__":
    # test_gradient_norm_decreases_along_gibbs_iterates()
    test_online_gibbs_decreasing_gamma_log_likelihood()
