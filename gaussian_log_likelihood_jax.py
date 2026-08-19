import numpy as np
import jax
import jax.numpy as jnp

from em_miss import obs_log_lkh


def gaussian_log_likelihood(S, mu, M, X):
    """Observed-data log-likelihood of X under N(mu, S), with M[i,j]=1 if
    X[i,j] is missing and M[i,j]=0 if observed (same convention as
    em_miss.obs_log_lkh). Fully vectorized so it is safe under jax.jit/grad.

    For each row, missing rows/cols of S are replaced by an identity block.
    This leaves the observed-observed block of S untouched, so the resulting
    d x d matrix has the same log-det and quadratic form as the true
    marginal covariance restricted to the observed entries, while keeping a
    fixed shape across rows.
    """
    n, d = X.shape
    obs = 1.0 - M

    S_row = S[None, :, :] * (obs[:, :, None] * obs[:, None, :])
    S_row = S_row + jnp.eye(d)[None, :, :] * (1.0 - obs)[:, :, None]

    X_safe = jnp.where(M > 0, mu[None, :], X)
    r = obs * (X_safe - mu[None, :])

    y = jnp.linalg.solve(S_row, r[..., None])[..., 0]
    quad = jnp.sum(r * y, axis=-1)
    logdet = jnp.linalg.slogdet(S_row)[1]
    n_obs = jnp.sum(obs, axis=-1)

    log_lik_per_row = -0.5 * (n_obs * jnp.log(2.0 * jnp.pi) + logdet + quad)
    return jnp.sum(log_lik_per_row)


gaussian_log_likelihood_grad = jax.jit(
    jax.grad(gaussian_log_likelihood, argnums=(0, 1))
)


def test_matches_obs_log_lkh(n=100, d=10, p_missing=0.3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    S = A @ A.T + d * np.eye(d)
    mu = rng.normal(size=d)
    X = rng.multivariate_normal(mu, S, size=n)
    M = (rng.uniform(size=(n, d)) < p_missing).astype(float)

    numpy_ll = obs_log_lkh(S, mu, M, X)
    jax_ll = float(gaussian_log_likelihood(jnp.array(S), jnp.array(mu), jnp.array(M), jnp.array(X)))

    assert np.isclose(numpy_ll, jax_ll), (numpy_ll, jax_ll)
    print("numpy obs_log_lkh:", numpy_ll)
    print("jax gaussian_log_likelihood:", jax_ll)
    print("match: OK")


if __name__ == "__main__":
    test_matches_obs_log_lkh()
