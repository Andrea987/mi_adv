import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path

from tsp import gibb_sampl_fast_sampling, online_gibbs_sampling
from gaussian_log_likelihood_jax import gaussian_log_likelihood, gaussian_log_likelihood_grad


np.random.seed(54321)

METHODS = [
    'fast sampling (sequential)',
    'fast sampling (random)',
    'online gibbs (sequential)',
    'online gibbs (random)',
]

STYLES = {
    'fast sampling (sequential)': dict(color='blue', linestyle='-'),
    'fast sampling (random)': dict(color='blue', linestyle='--'),
    'online gibbs (sequential)': dict(color='green', linestyle='-'),
    'online gibbs (random)': dict(color='green', linestyle='--'),
}

LABELS = {
    'fast sampling (sequential)': 'online multiple imputation (sequential)',
    'fast sampling (random)': 'online multiple imputation (random)',
    'online gibbs (sequential)': r'online gibbs (sequential, $\gamma_k=1/k$)',
    'online gibbs (random)': r'online gibbs (random, $\gamma_k=1/k$)',
}


def _neg_log_lik_and_grad_norm_per_iterate(res, M_j, X_j, method_label, print_every=10):
    neg_log_liks = []
    grad_norms = []
    for it, (mean_it, cov_it) in enumerate(zip(res['list_mean'], res['list_cov'])):
        S_j = jnp.array(cov_it)
        mu_j = jnp.array(mean_it)

        ll = gaussian_log_likelihood(S_j, mu_j, M_j, X_j)
        dS, dmu = gaussian_log_likelihood_grad(S_j, mu_j, M_j, X_j)
        grad_norm = float(jnp.sqrt(jnp.sum(dS ** 2) + jnp.sum(dmu ** 2)))

        neg_log_liks.append(-float(ll))
        grad_norms.append(grad_norm)

        if it % print_every == 0:
            print(f"    [{method_label}] it {it}: -log-lik = {-float(ll):.4f}, grad norm = {grad_norm:.4f}")
    return np.array(neg_log_liks), np.array(grad_norms)


def _run_once(seed, n, d, lbd, R, p_mcar):
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(d, d))
    cov_true = A @ A.T + (d ** 2) * np.eye(d)
    mean_true = rng.normal(size=d)
    X = rng.multivariate_normal(mean_true, cov_true, size=n)

    M = rng.binomial(1, p_mcar, size=(n, d))
    for j in range(d):
        if M[:, j].sum() == n:
            M[rng.integers(n), j] = 0

    X_j = jnp.array(X)
    M_j = jnp.array(M, dtype=jnp.float32)

    gamma = 1.0 / np.arange(1, R * d + 1)  # gamma[h] = 1/h, h = 1, 2, ...

    base_info = {
        'data': X,
        'imputed_data': None,
        'masks': M,
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

    run_results = {}

    for feature_order in ['sequential', 'random']:
        label = f'fast sampling ({feature_order})'
        print(f"\n>>> checking gibb_sampl_fast_sampling (feature_order='{feature_order}')")
        info_dic = {**base_info, 'feature_order': feature_order}
        res = gibb_sampl_fast_sampling(info_dic)
        run_results[label] = _neg_log_lik_and_grad_norm_per_iterate(res, M_j, X_j, label)

    for feature_order in ['sequential', 'random']:
        label = f'online gibbs ({feature_order})'
        print(f"\n>>> checking online_gibbs_sampling (feature_order='{feature_order}')")
        info_dic = {**base_info, 'feature_order': feature_order, 'gamma': gamma}
        res = online_gibbs_sampling(info_dic)
        run_results[label] = _neg_log_lik_and_grad_norm_per_iterate(res, M_j, X_j, label)

    return run_results


def stochastic_approximation_log_likelihood():
    # compares the fixed sweep order (feature_order='sequential') against the
    # i.i.d. uniform random feature order (feature_order='random') for both
    # gibb_sampl_fast_sampling and online_gibbs_sampling, tracking negative
    # observed-data log-likelihood and gradient norm across Gibbs iterations,
    # averaged (mean +/- 1 std) over independent repetitions
    print("\n\nstarting stochastic_approximation_log_likelihood()\n")
    n, d = 200, 10
    lbd = 0.0005
    R = 300
    n_repeats = 10
    p_mcar = 0.3  # MCAR missingness probability
    gamma = 1.0 / np.arange(1, R * d + 1)  # gamma[h] = 1/h, h = 1, 2, ... (decaying step size, online gibbs)

    all_neg_ll = {m: np.zeros((n_repeats, R)) for m in METHODS}
    all_grad = {m: np.zeros((n_repeats, R)) for m in METHODS}

    for rep in range(n_repeats):
        print(f"\n===== repetition {rep + 1}/{n_repeats} (seed={rep}) =====")
        run_results = _run_once(rep, n, d, lbd, R, p_mcar)
        for m in METHODS:
            neg_ll, grad_norms = run_results[m]
            all_neg_ll[m][rep] = neg_ll
            all_grad[m][rep] = grad_norms

    folder = Path("results/experiment5_stochastic_approximation")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "n.npy", np.array([n]))
    np.save(folder / "d.npy", np.array([d]))
    np.save(folder / "lbd.npy", np.array([lbd]))
    np.save(folder / "R.npy", np.array([R]))
    np.save(folder / "n_repeats.npy", np.array([n_repeats]))
    np.save(folder / "p_mcar.npy", np.array([p_mcar]))
    np.save(folder / "gamma.npy", gamma)
    for m in METHODS:
        slug = m.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(folder / f"neg_ll_{slug}.npy", all_neg_ll[m])
        np.save(folder / f"grad_norm_{slug}.npy", all_grad[m])
    print(f"\nsaved raw results (mean/std computable from these) to {folder}/")

    _plot_stochastic_approximation_results(n, d, R, n_repeats, all_neg_ll, all_grad, folder)


def _plot_stochastic_approximation_results(n, d, R, n_repeats, all_neg_ll, all_grad, folder):
    # fig, (ax_ll, ax_grad) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig, ax_grad = plt.subplots(figsize=(12, 5))
    iters = np.arange(R)

    for m in METHODS:
        color = STYLES[m]['color']

        # mean_ll = all_neg_ll[m].mean(axis=0)
        # std_ll = all_neg_ll[m].std(axis=0)
        # ax_ll.plot(iters, mean_ll, label=m, **STYLES[m])
        # ax_ll.fill_between(iters, mean_ll - std_ll, mean_ll + std_ll, alpha=0.3, color=color)

        mean_grad = all_grad[m].mean(axis=0)
        std_grad = all_grad[m].std(axis=0)
        ax_grad.plot(iters, mean_grad, label=LABELS[m], **STYLES[m])
        ax_grad.fill_between(iters, np.maximum(mean_grad - std_grad, 1e-8), mean_grad + std_grad,
                              alpha=0.3, color=color)

    # ax_ll.set_ylabel("-log-likelihood")
    # ax_ll.set_title(f"-Log-likelihood vs Gibbs iteration (n={n}, d={d}, {n_repeats} repeats, mean ± 1 std)")
    # ax_ll.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax_ll.grid(which='both')

    ax_grad.set_xlabel("Iterations", fontsize=16)
    ax_grad.set_ylabel("Gradient norm", fontsize=16)
    # ax_grad.set_title("Gradient norm vs Gibbs iteration", fontsize=15, fontweight='bold')
    ax_grad.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=13)
    ax_grad.grid(which='both')

    plt.tight_layout()

    fig.savefig(folder / "plot.pdf")
    print(f"\nsaved plot to {folder / 'plot.pdf'}")

    plt.show()


def plot_stochastic_approximation_log_likelihood():
    # re-creates the experiment5 plot from the .npy files already saved by
    # stochastic_approximation_log_likelihood(), without rerunning the Gibbs sampling
    folder = Path("results/experiment5_stochastic_approximation")

    n = int(np.load(folder / "n.npy")[0])
    d = int(np.load(folder / "d.npy")[0])
    R = int(np.load(folder / "R.npy")[0])
    n_repeats = int(np.load(folder / "n_repeats.npy")[0])

    all_neg_ll = {}
    all_grad = {}
    for m in METHODS:
        slug = m.replace(" ", "_").replace("(", "").replace(")", "")
        all_neg_ll[m] = np.load(folder / f"neg_ll_{slug}.npy")
        all_grad[m] = np.load(folder / f"grad_norm_{slug}.npy")

    _plot_stochastic_approximation_results(n, d, R, n_repeats, all_neg_ll, all_grad, folder)


if __name__ == "__main__":
    # stochastic_approximation_log_likelihood()
    plot_stochastic_approximation_log_likelihood()
