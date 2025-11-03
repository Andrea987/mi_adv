import numpy as np
import time
from python_tsp.heuristics import solve_tsp_local_search
from utils import flip_matrix, generate_binary_arrays, matrix_switches, rk_1_update_inverse, swm_formula
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge
from imputations_method import multiple_imputation
from scipy.linalg import cho_factor, cho_solve, eigh
from my_bayes import BayesianRidge as my_br


## inspired by sklearn BayesianRidge()


def update_coeff():
    coef_ = np.linalg.multi_dot(
                [Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y]
            )


def bayesian_ridge_solver(max_iter, Q):
    #  Q = (omega * Id + X.T@X)ˆ(-1)
    for iter_ in range(max_iter):
        coef_, sse_ = update_coef_(
                    X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
                )
        


np.random.seed(53)
n = 5
d = 3
lbd = 1 + 0.0
X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
#X_orig = np.random.rand(n, d) + 0.0
print(X_orig.dtype)
print("max min ", )
mean = np.mean(X_orig, axis=0)
std = np.std(X_orig, axis=0)
# Standardize
#X = (X_orig - mean) / std
X = X_orig
print(np.max(X))
print(np.min(X))
M = np.random.binomial(1, 0.2, size=(n, d))
X_nan = X.copy()
X_nan[M==1] = np.nan
R_gibb = 50
l0 = 1e-4
a0 = 1e-4
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': R_gibb,
    'lbd_reg': lbd,
    'tsp': False,
    'recomputation': False,
    'lambda_init': l0,
    'alpha_init': a0,
    'max_iter_br': 2
}


def br_gibb_sampling(info):
    alpha_ = info['alpha_init']
    lambda_ = info['lambda_init']
    X = info['data']
    M = info['masks']
    max_iter_br = info['max_iter_br']
    max_iter_gs = info['max_iter_gs']
    print("shape M", M.shape)
    print("nbr masks ", np.sum(M, axis=0).shape)
    print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    print("\n ", first_mask)
    print("X\n", X)
    R = X[first_mask == 0, :]
    print("first set vct \n", R)
    print("first set vct shape ", R.shape)
    omega_ = lambda_ / alpha_
    #Rt_R = R.T @ R + omega_ * np.eye(d)
    eig_vl, eig_vc = eigh(R.T @ R)

    for i in range(max_iter_gs):
         for j in range(d):
              bayesian_ridge_solver(R)
              for iter_ in range(max_iter_br):
                # update posterior mean coef_ based on alpha_ and lambda_ and
                # compute corresponding sse (sum of squared errors)
                







br_gibb_sampling(info_dic)


ice = IterativeImputer(estimator=BayesianRidge(), max_iter=R, initial_strategy='mean')
start2 = time.time()   # tic
res1 = ice.fit_transform(X_nan)
end2 = time.time()     # toc
print(f"Elapsed time no 1 simple imputer  prec: {end2 - start2:.4f} seconds")

print("ciao")
start3 = time.time()   # tic
res2 = multiple_imputation({'mi_nbr':1, 'nbr_feature':None, 'max_iter': R}, X_nan)
#print(res2)
end3 = time.time()     # toc
print(f"Elapsed time no 2 iter imputer  prec: {end3 - start3:.4f} seconds")

#clf1 = BayesianRidge()
#clf1.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
#clf1.predict([[1, 1]])

#clf = my_br()
#clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
#clf.predict([[1, 1]])
print("end")










