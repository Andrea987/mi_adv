import numpy as np
import time
import sys
#sys.path.insert(0, '/home/user/projects/sklearn_dev')  # NOT /sklearn_dev/sklearn
#sys.path.insert(0, '/Users/abasteri/git_projects/mi_adv/sklearn_dev')
from sklearn.linear_model import BayesianRidge
from sklearn import sklearn
print("Using sklearn from:", sklearn.__file__)

from python_tsp.heuristics import solve_tsp_local_search
from utils import flip_matrix, generate_binary_arrays, matrix_switches, rk_1_update_inverse, swm_formula
#from sklearn.linear_model import Ridge, BayesianRidge
#from sklearn.impute import SimpleImputer
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
#from sklearn.linear_model import BayesianRidge
from imputations_method import multiple_imputation
from scipy.linalg import cho_factor, cho_solve, eigh
#from my_bayes import BayesianRidge as my_br

## inspired by sklearn BayesianRidge()



def test_bayesian_ridge_solver():
    clf = BayesianRidge()
    max_iter0 = 500
    n, d = 500, 10
    XX = np.random.randint(0, 6, size=(n, d))
    yy = np.random.randint(0, 6, n)
    RR = np.column_stack((yy, XX))
    eig_vl, eig_vc = eigh(RR.T @ RR)  # v.T @ A @ v = L -> A = v @ L @ v.T
    spectral_dec0 = {'eig_vl': eig_vl, 'eig_vc': eig_vc}
    print(XX)
    print(yy)
    clf = BayesianRidge(fit_intercept=False, alpha_init=1.0, max_iter=max_iter0)
    clf.fit(XX, yy)
    coeff = clf.coef_
    print("alpha ", clf.alpha_)
    print("lambda ", clf.lambda_)
    print("sklearn implementation ", coeff)

    coefff, _ = bayesian_ridge_solver(max_iter=max_iter0, R=RR, spectral_dec=spectral_dec0, omega=1.0, feature=0)
    print("our implementation :   ", coefff)


def bayesian_ridge_solver(max_iter, R, spectral_dec, omega, feature):
    # Q = (omega * Id + X.T@X)
    n, d = R.shape
    sw_sum = d
    alpha_, lambda_ = 1, 1
    alpha_1 = 1.0e-6
    alpha_2 = 1.0e-6
    lambda_1 = 1.0e-6
    lambda_2 = 1.0e-6
    eig_vl, eig_vc = spectral_dec['eig_vl'], spectral_dec['eig_vc']
    v = eig_vc.T[:, feature]
    for iter_ in range(max_iter):    
        vv = eig_vc @ (v / (eig_vl + omega))
        coef_ = -np.delete(vv, feature) / vv[feature]
        #print("....coeff in my fct ", coef_.dtype)
        R_i = np.delete(R, feature, axis=1)
        R_i_seen = R[:, feature]
        sse_ = np.sum((R_i_seen - np.dot(R_i, coef_)) ** 2)
        
        delet = np.delete(vv, feature)
        coef_sq = np.sum(delet**2) / vv[0]
        tr_inv = np.sum(1 / (eig_vl+omega)) - vv[0] - coef_sq
        gamma_ = omega * (vv[0] + coef_sq) + np.sum(eig_vl / (eig_vl+omega)) - 1
        #print("inverse trace in my Bayes ", tr_inv)
        #gamma_ = len(delet) - omega * tr_inv
        #print("gamma::::: ", gamma_)
        #print("gamm0::::: ", gamma0_)

        #gamma_ = np.sum((alpha_ * eig_vl) / (lambda_ + alpha_ * eig_vl))
        lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_**2) + 2 * lambda_2)
        alpha_ = (n - gamma_ + 2 * alpha_1) / (sse_ + 2 * alpha_2)
        #print("      my sse_", sse_)
        #print(". n  ", n)
        #print(".....gamma ", gamma_, "lambda ", gamma_, "alpha ", alpha_)
        omega = lambda_ / alpha_
        #print("alpha my fct ", alpha_)
        #print("lambd my fct ", lambda_)
    vv = eig_vc @ (v / (eig_vl + omega))
    coef_ = -np.delete(vv, feature) / vv[feature]
    R_i = np.delete(R, feature, axis=1)
    R_i_seen = R_i[:, feature]
    sse_ = np.sum((R_i_seen - np.dot(R_i, coef_)) ** 2)
    return coef_, sse_


def spectral_decomposition_update(spectral_dec, v, rho):
    # update the spectral decomposition of A + rho * v @ v.T
    eig_vl, eig_vc = spectral_dec['eig_vl'], spectral_dec['eig_vc']
    

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
    eig_vl, eig_vc = eigh(R.T @ R)  # v.T @ A @ v = L -> A = v @ L @ v.T
    spectral_dec = {'eig_vl': eig_vl, 'eig_vc': eig_vc}

    for i in range(max_iter_gs):
         for j in range(d):
                bayesian_ridge_solver(max_iter_br, R, spectral_dec, omega_, j)
              
                X = impute_matrix(X, Q, M, i)
                N = Ms[:, i]
                X_upd, X_dwd = split_upd(X, N)
                nupd, ndwd = X_upd[0], X_dwd[0]
                i_up = 0
                while i_up < nupd:
                    #print("current max ", (i_up + 1) * b_s, "total nbr upd ", nupd)
                    Q = swm_formula(Q, X_upd[i_up * b_s:(i_up + 1) * b_s, :].T, 1.0)
                    i_up = i_up + 1
                Q = swm_formula(Q, X_upd[i_up * b_s:nupd, :].T, 1.0)
                #print("cond nub Q before dwd: ", np.linalg.cond(Q))
                i_dw = 0
                while i_dw < ndwd:
                    i_dw = i_dw + 1



clf = BayesianRidge()
n, d = 5, 3
XX = np.random.randint(0, 6, size=(n, d))
yy = np.random.randint(0, 6, n)
RR = np.column_stack((yy, XX))
print(XX)
print(yy)
print(RR)
clf = BayesianRidge(fit_intercept=False)
clf.fit(XX, yy)
coeff = clf.coef_
print(coeff)

print("\nnew test\n")
test_bayesian_ridge_solver()

'''
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
'''









