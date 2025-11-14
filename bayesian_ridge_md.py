import numpy as np
import time
import sys
#sys.path.insert(0, '/home/user/projects/sklearn_dev')  # NOT /sklearn_dev/sklearn
#sys.path.insert(0, '/Users/abasteri/git_projects/mi_adv/sklearn_dev')
from sklearn.linear_model import BayesianRidge
from sklearn import sklearn
from scipy.optimize import root_scalar
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
    n, d = 5, 3
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


def compute_intervals(alpha, a, psi):
    # given alpha and a such that
    # f(lbd) = sum(a * (1/(alpha-lbd))) - psi, compute some confidence interval for the zeros of the function
    print("ciao")
    print("a: ", a)
    s = np.cumsum(a)
    ss = np.sum(a) - s
    print("s: ", s)
    print("ss:", ss)
    Delta = np.delete(alpha, 0) - np.delete(alpha, -1)
    alpha_min_first = np.delete(alpha, 0)
    alpha_min_last = np.delete(alpha, -1)
    a_min_first = np.delete(a, 0)
    a_min_last = np.delete(a, -1)
    print("Delta ", Delta)
    print("s invers ", s[::-1][0:-1])
    lbd_l0 = alpha_min_last + a_min_last * Delta / (a_min_last + ss[0:-1])
    lbd_r0 = alpha_min_first - a_min_first * Delta / s[1:]  # (a_min_first + s[:-1])
    print("alpha min last ", alpha_min_last)
    print("a min last ", a_min_last)
    print("a min last ", a_min_last)
    print("s[:-1] ", s[:-1])
    print("s[1:-1] ", s[1:])
    print(lbd_l0)
    if psi < 0:
        Delta_lbdR = alpha_min_first - lbd_r0
        lbd_l = alpha_min_last + a_min_last * Delta / (a_min_last + ss[0:-1] - psi * Delta)
        lbd_r = alpha_min_first - (a_min_first - psi * Delta_lbdR) * Delta / (s[1:] - psi * Delta_lbdR)

    return lbd_l, lbd_r

def test_compute_intervals():
    n, d = 4, 5
    # alpha = np.random.randint(1, 5, size=d)
    alpha = np.array([1, 3, 5, 7, 10])
    alpha = np.sort(np.random.randn(d))
    print("asymptos ", alpha)
    a = np.random.randint(1, 9, size=d)
    lbd_l, lbd_r = compute_intervals(alpha, a, -1)
    def f(x):
        return 1 + np.sum(a / (alpha-x))
    print("asymptos ", alpha)
    print("left intervals ", lbd_l)
    print("right intervals ", lbd_r)
    for i in range(len(lbd_l)):
        brack = [lbd_l[i] , lbd_r[i]]
        print(brack)
        print(f(lbd_l[i]))
        print(f(lbd_r[i]))
        sol = root_scalar(f, bracket=brack, method='brentq')
        print(f(sol.root))
        print(sol)
    

def test_spectral_decomposition_update():
    n, d = 9, 5
    RR = np.random.randint(1, 7, size=(n, d))
    A = RR.T @ RR #+ 1e-3 * np.eye(d)
    eig_vl, eig_vc = eigh(A)  # v.T @ A @ v = L -> A = v @ L @ v.T
    #print("eig_vl ", eig_vl)
    v = np.random.randint(1, 3, size=d)
    print("new vector ", v)
    psi = -1
    Anew = A + np.outer(v, v)
    print("outer ", np.outer(v, v))
    new_eig_vl, new_eig_vc = eigh(Anew)

    vv = eig_vc.T @ v
    lbd_l, lbd_r = compute_intervals(eig_vl, vv ** 2, psi)
    def f(x):
        return 1 + np.sum((vv ** 2) / (eig_vl-x))
    print("asymptos ", eig_vl)
    print("left intervals  ", lbd_l)
    print("right intervals ", lbd_r)
    roots = np.zeros(d)
    for i in range(len(lbd_l)):
        brack = [lbd_l[i] , lbd_r[i]]
        #print(brack)
        #print(f(lbd_l[i]))
        #print(f(lbd_r[i]))
        sol = root_scalar(f, bracket=brack, method='brentq')
        #print(f(sol.root))
        print(sol.root)
        roots[i] = sol.root
    sum_eig_vl = np.sum(eig_vl)
    new_trace = np.sum(eig_vl) + np.sum(v **2)
    roots[d-1] = new_trace - np.sum(roots)
    print(roots)
    print("new eigvls ", new_eig_vl)




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
    #print("\n ", first_mask)
    #print("X\n", X)
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
#print(XX)
#print(yy)
#print(RR)
clf = BayesianRidge(fit_intercept=False)
clf.fit(XX, yy)
coeff = clf.coef_
#print(coeff)

print("\nnew test\n")
#test_bayesian_ridge_solver()
test_compute_intervals()
test_spectral_decomposition_update()
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









