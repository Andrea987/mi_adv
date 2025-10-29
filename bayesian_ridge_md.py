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
from scipy.linalg import cho_factor, cho_solve
from my_bayes import BayesianRidge as my_br


## inspired by sklearn BayesianRidge()


def bayesian_ridge_solver(max_iter, Q):
    #  Q = (omega * Id + X.T@X)ˆ(-1) 
    for iter_ in range(max_iter):


        
        


np.random.seed(53)
n = 80
d = 8
lbd = 1 + 0.0
X_orig = np.random.randint(-9, 9, size=(n, d)) + 0.0
X_orig = np.random.rand(n, d) + 0.0
print(X_orig.dtype)
print("max min ", )
mean = np.mean(X_orig, axis=0)
std = np.std(X_orig, axis=0)
# Standardize
X = (X_orig - mean) / std
print(np.max(X))
print(np.min(X))
M = np.random.binomial(1, 0.2, size=(n, d))
X_nan = X.copy()
X_nan[M==1] = np.nan
R = 50
l0 = 1e-4
a0 = 1e-4
info_dic = {
    'data': X,
    'masks': M,
    'nbr_it_gibb_sampl': R,
    'lbd_reg': lbd,
    'tsp': False,
    'recomputation': False,
    'lambda_init': l0,
    'alpha_init': a0
}





def br_gibb_sampling(info):
    a = info['alpha_init']
    l = info['lambda_init']

    X = info['data']
    M = info['masks']
    print("shape M", M.shape)
    print("nbr masks ", np.sum(M, axis=0).shape)
    print("nbr masks ", np.sum(M, axis=0))
    r = info['nbr_it_gibb_sampl']
    lbd = info['lbd_reg']
    n, d = X.shape
    Ms = matrix_switches(M)
    first_mask = M[:, 0]
    #print("\n ", first_mask)
    R = X[first_mask == 0, :]
    #print("first set vct ", R)
    print("first set vct shape ", R.shape)
    Rt_R = R.T @ R + (1/n) * lbd * np.eye(d)
    Q = np.linalg.inv(Rt_R)












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


clf1 = BayesianRidge()
clf1.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
clf1.predict([[1, 1]])

clf = my_br()
clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
clf.predict([[1, 1]])
print("end")










