import numpy as np
import scipy as sp
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import BayesianRidge
import time

from utils import initialize

def initialize1(info):
    X = info['X']
    M = info['masks']
    X_nan = info['X_nan']
    if info['initialize'] == 'mean':
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        res = imp_mean.fit_transform(X_nan)
    if info['initialize'] == 'constant':
        imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        res = imp_constant.fit_transform(X_nan)
    return res


def inverse_2_times_2_sym(A):
    # A is a 2x2 symmetric matrix, compute the inverse
    C = np.zeros_like(A)
    C[0, 0] = A[1, 1]
    C[1, 1] = A[0, 0]
    C[0, 1] = -A[0, 1]
    C[1, 0] = -A[1, 0]
    detr = A[0, 0] * A[1, 1] - (A[0, 1]**2)
    #print(C)
    return C / detr

def update_inverse_rk2_sym(A_inv, W):
    # A_inv is the dxd symmetric inverse of the object matrix A
    # W is the matrix (u|w), dx2, such that 
    # A_new = A + uw.T + wu.T = A + W.T C W, with C = [[0, 1], [1, 0]]
    #print("shape in update inverse ", A_inv.shape)
    #print("shape in update inverse 2 ", W.shape)
    #print("W inside update rk 2 inverse \n", W)
    #print("A_inv inside fct \n", A_inv)
    WTA_inv = W.T @ A_inv
    #print("\n ", WTA_inv)
    WTA_invW = WTA_inv @ W
    #print("\n", WTA_invW)
    #print("multiplication performed")
    C = np.array([[0, 1], [1, 0]])
    N = C + WTA_invW
    #print(N)
    N = N + np.eye(2) * 1e-8
    N_inv = inverse_2_times_2_sym(N)
    #print(N_inv)
    res = A_inv - WTA_inv.T @ N_inv @ WTA_inv
    #print("res inside update 2 inv\n", res)
    return (res + res.T) / 2

'''
def shift_inverse(A):
    # given a matrix A1 = [[a, b.T],[b,C]], 
    # and given its inverse A, compute the inverse of A2 = [[C, b],[b.T,a]]
    # observe that A2 is obtain by A1 by permuting a row and a column
    A = np.roll(A, shift=-1, axis=0)
    return np.roll(A, shift=-1, axis=1)
'''
    
def shift(A):
    # given a matrix A1 = [[a, b.T],[b,C]], 
    # and given its inverse A, compute the inverse of A2 = [[C, b],[b.T,a]]
    # observe that A2 is obtain by A1 by permuting a row and a column
    A = np.roll(A, shift=-1, axis=1)
    return np.roll(A, shift=-1, axis=0)


def test_inverse_2_times_sym():
    t = np.random.randint(1, 9, size=(2, 2))
    t = np.random.randn(2, 2)
    t = t.T @ t + np.eye(2) * 1e-6  
    ti = inverse_2_times_2_sym(t)
    print(t @ ti)
    print(ti @ t)


def test_update_inverse_rk2_sym():
    n, d = 4, 2
    t = np.random.randint(1, 9, size=(n, d))
    a = t.T @ t + np.eye(d) * 1e-8
    a_inv = np.linalg.inv(a)
    w = np.random.randint(1, 9, size=(d, 2))
    w = np.random.rand(d, 2)
    print(w)
    C = np.array([[0, 1], [1, 0]])
    a_upd = a + w @ C @ w.T
    a_upd = (a_upd + a_upd) / 2
    inv_a_fct = update_inverse_rk2_sym(a_inv, w)
    print("right inverse \n ", a_upd @ inv_a_fct)
    print("left  inverse \n ", inv_a_fct @ a_upd)


def test_shift_inverse():
    n, d = 4, 3
    t = np.random.randint(1, 9, size=(n, d))
    a = t.T @ t #+ np.eye(d) * 1e-8
    #print(a)
    a_inv = np.linalg.inv(a)
    #print("a inv \n", a_inv)
    a_shift = shift(a)
    #print(a_shift)
    a_shift_inverse = np.linalg.inv(a_shift)
    #print("a shift inverse \n", a_shift_inverse)
    a_shift_inverse_fct = shift(a_inv)
    #print("a shift inverse fct \n", a_shift_inverse_fct)
    np.testing.assert_allclose(a_shift_inverse, a_shift_inverse_fct)
    print("end test shift inverse")



test_inverse_2_times_sym()
test_update_inverse_rk2_sym()
test_shift_inverse()

def cg_solver():
    c = 1


def Lin_op_mask(X_red, col_masks):
    # X_red (n, d-1) reduced matrix
    # n (,n) vector
    n, d_red = X_red.shape
    #print("d red ", d_red)
    def mv(v):
        return X_red.T @ (((1-col_masks) * X_red.T).T) @ v
    #print("kernel matrix with missingnesss")
    #print(X_red.T @ (((1-col_masks) * X_red.T).T))
    return LinearOperator((d_red, d_red), matvec=mv)

def update_column(x_i, m_i, X_i_del, theta):
    # x_i, (, n), i-th column of the original mask
    # m_i, (, n), mask associated to x_i
    first = x_i * (1 - m_i)
    #third = m_i * X_i_del.T
    second = (m_i * X_i_del.T).T @ theta #np.random.randn(len(theta)) * 0.1)  # sampling part
    second = second + m_i * np.random.randn(len(second)) * np.std(second)
    return first + second

def update_low_rank_matrix():
    # update the low rank matrix V such that A_(t+1) = A_t + V
    x = 1
    return x


def sampling(info):
    X = info['X']
    n, d = X.shape
    M = info['masks']
    R = info['it_MC']
    print(M)
    list_N = [M[:, i] for i in range(d)]
    print(list_N)
    X_nan = info['X_nan']
    X_ini = initialize(info)
    info['X_ini'] = X_ini
    info['list_N'] = list_N
    
    # first system to solve
    #x1 = X_ini[:, 0]
    #print("origin x1 \n", x1)
    #X_ini_del = np.delete(X_ini, 1, axis=1)
    #K_first = X_ini_del.T @ X_ini_del + np.eye(d-1) * 1e-8 # (d, d)
    #N_first = list_N[0]
    #K_first_inv = np.linalg.inv(K_first)
    #n1 = list_N[0]
    #K_mask1 = Lin_op_mask(X_ini_del, n1) # first system to solve
    #b1 = X_ini_del.T @ (n1 * x1)
    #start = time.time()   # tic
    #sol1 = cg(K_mask1, b1, x0=None, rtol=1e-05, atol=0.0, M=K_first_inv)[0]
    #end = time.time()     # toc
    #print(f"Elapsed time with prec: {end - start:.4f} seconds")

    #start1 = time.time()   # tic
    #sol1 = cg(K_mask1, b1, x0=None, rtol=1e-05, atol=0.0, M=None)[0]
    #end1 = time.time()     # toc
    #print(f"Elapsed time no   prec: {end1 - start1:.4f} seconds")

    
    ice = IterativeImputer(estimator=BayesianRidge())
    #start2 = time.time()   # tic
    #res1 = ice.fit_transform(X_nan)
    #end2 = time.time()     # toc
    #print(f"Elapsed time no   prec: {end1 - start1:.4f} seconds")
    #print(sol1)
    #current_x1 = update_column(x1, n1, X_ini_del, sol1)
    
    X_ini_del = np.delete(X_ini, 0, axis=1)
    K_j = X_ini_del.T @ X_ini_del  # + np.eye(d-1) * 1e-8 # (d, d)
    K_j_inv = np.linalg.inv(K_j)
    info['K_j_inv'] = K_j_inv
    print("K j ", K_j)
    print("K j inv ", K_j_inv)
    print(K_j @ K_j_inv)
    upd_j = np.zeros((d-1, 2))
    ptr_X = np.zeros_like(X_ini)
    print(upd_j)
    warm_start_j = None
    info['warm_start'] = warm_start_j

    print("start program\n\n")
    for i in range(R):  # nbr iteration markov chain
        print("\n next iter MC ", i, "\n" )
        for crr_j in range(d):  # 0, 1, ..., d-1
            #print("\n\ncrr_j ", crr_j)
            j = crr_j + 0
            vj = X_ini[:, j]
            X_del_j = np.delete(X_ini, j, axis=1)
            #print("X_del_j \n", X_del_j)
            km = X_del_j.T @ X_del_j
            #print("kernel matrix \n", km)
            nj = info['list_N'][j]
            K_maskj = Lin_op_mask(X_del_j, nj) # first system to solve
            bj = X_del_j.T @ (nj * vj)
            solj = cg(K_maskj, bj, x0=warm_start_j, rtol=1e-05, atol=1e-05, M=K_j_inv)[0]
            warm_start_j = solj
            vj_updated = update_column(vj, nj, X_del_j, solj)  # here we should add the sampling
            #print("solj ", solj)
            X_ini[:, j] = vj_updated
            vj_upd = vj_updated
            #print("X_ini after updt \n ", X_ini)
            if crr_j == d-1:
                j=0
            upd_j[j, 0] = 1
            pt_j = (vj_upd - X_del_j[:, j]) @ X_del_j # partial_j
            pt_j[j] = np.sum((vj_upd - X_del_j[:, j]) * (vj_upd + X_del_j[:, j])) / 2
            upd_j[:, 1] = pt_j
            #print("upd_j print \n ", upd_j)
            km_check = km + np.outer(upd_j[:, 0], upd_j[:, 1]) + np.outer(upd_j[:, 1], upd_j[:, 0])
            #print("km_check, kernel matrix updated \n", km_check)
            C = np.array([[0, 1], [1, 0]])
            km_check2 = km + upd_j @ C @ upd_j.T 
            #print("km_check1, kernel matrix updated \n", km_check2)
            #print("km check 2 \n", km_check2)git 
            #print("det km_check", np.linalg.det(km_check))
            #print("k_jinv times K_j", K_j_inv @ K_j)
            K_j_inv = update_inverse_rk2_sym(K_j_inv, upd_j)
            print("cond number inverse: ", np.linalg.cond(K_j_inv))
            if crr_j == d-1:
                #print("small check in the case d-1: \n", km_check @ K_j_inv)
                #print("old inverse ", K_j_inv)
                K_j_inv = shift(K_j_inv) 
                #print("shifted inverse \n", K_j_inv)
                km_check = shift(km_check)
                #print("shifted km\n", km_check)
            #print("\ninverse with fct update inverse\n", K_j_inv1)
            #print("\ninverse with inverse 2 times 2 system \n", K_j_inv2)
            #print("\ninverse np \n", np.linalg.inv(km_check))
            #print("\ninverse np(km check) @ km check \n", np.linalg.inv(km_check) @ km_check)
            #print("\nsmall test inv \n", K_j_inv @ km_check)
            #print("\nsmall test inv \n", np.sum(K_j_inv @ km_check))
            upd_j[j, 0] = 0  # set the index to 0 once again, in the next iteration it must be 0
    return X_ini

'''
            if j < d-1:
                print("\n\nj: ", j)
                vj = X_ini[:, j]
                X_del_j = np.delete(X_ini, j, axis=1)
                print("X_del_j \n", X_del_j)
                km = X_del_j.T @ X_del_j
                print("kernel matrix \n", km)
                nj = list_N[j]
                K_maskj = Lin_op_mask(X_del_j, nj) # first system to solve
                bj = X_del_j.T @ (nj * vj)
                solj = cg(K_maskj, bj, x0=warm_start_j, rtol=1e-05, atol=0.0, M=K_j_inv)[0]
                vj_updated = update_column(vj, nj, X_del_j, solj)  # here we should add the sampling
                print("solj ", solj)
                X_ini[:, j] = vj_updated
                print("X_ini after updt \n ", X_ini)
                upd_j[j, 0] = 1
                pt_j = (vj - X_del_j[:, j]) @ X_del_j # partial_j
                pt_j[j] = np.sum((vj - X_del_j[:, j]) * (vj + X_del_j[:, j])) / 2
                upd_j[:, 1] = pt_j
                #print("upd_j print \n ", upd_j)
                km_check = km + np.outer(upd_j[:, 0], upd_j[:, 1]) + np.outer(upd_j[:, 1], upd_j[:, 0])
                #print("km_check, kernel matrix updated \n", km_check)
                C = np.array([[0, 1], [1, 0]])
                km_check2 = km + upd_j @ C @ upd_j.T 
                #print("km check 2 \n", km_check2)git 
                #print("det km_check", np.linalg.det(km_check))
                #print("k_jinv times K_j", K_j_inv @ K_j)
                K_j_inv1 = update_inverse_rk2_sym(K_j_inv, upd_j)
                K_j_inv2 = inverse_2_times_2_sym(km_check)
                #print("\ninverse with fct update inverse\n", K_j_inv1)
                #print("\ninverse with inverse 2 times 2 system \n", K_j_inv2)
                #print("\ninverse np \n", np.linalg.inv(km_check))
                #print("\ninverse np(km check) @ km check \n", np.linalg.inv(km_check) @ km_check)
                print("\nsmall test inv \n", K_j_inv1 @ km_check)
                K_j_inv = K_j_inv1
                upd_j[j, 0] = 0  # set the index to 0 once again, in the next iteration it must be 0  
            else:  # j == d-1, we need to make a special move
                print("\n\nj: ", j)
                vj = X_ini[:, j]
                X_del_j = np.delete(X_ini, j, axis=1)
                print("X_del_j \n", X_del_j)
                km = X_del_j.T @ X_del_j
                print("kernel matrix \n", km)
                nj = list_N[j]
                K_maskj = Lin_op_mask(X_del_j, nj) # first system to solve
                bj = X_del_j.T @ (nj * vj)
                solj = cg(K_maskj, bj, x0=warm_start_j, rtol=1e-05, atol=0.0, M=K_j_inv)[0]
                vj_updated = update_column(vj, nj, X_del_j, solj)  # here we should add the sampling
                print("sold ", solj)
                X_ini[:, j] = vj_updated
                print("X_ini after updt \n ", X_ini)
                upd_j[0, 0] = 1  # change the first column and row, the one with vˆ(1)
                pt_j = (vj - X_del_j[:, 0]) @ X_del_j # partial_j
                print("shape X del j ", X_del_j.shape)
                print("axis 0 ", X_del_j[:, 0])
                pt_j[0] = np.sum((vj - X_del_j[:, 0]) * (vj + X_del_j[:, 0])) / 2
                upd_j[:, 1] = pt_j
                print("upd_j \n ", upd_j)
                km_check = km + np.outer(upd_j[:, 0], upd_j[:, 1]) + np.outer(upd_j[:, 1], upd_j[:, 0])
                print("kernel matrix updated \n", km_check)
                K_j_inv = update_inverse_rk2_sym(K_j_inv, upd_j)
                K_j_inv = shift(K_j_inv)
                #K_j_inv = np.roll(K_j_inv, shift=1, axis=0)
                #K_j_inv = np.roll(K_j_inv, shift=1, axis=1)
                print("kernel matrix updated \n", km_check)
                print("small test inv", K_j_inv @ km_check)
                upd_j[0, 0] = 0  # set the index to 0 once again, in the next iteration it must be 0  
'''
                
'''                
#def iteartion(X_ini, crr_j, upd_j, list_N, warm_start_j, K_j_inv):
def iteration(info, crr_j):
    ## remember that numpy array are mutable, so when they are passed to function they get modified
    ## so, upd_j is getting modified
    X_ini = info['X_ini']
    upd_j = info['upd_j']
    K_j_inv = info['K_j_inv']
    j = crr_j + 0
    vj = X_ini[:, j]
    X_del_j = np.delete(X_ini, j, axis=1)
    print("X_del_j \n", X_del_j)
    km = X_del_j.T @ X_del_j
    print("kernel matrix X_del_j.T @ X_del_j\n", km)
    nj = info['list_N'][j]
    K_maskj = Lin_op_mask(X_del_j, nj) # first system to solve
    bj = X_del_j.T @ (nj * vj)
    solj = cg(K_maskj, bj, x0=info['warm_start_j'], rtol=1e-05, atol=0.0, M=K_j_inv)[0]
    vj_updated = update_column(vj, nj, X_del_j, solj)  # here we should add the sampling
    print("solj ", solj)
    info['warm_start_j'] = solj
    X_ini[:, j] = vj_updated
    print("X_ini after updt \n ", X_ini)
    if crr_j == d-1:
        j=0
    upd_j[j, 0] = 1
    pt_j = (vj - X_del_j[:, j]) @ X_del_j # partial_j
    pt_j[j] = np.sum((vj - X_del_j[:, j]) * (vj + X_del_j[:, j])) / 2
    upd_j[:, 1] = pt_j
    #print("upd_j print \n ", upd_j)
    km_check = km + np.outer(upd_j[:, 0], upd_j[:, 1]) + np.outer(upd_j[:, 1], upd_j[:, 0])
    #print("km_check, kernel matrix updated \n", km_check)
    C = np.array([[0, 1], [1, 0]])
    km_check2 = km + upd_j @ C @ upd_j.T 
    #print("km check 2 \n", km_check2)git 
    #print("det km_check", np.linalg.det(km_check))
    #print("k_jinv times K_j", K_j_inv @ K_j)
    K_j_inv = update_inverse_rk2_sym(K_j_inv, upd_j)
    if crr_j == d-1:
        K_j_inv = shift(K_j_inv) 
    #print("\ninverse with fct update inverse\n", K_j_inv1)
    #print("\ninverse with inverse 2 times 2 system \n", K_j_inv2)
    #print("\ninverse np \n", np.linalg.inv(km_check))
    #print("\ninverse np(km check) @ km check \n", np.linalg.inv(km_check) @ km_check)
    print("\nsmall test inv \n", K_j_inv @ km_check)
    upd_j[j, 0] = 0  # set the index to 0 once again, in the next iteration it must be 0


    #print("curren x1 \n", current_x1)
    #print("origin x1 \n", x1)
'''

np.random.seed(42)
n, d = 400, 100
R = 1  # iteration MC
m = np.random.binomial(1, 0.4, size=(n, d))
X = np.random.rand(n, d)
X_nan = X.copy()
X_nan[m==1] = np.nan
print('wewew ', X_nan)

infoo = {'masks': m, 
         'X': X,
         'initialize': 'mean',
         'X_nan': X_nan,
         'it_MC': 5
        }
print(infoo)
start1 = time.time()   # tic
res = sampling(infoo)
end1 = time.time()     # toc
print(f"Elapsed time no   prec: {end1 - start1:.4f} seconds")

ice = IterativeImputer(estimator=BayesianRidge(), max_iter=d * R, initial_strategy='mean')
start2 = time.time()   # tic
res1 = ice.fit_transform(X_nan)
end2 = time.time()     # toc
print(f"Elapsed time no   prec: {end2 - start2:.4f} seconds")
#print("res sampling \n", res)

#print("res iter imputer \n", res1)



