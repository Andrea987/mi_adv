import numpy as np
import scipy as sp
from sklearn.impute import SimpleImputer
from scipy.sparse.linalg import LinearOperator, cg
import time

def initialize(info):
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
    return C / detr

def update_inverse_rk2_sym(A_inv, W):
    # A_inv is the dxd symmetric inverse of the object matrix A
    # W is the matrix (u|w), dx2, such that 
    # A_new = A + uw.T + wu.T = A + W.T C W, with C = [[0, 1], [1, 0]]
    WTA_inv = W.T @ A_inv
    WTA_invW = WTA_inv @ W
    C = np.array([[0, 1], [1, 0]])
    N = C + WTA_invW
    N_inv = inverse_2_times_2_sym(N)
    res = A_inv - WTA_inv.T @ N_inv @ WTA_inv
    return (res + res.T) / 2

def test_inverse_2_times_sym():
    t = np.random.randint(1, 9, size=(2, 2))
    t = np.random.randn(2, 2)
    t = t.T @ t + np.eye(2) * 1e-6  
    ti = inverse_2_times_2_sym(t)
    print(t @ ti)
    print(ti @ t)


def test_update_inverse_rk2_sym():
    n, d = 5, 4
    t = np.random.randint(1, 9, size=(n, d))
    a = t.T @ t + np.eye(d)
    a_inv = np.linalg.inv(a)
    w = np.random.randint(1, 9, size=(d, 2))
    C = np.array([[0, 1], [1, 0]])
    a_upd = a + w @ C @ w.T
    a_upd = (a_upd + a_upd) / 2
    inv_a_fct = update_inverse_rk2_sym(a_inv, w)
    print("\n ", a_upd @ inv_a_fct)


test_inverse_2_times_sym()
test_update_inverse_rk2_sym()

def cg_solver():
    c = 1


def Lin_op_mask(X_red, col_masks):
    # X_red (n, d-1) reduced matrix
    # n (,n) vector
    n, d_red = X_red.shape
    print("d red ", d_red)
    def mv(v):
        return X_red.T @ (((1-col_masks) * X_red.T).T) @ v
    return LinearOperator((d_red, d_red), matvec=mv)

def update_column(x_i, m_i, X_i_del, theta):
    # x_i, (, n), i-th column of the original mask
    # m_i, (, n), mask associated to x_i
    first = x_i * (1 - m_i)
    #third = m_i * X_i_del.T
    second = (m_i * X_i_del.T).T @ theta 
    return first + second

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
    
    # first system to solve
    x1 = X_ini[:, 0]
    print("origin x1 \n", x1)
    X_ini_del = np.delete(X_ini, 1, axis=1)
    K_first = X_ini_del.T @ X_ini_del + np.eye(d-1) * 1e-8 # (d, d)
    N_first = list_N[0]
    K_first_inv = np.linalg.inv(K_first)
    n1 = list_N[0]
    K_mask1 = Lin_op_mask(X_ini_del, n1) # first system to solve
    b1 = X_ini_del.T @ (n1 * x1)
    start = time.time()   # tic
    sol1 = cg(K_mask1, b1, x0=None, rtol=1e-05, atol=0.0, M=K_first_inv)[0]
    end = time.time()     # toc
    print(f"Elapsed time with prec: {end - start:.4f} seconds")

    start1 = time.time()   # tic
    sol1 = cg(K_mask1, b1, x0=None, rtol=1e-05, atol=0.0, M=None)[0]
    end1 = time.time()     # toc
    print(f"Elapsed time no   prec: {end1 - start1:.4f} seconds")

    #print(sol1)
    current_x1 = update_column(x1, n1, X_ini_del, sol1)
    #print("curren x1 \n", current_x1)
    #print("origin x1 \n", x1)

np.random.seed(42)
n, d = 2000, 1000
m = np.random.binomial(1, 0.6, size=(n, d))
X = np.random.rand(n, d)
X_nan = X.copy()
X_nan[m==1] = np.nan
print('wewew ', X_nan)

infoo = {'masks': m, 
         'X': X,
         'initialize': 'mean',
         'X_nan': X_nan,
         'it_MC': 2
        }
print(infoo)
sampling(infoo)





