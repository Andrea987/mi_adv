import numpy as np
from sklearn.impute import SimpleImputer
from itertools import product


def generate_binary_arrays(n):
    return np.array(list(product([0, 1], repeat=n)))


def best_predictor(X, coeff, y):
  hat_y = (X @ coeff).T  # (n, d) @ (d, m) = (n, m)
  r = hat_y - y  # residual
  score = np.mean(r * r, axis=1)
  print("scores:  ", score)
  i_min = np.argmin(score)
  return coeff[:, i_min], score[i_min]


def best_idx_predictor(X, coeff, y):
  hat_y = (X @ coeff).T  # (n, d) @ (d, m) = (n, m)
  r = hat_y - y  # residual
  #score = np.mean(r * r, axis=1)
  score = np.mean(r * r, axis=1)
  #print("score in best idx", score)
  i_min = np.argmin(score)
  #### find the minimum value with a threshold, so we get bigger uncertainty set that are visible
  min = np.min(score)
  max = np.max(score)
  score[ score < min + -1 ] = max
  ####
  #print("score after ", score)
  i_min = np.argmin(score)
  return i_min, score[i_min]


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


def nb_flip(m1, m2):
  # m1 and m2 are 0, 1 mask, 0 seen, 1 missing. 
  # Count the number of components that do not match
  return np.sum(m1 != m2)


def flip_matrix_manual(M):
  #print("who is M, flip matrix\n", M)
  d, n = M.shape
  #print(n," ", d)
  ret = np.zeros((d, d))
  for i in range(d):
    for j in range(d):
      #print(i," ", j)
      ret[i, j] = nb_flip(M[i, :], M[j, :])
  return ret


def swm_formula(Q, U, c):
    # sherman woodbury morrison formula
    # compute the inverse of (Q + c*U.T@U)ˆ(-1)
    if U.ndim == 1 or U.shape[0] == 1 or U.shape[1] == 1:
        #print("rk 1 modification")
        ret = rk_1_update_inverse(Q, U, c)
    else:
        d, m = U.shape  # U = [u_1|..|u_m], size = (d, m)
        #print("shape U ", d, m)
        #print(U)
        #print(Q)
        #print(Q.dtype)
        #print(U.dtype)
        #print("cond numb ", np.linalg.cond(Q))
        with np.errstate(over='raise'):
            w = U.T @ Q  # x = np.exp(1000)
            #cn = np.linalg.cond(Q)
            #print("cond nbr ", cn)
            #print("max Q: ", np.max(Q), ", min Q ", np.min(Q))
            #cn = 1e1
            #if cn > 1e8:
            #    print(cn)
            
            if not np.all(np.isfinite(w)):
                print("Q: ", Q)
                print("cond numb:", np.linalg.cond(Q))
                print("Overflow detected inside block.")
                input("Paused. Press Enter to continue...")
            #try:
            #    w = U.T @ Q  # x = np.exp(1000)
            #except FloatingPointError:
            #    print("cond numb ", np.linalg.cond(Q))
            #    print("Overflow detected inside block.")
            #    input("Paused. Press Enter to continue...")
        
        #print("w \n\n\n", w)
        #print(w @ U)
        #print(w.shape)
        #print(U.shape)
        #cc, low = cho_factor(np.eye(m) / c + w @ U)
        #sol = cho_solve((cc, low), w)
        sol = np.linalg.solve(np.eye(m) / c + w @ U, w)
        #print("sol ", sol)
        #print("trial ", (np.eye(m) / c + w @ U) @ sol)
        #print("w", w)
        ret = Q - w.T @ sol # the identity should be cancelled, it is just to mitigate the numerical errors but it shouldn't be there
    return ret


def rk_1_update_inverse(Q, u, c):
    #print(Q)
    #print("u in rk. upd inverse ", u.ndim)
    if u.ndim > 1:
        #print("vector squeezed in rk 1 upd")
        u = np.squeeze(u)
    w = Q @ u
    #print(Q)
    #print(u)
    #print(w)
    return Q - np.outer(w, w) / (1/c + np.sum(u * w))


def matrix_switches(M):
    # this matrix encode which vectors should move
    # from one side to the other
    #M1 = M.copy()
    #m1 = M1[0, :]
    #M[0, :] = M[-1, :]
    #M[-1, :] = m1
    M1 = np.roll(M, -1, axis=1)
    #print(M1)
    return M - M1


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
    N = N #+ np.eye(2) * 1e-8
    N_inv = inverse_2_times_2_sym(N)
    #print(N_inv)
    res = A_inv - WTA_inv.T @ N_inv @ WTA_inv
    #print("res inside update 2 inv\n", res)
    return (res + res.T) / 2


def split_upd(X, ms):
    # split the 1 rank perturbations in updates and downdates
    X_upd = X[ms == 1, :]
    X_dwd = X[ms == -1, :]
    return X_upd, X_dwd
    #return {'+': X_upd, '-': X_dwd}


def split_up_fx_dw(X, ms):
    # split the 1 rank perturbations in updates and downdates
    X_up = X[ms == 1, :]
    X_fx = X[ms == 0, :]
    X_dw = X[ms == -1, :]
    return X_up, X_fx, X_dw
    #return {'+': X_upd, '-': X_dwd}


def s(C, v):
   # compute s(C, v) such that (CˆT - v1ˆT)(C - 1vˆT) = C^TC + s(C, v)
   # C = [c1| ... | cns]^T is (ns, d), v is (, d)
   ns, d = C.shape 
   u = np.ones(ns)
   C_1_v = np.outer(C.T @ u, v)
   return -C_1_v -C_1_v.T + ns * np.outer(v, v)


def update_covariance(Cov1, C1, C2, v1, v2, U, D):
    return Cov1 - s(C1, v1) + U.T @ U - D.T @ D + s(C2, v2)


def make_centered_kernel_matrix(K, m):
    # given a kernel matrix K = X @ X.T and a mask m,
    # compute K_cent = (X - 1 @ mu.T) @ (X - 1 @ mu.T).T,
    # where mu is (X @ (1-m)) / np.sum(1-m) is the mean of a
    # subsample of X
    # m is a mask, such that m[i] = 0 iff component is seen, 0 otherwise
    ns = np.sum(1-m)
    ms = (1 - m) / ns if ns>0 else np.zeros_like(m)
    #print("ms ", ms)
    u = np.ones(K.shape[0])
    w = K @ ms
    sw = np.outer(w, u) + np.outer(u, w)
    return K - sw + np.outer(u, u) * np.sum(w * ms), w, np.sum(w * ms) 
    

def compute_stats(X, m, lbd, idx):
    # test function that compute some variances in different ways
    n, d = X.shape
    R = X[m==0, :]
    mean = np.mean(R, axis=0)
    ns = np.sum(1-m)  # number seen
    u = np.ones(R.shape[0])
    uu = np.ones(X.shape[0])
    R_centered = R - np.outer(u, mean)
    R_centered_del = np.delete(R_centered, idx, 1)
    X_centered = X - np.outer(uu, mean)
    X_centered_del = np.delete(X_centered, idx, 1)
    Cov = R_centered.T @ R_centered + lbd * np.eye(d)
    K = X_centered_del @ X_centered_del.T + lbd * np.eye(n)
    K_S = K[m == 0, :][:, m == 0]  # submatrix of seen components
    K_S_test = R_centered_del @ R_centered_del.T + lbd * np.eye(ns)
    np.testing.assert_allclose(K_S, K_S_test)
    Q = np.linalg.inv(Cov)
    cov_i_given_rest_test = 1 / Q[idx, idx]


    Ri = R_centered[:, idx]
    #R_i = np.delete(R_centered, idx, axis=1)
    inv_cov = np.linalg.inv(R_centered_del.T @ R_centered_del + lbd * np.eye(d-1))
    inv_kernel = np.linalg.inv(K_S)
    first_matrix = R_centered_del @ inv_cov @ R_centered_del.T
    cov_i_given_rest_test2 = np.sum(Ri * Ri) - np.sum(Ri * (first_matrix @ Ri)) + lbd
    one = (R_centered_del @ R_centered_del.T @ Ri) * (inv_kernel @ Ri)
    second_matrix = R_centered_del @ R_centered_del.T @ inv_kernel
    #print("first matrix \n", first_matrix)
    #print("second matrix \n", second_matrix)
    cov_i_given_rest_test3 = np.sum(Ri * Ri) - np.sum(Ri * (second_matrix @ Ri)) + lbd
    cov_i_given_rest_test4 = Cov[idx, idx] - np.sum(one)
    #print("cov i given rest ", cov_i_given_rest)
    print("utils cov i give test ", cov_i_given_rest_test)
    print("cov i given rest test2", cov_i_given_rest_test2)
    print("cov i given rest test3", cov_i_given_rest_test3)
    print("cov i given rest test4", cov_i_given_rest_test4)
    return cov_i_given_rest_test, cov_i_given_rest_test2, cov_i_given_rest_test3, cov_i_given_rest_test4


def compute_stats_mean(X, m, lbd, idx, intercept):
    # test function that compute some means in different ways
    n, d = X.shape
    R = X[m==0, :]
    mean = np.mean(R, axis=0) if intercept else np.zeros(R.shape[1])
    mean_i = np.delete(mean, idx, 0)
    uuu = np.ones(R.shape[0])
    R_centered = R - np.outer(uuu, mean)
    Cov = R_centered.T @ R_centered + lbd * np.eye(d)
    #print(Cov)
    Cov_i = np.delete(Cov, idx, axis=0)
    Cov_ii = np.delete(Cov_i, idx, axis=1)
    #print(Cov_i[:, idx])
    Q = np.linalg.inv(Cov)
    X_i = np.delete(X, idx, axis=1)
    Q_i = np.delete(Q, idx, axis=0)
    Q_ii = np.delete(Q_i, idx, axis=1)
    v = -(1 / Q[idx, idx]) * Q_i[:, idx]
    vv = np.linalg.inv(Cov_ii) @ Cov_i[:, idx] #np.linalg.solve(Q_ii, Cov_i)
    #print("v  ", v, "\nvv ", vv)
    np.testing.assert_allclose(vv, v)
    uuuu = np.ones(X.shape[0])
    prediction1 = mean[idx] + (X_i - np.outer(uuuu, mean_i)) @ v[:, None]  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
    nm = np.sum(m)
    uuuuu = np.ones(nm)
    prediction1 = mean[idx] + (X_i[m == 1, :] - np.outer(uuuuu, mean_i)) @ v[:, None]  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
    return prediction1




def compute_centered_kernel_matrix_regulirized_manually(K, m, lbd, intercept):
    # just a test function which computes the kernel matrix, manually
    n = K.shape[0]
    u = np.ones(n)
    ns = np.sum(1-m)
    ms = (1 - m) / ns if intercept else np.zeros_like(m)
    #ms = (1 - current_mask) / np.sum(1 - current_mask)  # ms[i] = 1 iff component is seen
    A = np.eye(n) - np.outer(u, ms)
    return A @ K @ A.T + np.eye(n) * lbd


def compute_centered_kernel_matrix_regulirized_manually_2(X, m, lbd, intercept):
    n, d = X.shape  # suppose n < d
    R = X[m == 0, :]
    mean = np.mean(R, axis=0) if intercept else 0
    u = np.ones(X.shape[0])
    X_del_centered = np.delete(X - np.outer(u, mean), 0, axis=1)
    return X_del_centered @ X_del_centered.T + lbd * np.eye(n)  # (n, n)


