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

def flip_matrix(M):
  #print("who is M, flip matrix\n", M)
  d, n = M.shape
  print(n," ", d)
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






