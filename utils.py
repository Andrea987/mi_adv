import numpy as np



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





def iteartion(X_ini, crr_j, upd_j, list_N, warm_start_j, K_j_inv):
    ## remember that numpy array are mutable, so when they are passed to function they get modified
    ## so, upd_j is getting modified
    j = crr_j + 0
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
    warm_start_j = solj
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
    return K_j_inv, warm_start_j








