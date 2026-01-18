import numpy as np
from utils import compute_stats



def impute_matrix_under_parametrized(XX, Q, M, i):
    # X input matrix
    # Q current inverse, Q = (X.T@X + lbd*Id)^(-1)
    # M masks, 0 seen, 1 missing
    # i current iteration when sweeping the column
    X = XX.copy()
    #print("masks \n", M)
    n, d = X.shape
    xi = X[:, i]
    X_i = np.delete(X, i, axis=1)
    Q_i = np.delete(Q, i, axis=0)
    v = np.zeros(d-1)
    v = -(1 / Q[i, i]) * Q_i[:, i]
    #if i == 0:
    #    v = -(1/Q[0, 0]) * Q[1:, 0]
    #elif i== d:
    #    v = -(1/Q[d, d]) * Q[0:d-1, 0]
    #else:
    #    v[0:i] = Q[0:i, 0]
    #    v[(i+1):d] = Q[(i+1):d, 0]
    #    v = -(1/Q[i, i]) * v
    prediction = X_i @ v[:, None]  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
    #vvv = -30 * (vv < 4).astype(int) + 18 * (vv >= 6).astype(int) + vv * ((vv >= 4) & (vv < 6)).astype(int) 
    #thr = 1.8
    #prediction = -thr * (prediction < -thr).astype(int) + thr * (prediction >= thr).astype(int) + prediction * ((prediction >= -thr) & (prediction < thr)).astype(int) 
    #print(v[:, None])
    #print("test in impute matrix, who is v\n ", v)
    #print(-Q * (1 / Q[i, i]))
    #print(prediction, prediction.shape)
    #print(prediction.squeeze())
    #print(X[:, i])
    #print(M[:, i])
    #print(X[:, i] * (1 - M[:, i]))
    #print(prediction * M[:, i])
    # print(X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i])
    # print(X[:, i])
    X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i] + 0.0
    #X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i]
    #X[:, i] = np.zeros_like(X[:, i])
    #print("new X\n", X)
    return X, v  # imputed matrix, coeff


def impute_matrix_overparametrized(X, M, K ,K_inv, lbd, idx):
    n_m = np.sum(M[:, idx])  #  nbr missing, M_ij = 1 iff component is missing
    n_s = np.sum(1-M[:, idx])
    _, d = X.shape

    #print("K\n", K)
    #print("K_inv\n", K_inv)
    #print("nplinalg.inv(K)\n", np.linalg.inv(K))

    X_idx = X[:, idx]
    X_s = X_idx[M[:, idx] == 0]  # (n_s,)
    A = K_inv[M[:, idx] == 1][:, M[:, idx] == 0]  # (n_m, n_s)
    #print("dim A ", A.shape, ", nbr miss ", n_m, "nbr seen ", n_s) 
    if A.ndim == 1:
        A = np.array([A])
    if n_m < n_s:  # not many missing components 
        #print("n_m < n_s")
        #print("M \n", M)
        #X_del = np.delete(X, idx, axis=1)
        
        #X.copy()
        #C = K_inv[M[:, idx] == 1, :]  #  (n_m, n)
        #print("C \n", C)
        #S_C = C[0:n_m, 0:n_m]  #  Schur Complement
        #print(K_inv)
        S_C = K_inv[M[:, idx] == 1, :][:, M[:, idx] == 1]
        if S_C.ndim == 1:
            S_C = np.array([S_C])
        #print("S_C \n", S_C)
        #print("A\n", A)
        
        #print("X_s ", X_s)
        x = np.linalg.solve(S_C, A @ X_s)
        #print("x\n ", x)
        #print("check ", np.sqrt( np.sum( (S_C @ x - A@X_s)**2) ) )
        #print("M \n ", M[:, idx])
        #X[M[:, idx] == 1, idx] = -x
    else:  # many missing components, it's better to work with the the submatrix of seen components
        #print("n_s < n_m")
        K_S = K[M[:, idx] == 0, :][:, M[:, idx] == 0]  # submatrix of seen components
        #print("K_S\n", K_S)
        v = A @ X_s  # (n_m, n_s) * (n_s,) = (n_m,) 
        X_del = np.delete(X, idx, axis=1)
        Xm = X_del[M[:, idx] == 1, :]  # (n_m, d-1) 
        Xs = X_del[M[:, idx] == 0, :]  # (n_s, d-1)
        #print("Xs @ Xs.T + lbd * Id\n", Xs @ Xs.T + lbd * np.eye(n_s))
        w = Xm.T @ v  # (d-1, n_m) * (n_m,) = (d-1,)
        partial = w - Xs.T @ np.linalg.solve(K_S, Xs @ w ) 
        x = Xm @ partial + lbd * v
#        print("x\n ", x)
#        S_CC = Xm @ Xm.T - (Xm @ Xs.T) @ np.linalg.inv(K_S) @ (Xs @ Xm.T) + lbd * np.eye(n_m)
#        S_C_inv = np.linalg.inv(S_CC)
#        S_C_true = K_inv[M[:, idx] == 1, :][:, M[:, idx] == 1]  # this works in the other branch
#        print("SC_inv\n", S_C_inv)
#        print("K_inv\n", K_inv)
#        print("\n\nSC true\n ", S_C_true)
#        print("S_C_inv @ v\n ", S_CC @ v)
#        S_C_true_inv = np.linalg.inv(S_C_true)
#        print("SC true inv (from K_inv)\n", S_C_true_inv)
#        print("SCS computed by hand\n", S_CC)
#        x = np.linalg.inv(S_C_true) @ v
    X[M[:, idx] == 1, idx] = -x

    '''
    Xm = X_del[M[:, idx] == 1, :]
    Xs = X_del[M[:, idx] == 0, :]
    #print("split X\n", X)
    #print("\n", Xm)
    #print(Xs)
    one = Xm @ Xs.T
    two = np.linalg.inv((Xs @ Xs.T + lbd * np.eye(n_s)))
    res = -one @ two @ X_s
    #print("check result1 in impute matrix overparamettrized\n", res)

    one1 = Xm
    two1 = np.linalg.inv((Xs.T @ Xs + lbd * np.eye(d-1)))
    res1 = -one1 @ two1 @ (Xs.T @ X_s)
    #print("check result1 in impute matrix overparamettrized\n", res1)
    '''
    return X


'''
def impute_matrix_under_parametrized_sampling_max_likelihood(XX, mu, S, Q, M, i):
    # XX input matrix
    # mu current mean
    # S current covariance matrix
    # Q current inverse of S, i.e. Q = (X.T@X + lbd*Id)^(-1)
    # M masks, 0 seen, 1 missing
    # i current iteration when sweeping the column
    X = XX.copy()
    #print("masks \n", M)
    n, d = X.shape
    xi = X[:, i]
    X_i = np.delete(X, i, axis=1)
    Q_i = np.delete(Q, i, axis=0)
    S_i = np.delete(S, i, axis=1)
    mu_i = np.delete(mu, i, axis=0)
    v = np.zeros(d-1)
    v = -(1 / Q[i, i]) * Q_i[:, i]

    check_v = S_i[i, :] @ np.linalg.inv( np.delete(S_i, i, axis=0) )
    #print("v       ", v)
    #print("check_v ", check_v)

    S_current_check = S[i, i] - np.sum( S_i[i, :] * v)
    S_current = 1 / Q[i, i]
    #print("S_current      ", S_current)
    #print("S_current check", S_current_check)

    #if i == 0:
    #    v = -(1/Q[0, 0]) * Q[1:, 0]
    #elif i== d:
    #    v = -(1/Q[d, d]) * Q[0:d-1, 0]
    #else:
    #    v[0:i] = Q[0:i, 0]
    #    v[(i+1):d] = Q[(i+1):d, 0]
    #    v = -(1/Q[i, i]) * v
    u = np.ones(n)
    prediction = mu[i] + (X_i - np.outer(u, mu_i)) @ v[:, None]

    prediction = prediction.squeeze()  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
    print(prediction.shape)
    print("S_current ", S_current)
    sample = np.random.multivariate_normal(mean = prediction, 
                                           cov = S_current * np.eye(n))
    #vvv = -30 * (vv < 4).astype(int) + 18 * (vv >= 6).astype(int) + vv * ((vv >= 4) & (vv < 6)).astype(int) 
    #thr = 1.8
    #prediction = -thr * (prediction < -thr).astype(int) + thr * (prediction >= thr).astype(int) + prediction * ((prediction >= -thr) & (prediction < thr)).astype(int) 
    #print(v[:, None])
    #print("test in impute matrix, who is v\n ", v)
    #print(-Q * (1 / Q[i, i]))
    #print(prediction, prediction.shape)
    #print(prediction.squeeze())
    #print(X[:, i])
    #print(M[:, i])
    #print(X[:, i] * (1 - M[:, i]))
    #print(prediction * M[:, i])
    # print(X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i])
    # print(X[:, i])
    X[:, i] = X[:, i] * (1 - M[:, i]) + sample.squeeze() * M[:, i] + 0.0
    #X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i]
    #X[:, i] = np.zeros_like(X[:, i])
    #print("new X\n", X)
    return X, v  # imputed matrix, coeff
'''


def impute_matrix_under_parametrized_sampling(XX, mu, S, Q, M, i):
    # XX input matrix
    # mu current mean
    # S current covariance matrix
    # Q current inverse of S, i.e. Q = (X.T@X + lbd*Id)^(-1)
    # M masks, 0 seen, 1 missing
    # i current iteration when sweeping the column
    X = XX.copy()
    #print("masks \n", M)
    n, d = X.shape
    xi = X[:, i]
    X_i = np.delete(X, i, axis=1)
    Q_i = np.delete(Q, i, axis=0)
    S_i = np.delete(S, i, axis=1)
    mu_i = np.delete(mu, i, axis=0)
    v = np.zeros(d-1)
    v = -(1 / Q[i, i]) * Q_i[:, i]

    check_v = S_i[i, :] @ np.linalg.inv( np.delete(S_i, i, axis=0) )
    #print("v       ", v)
    #print("check_v ", check_v)

    S_current_check = S[i, i] - np.sum( S_i[i, :] * v)
    S_current = 1 / Q[i, i]
    #if ml_or_bs == 'bayes':
    #    S_current = v / np.sum(M[:, i])  # the conditional variance is rescaled by the number of seen components  
    #print("S_current      ", S_current)
    #print("S_current check", S_current_check)

    #if i == 0:
    #    v = -(1/Q[0, 0]) * Q[1:, 0]
    #elif i== d:
    #    v = -(1/Q[d, d]) * Q[0:d-1, 0]
    #else:
    #    v[0:i] = Q[0:i, 0]
    #    v[(i+1):d] = Q[(i+1):d, 0]
    #    v = -(1/Q[i, i]) * v
    u = np.ones(n)
    prediction = mu[i] + (X_i - np.outer(u, mu_i)) @ v[:, None]

    prediction = prediction.squeeze()  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
    #print(prediction.shape)
    #print("S_current ", S_current)

    sample = np.random.multivariate_normal(mean = prediction, 
                                           cov = S_current * np.eye(n))

    #vvv = -30 * (vv < 4).astype(int) + 18 * (vv >= 6).astype(int) + vv * ((vv >= 4) & (vv < 6)).astype(int) 
    #thr = 1.8
    #prediction = -thr * (prediction < -thr).astype(int) + thr * (prediction >= thr).astype(int) + prediction * ((prediction >= -thr) & (prediction < thr)).astype(int) 
    #print(v[:, None])
    #print("test in impute matrix, who is v\n ", v)
    #print(-Q * (1 / Q[i, i]))
    #print(prediction, prediction.shape)
    #print(prediction.squeeze())
    #print(X[:, i])
    #print(M[:, i])
    #print(X[:, i] * (1 - M[:, i]))
    #print(prediction * M[:, i])
    # print(X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i])
    # print(X[:, i])
    X[:, i] = X[:, i] * (1 - M[:, i]) + sample.squeeze() * M[:, i] + 0.0
    #X[:, i] = X[:, i] * (1 - M[:, i]) + prediction.squeeze() * M[:, i]
    #X[:, i] = np.zeros_like(X[:, i])
    #print("new X\n", X)
    return X, v  # imputed matrix, coeff



def impute_matrix_over_parametrized_sampling(X, m, K ,K_inv, lbd, idx, sampling):#XX, mu, S, Q, M, i):
    # K: kernel matrix centered and regulirized
    # K_inv: inverse of K
    # lbd: regulirizer
    # idx: current imputed index
    # m: current_mask
    n_m = np.sum(m)  #  nbr missing, M_ij = 1 iff component is missing
    n_s = np.sum(1-m)
    _, d = X.shape

    #print("K\n", K)
    #print("K_inv\n", K_inv)
    #print("nplinalg.inv(K)\n", np.linalg.inv(K))

    X_idx = X[:, idx]
    X_s = X_idx[m == 0]  # (n_s,)
    A = K[m == 1][:, m == 0]  # (n_m, n_s)
    #print("dim A ", A.shape, ", nbr miss ", n_m, "nbr seen ", n_s) 
    if A.ndim == 1:
        A = np.array([A])
    if n_m < n_s * 1000:  # not many missing components 
        print("less miss components that seen")
        #print("n_m < n_s")
        #print("M \n", M)
        #X_del = np.delete(X, idx, axis=1)
        
        #X.copy()
        #C = K_inv[M[:, idx] == 1, :]  #  (n_m, n)
        #print("C \n", C)
        #S_C = C[0:n_m, 0:n_m]  #  Schur Complement
        #print(K_inv)
        S_C = K_inv[m == 1, :][:, m == 1]  # (n_m, n_m)
        if S_C.ndim == 1:
            S_C = np.array([S_C])
        #print("S_C \n", S_C)
        #print("A\n", A)
        
        #print("X_s ", X_s)
        mean_idx = np.mean(X_s) 
        print(mean_idx)
        #x = mean_idx - np.linalg.solve(S_C, A @ (X_s - mean_idx))
        x = mean_idx - np.linalg.solve(S_C, A @ (X_s - mean_idx))  #
        print("predicitons ", x)  
        R = X[m==0, :]
        mean = np.mean(R, axis=0)
        uuu = np.ones(R.shape[0])
        R_centered = R - np.outer(uuu, mean)
        Cov = R_centered.T @ R_centered + lbd * np.eye(d)
        Q = np.linalg.inv(Cov)
        X_i = np.delete(X, idx, axis=1)
        Q_i = np.delete(Q, idx, axis=0)
        v = np.zeros(d-1)
        v = -(1 / Q[idx, idx]) * Q_i[:, idx]
        prediction1 = X_i @ v[:, None]  #  (n, d-1) * (d-1,) = (n,), cost O(n d)
        print("prediction 1 ", prediction1)

        # TO DO: FIND BECUASE THE TWO PREDICTIONS ARE DIFFERENT

        # as a test you can compute the conditional mean in the standard way

        #print("who is x in impute overparameterized sampling ", -x)
        #print("who is x in impute overparameterized sampling ", x.shape)
        #print("x\n ", x)
        #print("check ", np.sqrt( np.sum( (S_C @ x - A@X_s)**2) ) )
        #print("M \n ", M[:, idx])
        #X[M[:, idx] == 1, idx] = -x
    else:  # many missing components, it's better to work with the the submatrix of seen components
        #print("n_s < n_m")
        K_S = K[m == 0, :][:, m == 0]  # submatrix of seen components
        #print("K_S\n", K_S)
        v = A @ X_s  # (n_m, n_s) * (n_s,) = (n_m,) 
        X_del = np.delete(X, idx, axis=1)
        Xm = X_del[m == 1, :]  # (n_m, d-1) 
        Xs = X_del[m == 0, :]  # (n_s, d-1)
        #print("Xs @ Xs.T + lbd * Id\n", Xs @ Xs.T + lbd * np.eye(n_s))
        w = Xm.T @ v  # (d-1, n_m) * (n_m,) = (d-1,)
        partial = w - Xs.T @ np.linalg.solve(K_S, Xs @ w ) 
        x = Xm @ partial + lbd * v
        x = - x
    if sampling and n_m>0:
        print("K\n ", K)  # K is the regularize centered kernel matrix
        K_S = K[m == 0, :][:, m == 0]  # submatrix of seen components
        Xss = X_s - np.mean(X_s)
        print("test variacen ", np.sum(Xss * Xss))
        print("Xss ", Xss)
        x1 = np.linalg.solve(K_S, Xss)
        K_S_not_reg = K_S - np.eye(n_s) * lbd
        cov_i_given_rest = (np.sum(Xss * Xss) - np.sum(Xss * (K_S_not_reg @ x1)) + lbd) / n_s
        print("my cov   ", cov_i_given_rest)
        cov_test, _, _, _ = compute_stats(X, m, lbd, idx)
        print("test cov ", cov_test / n_s)
        #input()
        prediction = x
        print("predictions ", prediction)
        sample = np.random.multivariate_normal(mean = prediction, cov = cov_i_given_rest * np.eye(n_m))
        x = sample
    X[m == 1, idx] = x
    return X



