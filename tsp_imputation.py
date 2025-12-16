import numpy as np




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
