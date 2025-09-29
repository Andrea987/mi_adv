import numpy as np

np.random.seed(42)

def separate_seen_and_not_seen(X, M):
    # X dataset
    # M masks, 1 seen, 0 not seen
    # return X_observed, the seen entries, and set to zero the missing one
    # and X_masked, filled with the missing entries, and set to zero the seen entries
    X_masked = X * M
    X_observed = X * (1-M)
    return X_masked, X_observed

def expand_masks(M, x):
    # given a mask nxd, make a mask (d, n, d) such that
    # M = [M1|..|Md], then 
    # M_ret = [[M1|..|M1], [M2|..|M2], .., [Md|..|Md]]
    # x is the number of repetiton, could be a tuple
    if M.ndim != 2:
        raise ValueError("The mask M must ndim == 2")
    _, d = M.shape
    MM = M.T[:, np.newaxis, :]
    MM = np.swapaxes(MM, -1, -2)
    mm = np.tile(MM, x)
    return mm


def first_cycle(X, M):
    # X observations, 
    # M masks, 0 seen, 1 missing
    n, d = X.shape 
    X_masked, X_observed = separate_seen_and_not_seen(X, M) 
    X_del = np.zeros(shape=(d, n, d-1))
    X_del_orig = np.zeros(shape=(d, n, d-1)) # 
    for i in range(d):
        X_del[i, :, :] = np.delete(X_observed, i, axis=1)
        X_del_orig[i, :, :] = np.delete(X, i, axis=1)
    M_cp = expand_masks(M, M.shape[1]-1)
    print("X_del ", X_del)
    print("X_obs \n", X_observed)
    X_fin = X_del * (1-M_cp)
    X_finT = np.swapaxes(X_fin, -1, -2)
    X_ker = X_finT @ X_fin  # kernel matrix
    X_aug = expand_masks(X_observed, 1)

    XT_y = np.swapaxes(X_del_orig, -1, -2) @ X_aug
    
    print("M: \n", M)
    print("X_fin \n", X_fin)
    print("X_ker:\n ", X_ker)
    print("X_aug \n", X_aug)
    print("X_del_orig \n", X_del_orig)
    print("XT_y ", XT_y)

    #print("ciao ", X_cp )
    #print("masks ", M, "exp mask ", M_cp)



X = np.random.randint(1, 10, size=(4, 3))
M = np.random.binomial(1, 0.3, size=(4, 3))
print(X)
print(X[:, -1])
print(M)
print("test interni")
first_cycle(X, M)   


#M = np.random.binomial(1, 0.5, size=(2, 3))
#print("em \n", expand_masks(M))



