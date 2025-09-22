import numpy as np


def separate_seen_and_not_seen(X, M):
    # X dataset
    # M masks, 1 seen, 0 not seen
    X_masked = X * M
    X_observed = X * (1-M)
    return X_masked, X_observed


def first_cycle(X, M):
    # X observations, 
    # M masks, 0 seen, 1 missing
    x = 1
    n, d = X.shape 
    X_masked, X_observed = separate_seen_and_not_seen(X, M) 
    X_cp = np.zeros(shape=(d, n, d-1))
    for i in range(d):
        X_cp[i, :, :] = np.delete(X_observed, i, axis=1)
    print(X_cp)
    print(X_observed)


X = np.random.randint(1, 10, size=(4, 3))
M = np.random.binomial(1, 0.5, size=(4, 3))
print(X)
print(X[:, -1])
print(M)
first_cycle(X, M)   


