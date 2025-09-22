import numpy as np
from conjugate_gradient import separate_seen_and_not_seen


def test_separate_seen_and_not_seen():
    n, d = 4, 3

    X = np.random.randint(1, 3, size=(n, d))
    M = np.random.randint(0, 2, size=(n, d))
    Xmasked, Xobs = separate_seen_and_not_seen(X, M)
    np.testing.assert_allclose(X, Xmasked + Xobs)
    print(X)
    print(M)
    print(Xmasked)
    print(Xobs)


test_separate_seen_and_not_seen()






