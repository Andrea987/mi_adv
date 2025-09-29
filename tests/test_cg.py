import numpy as np
from conjugate_gradient import separate_seen_and_not_seen, expand_masks


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

def test_expand_masks():
    M_test = np.array(
        [[0, 1, 0], [1, 0, 1]]
    )
    M_res = np.array(
        [
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
    )
    M_fct = expand_masks(M_test)
    np.testing.assert_allclose(M_fct, M_res)
    z = np.zeros(shape=(7, 5))
    zz = np.zeros(shape=(5, 7, 4))
    M_fct2 = expand_masks(z)
    np.testing.assert_allclose(M_fct2, zz)



test_separate_seen_and_not_seen()
test_expand_masks





