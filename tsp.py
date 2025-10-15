import numpy as np
from python_tsp.heuristics import solve_tsp_local_search

n = 6
d = 4
M = np.random.randint(0, 2, size=(n, d))
print(M)

permutation, distance = solve_tsp_local_search(distance_matrix)


