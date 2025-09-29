import numpy as np
from my_miceforest.miceforest import ImputationKernel
import miceforest as mf
import pandas as pd

np.random.seed(42)

n = 15
d = 3
X = np.random.randint(1, 10, size=(n, d)) + 0.0
M = np.random.binomial(1, 0.2, size=(n, d)) 
print(M==1)
X_nan = X
X_nan[M==1] = np.nan
print(X_nan)

x = pd.DataFrame(X_nan)
x.columns = x.columns.astype(str)


#df = pd.DataFrame({
#    "0": [1, 2, np.nan, 4],
#    "1": [5, np.nan, 7, 8]
#})

#k = mf.ImputationKernel(x, num_datasets=2)
#k.mice(2)
#print(k.complete_data)

kernel = ImputationKernel(x, num_datasets=3, mean_match_candidates=3)
kernel.mice(2, verbose=False)
completed = kernel.complete_data(0)
print(completed)


