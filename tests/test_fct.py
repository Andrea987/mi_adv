import numpy as np
import miceforest as mf
from imputations_method import gc_imputation, miceforest_imputation


'''
def test_gc_impute(n, d, p_seen, nbr_c, mi_nbr):
    xt = np.random.randint(low=0, high=99, size=(n, d)) + 0.0
    mt = np.random.binomial(1, p_seen, size=(n, d))
    xt[mt == 1] = np.nan
    print("original nan\n", xt)
    info_gc = {'nbr_mi':mi_nbr}
    res = gc_imputation(info_gc, xt)
    print(res)



test_gc_impute(n=13, d=5, p_seen=0.3, nbr_c=0, mi_nbr=2)

# test miceforest

np.random.seed(11)
'''

def test_miceforest():
    n=13 
    d=5
    p_seen=0.3 
    nbr_c=0
    
    x_mf_test = np.random.randint(low=0, high=99, size=(n, d)) + 0.0
    m_mf_test = np.random.binomial(1, p_seen, size=(n, d))
    x_mf_test[m_mf_test == 1] = np.nan
    print(x_mf_test)
    info_mf_test = {'mi_nbr': 2, 'nbr_candidates_mm': nbr_c, 'it_mc': 5}

    print(np.isnan(x_mf_test))
    print(np.isnan(x_mf_test).sum() )


    res = miceforest_imputation(info_mf_test, x_mf_test)
    print(res)



test_miceforest()









