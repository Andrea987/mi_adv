import numpy as np
from tsp import gibb_sampl_under_parametrized_sampling
from utils import plot2D
from hyppo.ksample import Energy
from sklearn.impute import SimpleImputer



def plot2D_gaussian():
    # given a sample from a Gaussian distribution, with some hidden components,
    # run the algorithm and see if the final sampling resemble the initial one
    print("test gibb sampling udnerparametrized_sampling began")
    n, d = 70, 2
    n0 = 20
    lbd = 0.8 + 0.0
    mean = np.array([5, 5])
    cov = np.array([[2.5, 1.5],[1.5, 1]])
    X_orig = np.random.multivariate_normal(mean, cov, size=n)
    stat, pvalue = Energy().test(X_orig, X_orig)
    print("stat ", stat , "pvalue ", pvalue)
    extra_info = {'current_stat': stat, 'current_p_value': pvalue, 'extra_text': None}
    X = X_orig.copy()
    #X = X / np.sqrt(n)  # normalization, so that X.T @ X is the true covariance matrix, and the result should not explode
    #M = np.random.binomial(1, 0.3, size=(n, d))
    M = np.zeros_like(X)
    for ii in range(d):
        nbr = np.random.randint(0, n)
        #print("SUM OF COLUMNS MASKS ", np.sum(M[:, ii]))
        if np.sum(M[:, ii]) == n:
            print("add a random seen component")
            M[nbr, ii] = 0
    for i in range(n0, n):
        idx = np.random.randint(0, 2, 1)
        M[i, idx] = 1
    print("in plot2DGaussian ", M)
    extra_info['extra_text'] = ', original data'
    plot2D(X_orig, M, extra_info)
    extra_info['extra_text'] = None
    
    #input()
    #plot2D(X_orig, M, )
    #print(M) if n<=10 else print("too many observations, no printing of the mask in tsp_test: ", n)
    #input()
    #M[-1, 0] = 0
    #print("exponent", exponent)
    #M = make_mask_with_bounded_flip(n=n, d=d, p_miss=0.2, p_flip=exponent)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    #print("X_nan \n", X_nan)
    #print(X_nan)
    R = 1
    info_dic = {
        'data': X,
        'imputed_data': None,
        'masks': M,
        'nbr_it_gibb_sampl': R,
        'lbd_reg': lbd,
        'tsp': False,
        'recomputation': False,
        'batch_size': 64,
        'verbose': 0,
        'initial_strategy': 'constant',
        'exponent_d': 0.75,
        'sampling': True,
        'intercept': True,
    }
    #ress = gibb_sampl_under_parametrized_sampling(info_dic)
    #info_dic['data'] = res
    #res = X_orig
    #stat, pvalue = Energy().test(X_orig, res)
    #print("stat ", stat , "pvalue ", pvalue)
    #extra_info = {'current_stat': stat, 'current_p_value': pvalue}
    #plot2D(X, M, extra_info)
    X_nan = X.copy()
    X_nan[M==1] = np.nan
    initial_imputation = SimpleImputer(missing_values=np.nan, strategy=info_dic['initial_strategy'])
    X_imputed = initial_imputation.fit_transform(X_nan)
    stat, pvalue = Energy().test(X_orig, X_imputed)
    print("stat ", stat , "pvalue ", pvalue)
    extra_info = {'current_stat': stat, 'current_p_value': pvalue}
    plot2D(X_imputed, M, extra_info)
    res = gibb_sampl_under_parametrized_sampling(info_dic)
    stat, pvalue = Energy().test(X_orig, res)
    print("stat ", stat , "pvalue ", pvalue)
    extra_info = {'current_stat': stat, 'current_p_value': pvalue}
    plot2D(res, M, extra_info)
    RR = 6
    pvalue_list = []
    for i in range(RR):
        res = gibb_sampl_under_parametrized_sampling(info_dic)
        info_dic['imputed_data'] = res
        stat, pvalue = Energy().test(X_orig, res)
        print("stat ", stat , "pvalue ", pvalue)
        extra_info = {'current_stat': stat, 'current_p_value': pvalue}
        print("iterat ", i)
        pvalue_list.append(pvalue)
        plot2D(res, M, extra_info)


plot2D_gaussian()























