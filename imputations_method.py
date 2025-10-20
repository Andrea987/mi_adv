# imputation's block
import numpy as np
import pandas as pd
from gcimpute.gaussian_copula import GaussianCopula
import miceforest as mf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge



def clear_dataset(X, y, masks):
  # remove observations full NaN
  # X is an (n, d) matrix, y is a (n,) vector,
  # masks is an (n, d) binary matrix associated to X. 1 missing, 0 seen
  M = np.sum(1 - masks, axis=1) > 0
  print("X shape in clear data ", X.shape)
  print("y shape in clear data ", y.shape)
  print("M shape in clear data ", M.shape)
  M_col = np.sum(1 - masks, axis=0) > 0  # True if in the column there is at least one seen component
  if np.sum(M_col) < masks.shape[1]:
    print("Careful, there is one column full of nan")
  return X[M, :][:, M_col], y[M], masks[M, :][:, M_col]


def single_imputation(X_nan, impute_estimator):
    ice = IterativeImputer(estimator=impute_estimator)
    return ice.fit_transform(X_nan)


def multiple_imputation(info_mi, X_nan):
    n, d = X_nan.shape
    print("info mi", info_mi)
    nbr_mi = info_mi['mi_nbr']
    nbr_feature = info_mi['nbr_feature']
    max_iter = info_mi['max_iter'] if 'max_iter' in info_mi.keys() else d
    res = np.zeros((nbr_mi, n, d))
    for i in range(nbr_mi):
       n_i = np.random.randint(0, 100000)
       print("nbr features ", nbr_feature)
       ice = IterativeImputer(random_state=n_i, max_iter=max_iter, sample_posterior=True, n_nearest_features=nbr_feature)
       res[i, :, :] = ice.fit_transform(X_nan)
       #print("fin res shape", res.shape)
       #if nbr_mi == 1:
        #res = res[0, :, :]
        #print("fin res shape", res.shape)
    return res



def imputation_elliptic(mu, sigma, x, masks):
  # mu, mean elliptical distribution (,d)
  # sigma, cov matrix elliptical distribution (d, d)
  # x: dataset (n, d)
  # masks: mask data, 0 seen, 1 missing
  n, d = x.shape
  print(n, d)
  x_imp = x.copy()
  #print("x_imp clean", x_imp)
  for i in range(n):
    if not (masks[i, :] == 0).all():  # if we have at least one missing component
      #print("nbr : ", i)
      x_c = x[i, :]
      m_bool = (masks[i, :] == 0)  # True seen, False missing
      sigma_aa_inv = np.linalg.inv(sigma[m_bool, :][:, m_bool])
      sigma_ma = sigma[~m_bool, :][:, m_bool]
      mu_cond = mu[~m_bool] + sigma_ma @ sigma_aa_inv @ (x_c[m_bool] - mu[m_bool])
      x_imp[i, ~m_bool] = mu_cond
  return x_imp


def listwise_delection(X, masks):
  # masks: 1 missing, 0 seen
    M = np.sum(masks, axis=1) == 0  # zeros components are the one with full entries
    ret = X[M, :] if X.ndim == 2 else X[M]
    return ret

def miceforest_imputation(info_mf, X_nan):
    n, d = X_nan.shape
    nbr_mi = info_mf['mi_nbr']

def gc_imputation(info_gc, X_nan):
  nbr_mi = info_gc['mi_nbr']
  model = GaussianCopula()
  model.fit_transform(X=X_nan)
  Xmul = model.sample_imputation(X=X_nan, num=nbr_mi).swapaxes(0, 2)
  Xmul = Xmul.swapaxes(-1, -2)
  return Xmul

def miceforest_imputation(info_mf, X_nan):
    n, d = X_nan.shape
    nbr_mi = info_mf['mi_nbr']
    nbr_iterations_mc = info_mf['it_mc']  # number iteration Markov Chain
    res = np.zeros((nbr_mi, n, d))
    if np.isnan(X_nan).sum() == 0:  # no missing component
      print("CAREFUL::: no missing component, so no multiple imputation")
      res = [X_nan] * nbr_mi
    else:
      x = pd.DataFrame(X_nan)
      x.columns = x.columns.astype(str)
      # Create kernel.
      kernel = mf.ImputationKernel(
      x,
      num_datasets=nbr_mi,
      #random_state=1,
      mean_match_candidates=info_mf['nbr_candidates_mm']
      )
      # Run the MICE algorithm for 2 iterations on each of the datasets
      variables = kernel.model_training_order
      print("variab ", variables)
      #dic_param = {key:{'random_state':42, 'n_estimator': 25, 'bagging_fraction': 0.9999999999, 'feature_fraction': 0.999999999999} for key in variables}
      '''
      dic_param = {
        key: {
          'random_state': 42,
         'n_estimators': 25,
         'bagging_fraction': 0.9999999999999999,
         'feature_fraction': 0.9999999999999999,
         'force_col_wise': True,
         'num_threads': 1
        } for key in variables
      }'''
      #kernel.mice(iterations=0,
                  #variable_parameters=dic_param,
      #            mean_match_candidates=0
      #            )
      kernel.mice(nbr_iterations_mc)
      for i in range(nbr_mi):
        res[i, :, :] = kernel.complete_data(dataset=i)
    return res



def imputations(info, dict_obs_for_imp):  # X_nan, y):
  # info contains the method and possible extra information
  # X_nan is the dataset with nan in place of the missing components
  # y is return as it is, unless the method require to change it, like in
  # listwise deletion
    #print(info)
    X_nan = dict_obs_for_imp['X_nan']
    y = dict_obs_for_imp['y_train']
    mask_from_X_nan = np.isnan(X_nan).astype(int)
    if info['imp_method'] == 'BR_si':  # Baeysian_Ridge_single_imputation
        X = single_imputation(X_nan, BayesianRidge())
    elif info['imp_method'] in  ['mi', 'mi_pure']:
        print("info in imputations ", info)
        X = multiple_imputation(info, X_nan)  # size (info['mi_nbr], n, d)
    elif info['imp_method'] in ['mf_imp','mf_imp1']:
        X = miceforest_imputation(info, X_nan)
    elif info['imp_method'] == 'gc':
        X = gc_imputation(info, X_nan)
    elif info['imp_method'] == 'l_d':  
        # listwise_deletion
        #mask_from_X_nan = np.isnan(X_nan).astype(int)
        X = listwise_delection(X_nan, mask_from_X_nan)
        y = listwise_delection(y, mask_from_X_nan)
        if len(X) == 0:  # no elements left, add an artificial element
            X = np.zeros((1, X_nan.shape[-1]))
            y = np.zeros(1)
        mask_from_X_nan = np.zeros_like(X)
    elif info['imp_method'] == 'oracle':
        X = dict_obs_for_imp['X_train_masked'][0]
        mask_from_X_nan = np.zeros_like(X)
    else:
      print("-------------------> ERROR: WRONG KEYWORD (in imputations)")
    return X, y, mask_from_X_nan




def post_imputation(info_imp, dict_dataset):
  # X_imptued should be a matrix (n, d) or tensor (m, d, n) (in multiple imputations methods)
    X_imputed = dict_dataset['X_imputed']
    y_train = dict_dataset['y_from_X_imputed']
    #print("info imp in post_imp", info_imp)
    print("shape X_imputed in post_imputation ", X_imputed.shape)
    mask_train = dict_dataset['masks_after_imputation']
    if 'post_imp' not in info_imp.keys():
      X_train = X_imputed
    elif info_imp['post_imp'] == 'mean':
      #print("entered in pst_iputation, in mi_mean")
      X_train = np.mean(X_imputed, axis=0)
    elif info_imp['post_imp'] == 'conc':
      print("shape X_imputed ", X_imputed.shape)
      X_train = np.concatenate(X_imputed)
      y_train = np.tile(y_train, X_imputed.shape[0])
    else:
      X_train = X_imputed
    return X_train, y_train, mask_train





